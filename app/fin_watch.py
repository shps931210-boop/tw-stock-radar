"""新財報公告偵測與補抓佇列（V19）。

以前季報固定 20 天才重抓一次，公司公布新一季財報後可能要等兩三週網站才更新。現在改成：
1. 公告偵測：每次更新先讀證交所／櫃買中心 OpenAPI 的「最新一季綜合損益表」，得到每家公司官方已公布的最新季度
   （與 FinMind 同步分開記錄；FinMind 還沒收錄新季度時，不會把公司標成已更新）。
2. 補抓佇列：官方已公布、資料庫還沒有的季度，排進高優先佇列（正式推薦與候選 → 會員自選與申請 → 成交金額），
   每次更新優先向 FinMind 補抓；抓不到就記一次嘗試，隔 6 小時再試，不會永久放棄。
3. 財報季保底：偵測端點失敗時，財報申報期間內缺最新季的股票改成 5 天重抓一次（平常 20 天）。
4. 時間紀錄：每一季財報第一次出現在資料庫的時間（fin_seen）、官方公告偵測時間、同步時間與延遲；
   重抓時數值變動超過 0.5% 記成更正（fin_revision）。
5. 重新評分：快照重算後，把同步前後的 AI 均衡型分數寫進紀錄（history/fin/updates.json），網站可以看到哪些股票因新財報改分。

官方 OpenAPI 只提供「最新一季」，所以歷史公告時間仍只能用法定期限推估（見 data_inventory）。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import requests

from . import config, db
from .sources import http
from .sources.twse import HEADERS

SCHEMA = """
CREATE TABLE IF NOT EXISTS fin_announce (stock_id TEXT, period TEXT, report_date TEXT, source TEXT, first_seen TEXT,
  eps REAL, revenue REAL, status TEXT, attempts INTEGER DEFAULT 0, last_try TEXT, synced_at TEXT,
  score_before REAL, score_after REAL, PRIMARY KEY (stock_id, period));
CREATE TABLE IF NOT EXISTS fin_seen (stock_id TEXT, period TEXT, first_seen TEXT, source TEXT, fields INTEGER,
  PRIMARY KEY (stock_id, period));
CREATE TABLE IF NOT EXISTS fin_revision (stock_id TEXT, period TEXT, detected_at TEXT, detail TEXT,
  PRIMARY KEY (stock_id, period, detected_at));
"""

TWSE = "https://openapi.twse.com.tw/v1/opendata/"
TPEX = "https://www.tpex.org.tw/openapi/v1/"
# 一般業、銀行、證券、金控、保險、其他業的「最新一季綜合損益表」
KINDS = ("ci", "basi", "bd", "fh", "ins", "mim")
ENDPOINTS = [("twse", TWSE + f"t187ap06_L_{k}") for k in KINDS] + [("tpex", TPEX + f"mopsfin_t187ap06_O_{k}") for k in KINDS]

# 法定申報期限（月, 日）：Q1 5/15、Q2 8/14、Q3 11/14、年報（Q4）隔年 3/31
DEADLINE = {1: (5, 15), 2: (8, 14), 3: (11, 14), 4: (3, 31)}
CORE = ("revenue", "gross_profit", "operating_income", "net_income", "eps", "total_assets", "total_liabilities", "equity", "ocf")
RETRY_HOURS = 6
MAX_ATTEMPTS = 20


def ensure(con) -> None:
    con.executescript(SCHEMA)


def _qend(y: int, q: int) -> date:
    return date(y, q * 3, 30 if q in (2, 3) else 31)


def _deadline(y: int, q: int) -> date:
    m, d = DEADLINE[q]
    return date(y + 1 if q == 4 else y, m, d)


def _prev(y: int, q: int) -> tuple[int, int]:
    return (y - 1, 4) if q == 1 else (y, q - 1)


def due_quarter(d: date) -> str:
    """法定期限已過、所有公司都應該公布的最新一季。"""
    y, q = d.year, (d.month - 1) // 3 + 1
    y, q = _prev(y, q)
    while _deadline(y, q) >= d:
        y, q = _prev(y, q)
    return f"{y}-Q{q}"


def season_quarter(d: date) -> str | None:
    """正在申報期間的季度（季末之後到期限後 15 天）；不在財報季回傳 None。"""
    y, q = _prev(d.year, (d.month - 1) // 3 + 1)
    for _ in range(2):
        if _qend(y, q) < d <= _deadline(y, q) + timedelta(days=15):
            return f"{y}-Q{q}"
        y, q = _prev(y, q)
    return None


def _key(row: dict, *names: str, contains: tuple[str, ...] = ()) -> str | None:
    for n in names:
        if n in row:
            return n
    for k in row:
        if contains and all(c in k for c in contains):
            return k
    return None


def _num(v) -> float | None:
    try:
        s = str(v).replace(",", "").strip()
        return float(s) if s not in ("", "-", "--", "N/A") else None
    except ValueError:
        return None


def parse(rows: list[dict]) -> tuple[dict, dict]:
    """把 OpenAPI 的一批資料轉成 {股票代號: {period, report_date, eps, revenue}}。欄位用名稱比對，民國年轉西元。"""
    out = {}
    if not rows:
        return out, {"rows": 0}
    r0 = rows[0]
    k_id = _key(r0, "公司代號", "SecuritiesCompanyCode", "CompanyCode", "Code")
    k_y = _key(r0, "年度", "Year")
    k_q = _key(r0, "季別", "Season", "Quarter")
    k_d = _key(r0, "出表日期", "Date")
    k_eps = _key(r0, contains=("基本每股盈餘",)) or _key(r0, contains=("每股盈餘",)) or _key(r0, "EPS")
    k_rev = _key(r0, "營業收入", "收益合計", "淨收益", "收入") or _key(r0, contains=("營業收入",))
    info = {"rows": len(rows), "fields": list(r0)[:40], "id": k_id, "year": k_y, "quarter": k_q, "date": k_d, "eps": k_eps, "revenue": k_rev}
    if not (k_id and k_y and k_q):
        return out, {**info, "error": "找不到公司代號、年度或季別欄位"}
    for r in rows:
        try:
            sid = str(r[k_id]).strip()
            y, q = int(str(r[k_y]).strip()), int(str(r[k_q]).strip())
        except (ValueError, TypeError):
            continue
        if y < 1911:
            y += 1911
        if not (1 <= q <= 4):
            continue
        out[sid] = {"period": f"{y}-Q{q}", "report_date": str(r.get(k_d) or "").strip() or None,
                    "eps": _num(r.get(k_eps)) if k_eps else None, "revenue": _num(r.get(k_rev)) if k_rev else None}
    return out, info


def detect(session: requests.Session | None = None) -> tuple[dict, list]:
    """讀官方 OpenAPI，回傳 ({股票代號: 公告資料}, 每個端點的診斷)。單一端點失敗不影響其他端點。"""
    s = session or requests.Session()
    found, diag = {}, []
    for mkt, url in ENDPOINTS:
        d = {"market": mkt, "url": url}
        try:
            r = http.get(s, url, headers=HEADERS, timeout=60)
            js = r.json()
            got, info = parse(js if isinstance(js, list) else [])
            d.update(info, ok=bool(got), stocks=len(got))
            for sid, v in got.items():
                if sid not in found or v["period"] > found[sid]["period"]:
                    found[sid] = {**v, "source": f"{mkt}:{url.rsplit('/', 1)[-1]}"}
        except Exception as e:
            d.update(ok=False, error=http.describe(e)[:160])
        diag.append(d)
    return found, diag


def _latest_db(con) -> dict:
    return {sid: p for sid, p in con.execute("SELECT stock_id, MAX(period) FROM financials GROUP BY stock_id")}


def record(con, found: dict, diag: list, now: datetime | None = None) -> dict:
    """把偵測結果寫進 fin_announce：新公告記第一次看到的時間；資料庫已經有該季的標成已同步。"""
    ensure(con)
    now = now or config.now()
    ts = now.isoformat(timespec="seconds")
    have = _latest_db(con)
    new = 0
    for sid, v in found.items():
        if not ("1000" <= sid <= "9999" and len(sid) == 4):
            continue
        cur = con.execute("SELECT status FROM fin_announce WHERE stock_id=? AND period=?", (sid, v["period"])).fetchone()
        synced = (have.get(sid) or "") >= v["period"]
        if cur is None:
            new += 1
            con.execute("INSERT INTO fin_announce (stock_id, period, report_date, source, first_seen, eps, revenue, status, synced_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?)", (sid, v["period"], v.get("report_date"), v.get("source"), ts, v.get("eps"), v.get("revenue"),
                                                      "already" if synced else "pending", None))
        elif synced and cur[0] in ("pending", "waiting_source"):
            con.execute("UPDATE fin_announce SET status='synced', synced_at=? WHERE stock_id=? AND period=?", (ts, sid, v["period"]))
    ok = sum(1 for d in diag if d.get("ok"))
    db.set_meta(con, "fin_watch_detect", json.dumps({"at": ts, "endpoints_ok": ok, "endpoints": len(diag), "stocks": len(found), "new": new,
                                                     "diag": diag}, ensure_ascii=False))
    con.commit()
    return {"stocks": len(found), "new": new, "endpoints_ok": ok}


def run_detect(errors: list | None = None) -> dict | None:
    """更新流程呼叫：偵測失敗只記錄警告，不中斷其他更新。"""
    try:
        found, diag = detect()
        with db.connect() as con:
            res = record(con, found, diag)
        if not res["endpoints_ok"] and errors is not None:
            errors.append("財報公告偵測：官方 OpenAPI 全部端點失敗，改用財報季保底重抓")
        return res
    except Exception as e:
        if errors is not None:
            errors.append(f"財報公告偵測失敗：{http.describe(e)}")
        return None


def _priority(con) -> list[str]:
    """補抓順序的前段：今天正式推薦與候選（各模式前 30 名）→ 會員自選與設定的追蹤清單。"""
    out = []
    try:
        s = json.loads(db.get_meta(con, "scores_today") or "{}").get("s") or {}
        ranked = sorted(((min((x for x in (v.get("r") or {}).values() if x), default=9999), sid) for sid, v in s.items()))
        out += [sid for r, sid in ranked if r <= 30]
    except ValueError:
        pass
    return out


def hot_list(con, front: list[str], now: datetime | None = None) -> list[tuple[str, str]]:
    """官方已公布、資料庫還沒有的季度：回傳 [(股票代號, 季度)]，依優先順序。上次嘗試未滿 6 小時的先跳過。"""
    ensure(con)
    now = now or config.now()
    rows = con.execute("SELECT stock_id, period, attempts, last_try FROM fin_announce WHERE status IN ('pending','waiting_source')").fetchall()
    have = _latest_db(con)
    todo = []
    for sid, per, n, last in rows:
        if (have.get(sid) or "") >= per:
            continue
        if n and n >= MAX_ATTEMPTS:
            continue
        if last and now - datetime.fromisoformat(last) < timedelta(hours=RETRY_HOURS):
            continue
        todo.append((sid, per))
    rank = {sid: i for i, sid in enumerate(dict.fromkeys(_priority(con) + front))}
    todo.sort(key=lambda x: (rank.get(x[0], len(rank)), x[0]))
    return todo


def score_now(con, sid: str) -> float | None:
    try:
        return ((json.loads(db.get_meta(con, "scores_today") or "{}").get("s") or {}).get(sid) or {}).get("sc", {}).get("balanced")
    except ValueError:
        return None


def after_fetch(con, sid: str, period: str, now: datetime | None = None) -> bool:
    """補抓完一檔：資料庫有了該季就標成已同步並記下改分前的分數；還沒有就記一次嘗試（來源尚未同步）。"""
    now = now or config.now()
    ts = now.isoformat(timespec="seconds")
    have = (_latest_db(con).get(sid) or "") >= period
    if have:
        con.execute("UPDATE fin_announce SET status='synced', synced_at=?, last_try=?, attempts=attempts+1, score_before=? WHERE stock_id=? AND period=?",
                    (ts, ts, score_now(con, sid), sid, period))
    else:
        con.execute("UPDATE fin_announce SET status='waiting_source', last_try=?, attempts=attempts+1 WHERE stock_id=? AND period=?", (ts, sid, period))
    return have


def fin_age(con, sid: str, today: date, base: int = 20) -> int:
    """定期重抓間隔：財報申報期間內、資料庫還沒有該季的股票改成 5 天（偵測端點失敗時的保底）。"""
    q = season_quarter(today)
    if not q:
        return base
    r = con.execute("SELECT MAX(period) FROM financials WHERE stock_id=?", (sid,)).fetchone()
    return 5 if (r and r[0] or "") < q else base


def compare(con, sid: str, new_rows: list[dict], now: datetime | None = None) -> list[dict]:
    """寫入前比對：已存在的季度數值變動超過 0.5% 記成更正；第一次出現的季度記下取得時間與科目完整數。"""
    ensure(con)
    ts = (now or config.now()).isoformat(timespec="seconds")
    old = {r["period"]: r for r in db.read(con, "SELECT * FROM financials WHERE stock_id=?", (sid,)).to_dict("records")}
    revs = []
    for r in new_rows:
        p = r.get("period")
        nf = sum(1 for k in CORE if r.get(k) is not None)
        con.execute("INSERT OR IGNORE INTO fin_seen VALUES (?,?,?,?,?)", (sid, p, ts, "FinMind", nf))
        o = old.get(p)
        if not o:
            continue
        ch = {}
        for k in CORE:
            a, b = o.get(k), r.get(k)
            if a is None or b is None or a != a or b != b:
                continue
            if abs(b - a) > max(abs(a) * 0.005, 1e-6):
                ch[k] = [a, b]
        if ch:
            revs.append({"stock_id": sid, "period": p, "fields": ch})
            con.execute("INSERT OR IGNORE INTO fin_revision VALUES (?,?,?,?)", (sid, p, ts, json.dumps(ch, ensure_ascii=False)))
    return revs


def _hours(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600, 1)


def summary(con, today: date | None = None) -> dict:
    """網站與報告用：本季公告與同步進度、等待中的股票、同步延遲、改分紀錄、更正紀錄。"""
    ensure(con)
    today = today or config.today()
    q = season_quarter(today) or due_quarter(today)
    det = json.loads(db.get_meta(con, "fin_watch_detect") or "null")
    rows = db.read(con, "SELECT * FROM fin_announce WHERE period=? ORDER BY first_seen", (q,)).to_dict("records")
    have = _latest_db(con)
    names = {sid: n for sid, n in con.execute("SELECT stock_id, name FROM stocks")}
    n_common = sum(1 for sid in have if len(sid) == 4)
    in_db = sum(1 for sid, p in have.items() if len(sid) == 4 and p >= q)
    lat = sorted(h for r in rows if r["status"] == "synced" and (h := _hours(r["first_seen"], r["synced_at"])) is not None)
    waiting = [{"stock_id": r["stock_id"], "name": names.get(r["stock_id"]), "first_seen": r["first_seen"], "attempts": r["attempts"],
                "eps": r["eps"], "status": r["status"]} for r in rows if r["status"] in ("pending", "waiting_source")]
    changed = [{"stock_id": r["stock_id"], "name": names.get(r["stock_id"]), "period": r["period"], "synced_at": r["synced_at"],
                "before": r["score_before"], "after": r["score_after"], "latency_h": _hours(r["first_seen"], r["synced_at"])}
               for r in db.read(con, "SELECT * FROM fin_announce WHERE status='synced' ORDER BY synced_at DESC LIMIT 200").to_dict("records")]
    revs = [{"stock_id": a, "period": b, "detected_at": c, "fields": json.loads(d)}
            for a, b, c, d in con.execute("SELECT * FROM fin_revision ORDER BY detected_at DESC LIMIT 50")]
    return {
        "schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
        "quarter": q, "in_season": bool(season_quarter(today)), "deadline": _deadline(int(q[:4]), int(q[-1])).isoformat(),
        "due_quarter": due_quarter(today),
        "detect": {k: v for k, v in (det or {}).items() if k != "diag"} if det else None,
        "endpoints": [{k: d.get(k) for k in ("market", "url", "ok", "stocks", "error", "rows")} for d in (det or {}).get("diag", [])],
        "announced": len(rows), "synced": sum(r["status"] in ("synced", "already") for r in rows), "waiting": len(waiting),
        "db_has_quarter": in_db, "db_stocks": n_common,
        "latency_hours": {"n": len(lat), "median": lat[len(lat) // 2] if lat else None, "max": lat[-1] if lat else None},
        "waiting_list": waiting[:100], "rescored": changed, "revisions": revs,
        "note": "官方公告偵測讀證交所／櫃買中心 OpenAPI 最新一季綜合損益表；同步指 FinMind 完整財報（損益、資產負債、現金流量）已寫入資料庫。"
                "同步延遲＝第一次偵測到公告到資料庫有完整財報的時間。",
    }


def after_snapshot(con, out_dir=None) -> dict:
    """快照重算後呼叫：補上同步後的 AI 分數，寫出網站資料 data/fin_watch.json 與 history/fin/updates.json。"""
    ensure(con)
    for sid, per in con.execute("SELECT stock_id, period FROM fin_announce WHERE status='synced' AND score_after IS NULL").fetchall():
        sc = score_now(con, sid)
        if sc is not None:
            con.execute("UPDATE fin_announce SET score_after=? WHERE stock_id=? AND period=?", (sc, sid, per))
    con.commit()
    s = summary(con)
    (config.DATA_DIR / "fin_watch.json").write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")), "utf-8")
    h = config.ROOT / "history" / "fin"
    h.mkdir(parents=True, exist_ok=True)
    (h / "updates.json").write_text(json.dumps({k: s[k] for k in ("generated_at", "quarter", "announced", "synced", "waiting", "db_has_quarter",
                                                                  "latency_hours", "rescored", "revisions", "detect")},
                                               ensure_ascii=False, indent=1), "utf-8")
    return s
