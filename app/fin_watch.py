"""新財報公告偵測與補抓佇列（V19）。

以前季報固定 20 天才重抓一次，公司公布新一季財報後可能要等兩三週網站才更新。現在改成：
1. 公告偵測：每次更新先讀證交所／櫃買中心 OpenAPI 的「最新一季綜合損益表」，得到每家公司官方已公布的最新季度
   （與 FinMind 同步分開記錄；FinMind 還沒收錄新季度時，不會把公司標成已更新）。
2. 補抓佇列：官方已公布、資料庫還沒有的季度，排進高優先佇列（正式推薦與候選 → 會員自選與申請 → 成交金額），
   每次更新優先向 FinMind 補抓；抓不到就記一次嘗試，隔 6 小時再試，不會永久放棄。
3. 財報季保底：偵測端點失敗時，財報申報期間內缺最新季的股票改成 5 天重抓一次（平常 20 天）。
4. 時間紀錄：官方公告偵測時間、補抓時間、寫入資料庫時間、重新評分時間（fin_announce）、每季第一次出現在資料庫的時間（fin_seen）；
   重抓時數值變動超過 0.5% 記成更正（fin_revision）。
5. 更正偵測（V20.1）：官方 EPS、營業收入、營業利益、本期淨利、歸屬母公司淨利任一項變動都算更正，不只看 EPS。
6. 舊資料保護（V20.1）：官方公布的是「年初至今累計」數字；資料庫各季加總和官方一致時，重抓到和官方不一致的舊數字不會覆蓋；
   空白欄位也不會蓋掉已有的數字。資料庫和官方不一致時排進補抓，補到一致才算同步。
7. 重新評分：快照重算後，把同步前後的 AI 均衡型分數寫進紀錄（history/fin/updates.json），網站可以看到哪些股票因新財報改分。

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
  score_before REAL, score_after REAL, corrected_at TEXT, official TEXT, rescored_at TEXT, note TEXT,
  PRIMARY KEY (stock_id, period));
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
# 官方欄位（年初至今累計，金額單位千元）→ 資料庫欄位；更正偵測比對全部，資料一致性只比金額（EPS 因加權股數不同，各季加總不等於累計）
OFFICIAL = ("eps", "revenue", "operating_income", "net_income", "parent_ni")
CHECK = ("revenue", "net_income")
REV_KINDS = ("ci", "ins", "mim")   # 金控、銀行、證券的「淨收益／收益」定義和 FinMind 的營收不同，不比營收
MAX_ATTEMPTS = 20      # 高頻重試次數（每 6 小時，約 5 天）；之後改成低頻持續追蹤，不會永久放棄
SLOW_HOURS = 72        # 低頻追蹤：每 3 天再試一次，直到資料來源同步或官方資料改變


def ensure(con) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(fin_announce)")}
    for c in ("corrected_at", "official", "rescored_at", "note", "rescore_check"):
        if c not in have:
            con.execute(f"ALTER TABLE fin_announce ADD COLUMN {c} TEXT")


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
    k_op = _key(r0, "營業利益（損失）", "營業利益")
    k_ni = _key(r0, "本期淨利（淨損）", "本期稅後淨利（淨損）")
    k_pni = next((k for k in r0 if k.startswith("淨利") and "母公司" in k), None)
    info = {"rows": len(rows), "fields": list(r0)[:40], "id": k_id, "year": k_y, "quarter": k_q, "date": k_d, "eps": k_eps, "revenue": k_rev,
            "operating_income": k_op, "net_income": k_ni, "parent_ni": k_pni}
    if not (k_id and k_y and k_q):
        return out, {**info, "error": "找不到公司代號、年度或季別欄位"}
    blank = 0
    for r in rows:
        if not str(r.get(k_id) or "").strip():
            blank += 1  # 櫃買中心沒有該產業公司時會回一列空白
            continue
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
                    **{f: (_num(r.get(k)) if k else None) for f, k in (("eps", k_eps), ("revenue", k_rev), ("operating_income", k_op),
                                                                       ("net_income", k_ni), ("parent_ni", k_pni))}}
    if blank == len(rows):
        info["empty"] = True  # 端點正常，只是目前沒有這類公司的資料
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
            d.update(info, ok=bool(got) or bool(info.get("empty")), stocks=len(got))
            if not got and isinstance(js, list) and js and not info.get("empty"):
                d["sample"] = {k: str(v)[:20] for k, v in list(js[0].items())[:6]}  # 解析不到時留下第一列，方便查欄位
            for sid, v in got.items():
                if sid not in found or v["period"] > found[sid]["period"]:
                    found[sid] = {**v, "source": f"{mkt}:{url.rsplit('/', 1)[-1]}"}
        except Exception as e:
            d.update(ok=False, error=http.describe(e)[:160])
        diag.append(d)
    return found, diag


def _latest_db(con) -> dict:
    return {sid: p for sid, p in con.execute("SELECT stock_id, MAX(period) FROM financials GROUP BY stock_id")}


def _official(v: dict) -> dict:
    return {k: v.get(k) for k in OFFICIAL if v.get(k) is not None}


def _cum(con, sid: str, period: str, override: dict | None = None) -> dict | None:
    """資料庫該年第 1 季到該季的金額加總（元）；缺任何一季回傳 None。override 是準備寫入、還沒寫進去的那一季。"""
    y, q = period[:4], int(period[-1])
    rows = {r["period"]: r for r in db.read(con, "SELECT * FROM financials WHERE stock_id=? AND period>=? AND period<=?",
                                            (sid, f"{y}-Q1", period)).to_dict("records")}
    if override:
        rows[period] = {**rows.get(period, {}), **override}
    if any(f"{y}-Q{i}" not in rows for i in range(1, q + 1)):
        return None
    out = {}
    for k in CHECK:
        vals = [rows[f"{y}-Q{i}"].get(k) for i in range(1, q + 1)]
        if all(v is not None and v == v for v in vals):
            out[k] = sum(vals)
    return out


def mismatch(con, sid: str, period: str, official: dict, kind: str = "ci", override: dict | None = None) -> dict | None:
    """資料庫（或準備寫入的資料）和官方累計數字比對：回傳不一致的欄位 {欄位: [資料庫, 官方]}；資料不夠比對時回傳 None。
    營收差 0.5% 以上、淨利差 1% 以上（另加 2 千元的四捨五入誤差）算不一致（淨利可對本期淨利或歸屬母公司淨利任一項）。"""
    cum = _cum(con, sid, period, override)
    if not cum or not official:
        return None
    bad, n = {}, 0
    if "revenue" in cum and official.get("revenue") and kind in REV_KINDS:
        n += 1
        if abs(cum["revenue"] / 1000 - official["revenue"]) > abs(official["revenue"]) * 0.005 + 2:
            bad["revenue"] = [round(cum["revenue"] / 1000), official["revenue"]]
    targets = [official[k] for k in ("net_income", "parent_ni") if official.get(k) is not None]
    if "net_income" in cum and targets:
        n += 1
        if all(abs(cum["net_income"] / 1000 - t) > abs(t) * 0.01 + 1 for t in targets):
            bad["net_income"] = [round(cum["net_income"] / 1000), targets[0]]
    return bad if n else None


def _kind(source: str | None) -> str:
    return (source or "").rsplit("_", 1)[-1]


def record(con, found: dict, diag: list, now: datetime | None = None) -> dict:
    """把偵測結果寫進 fin_announce：新公告記第一次看到的時間；資料庫已經有該季的標成已同步。
    官方任一欄位（EPS、營收、營業利益、淨利、母公司淨利）變動算更正，重新補抓；資料庫和官方累計數字不一致也排進補抓。"""
    ensure(con)
    now = now or config.now()
    ts = now.isoformat(timespec="seconds")
    have = _latest_db(con)
    new = corrected = mism = 0
    for sid, v in found.items():
        if not ("1000" <= sid <= "9999" and len(sid) == 4):
            continue
        off = _official(v)
        cur = con.execute("SELECT status, eps, revenue, official, note FROM fin_announce WHERE stock_id=? AND period=?", (sid, v["period"])).fetchone()
        synced = (have.get(sid) or "") >= v["period"]
        bad = mismatch(con, sid, v["period"], off, _kind(v.get("source"))) if synced else None
        if cur is None:
            new += 1
            mism += bool(bad)
            con.execute("INSERT INTO fin_announce (stock_id, period, report_date, source, first_seen, eps, revenue, status, synced_at, official, note) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (sid, v["period"], v.get("report_date"), v.get("source"), ts, v.get("eps"), v.get("revenue"),
                                                          "pending" if bad or not synced else "already", None, json.dumps(off), _note(bad)))
            continue
        old = json.loads(cur[3]) if cur[3] else {"eps": cur[1], "revenue": cur[2]}
        diff = {k: [old.get(k), off.get(k)] for k in OFFICIAL if _changed(old.get(k), off.get(k))}
        if diff:
            # 官方數字變了（更正或重新公告）：重新觸發高頻補抓，重試次數歸零，改分紀錄重來；已同步的也要重抓一次讓資料庫跟上更正
            corrected += 1
            con.execute("UPDATE fin_announce SET status='pending', attempts=0, last_try=NULL, eps=?, revenue=?, official=?, corrected_at=?, "
                        "score_before=NULL, score_after=NULL, rescored_at=NULL, note=? WHERE stock_id=? AND period=?",
                        (v.get("eps"), v.get("revenue"), json.dumps(off), ts, "官方更正：" + "、".join(LABEL.get(k, k) for k in diff), sid, v["period"]))
            con.execute("INSERT OR IGNORE INTO fin_revision VALUES (?,?,?,?)",
                        (sid, v["period"], ts, json.dumps({"source": "官方公告", **diff}, ensure_ascii=False)))
        elif synced and cur[0] in ("pending", "waiting_source") and not bad:
            con.execute("UPDATE fin_announce SET status='synced', synced_at=?, official=?, note=NULL WHERE stock_id=? AND period=?", (ts, json.dumps(off), sid, v["period"]))
        elif bad and cur[0] in ("already", "synced"):
            mism += 1  # 資料庫有該季但和官方對不上（來源資料錯誤或舊版）：排進補抓，補到一致才算同步
            con.execute("UPDATE fin_announce SET status='pending', official=?, note=? WHERE stock_id=? AND period=?", (json.dumps(off), _note(bad), sid, v["period"]))
        elif not cur[3]:
            con.execute("UPDATE fin_announce SET official=? WHERE stock_id=? AND period=?", (json.dumps(off), sid, v["period"]))
    ok = sum(1 for d in diag if d.get("ok"))
    db.set_meta(con, "fin_watch_detect", json.dumps({"at": ts, "endpoints_ok": ok, "endpoints": len(diag), "stocks": len(found), "new": new,
                                                     "corrected": corrected, "mismatch": mism, "diag": diag}, ensure_ascii=False))
    con.commit()
    return {"stocks": len(found), "new": new, "endpoints_ok": ok, "corrected": corrected, "mismatch": mism}


LABEL = {"eps": "EPS", "revenue": "營業收入", "operating_income": "營業利益", "net_income": "本期淨利", "parent_ni": "歸屬母公司淨利"}


def _note(bad: dict | None) -> str | None:
    if not bad:
        return None
    return "資料庫和官方累計數字不一致：" + "、".join(f"{LABEL.get(k, k)} 資料庫 {round(a):,} 千元／官方 {round(b):,} 千元" for k, (a, b) in bad.items())


def _changed(a, b) -> bool:
    return a is not None and b is not None and abs(b - a) > max(abs(a) * 0.005, 0.005)


def run_detect(errors: list | None = None, con=None) -> dict | None:
    """更新流程呼叫：偵測失敗只記錄警告，不中斷其他更新。
    呼叫端已經開著資料庫連線（而且可能有還沒提交的寫入）時要傳進來，另開連線會等鎖 30 秒後失敗。"""
    try:
        found, diag = detect()
        if con is not None:
            res = record(con, found, diag)
        else:
            with db.connect() as c:
                res = record(c, found, diag)
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
    rows = con.execute("SELECT stock_id, period, attempts, last_try, corrected_at, note FROM fin_announce WHERE status IN ('pending','waiting_source')").fetchall()
    have = _latest_db(con)
    todo = []
    for sid, per, n, last, corr, note in rows:
        if (have.get(sid) or "") >= per and not (corr or note):
            continue  # 資料庫已有該季（官方更正或和官方對不上的除外：要重抓）
        mism = (note or "").startswith("資料庫和官方") and not corr
        # 高頻重試用完改低頻，持續追蹤；資料庫和官方對不上的（多半是資料來源本身的問題）重抓 2 次還一樣就改低頻，不占用補抓額度
        wait = SLOW_HOURS if (n or 0) >= (2 if mism else MAX_ATTEMPTS) else RETRY_HOURS
        if last and now - datetime.fromisoformat(last) < timedelta(hours=wait):
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
    """補抓完一檔：資料庫有了該季、而且和官方累計數字一致，才標成已同步並記下改分前的分數；
    還沒有或對不上就記一次嘗試（來源尚未同步／來源還是舊數字），之後照重試規則再抓。"""
    now = now or config.now()
    ts = now.isoformat(timespec="seconds")
    row = con.execute("SELECT official, source FROM fin_announce WHERE stock_id=? AND period=?", (sid, period)).fetchone()
    have = (_latest_db(con).get(sid) or "") >= period
    bad = mismatch(con, sid, period, json.loads(row[0]) if row and row[0] else {}, _kind(row[1] if row else None)) if have else None
    if have and not bad:
        con.execute("UPDATE fin_announce SET status='synced', synced_at=?, last_try=?, attempts=attempts+1, score_before=?, score_after=NULL, "
                    "rescored_at=NULL, note=NULL WHERE stock_id=? AND period=?", (ts, ts, score_now(con, sid), sid, period))
        return True
    note = _note(bad) if bad else "資料來源（FinMind）尚未收錄這一季"
    con.execute("UPDATE fin_announce SET status='waiting_source', last_try=?, attempts=attempts+1, note=? WHERE stock_id=? AND period=?",
                (ts, note, sid, period))
    return False


def guard(con, sid: str, new_rows: list[dict], now: datetime | None = None) -> list[dict]:
    """寫入前的舊資料保護：
    1. 新資料某欄是空的、資料庫已有數字 → 保留資料庫的數字（不讓缺欄的回應蓋掉完整資料）。
    2. 資料庫和官方累計數字一致、新資料卻和官方不一致 → 這一季整列不寫入（來源給的是舊版或錯誤數字），記一筆紀錄。"""
    ensure(con)
    ts = (now or config.now()).isoformat(timespec="seconds")
    old = {r["period"]: r for r in db.read(con, "SELECT * FROM financials WHERE stock_id=?", (sid,)).to_dict("records")}
    off = {p: (json.loads(o) if o else {}, src) for p, o, src in con.execute("SELECT period, official, source FROM fin_announce WHERE stock_id=?", (sid,))}
    out = []
    for r in new_rows:
        p = r.get("period")
        o = old.get(p)
        if o:
            r = {**r, **{k: o[k] for k in r if r.get(k) is None and o.get(k) is not None and o[k] == o[k]}}
        if o and p in off and off[p][0]:
            kind = _kind(off[p][1])
            if mismatch(con, sid, p, off[p][0], kind) == {} and mismatch(con, sid, p, off[p][0], kind, override=r):
                con.execute("INSERT OR IGNORE INTO fin_revision VALUES (?,?,?,?)",
                            (sid, p, ts, json.dumps({"source": "拒絕寫入", "reason": "新抓到的數字和官方公告不一致，資料庫原本的數字和官方一致，保留原本的",
                                                     **mismatch(con, sid, p, off[p][0], kind, override=r)}, ensure_ascii=False)))
                continue
        out.append(r)
    return out


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


def _s(v):
    return v if isinstance(v, str) else None  # pandas 把空值讀成 NaN


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
    # 等待清單不限本季：申報期內上一季還沒同步、或和官方對不上的也要列出來
    pend = db.read(con, "SELECT * FROM fin_announce WHERE status IN ('pending','waiting_source') ORDER BY period DESC, first_seen").to_dict("records")
    waiting = [{"stock_id": r["stock_id"], "name": names.get(r["stock_id"]), "period": r["period"], "first_seen": r["first_seen"], "attempts": r["attempts"],
                "eps": r["eps"], "status": r["status"], "slow": (r["attempts"] or 0) >= MAX_ATTEMPTS, "last_try": r["last_try"],
                "corrected_at": _s(r.get("corrected_at")), "note": _s(r.get("note"))}
               for r in pend]
    changed = [{"stock_id": r["stock_id"], "name": names.get(r["stock_id"]), "period": r["period"], "synced_at": r["synced_at"],
                "before": r["score_before"], "after": r["score_after"], "latency_h": _hours(r["first_seen"], r["synced_at"]),
                "first_seen": r["first_seen"], "fetched_at": r["last_try"], "rescored_at": _s(r.get("rescored_at")), "corrected_at": _s(r.get("corrected_at")),
                "check": _j(r.get("rescore_check"))}
               for r in db.read(con, "SELECT * FROM fin_announce WHERE status='synced' ORDER BY synced_at DESC LIMIT 200").to_dict("records")]
    revs = [{"stock_id": a, "period": b, "detected_at": c, "fields": json.loads(d)}
            for a, b, c, d in con.execute("SELECT * FROM fin_revision ORDER BY detected_at DESC LIMIT 50")]
    return {
        "schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
        "quarter": q, "in_season": bool(season_quarter(today)), "deadline": _deadline(int(q[:4]), int(q[-1])).isoformat(),
        "due_quarter": due_quarter(today),
        "detect": {k: v for k, v in (det or {}).items() if k != "diag"} if det else None,
        "endpoints": [{k: d.get(k) for k in ("market", "url", "ok", "stocks", "error", "rows", "empty")} for d in (det or {}).get("diag", [])],
        "announced": len(rows), "synced": sum(r["status"] in ("synced", "already") for r in rows), "waiting": len(waiting),
        "waiting_this_quarter": sum(r["status"] in ("pending", "waiting_source") for r in rows),
        "slow": sum(w["slow"] for w in waiting), "corrected": sum(1 for r in rows if r.get("corrected_at")),
        "mismatch": sum(1 for r in pend if str(r.get("note") or "").startswith("資料庫和官方")),
        "checks": "官方 EPS、營業收入、營業利益、本期淨利、歸屬母公司淨利任一項變動算更正；資料庫各季加總和官方累計營收（一般業、保險、其他業）與淨利比對，"
                  "不一致就重抓，抓到的數字比原本更偏離官方時不寫入",
        "retry_rule": f"每 {RETRY_HOURS} 小時重試，{MAX_ATTEMPTS} 次後改成每 {SLOW_HOURS // 24} 天低頻追蹤，直到同步；官方數字變動會重新觸發",
        "db_has_quarter": in_db, "db_stocks": n_common,
        "latency_hours": {"n": len(lat), "median": lat[len(lat) // 2] if lat else None, "max": lat[-1] if lat else None},
        "waiting_list": waiting[:100], "rescored": changed, "revisions": revs,
        "note": "官方公告偵測讀證交所／櫃買中心 OpenAPI 最新一季綜合損益表；同步指 FinMind 完整財報（損益、資產負債、現金流量）已寫入資料庫。"
                "同步延遲＝第一次偵測到公告到資料庫有完整財報的時間。",
    }


def _j(x):
    try:
        return json.loads(x) if isinstance(x, str) and x else None
    except ValueError:
        return None


def rescore_check(row: dict | None, period: str, sc: float | None) -> dict:
    """同步後的財報有沒有真的進到 AI 評分：逐項記錄（V20.4）。沒有正式分數時寫清楚原因，不留沒有解釋的空值。"""
    r = row or {}
    in_db = bool(r.get("fin_period")) and str(r.get("fin_period")) >= period
    ex = r.get("exclude_reasons") or []
    miss = r.get("missing_factors") or []
    elig = ((((r.get("ai") or {}).get("rec") or {}).get("balanced") or {}).get("recommendation_eligibility"))
    if sc is not None:
        why = "已產生新的正式 AI 綜合分數"
    elif not row:
        why = "這次快照沒有這檔股票（可能不是普通股或已下市）"
    elif ex:
        why = f"財報已入庫、因子已重算，但不在正式排名母體（{'、'.join(ex)}），依規則不產生正式綜合分數" + \
              (f"；同一套公式的母體外綜合分數 {r['ref_composite']}" if r.get("ref_composite") is not None else "")
    elif miss:
        why = f"財報已入庫，但還缺 {'、'.join(miss)}，五個因子沒有全部算出，不產生綜合分數"
    else:
        why = "財報已入庫，但沒有取得分數（需要檢查）"
    return {"fin_in_db": in_db, "fin_period": r.get("fin_period"), "factors_recomputed": bool(row), "in_rank_universe": bool(r.get("base_ok")),
            "complete": bool(r.get("complete")), "score": sc, "ref_composite": r.get("ref_composite"),
            "eligible": elig == "normal" and bool(r.get("base_ok")) and bool(r.get("complete")), "site_updated": bool(row), "why": why}


def after_snapshot(con, out_dir=None, rows: list[dict] | None = None) -> dict:
    """快照重算後呼叫：補上同步後的 AI 分數，寫出網站資料 data/fin_watch.json 與 history/fin/updates.json。
    rows：這次快照的股票（有傳就逐項記錄財報有沒有進到評分、沒有分數的原因）。"""
    ensure(con)
    ts = config.now().isoformat(timespec="seconds")
    R = {r["stock_id"]: r for r in rows or []}
    for sid, per in con.execute("SELECT stock_id, period FROM fin_announce WHERE status='synced' AND (score_after IS NULL OR rescore_check IS NULL)").fetchall():
        sc = score_now(con, sid)
        if sc is not None:
            con.execute("UPDATE fin_announce SET score_after=?, rescored_at=? WHERE stock_id=? AND period=?", (sc, ts, sid, per))
        if rows is not None:
            chk = rescore_check(R.get(sid), per, sc)
            con.execute("UPDATE fin_announce SET rescore_check=?, rescored_at=COALESCE(rescored_at, ?) WHERE stock_id=? AND period=?",
                        (json.dumps(chk, ensure_ascii=False), ts if chk["factors_recomputed"] else None, sid, per))
    con.commit()
    s = summary(con)
    (config.DATA_DIR / "fin_watch.json").write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")), "utf-8")
    h = config.ROOT / "history" / "fin"
    h.mkdir(parents=True, exist_ok=True)
    (h / "updates.json").write_text(json.dumps({k: s[k] for k in ("generated_at", "quarter", "announced", "synced", "waiting", "slow", "corrected", "mismatch", "checks", "db_has_quarter",
                                                                  "latency_hours", "waiting_list", "rescored", "revisions", "detect", "endpoints")},
                                               ensure_ascii=False, indent=1), "utf-8")
    return s
