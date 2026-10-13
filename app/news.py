"""市場新聞中心（data/news/latest.json）與個股新聞。

兩種資料，畫面上分開標示：
- 公司公告（announcement）：證交所 OpenAPI「上市公司每日重大訊息」t187ap04_L、櫃買中心 OpenAPI 上櫃重大訊息。
  官方資料，公司代號由公告本身提供，不靠標題猜測。這個端點只提供「當天」的公告，所以每次更新都存起來，慢慢累積歷史。
- 新聞（news）：FinMind「台股相關新聞」TaiwanStockNews，每次請求只提供一天。股票代號由 FinMind 標註。
  只顯示標題、來源、時間與原文連結，內容版權屬於原媒體。
  先試一次抓全市場（不帶股票代號）；不行就改抓重點股票（AI 均衡型前 50 名、技術訊號、市值前 100 大、會員自選股）。

偏多／偏空只用標題關鍵字判讀，大部分標題會是「無法判定」；這不是 AI 情緒分析。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

from . import config, db
from .sources.twse import HEADERS, OPENAPI, roc_date

SCHEMA = """CREATE TABLE IF NOT EXISTS news_items (id TEXT PRIMARY KEY, kind TEXT, stock_id TEXT, date TEXT, time TEXT, title TEXT,
  summary TEXT, source TEXT, link TEXT, rule TEXT, fetched_at TEXT);
CREATE INDEX IF NOT EXISTS news_items_date ON news_items (date);
CREATE INDEX IF NOT EXISTS news_items_stock ON news_items (stock_id, date);"""
TWSE_ANN = f"{OPENAPI}/opendata/t187ap04_L"
TPEX_ANN = ["https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O", "https://www.tpex.org.tw/openapi/v1/t187ap04_O"]
MOPS = "https://mops.twse.com.tw/mops/#/web/t05sr01_1"
KEEP_DAYS = 30
POS = ["創新高", "新高", "成長", "年增", "大單", "接單", "調升", "調高", "上修", "利多", "看旺", "擴產", "優於預期", "轉盈", "加碼", "受惠", "突破", "目標價"]
NEG = ["衰退", "年減", "下滑", "下修", "調降", "利空", "砍單", "虧損", "轉虧", "疲軟", "減產", "低於預期", "庫存調整", "違約", "訴訟", "裁罰", "處分", "停工", "轉讓持股", "下調"]


def tone(title: str) -> str:
    """標題關鍵字：同時有正負或都沒有 → 無法判定。"""
    p, n = any(k in title for k in POS), any(k in title for k in NEG)
    return "positive" if p and not n else "negative" if n and not p else "unclear"


def _id(*parts) -> str:
    return hashlib.sha1("|".join(str(x) for x in parts).encode()).hexdigest()[:16]


def _pick(rec: dict, *names):
    """依 names 的順序找欄位：先完全相同（忽略前後空白，例如「主旨 」），再找包含。"""
    keys = {str(k).strip(): v for k, v in rec.items()}
    for n in names:
        if n in keys:
            return keys[n]
    for n in names:
        for k, v in keys.items():
            if n in k:
                return v
    return None


def _date(v) -> str | None:
    """1151013、115/10/13、2026-10-13、20261013 → 2026-10-13。"""
    s = str(v or "").strip().replace("/", "-")
    if len(s) == 7 and s.isdigit():
        return roc_date(s)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    p = s.split("-")
    if len(p) == 3 and all(x.isdigit() for x in p):
        y = int(p[0])
        return f"{y + 1911 if y < 1911 else y}-{int(p[1]):02d}-{int(p[2][:2]):02d}"
    return None


def parse_announcements(rows: list[dict], source: str) -> list[dict]:
    """重大訊息 → news_items。欄位名稱用「包含」比對，中英文欄位都可以（櫃買 OpenAPI 可能是英文）。"""
    out, now = [], config.now().isoformat(timespec="seconds")
    for r in rows:
        sid = str(_pick(r, "公司代號", "SecuritiesCompanyCode", "CompanyCode") or "").strip()
        title = str(_pick(r, "主旨", "Subject") or "").strip()
        d = _date(_pick(r, "發言日期", "AnnouncementDate", "Date"))
        if not sid or not title or not d:
            continue
        t = str(_pick(r, "發言時間", "AnnouncementTime", "Time") or "").strip().zfill(6)
        tm = f"{t[:2]}:{t[2:4]}" if t.isdigit() else None
        out.append({"id": _id("ann", sid, d, t, title), "kind": "announcement", "stock_id": sid, "date": d, "time": tm, "title": title[:300],
                    "summary": str(_pick(r, "說明", "Description") or "").strip()[:600], "source": source, "link": MOPS,
                    "rule": str(_pick(r, "符合條款", "Clause") or "").strip()[:40], "fetched_at": now})
    return out


def parse_finmind(rows: list[dict]) -> list[dict]:
    out, now = [], config.now().isoformat(timespec="seconds")
    for r in rows:
        title, sid = (r.get("title") or "").strip(), str(r.get("stock_id") or "").strip()
        if not title or not sid:
            continue
        dt = str(r.get("date") or "")
        out.append({"id": _id("news", sid, r.get("link") or title), "kind": "news", "stock_id": sid, "date": dt[:10],
                    "time": dt[11:16] or None, "title": title[:300], "summary": (r.get("description") or "").strip()[:300],
                    "source": (r.get("source") or "").strip()[:100], "link": r.get("link") or None, "rule": None, "fetched_at": now})
    return out


def _priority(con, snap: dict | None) -> list[str]:
    ids: list[str] = []
    stocks = (snap or {}).get("stocks") or []
    common = [s for s in stocks if s.get("is_common")]
    rk = lambda s: (((s.get("ai") or {}).get("modes") or {}).get("balanced") or {}).get("rank")  # noqa: E731
    ranked = sorted((s for s in common if rk(s)), key=rk)
    ids += [s["stock_id"] for s in ranked[:50]]
    ids += [s["stock_id"] for s in common if (s.get("timing") or {}).get("status") == "buy"]
    ids += [s["stock_id"] for s in sorted(common, key=lambda s: -(s.get("market_cap") or 0))[:100]]
    return list(dict.fromkeys(ids))


def update(cfg: dict, errors: list, snap: dict | None = None, extra_ids: list[str] | None = None) -> dict:
    """抓公告與新聞，存進資料庫。回傳統計（寫進 data/news/status）。任何來源失敗只記錄警告。"""
    import requests
    from .sources import http
    from .sources.finmind import FinMindSource
    stat = {"announcements": {}, "news": {"mode": None, "requests": 0, "rows": 0}}
    s = requests.Session()
    s.headers.update(HEADERS)
    with db.connect() as con:
        con.executescript(SCHEMA)
        # 公司公告
        try:
            rows = parse_announcements(http.get(s, TWSE_ANN, timeout=60).json(), "公開資訊觀測站（上市重大訊息）")
            db.upsert(con, "news_items", rows)
            stat["announcements"]["twse"] = len(rows)
        except Exception as e:
            errors.append(f"上市重大訊息: {http.describe(e)}")
        if "tpex" in cfg.get("markets", ["twse", "tpex"]):
            for u in TPEX_ANN:
                try:
                    js = http.get(s, u, timeout=60, tries=1).json()
                    if not isinstance(js, list):
                        raise ValueError("不是清單")
                    rows = parse_announcements(js, "公開資訊觀測站（上櫃重大訊息）")
                    db.upsert(con, "news_items", rows)
                    stat["announcements"]["tpex"] = len(rows)
                    stat["announcements"]["tpex_url"] = u
                    break
                except Exception as e:
                    stat["announcements"].setdefault("tpex_errors", []).append(f"{u.rsplit('/', 1)[-1]}: {http.describe(e)}")
            if "tpex" not in stat["announcements"]:
                errors.append("上櫃重大訊息: 找不到可用的端點（見 news/latest.json 的 status）")
        # 新聞：FinMind 每次請求只給一天，抓今天
        fm = FinMindSource(cfg.get("finmind_token", ""), cfg.get("finmind_interval_sec", 0.7))
        today = config.today()
        days = [(today - timedelta(days=i)).isoformat() for i in (0,)]
        # FinMind 的 TaiwanStockNews 一定要指定股票（不給代號會回 400，2026-10-13 實測），所以逐檔抓重點股票
        got, ids = [], list(dict.fromkeys(_priority(con, snap) + (extra_ids or [])))[: cfg.get("news_max_stocks", 150)]
        stat["news"]["mode"] = "stocks"
        stat["news"]["stocks"] = len(ids)
        for sid in ids:
            t = db.fetched_at(con, "fm_news", sid)  # 同一檔 6 小時內抓過就跳過，省 FinMind 額度（股價更新也要用）
            if t and config.now() - t < timedelta(hours=6):
                continue
            try:
                got += fm._get("TaiwanStockNews", sid, days[0])
                stat["news"]["requests"] += 1
                db.log_fetch(con, "fm_news", sid)
            except Exception as e2:
                stat["news"]["error"] = str(e2)[:200]
                if "請求次數" in stat["news"]["error"]:
                    break
        rows = parse_finmind(got)
        db.upsert(con, "news_items", rows)
        stat["news"]["rows"] = len(rows)
        cut = (today - timedelta(days=KEEP_DAYS * 3)).isoformat()
        con.execute("DELETE FROM news_items WHERE date < ?", (cut,))
        db.log_fetch(con, "news_center", today.isoformat(), True)
        db.set_meta(con, "news_status", json.dumps(stat, ensure_ascii=False))
    return stat


def fetch_stock(sid: str, cfg: dict | None = None) -> None:
    """本機版打開個股頁時，順便抓這檔今天的新聞（3 小時內抓過就不再抓）。"""
    from .sources.finmind import FinMindSource
    cfg = cfg or config.load()
    with db.connect() as con:
        con.executescript(SCHEMA)
        t = db.fetched_at(con, "fm_news", sid)
        if t and config.now() - t < timedelta(hours=3):
            return
        try:
            db.upsert(con, "news_items", parse_finmind(FinMindSource(cfg.get("finmind_token", ""))._get("TaiwanStockNews", sid, config.today().isoformat())))
            db.log_fetch(con, "fm_news", sid)
        except Exception:
            db.log_fetch(con, "fm_news", sid, ok=False)


def for_stock(con, sid: str, limit: int = 20) -> list[dict]:
    con.executescript(SCHEMA)
    cur = con.execute("SELECT id, kind, stock_id, date, time, title, summary, source, link, rule FROM news_items WHERE stock_id=? ORDER BY date DESC, time DESC LIMIT ?", (sid, limit))
    cols = [c[0] for c in cur.description]
    out = []
    for r in cur.fetchall():
        x = dict(zip(cols, r))
        if x["kind"] == "news":
            x["title"] = clean_title(x["title"], x["source"])
        x["tone"] = tone(x["title"]) if x["kind"] == "news" else None
        out.append(x)
    group_similar(out)
    return out


def clean_title(title: str, source: str | None = None) -> str:
    """去掉新聞標題結尾的「 - 媒體名稱」（FinMind 的標題常帶著來源）。"""
    t = (title or "").strip()
    if " - " in t:
        head, tail = t.rsplit(" - ", 1)
        if len(tail) <= 30 and ((source and tail.strip() == source.strip()) or len(head) >= 8):
            t = head.strip()
    return t


def _bigrams(t: str) -> set[str]:
    t = "".join(ch for ch in t if ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def group_similar(items: list[dict], threshold: float = 0.5) -> None:
    """同一檔股票、前後一天內、標題字詞重疊多的新聞歸成同一組（group = 第一篇的 id）。公告不分組。"""
    seen: list[tuple[dict, set[str]]] = []
    for it in items:
        it["group"] = it["id"] if "id" in it else None
        if it["kind"] != "news":
            continue
        bg = _bigrams(it["title"])
        for first, fb in seen:
            if first["stock_id"] == it["stock_id"] and abs(date.fromisoformat(first["date"]).toordinal() - date.fromisoformat(it["date"]).toordinal()) <= 1 \
                    and bg and fb and len(bg & fb) / len(bg | fb) >= threshold:
                it["group"] = first["group"]
                break
        else:
            seen.append((it, bg))


def latest(snap: dict, dbname: str = "stocks.db", days: int = 7, cap: int = 1500) -> dict:
    """新聞中心資料：最近幾天的公告與新聞，附上股票名稱、產業、目前的 AI 量化評分與當日漲跌。"""
    by = {s["stock_id"]: s for s in snap.get("stocks", [])}
    today = config.today()
    with db.connect(dbname) as con:
        con.executescript(SCHEMA)
        cur = con.execute("SELECT id, kind, stock_id, date, time, title, summary, source, link, rule FROM news_items WHERE date >= ? ORDER BY date DESC, time DESC LIMIT ?",
                          ((today - timedelta(days=days)).isoformat(), cap))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        st = db.get_meta(con, "news_status")
        last = con.execute("SELECT fetched_at FROM fetch_log WHERE source='news_center' AND ok=1 ORDER BY fetched_at DESC LIMIT 1").fetchone()
    items = []
    for r in rows:
        if r["kind"] == "news":
            r["title"] = clean_title(r["title"], r["source"])
        s = by.get(r["stock_id"]) or {}
        m = ((s.get("ai") or {}).get("modes") or {}).get("balanced") or {}
        items.append({**r, "name": s.get("name"), "industry": s.get("industry") or None, "tone": tone(r["title"]) if r["kind"] == "news" else None,
                      "ctx": {"close": s.get("close"), "chg_pct": s.get("chg_pct"), "price_date": s.get("date"), "score": m.get("score"), "rank": m.get("rank")} if s else None})
    group_similar(items)
    status = json.loads(st) if st else None
    warn = []
    if not last:
        warn.append("還沒有成功抓過新聞與公告")
    if status and status.get("news", {}).get("mode") == "stocks":
        warn.append(f"新聞只涵蓋 {status['news'].get('stocks')} 檔重點股票（AI 前段班、技術訊號、大型股、會員自選股），不是全市場")
    if status and "tpex" not in status.get("announcements", {}):
        warn.append("上櫃公司重大訊息還沒有接上")
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "data_as_of": today.isoformat(), "timezone": "Asia/Taipei",
            "source": "公開資訊觀測站重大訊息（證交所／櫃買中心 OpenAPI）；新聞標題與連結來自 FinMind 台股相關新聞，內容版權屬原媒體",
            "last_fetched_at": last[0] if last else None, "status": status, "warnings": warn,
            "coverage": {"days": days, "items": len(items), "announcements": sum(1 for x in items if x["kind"] == "announcement"),
                         "news": sum(1 for x in items if x["kind"] == "news")},
            "tone_note": "「標題用詞」只看標題裡有沒有正面或負面字詞，大多數標題是「無法判定」；這不是 AI 情緒分析，不能用來判斷股價方向。",
            "items": items}


def main() -> int:
    """GitHub Actions：python -m app.news（在更新資料之後、產生網站之前執行）。"""
    errors: list[str] = []
    p = config.DATA_DIR / "snapshot.json"
    snap = json.loads(p.read_text("utf-8")) if p.exists() else None
    stat = update(config.load(), errors, snap)
    print("新聞與公告：", json.dumps(stat, ensure_ascii=False))
    for e in errors:
        print("警告：", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
