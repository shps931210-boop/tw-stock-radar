"""財報與事件日曆（data/events/calendar.json）。

事件分兩種，畫面上明確區分：
- 已確認（confirmed）：官方已公告的日期。目前來源：證交所「除權除息預告表」TWT48U（上市）。
  日期過了標為「已完成」（completed）。每次更新都重新核對，記錄最後核對時間；之前有、這次預告表裡沒有的未來事件，
  標為「未再確認」（unverified），可能延期或取消，請以公開資訊觀測站為準。
- 法定期限（estimated）：依法規推估的「最晚公布日」，不是公司公告的日期：
  月營收每月 10 日前；財報 Q1 5/15、Q2 8/14、Q3 11/14、年報 3/31（一般產業，遇假日可能順延，金融保險業規定略有不同）。
- 法說會：還沒有找到可以穩定、合法自動取得的官方資料，先不顯示，不猜測。
所有日期都是台灣時間（Asia/Taipei）。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

from . import config, db
from .sources.twse import any_date, num

TZ = "Asia/Taipei"
TWT48U = "https://www.twse.com.tw/rwd/zh/exRight/TWT48U"
TWT48U_PAGE = "https://www.twse.com.tw/zh/announcement/ex-right/twt48u.html"
FIN_DEADLINES = [((5, 15), "Q1", "第一季財報"), ((8, 14), "Q2", "第二季（半年）財報"), ((11, 14), "Q3", "第三季財報"), ((3, 31), "Y", "年度財報")]


def parse_twt48u(fields: list, data: list) -> list[dict]:
    """除權除息預告表：日期、代號、名稱、權／息、現金股利、無償配股率（欄位名稱 2026-10-12 核對過）。"""
    f = [str(x).replace(" ", "") for x in fields]
    def col(*keys):
        for i, x in enumerate(f):
            if all(k in x for k in keys):
                return i
        raise KeyError(f"除權除息預告表欄位找不到 {keys}，實際欄位：{fields}")
    di, si, ni, ki, ci, sti = col("除權除息日期"), col("股票代號"), col("名稱"), col("除權息"), col("現金股利"), col("無償配股率")
    out = []
    for r in data:
        d = any_date(r[di])
        if not d:
            continue
        cash = num(r[ci])
        out.append({"stock_id": str(r[si]).strip(), "name": str(r[ni]).strip(), "date": d, "kind": str(r[ki]).strip(),
                    "cash": cash, "stock_rate": num(r[sti]), "cash_note": None if cash is not None else "待公告實際金額"})
    return out


def _title(x: dict) -> str:
    kind = {"息": "除息", "權": "除權", "權息": "除權息"}.get(x["kind"], f"除{x['kind']}")
    parts = []
    if x.get("cash"):
        parts.append(f"現金股利 {x['cash']:g} 元")
    elif x.get("cash_note"):
        parts.append(x["cash_note"])
    if x.get("stock_rate"):
        parts.append(f"無償配股率 {round(x['stock_rate'], 6):g}")
    return f"{kind}" + (f"（{'、'.join(parts)}）" if parts else "")


def store_exdiv(con, rows: list[dict], today: date) -> int:
    """寫入除權息事件。保留第一次看到的時間；這次沒出現的未來事件標為未再確認。"""
    now = config.now().isoformat(timespec="seconds")
    seen = set()
    for x in rows:
        eid = f"exdiv:{x['stock_id']}:{x['date']}"
        seen.add(eid)
        status = "completed" if x["date"] < today.isoformat() else "confirmed"
        con.execute("""INSERT INTO events (event_id, stock_id, event_type, title, event_date, status, source_name, source_url, detail,
                         first_seen, last_verified_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(event_id) DO UPDATE SET title=excluded.title, status=excluded.status, detail=excluded.detail,
                         last_verified_at=excluded.last_verified_at,
                         updated_at=CASE WHEN events.title<>excluded.title OR events.status<>excluded.status THEN excluded.updated_at ELSE events.updated_at END""",
                    (eid, x["stock_id"], "dividend", _title(x), x["date"], status, "臺灣證券交易所 除權除息預告表", TWT48U_PAGE,
                     json.dumps({k: x[k] for k in ("kind", "cash", "stock_rate", "name")}, ensure_ascii=False), now, now, now))
    # 之前預告過、這次沒出現、日期還沒到：可能延期或取消
    for (eid,) in con.execute("SELECT event_id FROM events WHERE event_type='dividend' AND event_date>=? AND status='confirmed'", (today.isoformat(),)).fetchall():
        if eid not in seen:
            con.execute("UPDATE events SET status='unverified', updated_at=? WHERE event_id=?", (now, eid))
    con.execute("UPDATE events SET status='completed' WHERE event_type='dividend' AND status='confirmed' AND event_date<?", (today.isoformat(),))
    return len(rows)


def update(cfg: dict, errors: list) -> None:
    """抓證交所除權除息預告表（上市）。失敗只記錄警告，不影響其他資料。"""
    import requests
    from .sources import http
    from .sources.twse import HEADERS
    try:
        s = requests.Session()
        s.headers.update(HEADERS)
        r = http.get(s, TWT48U, params={"response": "json"}, timeout=30)
        js = r.json()
        if js.get("stat") != "OK":
            raise ValueError(f"stat={js.get('stat')}")
        rows = parse_twt48u(js.get("fields") or [], js.get("data") or [])
    except Exception as e:
        errors.append(f"twse_exdiv_forecast: {http.describe(e)}")
        return
    with db.connect() as con:
        store_exdiv(con, rows, config.today())
        db.log_fetch(con, "twse_exdiv_forecast", config.today().isoformat(), True)


def deadlines(today: date, months_ahead: int = 2) -> list[dict]:
    """法定最晚公布日（推估）：月營收、各季財報。"""
    out = []
    first = today.replace(day=1)
    for i in range(-1, months_ahead + 1):
        m = (first + timedelta(days=32 * i)).replace(day=1)
        prev = (m - timedelta(days=1)).replace(day=1)
        out.append({"event_id": f"deadline:revenue:{prev:%Y-%m}", "stock_id": None, "event_type": "revenue",
                    "title": f"{prev.month} 月營收公布期限（所有上市櫃公司）", "event_date": m.replace(day=10).isoformat(),
                    "status": "estimated", "basis": "法定期限：每月 10 日前公布上月營收",
                    "source_name": "公開資訊觀測站（依法規推估）", "source_url": "https://mops.twse.com.tw/", "period": f"{prev:%Y-%m}"})
    for y in (today.year - 1, today.year, today.year + 1):
        for (mo, d), q, name in FIN_DEADLINES:
            dd = date(y, mo, d)
            if not (today - timedelta(days=45) <= dd <= today + timedelta(days=120)):
                continue
            fy = y - 1 if q == "Y" else y
            out.append({"event_id": f"deadline:fin:{fy}{q}", "stock_id": None, "event_type": "financials",
                        "title": f"{fy} 年{name}公布期限（一般產業）", "event_date": dd.isoformat(), "status": "estimated",
                        "basis": "法定期限，遇假日可能順延；金融保險業規定略有不同", "source_name": "公開資訊觀測站（依法規推估）",
                        "source_url": "https://mops.twse.com.tw/", "period": f"{fy}{q}"})
    return sorted(out, key=lambda x: x["event_date"])


def calendar(snap: dict, dbname: str = "stocks.db") -> dict:
    today = config.today()
    lo, hi = (today - timedelta(days=30)).isoformat(), (today + timedelta(days=120)).isoformat()
    warn = ["上櫃公司的除權息預告還沒有接上，目前只有上市公司", "法說會還沒有可以穩定自動取得的官方來源，暫不顯示"]
    names = {s["stock_id"]: s["name"] for s in snap.get("stocks", [])}
    rows = []
    with db.connect(dbname) as con:
        cur = con.execute("SELECT event_id, stock_id, event_type, title, event_date, status, source_name, source_url, first_seen, last_verified_at, updated_at "
                          "FROM events WHERE event_date BETWEEN ? AND ? ORDER BY event_date", (lo, hi))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        last = con.execute("SELECT fetched_at FROM fetch_log WHERE source='twse_exdiv_forecast' AND ok=1 ORDER BY fetched_at DESC LIMIT 1").fetchone()
        rev_month = (snap.get("data_dates") or {}).get("revenue")
        rev_done = con.execute("SELECT COUNT(DISTINCT stock_id) FROM revenue WHERE ym=?", (rev_month,)).fetchone()[0] if rev_month else 0
    if not last:
        warn.insert(0, "還沒有成功抓到除權除息預告表")
    for r in rows:
        r["name"] = names.get(r["stock_id"])
        r["timezone"] = TZ
    dl = deadlines(today)
    for d in dl:
        d["timezone"] = TZ
        if d["event_type"] == "revenue" and d["period"] == rev_month:
            d["progress"] = {"reported": rev_done, "note": f"網站已收到 {rev_done} 家公司的 {rev_month} 營收"}
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "data_as_of": today.isoformat(),
            "timezone": TZ, "source": "臺灣證券交易所 除權除息預告表（上市）；月營收與財報期限依法規推估",
            "last_verified_at": last[0] if last else None, "warnings": warn,
            "status_labels": {"confirmed": "已確認", "completed": "已完成", "estimated": "法定期限（推估）", "unverified": "未再確認（可能延期或取消）"},
            "events": rows + dl}
