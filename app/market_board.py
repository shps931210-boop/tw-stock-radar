"""首頁市場看板（snapshot.market_board）：上市、上櫃分開的漲跌家數，真的漲停／跌停名單，漲跌幅與成交金額排行。

漲跌停不用「漲幅 ≥ 9.5%」猜，而是照證交所規則算出當天的漲停價、跌停價，收盤價等於它才算：
- 參考價：前一天的收盤價（原始價格，沒有還原權息）；除權息當天改用交易所公布的「除權息參考價」。
- 漲停價 = 參考價 × 1.10 往下取到合法的升降單位；跌停價 = 參考價 × 0.90 往上取到合法的升降單位。
- 升降單位（股票）：未滿 10 元 0.01、10–50 元 0.05、50–100 元 0.1、100–500 元 0.5、500–1000 元 1、1000 元以上 5。
限制：新上市（櫃）前五個交易日沒有漲跌幅限制，這裡排除不判斷；減資、變更面額等其他重新訂參考價的情況沒有資料，
可能誤判，所以清單標示「依規則推算」。
"""
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from . import db

TICKS = [(10, "0.01"), (50, "0.05"), (100, "0.1"), (500, "0.5"), (1000, "1")]
NEW_LISTING_DAYS = 5
TOP_N = 20
MARKETS = {"twse": "上市", "tpex": "上櫃"}


def tick(price: Decimal) -> Decimal:
    for lim, t in TICKS:
        if price < lim:
            return Decimal(t)
    return Decimal("5")


def _snap(x: Decimal, rounding) -> Decimal:
    """取到 x 所在價位的升降單位；若取整後跨到別的價位區間，用新區間的單位再取一次。"""
    for _ in range(3):
        t = tick(x)
        y = (x / t).to_integral_value(rounding=rounding) * t
        if tick(y) == t:
            return y
        x = y
    return y


def limits(ref: float) -> tuple[float, float]:
    """參考價 → (漲停價, 跌停價)。"""
    r = Decimal(str(ref))
    up = _snap(r * Decimal("1.1"), ROUND_FLOOR)
    dn = _snap(r * Decimal("0.9"), ROUND_CEILING)
    return float(up), float(dn)


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def build(snap: dict, dbname: str = "stocks.db") -> dict:
    day = (snap.get("data_dates") or {}).get("prices")
    rows = {s["stock_id"]: s for s in snap.get("stocks", []) if s.get("is_common")}
    out = {"schema_version": 1, "date": day, "markets": {}, "limit_up": [], "limit_down": [], "touched_up": 0, "touched_down": 0,
           "gainers": [], "losers": [], "by_value": [], "warnings": [],
           "method": "漲跌停依證交所規則推算：前一日收盤價（除權息日用除權息參考價）× 1.10／0.90，再取到合法升降單位；收盤價等於漲停價才算漲停。"
                     "新上市（櫃）前五個交易日沒有漲跌幅限制，不列入。減資等其他重新訂參考價的情況可能誤判。",
           "note": "上漲家數是市場廣度，不等於漲停家數；漲跌停只代表收盤鎖在限制價。"}
    if not day or not rows:
        out["warnings"].append("沒有最新行情，無法計算市場統計")
        return out
    with db.connect(dbname) as con:
        dates = [d for (d,) in con.execute("SELECT DISTINCT date FROM prices WHERE date<=? ORDER BY date DESC LIMIT 2", (day,))]
        prev_day = dates[1] if len(dates) > 1 else None
        first_day = con.execute("SELECT MIN(date) FROM prices").fetchone()[0]
        today = {r[0]: r for r in con.execute("SELECT stock_id, open, high, low, close, volume, value FROM prices WHERE date=?", (day,))}
        prev = {}
        if prev_day:
            for sid, c in con.execute("""SELECT p.stock_id, p.close FROM prices p JOIN (SELECT stock_id, MAX(date) d FROM prices
                                         WHERE date<? GROUP BY stock_id) m ON p.stock_id=m.stock_id AND p.date=m.d""", (day,)):
                prev[sid] = c
        nrows = dict(con.execute("SELECT stock_id, COUNT(*) FROM prices WHERE date<=? GROUP BY stock_id", (day,)).fetchall())
        firsts = dict(con.execute("SELECT stock_id, MIN(date) FROM prices GROUP BY stock_id").fetchall())
        refs = dict(con.execute("SELECT stock_id, ref_price FROM exrights WHERE date=? AND ref_price>0", (day,)).fetchall())
        news = {}
        for sid, d, title, src, link in con.execute(
                "SELECT stock_id, date, title, source, link FROM news WHERE date>=? ORDER BY date DESC", (prev_day or day,)):
            news.setdefault(sid, {"date": d, "title": title, "source": src, "link": link})
    if not prev_day:
        out["warnings"].append("只有一天的行情，無法計算漲跌")
        return out
    stats = {m: {"label": lab, "n": 0, "up": 0, "down": 0, "flat": 0, "no_trade": 0, "limit_up": 0, "limit_down": 0,
                 "value": 0.0, "new_listing": 0} for m, lab in MARKETS.items()}
    items, excluded_new = [], 0
    for sid, s in rows.items():
        m = s.get("market")
        if m not in stats:
            continue
        st = stats[m]
        t = today.get(sid)
        if not t or not t[4]:
            st["no_trade"] += 1
            continue
        _, o, h, lo, c, vol, val = t
        base = refs.get(sid) or prev.get(sid)
        if not base:
            continue
        st["n"] += 1
        st["value"] += val or 0
        chg = (c / base - 1) * 100
        st["up" if chg > 0 else "down" if chg < 0 else "flat"] += 1
        new = nrows.get(sid, 0) <= NEW_LISTING_DAYS and firsts.get(sid, first_day) > first_day
        it = {"stock_id": sid, "name": s.get("name"), "market": m, "industry": s.get("industry"), "close": _r(c),
              "ref": _r(base), "chg_pct": _r(chg), "volume_lots": _r((vol or 0) / 1000, 0), "value": _r(val, 0),
              "ex_rights": sid in refs}
        a = ((s.get("ai") or {}).get("modes") or {}).get("balanced") or {}
        it["ai_score"], it["ai_rank"] = _r(a.get("score"), 0), a.get("rank")
        if news.get(sid):
            it["news"] = news[sid]
        if new:
            st["new_listing"] += 1
            excluded_new += 1
        else:
            up, dn = limits(base)
            it["limit_up_price"], it["limit_down_price"] = up, dn
            eps = 1e-6
            if abs(c - up) < eps:
                st["limit_up"] += 1
                out["limit_up"].append(it)
            elif abs(c - dn) < eps:
                st["limit_down"] += 1
                out["limit_down"].append(it)
            if h is not None and abs(h - up) < eps and abs(c - up) >= eps:
                out["touched_up"] += 1   # 盤中摸到漲停、收盤打開
            if lo is not None and abs(lo - dn) < eps and abs(c - dn) >= eps:
                out["touched_down"] += 1
        items.append(it)
    for st in stats.values():
        st["value"] = _r(st["value"], 0)
        st["up_ratio"] = _r(st["up"] / st["n"] * 100, 1) if st["n"] else None
    tot = {k: sum(st[k] for st in stats.values()) for k in ("n", "up", "down", "flat", "no_trade", "limit_up", "limit_down", "value", "new_listing")}
    tot["label"] = "全部"
    tot["up_ratio"] = _r(tot["up"] / tot["n"] * 100, 1) if tot["n"] else None
    out["markets"] = {**stats, "all": tot}
    out["prev_date"] = prev_day
    key = lambda f, rev=True: sorted([i for i in items if i.get(f) is not None], key=lambda i: i[f], reverse=rev)
    out["limit_up"].sort(key=lambda i: -(i["value"] or 0))
    out["limit_down"].sort(key=lambda i: -(i["value"] or 0))
    out["gainers"] = key("chg_pct")[:TOP_N]
    out["losers"] = key("chg_pct", False)[:TOP_N]
    out["by_value"] = key("value")[:TOP_N]
    if excluded_new:
        out["warnings"].append(f"{excluded_new} 檔新上市（櫃）前五日沒有漲跌幅限制，不列入漲跌停判斷")
    missing = sum(st["no_trade"] for st in stats.values())
    if missing:
        out["warnings"].append(f"{missing} 檔今天沒有成交或沒有行情資料，不列入家數")
    return out
