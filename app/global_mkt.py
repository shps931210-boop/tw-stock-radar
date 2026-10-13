"""國際市場（data/global_markets.json → 網站 data/global/markets.json）：美股主要指數、費城半導體、台積電 ADR、美元兌台幣。

給 AI 研究助理分析「台股為什麼漲跌」時查前一晚的外部市場。資料來自 FinMind（USStockPrice、TaiwanExchangeRate），
美股日期是美國當地交易日：美國 10/12 收盤 → 影響台灣 10/13 開盤。失敗的項目只記錄警告，不影響其他資料。
執行：python -m app.global_mkt（GitHub Actions 在新聞之後執行）。
"""
import json
from datetime import timedelta

from . import config

SYMBOLS = [
    ("^SOX", "費城半導體指數"),
    ("^IXIC", "那斯達克綜合指數"),
    ("^GSPC", "S&P 500 指數"),
    ("^DJI", "道瓊工業指數"),
    ("TSM", "台積電 ADR（美股）"),
]


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def summarize(symbol: str, name: str, rows: list[dict]) -> dict | None:
    """rows：FinMind USStockPrice（date, Close…），回傳最新一天與前一天比較。"""
    rows = sorted((r for r in rows if r.get("Close") not in (None, 0)), key=lambda r: r["date"])
    if len(rows) < 2:
        return None
    last, prev = rows[-1], rows[-2]
    chg = (last["Close"] / prev["Close"] - 1) * 100
    streak, sign = 0, (1 if chg > 0 else -1 if chg < 0 else 0)
    for a, b in zip(reversed(rows[:-1]), reversed(rows[1:])):
        s = 1 if b["Close"] > a["Close"] else -1 if b["Close"] < a["Close"] else 0
        if s != sign or s == 0:
            break
        streak += 1
    return {"symbol": symbol, "name": name, "date": last["date"][:10], "close": _r(last["Close"]), "prev_close": _r(prev["Close"]),
            "chg_pct": _r(chg), "streak": streak * sign, "ret_5d": _r((last["Close"] / rows[-6]["Close"] - 1) * 100) if len(rows) >= 6 else None,
            "recent": [[r["date"][:10], _r(r["Close"])] for r in rows[-6:]]}


def fx(rows: list[dict]) -> dict | None:
    """FinMind TaiwanExchangeRate（台灣銀行牌告）：用即期買賣中價。"""
    ok = sorted((r for r in rows if (r.get("spot_buy") or 0) > 0 and (r.get("spot_sell") or 0) > 0), key=lambda r: r["date"])
    if len(ok) < 2:
        return None
    mid = lambda r: (r["spot_buy"] + r["spot_sell"]) / 2
    a, b = mid(ok[-2]), mid(ok[-1])
    return {"name": "美元兌新台幣（台灣銀行即期中價）", "date": ok[-1]["date"][:10], "rate": _r(b, 3), "prev": _r(a, 3), "chg": _r(b - a, 3),
            "note": "數字變大代表新台幣貶值"}


def build(get, today) -> dict:
    """get(dataset, data_id, start) → rows。"""
    start = (today - timedelta(days=14)).isoformat()
    out, warn = [], []
    for sym, name in SYMBOLS:
        try:
            s = summarize(sym, name, get("USStockPrice", sym, start))
            if s:
                out.append(s)
            else:
                warn.append(f"{name}：資料不足")
        except Exception as e:
            warn.append(f"{name}：{str(e)[:120]}")
    try:
        f = fx(get("TaiwanExchangeRate", "USD", start))
    except Exception as e:
        f = None
        warn.append(f"美元匯率：{str(e)[:120]}")
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds") + "+08:00",
            "source": "FinMind（USStockPrice、TaiwanExchangeRate）",
            "note": "美股日期是美國當地交易日；美國前一晚的收盤通常影響隔天台股開盤。這是外部市場資料，股栗子沒有另外驗證。",
            "markets": out, "fx_usd": f, "warnings": warn}


def main() -> int:
    from .sources.finmind import FinMindSource
    cfg = config.load()
    fm = FinMindSource(cfg.get("finmind_token", ""), cfg.get("finmind_interval_sec", 0.7))
    res = build(lambda ds, sid, start: fm._get(ds, sid, start), config.today())
    (config.DATA_DIR / "global_markets.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print("國際市場：", ", ".join(f"{m['name']} {m['date']} {m['chg_pct']:+.2f}%" for m in res["markets"]) or "沒有資料", res["warnings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
