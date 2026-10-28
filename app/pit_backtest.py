"""AI 多因子歷史回測（point-in-time）：用「當時已經公布」的資料重建過去每個月的 AI 排名，再模擬持有前 N 名。

和每日存檔的事後追蹤（app/track.py）不同，這是回測：模型規則是 2026-10 訂的，套回過去的資料計算。
規則在看到回測結果之前就已固定，沒有用回測結果調整權重或門檻。

不偷看未來（point-in-time）：
- 股價：只用調整日當天（含）以前的收盤資料；除權息還原只用當天以前已發生的事件。
- 季報：要過法定公布期限的隔天才算「已知」（Q1 5/15、Q2 8/14、Q3 11/14、年報隔年 3/31；金融業 Q2 延到 8/31）。
  公司常常提早公布，用期限比較保守，不會用到當時還沒公布的財報。
- 月營收：隔月 10 日公布，11 日起才使用。
- 估值（本益比、股價淨值比）：交易所的歷史估值資料只從 2026-08 開始有，所以改用當時的股價 ÷ 當時已公布的近四季 EPS、
  股價 × 股數 ÷ 股東權益計算，和交易所的數字可能有小差異。
- 股利：只用前一年以前的資料。處置股票沒有歷史紀錄，回測不排除處置股。
限制（網站會列出）：FinMind 的財報是更正後的版本，不是當時第一次公布的數字；約一半股票有財報資料。

交易規則：每月第一個交易日開盤買進前一個交易日收盤算出的前 N 名（等權），持有到下個月第一個交易日開盤賣出。
進場日沒有成交、開盤就鎖漲停、成交金額不到 300 萬元的股票買不到，那一份資金留現金。
持有期間停牌或下市：用最後一筆收盤價計算，之後留現金（不剔除，避免存活者偏差）。
換股成本：新買進的股票扣一次來回成本（手續費 ×2 + 證交稅，和每日追蹤相同）；續抱的股票不扣。
基準：0050 同一段期間（開盤到開盤，還原除權息）；另列「所有可排名股票等權」作為全市場基準。

執行：python -m app.pit_backtest（GitHub Actions「AI 歷史回測」每週執行），輸出 history/backtest/ai_pit.json。
"""
import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import ai_model, config, db, metrics, snapshot, track

METHOD = 1
START = "2025-03-01"  # 舊的固定起點；現在改由 auto_start 依資料長度決定，這個值只當作沒有資料時的預設
WARMUP_DAYS = 260      # 動能因子要一年股價：股價第一天往後約 260 個交易日，大多數股票才算得出來
FIN_COVER = 0.6        # 季報覆蓋率：某一季有季報的股票達到最多那一季的六成，才算「那時候的財報可用」
TOP_NS = (5, 10, 20)
COST = track.COST
THIN_VALUE = track.THIN_VALUE
BENCH = "0050"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "ai_pit.json"


# ---------------- 資料公布時間 ----------------
def fin_available(period: str, is_financial: bool = False) -> str:
    """季報（例如 2025-Q2）最早可以使用的日期：法定期限的隔天。"""
    y, q = period.split("-Q")
    y, q = int(y), int(q)
    if q == 1:
        return f"{y}-05-16"
    if q == 2:
        return f"{y}-09-01" if is_financial else f"{y}-08-15"
    if q == 3:
        return f"{y}-11-15"
    return f"{y + 1}-04-01"


def rev_available(ym: str) -> str:
    """月營收（例如 2025-07）最早可以使用的日期：隔月 11 日。"""
    p = pd.Period(ym, "M") + 1
    return f"{p.year}-{p.month:02d}-11"


# ---------------- 讀資料 ----------------
def load(dbname: str = "stocks.db") -> dict:
    with db.connect(dbname) as con:
        R = lambda sql: db.read(con, sql)
        px = R("SELECT * FROM prices ORDER BY stock_id, date")
        D = {"stocks": R("SELECT * FROM stocks"), "fin": R("SELECT * FROM financials"), "rev": R("SELECT * FROM revenue ORDER BY stock_id, ym"),
             "div": R("SELECT * FROM dividends"), "ex": R("SELECT * FROM exrights"), "idx": R("SELECT * FROM index_prices ORDER BY date")}
    G = lambda df: {k: g.reset_index(drop=True) for k, g in df.groupby("stock_id")} if len(df) else {}
    D["px"] = G(px)
    D["fin"] = G(D["fin"])
    D["rev"] = G(D["rev"])
    D["div"] = G(D["div"])
    D["ex"] = G(D["ex"])
    D["info"] = D["stocks"].set_index("stock_id").to_dict("index") if len(D["stocks"]) else {}
    D["dates"] = sorted(px["date"].unique())
    D["idx_s"] = D["idx"].set_index("date")["close"] if len(D["idx"]) else None
    # 報酬用：全期間還原除權息的價格（比例在任何一段期間都正確）
    D["rawi"] = {sid: p.set_index("date") for sid, p in D["px"].items()}
    D["adj"] = {sid: fix_gaps(metrics.adjust_prices(p, D["ex"].get(sid, pd.DataFrame()))[0]).set_index("date") for sid, p in D["px"].items()}
    return D


GAP = (0.89, 1.11)  # 台股漲跌幅 10%：開盤價和前一天收盤差超過這個範圍，一定是參考價改了（分割、減資、缺除權息資料）


def fix_gaps(px: pd.DataFrame) -> pd.DataFrame:
    """報酬計算用：把分割、減資等造成的跳空當成公司行動還原（例如 0050 在 2025-06-18 一拆四）。
    只用在持有報酬；選股評分照網站目前的模型，不另外處理。"""
    if len(px) < 2:
        return px
    px = px.sort_values("date").reset_index(drop=True)
    r = (px["open"] / px["close"].shift()).to_numpy()
    factor = np.ones(len(px))
    for k in np.flatnonzero((r < GAP[0]) | (r > GAP[1])):
        factor[:k] *= r[k]
    if (factor != 1).any():
        px = px.copy()
        for c in ("open", "high", "low", "close"):
            px[c] = px[c] * factor
    return px


def auto_start(D: dict) -> str:
    """回測起點：股價累積滿一年（動能因子算得出來）且季報已大致補齊的下一個月 1 日。
    夜間長期歷史回補把股價、財報往前補之後，起點會自動跟著往前，不必改程式。"""
    dates = D["dates"]
    if not dates:
        return START
    px = dates[min(WARMUP_DAYS, len(dates) - 1)]
    cnt: dict = {}
    for g in D["fin"].values():
        for p in g["period"].dropna().unique():
            cnt[p] = cnt.get(p, 0) + 1
    ok = sorted(p for p, n in cnt.items() if n >= FIN_COVER * max(cnt.values())) if cnt else []
    fin = fin_available(ok[0]) if ok else px
    d = max(px, fin)
    y, m = int(d[:4]), int(d[5:7]) + 1
    return f"{y + (m > 12)}-{(m - 1) % 12 + 1:02d}-01"


def rebalance_dates(dates: list[str], start: str = START) -> list[tuple[str, str]]:
    """每月第一個交易日（進場日）與前一個交易日（訊號日）。"""
    out, seen = [], set()
    for i, d in enumerate(dates):
        if d < start or i == 0 or d[:7] in seen:
            continue
        seen.add(d[:7])
        out.append((dates[i - 1], d))
    return out


# ---------------- 某一天的歷史評分 ----------------
def frame_at(D: dict, sig: str, max_rows: int = 520) -> pd.DataFrame:
    rows, empty = [], pd.DataFrame()
    idx = D["idx_s"][D["idx_s"].index <= sig] if D["idx_s"] is not None else None
    sig_year = int(sig[:4])
    for sid, p in D["px"].items():
        if not metrics.is_common_id(sid):
            continue
        k = int(np.searchsorted(p["date"].to_numpy(), sig, side="right"))
        if k == 0 or p["date"].iloc[k - 1] < (pd.Timestamp(sig) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"):
            continue  # 訊號日前十天都沒有成交：當時已下市或長期停牌
        pp = p.iloc[max(0, k - max_rows):k].reset_index(drop=True)
        info = D["info"].get(sid, {})
        m = {"stock_id": sid, "name": info.get("name", sid), "industry": info.get("industry") or "", "market": info.get("market") or ""}
        m["is_common"] = True
        m["is_financial"] = "金融" in m["industry"]
        m["disposition"] = False
        ex = D["ex"].get(sid, empty)
        ex = ex[ex["date"] <= sig] if len(ex) else ex
        adj, m["adj_events"] = metrics.adjust_prices(pp, ex)
        m.update(metrics.price_metrics(adj, idx))
        close = float(pp["close"].iloc[-1])
        m["close"] = close
        fin = D["fin"].get(sid, empty)
        if len(fin):
            fin = fin[fin["period"].map(lambda x: fin_available(x, m["is_financial"]) <= sig)]
        rev = D["rev"].get(sid, empty)
        if len(rev):
            rev = rev[rev["ym"].map(rev_available) <= sig]
        div = D["div"].get(sid, empty)
        if len(div):
            div = div[div["year"] < sig_year]
        f0 = metrics.fundamental_metrics(fin, rev, div, empty)
        val = empty
        if len(fin):
            last = fin.sort_values("period_end").iloc[-1]
            eps, eq, cap = f0.get("eps_ttm"), metrics._f(last.get("equity")), metrics._f(last.get("share_capital"))
            per = close / eps if eps and eps > 0 else None
            pbr = close * cap / 10 / eq if eq and eq > 0 and cap and cap > 0 else None
            dy = f0.get("div_cash_last") / close * 100 if f0.get("div_cash_last") else None
            val = pd.DataFrame([{"date": m.get("date"), "per": per or 0, "pbr": pbr or 0, "dy": dy}])
        m.update(metrics.fundamental_metrics(fin, rev, div, val))
        rows.append(m)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == object and not df[c].map(lambda x: isinstance(x, (bool, str, list))).any():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def scored_rows(D: dict, sig: str, cfg: dict) -> list[dict]:
    """訊號日收盤後、用當時資料算出的每檔股票因子分數與三種模型分數（和網站同一套程式）。"""
    df = frame_at(D, sig)
    df, _ = snapshot.score_frame(df, cfg, date.fromisoformat(sig))
    rows = [{k: snapshot._clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    ai_model.annotate(rows)
    return rows


def ranks_at(D: dict, sig: str, cfg: dict, rows: list[dict] | None = None) -> dict:
    """訊號日收盤後的三種模型排名：{mode: [(stock_id, score), ...]}，以及可排名股票數。"""
    rows = scored_rows(D, sig, cfg) if rows is None else rows
    out = {}
    for mode in ai_model.MODES:
        rk = sorted(((r["stock_id"], r["ai"]["modes"][mode]["score"], r["ai"]["modes"][mode]["rank"]) for r in rows if r["ai"]["modes"][mode]["rank"]),
                    key=lambda x: x[2])
        out[mode] = [(sid, sc) for sid, sc, _ in rk]
    out["_n"] = {"stocks": len(rows), "base_ok": int(sum(bool(r.get("base_ok")) for r in rows)),
                 "complete": int(sum(bool(r.get("complete")) for r in rows))}
    return out


# ---------------- 持有報酬 ----------------
def _limit_up(prev_close: float) -> float:
    from .market_board import limits
    return limits(prev_close)[0]


def window(dates: list[str], sig: str, h: int) -> tuple[str, str] | None:
    """唯一的持有期定義：訊號日 sig 收盤後 → 下一個交易日開盤進場 → 進場後第 h 個交易日開盤出場。
    dates 是真正有行情的交易日（休市、颱風停市都不在裡面），所以 h 一律以實際交易日計算。還沒到期回傳 None。"""
    if sig not in dates:
        return None
    i = dates.index(sig)
    if i + 1 + int(h) >= len(dates):
        return None
    return dates[i + 1], dates[i + 1 + int(h)]


def close_return(D: dict, sid: str, a: str, b: str) -> float | None:
    """a 收盤到 b 收盤的還原報酬（和只有收盤價的加權指數比較時用，兩邊同一段期間）。"""
    x = D["adj"].get(sid)
    if x is None or a not in x.index or b not in x.index or not x.at[a, "close"]:
        return None
    return float(x.at[b, "close"] / x.at[a, "close"] - 1)


def hold_return(D: dict, sid: str, entry: str, exit_: str) -> tuple[float | None, str]:
    """進場日開盤 → 出場日開盤的還原報酬。回傳 (報酬, 狀態)；None 代表買不到。"""
    a = D["adj"].get(sid)
    if a is None or entry not in a.index:
        return None, "not_traded"
    r = D["rawi"][sid]
    row = r.loc[entry]
    if not row["volume"] or not row["open"]:
        return None, "not_traded"
    if (row["value"] or 0) < THIN_VALUE:
        return None, "thin"
    prev = r[r.index < entry]
    if len(prev) and row["open"] == row["high"] == row["low"] and abs(row["open"] - _limit_up(float(prev["close"].iloc[-1]))) < 1e-6:
        return None, "locked"
    o = float(a.loc[entry, "open"])
    if exit_ in a.index and a.loc[exit_, "open"]:
        return float(a.loc[exit_, "open"]) / o - 1, "ok"
    held = a[(a.index >= entry) & (a.index < exit_)]
    return float(held["close"].iloc[-1]) / o - 1, "stale"  # 停牌或下市：用最後收盤價，之後留現金


def _bench(D: dict, entry: str, exit_: str) -> float | None:
    a = D["adj"].get(BENCH)
    if a is None or entry not in a.index or exit_ not in a.index:
        return None
    return float(a.loc[exit_, "open"] / a.loc[entry, "open"] - 1)


def _stats(rets: list[float], bench: list[float | None]) -> dict:
    r = np.array(rets, float)
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    years = len(r) / 12
    pairs = [(x, b) for x, b in zip(rets, bench) if b is not None]
    bc = np.cumprod(1 + np.array([b for _, b in pairs], float)) if pairs else None
    return {"periods": len(r), "total": round(float(curve[-1] - 1) * 100, 2) if len(r) else None,
            "cagr": round(float(curve[-1] ** (1 / years) - 1) * 100, 2) if len(r) >= 6 else None,
            "mdd": round(float((curve / peak - 1).min()) * 100, 2) if len(r) else None,
            "win_vs_bench": round(float(np.mean([x > b for x, b in pairs])) * 100, 1) if pairs else None,
            "avg_excess": round(float(np.mean([x - b for x, b in pairs])) * 100, 2) if pairs else None,
            "bench_total": round(float(bc[-1] - 1) * 100, 2) if bc is not None else None,
            "vol_ann": round(float(r.std(ddof=1) * np.sqrt(12)) * 100, 2) if len(r) > 2 else None}


def model_hash() -> str:
    src = (Path(__file__).resolve().parent / "ai_model.py").read_text("utf-8") + (Path(__file__).resolve().parent / "snapshot.py").read_text("utf-8")
    cfg = config.load()
    src += json.dumps({k: cfg.get(k) for k in ("composite_weights", "base_filter")}, sort_keys=True)
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def run(dbname: str = "stocks.db", start: str | None = None, log=print) -> dict:
    t0 = time.time()
    cfg = config.load()
    D = load(dbname)
    start = start or auto_start(D)
    rb = rebalance_dates(D["dates"], start)
    periods = list(zip(rb, rb[1:]))  # 只算已經走完的月份
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(periods)} 個完整月份")
    modes = list(ai_model.MODES)
    res = {m: {n: {"rets": [], "picks": []} for n in TOP_NS} for m in modes}
    uni = {m: [] for m in modes}
    bench, timeline, prev = [], [], {(m, n): set() for m in modes for n in TOP_NS}
    for (sig, entry), (_, exit_) in periods:
        rk = ranks_at(D, sig, cfg)
        b = _bench(D, entry, exit_)
        bench.append(b)
        timeline.append({"signal": sig, "entry": entry, "exit": exit_, "bench": None if b is None else round(b * 100, 2), **rk["_n"]})
        cache: dict = {}
        hr = lambda sid: cache.setdefault(sid, hold_return(D, sid, entry, exit_))
        for m in modes:
            ids = [sid for sid, _ in rk[m]]
            got = [hr(sid)[0] for sid in ids]
            uni[m].append(float(np.mean([x for x in got if x is not None])) if any(x is not None for x in got) else 0.0)
            for n in TOP_NS:
                top = rk[m][:n]
                legs = []
                for sid, sc in top:
                    r, st = hr(sid)
                    new = sid not in prev[(m, n)]
                    legs.append({"id": sid, "score": sc, "ret": None if r is None else round(r * 100, 2), "status": st, "new": new})
                w = 1 / n if top else 0
                gross = sum((l["ret"] or 0) / 100 * w for l in legs if l["ret"] is not None)
                cost = sum(COST * w for l in legs if l["ret"] is not None and l["new"])
                res[m][n]["rets"].append(gross - cost)
                res[m][n]["picks"].append({"entry": entry, "ret": round((gross - cost) * 100, 2), "legs": legs if n == 20 else None,
                                           "skipped": sum(1 for l in legs if l["ret"] is None), "n": len(top)})
                prev[(m, n)] = {sid for sid, _ in top}
        log(f"{entry}：完成（{time.time() - t0:.0f} 秒）")
    out = {"schema_version": 1, "method": METHOD, "model_version": ai_model.MODEL_VERSION, "model_hash": model_hash(),
           "generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "start": start,
           "data_end": D["dates"][-1] if D["dates"] else None, "cost_pct": round(COST * 100, 3), "bench": BENCH,
           "timeline": timeline, "modes": {}, "seconds": round(time.time() - t0)}
    for m in modes:
        out["modes"][m] = {"name": ai_model.MODES[m]["name"], "universe": _stats(uni[m], bench), "universe_monthly": [round(x * 100, 2) for x in uni[m]],
                           "top": {str(n): {**_stats(res[m][n]["rets"], bench), "monthly": [round(x * 100, 2) for x in res[m][n]["rets"]],
                                            "picks": res[m][n]["picks"]} for n in TOP_NS}}
    out["bench_stats"] = _stats([b or 0 for b in bench], bench) if bench else None
    return out


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), "utf-8")
    for m, x in res["modes"].items():
        t = x["top"]["20"]
        print(f"{x['name']} 前 20 名：累積 {t['total']}%，0050 {t['bench_total']}%，最大回撤 {t['mdd']}%，贏 0050 月數比例 {t['win_vs_bench']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
