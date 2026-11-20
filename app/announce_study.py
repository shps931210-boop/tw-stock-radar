"""公告後股價反應研究（事前登記：docs/prereg_announce_1120.md）→ history/backtest/announce.json。

月營收、季報公布期一結束就進場，看「好消息＋市場認同」能不能在扣成本後贏 0050。
只用當時看得到的資料（月營收隔月 11 日、季報法定期限隔天），包含已下市股票。

用法：python -m app.announce_study [資料庫檔名]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db, metrics, track
from . import pit_backtest as pb

STUDY_VERSION = "2026-11-20"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "announce.json"
HOLDOUT_FROM = "2025-01-01"
START = "2019-01-01"
COST = pb.COST + 0.002          # 成本＋來回滑價 0.2%
N = 10
MIN_VALUE = 5e7                 # 前 20 日平均成交金額
MIN_PRICE = 10.0
LARGE = 150
HOLD = 20
T_NEED = 2.5
NAMES = {"R1": "月營收：營收驚喜最高 10 檔",
         "R2": "月營收：營收驚喜＞0 且公告期股價強過 0050，兩者綜合最高 10 檔",
         "R3": "R2，只在市值前 150 大",
         "Q1": "季報：獲利驚喜＞0 且公告期股價強過 0050，兩者綜合最高 10 檔"}


# ---------------- 資料 ----------------
def load(dbname: str = "stocks.db") -> dict:
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT stock_id, date, open, high, low, close, value FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        rev = db.read(con, "SELECT stock_id, ym, revenue FROM revenue")
        fin = db.read(con, "SELECT stock_id, period, eps, share_capital FROM financials")
        bench, _ = track.benchmark(con)
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    O, C, CR, V = {}, {}, {}, {}
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid):
            continue
        p = p.reset_index(drop=True)
        raw = p.set_index("date")["close"].astype(float)
        a, _ = metrics.adjust_prices(p, exg.get(sid, pd.DataFrame()))
        a = a.set_index("date")
        o = a["open"].astype(float)
        O[sid] = o.where(o > 0, a["close"].astype(float))
        C[sid], CR[sid], V[sid] = a["close"].astype(float), raw, p.set_index("date")["value"].astype(float)
    frame = lambda d: pd.DataFrame(d).sort_index()
    O, C, CR, V = frame(O), frame(C), frame(CR), frame(V)
    dates = list(C.index)
    b = bench.reindex(dates)
    return {"O": O, "C": C, "CR": CR, "V20": V.rolling(20, min_periods=15).mean(), "dates": dates,
            "bo": b["open"].astype(float), "bc": b["close"].astype(float), "rev": rev, "fin": fin}


def _first_on_or_after(dates: list[str], d: str) -> int | None:
    i = int(np.searchsorted(dates, d))
    return i if i < len(dates) else None


# ---------------- 訊號 ----------------
def surge_table(rev: pd.DataFrame) -> pd.DataFrame:
    """每檔每月的營收驚喜（與 academic_study 的 MSURGE 同一個算法）。index=(ym)，columns=stock_id。"""
    R = rev.pivot_table(index="ym", columns="stock_id", values="revenue", aggfunc="last")
    R.index = pd.PeriodIndex(R.index, freq="M")
    R = R.reindex(pd.period_range(R.index.min(), R.index.max(), freq="M")).astype(float)
    ur = R - R.shift(12)                              # 年增額
    hist = pd.concat([ur.shift(k) for k in range(1, 13)], keys=range(12))
    cnt = hist.groupby(level=1).count()
    mu, sd = hist.groupby(level=1).mean(), hist.groupby(level=1).std(ddof=1)
    s = (ur - mu) / sd
    return s.where((cnt >= 9) & (sd > 0))


def sue_table(fin: pd.DataFrame) -> pd.DataFrame:
    F = fin.pivot_table(index="period", columns="stock_id", values="eps", aggfunc="last").astype(float).sort_index()
    d = F - F.shift(4)
    hist = pd.concat([d.shift(k) for k in range(1, 9)], keys=range(8))
    cnt = hist.groupby(level=1).count()
    sd = hist.groupby(level=1).std(ddof=1)
    return (d / sd).where((cnt >= 6) & (sd > 0))


def shares_at(fin: pd.DataFrame, sig: str) -> pd.Series:
    f = fin[fin["period"].map(pb.fin_available) <= sig].dropna(subset=["share_capital"]).sort_values("period")
    return f.groupby("stock_id")["share_capital"].last().astype(float) / 10


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


# ---------------- 交易 ----------------
def hold(D: dict, picks: list[str], i_in: int, i_out: int) -> dict:
    O, C, dates = D["O"], D["C"], D["dates"]
    rets = []
    for sid in picks:
        o_in = O[sid].iloc[i_in]
        if not np.isfinite(o_in) or o_in <= 0:
            continue                                     # 隔天沒有成交：買不到，不算
        o_out = O[sid].iloc[i_out] if i_out < len(dates) else np.nan
        if not np.isfinite(o_out):                       # 停牌或下市：用出場日以前最後一個收盤價
            last = C[sid].iloc[i_in:i_out + 1].dropna()
            o_out = float(last.iloc[-1]) if len(last) else np.nan
        if np.isfinite(o_out):
            rets.append(o_out / o_in - 1 - COST)
    b = D["bo"].iloc[i_out] / D["bo"].iloc[i_in] - 1
    if not rets or not np.isfinite(b):
        return {}
    r = np.array(rets)
    return {"net": float(r.mean()), "bench": float(b), "excess": float(r.mean() - b), "n": len(r),
            "big_loss": float((r <= -0.10).mean())}


def universe(D: dict, i: int) -> pd.Index:
    c, v = D["CR"].iloc[i], D["V20"].iloc[i]
    ok = (c >= MIN_PRICE) & (v >= MIN_VALUE)
    return ok[ok].index


def _combo(g: pd.DataFrame) -> pd.Series:
    pos = g[(g["s"] > 0) & (g["r"] > 0)]
    return ((_pct(pos["s"]) + _pct(pos["r"])) / 2).sort_values(ascending=False)


def monthly_picks(D: dict, S: pd.DataFrame, ym, i: int) -> dict:
    """訊號日（第 i 個交易日）收盤後就能算出的三份名單；只用當天以前的資料。"""
    dates, sig = D["dates"], D["dates"][i]
    pre = _first_on_or_after(dates, f"{(ym + 1).year}-{(ym + 1).month:02d}-01") - 1   # 公告月 1 日前最後一個交易日
    U = universe(D, i)
    s = S.loc[ym].reindex(U).dropna() if ym in S.index else pd.Series(dtype=float)
    react = (D["C"].iloc[i] / D["C"].iloc[pre] - 1).reindex(s.index) - (D["bc"].iloc[i] / D["bc"].iloc[pre] - 1)
    mcap = (D["CR"].iloc[i] * shares_at(D["fin"], sig)).reindex(U).dropna()
    large = set(mcap.sort_values(ascending=False).index[:LARGE])
    g = pd.DataFrame({"s": s, "r": react}).dropna()
    return {"R1": list(s.sort_values(ascending=False).index[:N]), "R2": list(_combo(g).index[:N]),
            "R3": list(_combo(g[g.index.isin(large)]).index[:N]), "universe": len(U)}


def quarterly_picks(D: dict, Q: pd.DataFrame, per: str, i: int) -> dict:
    dates = D["dates"]
    pre = _first_on_or_after(dates, str((pd.Timestamp(pb.fin_available(per)) - pd.Timedelta(days=45)).date())) - 1
    U = universe(D, i)
    s = Q.loc[per].reindex(U).dropna() if per in Q.index else pd.Series(dtype=float)
    react = (D["C"].iloc[i] / D["C"].iloc[pre] - 1).reindex(s.index) - (D["bc"].iloc[i] / D["bc"].iloc[pre] - 1)
    return {"Q1": list(_combo(pd.DataFrame({"s": s, "r": react}).dropna()).index[:N]), "universe": len(U)}


def monthly_events(D: dict) -> list[dict]:
    S = surge_table(D["rev"])
    dates, out = D["dates"], []
    months = [p for p in S.index if pb.rev_available(str(p)) >= START]
    sig_idx = [_first_on_or_after(dates, pb.rev_available(str(p))) for p in months]
    for k, (ym, i) in enumerate(zip(months, sig_idx)):
        nxt = sig_idx[k + 1] if k + 1 < len(sig_idx) else None
        if i is None or nxt is None or nxt + 1 >= len(dates):
            continue                                     # 還沒到期
        P = monthly_picks(D, S, ym, i)
        row = {"ym": str(ym), "signal": dates[i], "entry": dates[i + 1], "exit": dates[nxt + 1], "universe": P["universe"]}
        for k2 in ("R1", "R2", "R3"):
            row[k2] = hold(D, P[k2], i + 1, nxt + 1) | {"picks": P[k2]}
        out.append(row)
    return out


def quarterly_events(D: dict) -> list[dict]:
    Q = sue_table(D["fin"])
    dates, out = D["dates"], []
    for per in Q.index:
        av = pb.fin_available(per)
        if av < START:
            continue
        i = _first_on_or_after(dates, av)
        if i is None or i + 1 + HOLD >= len(dates):
            continue
        P = quarterly_picks(D, Q, per, i)
        out.append({"period": per, "signal": dates[i], "entry": dates[i + 1], "exit": dates[i + 1 + HOLD],
                    "universe": P["universe"], "Q1": hold(D, P["Q1"], i + 1, i + 1 + HOLD) | {"picks": P["Q1"]}})
    return out


# ---------------- 判定 ----------------
def _t(x: np.ndarray) -> float | None:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) >= 3 and x.std(ddof=1) > 0 else None


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def judge(events: list[dict], key: str) -> dict:
    rows = [(e["signal"], e[key]) for e in events if e.get(key, {}).get("excess") is not None]
    dev = np.array([r["excess"] for s, r in rows if s < HOLDOUT_FROM])
    ho = np.array([r["excess"] for s, r in rows if s >= HOLDOUT_FROM])
    net = lambda sel: np.array([r["net"] for s, r in rows if sel(s)])
    bench = lambda sel: np.array([r["bench"] for s, r in rows if sel(s)])
    isdev, isho = (lambda s: s < HOLDOUT_FROM), (lambda s: s >= HOLDOUT_FROM)
    h = len(dev) // 2
    v = {"name": NAMES[key], "events_dev": int(len(dev)), "events_holdout": int(len(ho)),
         "dev_excess": _r(dev.mean() * 100) if len(dev) else None, "dev_t": _r(_t(dev)),
         "dev_first_half": _r(dev[:h].mean() * 100) if h else None, "dev_second_half": _r(dev[h:].mean() * 100) if h else None,
         "dev_beat_share": _r((dev > 0).mean() * 100, 1) if len(dev) else None,
         "dev_net": _r(net(isdev).mean() * 100) if len(dev) else None, "dev_0050": _r(bench(isdev).mean() * 100) if len(dev) else None,
         "holdout_excess": _r(ho.mean() * 100) if len(ho) else None,
         "holdout_beat_share": _r((ho > 0).mean() * 100, 1) if len(ho) else None,
         "holdout_net": _r(net(isho).mean() * 100) if len(ho) else None, "holdout_0050": _r(bench(isho).mean() * 100) if len(ho) else None,
         "big_loss_pct": _r(np.mean([r["big_loss"] for _, r in rows]) * 100, 1) if rows else None}
    checks = {"t": v["dev_t"] is not None and v["dev_t"] >= T_NEED and v["dev_excess"] > 0,
              "halves": bool(h) and dev[:h].mean() > 0 and dev[h:].mean() > 0,
              "beat_share": v["dev_beat_share"] is not None and v["dev_beat_share"] > 50,
              "holdout": v["holdout_excess"] is not None and v["holdout_excess"] > 0}
    v["checks"] = {k: bool(x) for k, x in checks.items()}
    v["passed"] = bool(all(checks.values()))
    return v


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    D = load(dbname)
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(D['dates'])} 個交易日、{D['C'].shape[1]} 檔")
    M, Q = monthly_events(D), quarterly_events(D)
    res = {k: judge(M, k) for k in ("R1", "R2", "R3")} | {"Q1": judge(Q, "Q1")}
    return {"schema_version": 1, "study_version": STUDY_VERSION, "prereg": "docs/prereg_announce_1120.md",
            "generated_at": config.now().isoformat(timespec="seconds"), "data_end": D["dates"][-1],
            "holdout_from": HOLDOUT_FROM, "cost_pct": round(COST * 100, 3), "t_need": T_NEED,
            "results": res, "passed": [k for k, v in res.items() if v["passed"]],
            "monthly": [{k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "picks"}) for k, v in e.items()} for e in M],
            "quarterly": [{k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "picks"}) for k, v in e.items()} for e in Q],
            "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "stocks.db")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(r["results"], ensure_ascii=False, indent=1))
