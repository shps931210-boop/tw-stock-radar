"""網站買賣訊號本身的回測（事前登記：docs/prereg_plan_audit_1121.md）→ history/backtest/plan_audit.json。

檢查 app/buyplan.py 現在真的顯示給使用者的結論，過去扣成本後有沒有贏 0050：
波段 S-BUY／S-WAIT／S-AVOID（每一筆波段訊號，現行出場），長期 L-BUY（每 20 個交易日檢查，持有 120 個交易日）。

用法：python -m app.plan_audit [資料庫檔名]；PLAN_CACHE=1 時沿用 data/plan_periods.pkl 的歷史分數。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import buyplan, config, db, metrics, track, valuation
from . import pit_backtest as pb
from . import trade_rule_study as tr
from .selection_study import HOLDOUT_FROM, signal_dates

STUDY_VERSION = "2026-11-21"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "plan_audit.json"
CACHE = config.DATA_DIR / "plan_periods.pkl"   # 不進 git
START = tr.START
T_NEED = tr.T_NEED
LONG_HOLD = 120
NW_LAG = 5
NAMES = {"S-BUY": "波段「可以考慮進場」：分數 ≥ 60、本益比位置 < 70%、波段訊號成立",
         "S-WAIT": "波段「好公司，等價格回落」：分數 ≥ 60、本益比位置 70%～85%",
         "S-AVOID": "波段「先不要買」：分數 < 45 或本益比位置 ≥ 85%",
         "S-ALL": "所有波段訊號（對照）",
         "L-BUY": "長期「可以考慮分批買進」：品質、成長、分數都 ≥ 60，本益比 ≤ 自己兩年中位數，不低於年線 10%",
         "L-ALL": "所有有完整分數的股票等權（對照）"}

_D = _CFG = None


# ---------------- 歷史分數 ----------------
def _scores(sig: str) -> tuple[str, dict]:
    rows = pb.scored_rows(_D, sig, _CFG)
    out = {}
    for r in rows:
        sc = r["ai"]["modes"]["balanced"]["score"] if r.get("complete") else None
        if sc is not None:
            out[r["stock_id"]] = (sc, r.get("score_quality"), r.get("score_growth"))
    return sig, out


def score_periods(dbname: str, log=print) -> dict[str, dict]:
    global _D, _CFG
    if CACHE.exists() and os.environ.get("PLAN_CACHE") == "1":
        return pickle.loads(CACHE.read_bytes())
    t0 = time.time()
    _D, _CFG = pb.load(dbname), config.load()
    sigs = [s for s, _, _ in signal_dates(_D["dates"], START)]
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(sigs)} 期分數")
    import multiprocessing as mp
    with ProcessPoolExecutor(max(1, min(4, os.cpu_count() or 1)), mp_context=mp.get_context("fork")) as ex:
        out = dict(ex.map(_scores, sigs))
    CACHE.write_bytes(pickle.dumps(out))
    log(f"分數完成 {time.time() - t0:.0f} 秒")
    return out


# ---------------- 當時的本益比位置 ----------------
class PE:
    """每檔股票：實際收盤價 ÷ 當時已公布的近四季 EPS；位置＝前 500 個有效交易日裡比現在低的比例（和 valuation.band 相同）。"""

    def __init__(self, raw: pd.DataFrame, fin: pd.DataFrame):
        self.s = {}
        fg = {k: g for k, g in fin.groupby("stock_id")}
        for sid, p in raw.groupby("stock_id"):
            ser = valuation.eps_ttm_series(fg.get(sid, pd.DataFrame()), False)
            if not ser:
                continue
            d = p["date"].to_numpy(str)
            c = p["close"].to_numpy(float)
            when = [x[0] for x in ser]
            k = np.searchsorted(when, d, side="right") - 1
            eps = np.array([ser[j][1] if j >= 0 else np.nan for j in k])
            with np.errstate(divide="ignore", invalid="ignore"):
                pe = np.where((eps > 0) & np.isfinite(c), c / eps, np.nan)
            self.s[sid] = (d, pe)

    def at(self, sid: str, day: str) -> tuple[float | None, float | None, float | None]:
        """(本益比, 位置 %, 兩年中位數)；資料不足回傳 None。"""
        x = self.s.get(sid)
        if x is None:
            return None, None, None
        d, pe = x
        i = int(np.searchsorted(d, day, side="right")) - 1
        if i < 0 or not np.isfinite(pe[i]):
            return None, None, None
        h = pe[: i + 1]
        h = h[np.isfinite(h)][-500:]
        if len(h) < valuation.MIN_DAYS:
            return float(pe[i]), None, None
        return float(pe[i]), float((h < pe[i]).mean() * 100), float(np.median(h))


# ---------------- 波段 ----------------
def swing_groups(T: pd.DataFrame) -> dict[str, pd.DataFrame]:
    X = T[T["rule"] == "swing"]
    pct = X["pe_pct"]
    good = X["score"] >= buyplan.GOOD_CO
    return {"S-ALL": X,
            "S-BUY": X[good & ~(pct >= buyplan.RICH)],
            "S-WAIT": X[good & (pct >= buyplan.RICH) & (pct < buyplan.HOT)],
            "S-AVOID": X[(X["score"] < buyplan.WEAK_CO) | (pct >= buyplan.HOT)]}


def judge_swing(X: pd.DataFrame, warn: bool = False) -> dict:
    dev, hold = X[X["date"] < HOLDOUT_FROM], X[X["date"] >= HOLDOUT_FROM]
    months = sorted(dev["date"].str[:7].unique())
    half = months[len(months) // 2] if months else ""
    d1, d2 = dev[dev["date"].str[:7] < half], dev[dev["date"].str[:7] >= half]
    sd, sh, s1, s2 = (tr.summarize(x) for x in (dev, hold, d1, d2))
    me = lambda s: s.get("month_excess") if s.get("n") else None
    if warn:
        ck = {"開發段輸 0050 且 t ≤ −2.6": me(sd) is not None and me(sd) < 0 and (sd["t"] or 0) <= -T_NEED,
              "保留段也輸 0050": me(sh) is not None and me(sh) < 0}
    else:
        ck = {"開發段贏 0050 且 t ≥ 2.6": me(sd) is not None and me(sd) > 0 and (sd["t"] or 0) >= T_NEED,
              "開發段前後兩半都贏": (me(s1) or 0) > 0 and (me(s2) or 0) > 0,
              "保留段贏 0050": me(sh) is not None and me(sh) > 0}
    return {"dev": sd, "dev_first": s1, "dev_second": s2, "holdout": sh,
            "checks": {k: bool(v) for k, v in ck.items()}, "passed": bool(all(ck.values()))}


# ---------------- 長期 ----------------
def _nw_t(x: np.ndarray, lag: int = NW_LAG) -> float | None:
    x = np.asarray(x, float)
    n = len(x)
    if n < 6:
        return None
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return None if v <= 0 else float(x.mean() / np.sqrt(v / n))


def long_events(dbname: str, S: dict, pe: PE) -> list[dict]:
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT stock_id, date, open, high, low, close FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        bench, _ = track.benchmark(con)
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    O, C = {}, {}
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid):
            continue
        a, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        a = a.set_index("date")
        o = a["open"].astype(float)
        O[sid], C[sid] = o.where(o > 0, a["close"].astype(float)), a["close"].astype(float)
    O, C = pd.DataFrame(O).sort_index(), pd.DataFrame(C).sort_index()
    MA = C.rolling(240, min_periods=200).mean()
    dates = list(C.index)
    bo = bench["open"].reindex(dates).astype(float)
    out = []
    for sig in sorted(S):
        i = dates.index(sig) if sig in dates else None
        if i is None or i + 1 + LONG_HOLD >= len(dates):
            continue
        a, b = i + 1, i + 1 + LONG_HOLD
        br = bo.iloc[b] / bo.iloc[a] - 1
        if not np.isfinite(br):
            continue
        buy, alls = [], []
        for sid, (sc, q, g) in S[sig].items():
            if sid not in O:
                continue
            o_in = O[sid].iloc[a]
            if not np.isfinite(o_in) or o_in <= 0:
                continue
            o_out = O[sid].iloc[b]
            if not np.isfinite(o_out):
                last = C[sid].iloc[a:b + 1].dropna()
                o_out = float(last.iloc[-1]) if len(last) else np.nan
            if not np.isfinite(o_out):
                continue
            r = o_out / o_in - 1 - tr.COST
            alls.append(r)
            if sc is None or q is None or g is None or sc < 60 or q < 60 or g < 60:
                continue
            cur, _, p50 = pe.at(sid, sig)
            if cur is None or p50 is None or cur > p50:
                continue
            ma = MA[sid].iloc[i]
            if not np.isfinite(ma) or C[sid].iloc[i] < ma * 0.9:
                continue
            buy.append(r)
        out.append({"date": sig, "bench": float(br), "n_buy": len(buy), "n_all": len(alls),
                    "buy": float(np.mean(buy)) if buy else None, "all": float(np.mean(alls)) if alls else None})
    return out


def judge_long(E: list[dict], key: str) -> dict:
    rows = [e for e in E if e.get(key) is not None]
    ex = lambda sel: np.array([e[key] - e["bench"] for e in rows if sel(e["date"])])
    dev, ho = ex(lambda d: d < HOLDOUT_FROM), ex(lambda d: d >= HOLDOUT_FROM)
    h = len(dev) // 2
    r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 2)
    v = {"periods_dev": int(len(dev)), "periods_holdout": int(len(ho)),
         "avg_stocks": r2(np.mean([e["n_buy" if key == "buy" else "n_all"] for e in rows])) if rows else None,
         "dev_excess": r2(dev.mean() * 100) if len(dev) else None, "dev_t_nw": r2(_nw_t(dev)),
         "dev_first": r2(dev[:h].mean() * 100) if h else None, "dev_second": r2(dev[h:].mean() * 100) if h else None,
         "dev_beat": r2((dev > 0).mean() * 100) if len(dev) else None,
         "holdout_excess": r2(ho.mean() * 100) if len(ho) else None, "holdout_beat": r2((ho > 0).mean() * 100) if len(ho) else None,
         "by_year": {y: r2(np.mean([e[key] - e["bench"] for e in rows if e["date"][:4] == y]) * 100)
                     for y in sorted({e["date"][:4] for e in rows})}}
    ck = {"開發段贏 0050 且 t ≥ 2.6": v["dev_excess"] is not None and v["dev_excess"] > 0 and (v["dev_t_nw"] or 0) >= T_NEED,
          "開發段前後兩半都贏": (v["dev_first"] or 0) > 0 and (v["dev_second"] or 0) > 0,
          "保留段贏 0050": (v["holdout_excess"] or 0) > 0}
    v["checks"] = {k: bool(x) for k, x in ck.items()}
    v["passed"] = bool(all(ck.values()))
    return v


def site_summary(path: Path = OUT) -> dict | None:
    """給網站個股卡用的精簡結果：每個結論過去比 0050 好或差多少。"""
    try:
        r = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    sw = {k: {"dev": v["dev"].get("month_excess"), "dev_n": v["dev"].get("n"), "dev_t": v["dev"].get("t"),
              "holdout": v["holdout"].get("month_excess"), "holdout_n": v["holdout"].get("n"), "passed": v["passed"]}
          for k, v in r["swing"].items()}
    lg = {k: {"dev": v["dev_excess"], "dev_beat": v["dev_beat"], "holdout": v["holdout_excess"], "holdout_beat": v["holdout_beat"],
              "periods": v["periods_dev"], "passed": v["passed"]} for k, v in r["long"].items()}
    return {"study_version": r["study_version"], "data_end": r["data_end"], "holdout_from": r["holdout_from"], "start": START,
            "long_hold": r["long_hold"], "swing": sw, "long": lg}


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    S = score_periods(dbname, log)
    with db.connect(dbname) as con:
        raw = db.read(con, "SELECT stock_id, date, close FROM prices ORDER BY stock_id, date")
        fin = db.read(con, "SELECT stock_id, period, eps FROM financials")
    pe = PE(raw, fin)
    log(f"本益比序列 {len(pe.s)} 檔，{time.time() - t0:.0f} 秒")
    T = tr.trades(dbname)
    sigs = sorted(S)
    k = np.searchsorted(np.array(sigs), T["date"].to_numpy(), side="right") - 1
    T["score"] = [S[sigs[j]].get(sid, (None,))[0] if j >= 0 else None for j, sid in zip(k, T["sid"])]
    T["score"] = pd.to_numeric(T["score"], errors="coerce")
    T = T[k >= 0].copy()
    T["pe_pct"] = [pe.at(s, d)[1] for s, d in zip(T["sid"], T["date"])]
    T["pe_pct"] = pd.to_numeric(T["pe_pct"], errors="coerce")
    log(f"波段訊號 {int((T['rule'] == 'swing').sum())} 筆，{time.time() - t0:.0f} 秒")
    G = swing_groups(T)
    swing = {k: {"name": NAMES[k], **judge_swing(v, warn=(k == "S-AVOID"))} for k, v in G.items()}
    swing["S-BUY"]["no_pe_history_share"] = round(float(G["S-BUY"]["pe_pct"].isna().mean() * 100), 1) if len(G["S-BUY"]) else None
    E = long_events(dbname, S, pe)
    lng = {"L-BUY": {"name": NAMES["L-BUY"], **judge_long(E, "buy")}, "L-ALL": {"name": NAMES["L-ALL"], **judge_long(E, "all")}}
    return {"schema_version": 1, "study_version": STUDY_VERSION, "prereg": "docs/prereg_plan_audit_1121.md",
            "generated_at": config.now().isoformat(timespec="seconds"), "data_end": raw["date"].max(),
            "holdout_from": HOLDOUT_FROM, "cost_pct": round(tr.COST * 100, 3), "long_hold": LONG_HOLD,
            "swing": swing, "long": lng, "long_periods": E, "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else "stocks.db")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    for k, v in r["swing"].items():
        print(k, "開發", v["dev"].get("month_excess"), "t", v["dev"].get("t"), "n", v["dev"].get("n"), "前後",
              v["dev_first"].get("month_excess"), v["dev_second"].get("month_excess"), "保留", v["holdout"].get("month_excess"),
              "通過" if v["passed"] else "未過")
    for k, v in r["long"].items():
        print(k, json.dumps({x: v[x] for x in v if x not in ("name", "by_year")}, ensure_ascii=False))
