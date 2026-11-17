"""「可以虧，但不能輸 0050」兩種做法的歷史研究（事前登記：docs/prereg_bench_1117.md）→ history/backtest/bench_study.json。

做法 A：0050 為主（50／70／80%）＋該模式前 10 名等權。
做法 B：只在上市市值前 50 大（0050 成分股近似）裡持有，市值加權；B1 前 10 名 ×1.5、最後 10 名 ×0.5；B2 前 10 名 ×2、最後 10 名 ×0。
B0 是不調整的市值加權，用來量近似誤差。規則、門檻都在事前登記文件，這裡只照做。

用法：python -m app.bench_study
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ai_model, config
from . import pit_backtest as pb
from .selection_study import HOLDOUT_FROM, _regime, signal_dates

STUDY_VERSION = "2026-11-17"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "bench_study.json"
COST = pb.COST
TOP = 10
BIG = 50
CORE = {"A50": 0.5, "A70": 0.7, "A80": 0.8}
TILT = {"B1": (1.5, 0.5), "B2": (2.0, 0.0)}
GATE = {"lag_pp": 3.0, "lag_share": 0.10, "mdd_gap": 3.0, "proxy_err": 1.0}
NAMES = {"A50": "0050 佔 50%＋前 10 名", "A70": "0050 佔 70%＋前 10 名", "A80": "0050 佔 80%＋前 10 名",
         "B0": "市值前 50 大市值加權（0050 近似，對照組）", "B1": "前 50 大市值加權，前 10 名 ×1.5、最後 10 名 ×0.5",
         "B2": "前 50 大市值加權，前 10 名 ×2、最後 10 名不持有", "TOP10": "原本的前 10 名等權（對照）"}

_D = _CFG = None


def _period(p: tuple[str, str, str]) -> dict:
    sig, entry, exit_ = p
    D = _D
    rows = pb.scored_rows(D, sig, _CFG)
    df = pd.DataFrame([{"id": r["stock_id"], "market": r.get("market"), "cap": r.get("market_cap"),
                        **{f"rk_{m}": r["ai"]["modes"][m]["rank"] for m in ai_model.MODES}} for r in rows]).set_index("id")
    big = df[(df["market"] == "twse") & df["cap"].notna()]["cap"].nlargest(BIG)
    need = set(big.index)
    for m in ai_model.MODES:
        need |= set(df[df[f"rk_{m}"].notna()].sort_values(f"rk_{m}").index[:TOP])
    ret = {s: pb.hold_return(D, s, entry, exit_)[0] for s in need}
    return {"signal": sig, "entry": entry, "exit": exit_, "bench": pb._bench(D, entry, exit_), "regime": _regime(D, sig),
            "big": big.to_dict(), "ret": ret,
            "rk": {m: {s: df.at[s, f"rk_{m}"] for s in big.index if pd.notna(df.at[s, f"rk_{m}"])} for m in ai_model.MODES},
            "top": {m: list(df[df[f"rk_{m}"].notna()].sort_values(f"rk_{m}").index[:TOP]) for m in ai_model.MODES}}


def collect(dbname: str = "stocks.db", workers: int | None = None, log=print) -> list[dict]:
    global _D, _CFG
    t0 = time.time()
    _D, _CFG = pb.load(dbname), config.load()
    ps = signal_dates(_D["dates"])
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(ps)} 期")
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    if workers > 1 and sys.platform != "win32":
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            return [P for P in ex.map(_period, ps) if P["bench"] is not None]
    return [P for P in (_period(p) for p in ps) if P["bench"] is not None]


# ---------------- 權重 ----------------
def weights(P: dict, mode: str, rule: str) -> dict[str, float]:
    """這一期的目標權重（不含 0050 核心）；總和 = 股票部位比例。"""
    if rule == "TOP10" or rule in CORE:
        ids = P["top"][mode]
        share = 1 - CORE.get(rule, 0.0)
        return {s: share / len(ids) for s in ids} if ids else {}
    w = dict(P["big"])
    if rule in TILT:
        up, down = TILT[rule]
        rk = sorted(P["rk"][mode], key=P["rk"][mode].get)
        for s in rk[:TOP]:
            w[s] *= up
        for s in rk[-TOP:] if len(rk) >= 2 * TOP else []:
            w[s] *= down
    tot = sum(w.values())
    return {s: v / tot for s, v in w.items() if v > 0}


def series(periods: list[dict], mode: str, rule: str) -> list[float]:
    """每期成本後報酬。買不到的留現金；成本 = 0.585% × 新增權重（單邊換手）；0050 核心只在第一期扣一次。"""
    out, prev = [], {}
    core = CORE.get(rule, 0.0)
    for k, P in enumerate(periods):
        w = weights(P, mode, rule)
        r = sum(v * (P["ret"].get(s) if P["ret"].get(s) is not None else 0.0) for s, v in w.items())
        turn = sum(max(0.0, v - prev.get(s, 0.0)) for s, v in w.items())
        cost = COST * (turn + (core if k == 0 else 0.0))
        out.append(r + core * P["bench"] - cost)
        prev = w
    return out


# ---------------- 統計 ----------------
def _mdd(r) -> float:
    eq = np.cumprod(1 + np.asarray(r, float))
    return float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


def stats(r: list[float], b: list[float]) -> dict:
    r, b = np.asarray(r, float), np.asarray(b, float)
    e = (r - b) * 100
    n = len(e)
    t = float(e.mean() / (e.std(ddof=1) / np.sqrt(n))) if n >= 3 and e.std(ddof=1) > 0 else None
    return {"n": n, "mean_net": round(float(r.mean() * 100), 2), "mean_bench": round(float(b.mean() * 100), 2),
            "mean_excess": round(float(e.mean()), 2), "t": None if t is None else round(t, 2),
            "beat_share": round(float((e > 0).mean()) * 100, 1), "lag3_share": round(float((e < -GATE["lag_pp"]).mean()) * 100, 1),
            "worst_excess": round(float(e.min()), 2), "mdd": round(_mdd(r), 2), "mdd_bench": round(_mdd(b), 2),
            "cum": round(float(np.prod(1 + r) - 1) * 100, 1), "cum_bench": round(float(np.prod(1 + b) - 1) * 100, 1)}


def gate(dev: dict, h1: dict, h2: dict, hold: dict) -> dict:
    checks = {"開發段平均不輸 0050": dev["mean_excess"] >= 0,
              "開發段前後兩半都不輸": h1["mean_excess"] >= 0 and h2["mean_excess"] >= 0,
              "保留段平均不輸 0050": hold["mean_excess"] >= 0,
              f"保留段落後超過 {GATE['lag_pp']:g} 點的期數 ≤ {GATE['lag_share']:.0%}": hold["lag3_share"] <= GATE["lag_share"] * 100,
              f"保留段最大回撤不比 0050 差超過 {GATE['mdd_gap']:g} 點": hold["mdd"] >= hold["mdd_bench"] - GATE["mdd_gap"]}
    return {"passed": all(checks.values()), "checks": checks}


def analyse(periods: list[dict]) -> dict:
    dev = [P for P in periods if P["signal"] < HOLDOUT_FROM]
    hold = [P for P in periods if P["signal"] >= HOLDOUT_FROM]
    half = len(dev) // 2
    out = {"study_version": STUDY_VERSION, "prereg": "docs/prereg_bench_1117.md", "generated_at": config.now().isoformat(timespec="seconds"),
           "periods": {"dev": len(dev), "holdout": len(hold), "first": periods[0]["signal"], "last": periods[-1]["signal"]},
           "names": NAMES, "gate_rule": GATE, "modes": {}}
    bench_all = [P["bench"] for P in periods]
    # 近似誤差：B0（市值前 50 大市值加權）vs 0050
    b0 = series(periods, "balanced", "B0")
    err = (np.asarray(b0) - np.asarray(bench_all)) * 100
    out["proxy"] = {"mean_diff": round(float(err.mean()), 2), "abs_diff": round(float(np.abs(err).mean()), 2),
                    "corr": round(float(np.corrcoef(b0, bench_all)[0, 1]), 3),
                    "reliable": bool(abs(err.mean()) <= GATE["proxy_err"])}
    for m in ai_model.MODES:
        res = {}
        for rule in ("TOP10", *CORE, "B0", *TILT):
            r = series(periods, m, rule)
            seg = lambda ps: (r[: len(dev)] if ps is dev else r[len(dev):], [P["bench"] for P in ps])
            sd, sh = stats(*seg(dev)), stats(*seg(hold))
            s1 = stats(r[:half], [P["bench"] for P in dev[:half]])
            s2 = stats(r[half:len(dev)], [P["bench"] for P in dev[half:]])
            reg = {}
            for g in ("bull", "bear"):
                ix = [k for k, P in enumerate(periods) if P["regime"] == g]
                if len(ix) >= 3:
                    reg[g] = stats([r[k] for k in ix], [periods[k]["bench"] for k in ix])
            res[rule] = {"name": NAMES[rule], "dev": sd, "dev_first": s1, "dev_second": s2, "holdout": sh, "regime": reg,
                         **({"gate": gate(sd, s1, s2, sh)} if rule not in ("TOP10", "B0") else {})}
        out["modes"][m] = res
    return out


CACHE = config.DATA_DIR / "bench_periods.pkl"   # 不進 git


def run(dbname: str = "stocks.db", log=print) -> dict:
    import pickle
    if CACHE.exists() and os.environ.get("BENCH_CACHE") == "1":
        periods = pickle.loads(CACHE.read_bytes())
    else:
        periods = collect(dbname, log=log)
        CACHE.write_bytes(pickle.dumps(periods))
    r = analyse(periods)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), "utf-8")
    return r


def main(argv: list[str]) -> int:
    r = run(argv[1] if len(argv) > 1 else "stocks.db")
    print(f"近似誤差：B0 − 0050 平均 {r['proxy']['mean_diff']}、相關 {r['proxy']['corr']}")
    for m, res in r["modes"].items():
        for k, v in res.items():
            g = v.get("gate")
            print(m, k, "開發", v["dev"]["mean_excess"], "保留", v["holdout"]["mean_excess"], "落後3點", v["holdout"]["lag3_share"],
                  "" if g is None else ("通過" if g["passed"] else "未過"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
