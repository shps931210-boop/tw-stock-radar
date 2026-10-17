"""推薦組合風險研究（V27）→ history/backtest/portfolio.json。

問題：高分冷門股、同產業集中、高波動會不會讓 Top10 看起來很好、實際上買不到或風險太集中？
比較六種名單（同一套資料截止時間、進場規則、交易成本、測試期間）：
  A 原始模型、B 排除流動性 E/F、C B＋D 級不進前 10 名（網站採用）、D C＋流動性×波動率聯合限制、
  E 原始＋同產業最多 30%、F D＋同產業最多 30%（聯合風險限制）。
- 每月第一個交易日開盤進場、下月第一個交易日開盤出場；只用訊號日收盤前的資料（PIT）；買不到的那一份留現金。
- 流動性規則和網站呼叫同一個 liquidity.order()。門檻在看結果前寫下（2026-10-17），不依回測結果調整。
- 另外研究：各流動性等級的平均超額報酬與買不到的比例、流動性 × 波動率、各流動性細項與下月報酬的等級相關（IC）。
- 判定：與 A 逐月比較，前後兩半段都改善且 t ≥ 2 才算「通過」。未通過的規則不能取代正式模型；
  網站上的流動性限制是「推薦資格」，原始排名與績效紀錄照舊。
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

from . import ai_model, config, liquidity
from . import pit_backtest as pb
from .factor_research import _cap_pick
from .risk_research import _rel_mdd, _verdict

METHOD = 1
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "portfolio.json"
TOPS = (10, 20)
IND_CAP = 0.3
VARIANTS = {**liquidity.VARIANTS, "E": "原始＋同產業最多 30%", "F": "D＋同產業最多 30%（聯合風險限制）"}
SLIP = (0.002, 0.005)       # 每次買進額外滑價（來回），敏感度測試
# 依流動性等級的買賣價差假設（單位：比例），只用在「依流動性的滑價」情境，是假設不是實測
SPREAD = {"A": 0.001, "B": 0.0015, "C": 0.0025, "D": 0.004, "E": 0.008, "F": 0.015, "?": 0.015}
CORR_DAYS, BETA_DAYS = 120, 250
LIQ_FIELDS = {"avg_value_20": "20 日平均成交金額", "value_med_20": "20 日成交金額中位數",
              "low_value_days_20": "20 日內低量日數（越多越冷門）", "value_cv_20": "成交金額變異係數",
              "zero_vol_days_60": "60 日內零成交日數"}

_D = _CFG = None


def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def pick(rows: list[dict], mode: str, variant: str, n: int) -> list[str]:
    ranked = sorted((r for r in rows if r["ai"]["modes"][mode]["rank"]), key=lambda r: r["ai"]["modes"][mode]["rank"])
    if variant in ("A", "E"):
        order = [r["stock_id"] for r in ranked]
    else:
        lv = "D" if variant == "F" else variant
        rec, _ = liquidity.order([(r["stock_id"], r["ai"]["modes"][mode]["score"], r["liq"]["grade"], r.get("vol_1y_pct")) for r in ranked], lv)
        order = [x["id"] for x in rec]
    ind = {r["stock_id"]: r.get("industry") or "" for r in rows}
    return _cap_pick([(s, ind[s]) for s in order], n, IND_CAP if variant in ("E", "F") else None)


def _daily(D: dict, sid: str, sig: str, days: int) -> pd.Series | None:
    a = D["adj"].get(sid)
    if a is None:
        return None
    c = a.loc[a.index <= sig, "close"].iloc[-days - 1:]
    return c.pct_change().dropna() if len(c) > days // 2 else None


def _corr_beta(D: dict, ids: list[str], sig: str) -> tuple[float | None, float | None]:
    R = {s: _daily(D, s, sig, BETA_DAYS) for s in ids}
    R = {s: r for s, r in R.items() if r is not None}
    corr = beta = None
    if len(R) >= 2:
        M = pd.DataFrame({s: r.iloc[-CORR_DAYS:] for s, r in R.items()}).corr().to_numpy()
        iu = np.triu_indices_from(M, 1)
        v = M[iu]
        corr = float(np.nanmean(v)) if np.isfinite(v).any() else None
    if D["idx_s"] is not None and R:
        m = D["idx_s"][D["idx_s"].index <= sig].astype(float).iloc[-BETA_DAYS - 1:].pct_change().dropna()
        bs = []
        for r in R.values():
            j = pd.concat([r, m], axis=1, join="inner").dropna()
            if len(j) > 60 and j.iloc[:, 1].var() > 0:
                bs.append(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
        beta = float(np.mean(bs)) if bs else None
    return corr, beta


def _ic(x: pd.Series, y: pd.Series) -> float | None:
    j = pd.concat([x, y], axis=1).dropna()
    if len(j) < 30 or j.iloc[:, 0].nunique() < 3:
        return None
    return float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()))


def _month(period) -> dict:
    (sig, entry), (sig2, exit_) = period
    D = _D
    rows = pb.scored_rows(D, sig, _CFG)
    for r in rows:
        r["liq"] = liquidity.grade(r)
    b = pb._bench(D, entry, exit_)
    tx = None
    if D["idx_s"] is not None and sig in D["idx_s"].index and sig2 in D["idx_s"].index:
        tx = float(D["idx_s"][sig2] / D["idx_s"][sig] - 1)
    cache = {}
    R = lambda sid: cache.setdefault(sid, pb.hold_return(D, sid, entry, exit_)[0])
    by = {r["stock_id"]: r for r in rows}
    out = {"signal": sig, "entry": entry, "exit": exit_, "bench": b, "taiex": tx, "p": {}}
    for m in ai_model.MODES:
        for v in VARIANTS:
            for n in TOPS:
                pk = pick(rows, m, v, n)
                got = [R(s) for s in pk]
                corr, beta = _corr_beta(D, pk, sig) if n == 10 else (None, None)
                out["p"][f"{m}|{v}|{n}"] = {
                    "ret": sum(x / n for x in got if x is not None), "ids": pk, "n": len(pk),
                    "inv": sum(x is not None for x in got) / n,
                    "grades": [by[s]["liq"]["grade"] for s in pk],
                    "ind_max": float(pd.Series([by[s].get("industry") or s for s in pk]).value_counts().iloc[0] / len(pk)) if pk else None,
                    "vol_pct": float(np.nanmean([_num(by[s].get("vol_1y_pct")) or np.nan for s in pk])) if pk else None,
                    "corr": corr, "beta": beta}
    # 全市場（通過基本門檻）：流動性等級、波動率與下月報酬
    uni = [r for r in rows if r.get("base_ok")]
    vol = pd.Series({r["stock_id"]: _num(r.get("vol_1y_pct")) for r in uni}, dtype=float)
    ret = pd.Series({r["stock_id"]: R(r["stock_id"]) for r in uni}, dtype=float)
    grade = {r["stock_id"]: r["liq"]["grade"] for r in uni}
    cells = {}
    for sid in ret.index:
        g = grade[sid]
        hv = "hv" if (vol.get(sid) or 0) >= liquidity.VOL_HI else "lv"
        for k in (g, f"{g}|{hv}"):
            c = cells.setdefault(k, {"n": 0, "untradable": 0, "ex": []})
            c["n"] += 1
            if pd.isna(ret[sid]):
                c["untradable"] += 1
            elif b is not None:
                c["ex"].append(float(ret[sid] - b))
    out["cells"] = {k: {"n": c["n"], "untradable": c["untradable"], "ex": float(np.mean(c["ex"])) if c["ex"] else None} for k, c in cells.items()}
    out["ic"] = {f: _ic(pd.Series({r["stock_id"]: _num(r.get(f)) for r in uni}, dtype=float), ret) for f in LIQ_FIELDS}
    return out


def _work(period):
    return _month(period)


def _costs(months: list[dict], key: str, extra: float = 0.0, spread: bool = False) -> list[float]:
    out, prev = [], set()
    for mo in months:
        P = mo["p"][key]
        new = [(s, g) for s, g in zip(P["ids"], P["grades"]) if s not in prev]
        k = max(len(P["ids"]), 1)
        c = sum(pb.COST + extra + (SPREAD.get(g, 0.015) if spread else 0) for _, g in new) / k * P["inv"]
        out.append(P["ret"] - c)
        prev = set(P["ids"])
    return out


def summarize(months: list[dict]) -> dict:
    bench = [mo["bench"] if mo["bench"] is not None else 0.0 for mo in months]
    taiex = [mo["taiex"] for mo in months]
    res = {}
    for m in ai_model.MODES:
        res[m] = {"name": ai_model.MODES[m]["name"]}
        for n in TOPS:
            base = np.array(_costs(months, f"{m}|A|{n}"))
            block = {}
            for v, label in VARIANTS.items():
                key = f"{m}|{v}|{n}"
                r = _costs(months, key)
                st = pb._stats(r, bench)
                tx = [(x, t) for x, t in zip(r, taiex) if t is not None]
                P = [mo["p"][key] for mo in months]
                gr = [g for p in P for g in p["grades"]]
                mean = lambda k: (round(float(np.mean([p[k] for p in P if p[k] is not None])), 3) if any(p[k] is not None for p in P) else None)
                st.update({
                    "variant": label, "rel_mdd": _rel_mdd(r, bench),
                    "taiex_total": round(float(np.prod([1 + t for _, t in tx]) - 1) * 100, 2) if tx else None,
                    "win_vs_taiex": round(float(np.mean([x > t for x, t in tx])) * 100, 1) if tx else None,
                    "coverage": round(float(np.mean([p["n"] / n for p in P])) * 100, 1),
                    "untradable": round(float(1 - np.mean([p["inv"] for p in P])) * 100, 1),
                    "ind_max": round(mean("ind_max") * 100, 1) if mean("ind_max") is not None else None,
                    "vol_pct": mean("vol_pct"), "corr": mean("corr"), "beta": mean("beta"),
                    "grade_mix": {g: round(gr.count(g) / max(len(gr), 1) * 100, 1) for g in "ABCDEF?" if g in gr},
                    "slippage": {f"+{x * 100:.1f}%": pb._stats(_costs(months, key, x), bench)["total"] for x in SLIP}
                                | {"依流動性": pb._stats(_costs(months, key, spread=True), bench)["total"]},
                    "monthly": [round(x * 100, 2) for x in r]})
                if v != "A":
                    st["vs_A"] = _verdict(np.array(r) - base)
                    st["vs_A_spread"] = _verdict(np.array(_costs(months, key, spread=True)) - np.array(_costs(months, f"{m}|A|{n}", spread=True)))
                block[v] = st
            res[m][str(n)] = block
    cells = {}
    for mo in months:
        for k, c in mo["cells"].items():
            x = cells.setdefault(k, {"n": 0, "untradable": 0, "ex": []})
            x["n"] += c["n"]
            x["untradable"] += c["untradable"]
            if c["ex"] is not None:
                x["ex"].append(c["ex"])
    grades = {k: {"stock_months": c["n"], "untradable_pct": round(c["untradable"] / c["n"] * 100, 2) if c["n"] else None,
                  "avg_excess": round(float(np.mean(c["ex"])) * 100, 2) if c["ex"] else None,
                  "t": round(float(np.mean(c["ex"]) / (np.std(c["ex"], ddof=1) / np.sqrt(len(c["ex"])))), 2) if len(c["ex"]) > 2 and np.std(c["ex"], ddof=1) > 0 else None}
              for k, c in sorted(cells.items())}
    ics = {}
    for f, label in LIQ_FIELDS.items():
        v = np.array([mo["ic"][f] for mo in months if mo["ic"].get(f) is not None])
        ics[f] = {"label": label, "months": len(v), "ic": round(float(v.mean()), 4) if len(v) else None,
                  "t": round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2) if len(v) > 2 and v.std(ddof=1) > 0 else None}
    return {"results": res, "by_grade": grades, "liquidity_ic": ics}


def run(dbname: str = "stocks.db", start: str | None = None, log=print, workers: int | None = None) -> dict:
    global _D, _CFG
    t0 = time.time()
    cfg = config.load()
    D = pb.load(dbname)
    start = start or pb.auto_start(D)
    rb = pb.rebalance_dates(D["dates"], start)
    periods = list(zip(rb, rb[1:]))
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(periods)} 個完整月份")
    _D, _CFG = D, cfg
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    if workers > 1 and sys.platform != "win32":
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            months = list(ex.map(_work, periods))
    else:
        months = [_work(p) for p in periods]
    log(f"逐月計算完成（{time.time() - t0:.0f} 秒）")
    S = summarize(months)
    return {"schema_version": 1, "method": METHOD, "model_version": ai_model.MODEL_VERSION, "model_hash": pb.model_hash(),
            "generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "start": start,
            "data_end": D["dates"][-1] if D["dates"] else None, "bench": pb.BENCH, "cost_pct": round(pb.COST * 100, 3),
            "variants": VARIANTS, "live_variant": liquidity.LIVE, "thresholds_fixed_at": "2026-10-17",
            "grades": [[g, v] for g, v in liquidity.GRADES], "spread_assumption": SPREAD,
            "taiex_note": "加權指數是價格指數（不含股利），以訊號日收盤計算；0050 含息、以開盤價計算",
            "months": [{k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in mo.items() if k in ("signal", "entry", "exit", "bench", "taiex")}
                       for mo in months],
            **S, "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    for m in ai_model.MODES:
        for v in VARIANTS:
            s = res["results"][m]["10"][v]
            print(f"{res['results'][m]['name']} 前10 {v} {s['variant']}: 總報酬 {s['total']}% 0050 {s['bench_total']}% 加權 {s['taiex_total']}% "
                  f"MDD {s['mdd']}% 買不到 {s['untradable']}% 產業最大 {s['ind_max']}% 依流動性滑價後 {s['slippage']['依流動性']}%"
                  + (f" → {s['vs_A']['label']}（差 {s['vs_A']['avg_diff']}%，t {s['vs_A']['t']}）" if "vs_A" in s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
