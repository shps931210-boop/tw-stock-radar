"""選股風險規則的逐項檢驗（消融回測）→ history/backtest/risk.json。

問題：積極型（以及另外兩種模型）是否過度追高、承擔太多波動或產業集中？加一條風險規則能不能改善成本後超額報酬與回撤？

做法（和 AI 歷史回測同一套資料與交易規則）：
- 每月第一個交易日開盤進場、下個月第一個交易日開盤出場，只用訊號日收盤前看得到的資料，扣交易成本，基準 0050。
- A 是原始模型；B～H 每次只加入一條規則；I 是事先寫好的組合（不含市場狀態與基準部位）。
- 所有門檻在看結果之前就固定在下面的 RULES，不依回測結果挑參數，也不使用 2026/10/12 那一週的結果。
- 判定：與原版逐月比較，前後兩半段都改善且 t ≥ 2 才算「通過」；只有方向一致算「方向一致、證據不足」。
  通過的規則也只會先以影子模式追蹤，不直接改正式模型。
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
from .factor_research import REGIME_MA, _cap_pick

METHOD = 1
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "risk.json"
TOPS = (10, 20)
# 事先固定的門檻（2026-10-17 寫下，之後不依回測結果調整）
VOL_PCT = 0.8        # B：一年波動率在可排名股票中前 20% 高的剔除
VALUE_MIN = 20       # C：估值分數低於 20（明顯偏貴）的剔除
IND_CAP = 0.3        # D：同一產業最多 30%
HOT_PCT = 0.9        # F：近 3 個月漲幅在前 10% 的剔除
HOT_MA20 = 15        # F：股價高於 20 日均線超過 15% 的剔除
LIQ_MIN = 5e7        # G：近 20 日平均成交金額低於 5000 萬的剔除
CORE = 0.5           # E、H：0050 基準部位比例
RULES = {
    "A": "原始模型",
    "B": "＋高波動過濾",
    "C": "＋估值過貴過濾",
    "D": "＋同產業最多 30%",
    "E": "＋空頭時一半改持 0050",
    "F": "＋短期過熱過濾",
    "G": "＋流動性門檻",
    "H": "＋固定一半 0050 基準部位",
    "I": "＋B、C、D、F、G 全部（事先寫好的組合）",
}
RULE_DESC = {
    "B": f"一年波動率在當月可排名股票中排前 {round((1 - VOL_PCT) * 100)}% 的剔除",
    "C": f"估值分數低於 {VALUE_MIN} 的剔除",
    "D": f"依排名往下挑，同一產業最多 {round(IND_CAP * 100)}%",
    "E": f"訊號日加權指數低於 {REGIME_MA} 日均線時，{round(CORE * 100)}% 資金改持 0050",
    "F": f"近 3 個月漲幅排前 {round((1 - HOT_PCT) * 100)}%，或股價高於 20 日均線超過 {HOT_MA20}% 的剔除",
    "G": f"近 20 日平均成交金額低於 {LIQ_MIN / 1e8:.1f} 億元的剔除",
    "H": f"永遠 {round(CORE * 100)}% 持有 0050、{round((1 - CORE) * 100)}% 持有選股",
    "I": "B、C、D、F、G 同時套用",
}

_D = _CFG = None


def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"id": r["stock_id"], "ind": r.get("industry") or "", "base": bool(r.get("base_ok")),
                        "vol": _num(r.get("vol_1y")), "ret3m": _num(r.get("ret_3m")), "ma20": _num(r.get("dist_ma20")),
                        "liq": _num(r.get("avg_value_20")), "value": _num(r.get("score_value")),
                        **{f"rk_{m}": r["ai"]["modes"][m]["rank"] for m in ai_model.MODES}} for r in rows]).set_index("id")
    base = df[df["base"]]
    df["vol_hi"] = df["vol"] > base["vol"].quantile(VOL_PCT) if base["vol"].notna().any() else False
    df["hot"] = (df["ret3m"] > base["ret3m"].quantile(HOT_PCT)) | (df["ma20"] > HOT_MA20)
    return df


def _filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {"B": ~df["vol_hi"].fillna(False).astype(bool),
            "C": ~(df["value"] < VALUE_MIN).fillna(False).astype(bool),
            "F": ~df["hot"].fillna(False).astype(bool),
            "G": (df["liq"] >= LIQ_MIN).fillna(False).astype(bool)}


def picks(df: pd.DataFrame, mode: str, rule: str, n: int) -> list[str]:
    """某一條規則下的前 N 名（規則只剔除或限制，不改變模型分數本身的排序）。"""
    ok = df[df[f"rk_{mode}"].notna()].sort_values(f"rk_{mode}")
    f = _filters(ok)
    if rule in f:
        ok = ok[f[rule]]
    if rule == "I":
        for k in "BCFG":
            ok = ok[f[k].reindex(ok.index).fillna(False)]
    cap = IND_CAP if rule in ("D", "I") else None
    return _cap_pick([(sid, ok.at[sid, "ind"]) for sid in ok.index], n, cap)


def _regime(D: dict, sig: str) -> str | None:
    s = D["idx_s"]
    if s is None:
        return None
    s = s[s.index <= sig].astype(float)
    if len(s) < REGIME_MA:
        return None
    return "bull" if s.iloc[-1] >= s.iloc[-REGIME_MA:].mean() else "bear"


def _month(period) -> dict:
    """一個月份：所有模型 × 規則 × 前 N 名的成本後報酬（買不到的那一份留現金）。"""
    (sig, entry), (_, exit_) = period
    D = _D
    df = _frame(pb.scored_rows(D, sig, _CFG))
    pos = {d: i for i, d in enumerate(D["dates"])}
    e5 = D["dates"][pos[entry] + 5] if pos[entry] + 5 < len(D["dates"]) else None
    b, b5 = pb._bench(D, entry, exit_), (pb._bench(D, entry, e5) if e5 else None)
    reg = _regime(D, sig)
    out = {"signal": sig, "entry": entry, "exit": exit_, "bench": b, "bench5": b5, "regime": reg, "p": {}}
    cache, cache5 = {}, {}
    R = lambda sid: cache.setdefault(sid, pb.hold_return(D, sid, entry, exit_)[0])
    R5 = lambda sid: cache5.setdefault(sid, pb.hold_return(D, sid, entry, e5)[0] if e5 else None)
    for m in ai_model.MODES:
        for rule in RULES:
            base_rule = "A" if rule in ("E", "H") else rule
            for n in TOPS:
                pk = picks(df, m, base_rule, n)
                got = [R(s) for s in pk]
                g5 = [R5(s) for s in pk]
                w = 1 / n
                gross = sum(x * w for x in got if x is not None)
                gross5 = sum(x * w for x in g5 if x is not None)
                core = CORE if rule == "H" or (rule == "E" and reg == "bear") else 0.0
                out["p"][f"{m}|{rule}|{n}"] = {
                    "ret": (1 - core) * gross + core * (b or 0), "ret5": (1 - core) * gross5 + core * (b5 or 0),
                    "ids": pk, "inv": sum(x is not None for x in got) / n, "n": len(pk), "core": core,
                    "ind_max": float(pd.Series([df.at[s, "ind"] or s for s in pk]).value_counts().iloc[0] / max(len(pk), 1)) if pk else None}
    return out


def _work(period):
    return _month(period)


def _costs(months: list[dict], key: str) -> list[float]:
    """扣交易成本：只對新買進的股票部位收成本（和 AI 歷史回測一致）。"""
    out, prev = [], set()
    for mo in months:
        P = mo["p"][key]
        new = [s for s in P["ids"] if s not in prev]
        out.append(P["ret"] - pb.COST * (1 - P["core"]) * len(new) / max(len(P["ids"]), 1) * P["inv"])
        prev = set(P["ids"])
    return out


def _rel_mdd(r: list[float], b: list[float]) -> float | None:
    """相對 0050 的最大回撤：組合淨值 ÷ 0050 淨值的最大跌幅。"""
    if not r:
        return None
    rel = np.cumprod(1 + np.array(r)) / np.cumprod(1 + np.array(b))
    peak = np.maximum.accumulate(np.concatenate([[1.0], rel]))[1:]
    return round(float((rel / peak - 1).min()) * 100, 2)


def _verdict(diff: np.ndarray) -> dict:
    n = len(diff)
    h = n // 2
    first, second = float(diff[:h].mean()) if h else None, float(diff[h:].mean()) if n - h else None
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n))) if n > 2 and diff.std(ddof=1) > 0 else None
    both = first is not None and second is not None and first > 0 and second > 0
    v = "pass" if both and t is not None and t >= 2 else "consistent" if both else "no"
    return {"avg_diff": round(float(diff.mean()) * 100, 2), "first_half": None if first is None else round(first * 100, 2),
            "second_half": None if second is None else round(second * 100, 2), "t": None if t is None else round(t, 2),
            "verdict": v, "label": {"pass": "通過（仍先以影子模式追蹤）", "consistent": "方向一致，但證據不足", "no": "沒有穩定改善"}[v]}


def summarize(months: list[dict]) -> dict:
    bench = [mo["bench"] if mo["bench"] is not None else 0.0 for mo in months]
    b5 = [mo["bench5"] for mo in months]
    res = {}
    for m in ai_model.MODES:
        res[m] = {"name": ai_model.MODES[m]["name"]}
        for n in TOPS:
            base = np.array(_costs(months, f"{m}|A|{n}"))
            block = {}
            for rule, label in RULES.items():
                key = f"{m}|{rule}|{n}"
                r = _costs(months, key)
                st = pb._stats(r, bench)
                ex = np.array(r) - np.array(bench)
                r5 = [mo["p"][key]["ret5"] - x for mo, x in zip(months, b5) if x is not None]
                st.update({"rule": label, "te": round(float(ex.std(ddof=1) * np.sqrt(12)) * 100, 2) if len(ex) > 2 else None,
                           "rel_mdd": _rel_mdd(r, bench),
                           "excess_5d": round(float(np.mean(r5)) * 100, 2) if r5 else None,
                           "ind_max": round(float(np.mean([mo["p"][key]["ind_max"] for mo in months if mo["p"][key]["ind_max"] is not None])) * 100, 1),
                           "by_regime": {g: round(float(np.mean([x for x, mo in zip(ex, months) if mo["regime"] == g])) * 100, 2)
                                         if any(mo["regime"] == g for mo in months) else None for g in ("bull", "bear")},
                           "monthly": [round(x * 100, 2) for x in r]})
                if rule != "A":
                    st["vs_A"] = _verdict(np.array(r) - base)
                    if rule in ("E", "H"):  # 改的是資金配置，不是選股；0050 越強，這兩個版本自然越好
                        st["vs_A"]["label"] += "；這是資金配置效果，不代表選股變準"
                block[rule] = st
            res[m][str(n)] = block
    return res


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
    res = summarize(months)
    return {"schema_version": 1, "method": METHOD, "model_version": ai_model.MODEL_VERSION, "model_hash": pb.model_hash(),
            "generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "start": start,
            "data_end": D["dates"][-1] if D["dates"] else None, "bench": pb.BENCH, "cost_pct": round(pb.COST * 100, 3),
            "rules": RULES, "rule_desc": RULE_DESC,
            "thresholds_fixed_at": "2026-10-17",
            "months": [{k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in mo.items() if k != "p"} for mo in months],
            "results": res, "bench_stats": pb._stats([mo["bench"] or 0 for mo in months], [mo["bench"] for mo in months]),
            "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    for n in TOPS:
        for rule in RULES:
            s = res["results"]["aggressive"][str(n)][rule]
            print(f"積極型 前{n} {rule} {s['rule']}: 總報酬 {s['total']}% 平均超額 {s['avg_excess']}% MDD {s['mdd']}% 相對MDD {s['rel_mdd']}%"
                  + (f" → {s['vs_A']['label']}（差 {s['vs_A']['avg_diff']}%，t {s['vs_A']['t']}）" if "vs_A" in s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
