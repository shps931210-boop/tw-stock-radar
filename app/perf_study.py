"""V22 績效優先候選實驗（事前登記：docs/prereg_v22.md）→ history/backtest/perf_v22.json。

三種模式各自和原模型（A）比較，每個候選只改一件事：
B1／B2 市場風險狀態、C1 跳空不追、C2 過熱不追、D 加入 MSURGE、R1 產業上限、R2 波動上限、R3 R1＋R2、R4 反波動加權。
E（題材催化）沒有歷史新聞資料，登記為無法驗證。

樣本：academic_study 的特徵快取（87 期不重疊 20 日，近似 PIT），隔日開盤進、第 20 個交易日後開盤出，
每期每檔都扣來回成本 0.585%（不考慮續抱省下的成本，對所有候選一樣），相對 0050 同期開盤到開盤。
只做研究，不改正式選股規則。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import pit_backtest as pb
from . import academic_study as ac

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "perf_v22.json"
STUDY_VERSION = "2026-11-01"
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}
N = 10
T_NEED = 2.90
RISK_SLACK = 0.20      # 百分點
BIG_SLACK = 1.0        # 百分點
BIG_LOSS = -0.10
GAP = 0.03
HOT_PCT = 0.90
VOL_PCT = 0.80
IND_CAP = 2
W_CAP = 0.20
ETF_COST = 0.00385
HOLDOUT_FROM = ac.HOLDOUT_FROM
RULES = {
    "A": "原模型前 10 名（對照組）",
    "B1": "市場風險狀態（加權指數 < 60 日均線且 20 日波動高於過去一年中位數）→ 該期改持有 0050",
    "B2": "市場風險狀態 → 只從低波動一半的股票裡選",
    "C1": "進場日開盤跳空 ≥ 3% 不買，改買下一名沒有跳空的",
    "C2": "距 20 日均線在當天全體前 10% 的不選",
    "D": "0.75 × 模式分數百分位 ＋ 0.25 × MSURGE 百分位",
    "R1": "同產業最多 2 檔",
    "R2": "年化波動在當天全體前 20% 的不選",
    "R3": "R1 ＋ R2",
    "R4": "反波動加權（單檔上限 20%）",
}
NOT_RUN = {"E": "題材催化：新聞資料 2026-10-11 才開始，沒有歷史期間可以驗證（RT-023 新聞存檔累積中）"}


def risk_off(D: dict) -> dict[str, bool]:
    s = D["idx_s"].astype(float).sort_index()
    ma = s.rolling(60).mean()
    v = s.pct_change().rolling(20).std()
    med = v.rolling(250, min_periods=120).median()
    return {d: bool(s[d] < ma[d] and v[d] > med[d]) for d in s.index if not np.isnan(ma[d]) and not np.isnan(med[d])}


def gaps(D: dict, P: dict, ids) -> pd.Series:
    out = {}
    for sid in ids:
        a = D["adj"].get(sid)
        if a is None or P["entry"] not in a.index or P["signal"] not in a.index:
            continue
        o, c = float(a.at[P["entry"], "open"]), float(a.at[P["signal"], "close"])
        if o > 0 and c > 0:
            out[sid] = o / c - 1
    return pd.Series(out, dtype=float)


def order(X: pd.DataFrame, mode: str, rule: str) -> list[str]:
    d = X[X[f"rk_{mode}"].notna()]
    if rule == "D":
        ms = d["msurge"].rank(pct=True).fillna(0.5) if "msurge" in d else 0.5
        s = 0.75 * d[f"sc_{mode}"].rank(pct=True) + 0.25 * ms
        return list(s.sort_values(ascending=False).index)
    return list(d.sort_values(f"rk_{mode}").index)


def pick(D: dict, P: dict, X: pd.DataFrame, mode: str, rule: str, off: bool) -> tuple[list[str], np.ndarray | None]:
    ids = order(X, mode, "D" if rule == "D" else "A")
    if rule == "B2" and off:
        lowv = X["vol_1y"] <= X["vol_1y"].median()
        ids = [s for s in ids if lowv.get(s, False)]
    if rule == "C2":
        hot = X["dist_ma20"] >= X["dist_ma20"].quantile(HOT_PCT)
        ids = [s for s in ids if not hot.get(s, False)]
    if rule in ("R2", "R3"):
        hv = X["vol_1y"] >= X["vol_1y"].quantile(VOL_PCT)
        ids = [s for s in ids if not hv.get(s, False)]
    if rule in ("R1", "R3"):
        out, cnt = [], {}
        for s in ids:
            g = X.at[s, "industry"] or ""
            if cnt.get(g, 0) < IND_CAP:
                out.append(s)
                cnt[g] = cnt.get(g, 0) + 1
            if len(out) == N:
                break
        ids = out
    if rule == "C1":
        g = gaps(D, P, ids[:60])
        ids = [s for s in ids[:60] if not (g.get(s, 0) >= GAP)]
    ids = ids[:N]
    w = None
    if rule == "R4":
        v = X["vol_1y"].reindex(ids).astype(float)
        w = (1 / v.fillna(v.median())).values
        w = w / w.sum()
        for _ in range(10):
            w = np.minimum(w, W_CAP)
            w = w / w.sum()
    return ids, w


def period_stats(P: dict, X: pd.DataFrame, ids: list[str], w) -> dict:
    r = X["ret"].reindex(ids).astype(float)
    net = r - pb.COST
    got = net.notna().values
    ww = np.full(len(ids), 1 / max(1, len(ids))) if w is None else np.asarray(w)
    port = float(np.nansum(np.where(got, net.values, 0.0) * ww)) if len(ids) else 0.0
    nn = net.dropna()
    return {"ret": port, "big": float((nn < BIG_LOSS).mean()) if len(nn) else 0.0,
            "worst": float(nn.min()) if len(nn) else 0.0, "ex_stock": list((nn - P["bench"]).values)}


def run_rule(D: dict, FS: list, mode: str, rule: str, ro: dict) -> dict:
    rows, picks = [], []
    for P, X in FS:
        off = ro.get(P["signal"], False)
        if rule == "B1" and off:
            rows.append({"ret": P["bench"] - ETF_COST, "big": 0.0, "worst": P["bench"] - ETF_COST, "ex_stock": []})
            picks.append([])
            continue
        ids, w = pick(D, P, X, mode, rule, off)
        rows.append(period_stats(P, X, ids, w))
        picks.append(ids)
    return {"rows": rows, "picks": picks}


def _half(dd):
    h = len(dd) // 2
    return float(dd[:h].mean()), float(dd[h:].mean())


def compare(R: dict, A: dict, bench: np.ndarray, dv: np.ndarray, rule: str) -> dict:
    r, a = np.array([x["ret"] for x in R["rows"]]), np.array([x["ret"] for x in A["rows"]])
    bg, ba = np.array([x["big"] for x in R["rows"]]), np.array([x["big"] for x in A["rows"]])
    d, db = r - a, bg - ba
    dd = d[dv]
    h1, h2 = _half(dd)
    v = {"dev_diff": ac._r(dd.mean() * 100), "dev_t": ac._r(ac._t(dd)), "first_half": ac._r(h1 * 100), "second_half": ac._r(h2 * 100),
         "holdout_diff": ac._r(d[~dv].mean() * 100), "big_diff_dev": ac._r(db[dv].mean() * 100), "big_t_dev": ac._r(ac._t(db[dv])),
         "big_diff_holdout": ac._r(db[~dv].mean() * 100)}
    v["return_pass"] = bool(v["dev_t"] is not None and v["dev_t"] >= T_NEED and h1 > 0 and h2 > 0 and d[~dv].mean() > 0
                            and db[dv].mean() * 100 <= BIG_SLACK)
    v["risk_pass"] = bool(rule != "D" and v["big_t_dev"] is not None and v["big_t_dev"] <= -T_NEED
                          and dd.mean() * 100 >= -RISK_SLACK and d[~dv].mean() * 100 >= -RISK_SLACK)
    # 交易機會損失：被排除（A 有、候選沒有）的股票 vs 補進來的股票，各期平均超額
    lost, added = [], []
    for (P, X), pa, pr in zip(R["_fs"], A["picks"], R["picks"]):
        out_ = [s for s in pa if s not in pr]
        in_ = [s for s in pr if s not in pa]
        if out_:
            lost += list((X["ret"].reindex(out_) - pb.COST - P["bench"]).dropna())
        if in_:
            added += list((X["ret"].reindex(in_) - pb.COST - P["bench"]).dropna())
    v["excluded_mean_excess"] = ac._r(np.mean(lost) * 100) if lost else None
    v["added_mean_excess"] = ac._r(np.mean(added) * 100) if added else None
    v["swaps"] = len(lost)
    return v


def summarize(R: dict, bench: np.ndarray, dv: np.ndarray) -> dict:
    from . import selection_study as ss
    r = [x["ret"] for x in R["rows"]]
    big = np.array([x["big"] for x in R["rows"]])
    worst = np.array([x["worst"] for x in R["rows"]])
    exs = [e for x in R["rows"] for e in x["ex_stock"]]
    return {**ss.summary(r, list(bench)), "stock_big_loss": ac._r(big.mean() * 100), "worst_stock_avg": ac._r(worst.mean() * 100),
            "stock_beat_0050": ac._r(np.mean([e > 0 for e in exs]) * 100) if exs else None,
            "up_hit": ac._r(np.mean([x > 0 for x in r]) * 100),
            "dev": {**ss.summary(list(np.array(r)[dv]), list(bench[dv])), "stock_big_loss": ac._r(big[dv].mean() * 100)},
            "holdout": {**ss.summary(list(np.array(r)[~dv]), list(bench[~dv])), "stock_big_loss": ac._r(big[~dv].mean() * 100)}}


def stable_vs_balanced(FS: list) -> dict:
    """穩健型與均衡型差在哪（描述）。"""
    ov, ex = [], {m: [] for m in MODES}
    expo = {m: {k: [] for k in ("quality", "growth", "value", "momentum", "trend", "vol_1y", "dist_ma20", "ret_3m")} for m in MODES}
    cap, ind, reg = {m: [] for m in MODES}, {m: [] for m in MODES}, {m: {"bull": [], "bear": []} for m in MODES}
    for P, X in FS:
        tops = {m: order(X, m, "A")[:N] for m in MODES}
        ov.append(len(set(tops["stable"]) & set(tops["balanced"])) / N)
        for m, t in tops.items():
            T = X.loc[t]
            e = float(((T["ret"] - pb.COST).fillna(0) - P["bench"]).mean())
            ex[m].append(e)
            if P.get("regime") in ("bull", "bear"):
                reg[m][P["regime"]].append(e)
            for k in expo[m]:
                expo[m][k].append(float(T[k].mean()))
            cap[m].append(float(T["market_cap"].median()))
            ind[m].append(int(T["industry"].fillna("").value_counts().iloc[0]))
    E = {m: np.array(v) for m, v in ex.items()}
    return {"overlap_stable_balanced_pct": ac._r(np.mean(ov) * 100), "overlap_balanced_aggressive_pct":
            ac._r(np.mean([len(set(order(X, "balanced", "A")[:N]) & set(order(X, "aggressive", "A")[:N])) / N for _, X in FS]) * 100),
            "excess_corr_stable_balanced": ac._r(np.corrcoef(E["stable"], E["balanced"])[0, 1], 3),
            "balanced_minus_stable": {"mean": ac._r((E["balanced"] - E["stable"]).mean() * 100), "t": ac._r(ac._t(E["balanced"] - E["stable"]))},
            "modes": {m: {"exposure": {k: ac._r(np.mean(v)) for k, v in expo[m].items()},
                          "median_cap_billion": ac._r(np.mean(cap[m]) / 1e9, 1), "max_same_industry": ac._r(np.mean(ind[m]), 1),
                          "bull": ac._r(np.mean(reg[m]["bull"]) * 100), "bear": ac._r(np.mean(reg[m]["bear"]) * 100),
                          "n_bull": len(reg[m]["bull"]), "n_bear": len(reg[m]["bear"])} for m in MODES}}


def run(D: dict | None = None, FS: list | None = None, log=print) -> dict:
    from .factor_audit import load_fs
    D = D or pb.load("stocks.db")
    FS = FS or load_fs()
    FS = [(P, X) for P, X in FS if P.get("bench") is not None and len(X)]
    ro = risk_off(D)
    bench = np.array([P["bench"] for P, _ in FS])
    dv = np.array([P["signal"] < HOLDOUT_FROM for P, _ in FS])
    out = {}
    for m in MODES:
        A = run_rule(D, FS, m, "A", ro)
        res = {"A": {"rule": RULES["A"], **summarize(A, bench, dv)}}
        for k in RULES:
            if k == "A":
                continue
            R = run_rule(D, FS, m, k, ro)
            R["_fs"] = FS
            res[k] = {"rule": RULES[k], **summarize(R, bench, dv), "vs_A": compare(R, A, bench, dv, k)}
            log(m, k, res[k]["vs_A"])
        out[m] = res
    passed = [f"{m}:{k}" for m, r in out.items() for k, v in r.items() if k != "A" and (v["vs_A"]["return_pass"] or v["vs_A"]["risk_pass"])]
    return {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
            "prereg": "docs/prereg_v22.md", "periods": len(FS), "first": FS[0][0]["signal"], "last": FS[-1][0]["signal"],
            "dev_periods": int(dv.sum()), "holdout_periods": int((~dv).sum()), "holdout_from": HOLDOUT_FROM,
            "risk_off_periods": int(sum(ro.get(P["signal"], False) for P, _ in FS)),
            "t_need": T_NEED, "tests": (len(RULES) - 1) * len(MODES), "rules": RULES, "not_run": NOT_RUN,
            "results": out, "passed": passed, "stable_vs_balanced": stable_vs_balanced(FS),
            "note": "保留段 2025 年以後已在 10/28、10/29 看過，屬重複使用；近似 PIT；每期每檔都扣成本。研究結果，不改正式推薦。"}


def main(argv: list[str]) -> int:
    D = pickle.loads(Path(argv[0]).read_bytes()) if argv else None
    res = run(D=D)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps({"passed": res["passed"], "risk_off": res["risk_off_periods"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
