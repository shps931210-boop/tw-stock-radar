"""V13 LightGBM 專項研究（只做研究，不改正式模型）：預測目標、訓練視窗、模型融合。

事前固定的比較（2026-10-25 定案，之後不能看結果再加減）：
- 基準 base：目前候選模型的做法，預測「20 日超額報酬的當日排名」，用全部過去資料（擴張視窗）。
- 目標一 t_excess：直接預測 20 日超額報酬（每天先把頭尾 1% 截掉，避免單一飆股主導）。
- 目標三 t_riskadj：超額報酬 ÷ 訊號日已知的一年波動率（風險調整後報酬，分母只用當時已知的資料）。
- 訓練視窗 w_25／w_50／w_100：只用最近 25／50／100 個訊號日（約 1／2／4 年）。
- 融合 fuse_targets：三種目標預測的當日百分位平均；ens_v21：base 和 V21 的百分位平均。
共 7 個挑戰者，和 base 比較；多重比較用 Bonferroni（雙尾 5% ÷ 7），|t| 要 ≥ 2.69 才算有證據。
所有變體共用同一組測試日、同一個 purge 規則（訓練標籤要在測試日之前就到期）、同一個交易成本與持股規則（model_race.portfolio）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import factor_research as fr
from . import model_race as mr

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "lgbm_study.json"
STUDY_VERSION = "2026-10-25"
VARIANTS = {
    "base": ("rank", None, "基準：預測 20 日超額報酬的當日排名，用全部過去資料"),
    "t_excess": ("excess", None, "目標一：直接預測 20 日超額報酬（每天截掉頭尾 1%）"),
    "t_riskadj": ("riskadj", None, "目標三：超額報酬 ÷ 已知的一年波動率（風險調整）"),
    "w_25": ("rank", 25, "只用最近 25 個訊號日訓練（約 1 年）"),
    "w_50": ("rank", 50, "只用最近 50 個訊號日訓練（約 2 年）"),
    "w_100": ("rank", 100, "只用最近 100 個訊號日訓練（約 4 年）"),
}
FUSED = {"fuse_targets": "融合：三種目標預測的百分位平均", "ens_v21": "融合：基準 LightGBM 和 V21 的百分位平均"}
T_NEED = 2.69
MDD_GAP = 5.0


def _targets(P: pd.DataFrame, y: str) -> pd.DataFrame:
    P = P.copy()
    P["tg_rank"] = P.groupby("date")[y].rank(pct=True)
    lo, hi = P.groupby("date")[y].transform(lambda s: s.quantile(0.01)), P.groupby("date")[y].transform(lambda s: s.quantile(0.99))
    P["tg_excess"] = P[y].clip(lo, hi)
    vol = P["vol_1y"].where(P["vol_1y"] > 0)
    vol = vol.fillna(P.groupby("date")["vol_1y"].transform("median"))
    floor = P.groupby("date")["vol_1y"].transform(lambda s: s.quantile(0.05))
    P["tg_riskadj"] = P[y] / np.maximum(vol, floor)
    return P


def walk(panel: pd.DataFrame, sigs: list[str], dates: list[str], idx: pd.Series | None, variants=VARIANTS) -> tuple[pd.DataFrame, list[str]]:
    import lightgbm as lgb
    from . import ml_research as ml
    H, y = mr.H, f"y{mr.H}"
    P = _targets(ml.prepare(mr.add_features(panel, idx), H), y)
    rf = [f"r_{f}" for f in ml.FEATS]
    pos = {d: i for i, d in enumerate(dates)}
    ex = {s: pos[s] + 1 + H for s in sigs}
    have = sorted(P["date"].unique())
    prm = mr._lgb_params(ml.LGB_PARAMS)
    out, tests = [], []
    for t in have:
        tr_all = [s for s in have if ex[s] < pos[t]]   # purge：訓練標籤要在測試日之前到期
        if len(tr_all) < ml.MIN_TRAIN:
            continue
        te = P[P["date"] == t]
        o = te[["date", "sid", "industry", y, "v21"] + [c for c in ("market_cap",) if c in te]].copy()
        for name, (target, win) in ((k, v[:2]) for k, v in variants.items()):
            tr_d = tr_all[-win:] if win else tr_all
            tr = P[P["date"].isin(tr_d)]
            m = lgb.train(prm, lgb.Dataset(tr[rf].to_numpy(), tr[f"tg_{target}"].to_numpy()), num_boost_round=ml.LGB_PARAMS["n_estimators"])
            o[name] = m.predict(te[rf].to_numpy())
        out.append(o)
        tests.append(t)
    preds = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(preds):
        pr = lambda c: preds.groupby("date")[c].rank(pct=True)
        preds["fuse_targets"] = (pr("base") + pr("t_excess") + pr("t_riskadj")) / 3
        preds["ens_v21"] = (pr("base") + pr("v21")) / 2
    return preds, tests


def _diff(a: dict, b: dict) -> dict:
    x, z = a.get("series") or [], b.get("series") or []
    n = min(len(x), len(z))
    d = np.array(x[:n]) - np.array(z[:n])
    t = mr._t(d)
    return {"diff": round(float(d.mean()), 2) if n else None, "t": round(t, 2) if t is not None else None, "n": n}


def run(panel: pd.DataFrame, sigs: list[str], D: dict, log=print) -> dict:
    from . import robustness as rb
    t0 = time.time()
    dates, idx = D["dates"], D.get("idx_s")
    preds, tests = walk(panel, sigs, dates, idx)
    if not len(preds):
        return {"ready": False, "reason": "測試日不足"}
    cut = tests[int(len(tests) * (1 - mr.HOLDOUT))]
    B = {d: mr._bench(D, dates, d) for d in tests}
    use, hold_use = tests[0::2], [d for d in tests[0::2] if d >= cut]
    hold = preds[preds["date"] >= cut]
    names = {**{k: v[2] for k, v in VARIANTS.items()}, **FUSED, "v21": "V21 固定權重（正式）"}
    models = ["v21", *VARIANTS, *FUSED]
    R = {}
    for m in models:
        p = mr.portfolio(preds, m, use, B)
        R[m] = {"name": names[m], "ic": mr.ic_stats(preds, m), "ic_holdout": mr.ic_stats(hold, m), "portfolio": p,
                "holdout": mr.portfolio(preds, m, hold_use, B), "offset1": {k: mr.portfolio(preds, m, tests[1::2], B).get(k) for k in ("net_excess", "t", "mdd")},
                "industries": mr.industries(preds, m)[:3]}
    base = R["base"]
    for m in models:
        if m in ("base", "v21"):
            continue
        r = R[m]
        r["vs_base"] = _diff(r["portfolio"], base["portfolio"])
        r["vs_base_holdout"] = _diff(r["holdout"], base["holdout"])
        ok_t = (r["vs_base"]["t"] or 0) >= T_NEED
        ok_hold = (r["holdout"].get("net_excess") or -1e9) > (base["holdout"].get("net_excess") or -1e9)
        ok_mdd = (r["portfolio"].get("mdd") or -1e9) >= (base["portfolio"].get("mdd") or -1e9) - MDD_GAP
        r["verdict"] = {"passed": bool(ok_t and ok_hold and ok_mdd), "checks": {"t": ok_t, "holdout": ok_hold, "mdd": ok_mdd},
                        "label": "有足夠證據勝過基準" if ok_t and ok_hold and ok_mdd else
                        ("方向較好但證據不足" if (r["vs_base"]["diff"] or 0) > 0 else "沒有比基準好")}
    passed = [m for m in models if R[m].get("verdict", {}).get("passed")]
    reg = rb.regime4(idx)
    res = {"schema_version": 1, "study_version": STUDY_VERSION, "horizon": mr.H, "top": mr.TOPN, "t_need": T_NEED, "mdd_gap": MDD_GAP,
           "tests": len(tests), "first_test": tests[0], "last_test": tests[-1], "holdout_from": cut,
           "independent_periods": R["v21"]["portfolio"].get("periods"), "names": names, "results": R,
           "size": rb.size_study(preds, models, use, mr.portfolio), "regimes": rb.regime_study(preds, models, reg),
           "passed": passed, "seconds": round(time.time() - t0),
           "conclusion": ("、".join(names[m] for m in passed) + " 通過研究門檻，下一步是放進影子預測做真實前瞻比較；正式模型不變") if passed
           else "沒有任何變體在事前定好的門檻下勝過基準：候選模型維持目前做法，正式模型維持 V21",
           "rules": "7 個挑戰者都和基準比；成本、持股、測試日、purge 全部相同；|t| ≥ 2.69（Bonferroni）、保留期也要較好、最大回撤不能差超過 5 個百分點才算通過"}
    log(f"LightGBM 專項研究：{res['conclusion']}（{res['seconds']} 秒）")
    return res


def main(argv: list[str]) -> int:
    from . import config, ml_research as ml, pit_backtest as pb
    path = Path(argv[0]) if argv else config.DATA_DIR / "ml_panel.pkl"
    if not path.exists():
        print("沒有研究資料集（data/ml_panel.pkl），跳過")
        return 0
    D = pb.load()
    sigs = ml.signal_dates(D["dates"], pb.auto_start(D))
    res = run(pd.read_pickle(path), sigs, D)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
