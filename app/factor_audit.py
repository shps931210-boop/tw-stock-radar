"""V21.2 五大因子完整檢查（事前登記：docs/prereg_v21_2.md 第一節）→ history/backtest/factor_audit.json。

只做診斷，不改權重。樣本與交易規則同 academic_study（87 期、20 日、開盤進出、相對 0050）。
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

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "factor_audit.json"
STUDY_VERSION = "2026-10-29"
FACTORS = ("quality", "growth", "value", "momentum", "trend")
T_NEED = 2.94          # 5 因子 × 3 期間 = 15 個檢定，Bonferroni
HOLDOUT_FROM = ac.HOLDOUT_FROM
MIN_IND = 15


def _ic(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())) if len(d) >= ac.MIN_N else np.nan


def _m(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return {"mean": ac._r(np.mean(v), 4) if v else None, "t": ac._r(ac._t(v)), "n": len(v)}


def _pp(v):
    v = [x for x in v if not np.isnan(x)]
    return {"mean_pct": ac._r(np.mean(v) * 100) if v else None, "t": ac._r(ac._t(v)), "n": len(v)}


def audit(FS: list[tuple[dict, pd.DataFrame]]) -> dict:
    res = {}
    dv = [P["signal"] < HOLDOUT_FROM for P, _ in FS]
    for f in FACTORS:
        r = {"ic": {}, "deciles": None, "top10_net": None, "regime": {}, "size": {}, "industry": None, "tail": None, "incremental": None}
        for h in (5, 20, 60):
            v = [_ic(X[f], X[f"ex{h}"]) if f"ex{h}" in X else np.nan for _, X in FS]
            r["ic"][str(h)] = {**_m(v), "dev": _m([x for x, d in zip(v, dv) if d]), "holdout": _m([x for x, d in zip(v, dv) if not d]),
                               "passed": bool(ac._t(v) is not None and abs(ac._t(v)) >= T_NEED)}
        dec, top, tail_hi, tail_lo = [[] for _ in range(10)], [], [], []
        for P, X in FS:
            d = X[[f, "ex20", "ret"]].dropna()
            if len(d) < 50:
                continue
            q = pd.qcut(d[f].rank(method="first"), 10, labels=False)
            for k in range(10):
                dec[k].append(float(d.loc[q == k, "ex20"].mean()))
            t10 = d.nlargest(10, f)
            top.append(float((t10["ret"] - pb.COST - P["bench"]).mean()))
            tail_hi.append(float((d.loc[q == 9, "ex20"] < -0.10).mean()))
            tail_lo.append(float((d.loc[q == 0, "ex20"] < -0.10).mean()))
        r["deciles"] = [ac._r(np.mean(x) * 100) for x in dec]
        r["top_minus_bottom"] = _pp([a - b for a, b in zip(dec[9], dec[0])])
        r["top10_net"] = {**_pp(top), "dev": _pp([x for x, d in zip(top, dv) if d]), "holdout": _pp([x for x, d in zip(top, dv) if not d])}
        r["tail"] = {"top_decile_big_loss_pct": ac._r(np.mean(tail_hi) * 100), "bottom_decile_big_loss_pct": ac._r(np.mean(tail_lo) * 100)}
        for reg in ("bull", "bear"):
            r["regime"][reg] = _m([_ic(X[f], X["ex20"]) for P, X in FS if P.get("regime") == reg])
        for lab, k in (("大型", 2), ("中型", 1), ("小型", 0)):
            v = []
            for _, X in FS:
                t = pd.qcut(X["market_cap"].rank(method="first"), 3, labels=False)
                v.append(_ic(X.loc[t == k, f], X.loc[t == k, "ex20"]))
            r["size"][lab] = _m(v)
        ind = {}
        for _, X in FS:
            for g, Y in X.groupby(X["industry"].fillna("")):
                if len(Y.dropna(subset=[f, "ex20"])) >= MIN_IND:
                    ind.setdefault(g, []).append(_ic(Y[f], Y["ex20"]))
        r["industry"] = sorted([{"industry": g, "periods": len(v), "pos_pct": ac._r(np.mean([x > 0 for x in v]) * 100, 1), "ic": ac._r(np.nanmean(v), 3)}
                                for g, v in ind.items() if len(v) >= 20], key=lambda x: -(x["ic"] or 0))
        inc = []
        for _, X in FS:
            d = X[[f, "sc_balanced", "ex20"]].dropna()
            if len(d) < ac.MIN_N:
                continue
            a, b = d[f].rank(pct=True), d["sc_balanced"].rank(pct=True)
            beta = np.polyfit(b, a, 1)
            inc.append(_ic(a - np.polyval(beta, b), d["ex20"]))
        r["incremental"] = _m(inc)
        res[f] = r
    cor = []
    for _, X in FS:
        cor.append(X[list(FACTORS)].rank().corr().values)
    C = np.nanmean(np.array(cor), axis=0)
    return {"factors": res, "corr": {a: {b: round(float(C[i, j]), 2) for j, b in enumerate(FACTORS)} for i, a in enumerate(FACTORS)}}


def load_fs() -> list:
    """academic_study 存下的特徵快取（data/academic_features.pkl）；沒有就整個重建（約 30 分鐘）。"""
    f = config.DATA_DIR / "academic_features.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    D = pb.load("stocks.db")
    periods = pickle.loads(ac.CACHE.read_bytes())
    return [(P, ac.features(D, P)) for P in periods if len(P["df"]) and P["bench"] is not None]


def run(log=print) -> dict:
    FS = load_fs()
    out = audit(FS)
    out.update({"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
                "periods": len(FS), "first": FS[0][0]["signal"], "last": FS[-1][0]["signal"], "t_need": T_NEED, "holdout_from": HOLDOUT_FROM,
                "pit": "近似 PIT（月營收次月 11 日、財報法定期限＋緩衝）", "note": "只做診斷，不改正式權重；保留段 10/28 已看過一次，屬重複使用。"})
    return out


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps({f: {"ic20": v["ic"]["20"], "top10": v["top10_net"]["mean_pct"], "inc": v["incremental"]} for f, v in res["factors"].items()},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
