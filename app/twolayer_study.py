"""V21.2 投資價值 × 進場機會：兩層分數是否各自有用（事前登記）→ history/backtest/twolayer.json。

使用者 10/28 的方向：把「這家公司值不值得投資」和「現在是不是好的進場時機」分開。這裡先用歷史資料檢驗兩層是否真的各有資訊，
通過後才放到個股頁當正式說明；權重不自己編，用事前寫死的等權定義，結果不好就不採用。

事前固定（2026-10-28 寫下，跑之前推上 GitHub）：
- 樣本：selection_study 的 87 期（20 日不重疊，2019-07 起）、均衡型可排名母體、同一套開盤進出與成本；保留段 ≥ 2025-01-01。
- 投資價值分數 V ＝ 品質、成長、價值、月營收驚喜（MSURGE，academic_study 達 E2）四項百分位等權平均。
- 進場機會分數 E ＝ 趨勢分數、短期反轉（過去 20 日跌越多越高）兩項百分位等權平均（兩者在 academic／selection 研究中各有 t > 3 的證據）。
- 問題一：V、E 各自的 5／20／60 日 Rank IC。
- 問題二（關鍵）：在 V 前 30% 的股票裡，E 高的一半是否比 E 低的一半好（5、20 日，未扣成本，比的是時機）。
- 問題三：組合，前 10 名、扣成本、相對 0050：
  A＝均衡型原版；B＝只看 V；C＝V 前 30% 裡再挑 E 最高的 10 檔。C 對 A、C 對 B 逐期比較。
  門檻：開發段 t ≥ 2.4（2 個比較的 Bonferroni）、前後半段都 > 0、保留段 > 0、最大回撤不差超過 5 個百分點。
- 問題四：大型／中型／小型（市值三等分）各自的 V、E 的 IC，看是否只在某一群有效。
- 問題五：市值前 50 大（0050 成分股的近似，沒有歷史成分名單）在可排名母體與各組合前 10 名中的比例。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import pit_backtest as pb
from . import academic_study as ac
from . import selection_study as ss

STUDY_VERSION = "2026-10-28"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "twolayer.json"
FEAT_CACHE = config.DATA_DIR / "academic_features.pkl"
V_PARTS = ("quality", "growth", "value", "msurge")
E_PARTS = ("trend", "st_reversal")
TOP_V = 0.3
T_NEED = 2.4
MDD_GAP = 5.0
HOLDOUT_FROM = ss.HOLDOUT_FROM


def layers(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["V"] = pd.concat([X[c].rank(pct=True) for c in V_PARTS], axis=1).mean(axis=1, skipna=True)
    X["E"] = pd.concat([X[c].rank(pct=True) for c in E_PARTS], axis=1).mean(axis=1, skipna=True)
    return X


def _t(x):
    return ac._t(x)


def _ic(X, col, y):
    d = X[[col, y]].dropna()
    return float(d[col].rank().corr(d[y].rank())) if len(d) >= ac.MIN_N else np.nan


def picks(X: pd.DataFrame, rule: str, n: int = 10) -> list[str]:
    if rule == "A":
        return list(X.sort_values("rk_balanced").index[:n])
    if rule == "B":
        return list(X["V"].sort_values(ascending=False).index[:n])
    top = X[X["V"] >= X["V"].quantile(1 - TOP_V)]
    return list(top["E"].sort_values(ascending=False).index[:n])


def analyse(FS: list[tuple[dict, pd.DataFrame]]) -> dict:
    FS = [(P, layers(X)) for P, X in FS]
    res = {"ic": {}, "timing": {}, "portfolio": {}, "size": {}, "big50": {}}
    for col in ("V", "E"):
        res["ic"][col] = {}
        for h in (5, 20, 60):
            v = [_ic(X, col, f"ex{h}") for _, X in FS if f"ex{h}" in X]
            v = [x for x in v if not np.isnan(x)]
            res["ic"][col][str(h)] = {"mean": ac._r(np.mean(v), 4), "t": ac._r(_t(v)), "n": len(v)}
    for h in (5, 20):
        sp = []
        for _, X in FS:
            top = X[X["V"] >= X["V"].quantile(1 - TOP_V)].dropna(subset=["E", f"ex{h}"])
            if len(top) < 20:
                continue
            hi = top["E"] >= top["E"].median()
            sp.append(float(top.loc[hi, f"ex{h}"].mean() - top.loc[~hi, f"ex{h}"].mean()))
        h2 = len(sp) // 2
        res["timing"][str(h)] = {"high_minus_low": ac._r(np.mean(sp) * 100), "t": ac._r(_t(sp)), "n": len(sp),
                                 "first_half": ac._r(np.mean(sp[:h2]) * 100), "second_half": ac._r(np.mean(sp[h2:]) * 100)}
    periods = [P for P, _ in FS]
    bench = [P["bench"] for P in periods]
    dv = np.array([P["signal"] < HOLDOUT_FROM for P in periods])
    R, PK = {}, {}
    for rule in "ABC":
        PK[rule] = [picks(X, rule) for _, X in FS]
        R[rule] = ss._net(periods, PK[rule])
        res["portfolio"][rule] = {"rule": {"A": "均衡型原版前 10 名", "B": "只看投資價值 V 前 10 名", "C": "V 前 30% 裡進場機會 E 最高 10 檔"}[rule],
                                  **ss.summary(R[rule], bench), "dev": ss.summary(list(np.array(R[rule])[dv]), list(np.array(bench)[dv])),
                                  "holdout": ss.summary(list(np.array(R[rule])[~dv]), list(np.array(bench)[~dv]))}
    for a, b in (("C", "A"), ("C", "B"), ("B", "A")):
        d = np.array(R[a]) - np.array(R[b])
        dd = d[dv]
        h2 = len(dd) // 2
        gap = res["portfolio"][a]["mdd"] - res["portfolio"][b]["mdd"]
        v = {"dev_diff": ac._r(dd.mean() * 100), "dev_t": ac._r(_t(dd)), "first_half": ac._r(dd[:h2].mean() * 100),
             "second_half": ac._r(dd[h2:].mean() * 100), "holdout_diff": ac._r(d[~dv].mean() * 100), "mdd_gap": ac._r(gap)}
        v["passed"] = bool(v["dev_t"] is not None and v["dev_t"] >= T_NEED and dd[:h2].mean() > 0 and dd[h2:].mean() > 0
                           and d[~dv].mean() > 0 and gap >= -MDD_GAP)
        res["portfolio"][f"{a}_vs_{b}"] = v
    for lab, q in (("大型", 2), ("中型", 1), ("小型", 0)):
        out = {}
        for col in ("V", "E"):
            v = []
            for _, X in FS:
                t = pd.qcut(X["market_cap"].rank(method="first"), 3, labels=False)
                v.append(_ic(X[t == q], col, "ex20"))
            v = [x for x in v if not np.isnan(x)]
            out[col] = {"ic": ac._r(np.mean(v), 4), "t": ac._r(_t(v))}
        res["size"][lab] = out
    share = {k: [] for k in ("universe", "A", "B", "C")}
    for (P, X), i in zip(FS, range(len(FS))):
        big = set(P["df"]["market_cap"].dropna().nlargest(50).index)
        share["universe"].append(len(big & set(X.index)) / 50)
        for r in "ABC":
            share[r].append(len(big & set(PK[r][i])) / 10)
    res["big50"] = {k: round(float(np.mean(v)) * 100, 1) for k, v in share.items()}
    res["big50"]["note"] = "沒有歷史 0050 成分股名單，以訊號日市值前 50 大近似；universe 是前 50 大中有多少在可排名母體"
    res["periods"], res["first"], res["last"] = len(FS), periods[0]["signal"], periods[-1]["signal"]
    return res


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    if FEAT_CACHE.exists():
        FS = pickle.loads(FEAT_CACHE.read_bytes())
    else:
        D = pb.load(dbname)
        periods = pickle.loads(ac.CACHE.read_bytes()) if ac.CACHE.exists() else None
        if periods is None:
            ss._D, ss._CFG = D, config.load()
            periods = ss.collect(dbname, log=log)
        FS = [(P, ac.features(D, P)) for P in periods if len(P["df"]) and P["bench"] is not None]
    res = analyse(FS)
    res.update({"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds") + "+08:00",
                "v_parts": V_PARTS, "e_parts": E_PARTS, "top_v": TOP_V, "t_need": T_NEED, "mdd_gap": MDD_GAP,
                "holdout_from": HOLDOUT_FROM, "cost_pct": round(pb.COST * 100, 3), "seconds": round(time.time() - t0),
                "note": "歷史回測（研究）；兩層分數通過才放上個股頁當說明，正式排行榜不因本研究自動改變。"})
    return res


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps(res["portfolio"], ensure_ascii=False, default=float)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
