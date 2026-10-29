"""V21.2 MSURGE 組合研究（事前登記：docs/prereg_v21_2.md 第二節）→ history/backtest/msurge.json。

A 均衡型原版；B 均衡＋0.25 MSURGE；C MSURGE＋營收年增；D MSURGE＋獲利品質；E MSURGE＋產業動能；
F B 的前 30 名裡進場機會（趨勢＋短期反轉）最高 10 檔。B～F 各自對 A。通過也只進影子紀錄。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import academic_study as ac
from . import selection_study as ss
from .factor_audit import load_fs, _ic, _m

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "msurge.json"
STUDY_VERSION = "2026-10-29"
T_NEED = 2.58      # 5 個比較，Bonferroni
MDD_GAP = 5.0
HOLDOUT_FROM = ac.HOLDOUT_FROM
RULES = {"A": "均衡型原版前 10 名", "B": "0.75×均衡型＋0.25×MSURGE", "C": "MSURGE＋營收年增（等權）",
         "D": "MSURGE＋獲利品質（營業現金流／淨利、低應計）", "E": "MSURGE＋產業動能", "F": "B 的前 30 名裡進場機會最高 10 檔"}


def _p(d, c):
    return d[c].rank(pct=True) if c in d else pd.Series(0.5, index=d.index)


def picks(X: pd.DataFrame, rule: str, n: int = 10) -> list[str]:
    d = X[X["rk_balanced"].notna()]
    if rule == "A":
        return list(d.sort_values("rk_balanced").index[:n])
    ms = _p(d, "msurge").fillna(0.5)
    b = 0.75 * d["sc_balanced"].rank(pct=True) + 0.25 * ms
    if rule == "B":
        s = b
    elif rule == "C":
        s = (ms + _p(d, "rev_yoy").fillna(0.5)) / 2
    elif rule == "D":
        s = (ms + (_p(d, "ocf_ni").fillna(0.5) + _p(d, "low_accruals").fillna(0.5)) / 2) / 2
    elif rule == "E":
        s = (ms + _p(d, "ind_mom").fillna(0.5)) / 2
    else:
        top = b.sort_values(ascending=False).index[:30]
        e = (_p(d, "trend") + _p(d, "st_reversal")).loc[top] / 2
        return list(e.sort_values(ascending=False).index[:n])
    return list(s.sort_values(ascending=False).index[:n])


def compare(r, base, dv, s, sb):
    d = np.array(r) - np.array(base)
    dd = d[dv]
    h = len(dd) // 2
    v = {"dev_diff": ac._r(dd.mean() * 100), "dev_t": ac._r(ac._t(dd)), "first_half": ac._r(dd[:h].mean() * 100),
         "second_half": ac._r(dd[h:].mean() * 100), "holdout_diff": ac._r(d[~dv].mean() * 100), "mdd_gap": ac._r(s["mdd"] - sb["mdd"])}
    v["passed"] = bool(v["dev_t"] is not None and v["dev_t"] >= T_NEED and dd[:h].mean() > 0 and dd[h:].mean() > 0
                       and d[~dv].mean() > 0 and v["mdd_gap"] >= -MDD_GAP)
    return v


def side(FS) -> dict:
    """附帶檢查（只描述）：是否已被價格反映、產業、市值、農曆年季節性。"""
    pre_lo, pre_hi, jan, other = [], [], [], []
    size = {"大型": [], "中型": [], "小型": []}
    ind = {}
    for P, X in FS:
        d = X[X["rk_balanced"].notna()].dropna(subset=["msurge", "ex20"])
        if len(d) < 50:
            continue
        top = d[d["msurge"] >= d["msurge"].quantile(0.9)]
        if "st_reversal" in top and top["st_reversal"].notna().sum() >= 6:   # st_reversal＝過去 20 日報酬取負號
            m = top["st_reversal"].median()
            pre_hi.append(float(top.loc[top["st_reversal"] < m, "ex20"].mean()))    # 訊號前漲比較多（已反映）
            pre_lo.append(float(top.loc[top["st_reversal"] >= m, "ex20"].mean()))
        v = _ic(d["msurge"], d["ex20"])
        (jan if P["signal"][5:7] in ("01", "02", "03") else other).append(v)
        t = pd.qcut(d["market_cap"].rank(method="first"), 3, labels=False)
        for lab, k in (("大型", 2), ("中型", 1), ("小型", 0)):
            size[lab].append(_ic(d.loc[t == k, "msurge"], d.loc[t == k, "ex20"]))
        for g, Y in d.groupby(d["industry"].fillna("")):
            if len(Y) >= 15:
                ind.setdefault(g, []).append(_ic(Y["msurge"], Y["ex20"]))
    pct = lambda v: {"mean_pct": ac._r(np.nanmean(v) * 100), "t": ac._r(ac._t([x for x in v if not np.isnan(x)])), "n": len(v)}
    return {"already_priced": {"pre_up_half": pct(pre_hi), "pre_down_half": pct(pre_lo),
                               "note": "MSURGE 前十分位，依訊號前 20 日漲跌分兩半，20 日超額（未扣成本）"},
            "season": {"jan_mar_signals": _m(jan), "other_months": _m(other), "note": "1～3 月訊號會用到 12、1 月營收（農曆年月份），季節性最強"},
            "size": {k: _m(v) for k, v in size.items()},
            "industry": sorted([{"industry": g, "periods": len(v), "ic": ac._r(np.nanmean(v), 3), "pos_pct": ac._r(np.mean([x > 0 for x in v]) * 100, 1)}
                                for g, v in ind.items() if len(v) >= 20], key=lambda x: -(x["ic"] or 0))}


def run(log=print) -> dict:
    FS = load_fs()
    periods = [P for P, _ in FS]
    bench = [P["bench"] for P in periods]
    dv = np.array([P["signal"] < HOLDOUT_FROM for P in periods])
    R, S, PK = {}, {}, {}
    for k in RULES:
        PK[k] = [picks(X, k) for _, X in FS]
        R[k] = ss._net(periods, PK[k])
        S[k] = {"rule": RULES[k], **ss.summary(R[k], bench), "dev": ss.summary(list(np.array(R[k])[dv]), list(np.array(bench)[dv])),
                "holdout": ss.summary(list(np.array(R[k])[~dv]), list(np.array(bench)[~dv]))}
    cmp = {f"{k}_vs_A": compare(R[k], R["A"], dv, S[k], S["A"]) for k in "BCDEF"}
    over = {k: round(float(np.mean([len(set(a) & set(b)) for a, b in zip(PK[k], PK["A"])])), 1) for k in "BCDEF"}
    return {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
            "periods": len(FS), "first": periods[0]["signal"], "last": periods[-1]["signal"], "t_need": T_NEED, "mdd_gap": MDD_GAP,
            "holdout_from": HOLDOUT_FROM, "portfolios": S, "compare": cmp, "overlap_with_A": over,
            "passed": [k for k, v in cmp.items() if v["passed"]], "side": side(FS),
            "pit": "近似 PIT：月營收以次月 11 日起可用，沒有實際公告時間與更正紀錄",
            "note": "歷史回測（研究）；保留段 10/28 已看過一次，屬重複使用。通過也只進影子紀錄，正式模型不變。"}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps({"compare": res["compare"], "passed": res["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
