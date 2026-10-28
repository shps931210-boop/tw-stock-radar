"""V21.1 學術假說的台股驗證（事前登記）→ history/backtest/academic.json。

每個因子都來自 history/research/literature.json 的文獻；文獻只提供假說，台股結果才決定證據等級。

事前固定（2026-10-28 寫下，跑結果之前推上 GitHub，之後不能看結果再加減因子、方向、權重或門檻）：
- 樣本：和 selection_study 同一套 87 個不重疊 20 日期間（2019-07 起）、同一個股票母體（當天均衡型可排名）、
  同一套 hold_return（隔日開盤進、20 日後開盤出、還原除權息）、同樣成本 0.585%（只對新買進的部位扣）。
  開發段訊號日 < 2025-01-01；保留段 ≥ 2025-01-01。保留段月份過去其他研究看過，所以通過也只算 E2，不算 E3。
- 因子（方向照原論文，不看台股結果決定）：見 FACTORS。只用訊號日收盤前已公開的資料：
  月營收隔月 11 日起才用、季報法定期限隔天起才用（和正式評分同一套規則）。金融保險業不算財報品質因子。
- 每個因子做四種比較（共同樣本、相同成本）：
  1. 單因子：Rank IC（20 日；另列 5、60 日）、前 10% 等權相對母體平均（扣換手成本）。
  2. 增量：控制均衡型分數之後的殘差 Rank IC（月營收驚喜另外控制營收年增率）。
  3. 原模型＋因子：三種模式各自用「0.75×模式分數百分位＋0.25×因子百分位」重新排前 10 名，和原模型前 10 名逐期比較。
  4. 簡單基準：母體等權平均、0050。
- H05 風險因子另做「從原模型前 10 名剔除該因子最差 10% 的股票」，看重大虧損比例和最大回撤。
- 證據等級（門檻全部要過）：
  E2（因子層級）：開發段前 10% 扣成本超額 t ≥ T_FACTOR（13 個因子 Bonferroni）、前後半段都 > 0、保留段 > 0、增量 IC t ≥ 2。
  E2（模型層級）：原模型＋因子 vs 原模型，開發段 t ≥ T_MODEL（13 因子 × 3 模式 Bonferroni）、前後半段都 > 0、保留段 > 0、
  最大回撤不差超過 5 個百分點。
  E3 要新的、沒人看過的資料（前瞻影子紀錄）；E4 要多批正式到期成績。本研究最多只能給到 E2。
- H03（法人 × 市況）：資料庫的法人買賣超只從 2026-08 開始，歷史不夠，本次不測，列為資料缺口。
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
from . import selection_study as ss

STUDY_VERSION = "2026-10-28"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "academic.json"
CACHE = config.DATA_DIR / "selection_periods.pkl"
HOLDOUT_FROM = ss.HOLDOUT_FROM
BLEND = 0.25
T_FACTOR = 2.7
T_MODEL = 2.9
MDD_GAP = 5.0
MIN_N = 50
# key: (假說, 名稱, 文獻 ID, 方向說明)；值越大＝論文預期報酬越高
FACTORS = {
    "msurge": ("H01", "月營收驚喜（MSURGE）", "P022", "本月營收年增額相對過去 12 個月年增額的標準化偏離，越高越好"),
    "rev_yoy": ("H01", "營收年增率（現行基準）", "P022", "對照組：目前成長分數用的營收年增率"),
    "rev_mom": ("H01", "營收月增率（現行基準）", "P022", "對照組：營收月增率"),
    "ind_mom": ("H02", "產業動能", "P005", "同產業股票過去 6 個月（跳過最近 1 個月）平均報酬，越高越好"),
    "rel_ind_mom": ("H02", "相對產業動能", "P002", "個股 6-1 個月報酬減產業平均，越高越好"),
    "hi52": ("H02", "接近 52 週高點", "P004", "收盤價除以過去 252 日最高價，越接近越好"),
    "st_reversal": ("H02", "短期反轉", "P003", "過去 20 日報酬取負號，跌越多越好"),
    "gp_assets": ("H04", "毛利／總資產", "P007", "近四季毛利除以總資產，越高越好"),
    "ocf_ni": ("H04", "現金流／淨利", "P008", "近四季營業現金流除以淨利（淨利 > 0），越高越好"),
    "low_accruals": ("H04", "低應計項目", "P008", "－（近四季淨利－營業現金流）÷平均總資產，越高越好"),
    "low_asset_growth": ("H04", "低資產成長", "P006", "－總資產年增率，越高越好"),
    "illiq": ("H05", "Amihud 非流動性", "P011", "過去 20 日 |報酬|÷成交金額平均，論文預期越高報酬越高（但較難成交）"),
    "low_max": ("H05", "低極端漲幅（MAX）", "P012", "－過去 20 日最大單日報酬，越高越好"),
}
RISK_EXCLUDE = {"illiq": "最不流動 10%", "low_max": "單日暴漲最大 10%", "low_dvol": "下行波動最大 10%"}


# ---------------- 特徵（只用訊號日以前看得到的資料）----------------
def _rev_feats(R: pd.DataFrame, sig: str) -> dict:
    if R is None or not len(R):
        return {}
    avail = R[R["ym"].map(pb.rev_available) <= sig]
    if not len(avail):
        return {}
    s = pd.Series(avail["revenue"].astype(float).values, index=pd.PeriodIndex(avail["ym"], freq="M"))
    t = s.index[-1]
    if (pd.Period(sig[:7], "M") - t).n > 3:
        return {}
    g = lambda p: s.get(p)
    out = {}
    cur, ly, pm = g(t), g(t - 12), g(t - 1)
    if cur is not None and ly and ly > 0:
        out["rev_yoy"] = cur / ly - 1
    if cur is not None and pm and pm > 0:
        out["rev_mom"] = cur / pm - 1
    ur = []
    for k in range(13):
        a, b = g(t - k), g(t - k - 12)
        ur.append(None if a is None or b is None else a - b)
    hist = [x for x in ur[1:] if x is not None]
    if ur[0] is not None and len(hist) >= 9:
        sd = float(np.std(hist, ddof=1))
        if sd > 0:
            out["msurge"] = (ur[0] - float(np.mean(hist))) / sd
    return out


def _fin_feats(F: pd.DataFrame, sig: str, financial: bool) -> dict:
    if F is None or financial or len(F) < 5:
        return {}
    f = F[F["period"].map(lambda p: pb.fin_available(p)) <= sig].sort_values("period_end")
    if len(f) < 5:
        return {}
    last4 = f.iloc[-4:]
    num = lambda k: last4[k].astype(float)
    if num("gross_profit").isna().any() or num("net_income").isna().any():
        return {}
    ta, ta4 = float(f["total_assets"].iloc[-1] or 0), float(f["total_assets"].iloc[-5] or 0)
    gp, ni = float(num("gross_profit").sum()), float(num("net_income").sum())
    ocf = None if num("ocf").isna().any() else float(num("ocf").sum())
    out = {}
    if ta > 0:
        out["gp_assets"] = gp / ta
    if ta > 0 and ta4 > 0:
        out["low_asset_growth"] = -(ta / ta4 - 1)
        if ocf is not None:
            out["low_accruals"] = -(ni - ocf) / ((ta + ta4) / 2)
    if ocf is not None and ni > 0:
        out["ocf_ni"] = ocf / ni
    return out


def _px_feats(a: pd.DataFrame, sig: str) -> dict:
    if a is None:
        return {}
    a = a[a.index <= sig]
    if len(a) < 253:
        return {}
    c = a["close"].astype(float).to_numpy()
    r = np.diff(c) / c[:-1]
    v = a["value"].astype(float).to_numpy()[1:]
    out = {"hi52": c[-1] / np.nanmax(c[-252:]), "r6_1": c[-21] / c[-126] - 1, "st_reversal": -(c[-1] / c[-21] - 1),
           "low_max": -float(np.nanmax(r[-20:]))}
    ok = v[-20:] > 0
    if ok.sum() >= 10:
        out["illiq"] = float(np.mean(np.abs(r[-20:][ok]) / v[-20:][ok]) * 1e9)
    neg = r[-60:][r[-60:] < 0]
    out["low_dvol"] = -float(np.sqrt(np.mean(neg ** 2))) if len(neg) >= 5 else None
    return out


def features(D: dict, P: dict) -> pd.DataFrame:
    sig = P["signal"]
    df = P["df"]
    df = df[df["rk_balanced"].notna() & df["ret"].notna()].copy()
    rows = {}
    for sid in df.index:
        ind = df.at[sid, "industry"] or ""
        x = {**_px_feats(D["adj"].get(sid), sig), **_rev_feats(D["rev"].get(sid), sig),
             **_fin_feats(D["fin"].get(sid), sig, "金融" in ind)}
        rows[sid] = x
    X = pd.DataFrame.from_dict(rows, orient="index").reindex(df.index)
    out = df.join(X)
    if "r6_1" in out:
        g = out.groupby(out["industry"].fillna(""))["r6_1"]
        n = g.transform("count")
        out["ind_mom"] = g.transform("mean").where(n >= 3)
        out["rel_ind_mom"] = out["r6_1"] - out["ind_mom"]
    # 5／60 日報酬（輔助期限）
    pos = {d: i for i, d in enumerate(D["dates"])}
    i = pos[P["entry"]]
    for h in (5, 60):
        if i + h < len(D["dates"]):
            ex = D["dates"][i + h]
            b = pb._bench(D, P["entry"], ex)
            out[f"ex{h}"] = [(pb.hold_return(D, s, P["entry"], ex)[0] or np.nan) - (b or 0) for s in out.index]
    out["ex20"] = out["ret"] - P["bench"]
    return out


# ---------------- 統計 ----------------
def _t(x) -> float | None:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def _r(x, k=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def _resid_ic(sig: pd.Series, ctrl: list[pd.Series], y: pd.Series) -> float | None:
    d = pd.concat([sig.rank(pct=True), *[c.rank(pct=True) for c in ctrl], y.rank(pct=True)], axis=1).dropna()
    if len(d) < MIN_N:
        return None
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1:-1].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d.iloc[:, 0].to_numpy(), rcond=None)
    e = d.iloc[:, 0].to_numpy() - X @ beta
    return float(np.corrcoef(pd.Series(e).rank(), d.iloc[:, -1].rank().to_numpy())[0, 1])


def factor_eval(FS: list[tuple[dict, pd.DataFrame]], f: str) -> dict:
    ic = {h: [] for h in (5, 20, 60)}
    inc, inc2, spread, dev = [], [], [], []
    prev = set()
    for P, X in FS:
        d = X[X[f].notna()]
        if len(d) < MIN_N:
            continue
        for h in ic:
            y = d.get(f"ex{h}")
            if y is not None and y.notna().sum() >= MIN_N:
                ic[h].append(float(d[f].rank().corr(y.rank())))
        inc.append(_resid_ic(d[f], [d["sc_balanced"]], d["ex20"]))
        if f == "msurge" and "rev_yoy" in d:
            inc2.append(_resid_ic(d[f], [d["sc_balanced"], d["rev_yoy"]], d["ex20"]))
        k = max(1, len(d) // 10)
        top = d.nlargest(k, f)
        ids = set(top.index)
        turn = len(ids - prev) / len(ids)
        prev = ids
        spread.append(float(top["ret"].mean() - X["ret"].mean() - pb.COST * turn))
        dev.append(P["signal"] < HOLDOUT_FROM)
    sp, dv = np.array(spread), np.array(dev)
    dd = sp[dv]
    h = len(dd) // 2
    out = {"periods": int(len(sp)),
           "ic": {str(k): {"mean": _r(np.mean(v), 4) if v else None, "t": _r(_t(v)), "n": len(v)} for k, v in ic.items()},
           "inc_ic": _r(np.nanmean([x for x in inc if x is not None]), 4), "inc_ic_t": _r(_t([x for x in inc if x is not None])),
           "top10pct_net": _r(sp.mean() * 100), "dev_net": _r(dd.mean() * 100), "dev_t": _r(_t(dd)),
           "first_half": _r(dd[:h].mean() * 100), "second_half": _r(dd[h:].mean() * 100),
           "holdout_net": _r(sp[~dv].mean() * 100) if (~dv).any() else None}
    if inc2:
        x = [v for v in inc2 if v is not None]
        out["inc_ic_vs_yoy"], out["inc_ic_vs_yoy_t"] = _r(np.mean(x), 4), _r(_t(x))
    out["e2_factor"] = bool(out["dev_t"] is not None and out["dev_t"] >= T_FACTOR and dd[:h].mean() > 0 and dd[h:].mean() > 0
                            and out["holdout_net"] is not None and out["holdout_net"] > 0
                            and out["inc_ic_t"] is not None and out["inc_ic_t"] >= 2)
    return out


def _top(X: pd.DataFrame, mode: str, f: str | None, n: int = 10, exclude: str | None = None) -> list[str]:
    d = X[X[f"rk_{mode}"].notna()] if f"rk_{mode}" in X else X.iloc[0:0]
    if exclude:   # 剔除風險最差的 10%：illiq 越高越差；low_max、low_dvol 越低越差
        bad = d[exclude] >= d[exclude].quantile(0.9) if exclude == "illiq" else d[exclude] <= d[exclude].quantile(0.1)
        d = d[~bad.fillna(False)]
    if f is None:
        return list(d.sort_values(f"rk_{mode}").index[:n])
    s = (1 - BLEND) * d[f"sc_{mode}"].rank(pct=True) + BLEND * d[f].rank(pct=True).fillna(0.5)
    return list(s.sort_values(ascending=False).index[:n])


def blend_eval(FS, _unused, mode: str, f: str | None, exclude: str | None = None) -> dict:
    """原模型（或＋因子、或剔除風險）前 10 名的逐期扣成本報酬；買不到留現金，和 selection_study 同一套成本。"""
    picks = []
    for P, X in FS:
        full = P["df"]
        if f is not None or exclude:   # 因子只算在均衡型母體；其他可排名股票因子值當中位數（不加分也不扣分）
            full = full.join(X[[c for c in X.columns if c not in full.columns]])
        picks.append(_top(full, mode, f, exclude=exclude))
    return ss._net([P for P, _ in FS], picks), picks


def model_eval(FS, mode: str, base: list[float], f: str | None = None, exclude: str | None = None) -> dict:
    r, pk = blend_eval(FS, None, mode, f, exclude)
    bench = [P["bench"] for P, _ in FS]
    dv = np.array([P["signal"] < HOLDOUT_FROM for P, _ in FS])
    diff = np.array(r) - np.array(base)
    dd = diff[dv]
    h = len(dd) // 2
    s = ss.summary(r, bench)
    sb = ss.summary(base, bench)
    v = {"dev_diff": _r(dd.mean() * 100), "dev_t": _r(_t(dd)), "first_half": _r(dd[:h].mean() * 100), "second_half": _r(dd[h:].mean() * 100),
         "holdout_diff": _r(diff[~dv].mean() * 100), "mdd_gap": _r(s["mdd"] - sb["mdd"]), "big_loss_gap": _r(s["big_loss"] - sb["big_loss"], 1)}
    v["e2_model"] = bool(v["dev_t"] is not None and v["dev_t"] >= T_MODEL and dd[:h].mean() > 0 and dd[h:].mean() > 0
                         and v["holdout_diff"] > 0 and v["mdd_gap"] >= -MDD_GAP)
    return {**s, "vs_base": v}


def analyse(D: dict, periods: list[dict], log=print) -> dict:
    periods = [P for P in periods if len(P["df"]) and P["bench"] is not None]
    t0 = time.time()
    FS = [(P, features(D, P)) for P in periods]
    log(f"特徵 {time.time() - t0:.0f} 秒")
    cov = {f: round(float(np.mean([X[f].notna().mean() if f in X else 0 for _, X in FS])) * 100, 1) for f in list(FACTORS) + ["low_dvol"]}
    res = {"factors": {}, "coverage_pct": cov, "modes": {}}
    for f, (hyp, name, ref, desc) in FACTORS.items():
        res["factors"][f] = {"hypothesis": hyp, "name": name, "ref": ref, "direction": desc, **factor_eval(FS, f)}
    bench = [P["bench"] for P, _ in FS]
    for m in ("stable", "balanced", "aggressive"):
        base, _ = blend_eval(FS, None, m, None)
        M = {"base": ss.summary(base, bench), "blend": {}, "risk_exclude": {}}
        for f in FACTORS:
            M["blend"][f] = model_eval(FS, m, base, f)
        for f in RISK_EXCLUDE:
            M["risk_exclude"][f] = model_eval(FS, m, base, None, exclude=f)
        res["modes"][m] = M
    ew = [float(X["ret"].mean()) for _, X in FS]
    res["simple"] = {"universe_ew": ss.summary(ew, bench)}
    res["e2_factor"] = [f for f, v in res["factors"].items() if v["e2_factor"]]
    res["e2_model"] = [f"{m}:{f}" for m, M in res["modes"].items() for f, v in M["blend"].items() if v["vs_base"]["e2_model"]]
    res["periods"], res["first"], res["last"] = len(FS), FS[0][0]["signal"], FS[-1][0]["signal"]
    return res


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    D = pb.load(dbname)
    if CACHE.exists():
        periods = pickle.loads(CACHE.read_bytes())
    else:
        ss._D, ss._CFG = D, config.load()
        periods = ss.collect(dbname, log=log)
    res = analyse(D, periods, log)
    res.update({"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds") + "+08:00",
                "holdout_from": HOLDOUT_FROM, "blend_weight": BLEND, "t_factor": T_FACTOR, "t_model": T_MODEL, "mdd_gap": MDD_GAP,
                "cost_pct": round(pb.COST * 100, 3), "data_end": D["dates"][-1], "seconds": round(time.time() - t0),
                "gaps": {"H03": "法人買賣超資料只從 2026-08 開始，歷史不足，本次未測"},
                "note": "歷史回測（研究），最多只能給到 E2；正式模型不因本研究自動改變。"})
    return res


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print("E2 因子：", res["e2_factor"] or "沒有", "；E2 模型：", res["e2_model"] or "沒有")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
