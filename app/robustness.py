"""V30 LightGBM 可信度檢查（研究，影子模式）→ race.json 的 "robust"。

模型比賽（app/model_race.py）回答「誰比較好」；這裡回答「好的那個是不是靠運氣」：
1. 去掉最賺的 1%、5% 持股（每個模型去掉自己的），重算每期超額報酬 —— 看是不是少數爆發股撐起來的。
2. 去掉最好的一期 —— 看是不是一個月撐起來的。
3. 大型、中型、小型股（訊號日當天市值三等分）各自的 Rank IC 與前 10 名 —— 看是不是只在某一群股票有效。
4. 四種市場環境（加權指數在半年線上／下 × 波動高／低，波動門檻只用當天以前的資料）。
5. 交易成本敏感度：成本變成 1.5 倍、2 倍、3 倍，超額報酬還剩多少。
6. 模型分歧：LightGBM 前 10 名裡，V21 也看好的股票和 V21 不看好的股票，之後表現差多少 —— 分歧能不能預測選錯。
7. 特徵消融：LightGBM 拿掉一組特徵重新訓練（同一套 walk-forward），看哪一組真的有用。
8. 特徵重要性每年的變化。
這些都是「診斷」，不改正式模型、不改比賽門檻；消融結果若要拿來改特徵，必須另外事先寫下新的比賽規則再驗證一次。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import factor_research as fr
from . import track

H = 20
TOPN = 10
TRIM = (0.01, 0.05)
COST_X = (1.5, 2.0, 3.0)
MODELS = ("v21", "lgbm")
SIZE = {"small": "小型股", "mid": "中型股", "large": "大型股"}
REG4 = {"up_calm": "多頭・低波動", "up_wild": "多頭・高波動", "down_calm": "空頭・低波動", "down_wild": "空頭・高波動"}
# 特徵分組（V30 事先固定）；資料庫目前沒有法人、融資券等籌碼資料，所以沒有籌碼組
GROUPS = {
    "fundamental": ("基本面（品質、成長、價值、財報、營收）",
                    ["score_quality", "score_growth", "score_value", "rev_yoy", "rev_yoy_3m", "eps_q_yoy", "op_q_yoy", "roe", "gross_margin",
                     "op_margin", "debt_ratio", "earnings_yield", "book_to_price", "dy", "fcf_yield", "fscore"]),
    "price": ("價格與技術（動能、趨勢、相對強弱、量能）",
              ["score_momentum", "score_trend", "mom_12_1", "ret_1m", "ret_3m", "rs_rank", "dist_ma60", "ma60_slope_20", "rsi", "vol_ratio"]),
    "risk": ("風險（波動、下檔波動、Beta、最大回撤）", ["vol_1y", "downside_vol", "beta", "mdd_1y"]),
    "size_liq": ("規模與流動性（市值、成交金額）", ["market_cap", "avg_value_20"]),
}


def _t(x) -> float | None:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2)


def holdings(preds: pd.DataFrame, m: str, dates_used: list[str]) -> pd.DataFrame:
    """不重疊期間每期的前 10 名持股與換股成本（和 model_race.portfolio 同一套規則）。"""
    y = f"y{H}"
    rows, prev = [], set()
    for d in dates_used:
        g = preds[preds["date"] == d]
        if m not in g or g[m].notna().sum() < TOPN:
            continue
        top = g.nlargest(TOPN, m)
        ids = set(top["sid"])
        cost = track.COST * len(ids - prev) / TOPN
        for _, r in top.iterrows():
            rows.append({"date": d, "sid": r["sid"], "y": float(r[y]), "cost": cost})
        prev = ids
    return pd.DataFrame(rows, columns=["date", "sid", "y", "cost"])


def _period_net(h: pd.DataFrame, drop: pd.Series | None = None) -> pd.Series:
    x = h if drop is None else h[~drop]
    return x.groupby("date").agg(y=("y", "mean"), cost=("cost", "first")).eval("y - cost")


def trim_study(h: pd.DataFrame) -> dict:
    """去掉最賺的持股、最好的一期後的每期超額報酬（%）。"""
    if not len(h):
        return {}
    base = _period_net(h)
    out = {"all": round(float(base.mean()) * 100, 2), "periods": int(len(base)), "positions": int(len(h))}
    for q in TRIM:
        cut = h["y"].quantile(1 - q)
        s = _period_net(h, h["y"] > cut)
        out[f"no_top{int(q * 100)}"] = round(float(s.mean()) * 100, 2)
    if len(base) > 1:
        out["no_best_period"] = round(float(base.drop(base.idxmax()).mean()) * 100, 2)
        out["best_period"] = {"date": base.idxmax(), "net": round(float(base.max()) * 100, 2)}
    out["top_contrib"] = [{"date": r.date, "sid": r.sid, "excess": round(r.y * 100, 1)} for r in h.nlargest(5, "y").itertuples()]
    return out


def cost_study(h: pd.DataFrame) -> dict:
    if not len(h):
        return {}
    g = h.groupby("date").agg(y=("y", "mean"), cost=("cost", "first"))
    return {f"x{k:g}": round(float((g["y"] - g["cost"] * k).mean()) * 100, 2) for k in (1.0, *COST_X)}


def size_study(preds: pd.DataFrame, models: list[str], dates_used: list[str], portfolio) -> dict:
    """訊號日當天市值三等分，每一群裡各自排名、買前 10 名。"""
    if "market_cap" not in preds:
        return {}
    P = preds[preds["market_cap"].notna()].copy()
    P["size"] = P.groupby("date")["market_cap"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=list(SIZE)))
    y = f"y{H}"
    out = {}
    for k in SIZE:
        g = P[P["size"] == k]
        row = {}
        for m in models:
            ics = [fr._ic(x[m], x[y]) for _, x in g.groupby("date") if x[m].notna().sum() > 30]
            ics = np.array([v for v in ics if v is not None and not np.isnan(v)])
            p = portfolio(g, m, dates_used, {})
            row[m] = {"ic": round(float(ics.mean()), 4) if len(ics) else None, "ic_t": round(float(ics.mean() / fr._nw_se(ics, 2)), 2) if len(ics) >= 4 else None,
                      "net_excess": p.get("net_excess"), "t": p.get("t"), "periods": p.get("periods", 0)}
        out[k] = row
    return out


def regime4(idx: pd.Series | None) -> dict:
    """每個交易日的四種市場環境；波動門檻是「當天以前所有 20 日波動的中位數」，不用未來資料。"""
    if idx is None or not len(idx):
        return {}
    s = idx.astype(float).sort_index()
    ma = s.rolling(120).mean()
    vol = s.pct_change().rolling(20).std()
    med = vol.expanding(60).median()
    out = {}
    for d in s.index:
        if pd.isna(ma[d]) or pd.isna(vol[d]) or pd.isna(med[d]):
            continue
        out[d] = ("up" if s[d] >= ma[d] else "down") + "_" + ("wild" if vol[d] > med[d] else "calm")
    return out


def regime_study(preds: pd.DataFrame, models: list[str], reg: dict) -> dict:
    y = f"y{H}"
    acc: dict = {}
    for d, g in preds.groupby("date"):
        k = reg.get(d)
        if not k:
            continue
        for m in models:
            o = acc.setdefault(k, {}).setdefault(m, {"ic": [], "top": []})
            o["ic"].append(fr._ic(g[m], g[y]))
            o["top"].append(float(g.nlargest(TOPN, m)[y].mean()) - track.COST)
    out = {}
    for k in REG4:
        if k not in acc:
            continue
        out[k] = {m: {"dates": len(v["ic"]), "ic": round(float(np.nanmean(v["ic"])), 4), "top_net_excess": round(float(np.mean(v["top"])) * 100, 2)}
                  for m, v in acc[k].items()}
    return out


def disagreement(preds: pd.DataFrame) -> dict:
    """LightGBM 前 10 名裡：V21 也排前半的（同意）和 V21 排後半的（不同意），之後 20 日超額報酬與大幅落後比例。
    另看每一天兩個模型排名的相關係數高低，和當天 LightGBM 的 Rank IC 有沒有關係。"""
    if "lgbm" not in preds:
        return {}
    y = f"y{H}"
    rows, days = [], []
    for d, g in preds.groupby("date"):
        if g["lgbm"].notna().sum() < TOPN:
            continue
        v = g["v21"].rank(pct=True)
        top = g.nlargest(TOPN, "lgbm")
        for i, r in top.iterrows():
            rows.append({"date": d, "agree": bool(v[i] >= 0.5), "y": float(r[y])})
        days.append({"date": d, "corr": fr._ic(g["v21"], g["lgbm"]), "ic": fr._ic(g["lgbm"], g[y])})
    R, Dd = pd.DataFrame(rows), pd.DataFrame(days).dropna()
    out = {}
    for k, name in ((True, "agree"), (False, "disagree")):
        x = R[R["agree"] == k]
        out[name] = {"n": int(len(x)), "avg_excess": round(float(x["y"].mean()) * 100, 2) if len(x) else None,
                     "big_loss_pct": round(float((x["y"] < -0.10).mean()) * 100, 1) if len(x) else None}
    diffs = [float(g.loc[g["agree"], "y"].mean() - g.loc[~g["agree"], "y"].mean()) for _, g in R.groupby("date") if g["agree"].any() and (~g["agree"]).any()]
    out["diff"] = round(float(np.mean(diffs)) * 100, 2) if diffs else None
    out["t"] = round(float(np.mean(diffs) / fr._nw_se(np.array(diffs), 2)), 2) if len(diffs) >= 4 else None
    if len(Dd) >= 6:
        c = float(np.corrcoef(Dd["corr"], Dd["ic"])[0, 1])
        out["day_corr_vs_ic"] = round(c, 3)
        lo = Dd["corr"] <= Dd["corr"].median()
        out["ic_low_agreement_days"] = round(float(Dd.loc[lo, "ic"].mean()), 4)
        out["ic_high_agreement_days"] = round(float(Dd.loc[~lo, "ic"].mean()), 4)
    return out


def ablation_study(preds: pd.DataFrame, dates_used: list[str], hold_use: list[str], portfolio, ic_stats) -> dict:
    if "lgbm" not in preds:
        return {}
    base = portfolio(preds, "lgbm", dates_used, {})
    base_h = portfolio(preds, "lgbm", hold_use, {})
    out = {"lgbm": {"net_excess": base.get("net_excess"), "holdout": base_h.get("net_excess"), "ic": ic_stats(preds, "lgbm").get("ic")}}
    for g, (name, feats) in GROUPS.items():
        c = f"abl_{g}"
        if c not in preds:
            continue
        p, ph = portfolio(preds, c, dates_used, {}), portfolio(preds, c, hold_use, {})
        a, b = p.get("series") or [], base.get("series") or []
        n = min(len(a), len(b))
        d = np.array(a[:n]) - np.array(b[:n])
        out[g] = {"name": name, "features": len(feats), "net_excess": p.get("net_excess"), "holdout": ph.get("net_excess"), "ic": ic_stats(preds, c).get("ic"),
                  "diff_vs_full": round(float(d.mean()), 2) if n else None, "t": _t(d)}
        dn, di = out[g]["diff_vs_full"], (out[g]["ic"] or 0) - (out["lgbm"]["ic"] or 0)
        out[g]["reading"] = ("拿掉後明顯變差：這組特徵有用" if (out[g]["t"] or 0) <= -2 else
                             "拿掉後明顯變好：這組特徵可能是雜訊" if (out[g]["t"] or 0) >= 2 else
                             "差異不顯著" + ("（IC 下降）" if di < -0.005 else "（IC 上升）" if di > 0.005 else ""))
    return out


def importance_by_year(imp: list) -> dict:
    """每年 LightGBM 各組特徵的重要性占比（gain，%）。重要性不是因果，只看穩不穩定。"""
    if not imp:
        return {}
    acc: dict = {}
    for d, m in imp:
        tot = sum(m.values()) or 1.0
        yr = d[:4]
        for g, (_, feats) in GROUPS.items():
            acc.setdefault(yr, {}).setdefault(g, []).append(sum(m.get(f, 0.0) for f in feats) / tot)
    return {yr: {g: round(float(np.mean(v)) * 100, 1) for g, v in gs.items()} for yr, gs in sorted(acc.items())}


def run(preds: pd.DataFrame, tests: list[str], cut: str, idx: pd.Series | None, imp: list, portfolio, ic_stats) -> dict:
    use = tests[0::2]
    hold_use = [d for d in use if d >= cut]
    models = [m for m in MODELS if m in preds]
    res = {"version": "2026-10-22", "models": models, "trim": {}, "trim_holdout": {}, "cost": {}}
    for m in models:
        h = holdings(preds, m, use)
        res["trim"][m] = trim_study(h)
        res["trim_holdout"][m] = trim_study(h[h["date"] >= cut])
        res["cost"][m] = cost_study(h)
    res["size"] = size_study(preds, models, use, portfolio)
    res["regime4"] = regime_study(preds, models, regime4(idx))
    res["regime4_names"] = REG4
    res["disagreement"] = disagreement(preds)
    res["ablation"] = ablation_study(preds, use, hold_use, portfolio, ic_stats)
    res["importance"] = importance_by_year(imp)
    res["groups"] = {g: {"name": n, "features": f} for g, (n, f) in GROUPS.items()}
    res["findings"] = findings(res)
    return res


def findings(r: dict) -> list[str]:
    """把結果寫成白話；只描述數字，不下「模型變好」的結論。"""
    out = []
    t = r["trim"].get("lgbm") or {}
    th = r["trim_holdout"].get("lgbm") or {}
    if th.get("all") is not None:
        out.append(f"LightGBM 保留測試期每期 {th['all']:+.2f}%；去掉最賺的 5% 持股後 {th.get('no_top5', 0):+.2f}%，"
                   f"再去掉最好的一期後 {th.get('no_best_period', 0):+.2f}%（只有 {th.get('periods')} 期）。")
    if t.get("all") is not None:
        v = r["trim"].get("v21") or {}
        out.append(f"全部期間：LightGBM {t['all']:+.2f}% → 去掉最賺 5% 後 {t.get('no_top5', 0):+.2f}%；V21 {v.get('all', 0):+.2f}% → {v.get('no_top5', 0):+.2f}%。")
    c = r["cost"].get("lgbm") or {}
    if c:
        out.append(f"交易成本變 2 倍時，LightGBM 每期超額 {c.get('x2', 0):+.2f}%（原本 {c.get('x1', 0):+.2f}%）。")
    dz = r.get("disagreement") or {}
    if dz.get("diff") is not None:
        out.append(f"LightGBM 前 10 名中，V21 也看好的股票比 V21 不看好的平均多 {dz['diff']:+.2f}%（t {dz.get('t')}）。")
    return out
