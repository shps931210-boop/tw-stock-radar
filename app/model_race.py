"""V28 20 日模型比賽（研究，影子模式）→ history/backtest/race.json。

選股準確度是最高優先：這裡回答「哪一個模型在沒看過的資料上，前 10 名扣成本後真的比較好？」
規則在 2026-10-17 看結果之前寫下（RACE_VERSION），之後不依結果修改期間、參數或門檻：

- 母體與標籤：和 ml_research 同一個資料集（PIT 特徵、訊號日隔天開盤進場、20 個交易日後開盤出場、減 0050）。
  訓練目標是「當天所有股票的超額報酬名次」：同一天每檔股票減掉的是同一個基準，所以名次就是「誰比較會跑贏大盤」，
  不受大盤當期漲跌干擾；用加權指數或 0050 當基準，名次完全相同，只有超額報酬的數值不同（兩個都報告）。
- 四個模型：V21 固定權重（正式）、Ridge、LightGBM、LightGBM＋產業與市場特徵。參數全部事先固定。
- Purged walk-forward：每個測試日只用標籤已經到期的歷史訓練；最後 30% 的測試日是最終保留測試期。
- 主要指標：前 10 名扣成本後超額報酬（不重疊的 20 日期間，換股才收成本）。輔助：Rank IC、命中率、最大回撤。
- 升級條件（全部要過）：
  1. 前 10 名扣成本超額報酬比 V21 高，t ≥ T_NEED（三個挑戰者，Bonferroni 修正）；
  2. 最終保留測試期也比 V21 高；
  3. Rank IC 平均高於 V21；
  4. 最大回撤不比 V21 差超過 5 個百分點。
  沒有全過就維持 V21，而且必須如實寫「沒有證據變好」。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import factor_research as fr
from . import pit_backtest as pb
from . import track

RACE_VERSION = "2026-10-17"
H = 20
TOPN = 10
HOLDOUT = 0.3
MODELS = {"v21": "V21 固定權重（正式）", "ridge": "Ridge 線性迴歸", "lgbm": "LightGBM", "lgbm_x": "LightGBM＋產業與市場特徵"}
CHALLENGERS = ("ridge", "lgbm", "lgbm_x")
T_NEED = 2.39          # 雙尾 5% ÷ 3 個挑戰者（Bonferroni）
MDD_GAP = 5.0
BIG_LOSS = -0.10       # 20 日落後 0050 超過 10% 算「大幅落後」
IND_MIN = 5            # 產業特徵：當天至少 5 檔才算
IND_FEATS = ["ind_ret_1m", "ind_ret_3m", "ind_mom", "rel_ind_3m", "ind_n"]
MKT_FEATS = ["mkt_ret_20", "mkt_ma120", "mkt_vol_20", "breadth"]
# 積極型風險組合研究（事先固定的門檻；用全部歷史，不用 10/12 那一週調整）
COMBO = {"hot": ("ret_3m", ">=", 0.8, "近 3 個月漲幅前 20%"), "rich": ("earnings_yield", "<=", 0.2, "盈餘殖利率最低 20%（估值最高）"),
         "volatile": ("vol_1y", ">=", 0.8, "一年波動率前 20%"), "illiquid": ("avg_value_20", "<", math.log(1e8), "20 日均成交金額 < 1 億")}


def add_features(panel: pd.DataFrame, idx: pd.Series | None) -> pd.DataFrame:
    """產業與市場特徵：只用訊號日當天（含）以前的資料。"""
    P = panel.copy()
    g = P.groupby(["date", "industry"])
    n = g["sid"].transform("size")
    for col, src in (("ind_ret_1m", "ret_1m"), ("ind_ret_3m", "ret_3m"), ("ind_mom", "mom_12_1")):
        P[col] = g[src].transform("mean").where((n >= IND_MIN) & (P["industry"] != ""))
    P["rel_ind_3m"] = P["ret_3m"] - P["ind_ret_3m"]
    P["ind_n"] = np.log(n.astype(float))
    P["breadth"] = P.groupby("date")["ret_1m"].transform(lambda s: (s > 0).mean())
    for c in ("mkt_ret_20", "mkt_ma120", "mkt_vol_20"):
        P[c] = np.nan
    if idx is not None and len(idx):
        s = idx.astype(float).sort_index()
        r = s.pct_change()
        M = pd.DataFrame({"mkt_ret_20": s / s.shift(20) - 1, "mkt_ma120": s / s.rolling(120).mean() - 1,
                          "mkt_vol_20": r.rolling(20).std() * math.sqrt(252)})
        M = M.reindex(sorted(set(M.index) | set(P["date"]))).ffill()
        for c in M:
            P[c] = P["date"].map(M[c])
    return P


def _lgb_params(lgb_cfg: dict) -> dict:
    prm = {k: v for k, v in lgb_cfg.items() if k != "n_estimators"}
    prm.update(objective="regression", bagging_fraction=prm.pop("subsample"), bagging_freq=prm.pop("subsample_freq"),
               feature_fraction=prm.pop("colsample_bytree"), seed=prm.pop("random_state"), min_data_in_leaf=prm.pop("min_child_samples"),
               deterministic=True, force_row_wise=True, num_threads=1)
    return prm


def walk(panel: pd.DataFrame, sigs: list[str], dates: list[str], idx: pd.Series | None, use_lgbm: bool = True,
         ablate: dict[str, list[str]] | None = None, imp: list | None = None) -> tuple[pd.DataFrame, list[str]]:
    """ablate：額外訓練「LightGBM 拿掉某一組特徵」（V30 消融測試，只做研究）；imp：收集每個測試日 LightGBM 的特徵重要性。"""
    from . import ml_research as ml
    y = f"y{H}"
    A = add_features(panel, idx)
    A[[f"r_{f}" for f in IND_FEATS]] = A.groupby("date")[IND_FEATS].rank(pct=True).values   # 標籤篩選前的當日母體
    P = ml.prepare(A, H)
    if not len(P):
        return pd.DataFrame(), []
    rf = [f"r_{f}" for f in ml.FEATS]
    xf = rf + [f"r_{f}" for f in IND_FEATS] + MKT_FEATS
    P["y_rank"] = P.groupby("date")[y].rank(pct=True)
    pos = {d: i for i, d in enumerate(dates)}
    ex = {s: pos[s] + 1 + H for s in sigs}
    have = sorted(P["date"].unique())
    lgb = None
    if use_lgbm:
        try:
            import lightgbm as lgb  # noqa: F811
        except Exception:
            lgb = None
    prm = _lgb_params(ml.LGB_PARAMS)
    out, tests = [], []
    for t in have:
        tr_d = [s for s in have if ex[s] < pos[t]]   # purge
        if len(tr_d) < ml.MIN_TRAIN:
            continue
        tr, te = P[P["date"].isin(tr_d)], P[P["date"] == t]
        o = te[["date", "sid", "industry", y, "v21"] + [c for c in ("market_cap",) if c in te]].copy()
        o["ridge"] = ml._pred_lin(ml._ridge(tr[rf].fillna(0.5).to_numpy(), tr["y_rank"].to_numpy()), te[rf].fillna(0.5).to_numpy())
        if lgb is not None:
            for name, cols in (("lgbm", rf), ("lgbm_x", xf)):
                m = lgb.train(prm, lgb.Dataset(tr[cols].to_numpy(), tr["y_rank"].to_numpy()), num_boost_round=ml.LGB_PARAMS["n_estimators"])
                o[name] = m.predict(te[cols].to_numpy())
                if name == "lgbm" and imp is not None:
                    imp.append((t, dict(zip(ml.FEATS, m.feature_importance("gain").tolist()))))
            for g, drop in (ablate or {}).items():
                cols = [f"r_{f}" for f in ml.FEATS if f not in drop]
                m = lgb.train(prm, lgb.Dataset(tr[cols].to_numpy(), tr["y_rank"].to_numpy()), num_boost_round=ml.LGB_PARAMS["n_estimators"])
                o[f"abl_{g}"] = m.predict(te[cols].to_numpy())
        out.append(o)
        tests.append(t)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), tests


def _bench(D: dict, dates: list[str], sig: str) -> tuple[float | None, float | None]:
    """(0050 同期報酬, 加權指數同期報酬)。指數只有收盤價：用訊號日收盤到出場前一天收盤，是近似值。"""
    pos = dates.index(sig)
    if pos + 1 + H >= len(dates):
        return None, None
    entry, exit_ = dates[pos + 1], dates[pos + 1 + H]
    b = pb._bench(D, entry, exit_)
    s = D.get("idx_s")
    tx = None
    if s is not None and sig in s.index and dates[pos + H] in s.index:
        tx = float(s[dates[pos + H]] / s[sig] - 1)
    return b, tx


def _mdd(r: list[float]) -> float | None:
    if not r:
        return None
    c = np.cumprod(1 + np.array(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
    return round(float((c / peak - 1).min()) * 100, 2)


def _t(x: np.ndarray) -> float | None:
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def portfolio(preds: pd.DataFrame, m: str, dates_used: list[str], B: dict) -> dict:
    """不重疊的 20 日期間：每期買前 10 名、等權重、換股才收成本（來回 track.COST）。"""
    y = f"y{H}"
    rows, prev, trades = [], set(), 0
    for d in dates_used:
        g = preds[preds["date"] == d]
        if m not in g or g[m].notna().sum() < TOPN:
            continue
        top = g.nlargest(TOPN, m)
        ids = set(top["sid"])
        new = len(ids - prev)
        trades += new
        gross = float(top[y].mean())
        net = gross - track.COST * new / TOPN
        b, tx = B.get(d, (None, None))
        rows.append({"date": d, "net_excess": net, "hit": float((top[y] > 0).mean()), "big_loss": float((top[y] < BIG_LOSS).mean()),
                     "abs": net + b if b is not None else None, "bench": b, "taiex": tx,
                     "vs_taiex": net + b - tx if b is not None and tx is not None else None})
        prev = ids
    if not rows:
        return {"periods": 0}
    x = np.array([r["net_excess"] for r in rows])
    ab = [r["abs"] for r in rows if r["abs"] is not None]
    bb = [r["bench"] for r in rows if r["bench"] is not None]
    vt = [r["vs_taiex"] for r in rows if r["vs_taiex"] is not None]
    tx = [r["taiex"] for r in rows if r["taiex"] is not None]
    t = _t(x)
    return {"periods": len(rows), "trades": trades, "net_excess": round(float(x.mean()) * 100, 2), "t": round(t, 2) if t is not None else None,
            "beat_0050": round(float((x > 0).mean()) * 100, 1), "hit": round(float(np.mean([r["hit"] for r in rows])) * 100, 1),
            "big_loss": round(float(np.mean([r["big_loss"] for r in rows])) * 100, 1),
            "total": round(float(np.prod([1 + a for a in ab]) - 1) * 100, 2) if ab else None,
            "bench_total": round(float(np.prod([1 + a for a in bb]) - 1) * 100, 2) if bb else None,
            "taiex_total": round(float(np.prod([1 + a for a in tx]) - 1) * 100, 2) if tx else None,
            "vs_taiex": round(float(np.mean(vt)) * 100, 2) if vt else None,
            "mdd": _mdd(ab), "series": [round(v * 100, 2) for v in x]}


def ic_stats(preds: pd.DataFrame, m: str) -> dict:
    y = f"y{H}"
    ics = np.array([fr._ic(g[m], g[y]) if g[m].notna().sum() > 30 else np.nan for _, g in preds.groupby("date")], float)
    ok = ics[~np.isnan(ics)]
    if len(ok) < 4:
        return {"ic": None, "t": None, "dates": int(len(ok))}
    return {"ic": round(float(ok.mean()), 4), "t": round(float(ok.mean() / fr._nw_se(ok, 2)), 2), "pos": round(float((ok > 0).mean()) * 100, 1), "dates": int(len(ok))}


def regimes(preds: pd.DataFrame, m: str, idx: pd.Series | None, B: dict) -> dict:
    if idx is None:
        return {}
    s = idx.astype(float).sort_index()
    ma = s.rolling(120).mean()
    y = f"y{H}"
    out = {}
    for d, g in preds.groupby("date"):
        if d not in s.index or pd.isna(ma.get(d)) or m not in g:
            continue
        k = "bull" if s[d] >= ma[d] else "bear"
        o = out.setdefault(k, {"ic": [], "top": []})
        o["ic"].append(fr._ic(g[m], g[y]))
        o["top"].append(float(g.nlargest(TOPN, m)[y].mean()) - track.COST)
    return {k: {"dates": len(v["ic"]), "ic": round(float(np.nanmean(v["ic"])), 4), "top_net_excess": round(float(np.mean(v["top"])) * 100, 2)}
            for k, v in out.items()}


def industries(preds: pd.DataFrame, m: str) -> list[dict]:
    y = f"y{H}"
    rows = []
    for ind, g in preds.groupby("industry"):
        ics = [fr._ic(x[m], x[y]) for _, x in g.groupby("date") if len(x) >= 10]
        ics = [v for v in ics if v is not None and not np.isnan(v)]
        if len(ics) >= 10:
            rows.append({"industry": ind or "（未分類）", "dates": len(ics), "ic": round(float(np.mean(ics)), 4)})
    return sorted(rows, key=lambda r: r["ic"])


def combo_study(panel: pd.DataFrame) -> dict:
    """積極型下跌風險：高動能＋高估值＋高波動＋低流動性，各自與組合在下一個 20 日的表現（全部歷史）。"""
    y = f"y{H}"
    P = panel[panel[y].notna()].copy()
    for k, (col, op, v, _) in COMBO.items():
        if k == "illiquid":
            P[k] = P[col] < v
        else:
            pct = P.groupby("date")[col].rank(pct=True)
            P[k] = (pct >= v) if op == ">=" else (pct <= v)
    P["n_flags"] = P[list(COMBO)].sum(axis=1)
    groups = {"none": P["n_flags"] == 0, **{k: P[k] for k in COMBO}, "hot+volatile": P["hot"] & P["volatile"],
              "three_plus": P["n_flags"] >= 3, "all_four": P["n_flags"] == 4}
    out = {}
    for name, mask in groups.items():
        diffs = []
        for _, g in P.groupby("date"):
            a, b = g[mask.loc[g.index]], g[~mask.loc[g.index]]
            if len(a) >= 3 and len(b) >= 3:
                diffs.append(float(a[y].mean() - b[y].mean()))
        d = np.array(diffs)
        sub = P[mask]
        out[name] = {"stock_dates": int(len(sub)), "dates": len(d), "avg_excess": round(float(sub[y].mean()) * 100, 2) if len(sub) else None,
                     "big_loss_pct": round(float((sub[y] < BIG_LOSS).mean()) * 100, 1) if len(sub) else None,
                     "vs_rest": round(float(d.mean()) * 100, 2) if len(d) else None,
                     "t": round(float(d.mean() / fr._nw_se(d, 2)), 2) if len(d) >= 4 else None}
    out["_base_big_loss_pct"] = round(float((P[y] < BIG_LOSS).mean()) * 100, 1)
    return out


def verdict(cur: dict, champ: dict, hold_c: dict, hold_v: dict, ic_c: dict, ic_v: dict, diff_t: float | None) -> dict:
    checks = [
        {"name": "前 10 名扣成本超額報酬勝過 V21", "need": f"t ≥ {T_NEED}", "value": f"{cur.get('net_excess')}% vs {champ.get('net_excess')}%（差的 t {diff_t if diff_t is not None else '—'}）",
         "ok": bool(diff_t is not None and diff_t >= T_NEED)},
        {"name": "最終保留測試期也勝過 V21", "need": "> V21", "value": f"{hold_c.get('net_excess')}% vs {hold_v.get('net_excess')}%",
         "ok": bool(hold_c.get("net_excess") is not None and hold_v.get("net_excess") is not None and hold_c["net_excess"] > hold_v["net_excess"])},
        {"name": "Rank IC 高於 V21", "need": "> V21", "value": f"{ic_c.get('ic')} vs {ic_v.get('ic')}",
         "ok": bool(ic_c.get("ic") is not None and ic_v.get("ic") is not None and ic_c["ic"] > ic_v["ic"])},
        {"name": "最大回撤不比 V21 差太多", "need": f"差距 ≤ {MDD_GAP:.0f} 個百分點", "value": f"{cur.get('mdd')}% vs {champ.get('mdd')}%",
         "ok": bool(cur.get("mdd") is not None and champ.get("mdd") is not None and cur["mdd"] >= champ["mdd"] - MDD_GAP)},
    ]
    ok = all(c["ok"] for c in checks)
    return {"checks": checks, "passed": ok, "label": "通過全部條件（仍需真實封存比賽與人工確認）" if ok else "未通過：沒有足夠證據比 V21 好，維持 V21"}


def run(panel: pd.DataFrame, sigs: list[str], D: dict, use_lgbm: bool = True, log=print) -> dict:
    from . import robustness as rb
    dates, idx = D["dates"], D.get("idx_s")
    imp: list = []
    preds, tests = walk(panel, sigs, dates, idx, use_lgbm, ablate={g: f for g, (_, f) in rb.GROUPS.items()}, imp=imp)
    if not len(preds):
        return {"ready": False, "reason": "測試日不足"}
    cut = tests[int(len(tests) * (1 - HOLDOUT))]
    B = {d: _bench(D, dates, d) for d in tests}
    models = [m for m in MODELS if m in preds]
    res = {"schema_version": 1, "race_version": RACE_VERSION, "horizon": H, "top": TOPN, "cost_pct": round(track.COST * 100, 3),
           "models": {m: MODELS[m] for m in models}, "t_need": T_NEED, "tests": len(tests), "first_test": tests[0], "last_test": tests[-1],
           "holdout_from": cut, "extra_features": IND_FEATS + MKT_FEATS, "results": {}, "offsets": {}}
    hold = preds[preds["date"] >= cut]
    for off in (0, 1):   # 每 10 個交易日一個訊號、持有 20 日：兩組不重疊的期間，第 0 組是主要結果，第 1 組看穩不穩
        use = tests[off::2]
        res["offsets"][str(off)] = {m: portfolio(preds, m, use, B) for m in models}
    main = res["offsets"]["0"]
    hold_use = [d for d in tests[0::2] if d >= cut]
    for m in models:
        res["results"][m] = {"name": MODELS[m], "ic": ic_stats(preds, m), "ic_holdout": ic_stats(hold, m), "portfolio": main[m],
                             "holdout": portfolio(preds, m, hold_use, B), "robust_offset1": {k: res["offsets"]["1"][m].get(k) for k in ("net_excess", "t", "total", "mdd")},
                             "regimes": regimes(preds, m, idx, B)}
    v = res["results"]["v21"]
    for m in models:
        if m == "v21":
            continue
        a, b = main[m].get("series") or [], v["portfolio"].get("series") or []
        n = min(len(a), len(b))
        d = np.array(a[:n]) - np.array(b[:n])
        t = _t(d)
        r = res["results"][m]
        r["vs_v21"] = {"diff": round(float(d.mean()), 2) if n else None, "t": round(t, 2) if t is not None else None}
        r["verdict"] = verdict(r["portfolio"], v["portfolio"], r["holdout"], v["holdout"], r["ic"], v["ic"], t)
    best = max((m for m in models if m != "v21"), key=lambda m: res["results"][m]["portfolio"].get("net_excess") or -1e9, default=None)
    res["industries"] = {m: industries(preds, m) for m in ("v21", best) if m}
    res["combo"] = combo_study(panel)
    res["combo_rules"] = {k: v[3] for k, v in COMBO.items()}
    passed = [m for m in models if res["results"][m].get("verdict", {}).get("passed")]
    res["summary"] = {"passed": passed, "best_by_net_excess": best,
                      "label": ("、".join(MODELS[m] for m in passed) + " 通過研究驗收，接著要在真實封存預測中再比一次") if passed
                      else "沒有任何挑戰者在樣本外有足夠證據勝過 V21：正式模型維持 V21"}
    res["effective_samples"] = {"test_dates": len(tests), "independent_periods": main["v21"].get("periods"),
                                "note": "每 10 個交易日一個訊號、持有 20 日，相鄰訊號重疊一半；不重疊的期間數才是有效樣本"}
    try:  # V30 LightGBM 可信度檢查（診斷，不影響比賽結果）
        res["robust"] = rb.run(preds, tests, cut, idx, imp, portfolio, ic_stats)
    except Exception as e:  # noqa: BLE001
        res["robust"] = {"error": f"可信度檢查失敗：{e}"}
    log(f"模型比賽：{res['summary']['label']}")
    return res
