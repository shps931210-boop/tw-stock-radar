"""V22 預測模型研究（影子模式，不影響正式排行榜）：現有資料夠不夠訓練 5／20／60 日超額報酬預測模型？

資料集（V22.1）
- 訊號日：從 pit_backtest.auto_start（股價滿一年且季報大致齊全）起每 10 個交易日一次。每個訊號日用 pit_backtest.scored_rows 重算「當時看得到的資料」：
  股價只到訊號日、季報在法定期限後才可用、月營收次月 10 日後才可用（和歷史回測同一套程式）。
- 股票母體：當天 V21 均衡型可以排名的股票（通過基本門檻、五因子齊全），新舊模型在同一批股票上比較。
- 特徵：V21 五個因子分數＋事先選定的 27 個原始指標（動能、波動、估值、成長、獲利、規模），
  每個訊號日轉成全市場百分位；缺值不補零、不用未來資料：線性模型用當天中間值（0.5），LightGBM 直接處理缺值。
- 標籤：訊號日隔天開盤進場，持有 5／20／60 個交易日後的開盤出場（還原除權息），減去同期間 0050 的報酬；
  進場日沒成交、漲停鎖死、成交太少的股票買不到，不列入（另外計數）。

驗證（V22.2）
- 依時間滾動：每個測試日只用「標籤在測試訊號日之前就已經確定」的歷史訓練（purge），至少 8 個訓練日。
- 模型：V21 均衡型固定權重（基準，不訓練）、Ridge 線性迴歸、Logistic 迴歸（跑贏 0050 的機率）、LightGBM。
  所有參數事先固定，不依測試結果調整；最後 30% 的測試日另外列為「最終保留測試期」。
- 指標：Rank IC（Newey-West t 值，處理重疊期間）、前 20 名平均超額報酬、方向命中率、Brier 分數與校準。
- 「資料夠不夠」：用實際 IC 波動算出這段資料「最小可偵測的 IC」，和合理可期待的 IC（0.03～0.05）比較。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import factor_research as fr
from . import pit_backtest as pb

METHOD = 2  # 2：特徵百分位改用標籤篩選前的當日母體
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "ml.json"
STEP = 10                     # 每 10 個交易日一個訊號日
HORIZONS = (5, 20, 60)
MAIN = 20                     # 主要研究目標：20 個交易日超額報酬
MIN_TRAIN = 8                 # 至少 8 個訓練日才預測
HOLDOUT = 0.3                 # 最後 30% 的測試日是最終保留測試期
TOP = 20
SCORES = ["score_quality", "score_growth", "score_value", "score_momentum", "score_trend"]
RAW = ["mom_12_1", "ret_1m", "ret_3m", "rs_rank", "vol_1y", "downside_vol", "beta", "mdd_1y", "dist_ma60", "ma60_slope_20",
       "rsi", "vol_ratio", "rev_yoy", "rev_yoy_3m", "eps_q_yoy", "op_q_yoy", "roe", "gross_margin", "op_margin", "debt_ratio",
       "earnings_yield", "book_to_price", "dy", "fcf_yield", "fscore", "market_cap", "avg_value_20"]
FEATS = SCORES + RAW
RIDGE_ALPHA = 10.0
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=7, verbose=-1)
NAMES = {"v21": "V21 均衡型（固定權重）", "ridge": "Ridge 線性迴歸", "logit": "Logistic 迴歸", "lgbm": "LightGBM", "ens": "V21＋LightGBM 平均"}

_D, _CFG = None, None


def signal_dates(dates: list[str], start: str = pb.START, step: int = STEP) -> list[str]:
    i0 = next(i for i, d in enumerate(dates) if d >= start)
    return [dates[i] for i in range(i0, len(dates) - 1, step)]


def _cross(D: dict, sig: str, cfg: dict) -> pd.DataFrame:
    """一個訊號日的特徵與標籤（每列一檔可排名股票）。"""
    dates = D["dates"]
    pos = dates.index(sig)
    entry = dates[pos + 1]
    rows = pb.scored_rows(D, sig, cfg)
    out = []
    for r in rows:
        bal = r["ai"]["modes"]["balanced"]
        if not bal.get("rank"):
            continue
        x = {"date": sig, "sid": r["stock_id"], "industry": r.get("industry") or "", "v21": bal.get("score")}
        for f in FEATS:
            v = r.get(f)
            x[f] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else np.nan
        for f in ("market_cap", "avg_value_20"):
            x[f] = math.log(x[f]) if x[f] and x[f] > 0 else np.nan
        for h in HORIZONS:
            e = pos + 1 + h
            if e >= len(dates):
                x[f"y{h}"], x[f"st{h}"] = np.nan, "no_exit"
                continue
            ret, st = pb.hold_return(D, r["stock_id"], entry, dates[e])
            b = pb._bench(D, entry, dates[e])
            x[f"y{h}"] = ret - b if ret is not None and b is not None else np.nan
            x[f"st{h}"] = st if b is not None else "no_bench"
        out.append(x)
    return pd.DataFrame(out)


def _work(sig: str) -> pd.DataFrame:
    return _cross(_D, sig, _CFG)


def build_panel(D: dict, cfg: dict, sigs: list[str], log=print, workers: int | None = None) -> pd.DataFrame:
    global _D, _CFG
    _D, _CFG = D, cfg
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    parts = []
    if workers > 1 and sys.platform != "win32":
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            for i, df in enumerate(ex.map(_work, sigs)):
                parts.append(df)
                log(f"{sigs[i]}：{len(df)} 檔")
    else:
        for s in sigs:
            parts.append(_work(s))
            log(f"{s}：{len(parts[-1])} 檔")
    return pd.concat(parts, ignore_index=True)


# ---------------- 模型 ----------------
def _ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("date")[cols].rank(pct=True)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    I = np.eye(Xc.shape[1]); I[0, 0] = 0
    return np.linalg.solve(Xc.T @ Xc + alpha * I, Xc.T @ y)


def _logit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA, iters: int = 30) -> np.ndarray:
    """L2 Logistic 迴歸（牛頓法）。"""
    Xc = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xc.shape[1])
    I = np.eye(len(w)); I[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xc @ w, -30, 30)))
        g = Xc.T @ (p - y) + alpha * I @ w
        H = (Xc * (p * (1 - p))[:, None]).T @ Xc + alpha * I
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def _pred_lin(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def prepare(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """先用訊號日當天的完整可排名母體算特徵百分位，再接上未來標籤、只留有標籤的列。
    不能先丟掉「之後停牌、買不到、沒有出場價」的股票再排名，否則當天的母體就被未來資訊改變了（選樣偏誤）。"""
    P = panel.copy()
    P[[f"r_{f}" for f in FEATS]] = _ranks(P, FEATS).values
    return P[P[f"y{h}"].notna()].copy()


def walk_forward(panel: pd.DataFrame, h: int, sigs: list[str], dates: list[str], use_lgbm: bool = True) -> dict:
    """依時間滾動預測；回傳每個測試日每檔股票的各模型預測。"""
    y = f"y{h}"
    P = prepare(panel, h)
    if not len(P):
        return {"preds": pd.DataFrame(), "tests": []}
    P["y_rank"] = P.groupby("date")[y].rank(pct=True)
    P["y_win"] = (P[y] > 0).astype(float)
    pos = {d: i for i, d in enumerate(dates)}
    exit_idx = {s: pos[s] + 1 + h for s in sigs}
    have = sorted(P["date"].unique())
    rf = [f"r_{f}" for f in FEATS]
    lgb = None
    if use_lgbm:
        try:
            import lightgbm as lgb  # noqa: F811
        except Exception:
            lgb = None
    preds, tests = [], []
    for t in have:
        train_dates = [s for s in have if exit_idx[s] < pos[t]]  # purge：標籤在測試訊號日之前就已確定
        if len(train_dates) < MIN_TRAIN:
            continue
        tr, te = P[P["date"].isin(train_dates)], P[P["date"] == t]
        Xtr, Xte = tr[rf].fillna(0.5).to_numpy(), te[rf].fillna(0.5).to_numpy()
        out = te[["date", "sid", "industry", y, "v21"]].copy()
        out["ridge"] = _pred_lin(_ridge(Xtr, tr["y_rank"].to_numpy()), Xte)
        out["logit"] = 1 / (1 + np.exp(-np.clip(_pred_lin(_logit(Xtr, tr["y_win"].to_numpy()), Xte), -30, 30)))
        out["base_rate"] = tr["y_win"].mean()
        if lgb is not None:
            prm = {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"}
            prm.update(objective="regression", bagging_fraction=prm.pop("subsample"), bagging_freq=prm.pop("subsample_freq"),
                       feature_fraction=prm.pop("colsample_bytree"), seed=prm.pop("random_state"), min_data_in_leaf=prm.pop("min_child_samples"))
            m = lgb.train(prm, lgb.Dataset(tr[rf].to_numpy(), tr["y_rank"].to_numpy()), num_boost_round=LGB_PARAMS["n_estimators"])
            out["lgbm"] = m.predict(te[rf].to_numpy())
            out["ens"] = ensemble(out["v21"], out["lgbm"])
        preds.append(out)
        tests.append(t)
    return {"preds": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(), "tests": tests}


def ensemble(v21, lgbm) -> np.ndarray:
    """V25.1 組合模型：同一天裡 V21 分數與 LightGBM 預測的百分位平均（事先固定各半，不依結果調整）。
    兩個模型的錯誤不完全相同，平均通常比單一模型穩定；是否真的比較準，交給樣本外驗證與實盤比賽。"""
    a, b = pd.Series(np.asarray(v21, float)).rank(pct=True), pd.Series(np.asarray(lgbm, float)).rank(pct=True)
    return ((a.fillna(0.5) + b.fillna(0.5)) / 2).to_numpy()


def _nw_t(x: np.ndarray, lag: int) -> tuple[float | None, float | None]:
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return None, None
    se = fr._nw_se(x, lag)
    return (float(x.mean() / se) if se else None), se


def evaluate(preds: pd.DataFrame, h: int, models: list[str]) -> dict:
    y = f"y{h}"
    lag = max(1, math.ceil(h / STEP))  # 相鄰訊號日的持有期間重疊幾期
    res = {}
    by = list(preds.groupby("date"))
    for m in models:
        if m not in preds:
            continue
        ics = np.array([fr._ic(g[m], g[y]) if g[m].notna().sum() > 30 else np.nan for _, g in by], float)
        top = np.array([g.nlargest(TOP, m)[y].mean() for _, g in by], float)
        allm = np.array([g[y].mean() for _, g in by], float)
        t, se = _nw_t(ics, lag)
        ok = ics[~np.isnan(ics)]
        res[m] = {"ic": round(float(np.nanmean(ics)), 4) if len(ok) else None, "ic_t": round(t, 2) if t is not None else None,
                  "ic_pos": round(float((ok > 0).mean()) * 100, 1) if len(ok) else None, "ic_sd": round(float(ok.std(ddof=1)), 4) if len(ok) > 1 else None,
                  "top_excess": round(float(np.nanmean(top)) * 100, 2), "top_vs_all": round(float(np.nanmean(top - allm)) * 100, 2),
                  "top_hit": round(float(np.nanmean(top > 0)) * 100, 1), "dates": int(len(ok))}
    if "logit" in preds:
        p, w = preds["logit"].to_numpy(), (preds[y] > 0).to_numpy(float)
        b0 = preds["base_rate"].to_numpy()
        brier, brier0 = float(np.mean((p - w) ** 2)), float(np.mean((b0 - w) ** 2))
        bins = pd.cut(preds["logit"], [0, .3, .4, .5, .6, .7, 1], include_lowest=True)
        cal = preds.assign(w=w).groupby(bins, observed=True).agg(n=("w", "size"), pred=("logit", "mean"), actual=("w", "mean"))
        res["prob"] = {"hit": round(float(np.mean((p > 0.5) == (w > 0.5))) * 100, 1), "base_hit": round(float(max(w.mean(), 1 - w.mean())) * 100, 1),
                       "win_rate": round(float(w.mean()) * 100, 1), "brier": round(brier, 4), "brier_base": round(brier0, 4),
                       "skill": round(1 - brier / brier0, 4) if brier0 else None,
                       "calibration": [{"bin": str(k), "n": int(r.n), "pred": round(float(r.pred) * 100, 1), "actual": round(float(r.actual) * 100, 1)}
                                       for k, r in cal.iterrows()]}
    return res


def _diff(preds: pd.DataFrame, h: int, a: str, b: str) -> dict | None:
    if a not in preds or b not in preds:
        return None
    y = f"y{h}"
    d = np.array([fr._ic(g[a], g[y]) - fr._ic(g[b], g[y]) for _, g in preds.groupby("date")], float)
    t, _ = _nw_t(d, max(1, math.ceil(h / STEP)))
    return {"diff": round(float(np.nanmean(d)), 4), "t": round(t, 2) if t is not None else None}


def sufficiency(res_main: dict, n_dates: int, h: int) -> dict:
    """這段資料能不能證明一個預測模型有效：最小可偵測 IC（雙尾 5%、檢定力 80%）。"""
    sd = res_main.get("ic_sd")
    span = n_dates * STEP                      # 測試期涵蓋的交易日
    n_eff = max(1.0, span / h)                 # 不重疊的持有期數
    mde = round(2.8 * sd / math.sqrt(n_eff), 4) if sd else None
    need = math.ceil((2.8 * sd / 0.03) ** 2) if sd else None   # 要偵測 IC 0.03 需要幾個獨立期間
    return {"horizon": h, "test_trading_days": span, "independent_periods": round(n_eff, 1), "ic_sd": sd, "min_detectable_ic": mde,
            "periods_needed_for_0.03": need, "years_needed_for_0.03": round(need * h / 245, 1) if need else None,
            "enough": bool(mde is not None and mde <= 0.03)}


def run(dbname: str = "stocks.db", start: str | None = None, log=print, use_lgbm: bool = True) -> dict:
    t0 = time.time()
    cfg = config.load()
    D = pb.load(dbname)
    start = start or pb.auto_start(D)
    sigs = signal_dates(D["dates"], start)
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(sigs)} 個訊號日（{sigs[0]}～{sigs[-1]}）")
    cache = config.DATA_DIR / "ml_panel.pkl"  # 只在本機重跑時沿用（Actions 每次重建）
    if os.environ.get("ML_PANEL_CACHE") and cache.exists():
        panel = pd.read_pickle(cache)
    else:
        panel = build_panel(D, cfg, sigs, log)
        if os.environ.get("ML_PANEL_CACHE"):
            panel.to_pickle(cache)
    log(f"資料集 {len(panel)} 列（股票 × 訊號日），{time.time() - t0:.0f} 秒")
    models = ["v21", "ridge", "logit", "lgbm", "ens"]
    H = {}
    errors = None
    for h in HORIZONS:
        wf = walk_forward(panel, h, sigs, D["dates"], use_lgbm)
        P, tests = wf["preds"], wf["tests"]
        if not len(P):
            H[str(h)] = {"tests": 0}
            continue
        cut = tests[int(len(tests) * (1 - HOLDOUT))] if len(tests) >= 4 else tests[-1]
        main = evaluate(P, h, models)
        if h == MAIN:
            try:
                from . import learning
                errors = learning.error_analysis(P, panel, h, D.get("idx_s"))
            except Exception as e:
                errors = {"error": str(e)}
        H[str(h)] = {"tests": len(tests), "first_test": tests[0], "last_test": tests[-1], "holdout_from": cut,
                     "all": main, "rolling": evaluate(P[P["date"] < cut], h, models), "holdout": evaluate(P[P["date"] >= cut], h, models),
                     "vs_v21": {m: _diff(P, h, m, "v21") for m in ("ridge", "logit", "lgbm", "ens")},
                     "sufficiency": sufficiency(main.get("v21") or {}, len(tests), h),
                     "rows": int(len(P))}
        log(f"{h} 日：{len(tests)} 個測試日，{time.time() - t0:.0f} 秒")
    lab = {str(h): {k: int(v) for k, v in panel[f"st{h}"].value_counts().items()} for h in HORIZONS}
    miss = {f: round(float(panel[f].isna().mean()) * 100, 1) for f in FEATS}
    S = H.get(str(MAIN), {}).get("sufficiency") or {}
    verdict = ("不足" if not S.get("enough") else "勉強足夠")
    res = {"schema_version": 1, "method": METHOD, "generated_at": config.now().isoformat(timespec="seconds"), "shadow": True,
            "start": sigs[0], "end": sigs[-1], "step": STEP, "signals": len(sigs), "rows": int(len(panel)), "stocks_avg": round(len(panel) / len(sigs)),
            "features": FEATS, "feature_missing_pct": miss, "label_status": lab, "models": NAMES, "params": {"ridge_alpha": RIDGE_ALPHA, "lgbm": LGB_PARAMS},
            "lgbm": "lgbm" in (H.get(str(MAIN), {}).get("all") or {}), "horizons": H, "main": MAIN, "verdict": verdict,
            "seconds": round(time.time() - t0), "errors": errors}
    try:  # V28 20 日模型比賽（事先固定的驗收規則，app/model_race.py）
        from . import model_race
        res["race"] = model_race.run(panel, sigs, D, use_lgbm, log)
    except Exception as e:
        res["race"] = {"ready": False, "reason": f"模型比賽失敗：{e}"}
        log(res["race"]["reason"])
    try:  # V25 持續學習：用全部已到期的資料訓練新版本，暫存檢查通過才上線（app/learning.py）
        from . import learning
        res["registered"] = learning.train_and_register(panel, res, D["dates"], use_lgbm, log)["version"]
    except Exception as e:
        log(f"新版本訓練或登記失敗，沿用上一版：{e}")
    return res


def main(argv: list[str]) -> int:
    res = run(log=lambda s: print(s, flush=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    race = res.pop("race", None)
    if race is not None:
        (OUT.parent / "race.json").write_text(json.dumps({**race, "generated_at": res["generated_at"], "data_start": res["start"], "data_end": res["end"]},
                                                          ensure_ascii=False, indent=1, default=str), "utf-8")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    m = res["horizons"].get(str(MAIN), {})
    print(json.dumps({"verdict": res["verdict"], "sufficiency": m.get("sufficiency"), "all": m.get("all"), "vs_v21": m.get("vs_v21")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
