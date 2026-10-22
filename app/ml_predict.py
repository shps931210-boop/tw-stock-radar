"""每日影子預測（V23.2）與預測成績單（V23.5）。

流程：
1. 每週「AI 歷史回測」跑完 ml_research 後，用全部已確定標籤的資料訓練一次模型，存到 history/models/
   （Ridge、Logistic 係數是 JSON，LightGBM 是文字檔），附上訓練資料截止日與模型雜湊。
2. 每天完整更新後（app.ml_predict），用「和歷史研究同一套程式」（pit_backtest.scored_rows）算當天特徵，
   套用最新一版模型，結果封存到 history/predictions/日期.json；同一天只寫一次，之後不覆寫。
3. 到期的預測（5／20／60 個交易日）自動用實際行情對帳：隔天開盤進場、第 h 個交易日後開盤出場、減 0050，
   和歷史研究的標籤定義完全相同。統計 Rank IC、前 20 名超額、方向命中率與 Brier 分數 → history/predictions/scorecard.json（網站 data/predictions/scorecard.json）。

這是影子模式：不改正式排行榜。Logistic 的「跑贏機率」還沒通過校準（歷史研究的 Brier 技巧分數 ≈ 0），
網站只把它當實驗數字，不當正式機率；個股頁顯示的是模型給的同日百分位排名。
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, db
from . import ml_research as ml
from . import pit_backtest as pb
from . import track

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "history" / "models"
PRED_DIR = ROOT / "history" / "predictions"
MODELS = ("v21", "ridge", "lgbm", "ens", "logit")


# ---------------- 訓練並存檔（每週） ----------------
def export_models(panel: pd.DataFrame, use_lgbm: bool = True, out: Path = MODEL_DIR) -> dict:
    """用全部已確定標籤的列訓練最終模型（不需要 purge：所有標籤都已經到期）。"""
    out.mkdir(parents=True, exist_ok=True)
    lgb = None
    if use_lgbm:
        try:
            import lightgbm as lgb
        except Exception:
            lgb = None
    rf = [f"r_{f}" for f in ml.FEATS]
    meta = {"method": ml.METHOD, "feats": ml.FEATS, "params": {"ridge_alpha": ml.RIDGE_ALPHA, "lgbm": ml.LGB_PARAMS},
            "trained_at": config.now().isoformat(timespec="seconds"), "horizons": {}}
    for h in ml.HORIZONS:
        P = ml.prepare(panel, h)
        if len(P) < 500:
            continue
        P["y_rank"] = P.groupby("date")[f"y{h}"].rank(pct=True)
        P["y_win"] = (P[f"y{h}"] > 0).astype(float)
        X = P[rf].fillna(0.5).to_numpy()
        H = {"rows": int(len(P)), "dates": int(P["date"].nunique()), "through": str(P["date"].max()),
             "ridge": ml._ridge(X, P["y_rank"].to_numpy()).round(6).tolist(),
             "logit": ml._logit(X, P["y_win"].to_numpy()).round(6).tolist(), "base_rate": round(float(P["y_win"].mean()), 4)}
        if lgb is not None:
            prm = {k: v for k, v in ml.LGB_PARAMS.items() if k != "n_estimators"}
            prm.update(objective="regression", bagging_fraction=prm.pop("subsample"), bagging_freq=prm.pop("subsample_freq"),
                       feature_fraction=prm.pop("colsample_bytree"), seed=prm.pop("random_state"), min_data_in_leaf=prm.pop("min_child_samples"),
                       deterministic=True, force_row_wise=True, num_threads=1)  # 同樣的資料一定訓練出同一個模型（雜湊才比得出「有沒有變」）
            m = lgb.train(prm, lgb.Dataset(P[rf].to_numpy(), P["y_rank"].to_numpy()), num_boost_round=ml.LGB_PARAMS["n_estimators"])
            (out / f"lgbm_{h}.txt").write_text(m.model_to_string(), "utf-8")
            H["lgbm"] = f"lgbm_{h}.txt"
            g = m.feature_importance("gain")
            H["importance"] = {f: round(float(v) / float(g.sum()) * 100, 2) for f, v in zip(ml.FEATS, g)} if g.sum() > 0 else {}
        meta["horizons"][str(h)] = H
    blob = json.dumps({k: v for k, v in meta.items() if k != "trained_at"}, sort_keys=True)
    for h in meta["horizons"].values():
        if h.get("lgbm"):
            blob += (out / h["lgbm"]).read_text("utf-8")
    meta["model_hash"] = hashlib.sha1(blob.encode()).hexdigest()[:10]
    (out / "ml.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return meta


def load_models(src: Path = MODEL_DIR) -> dict | None:
    p = src / "ml.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text("utf-8"))
    if meta.get("feats") != ml.FEATS:
        return None  # 特徵定義改過，舊模型不能用
    meta["_lgbm"] = {}
    try:
        import lightgbm as lgb
        for h, H in meta["horizons"].items():
            if H.get("lgbm") and (src / H["lgbm"]).exists():
                meta["_lgbm"][h] = lgb.Booster(model_str=(src / H["lgbm"]).read_text("utf-8"))
    except Exception:
        pass
    return meta


# ---------------- 每日預測 ----------------
def cross_section(D: dict, sig: str, cfg: dict) -> pd.DataFrame:
    """訊號日的特徵（和 ml_research._cross 同一套欄位，但不算標籤）。"""
    rows = pb.scored_rows(D, sig, cfg)
    out = []
    for r in rows:
        bal = r["ai"]["modes"]["balanced"]
        if not bal.get("rank"):
            continue
        x = {"date": sig, "sid": r["stock_id"], "name": r.get("name"), "industry": r.get("industry") or "", "v21": bal.get("score")}
        for f in ml.FEATS:
            v = r.get(f)
            x[f] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else np.nan
        for f in ("market_cap", "avg_value_20"):
            x[f] = math.log(x[f]) if x[f] and x[f] > 0 else np.nan
        out.append(x)
    return pd.DataFrame(out)


def predict(X: pd.DataFrame, M: dict) -> dict:
    """套用存檔的模型。輸出每個期間、每個模型的分數與同日百分位（1 = 當天最看好）。"""
    rf = [f"r_{f}" for f in ml.FEATS]
    X = X.copy()
    X[rf] = X[ml.FEATS].rank(pct=True).values
    Z = X[rf].fillna(0.5).to_numpy()
    out = {"v21": X["v21"].astype(float).to_numpy()}
    res = {}
    for h, H in M["horizons"].items():
        p = {"v21": out["v21"], "ridge": ml._pred_lin(np.array(H["ridge"]), Z)}
        lg = ml._pred_lin(np.array(H["logit"]), Z)
        p["logit"] = 1 / (1 + np.exp(-np.clip(lg, -30, 30)))
        if h in M.get("_lgbm", {}):
            p["lgbm"] = M["_lgbm"][h].predict(X[rf].to_numpy())
            p["ens"] = ml.ensemble(p["v21"], p["lgbm"])
        res[h] = {k: pd.Series(v, index=X["sid"]) for k, v in p.items()}
    return res


COMPLETE = 0.95   # 當天上市、上櫃各自的股價筆數至少要有前一個交易日的 95%


def completeness(con, day: str) -> dict:
    """當天的正式行情是否已經齊全（上市、上櫃都要有）。不齊全就不封存，避免用一半的舊價格產生「今日」預測。"""
    prev = con.execute("SELECT MAX(date) FROM prices WHERE date < ?", (day,)).fetchone()[0]
    cnt = lambda d: dict(con.execute("SELECT s.market, COUNT(*) FROM prices p JOIN stocks s USING(stock_id) WHERE p.date = ? GROUP BY s.market", (d,)).fetchall())
    now, before = cnt(day), cnt(prev) if prev else {}
    out = {"date": day, "counts": now, "prev_date": prev, "prev_counts": before}
    for m in ("twse", "tpex"):
        if before.get(m) and now.get(m, 0) < before[m] * COMPLETE:
            return {**out, "ok": False, "why": f"{'上市' if m == 'twse' else '上櫃'}只有 {now.get(m, 0)} 檔（前一交易日 {before[m]} 檔）"}
    return {**out, "ok": True}


def seal(dbname: str = "stocks.db", models: Path = MODEL_DIR, out: Path = PRED_DIR) -> dict | None:
    """產生並封存當天的影子預測。同一天已經封存過就不重寫（回傳 None）。"""
    M = load_models(models)
    if M is not None and models == MODEL_DIR:
        from . import learning
        bad = learning.validate(models)
        if bad:  # 上線版本壞掉：自動退回上一個檢查過的版本
            M = load_models(models) if learning.rollback("；".join(bad)) else None
    if not M:
        print("沒有可用的已訓練模型（等每週「AI 歷史回測」產生 history/models/ml.json）")
        return None
    with db.connect(dbname) as con:
        official = db.get_meta(con, "official_latest_twse")
        last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        data_version = {"prices": last, "revenue_ym": con.execute("SELECT MAX(ym) FROM revenue").fetchone()[0],
                        "fin_period": con.execute("SELECT MAX(period) FROM financials").fetchone()[0]}
        comp = completeness(con, last) if last else {"ok": False, "why": "資料庫沒有股價"}
    if not comp["ok"]:  # 快速更新可能只拿到上市、櫃買還沒公布：等資料齊了再封存（14:40、16:30、21:00 會再試）
        print(f"{last} 的行情還不完整，暫不封存：{comp['why']}")
        return None
    if last and (out / f"{last}.json").exists():
        print(f"{last} 已經封存過，不覆寫")
        return None
    D = pb.load(dbname)
    sig = D["dates"][-1]
    if official and official > sig:
        print(f"股價只到 {sig}，官方已公布 {official}：資料過期，不產生預測")
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{sig}.json"
    if path.exists():
        print(f"{sig} 已經封存過，不覆寫")
        return None
    X = cross_section(D, sig, config.load())
    if len(X) < 100:
        print(f"可排名股票只有 {len(X)} 檔，不產生預測")
        return None
    try:  # V31：封存當天的原始特徵（之後算特徵漂移與特徵有效性）；失敗不影響預測封存
        from . import daily_learning
        daily_learning.save_features(X, sig)
    except Exception as e:  # noqa: BLE001
        print(f"特徵快照沒有存成功：{e}")
    P = predict(X, M)
    stocks = []
    for i, sid in enumerate(X["sid"]):
        row = {"id": sid}
        for h, p in P.items():
            row[h] = {k: (None if not np.isfinite(v.iloc[i]) else round(float(v.iloc[i]), 4)) for k, v in p.items()}
            for k, v in p.items():
                if k != "logit":
                    row[h][f"{k}_pct"] = round(float(v.rank(pct=True).iloc[i]), 3)
        stocks.append(row)
    rec = {"schema_version": 1, "signal_date": sig, "generated_at": config.now().isoformat(timespec="seconds"),
           "feature_cutoff": f"{sig} 收盤；季報依法定期限、月營收依次月 10 日後才使用", "model_hash": M["model_hash"],
           "model_version": M.get("version") or M["model_hash"], "model_method": M["method"], "trained_through": {h: H["through"] for h, H in M["horizons"].items()},
           "data_version": data_version, "data_complete": comp, "sealed_in": os.environ.get("MODE") or "manual",
           "time_check": {"trained_before_signal": all((H["through"] or "") < sig for H in M["horizons"].values())},
           "status": "experimental", "universe": "當天均衡型可排名的股票", "n": len(stocks), "stocks": stocks}
    path.write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"已封存 {sig} 的影子預測：{len(stocks)} 檔，模型 {M['model_hash']}")
    return rec


# ---------------- 成績單 ----------------
def scorecard(dbname: str = "stocks.db", src: Path = PRED_DIR, D: dict | None = None) -> dict:
    files = sorted(src.glob("????-??-??.json")) if src.exists() else []
    if not files:
        return {"days": 0, "horizons": {}}
    D = D or pb.load(dbname)
    dates = D["dates"]
    pos = {d: i for i, d in enumerate(dates)}
    acc = {h: {m: {"ic": [], "top": [], "hit": [], "brier": [], "brier_base": []} for m in MODELS} for h in map(str, ml.HORIZONS)}
    counts = {h: 0 for h in acc}
    daily = []
    on_time = 0
    for f in files:
        rec = json.loads(f.read_text("utf-8"))
        sig = rec["signal_date"]
        if sig in pos and pos[sig] + 1 < len(dates):
            entry = dates[pos[sig] + 1]
        else:  # 進場日還沒到：用休市日曆推下一個交易日，「是否在開盤前封存」現在就判斷得出來
            from datetime import date as _d
            from . import premarket
            entry = premarket.next_trading_day(_d.fromisoformat(sig), premarket.holidays(refresh=False)).isoformat()
        on_time += bool(track.on_time(rec, entry))
        if sig not in pos or pos[sig] + 1 >= len(dates):
            continue
        for h in acc:
            e = pos[sig] + 1 + int(h)
            if e >= len(dates):
                continue  # 還沒到期
            b = pb._bench(D, entry, dates[e])
            if b is None:
                continue
            y = {}
            for s in rec["stocks"]:
                r, _ = pb.hold_return(D, s["id"], entry, dates[e])
                if r is not None:
                    y[s["id"]] = r - b
            if len(y) < 30:
                continue
            counts[h] += 1
            Y = pd.Series(y)
            day = {"sig": sig, "h": h, "ver": rec.get("model_version") or rec.get("model_hash"), "ic": {}, "top": {}}
            daily.append(day)
            for m in MODELS:
                p = pd.Series({s["id"]: (s.get(h) or {}).get(m) for s in rec["stocks"]}, dtype=float).reindex(Y.index)
                if p.notna().sum() < 30:
                    continue
                A = acc[h][m]
                A["ic"].append(float(p.rank().corr(Y.rank())))
                A["top"].append(float(Y[p.nlargest(ml.TOP).index].mean()))
                day["ic"][m], day["top"][m] = round(A["ic"][-1], 4), round(A["top"][-1], 5)
                if m == "logit":
                    w = (Y > 0).astype(float)
                    A["hit"].append(float(((p > 0.5) == (w > 0.5)).mean()))
                    A["brier"].append(float(((p - w) ** 2).mean()))
                    A["brier_base"].append(float(((w.mean() - w) ** 2).mean()))
    out = {}
    for h, mm in acc.items():
        out[h] = {"matured": counts[h]}
        for m, A in mm.items():
            if not A["ic"]:
                continue
            out[h][m] = {"ic": round(float(np.mean(A["ic"])), 3), "ic_pos": round(float(np.mean([x > 0 for x in A["ic"]])) * 100, 1),
                         "top_excess": round(float(np.mean(A["top"])) * 100, 2), "n": len(A["ic"])}
            if A["brier"]:
                out[h][m].update(hit=round(float(np.mean(A["hit"])) * 100, 1), brier=round(float(np.mean(A["brier"])), 4),
                                 brier_base=round(float(np.mean(A["brier_base"])), 4))
    return {"generated_at": config.now().isoformat(timespec="seconds"), "days": len(files), "on_time": on_time,
            "first": files[0].stem, "last": files[-1].stem, "horizons": out, "daily": daily}


def latest(src: Path = PRED_DIR) -> dict | None:
    """網站用：最新一天的預測（每檔只留 20 日的百分位，個股頁顯示）。"""
    files = sorted(src.glob("????-??-??.json")) if src.exists() else []
    if not files:
        return None
    rec = json.loads(files[-1].read_text("utf-8"))
    keep = {k: rec[k] for k in ("signal_date", "generated_at", "feature_cutoff", "model_hash", "trained_through", "status", "universe", "n")}
    keep["stocks"] = {s["id"]: {h: {k: v for k, v in x.items() if k.endswith("_pct")} for h, x in s.items() if h != "id"} for s in rec["stocks"]}
    return keep


def main(argv: list[str]) -> int:
    rec = seal()
    if rec is None and (PRED_DIR / "scorecard.json").exists() and "--score" not in argv:
        return 0  # 沒有新預測：成績單沿用上一次（每天第一次封存時才重算，省時間）
    sc = scorecard()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / "scorecard.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({h: v.get("matured") for h, v in sc.get("horizons", {}).items()}, ensure_ascii=False))
    from . import learning
    st = learning.status(sc)  # 網站用的學習報告在 build_site 每次發布時重算
    print(f"持續學習：候選 {st['active'] and st['active']['version']}，比賽 {st['gate']['label']}，漂移 {st['drift']['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
