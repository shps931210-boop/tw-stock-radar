"""V31 每日自主學習（AI Continuous Learning Engine 的每日部分）→ history/learning/YYYY-MM-DD.json。

每天 21:00 完整更新、封存預測之後執行。每天都會做的事：
1. 資料檢查：今天的上市、上櫃行情是否齊全，今天有沒有準時封存預測。
2. 到期對帳：每份封存預測的 5／20／60 日到期後，逐檔記下「當時怎麼預測、實際怎麼走」（history/learning/matured/，
   寫過就不再改，所以同一筆不會被重複加入，也不能事後改寫）。
3. 錯誤分析：用全部已到期的真實預測，統計哪些產業、市場環境、模型最常選錯（只報數字與樣本數，不寫沒有證據的原因）。
4. 特徵監控：今天特徵的缺漏率與分布有沒有突然改變（漂移），以及已到期樣本上各特徵的排序能力有沒有和研究期相反。
5. 候選模型：研究資料集（每週研究上傳的 ml_panel）補上今天以前新到期的標籤；只有「真的有新標籤」才重新訓練 LightGBM 候選模型，
   訓練完走 learning.register 的上線檢查（沒通過就沿用上一版）。沒有新標籤、資料不齊、時間不夠時不訓練，並寫明原因。
6. 比賽與晉升：沿用 learning.gate／acceptance（只用封存後到期的真實預測），通過也只寫「等待人工確認」。

每天跑學習流程 ≠ 每天變準。報告只陳述今天實際發生的事；沒有新標籤就寫「今天沒有新的學習資料」。
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import factor_research as fr
from . import ml_research as ml
from . import pit_backtest as pb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "history" / "learning"
MAT = OUT / "matured"
FEAT = OUT / "features"
PRED = ROOT / "history" / "predictions"
PANEL = config.DATA_DIR / "ml_panel.pkl"
BUDGET = 1500          # 秒：超過就跳過重新訓練（GitHub Actions 這一步另有 30 分鐘上限）
TRAIN_COST = 300       # 預估重新訓練要多少秒；剩餘時間不夠就不開始
TOPN = 10
MIN_GROUP = 30         # 分組統計至少幾筆（股票×期）才下結論，否則標「僅供觀察」
DRIFT_MISS = 10.0      # 缺漏率比過去 20 天平均多 10 個百分點 → 警示
DRIFT_PSI = 0.25       # PSI ≥ 0.25 → 分布明顯改變
EFFECT_MIN = 20        # 至少 20 個到期日才判斷特徵是否失效
MODELS = ("v21", "lgbm", "ens")


# ---------------- 1. 特徵快照（封存預測時呼叫） ----------------
def save_features(X: pd.DataFrame, sig: str, out: Path = FEAT) -> Path | None:
    """封存當天的原始特徵值（之後算漂移與特徵有效性用）。已經存在就不覆寫。"""
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{sig}.json.gz"
    if p.exists():
        return None
    rec = {"signal_date": sig, "ids": X["sid"].tolist(), "industry": X["industry"].tolist() if "industry" in X else None,
           "feats": {f: [None if pd.isna(v) else float(f"{v:.5g}") for v in X[f]] for f in ml.FEATS if f in X}}
    p.write_bytes(gzip.compress(json.dumps(rec, separators=(",", ":")).encode()))
    return p


def load_features(sig: str, src: Path = FEAT) -> pd.DataFrame | None:
    p = src / f"{sig}.json.gz"
    if not p.exists():
        return None
    r = json.loads(gzip.decompress(p.read_bytes()))
    return pd.DataFrame(r["feats"], index=r["ids"], dtype=float)


# ---------------- 2. 到期對帳 ----------------
def _regime(D: dict, sig: str) -> str | None:
    from . import robustness
    return robustness.regime4(D.get("idx_s")).get(sig) if D.get("idx_s") is not None else None


def mature(D: dict, src: Path = PRED, dst: Path = MAT, horizons=("5", "20", "60")) -> list[dict]:
    """把已到期、還沒對帳過的封存預測寫成對帳檔。回傳今天新增的對帳檔摘要（已存在的不重算、不覆寫）。"""
    dates = D["dates"]
    pos = {d: i for i, d in enumerate(dates)}
    dst.mkdir(parents=True, exist_ok=True)
    new = []
    for f in sorted(src.glob("????-??-??.json")) if src.exists() else []:
        sig = f.stem
        if sig not in pos or pos[sig] + 1 >= len(dates):
            continue
        rec = None
        for h in horizons:
            out = dst / f"{sig}_h{h}.json"
            e = pos[sig] + 1 + int(h)
            if out.exists() or e >= len(dates):
                continue   # 已經對帳過，或還沒到期（未到期的標籤絕對不用）
            rec = rec or json.loads(f.read_text("utf-8"))
            entry, exit_ = dates[pos[sig] + 1], dates[e]
            b = pb._bench(D, entry, exit_)
            if b is None:
                continue
            ids, ex, st = [], [], []
            pct = {m: [] for m in MODELS}
            for s in rec["stocks"]:
                p = s.get(h) or {}
                if p.get("v21_pct") is None:
                    continue
                r, status = pb.hold_return(D, s["id"], entry, exit_)
                ids.append(s["id"])
                ex.append(None if r is None else round(r - b, 5))
                st.append(status)
                for m in MODELS:
                    pct[m].append(p.get(f"{m}_pct"))
            if sum(v is not None for v in ex) < 30:
                continue
            info = D.get("info", {})
            doc = {"schema_version": 1, "signal_date": sig, "horizon": h, "entry": entry, "exit": exit_, "bench_0050": round(b, 5),
                   "model_version": rec.get("model_version"), "generated_at": rec.get("generated_at"), "regime": _regime(D, sig),
                   "matured_at": config.now().isoformat(timespec="seconds"),
                   "ids": ids, "industry": [(info.get(i) or {}).get("industry") or "" for i in ids], "excess": ex, "status": st, "pct": pct}
            out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), "utf-8")
            new.append({"signal_date": sig, "horizon": h, "stocks": sum(v is not None for v in ex), "model_version": rec.get("model_version")})
    return new


def load_matured(h: str = "20", src: Path = MAT) -> list[dict]:
    return [json.loads(p.read_text("utf-8")) for p in sorted(src.glob(f"*_h{h}.json"))] if src.exists() else []


def _frame(L: list[dict]) -> pd.DataFrame:
    rows = []
    for d in L:
        for i, sid in enumerate(d["ids"]):
            if d["excess"][i] is None:
                continue
            rows.append({"sig": d["signal_date"], "sid": sid, "industry": d["industry"][i] or "（未分類）", "regime": d.get("regime") or "—",
                         "y": d["excess"][i], **{m: d["pct"][m][i] for m in MODELS}})
    return pd.DataFrame(rows)


# ---------------- 3. 錯誤分析 ----------------
def errors(L: list[dict]) -> dict:
    """已到期真實預測的錯誤統計（20 日）。前 10 名＝當天模型百分位最高的 10 檔。"""
    F = _frame(L)
    if not len(F):
        return {"dates": 0, "note": "還沒有到期的 20 日預測"}
    out = {"dates": int(F["sig"].nunique()), "stock_dates": int(len(F)), "models": {}, "by_industry": {}, "by_regime": {}, "worst": []}
    for m in MODELS:
        if F[m].isna().all():
            continue
        ics, tops, hits = [], [], []
        for _, g in F.groupby("sig"):
            g = g[g[m].notna()]
            if len(g) < 30:
                continue
            v = fr._ic(g[m], g["y"])
            ics.append(np.nan if v is None else v)
            t = g.nlargest(TOPN, m)
            tops.append(float(t["y"].mean()))
            hits.append(float((t["y"] > 0).mean()))
        out["models"][m] = {"dates": len(ics), "ic": round(float(np.nanmean(ics)), 4) if ics and not np.isnan(ics).all() else None,
                            "top_excess": round(float(np.mean(tops)) * 100, 2) if tops else None,
                            "beat_0050": round(float(np.mean(hits)) * 100, 1) if hits else None}
    tops = pd.concat([g.nlargest(TOPN, m).assign(model=m) for m in ("v21", "lgbm") if not F[m].isna().all() for _, g in F[F[m].notna()].groupby("sig")]) \
        if len(F) else pd.DataFrame()
    if len(tops):
        for key, col in (("by_industry", "industry"), ("by_regime", "regime")):
            for m, gm in tops.groupby("model"):
                rows = []
                for k, g in gm.groupby(col):
                    rows.append({"group": k, "n": int(len(g)), "avg_excess": round(float(g["y"].mean()) * 100, 2),
                                 "big_loss_pct": round(float((g["y"] < -0.10).mean()) * 100, 1), "enough": len(g) >= MIN_GROUP})
                out[key][m] = sorted(rows, key=lambda r: r["avg_excess"])
        w = tops.sort_values("y").head(8)
        out["worst"] = [{"sig": r.sig, "sid": r.sid, "industry": r.industry, "model": r.model, "excess": round(r.y * 100, 2)} for r in w.itertuples()]
    out["note"] = f"分組少於 {MIN_GROUP} 筆的只供觀察；這裡只描述哪裡錯，原因要再用對照實驗驗證。"
    return out


# ---------------- 4. 特徵監控 ----------------
def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float | None:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 100 or len(cur) < 100:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


def feature_drift(sig: str, src: Path = FEAT, lookback: int = 20) -> dict:
    files = sorted(p.name[:10] for p in src.glob("????-??-??.json.gz")) if src.exists() else []
    if sig not in files:
        return {"status": "no_data", "label": "今天沒有特徵快照"}
    prev = [d for d in files if d < sig][-lookback:]
    if len(prev) < 5:
        return {"status": "accumulating", "label": f"特徵快照累積中（{len(prev)}／5 天）"}
    cur = load_features(sig, src)
    ref = pd.concat([load_features(d, src) for d in prev])
    rows = []
    for f in cur.columns:
        miss_now, miss_ref = float(cur[f].isna().mean() * 100), float(ref[f].isna().mean() * 100)
        psi = _psi(ref[f].to_numpy(float), cur[f].to_numpy(float))
        flag = miss_now - miss_ref >= DRIFT_MISS or (psi is not None and psi >= DRIFT_PSI)
        rows.append({"feature": f, "missing": round(miss_now, 1), "missing_ref": round(miss_ref, 1), "psi": None if psi is None else round(psi, 3), "flag": flag})
    bad = [r for r in rows if r["flag"]]
    return {"status": "warn" if bad else "ok", "label": f"{len(bad)} 項特徵缺漏或分布明顯改變" if bad else "特徵缺漏率與分布沒有明顯改變",
            "reference_days": len(prev), "flagged": bad, "rows": rows}


def feature_effect(L: list[dict], research_ic: dict | None, src: Path = FEAT) -> dict:
    """已到期真實樣本上每個特徵的 Rank IC，和研究期（ml_panel）比較；方向相反且顯著才標「可能失效」。"""
    per: dict = {}
    for d in L:
        X = load_features(d["signal_date"], src)
        if X is None:
            continue
        y = pd.Series(d["excess"], index=d["ids"], dtype=float).dropna()
        X = X.reindex(y.index)
        for f in X.columns:
            if X[f].notna().sum() >= 30:
                per.setdefault(f, []).append(fr._ic(X[f], y))
    if not per:
        return {"dates": 0, "label": "還沒有可以評估的到期樣本（特徵快照從 V31 開始累積）", "rows": [], "hypotheses": []}
    rows, hyp = [], []
    for f, v in per.items():
        a = np.array([x for x in v if x is not None and not np.isnan(x)])
        if not len(a):
            continue
        t = float(a.mean() / fr._nw_se(a, 2)) if len(a) >= 4 else None
        base = (research_ic or {}).get(f)
        st = "樣本不足"
        if len(a) >= EFFECT_MIN and base is not None and t is not None:
            st = "可能失效" if (np.sign(a.mean()) != np.sign(base) and abs(t) >= 2) else "和研究期一致"
        rows.append({"feature": f, "dates": int(len(a)), "live_ic": round(float(a.mean()), 4), "t": None if t is None else round(t, 2),
                     "research_ic": base, "status": st})
        if st == "可能失效":
            hyp.append(f"{f} 在最近 {len(a)} 個到期日的 IC {a.mean():+.3f}（t {t:.2f}），研究期是 {base:+.3f}：下次每週研究用消融測試檢查拿掉它是否較好（不直接改正式模型）")
    n = max(r["dates"] for r in rows) if rows else 0
    return {"dates": n, "label": f"{n} 個到期日" + ("" if n >= EFFECT_MIN else f"（少於 {EFFECT_MIN} 個，先不判斷是否失效）"),
            "rows": sorted(rows, key=lambda r: r["live_ic"]), "hypotheses": hyp}


def research_ic(panel: pd.DataFrame, h: int = 20) -> dict:
    y = f"y{h}"
    P = panel[panel[y].notna()]
    out = {}
    for f in ml.FEATS:
        ics = [fr._ic(g[f], g[y]) for _, g in P.groupby("date") if g[f].notna().sum() >= 30]
        ics = [v for v in ics if v is not None and not np.isnan(v)]
        if ics:
            out[f] = round(float(np.mean(ics)), 4)
    return out


# ---------------- 5. 候選模型（有新標籤才重新訓練） ----------------
def refresh_panel(panel: pd.DataFrame, D: dict, cfg: dict, log=print) -> tuple[pd.DataFrame, dict]:
    """補上新的訊號日（只有特徵）與今天以前新到期的標籤。同一個（日期、股票）只會有一列。"""
    dates = D["dates"]
    pos = {d: i for i, d in enumerate(dates)}
    have = set(panel["date"].unique())
    sigs = [s for s in ml.signal_dates(dates, str(panel["date"].min())) if s not in have and pos.get(s, len(dates)) + 1 < len(dates)]
    ml._D, ml._CFG = D, cfg
    add = [ml._work(s) for s in sigs]
    for s, df in zip(sigs, add):
        log(f"新訊號日 {s}：{len(df)} 檔")
    P = pd.concat([panel, *add], ignore_index=True) if add else panel.copy()
    new = {str(h): 0 for h in ml.HORIZONS}
    for h in ml.HORIZONS:
        y, st = f"y{h}", f"st{h}"
        need = P[(P[st] == "no_exit") & P["date"].map(lambda d: d in pos and pos[d] + 1 + h < len(dates))]
        for d, g in need.groupby("date"):
            entry, exit_ = dates[pos[d] + 1], dates[pos[d] + 1 + h]
            b = pb._bench(D, entry, exit_)
            for i, sid in zip(g.index, g["sid"]):
                ret, s = pb.hold_return(D, sid, entry, exit_)
                P.at[i, y] = ret - b if ret is not None and b is not None else np.nan
                P.at[i, st] = s if b is not None else "no_bench"
            new[str(h)] += int(P.loc[g.index, y].notna().sum())
    P = P.drop_duplicates(["date", "sid"], keep="last").reset_index(drop=True)
    return P, {"new_signals": sigs, "new_labels": new}


def labeled_through(panel: pd.DataFrame) -> dict:
    return {str(h): (str(panel.loc[panel[f"y{h}"].notna(), "date"].max()) if panel[f"y{h}"].notna().any() else None) for h in ml.HORIZONS}


def retrain_decision(through_now: dict, active: dict | None) -> tuple[bool, str]:
    """有任何期間的「最新已到期訊號日」晚於上線版本的訓練資料，才需要重新訓練。"""
    if not active:
        return True, "目前沒有上線的候選模型"
    tp = active.get("train_period") or {}
    newer = [h for h, d in through_now.items() if d and d > ((tp.get(h) or {}).get("through") or "")]
    if newer:
        return True, "有新到期的訓練標籤（" + "、".join(f"{h} 日到 {through_now[h]}" for h in newer) + "）"
    return False, "沒有新到期的訓練標籤（訊號每 10 個交易日一個，標籤要等持有期結束才確定），沿用目前的候選模型"


def candidate(D: dict, cfg: dict, t0: float, budget: float, log=print, panel_path: Path = PANEL, root: Path | None = None) -> dict:
    from . import learning
    root = root or learning.MODEL_DIR
    if not panel_path.exists():
        return {"retrained": False, "reason": "沒有研究資料集（等每週「AI 歷史回測」產生並上傳 ml_panel）"}
    panel = pd.read_pickle(panel_path)
    panel, info = refresh_panel(panel, D, cfg, log)
    panel.to_pickle(panel_path)
    thr = labeled_through(panel)
    reg = learning.load_registry(root / "registry.json")
    act = next((v for v in reg["versions"] if v["version"] == reg.get("active")), None)
    need, why = retrain_decision(thr, act)
    out = {"retrained": False, "reason": why, **info, "labeled_through": thr, "panel_rows": int(len(panel)),
           "research_ic": research_ic(panel)}
    if not need:
        return out
    left = budget - (time.time() - t0)
    if left < TRAIN_COST:
        out["reason"] = f"需要重新訓練，但今天的運算時間只剩 {left:.0f} 秒（預估要 {TRAIN_COST} 秒），安全跳過，下一次再訓練"
        return out
    research = json.loads(ml.OUT.read_text("utf-8")) if ml.OUT.exists() else None
    e = learning.train_and_register(panel, research, D["dates"], True, log, root)
    out.update(retrained=True, reason=why, version=e.get("version"), status=e.get("status"), problems=e.get("problems"),
               rows={h: (v or {}).get("rows") for h, v in (e.get("train_period") or {}).items()})
    return out


# ---------------- 6. 報告 ----------------
def run(dbname: str = "stocks.db", budget: float = BUDGET, log=print, out_dir: Path = OUT, train: bool = True) -> dict:
    from . import learning, ml_predict as mp, premarket, track
    from . import db
    t0 = time.time()
    cfg = config.load()
    with db.connect(dbname) as con:
        last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        comp = mp.completeness(con, last) if last else {"ok": False, "why": "資料庫沒有股價"}
    D = pb.load(dbname)
    day = last or date.today().isoformat()
    sealed_p = PRED / f"{day}.json"
    sealed = json.loads(sealed_p.read_text("utf-8")) if sealed_p.exists() else None
    on_time = None
    if sealed:
        entry = premarket.next_trading_day(date.fromisoformat(day), premarket.holidays(refresh=False)).isoformat()
        on_time = track.on_time(sealed, entry)
    steps, skipped = [], []
    new = mature(D)
    steps.append(f"到期對帳：新增 {len(new)} 份")
    L = load_matured("20")
    err = errors(L)
    drift_f = feature_drift(day)
    cand = {"retrained": False, "reason": "這次只做監控（沒有開啟訓練）"}
    if train and comp.get("ok"):
        try:
            cand = candidate(D, cfg, t0, budget, log)
        except Exception as e:  # noqa: BLE001  候選模型失敗不影響正式模型與網站
            cand = {"retrained": False, "reason": f"候選模型更新失敗，沿用上一版：{e}"}
    elif train:
        cand = {"retrained": False, "reason": f"今天行情不完整，不訓練：{comp.get('why')}"}
        skipped.append("重新訓練")
    effect = feature_effect(L, cand.get("research_ic"))
    sc = json.loads((PRED / "scorecard.json").read_text("utf-8")) if (PRED / "scorecard.json").exists() else {"days": 0, "horizons": {}}
    st = learning.status(sc, cfg)
    reg = learning.load_registry()
    rolled = [e for e in reg.get("events", []) if e.get("kind") in ("rollback", "rollback_failed") and (e.get("at") or "")[:10] == day]
    risks = []
    if not comp.get("ok"):
        risks.append(f"行情不完整：{comp.get('why')}")
    if not sealed:
        risks.append("今天沒有封存預測")
    elif on_time is False:
        risks.append("今天的預測晚於隔天開盤才封存，不列入真實成績")
    if drift_f.get("status") == "warn":
        risks.append(drift_f["label"])
    if st["drift"].get("status") == "warn":
        risks.append(st["drift"]["label"])
    if cand.get("status") == "rejected":
        risks.append("新的候選模型沒通過上線檢查：" + "；".join(cand.get("problems") or []))
    if rolled:
        risks.append("今天發生模型回滾")
    human = [x for x in ([st["gate"]["label"]] if st["gate"].get("status") == "passed" else []) + risks if x]
    matured_new = sum(x["stocks"] for x in new)
    labels_new = sum((cand.get("new_labels") or {}).values())
    summary = ("今天沒有新的學習資料：沒有預測到期、也沒有新的訓練標籤。" if not new and not labels_new else
               f"今天新增 {len(new)} 份到期對帳（{matured_new} 筆股票結果）" + (f"、{labels_new} 筆研究訓練標籤" if labels_new else "") + "。")
    summary += "重新訓練了候選模型。" if cand.get("retrained") else "沒有重新訓練：" + cand.get("reason", "")
    rep = {"schema_version": 1, "date": day, "generated_at": config.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0),
           "summary": summary,
           "data": {"ok": bool(comp.get("ok")), "prices": last, "why": comp.get("why")},
           "sealed": {"today": bool(sealed), "on_time": on_time, "model_version": (sealed or {}).get("model_version"), "n": (sealed or {}).get("n")},
           "matured_today": new, "matured_stocks_today": matured_new,
           "live": {"sealed_days": sc.get("days", 0), "matured": {h: v.get("matured", 0) for h, v in sc.get("horizons", {}).items()}},
           "candidate": {k: v for k, v in cand.items() if k != "research_ic"},
           "champion": st["champion"], "challenger": (st.get("active") or {}).get("version"),
           "gate": {k: st["gate"].get(k) for k in ("status", "label", "days", "need", "live", "challenger_name")},
           "acceptance": st["acceptance"]["label"], "promotion_ok": st["acceptance"]["ok"], "auto_promote": st["auto_promote"],
           "model_drift": st["drift"], "feature_drift": {k: v for k, v in drift_f.items() if k != "rows"}, "feature_effect": effect,
           "errors": err, "risks": risks, "needs_human": bool(human), "human_why": human, "steps": steps, "skipped": skipped,
           "limits": ["每天跑學習流程不代表每天變準；是否變好只看到期的真實預測與樣本外比賽",
                      "20 日預測要 20 個交易日後才知道對錯，真實樣本累積很慢",
                      "資料庫沒有法人、融資券等籌碼資料，也沒有歷史處置股、全額交割紀錄"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{day}.json"
    if p.exists():  # 同一天重跑：保留第一次的時間與今天已新增的對帳，其餘更新為最新狀態
        old = json.loads(p.read_text("utf-8"))
        rep["first_run_at"] = old.get("first_run_at") or old.get("generated_at")
        rep["runs"] = old.get("runs", 1) + 1
        rep["matured_today"] = old.get("matured_today", []) + new
        rep["matured_stocks_today"] = old.get("matured_stocks_today", 0) + matured_new
        if old.get("candidate", {}).get("retrained") and not cand.get("retrained"):
            rep["candidate"] = {**rep["candidate"], "retrained_earlier_today": old["candidate"].get("version")}
    else:
        rep["first_run_at"], rep["runs"] = rep["generated_at"], 1
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), "utf-8")
    log(summary)
    return rep


def site_summary(src: Path = OUT, days: int = 30) -> dict | None:
    """網站用的精簡版：最新一天完整報告＋最近 30 天每天一行。"""
    files = sorted(src.glob("????-??-??.json")) if src.exists() else []
    if not files:
        return None
    latest = json.loads(files[-1].read_text("utf-8"))
    hist = []
    for f in files[-days:][::-1]:
        r = json.loads(f.read_text("utf-8"))
        hist.append({"date": r["date"], "data_ok": r["data"]["ok"], "sealed": r["sealed"]["today"], "on_time": r["sealed"]["on_time"],
                     "matured": len(r.get("matured_today", [])), "labels": sum((r["candidate"].get("new_labels") or {}).values()),
                     "retrained": r["candidate"].get("retrained"), "version": r["candidate"].get("version"), "risks": len(r.get("risks", []))})
    return {"latest": latest, "history": hist, "run_status": run_status()}


RUNS = OUT / "runs.json"
ALERT_FAILS = 2        # 連續失敗幾次就讓 Actions 標成失敗（寄通知信）


def record(ok: bool, error: str | None = None, day: str | None = None, seconds: float | None = None, path: Path | None = None) -> dict:
    """每一次每日學習的執行結果（成功或失敗都記）。同一個 Actions 執行只記一次。"""
    import os
    path = path or RUNS
    log = json.loads(path.read_text("utf-8")) if path.exists() else {"runs": []}
    rid = os.environ.get("GITHUB_RUN_ID") or f"local-{time.time():.0f}"
    if any(r.get("run_id") == rid for r in log["runs"]):
        return log
    log["runs"].append({"at": config.now().isoformat(timespec="seconds"), "run_id": rid, "ok": ok, "date": day,
                        "error": (error or "")[:300] or None, "seconds": None if seconds is None else round(seconds)})
    log["runs"] = log["runs"][-200:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
    return log


def run_status(path: Path | None = None, last: int = 30) -> dict:
    """成功率、連續失敗次數、最近一次成功時間、最近一次失敗原因。"""
    path = path or RUNS
    runs = (json.loads(path.read_text("utf-8")) if path.exists() else {"runs": []})["runs"]
    if not runs:
        return {"runs": 0, "label": "還沒有執行紀錄"}
    recent = runs[-last:]
    streak = 0
    for r in reversed(runs):
        if r["ok"]:
            break
        streak += 1
    ok = [r for r in runs if r["ok"]]
    bad = [r for r in runs if not r["ok"]]
    lab = ("最近一次執行成功" if runs[-1]["ok"] else f"最近一次執行失敗（連續 {streak} 次）：{runs[-1].get('error') or '原因不明'}")
    return {"runs": len(runs), "recent": len(recent), "success_rate": round(sum(r["ok"] for r in recent) / len(recent) * 100, 1),
            "fail_streak": streak, "last_ok": ok[-1]["at"] if ok else None, "last_fail": bad[-1] if bad else None,
            "last": runs[-1], "alert": streak >= ALERT_FAILS, "label": lab}


def main(argv: list[str]) -> int:
    if argv[:1] == ["record"]:   # 工作流程補記：這一步被強制停止（逾時）時，程式自己來不及記錄
        outcome = argv[1] if len(argv) > 1 else "unknown"
        if outcome != "success":
            record(False, f"Actions 步驟結果：{outcome}（可能逾時或環境錯誤）")
        st = run_status()
        print(st["label"], f"成功率 {st.get('success_rate')}%")
        return 1 if st.get("alert") else 0
    t0 = time.time()
    try:
        rep = run(train="--no-train" not in argv)
    except Exception as e:  # noqa: BLE001  失敗要留下紀錄，網站不會顯示「今天已完成學習」
        import traceback
        traceback.print_exc()
        record(False, f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        return 1
    record(True, None, rep["date"], time.time() - t0)
    print(json.dumps({k: rep[k] for k in ("date", "summary", "needs_human")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
