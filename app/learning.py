"""V25 AI 持續學習引擎（影子模式）：每週重新訓練、版本登記、安全上線與回滾、新舊模型比賽、錯誤分析。

自動學習 ≠ 越來越準。這個模組只負責讓「重新訓練」安全、可追溯、可比較：
1. 版本登記（history/models/registry.json）：每次每週訓練產生一個版本，記錄訓練期間、研究驗證期間、特徵版本、
   模型雜湊與樣本外指標。先在暫存資料夾訓練並檢查（能載入、係數有限、試算預測不是常數、標籤都已到期），
   通過才換上線；沒通過的版本標「未通過檢查」，網站繼續用上一版。
2. 回滾：每日預測載入模型失敗時，自動退回上一個檢查過的版本（保留最近 KEEP 版的檔案；更舊的可從 git 歷史找回）。
3. Champion–Challenger：正式排行榜（Champion）是 V21 固定權重；候選（Challenger）是「每週重新訓練的 LightGBM」。
   比賽只用每天封存、事後到期的真實預測（不是訓練內績效），門檻 2026-10-17 事先定好、之後不依結果調整。
   通過門檻也只顯示「通過，等待人工確認」：換掉正式排行榜要由網站主人決定（config learning.auto_promote 預設 false）。
4. 錯誤分析：用 walk-forward 樣本外預測，依市場環境、波動、漲幅、估值、規模、產業分組，看模型在哪裡最常判斷錯。
   只報數字和 t 值，不寫因果；樣本不足的分組標「僅供觀察」。
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import factor_research as fr
from . import ml_research as ml

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "history" / "models"
PRED_DIR = ROOT / "history" / "predictions"
REGISTRY = MODEL_DIR / "registry.json"
VERSIONS = MODEL_DIR / "versions"
KEEP = 4                      # 保留最近幾個通過檢查的版本檔案（回滾用）
FILES = ("ml.json", "lgbm_5.txt", "lgbm_20.txt", "lgbm_60.txt")

CHAMPION = {"id": "v21", "name": "V21 固定權重", "role": "正式排行榜"}
CHALLENGER = {"id": "lgbm", "name": "LightGBM（每週重新訓練）", "role": "影子模式，不影響排行榜"}
# 升級門檻：2026-10-17 事先定好，之後不依結果調整（要改就換 GATE_VERSION，並在網站註明）
GATE_VERSION = "2026-10-17"
GATE = {"horizon": "20", "min_days": 60, "ic_t": 2.0, "min_beat": 50.0, "max_mdd_gap": 5.0}


def feature_version() -> str:
    return hashlib.sha1(json.dumps([ml.METHOD, ml.FEATS]).encode()).hexdigest()[:8]


# ---------------- 版本登記 ----------------
def load_registry(path: Path = REGISTRY) -> dict:
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return {"schema_version": 1, "champion": CHAMPION, "challenger": CHALLENGER, "active": None, "versions": [], "events": []}


def _save_registry(reg: dict, path: Path = REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reg["events"] = reg["events"][-50:]
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1), "utf-8")


def _event(reg: dict, kind: str, msg: str) -> None:
    reg["events"].append({"at": config.now().isoformat(timespec="seconds"), "kind": kind, "msg": msg})


def validate(src: Path, sample: pd.DataFrame | None = None, dates: list[str] | None = None) -> list[str]:
    """上線前檢查。回傳問題清單（空的＝通過）。"""
    from . import ml_predict as mp
    bad = []
    if not (src / "ml.json").exists():
        return ["沒有 ml.json"]
    M = mp.load_models(src)
    if M is None:
        return ["特徵定義和程式不一致，無法載入"]
    if not M.get("horizons"):
        return ["沒有任何期間的模型"]
    n = len(ml.FEATS) + 1
    for h, H in M["horizons"].items():
        for k in ("ridge", "logit"):
            w = np.array(H.get(k) or [], float)
            if len(w) != n or not np.all(np.isfinite(w)):
                bad.append(f"{h} 日 {k} 係數不完整")
        if H.get("lgbm") and h not in M["_lgbm"] and _has_lgbm():
            bad.append(f"{h} 日 LightGBM 檔案無法載入")
        if dates and H.get("through"):
            # 標籤到期檢查：最後一個訓練訊號日隔天進場、第 h 個交易日出場，出場日必須在資料範圍內（未到期的標籤不能拿來訓練）
            i = dates.index(H["through"]) if H["through"] in dates else None
            if i is None or i + 1 + int(h) >= len(dates):
                bad.append(f"{h} 日訓練資料截止 {H['through']}，標籤還沒到期")
    if sample is not None and len(sample) >= 50:
        try:
            P = mp.predict(sample, M)
            for h, p in P.items():
                for k, v in p.items():
                    if k == "v21":
                        continue
                    a = v.to_numpy(float)
                    if not np.all(np.isfinite(a)):
                        bad.append(f"{h} 日 {k} 試算出現無效數字")
                    elif np.std(a) == 0:
                        bad.append(f"{h} 日 {k} 試算每檔都一樣")
        except Exception as e:
            bad.append(f"試算失敗：{e}")
    return bad


def _has_lgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False  # 沒裝 LightGBM 的環境（例如快速更新）只用線性模型，不算模型壞掉


def _copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (src / f).exists():
            tmp = dst / (f + ".tmp")
            shutil.copyfile(src / f, tmp)
            tmp.replace(dst / f)
        elif (dst / f).exists():
            (dst / f).unlink()


def register(staging: Path, research: dict | None = None, sample: pd.DataFrame | None = None, dates: list[str] | None = None,
             seconds: float | None = None, root: Path = MODEL_DIR) -> dict:
    """暫存資料夾裡剛訓練好的模型 → 檢查 → 登記 → 通過才上線。回傳這一版的登記資料。"""
    reg_path, vers = root / "registry.json", root / "versions"
    reg = load_registry(reg_path)
    meta = json.loads((staging / "ml.json").read_text("utf-8"))
    ver = f"{meta['trained_at'][:10]}-{meta['model_hash']}"
    if reg.get("active") and reg["active"].endswith(meta["model_hash"]):
        _event(reg, "unchanged", f"重新訓練的模型和上線版本相同（{meta['model_hash']}），不換版")
        _save_registry(reg, reg_path)
        return next(v for v in reg["versions"] if v["version"] == reg["active"])
    R = (research or {}).get("horizons", {}).get(GATE["horizon"], {})
    entry = {"version": ver, "model_hash": meta["model_hash"], "trained_at": meta["trained_at"], "method": meta.get("method"),
             "feature_version": feature_version(), "seconds": round(seconds) if seconds else None,
             "train_period": {h: {"from": (research or {}).get("start"), "through": H.get("through"), "rows": H.get("rows"), "dates": H.get("dates")}
                              for h, H in meta.get("horizons", {}).items()},
             "validation": {"method": "walk-forward（每個測試日只用標籤已到期的資料訓練）", "from": R.get("first_test"), "to": R.get("last_test"),
                            "holdout_from": R.get("holdout_from"),
                            "metrics": {m: {k: (R.get("all", {}).get(m) or {}).get(k) for k in ("ic", "ic_t", "top_excess", "top_vs_all")}
                                        for m in ("v21", "ridge", "lgbm")} if R else None,
                            "holdout": {m: (R.get("holdout", {}).get(m) or {}).get("ic") for m in ("v21", "ridge", "lgbm")} if R else None,
                            "vs_v21": R.get("vs_v21", {}).get("lgbm") if R else None},
             "importance": {h: H.get("importance") for h, H in meta.get("horizons", {}).items() if H.get("importance")}}
    problems = validate(staging, sample, dates)
    if problems:
        entry.update(status="rejected", problems=problems)
        reg["versions"].append(entry)
        _event(reg, "rejected", f"{ver} 沒通過上線檢查：{'；'.join(problems)}。繼續使用 {reg.get('active') or '（無）'}")
        _save_registry(reg, reg_path)
        return entry
    meta["version"] = ver
    (staging / "ml.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    _copy(staging, vers / ver)
    _copy(staging, root)
    for v in reg["versions"]:
        if v.get("status") == "active":
            v["status"] = "previous"
    entry["status"] = "active"
    reg["versions"].append(entry)
    prev = reg.get("active")
    reg["active"] = ver
    _event(reg, "activated", f"{ver} 通過檢查並上線為候選模型" + (f"（取代 {prev}）" if prev else ""))
    _prune(reg, vers)
    _save_registry(reg, reg_path)
    return entry


def _prune(reg: dict, vers: Path) -> None:
    ok = [v for v in reg["versions"] if v.get("status") in ("active", "previous") and not v.get("files_pruned")]
    for v in ok[:-KEEP]:
        shutil.rmtree(vers / v["version"], ignore_errors=True)
        v["files_pruned"] = True


def rollback(reason: str, root: Path = MODEL_DIR) -> str | None:
    """上線版本載入失敗：退回最近一個還有檔案、而且能通過檢查的版本。回傳換上的版本，找不到回傳 None。"""
    reg_path, vers = root / "registry.json", root / "versions"
    reg = load_registry(reg_path)
    bad = reg.get("active")
    for v in reversed(reg["versions"]):
        if v["version"] == bad or v.get("status") not in ("active", "previous") or v.get("files_pruned"):
            continue
        if not (vers / v["version"] / "ml.json").exists() or validate(vers / v["version"]):
            continue
        _copy(vers / v["version"], root)
        for x in reg["versions"]:
            if x["version"] == bad:
                x["status"], x["problems"] = "rolled_back", [reason]
        v["status"] = "active"
        reg["active"] = v["version"]
        _event(reg, "rollback", f"{bad} 無法使用（{reason}），已退回 {v['version']}")
        _save_registry(reg, reg_path)
        return v["version"]
    _event(reg, "rollback_failed", f"{bad} 無法使用（{reason}），沒有可退回的版本，今天不產生影子預測")
    _save_registry(reg, reg_path)
    return None


def train_and_register(panel: pd.DataFrame, research: dict, dates: list[str], use_lgbm: bool = True, log=print, root: Path = MODEL_DIR) -> dict:
    """ml_research 每週跑完後呼叫：在暫存資料夾訓練，檢查通過才上線。"""
    from . import ml_predict as mp
    import time
    t0 = time.time()
    with tempfile.TemporaryDirectory() as t:
        st = Path(t)
        mp.export_models(panel, use_lgbm, out=st)
        last = panel["date"].max()
        sample = panel[panel["date"] == last][["sid", "v21"] + ml.FEATS].copy() if "v21" in panel else None
        e = register(st, research, sample, dates, time.time() - t0, root)
    log(f"模型版本 {e['version']}：{ {'active': '通過檢查並上線', 'rejected': '沒通過檢查，沿用上一版'}.get(e.get('status'), e.get('status')) }")
    return e


# ---------------- 新舊模型比賽（只用封存後到期的真實預測） ----------------
def _mdd(x: list[float]) -> float:
    eq, peak, worst = 1.0, 1.0, 0.0
    for r in x:
        eq *= 1 + r
        peak = max(peak, eq)
        worst = min(worst, eq / peak - 1)
    return worst * 100


def gate(daily: list[dict], cost_pct: float) -> dict:
    """Champion（V21）vs Challenger（LightGBM）在同一批到期日上的比較。"""
    h = GATE["horizon"]
    rows = sorted([d for d in daily if d["h"] == h and d["ic"].get("lgbm") is not None and d["ic"].get("v21") is not None],
                  key=lambda d: d["sig"])
    n = len(rows)
    out = {"gate_version": GATE_VERSION, "rule": GATE, "horizon": h, "days": n, "need": GATE["min_days"], "cost_pct": cost_pct}
    if not n:
        out.update(status="accumulating", label="累積中", criteria=[])
        return out
    c = cost_pct / 100
    ic = {m: np.array([d["ic"][m] for d in rows], float) for m in ("v21", "lgbm")}
    top = {m: np.array([d["top"][m] for d in rows], float) - c for m in ("v21", "lgbm")}
    diff = ic["lgbm"] - ic["v21"]
    t = float(diff.mean() / fr._nw_se(diff, int(h))) if n >= 4 else None
    step = int(h)  # 不重疊的持有期才能算回撤
    mdd = {m: _mdd(list(top[m][::step])) for m in top}
    beat = float((top["lgbm"] > 0).mean() * 100)
    crit = [
        {"name": "到期的真實預測天數", "need": f"≥ {GATE['min_days']} 天", "value": f"{n} 天", "ok": n >= GATE["min_days"]},
        {"name": "Rank IC 勝過 V21（Newey-West t）", "need": f"t ≥ {GATE['ic_t']}", "value": f"差 {diff.mean():+.3f}，t {t:.2f}" if t is not None else "樣本太少", "ok": bool(t is not None and t >= GATE["ic_t"])},
        {"name": "前 20 名扣成本超額報酬", "need": "> 0 且高於 V21", "value": f"{top['lgbm'].mean() * 100:+.2f}%（V21 {top['v21'].mean() * 100:+.2f}%）",
         "ok": bool(top["lgbm"].mean() > 0 and top["lgbm"].mean() > top["v21"].mean())},
        {"name": "跑贏 0050 的天數比例（扣成本）", "need": f"≥ {GATE['min_beat']:.0f}%", "value": f"{beat:.0f}%", "ok": beat >= GATE["min_beat"]},
        {"name": "最大回撤不比 V21 差太多", "need": f"差距 ≤ {GATE['max_mdd_gap']:.0f} 個百分點", "value": f"{mdd['lgbm']:.1f}%（V21 {mdd['v21']:.1f}%）",
         "ok": mdd["lgbm"] >= mdd["v21"] - GATE["max_mdd_gap"]},
    ]
    if n < GATE["min_days"]:
        st, lab = "accumulating", f"累積中（{n}／{GATE['min_days']} 天）"
    elif all(x["ok"] for x in crit):
        st, lab = "passed", "通過門檻，等待人工確認是否升級"
    else:
        st, lab = "kept", "未通過，維持正式模型"
    out.update(status=st, label=lab, criteria=crit,
               live={m: {"ic": round(float(ic[m].mean()), 4), "top_net": round(float(top[m].mean()) * 100, 2), "mdd": round(mdd[m], 1)} for m in ic})
    return out


def drift(daily: list[dict], research: dict | None) -> dict:
    """模型漂移監控：最近 60 個到期日的 V21 實盤 IC，和歷史研究的 IC 比較。"""
    h = GATE["horizon"]
    xs = [d["ic"]["v21"] for d in sorted(daily, key=lambda d: d["sig"]) if d["h"] == h and d["ic"].get("v21") is not None][-60:]
    hist = (((research or {}).get("horizons", {}).get(h, {}).get("all") or {}).get("v21") or {}).get("ic")
    if len(xs) < 20:
        return {"status": "accumulating", "label": f"樣本不足（{len(xs)}／20 天）", "recent_ic": None, "research_ic": hist}
    a = np.array(xs, float)
    t = float(a.mean() / fr._nw_se(a, int(h)))
    st = "warn" if t <= -2 else "ok"
    return {"status": st, "label": "最近實盤 IC 顯著為負，可能出現模型漂移" if st == "warn" else "沒有顯著漂移",
            "recent_ic": round(float(a.mean()), 4), "t": round(t, 2), "days": len(xs), "research_ic": hist}


def lessons(reg: dict) -> list[str]:
    """「AI 最近學到了什麼」：只根據 LightGBM 特徵重要性（增益）的實際變化，不寫因果。"""
    ok = [v for v in reg.get("versions", []) if v.get("status") in ("active", "previous", "rolled_back") and v.get("importance", {}).get(GATE["horizon"])]
    if not ok:
        return []
    cur = ok[-1]["importance"][GATE["horizon"]]
    lab = {**{f: f for f in ml.FEATS}, **_FEAT_LABEL}
    top = sorted(cur, key=cur.get, reverse=True)[:5]
    out = [f"20 日模型目前最依賴：{'、'.join(f'{lab[f]}（{cur[f]:.0f}%）' for f in top)}。"]
    if len(ok) > 1:
        old = ok[-2]["importance"][GATE["horizon"]]
        rk = lambda d: {f: i + 1 for i, f in enumerate(sorted(d, key=d.get, reverse=True))}
        a, b = rk(old), rk(cur)
        mv = sorted([f for f in cur if f in a], key=lambda f: a[f] - b[f])
        up = [f for f in mv[::-1] if a[f] - b[f] >= 3][:2]
        dn = [f for f in mv if b[f] - a[f] >= 3][:2]
        if up:
            out.append("和上一版相比重要性上升：" + "、".join(f"{lab[f]}（第 {a[f]} → {b[f]} 名）" for f in up) + "。")
        if dn:
            out.append("重要性下降：" + "、".join(f"{lab[f]}（第 {a[f]} → {b[f]} 名）" for f in dn) + "。")
        if not up and not dn:
            out.append("和上一版相比，特徵重要性排名沒有明顯變化。")
    out.append("這是模型內部的權重變化，只說明模型「看什麼」，不代表因果，也不代表變準。")
    return out


_FEAT_LABEL = {"score_quality": "品質分數", "score_growth": "成長分數", "score_value": "價值分數", "score_momentum": "動能分數", "score_trend": "趨勢分數",
               "mom_12_1": "12-1 月動能", "ret_1m": "近 1 月報酬", "ret_3m": "近 3 月報酬", "rs_rank": "相對強度", "vol_1y": "一年波動",
               "downside_vol": "下檔波動", "beta": "Beta", "mdd_1y": "一年最大回撤", "dist_ma60": "季線乖離", "ma60_slope_20": "季線斜率",
               "rsi": "RSI", "vol_ratio": "量比", "rev_yoy": "月營收年增", "rev_yoy_3m": "近三月營收年增", "eps_q_yoy": "EPS 年增",
               "op_q_yoy": "營益年增", "roe": "ROE", "gross_margin": "毛利率", "op_margin": "營益率", "debt_ratio": "負債比",
               "earnings_yield": "盈餘殖利率", "book_to_price": "淨值股價比", "dy": "殖利率", "fcf_yield": "自由現金流殖利率",
               "fscore": "F-Score", "market_cap": "市值", "avg_value_20": "成交金額"}


# ---------------- 錯誤分析（walk-forward 樣本外預測） ----------------
BUCKETS = [("vol_1y", "波動", ["最低 20%", "中低", "中間", "中高", "最高 20%"]),
           ("ret_3m", "近 3 月漲幅", ["最低 20%", "中低", "中間", "中高", "最高 20%"]),
           ("earnings_yield", "估值（盈餘殖利率）", ["最貴 20%", "偏貴", "中間", "偏便宜", "最便宜 20%"]),
           ("market_cap", "規模", ["最小 20%", "偏小", "中間", "偏大", "最大 20%"])]


def error_analysis(preds: pd.DataFrame, panel: pd.DataFrame, h: int, idx: pd.Series | None, models=("v21", "lgbm")) -> dict:
    """模型在哪些情況下最常判斷錯：各分組內的 Rank IC 與「模型前 20% 的超額報酬」。"""
    y = f"y{h}"
    cols = [c for c, _, _ in BUCKETS]
    P = preds.merge(panel[["date", "sid"] + cols], on=["date", "sid"], how="left")
    lag = max(1, math.ceil(h / ml.STEP))
    groups = []

    def stat(sub: pd.DataFrame, dim: str, name: str):
        rec = {"dim": dim, "bucket": name, "rows": int(len(sub)), "dates": int(sub["date"].nunique())}
        for m in models:
            if m not in sub:
                continue
            ics = np.array([fr._ic(g[m], g[y]) if len(g) >= 30 else np.nan for _, g in sub.groupby("date")], float)
            ok = ics[~np.isnan(ics)]
            t = float(ok.mean() / fr._nw_se(ok, lag)) if len(ok) >= 6 else None
            tops = [g.nlargest(max(3, len(g) // 5), m)[y].mean() - g[y].mean() for _, g in sub.groupby("date") if len(g) >= 15]
            if len(ok) < 6:
                ok = ok[:0]  # 少於 6 個測試日：不算 IC，避免一兩天的運氣被當成「最容易錯」
            rec[m] = {"ic": round(float(ok.mean()), 4) if len(ok) else None, "t": round(t, 2) if t is not None else None, "n": int(len(ok)),
                      "top_vs_bucket": round(float(np.mean(tops)) * 100, 2) if tops else None}
        groups.append(rec)

    if idx is not None and len(idx):
        ma = idx.rolling(120).mean()
        bull = {d: bool(idx.get(d, np.nan) > ma.get(d, np.nan)) for d in P["date"].unique() if d in ma.index and pd.notna(ma.get(d))}
        P["regime"] = P["date"].map(bull)
        for k, nm in ((True, "多頭（加權指數在半年線上）"), (False, "空頭（加權指數在半年線下）")):
            stat(P[P["regime"] == k], "市場環境", nm)
    for c, dim, names in BUCKETS:
        q = P.groupby("date")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() >= 25 else np.nan)
        for i, nm in enumerate(names):
            stat(P[q == i], dim, nm)
    big = P["industry"].value_counts() / max(1, P["date"].nunique())
    for ind in big[big >= 30].index[:12]:  # 平均每個測試日至少 30 檔才算得出產業內的 IC
        stat(P[P["industry"] == ind], "產業", ind)
    for g in groups:
        x = g.get("lgbm") or g.get("v21") or {}
        g["evidence"] = "IC 估計可靠（|t|≥2）" if x.get("t") is not None and abs(x["t"]) >= 2 else "樣本不足，僅供觀察"
    worst = sorted([g for g in groups if (g.get("lgbm") or {}).get("ic") is not None and g["lgbm"]["n"] >= 10], key=lambda g: g["lgbm"]["ic"])[:5]
    return {"horizon": h, "method": "分組內的 Rank IC（各測試日分別計算再平均，Newey-West t）與模型前 20% 相對同組平均的超額報酬",
            "groups": groups, "weakest": [f"{g['dim']}・{g['bucket']}" for g in worst]}


# ---------------- 每日：學習報告 ----------------
def status(scorecard: dict, cfg: dict | None = None, reg_path: Path = REGISTRY, research_path: Path = ml.OUT) -> dict:
    cfg = cfg or config.load()
    reg = load_registry(reg_path)
    research = json.loads(research_path.read_text("utf-8")) if research_path.exists() else None
    from . import track
    cost = round(track.COST * 100, 3)
    daily = scorecard.get("daily", [])
    act = next((v for v in reg["versions"] if v["version"] == reg.get("active")), None)
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "shadow": True,
            "auto_promote": bool(cfg.get("learning", {}).get("auto_promote", False)),
            "champion": reg.get("champion", CHAMPION), "challenger": reg.get("challenger", CHALLENGER), "active": act,
            "versions": [{k: v.get(k) for k in ("version", "trained_at", "status", "problems", "validation", "train_period", "seconds", "files_pruned")}
                         for v in reg["versions"][-12:]][::-1],
            "events": reg["events"][-10:][::-1],
            "live": {"days": scorecard.get("days", 0), "first": scorecard.get("first"), "last": scorecard.get("last"),
                     "matured": {h: v.get("matured", 0) for h, v in scorecard.get("horizons", {}).items()}, "horizons": scorecard.get("horizons", {})},
            "gate": gate(daily, cost), "drift": drift(daily, research), "lessons": lessons(reg),
            "errors": (research or {}).get("errors"), "research": {"generated_at": (research or {}).get("generated_at"), "verdict": (research or {}).get("verdict")},
            "schedule": [{"when": "每天盤後（完整更新）", "what": "用上線版本產生 5／20／60 日預測並封存；對帳已到期的預測；更新比賽與漂移監控"},
                         {"when": "每週六", "what": "用已到期的資料重新訓練候選模型，檢查通過才上線（沒通過就沿用上一版）"},
                         {"when": "門檻判定", "what": f"到期的 20 日預測滿 {GATE['min_days']} 天後才開始判定；通過也要人工確認才會換掉正式排行榜"}]}
