"""V14 三本帳：歷史研究、樣本外比較、真實前瞻，各自獨立計算，網站分開顯示，數字不互相借用。

1. 歷史研究（research）：PIT 回測。規則和權重是看過這段歷史之後才定的，只能當參考。
2. 樣本外比較（oos）：walk-forward 模型比賽、LightGBM 專項研究、投資組合研究。每個測試日只用之前的資料訓練；
   但 2024-09 之後的保留期已經被看過很多次，不再算完全獨立。
3. 真實前瞻（live）：每天開盤前封存、事後到期才對帳的預測。這是唯一沒有被看過的資料，只有它能證明模型真的變好。
   每天記一行（history/learning/live_ledger.json），只新增不改寫。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from . import factor_research as fr
from . import track

ROOT = Path(__file__).resolve().parent.parent
BT = ROOT / "history" / "backtest"
LIVE = ROOT / "history" / "learning" / "live_ledger.json"
H = 20
TOP = 10


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text("utf-8")) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _mdd(r: list[float]) -> float | None:
    if not r:
        return None
    c = np.cumprod(1 + np.array(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
    return round(float((c / peak - 1).min()) * 100, 2)


def live_metrics(L: list[dict], models=("v21", "lgbm", "ens")) -> dict:
    """已到期的 20 日封存預測：Rank IC、前 10 名扣成本超額、跑贏 0050 比例、命中率、最大回撤、成熟樣本數。"""
    from . import daily_learning as dl
    F = dl._frame(L)
    out = {"matured_dates": int(F["sig"].nunique()) if len(F) else 0, "matured_stocks": int(len(F)), "models": {}}
    if not len(F):
        return out
    for m in models:
        if m not in F or F[m].isna().all():
            continue
        ics, tops, hits, sigs = [], [], [], []
        for s, g in F.groupby("sig"):
            g = g[g[m].notna()]
            if len(g) < 30:
                continue
            v = fr._ic(g[m], g["y"])
            if v is not None and np.isfinite(v):
                ics.append(v)
            t = g.nlargest(TOP, m)
            tops.append(float(t["y"].mean()) - track.COST)
            hits.append(float((t["y"] > 0).mean()))
            sigs.append(s)
        if not tops:
            continue
        x = np.array(tops)
        out["models"][m] = {"dates": len(tops), "ic": round(float(np.mean(ics)), 4) if ics else None,
                            "top_net": round(float(x.mean()) * 100, 2), "beat_0050": round(float((x > 0).mean()) * 100, 1),
                            "hit": round(float(np.mean(hits)) * 100, 1), "mdd": _mdd(list(x[::H])),   # 每天一份，不重疊的 20 日期間才能算回撤
                            "first": sigs[0], "last": sigs[-1]}
    return out


def record_live(day: str, path: Path | None = None) -> dict:
    """每天記一行真實前瞻成績（同一天重跑只覆寫當天那一行）。"""
    from . import daily_learning as dl
    path = path or LIVE
    log = _read(path) or {"note": "真實前瞻帳本：只用開盤前封存、事後到期的 20 日預測；每天一行，不回頭修改過去的行", "rows": []}
    sc = _read(dl.PRED / "scorecard.json") or {}
    row = {"date": day, "at": config.now().isoformat(timespec="seconds"), "sealed_days": sc.get("days", 0), **live_metrics(dl.load_matured(str(H)))}
    log["rows"] = [r for r in log["rows"] if r["date"] != day] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")
    return row


def research() -> dict:
    d = _read(BT / "ai_pit.json")
    if not d:
        return {"ready": False}
    modes = {}
    for k, m in (d.get("modes") or {}).items():
        t10 = ((m.get("top") or {}).get("10") or {})
        modes[k] = {"name": m.get("name"), "periods": t10.get("periods"), "total": t10.get("total"), "cagr": t10.get("cagr"), "mdd": t10.get("mdd"),
                    "avg_excess": t10.get("avg_excess"), "bench_total": t10.get("bench_total")}
    return {"ready": True, "start": d.get("start"), "end": d.get("data_end"), "modes": modes,
            "caveat": "V21 權重和選股規則是看過這段歷史後才定的，回測成績會偏好看，只能當參考"}


def oos() -> dict:
    out = {"ready": False}
    r = _read(BT / "race.json")
    if r:
        R = r.get("results", {})
        pick = lambda k: {x: (R.get(k, {}).get(part) or {}).get(x) for part in ("portfolio",) for x in ("net_excess", "t", "mdd")} | \
            {"holdout": (R.get(k, {}).get("holdout") or {}).get("net_excess"), "ic": (R.get(k, {}).get("ic") or {}).get("ic")}
        out.update(ready=True, race={"first": r.get("first_test"), "last": r.get("last_test"), "holdout_from": r.get("holdout_from"),
                                     "v21": pick("v21"), "lgbm": pick("lgbm")})
    s = _read(BT / "lgbm_study.json")
    if s:
        out.update(ready=True, lgbm_study={"passed": s.get("passed"), "conclusion": s.get("conclusion"), "periods": s.get("independent_periods")})
    p = _read(BT / "portfolio_study.json")
    if p:
        R = p.get("results", {})
        out.update(ready=True, portfolio_study={"passed": p.get("passed"), "conclusion": p.get("conclusion"), "periods": p.get("periods"),
                                                "top10": {k: (R.get("top10") or {}).get(k) for k in ("net_excess", "t", "mdd", "total", "bench_total")},
                                                "walk_select": p.get("walk_select")})
    out["caveat"] = "每個測試日只用之前的資料訓練；但 2024-09 之後的保留期已被看過多次，不再算完全獨立的證據"
    return out


def build() -> dict:
    lv = _read(LIVE) or {"rows": []}
    return {"generated_at": config.now().isoformat(timespec="seconds"), "research": research(), "oos": oos(),
            "live": {"latest": lv["rows"][-1] if lv["rows"] else None, "history": lv["rows"][-60:],
                     "caveat": "唯一沒被看過的資料：開盤前封存、20 個交易日後才對帳。樣本少時數字波動很大"}}
