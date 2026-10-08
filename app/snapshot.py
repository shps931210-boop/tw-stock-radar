"""三層選股：① 基本門檻排除 → ② 策略條件篩選 → ③ 多因子評分排名，結果存成 data/snapshot*.json。"""
from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import ai_model, config, db, metrics
from .catalog import C as CATALOG
from .strategies import STRATEGIES, passes

FACTORS = {
    "quality": [("roe", 1), ("roa", 1), ("op_margin", 1), ("ocf_to_assets", 1), ("debt_ratio", -1)],
    "growth": [("rev_yoy_3m", 1), ("rev_yoy", 1), ("eps_q_yoy", 1), ("op_q_yoy", 1)],
    "value": [("earnings_yield", 1), ("book_to_price", 1), ("fcf_yield", 1)],
}
FACTOR_LABEL = {"quality": "財務品質", "growth": "成長能力", "value": "估值合理性", "momentum": "股價動能", "trend": "趨勢確認"}


def grade(score, value_style=False):
    if score is None:
        return "資料不足"
    if value_style:
        return "便宜" if score >= 70 else "合理" if score >= 40 else "偏貴"
    return "優秀" if score >= 80 else "良好" if score >= 60 else "普通" if score >= 40 else "偏弱"


def verdict(score, coverage):
    if score is None or coverage < 3:
        return "資料不足"
    return "值得優先研究" if score >= 75 else "值得進一步研究" if score >= 60 else "中性觀察" if score >= 45 else "條件不足"


def _pctrank(s: pd.Series, mask: pd.Series, sign: int = 1) -> pd.Series:
    x = (s * sign).where(mask)
    return x.rank(pct=True) * 100


def build(demo: bool = False) -> dict:
    cfg = config.load()
    dbname = "demo.db" if demo else "stocks.db"
    df = metrics.compute_all(dbname)
    bf = cfg["base_filter"]

    # ---------- 第一層：基本門檻 ----------
    reasons = []
    for _, r in df.iterrows():
        why = []
        if not r["is_common"]:
            why.append("非普通股（ETF 等）")
        if (r.get("avg_value_20") or 0) < bf["min_avg_value_20"]:
            why.append("成交金額過低")
        if (r.get("history_days") or 0) < bf["min_history_days"]:
            why.append("交易日數不足")
        if bf.get("exclude_long_term_loss") and r.get("long_term_loss") is True:
            why.append("長期虧損")
        if bf.get("exclude_disposition") and r.get("disposition"):
            why.append("處置股票")
        reasons.append(why)
    df["exclude_reasons"] = reasons
    df["base_ok"] = df["exclude_reasons"].map(len) == 0
    base = df["base_ok"]

    # ---------- 第三層：橫斷面百分位與多因子評分 ----------
    df["rs_rank"] = _pctrank(df.get("mom_12_1"), base) if "mom_12_1" in df else None
    df["vol_1y_pct"] = _pctrank(df.get("vol_1y"), base) if "vol_1y" in df else None
    nonfin = base & ~df["is_financial"]
    for f, parts in FACTORS.items():
        cols = []
        for k, sign in parts:
            if k in df:
                mask = nonfin if k in ("debt_ratio", "op_margin") else base
                cols.append(_pctrank(df[k], mask & df[k].notna(), sign))
        df[f"score_{f}"] = pd.concat(cols, axis=1).mean(axis=1, skipna=True) if cols else np.nan
    df["score_momentum"] = df["rs_rank"]
    chip_cols = [_pctrank(df[k], base & df[k].notna(), sg) for k, sg in (("inst_ratio_5d", 1), ("margin_chg_20d", -1)) if k in df]
    df["score_chips"] = pd.concat(chip_cols, axis=1).mean(axis=1, skipna=True) if chip_cols else np.nan
    tchecks = []
    for a, b in (("close", "ma20"), ("ma20", "ma60"), ("ma60", "ma120"), ("close", "ma240")):
        if a in df and b in df:
            tchecks.append((df[a] > df[b]).where(df[b].notna()))
    if "ma60_slope_20" in df:
        tchecks.append((df["ma60_slope_20"] > 0).where(df["ma60_slope_20"].notna()))
    df["score_trend"] = (pd.concat(tchecks, axis=1).astype(float).mean(axis=1) * 100).where(base) if tchecks else np.nan
    w = cfg["composite_weights"]
    S = pd.DataFrame({k: df[f"score_{k}"] for k in w})
    W = pd.DataFrame({k: np.where(S[k].notna(), v, 0.0) for k, v in w.items()})
    df["coverage"] = S.notna().sum(axis=1)
    # 資料完整性：五個因子都要有分數才列入綜合排名；缺的因子不當作 0 分，也不用其他因子硬補
    df["missing_factors"] = [[FACTOR_LABEL[k] for k in w if pd.isna(row[k])] for _, row in S.iterrows()]
    df["complete"] = df["missing_factors"].map(len) == 0
    df["composite"] = ((S.fillna(0) * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan)).where(base & df["complete"])
    if "earnings_yield" in df and "roa" in df:
        ok = base & ~df["is_financial"] & df["earnings_yield"].notna() & df["roa"].notna()
        combo = df["earnings_yield"].where(ok).rank(ascending=False) + df["roa"].where(ok).rank(ascending=False)
        df["magic_rank"] = combo.rank(ascending=False, pct=True) * 100

    rows = []
    for rec in df.to_dict("records"):
        rec = {k: _clean(v) for k, v in rec.items()}
        rec["grades"] = {f: grade(rec.get(f"score_{f}"), f == "value") for f in w}
        rec["verdict"] = (verdict(rec.get("composite"), rec.get("coverage") or 0) if rec["complete"] or not rec["base_ok"]
                          else "資料不足，暫不列入完整綜合排名")
        rec["strategies"] = [s["id"] for s in STRATEGIES if passes(s, rec)]
        rows.append(rec)
    ai_model.annotate(rows)
    ai_model.timing(rows, cfg.get("buy_point", {}))

    with db.connect(dbname) as con:
        meta = {k: db.get_meta(con, k) for k in ("last_update", "source", "last_errors", "updated_prices",
                                                 "updated_revenue", "updated_financials", "update_state", "official_latest_twse")}
        hist_days = con.execute("SELECT COUNT(DISTINCT date) FROM prices").fetchone()[0]
        idx = db.read(con, "SELECT * FROM index_prices ORDER BY date DESC LIMIT 260")
        health = [] if demo else _health(con)
    common = [r for r in rows if r["is_common"]]
    chg = [r.get("chg_pct") for r in common if r.get("chg_pct") is not None]
    ind = pd.DataFrame([{"industry": r["industry"], "chg": r.get("chg_pct"), "ret_1m": r.get("ret_1m")} for r in common if r.get("industry")])
    industries = []
    if len(ind):
        g = ind.groupby("industry").agg(n=("chg", "size"), chg=("chg", "mean"), ret_1m=("ret_1m", "mean")).reset_index()
        g = g[g["n"] >= 2].sort_values("ret_1m", ascending=False)
        industries = [{k: _clean(v) for k, v in x.items()} for x in g.to_dict("records")]
    out = {
        "generated_at": config.now().isoformat(timespec="seconds"),
        "source": meta["source"] or ("demo" if demo else "live"),
        "meta": meta,
        "health": health,
        "history": {"days": hist_days, "need": bf["min_history_days"], "ready": hist_days >= bf["min_history_days"]}, "expected_date": None if demo else (meta["official_latest_twse"] or _expected_date()),
        "expected_basis": None if demo else ("official" if meta["official_latest_twse"] else "estimate"),
        "data_dates": {
            "prices": max((r.get("date") or "" for r in rows), default=None),
            "revenue": _mode([r.get("rev_ym") for r in common]),
            "financials": _mode([r.get("fin_period") for r in common]),
        },
        "market": {"taiex": _clean(idx["close"].iloc[0]) if len(idx) else None,
                   "taiex_chg": _clean((idx["close"].iloc[0] / idx["close"].iloc[1] - 1) * 100) if len(idx) > 1 else None},
        "breadth": {"up": sum(c > 0 for c in chg), "down": sum(c < 0 for c in chg), "flat": sum(c == 0 for c in chg)},
        "industries": industries,
        "index_series": [{"time": d, "value": _clean(v)} for d, v in zip(idx["date"][::-1], idx["close"][::-1])],
        "ai_modes": ai_model.MODES,
        "triggers": ai_model.TRIGGERS, "buy_point": ai_model.buy_rule(cfg.get("buy_point", {})),
        "universe": {"total": len(common), "base_ok": int(sum(r["base_ok"] for r in common))},
        "weights": w, "factor_label": FACTOR_LABEL, "base_filter": bf,
        "strategies": [{**s, "count": sum(s["id"] in r["strategies"] for r in rows)} for s in STRATEGIES],
        "catalog": CATALOG,
        "stocks": rows,
    }
    name = "snapshot_demo.json" if demo else "snapshot.json"
    (config.DATA_DIR / name).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    return out


HEALTH = [("twse_quotes", "上市行情"), ("tpex_quotes", "上櫃行情"), ("twse_inst", "上市三大法人"), ("tpex_inst", "上櫃三大法人"),
          ("twse_margin", "上市融資融券"), ("tpex_margin", "上櫃融資融券"), ("twse_val", "上市本益比"), ("tpex_val", "上櫃本益比")]


def _health(con) -> list[dict]:
    """每個每日資料來源：最後成功取得的資料日、最後嘗試時間、是否動用 OpenAPI 備援。"""
    out = []
    for tag, label in HEALTH:
        ok = con.execute("SELECT MAX(key) FROM fetch_log WHERE source=? AND ok=1", (tag,)).fetchone()[0]
        tried = con.execute("SELECT MAX(fetched_at) FROM fetch_log WHERE source=?", (tag,)).fetchone()[0]
        out.append({"source": tag, "label": label, "last_ok": ok, "last_try": tried})
    return out


def _expected_date() -> str:
    """連不到 OpenAPI、不知道官方公布到哪天時的推估：平日 15:00 後算當天（不含國定假日，假日隔天可能誤報）。"""
    from datetime import date, timedelta
    now = config.now()
    d = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _mode(xs):
    xs = [x for x in xs if x]
    return max(set(xs), key=xs.count) if xs else None


def _clean(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return float(f"{x:.6g}")
    return x
