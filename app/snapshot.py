"""三層選股：① 基本門檻排除 → ② 策略條件篩選 → ③ 多因子評分排名，結果存成 data/snapshot*.json。"""
from __future__ import annotations

import json
import re
import math
from datetime import datetime

import numpy as np
import pandas as pd

from . import ai_model, backtest, config, db, events, liquidity, market_board, metrics, sectors, track
from .catalog import C as CATALOG
from .strategies import STRATEGIES, passes

FACTORS = {
    "quality": [("roe", 1), ("roa", 1), ("op_margin", 1), ("ocf_to_assets", 1), ("debt_ratio", -1)],
    "growth": [("rev_yoy_3m", 1), ("rev_yoy", 1), ("eps_q_yoy", 1), ("op_q_yoy", 1)],
    "value": [("earnings_yield", 1), ("book_to_price", 1), ("fcf_yield", 1)],
}
# 每個因子的必要欄位與最少可用項目數：只有一兩項資料剛好很高的公司，不能被當成完整的高分
REQUIRED = {
    "quality": (["roe", "roa"], 3),
    "growth": (["rev_yoy", "rev_yoy_3m"], 3),
    "value": (["earnings_yield", "book_to_price"], 2),
}
TREND_MIN = 4  # 五項趨勢檢查至少要有四項能算
ITEM_LABEL = {"roe": "ROE", "roa": "ROA", "op_margin": "營業利益率", "ocf_to_assets": "營業現金流", "debt_ratio": "負債比",
              "rev_yoy_3m": "近三月營收", "rev_yoy": "月營收年增", "eps_q_yoy": "EPS 年增", "op_q_yoy": "營業利益年增",
              "earnings_yield": "本益比", "book_to_price": "股價淨值比", "fcf_yield": "自由現金流", "mom_12_1": "一年股價"}
FIN_COLS = ["roe", "roa", "op_margin", "ocf_to_assets", "debt_ratio", "eps_q_yoy", "op_q_yoy", "fcf_yield"]
REV_COLS = ["rev_yoy", "rev_yoy_3m"]
VAL_COLS = ["earnings_yield", "book_to_price", "fcf_yield"]
VAL_MAX_LAG = 3  # 本益比資料最多比股價晚幾個交易日


def expected_fin_end(today) -> str:
    """依法定公告期限（Q1 5/15、Q2 8/14、Q3 11/14、年報 3/31），加 10 天緩衝，推算現在至少應該有哪一季的財報。"""
    from datetime import date, timedelta
    y = today.year
    cands = [(date(y - 1, 9, 30), date(y - 1, 11, 14)), (date(y - 1, 12, 31), date(y, 3, 31)),
             (date(y, 3, 31), date(y, 5, 15)), (date(y, 6, 30), date(y, 8, 14)), (date(y, 9, 30), date(y, 11, 14))]
    ok = [end for end, due in cands if due + timedelta(days=10) <= today]
    return max(ok).isoformat()


def expected_rev_ym(today) -> str:
    """月營收在次月 10 日前公告，加 5 天緩衝：15 日以後應該有上個月，之前至少要有前兩個月。"""
    p = pd.Period(today, "M") - (1 if today.day > 15 else 2)
    return str(p)
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


def _pct_vs(x: pd.Series, ref: pd.Series) -> pd.Series:
    """x 在參考分布 ref 裡的百分位（0～100，同值取中間）。"""
    r = np.sort(ref.dropna().to_numpy(float))
    if not len(r):
        return pd.Series(np.nan, index=x.index)
    v = x.to_numpy(float)
    p = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / 2 / len(r) * 100
    return pd.Series(np.where(np.isnan(v), np.nan, p), index=x.index)


def _ref_scores(df: pd.DataFrame, S_in: pd.DataFrame, base: pd.Series) -> list:
    """個股研究用的「參考分項分數」：沒通過基本門檻（成交金額過低、交易日數不足等）的普通股，
    拿自己的指標和「通過門檻的股票」比，算出各因子在全市場的相對位置。
    必要欄位規則和正式分數相同，缺資料的因子不給分；這些分數不進綜合分數、不進排行榜。
    通過門檻的股票直接用正式分數，不另外算。"""
    out = [None] * len(df)
    cand = df["is_common"].astype(bool) & ~base
    if not cand.any():
        return out
    fin = df["is_financial"].astype(bool)
    res = {}
    for f, parts in FACTORS.items():
        cols, have = [], []
        for k, sign in parts:
            x = S_in[k] * sign
            applies = ~fin if k in ("debt_ratio", "op_margin") else pd.Series(True, index=df.index)
            if f in ("quality", "value"):
                v = pd.concat([_pct_vs(x.where(g), x.where(base & g)) for g in (fin, ~fin)], axis=1).bfill(axis=1).iloc[:, 0]
            else:
                v = _pct_vs(x, x.where(base & applies))
            cols.append(v.where(applies))
            have.append((S_in[k].notna() & applies).rename(k))
        H = pd.concat(have, axis=1)
        req, need = REQUIRED[f]
        ok = H[req].all(axis=1) & (H.sum(axis=1) >= need)
        res[f] = pd.concat(cols, axis=1).mean(axis=1, skipna=True).where(ok)
    if "mom_12_1" in df:
        res["momentum"] = _pct_vs(df["mom_12_1"], df["mom_12_1"].where(base))
    tchecks = []
    for a, b in (("close", "ma20"), ("ma20", "ma60"), ("ma60", "ma120"), ("close", "ma240")):
        if a in df and b in df:
            tchecks.append((df[a] > df[b]).where(df[b].notna()))
    if "ma60_slope_20" in df:
        tchecks.append((df["ma60_slope_20"] > 0).where(df["ma60_slope_20"].notna()))
    if tchecks:
        T = pd.concat(tchecks, axis=1).astype(float)
        res["trend"] = (T.mean(axis=1) * 100).where(T.notna().sum(axis=1) >= TREND_MIN)
    for i in np.flatnonzero(cand.to_numpy()):
        d = {f: round(float(v.iloc[i]), 1) for f, v in res.items() if pd.notna(v.iloc[i])}
        out[i] = d or None
    return out


TURN_COL = {"eps_q_yoy": "eps_q_turn", "op_q_yoy": "op_q_turn"}


def _gap_label(df: pd.DataFrame, k: str, i: int) -> tuple[str, str]:
    """缺漏項目的種類：na＝去年同期虧損、年增率不適用；stale＝資料過期；miss＝真的沒有資料。"""
    lab = ITEM_LABEL[k]
    t = df[TURN_COL[k]].iloc[i] if k in TURN_COL and TURN_COL[k] in df else None
    if isinstance(t, str) and t:
        return "na", lab
    if k == "earnings_yield" and "eps_ttm" in df and pd.notna(df["eps_ttm"].iloc[i]) and df["eps_ttm"].iloc[i] <= 0:
        return "loss", lab
    if k in df and pd.notna(df[k].iloc[i]):
        return "stale", lab
    return "miss", lab


def _gap_text(gs: list) -> str:
    """把缺漏項目寫成一句話，例如「缺近三月營收，因去年同期虧損，EPS 年增不適用」。"""
    by = {kind: [lab for k2, lab in gs if k2 == kind] for kind in ("miss", "stale", "na", "loss")}
    seg = (["缺" + "、".join(by["miss"])] if by["miss"] else []) + (["、".join(by["stale"]) + "資料過期"] if by["stale"] else []) + \
          (["因去年同期虧損，" + "、".join(by["na"]) + "不適用"] if by["na"] else []) + (["近四季虧損，本益比不適用"] if by["loss"] else [])
    return "，".join(seg) if seg else "缺必要資料"


def score_frame(df: pd.DataFrame, cfg: dict, today) -> tuple[pd.DataFrame, dict]:
    """多因子評分（基本門檻、因子百分位、綜合分數）。today：判斷財報、營收是否過期的基準日（歷史回測傳當時的日期）。"""
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
    # ---------- 資料新鮮度：過期的財報、營收、本益比不能當成最新資料計分 ----------
    fin_exp, rev_exp = expected_fin_end(today), expected_rev_ym(today)
    older = lambda col, exp: df[col].map(lambda x: isinstance(x, str) and x < exp) if col in df else False
    df["fin_stale"] = older("fin_period_end", fin_exp)
    df["rev_stale"] = older("rev_ym", rev_exp)
    if "val_date" in df:
        pdates = sorted(set(df["date"].dropna()))
        lag = df["val_date"].map(lambda d: len(pdates) - int(np.searchsorted(pdates, d, side="right")) if isinstance(d, str) else None)
        df["val_stale"] = lag.notna() & (lag > VAL_MAX_LAG)
    else:
        df["val_stale"] = False
    S_in = df.copy()  # 計分用：過期欄位當作缺資料
    for flag, cols in (("fin_stale", FIN_COLS), ("rev_stale", REV_COLS), ("val_stale", VAL_COLS)):
        for c in cols:
            if c in S_in:
                S_in.loc[S_in[flag].astype(bool), c] = np.nan

    # ---------- 第三層：橫斷面百分位與多因子評分 ----------
    df["rs_rank"] = _pctrank(df.get("mom_12_1"), base) if "mom_12_1" in df else None
    df["vol_1y_pct"] = _pctrank(df.get("vol_1y"), base) if "vol_1y" in df else None
    nonfin = base & ~df["is_financial"]
    gaps = {f: [[] for _ in range(len(df))] for f in list(FACTORS) + ["momentum", "trend"]}
    for f, parts in FACTORS.items():
        cols, have = [], []
        for k, sign in parts:
            if k not in S_in:
                S_in[k] = np.nan
            applies = nonfin if k in ("debt_ratio", "op_margin") else base  # 金融業不比負債比、營業利益率
            # 品質與估值：金融業只和金融業比、其他產業和非金融業比（銀行的 ROA、股價淨值比和一般公司差很多）
            if f in ("quality", "value"):
                fin = df["is_financial"].astype(bool)
                cols.append(pd.concat([_pctrank(S_in[k], applies & S_in[k].notna() & g, sign) for g in (fin, ~fin)], axis=1)
                            .bfill(axis=1).iloc[:, 0])
            else:
                cols.append(_pctrank(S_in[k], applies & S_in[k].notna(), sign))
            have.append((S_in[k].notna() & applies).rename(k))
        H = pd.concat(have, axis=1)
        req, need = REQUIRED[f]
        ok = H[req].all(axis=1) & (H.sum(axis=1) >= need)
        df[f"score_{f}"] = pd.concat(cols, axis=1).mean(axis=1, skipna=True).where(ok)
        # 缺漏說明看「資料本身」有沒有（不看是否通過排行榜門檻），並分清楚是沒資料、資料過期，還是去年同期虧損而不適用
        fin_all = df["is_financial"].astype(bool)
        H_any = pd.concat([(S_in[k].notna() & ~(fin_all & (k in ("debt_ratio", "op_margin")))).rename(k) for k, _ in parts], axis=1)
        ok_any = H_any[req].all(axis=1) & (H_any.sum(axis=1) >= need)
        for i in np.flatnonzero(~ok_any.to_numpy()):
            fin = bool(fin_all.iloc[i])
            gaps[f][i] = [_gap_label(df, k, i) for k, _ in parts if not H_any[k].iloc[i]
                          and not (fin and k in ("debt_ratio", "op_margin"))][:3]
    df["score_momentum"] = df["rs_rank"]
    if "mom_12_1" in df:
        for i, v in enumerate(df["mom_12_1"]):
            if pd.isna(v):
                gaps["momentum"][i] = ["一年股價"]
    chip_cols = [_pctrank(df[k], base & df[k].notna(), sg) for k, sg in (("inst_ratio_5d", 1), ("margin_chg_20d", -1)) if k in df]
    df["score_chips"] = pd.concat(chip_cols, axis=1).mean(axis=1, skipna=True) if chip_cols else np.nan
    tchecks = []
    for a, b in (("close", "ma20"), ("ma20", "ma60"), ("ma60", "ma120"), ("close", "ma240")):
        if a in df and b in df:
            tchecks.append((df[a] > df[b]).where(df[b].notna()))
    if "ma60_slope_20" in df:
        tchecks.append((df["ma60_slope_20"] > 0).where(df["ma60_slope_20"].notna()))
    if tchecks:
        T = pd.concat(tchecks, axis=1).astype(float)
        df["score_trend"] = (T.mean(axis=1) * 100).where(base & (T.notna().sum(axis=1) >= TREND_MIN))
        for i, n in enumerate(T.notna().sum(axis=1)):
            if n < TREND_MIN:
                gaps["trend"][i] = ["歷史股價天數"]
    else:
        df["score_trend"] = np.nan
    w = cfg["composite_weights"]
    S = pd.DataFrame({k: df[f"score_{k}"] for k in w})
    W = pd.DataFrame({k: np.where(S[k].notna(), v, 0.0) for k, v in w.items()})
    df["coverage"] = S.notna().sum(axis=1)
    # 資料完整度：五個因子內所有子項目中，有幾成拿得到「未過期」的資料
    items = [k for parts in FACTORS.values() for k, _ in parts] + ["mom_12_1"]
    df["data_completeness"] = (pd.concat([S_in[k].notna() if k in S_in else pd.Series(False, index=df.index) for k in items], axis=1)
                               .mean(axis=1) * 100).round(0)
    # 資料完整性：五個因子都要通過必要欄位檢查才列入綜合排名；缺的因子不當作 0 分，也不用其他因子硬補
    df["missing_factors"] = [[FACTOR_LABEL[k] for k in w if pd.isna(row[k])] for _, row in S.iterrows()]
    stale_note = []
    for i, r in df.iterrows():
        n = []
        if r.get("fin_stale"):
            n.append(f"財報停在 {r.get('fin_period')}（應已公布到 {fin_exp[:7]}）")
        if r.get("rev_stale"):
            n.append(f"月營收停在 {r.get('rev_ym')}")
        if r.get("val_stale"):
            n.append(f"本益比資料停在 {r.get('val_date')}")
        stale_note.append(n)
    df["stale_notes"] = stale_note
    pos = {k: i for i, k in enumerate(df.index)}
    df["missing_detail"] = [[f"{FACTOR_LABEL[k]}（{_gap_text(gaps[k][pos[ix]]) if isinstance(gaps[k][pos[ix]][0], tuple) else '缺' + '、'.join(gaps[k][pos[ix]])}）" if gaps[k][pos[ix]] else
                             f"{FACTOR_LABEL[k]}（資料齊全，但未通過排行榜基本門檻，不計正式分數）" if not base.loc[ix] else
                             f"{FACTOR_LABEL[k]}（缺必要資料）" for k in w if pd.isna(S.loc[ix, k])]
                            for ix in df.index]
    df["complete"] = df["missing_factors"].map(len) == 0
    df["composite"] = ((S.fillna(0) * W).sum(axis=1) / W.sum(axis=1).replace(0, np.nan)).where(base & df["complete"])
    # 產業內排名：同產業至少 5 檔有綜合分數才排
    df["industry_rank"] = df.groupby("industry")["composite"].rank(ascending=False, method="min")
    df["industry_n"] = df.groupby("industry")["composite"].transform("count")
    small = (df["industry_n"] < 5) | (df["industry"].fillna("") == "")
    df.loc[small, ["industry_rank", "industry_n"]] = np.nan
    df["market_rank_pct"] = df["composite"].rank(pct=True) * 100
    df["ref_scores"] = _ref_scores(df, S_in, base)
    if "earnings_yield" in df and "roa" in df:
        ok = base & ~df["is_financial"] & df["earnings_yield"].notna() & df["roa"].notna()
        combo = df["earnings_yield"].where(ok).rank(ascending=False) + df["roa"].where(ok).rank(ascending=False)
        df["magic_rank"] = combo.rank(ascending=False, pct=True) * 100
    return df, w



INDEX_NAMES = {"櫃買指數": "上櫃指數", "電子類指數": "電子類股指數", "金融保險類指數": "金融類股指數", "半導體類指數": "半導體類股指數"}


def _sector_indices(con) -> list[dict]:
    """首頁「主要市場指標」：每個指數最近兩筆收盤算漲跌。沒有資料的指數照樣列出、值為 None，網站顯示「資料源確認中」。"""
    out = []
    for name, label in INDEX_NAMES.items():
        try:
            r = con.execute("SELECT date, close FROM sector_index WHERE name=? ORDER BY date DESC LIMIT 2", (name,)).fetchall()
        except Exception:
            r = []
        it = {"name": label, "date": r[0][0] if r else None, "close": _clean(r[0][1]) if r else None, "chg": None, "chg_pct": None}
        if len(r) > 1 and r[1][1]:
            it["chg"] = _clean(r[0][1] - r[1][1])
            it["chg_pct"] = _clean((r[0][1] / r[1][1] - 1) * 100)
        out.append(it)
    return out

def build(demo: bool = False) -> dict:
    cfg = config.load()
    dbname = "demo.db" if demo else "stocks.db"
    df = metrics.compute_all(dbname)
    bf = cfg["base_filter"]
    df, w = score_frame(df, cfg, config.today())
    bf = cfg["base_filter"]

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
    liquidity.annotate(rows, ai_model.MODES)  # 推薦資格另外存；原始分數與名次不動
    with db.connect(dbname) as con:
        fm_done = {k for (k,) in con.execute("SELECT key FROM fetch_log WHERE source='fm_financials' AND ok=1")} if not demo else None
        try:
            fm_queue = json.loads(db.get_meta(con, "fundamental_queue") or "null")
        except ValueError:
            fm_queue = None
        ann = None
        if not demo:  # 官方已公告、資料庫還沒有的新一季財報（app/fin_watch.py）
            from . import fin_watch
            fin_watch.ensure(con)
            ann = dict(con.execute("SELECT stock_id, MAX(period) FROM fin_announce WHERE status IN ('pending','waiting_source') GROUP BY stock_id").fetchall())
    decision = _decision(rows, max((r.get("date") or "" for r in rows), default=""), fm_done, fm_queue, ann)

    with db.connect(dbname) as con:
        official = None if demo else db.get_meta(con, "official_latest_twse")
        adj_mkts = [] if demo else [m for m in ("twse", "tpex") if con.execute(
            "SELECT 1 FROM fetch_log WHERE source=? AND ok=1", (f"{m}_exrights",)).fetchone()]
    # 發布阻擋：股價明確落後官方公布日時，不產生新的 AI 排名與技術訊號（避免用舊行情當成今天的建議）
    price_day = max((r.get("date") or "" for r in rows), default="")
    blocked = None
    if not demo:
        exp = official or _expected_date()
        behind = _weekdays_between(price_day, exp)
        if (official and behind >= 1) or behind >= 2:
            blocked = f"股價資料停在 {price_day}，{'官方已公布到' if official else '推估應有'} {exp}，暫停產生 AI 排名與技術訊號"
            for r in rows:
                for v in list(r["ai"]["modes"].values()) + list(r["ai"]["rec"].values()):
                    v["rank"] = None
                r["timing"].update(status="none", blocked=True)
                r["composite_blocked"] = True

    with db.connect(dbname) as con:
        meta = {k: db.get_meta(con, k) for k in ("last_update", "source", "last_errors", "updated_prices",
                                                 "updated_revenue", "updated_financials", "update_state", "official_latest_twse", "update_mode")}
        hist_days = con.execute("SELECT COUNT(DISTINCT date) FROM prices").fetchone()[0]
        idx = db.read(con, "SELECT * FROM index_prices ORDER BY date DESC LIMIT 260")
        indices = _sector_indices(con)
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
        "rankings_blocked": blocked,
        "model": {"version": ai_model.MODEL_VERSION, "changelog": ai_model.CHANGELOG},
        "backtest": _backtest(dbname, price_day),
        "track": {"days": 0, "logs": [], "scorecard": {}} if demo else track.performance(dbname),
        "adjusted_markets": adj_mkts,
        "freshness_rule": {"fin_expected": expected_fin_end(config.today()), "rev_expected": expected_rev_ym(config.today()),
                           "chip_max_lag": metrics.CHIP_MAX_LAG, "val_max_lag": VAL_MAX_LAG},
        "data_dates": {
            "prices": max((r.get("date") or "" for r in rows), default=None),
            "revenue": _mode([r.get("rev_ym") for r in common]),
            "financials": _mode([r.get("fin_period") for r in common]),
        },
        "market": {"taiex": _clean(idx["close"].iloc[0]) if len(idx) else None,
                   "taiex_chg": _clean((idx["close"].iloc[0] / idx["close"].iloc[1] - 1) * 100) if len(idx) > 1 else None,
                   "taiex_pts": _clean(idx["close"].iloc[0] - idx["close"].iloc[1]) if len(idx) > 1 else None,
                   "taiex_date": idx["date"].iloc[0] if len(idx) else None,
                   "indices": indices},
        "breadth": {"up": sum(c > 0 for c in chg), "down": sum(c < 0 for c in chg), "flat": sum(c == 0 for c in chg)},
        "industries": industries,
        "index_series": [{"time": d, "value": _clean(v)} for d, v in zip(idx["date"][::-1], idx["close"][::-1])],
        "ai_modes": ai_model.MODES,
        "liquidity_rule": {"grades": [[g, v] for g, v in liquidity.GRADES], "variant": liquidity.LIVE, "desc": liquidity.VARIANTS[liquidity.LIVE],
                           "top": liquidity.TOP, "level": liquidity.LEVEL},
        "triggers": ai_model.TRIGGERS, "buy_point": ai_model.buy_rule(cfg.get("buy_point", {})),
        "universe": {"total": len(common), "base_ok": int(sum(r["base_ok"] for r in common))},
        "weights": w, "factor_label": FACTOR_LABEL, "base_filter": bf,
        "strategies": [{**s, "count": sum(s["id"] in r["strategies"] for r in rows)} for s in STRATEGIES],
        "catalog": CATALOG,
        "stocks": [r for r in rows if r["is_common"] or re.fullmatch(r"00\d{2,4}[A-Z]?", r["stock_id"])],
    }
    out["changes"] = None if demo else track.changes(out)
    try:
        out["prev_scores"] = prev_scores(out, dbname)
    except Exception as e:  # 分數變化失敗不影響選股結果
        out["prev_scores"] = {"date": None, "warnings": [f"分數變化計算失敗：{e}"]}
    try:
        out["sectors"] = sectors.build(out, dbname)
    except Exception as e:  # 產業熱力圖失敗不影響選股結果
        out["sectors"] = {"schema_version": 1, "industries": [], "warnings": [f"產業熱力圖計算失敗：{e}"]}
    try:
        out["market_board"] = market_board.build(out, dbname)
    except Exception as e:  # 市場看板失敗不影響選股結果
        out["market_board"] = {"schema_version": 1, "markets": {}, "warnings": [f"市場統計計算失敗：{e}"]}
    try:
        out["calendar"] = events.calendar(out, dbname)
    except Exception as e:  # 日曆失敗不影響其他功能
        out["calendar"] = {"schema_version": 1, "events": [], "warnings": [f"財報與事件日曆計算失敗：{e}"]}
    try:  # 進出場策略回測（同一個資料日只算一次）
        out["strategy_bt"] = _strategy_bt(dbname, price_day)
        from . import strategy
        out["strategy_rules"] = {**{k: {x: y for x, y in v.items() if isinstance(y, str)} for k, v in strategy.RULES.items()}, "long": strategy.LONG_RULE}
    except Exception as e:
        out["strategy_bt"] = {"ready": False, "reason": f"策略回測失敗：{e}"}
    if blocked:
        decision = {sid: {**d, "blocked": blocked} for sid, d in decision.items()}
        for r in rows:
            if r.get("plan"):
                r["plan"].update(short="blocked", swing="blocked")
    name = "snapshot_demo.json" if demo else "snapshot.json"
    (config.DATA_DIR / name).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    (config.DATA_DIR / ("decision_demo.json" if demo else "decision.json")).write_text(json.dumps(decision, ensure_ascii=False, separators=(",", ":")), "utf-8")
    try:  # 個股研究資料覆蓋率（各模組狀態、缺漏原因排行）
        from . import coverage
        cov = {**coverage.build(rows, decision), "generated_at": out.get("generated_at"), "prices": price_day, "fundamental_queue": fm_queue}
        (config.DATA_DIR / ("coverage_demo.json" if demo else "coverage.json")).write_text(json.dumps(cov, ensure_ascii=False, separators=(",", ":")), "utf-8")
    except Exception as e:
        print("覆蓋率報告失敗：", e)
    if not demo:
        try:  # V19：新財報同步後的改分紀錄、全市場資料完整性稽核
            from . import coverage, fin_watch
            with db.connect(dbname) as con:
                fin_watch.after_snapshot(con)
                (config.DATA_DIR / "coverage_audit.json").write_text(json.dumps(coverage.audit(con, rows, decision), ensure_ascii=False, separators=(",", ":")), "utf-8")
        except Exception as e:
            print("財報偵測紀錄或資料完整性稽核失敗：", e)
    return out


# 研究模組的資料狀態（個股研究卡分開顯示，不因一項缺資料就整張卡失效）
STATUS_TXT = {"ready": "完成", "partial": "部分完成", "insufficient_history": "歷史不足", "missing_source": "尚未取得資料",
              "stale": "資料過期", "not_applicable": "不適用", "calculation_error": "計算異常"}


def _valuation_fill(r: dict, vb: dict | None) -> dict:
    """自己算不出歷史本益比區間時，仍提供交易所公布的目前本益比、股價淨值比、殖利率與同業中位數（標明來源）。"""
    v = dict(vb) if isinstance(vb, dict) else {"status": "missing_source", "why": "沒有估值計算結果"}
    v.setdefault("status", "ready" if v.get("p50") is not None else "missing_source")
    if v["status"] != "ready":
        per, pbr, dy = r.get("per"), r.get("pbr"), r.get("dy")
        if v.get("pe") is None and per and per > 0 and v["status"] != "not_applicable":
            v["pe"], v["pe_src"] = per, "official"
        v.update({k: x for k, x in (("pbr", pbr), ("dy", dy)) if x is not None and x > 0})
        v["val_date"] = r.get("val_date")
        if v["status"] == "missing_source" and (v.get("pe") or v.get("pbr")):
            v["status"] = "partial"
    return v


def _queue_txt(q: dict | None) -> str:
    """財報補齊佇列的實際狀態（來自上一次完整更新的紀錄，不是前端推測）。"""
    if not q or "done" not in q:
        return "（已排入補齊佇列，夜間依成交金額優先補抓）"
    when = (q.get("finished_at") or q.get("run_at") or "")[:16].replace("T", " ")
    tail = "這一小時的免費額度用完，夜間下一次補抓（22:30～06:30 每小時）接著補" if q.get("quota_hit") else "夜間補抓（22:30～06:30 每小時）會繼續"
    left = f"，還有 {q['never_fin']} 檔沒有財報" if q.get("never_fin") is not None else ""
    return f"（財報補齊中：{when} 補了 {q['done']} 筆{left}；{tail}）"


def _modules(r: dict, st: dict, v: dict, lg: dict, price_day: str, fm_done: set | None, fm_queue: dict | None = None,
             ann: dict | None = None) -> dict:
    """行情、基本面、估值、短線、波段、長期、AI 評分，各自的狀態與原因。"""
    M = {}
    d = r.get("date")
    M["quote"] = ({"status": "ready", "asof": d} if d and d >= price_day else
                  {"status": "stale", "asof": d, "why": f"最後成交 {d}，可能停牌或暫停交易"} if d else
                  {"status": "missing_source", "why": "沒有收盤行情"})
    fin, rev = r.get("fin_period"), r.get("rev_ym")
    why = []
    if not fin:
        why.append("尚未取得季報" + (_queue_txt(fm_queue) if fm_done is not None and r["stock_id"] not in fm_done else ""))
    elif r.get("fin_stale"):
        why.append(f"季報停在 {fin}（法定公告期限已過，應已有更新的季報）")
    a = (ann or {}).get(r["stock_id"])
    if a and (not fin or a > fin):
        why.append(f"{a[:4]} 年第 {a[-1]} 季財報已公告，資料來源尚未同步，夜間會自動補抓")
    if not rev:
        why.append("尚未取得月營收")
    elif r.get("rev_stale"):
        why.append(f"月營收停在 {rev}")
    M["fundamental"] = {"status": "ready" if not why or (fin and rev and not r.get("fin_stale") and not r.get("rev_stale")) else "stale" if fin and rev else "partial" if fin or rev else "missing_source",
                        "asof": " / ".join(x for x in (fin, rev) if x) or None, "why": "；".join(why) or None}
    M["valuation"] = {"status": v["status"], "why": v.get("why")}
    if st.get("ready"):
        M["short"] = {"status": "ready"}
        M["swing"] = {"status": (st.get("swing") or {}).get("status", "ready"), "why": ((st.get("swing") or {}).get("miss") or [None])[0]
                      if (st.get("swing") or {}).get("status") else None}
    else:
        M["short"] = M["swing"] = {"status": st.get("status", "calculation_error"), "why": st.get("why")}
    M["long"] = {"status": lg.get("status", "ready"), "why": lg.get("note") or ("；".join(lg.get("miss") or []) if lg.get("state") == "na" else None)}
    have = [f for f in SCORE_FACTORS if r.get(f"score_{f}") is not None] if r.get("base_ok") else list((r.get("ref_scores") or {}))
    if r.get("base_ok") and r.get("complete"):
        M["ai"] = {"status": "ready", "label": "正式排名"}
    else:
        miss = [FACTOR_LABEL[f] for f in SCORE_FACTORS if f not in have]
        ex = r.get("exclude_reasons") or []
        why = (f"未通過排行榜基本門檻（{'、'.join(ex)}），只提供參考分項分數" if ex else "五個因子沒有全部算得出來，不列入排行榜")
        M["ai"] = {"status": "partial" if have else "missing_source", "why": why + (f"；缺：{'、'.join(miss)}" if miss else "")}
        if have:
            M["ai"]["label"] = "參考分項，不排名"  # 和正式 AI 排名分開：沒有綜合分數、不進排行榜
    for m in M.values():
        m.setdefault("label", STATUS_TXT.get(m["status"], m["status"]))
    return M


def _decision(rows: list[dict], price_day: str = "", fm_done: set | None = None, fm_queue: dict | None = None, ann: dict | None = None) -> dict:
    """個股研究決策卡：進出場策略狀態、估值區間、長期判斷、各模組資料狀態。完整內容另存 data/decision.json（只放進個股 JSON），
    快照裡只留一行摘要（plan），避免整份快照變大。"""
    from . import strategy, valuation
    valuation.add_peers(rows)
    out = {}
    for r in rows:
        st, vb, ref = r.pop("strategy", None), r.pop("valuation_band", None), r.pop("ref_scores", None)
        if not r.get("is_common"):
            continue
        st = st if isinstance(st, dict) else {"ready": False, "status": "calculation_error", "why": "沒有策略計算結果"}
        vb = vb if isinstance(vb, dict) else None
        v = _valuation_fill(r, vb)
        lg = strategy.long_status({**r, "ref_scores": ref}, v)
        out[r["stock_id"]] = {"strategy": st, "long": lg, "valuation": v, "ref_scores": ref if isinstance(ref, dict) else None,
                              "modules": _modules({**r, "ref_scores": ref}, st, v, lg, price_day, fm_done, fm_queue, ann)}
        vb = v if v.get("p50") is not None else None
        tech = lambda m: (st.get(m) or {}).get("state", "na") if st.get("ready") else "na"
        r["plan"] = {"short": tech("short"), "swing": tech("swing"), "long": lg["state"],
                     "pe_pct": (vb or {}).get("pct"), "vs_base": round((r["close"] / vb["base"] - 1) * 100, 1) if vb and vb.get("base") and r.get("close") else None}
    return out


def _strategy_bt(dbname: str, price_day: str) -> dict:
    from . import strategy
    cache = config.DATA_DIR / f"strategy_bt_{dbname.split('.')[0]}.json"
    if cache.exists():
        old = json.loads(cache.read_text("utf-8"))
        if old.get("end") == price_day and old.get("method") == strategy.METHOD:
            return old
    res = strategy.backtest(dbname)
    cache.write_text(json.dumps(res, ensure_ascii=False), "utf-8")
    return res


SCORE_FACTORS = ("quality", "growth", "value", "momentum", "trend")


def prev_scores(snap: dict, dbname: str) -> dict:
    """上一個交易日的 AI 分數、名次與五個因子分數，給「和昨天相比」用。
    存在資料庫 meta：scores_today（這個交易日最新一次）、scores_prev（上一個交易日）。同一天重跑只更新 today，不會把今天當成昨天。"""
    day = (snap.get("data_dates") or {}).get("prices")
    cur = {"date": day, "s": {}}
    for r in snap["stocks"]:
        a = (r.get("ai") or {}).get("modes") or {}
        if not any((a.get(m) or {}).get("rank") for m in a):
            continue
        cur["s"][r["stock_id"]] = {"r": {m: v.get("rank") for m, v in a.items()}, "sc": {m: v.get("score") for m, v in a.items()},
                                   "f": {f: None if r.get(f"score_{f}") is None else round(r[f"score_{f}"], 1) for f in SCORE_FACTORS}}
    with db.connect(dbname) as con:
        today = json.loads(db.get_meta(con, "scores_today") or "null")
        prev = json.loads(db.get_meta(con, "scores_prev") or "null")
        if today and day and today.get("date") and today["date"] < day:
            prev = today
            db.set_meta(con, "scores_prev", json.dumps(prev, ensure_ascii=False, separators=(",", ":")))
        if day and cur["s"]:
            db.set_meta(con, "scores_today", json.dumps(cur, ensure_ascii=False, separators=(",", ":")))
    partial = False
    if not prev and day:  # 剛上線還沒有前一天的完整紀錄：先用每日存檔的前 20 名名單
        logs = [L for L in track.load(first_only=False) if L["date"] < day]
        if logs:
            last = max(logs, key=lambda L: (L["date"], L.get("version", 1)))
            st: dict = {}
            for m, picks in (last.get("picks") or {}).items():
                for p_ in picks:
                    x = st.setdefault(p_["stock_id"], {"r": {}, "sc": {}, "f": {k: (None if v is None else round(v, 1)) for k, v in (p_.get("factors") or {}).items()}})
                    x["r"][m], x["sc"][m] = p_.get("rank"), p_.get("score")
            prev, partial = {"date": last["date"], "s": st}, True
    if not prev or not day or prev.get("date") == day:
        return {"date": None, "stocks": {}}
    # 只送現在或之前有名次的股票
    keep = set(cur["s"]) | set(prev["s"])
    return {"date": prev["date"], "partial": partial, "stocks": {k: v for k, v in prev["s"].items() if k in keep}}


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


def _backtest(dbname: str, price_day: str) -> dict:
    """技術訊號歷史統計很花時間，同一個資料日只算一次（存在 data/backtest_*.json）。"""
    cache = config.DATA_DIR / f"backtest_{dbname.split('.')[0]}.json"
    if cache.exists():
        old = json.loads(cache.read_text("utf-8"))
        if old.get("end") == price_day and old.get("method") == backtest.METHOD:
            return old
    res = backtest.run(dbname)
    if res.get("ready"):
        cache.write_text(json.dumps(res, ensure_ascii=False), "utf-8")
    return res


def _weekdays_between(a: str, b: str) -> int:
    """a 之後到 b（含）有幾個平日；a >= b 時為 0。"""
    from datetime import date, timedelta
    if not a or not b or a >= b:
        return 0
    d, end, n = date.fromisoformat(a), date.fromisoformat(b), 0
    while d < end:
        d += timedelta(days=1)
        n += d.weekday() < 5
    return n


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
