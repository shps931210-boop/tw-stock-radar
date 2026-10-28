"""個股研究資料覆蓋率報告（data/coverage.json）。

每次重算快照時，統計所有普通股在各研究模組（行情、基本面、估值、短線、波段、長期、AI 評分）的資料狀態，
並把「資料不足」的原因整理成排行榜（依影響股票數排序），讓管理員知道該先補哪一種資料。
另外挑出八類代表股票（大型權值股、中小型上市、上櫃、新上市、金融、虧損、缺財報、歷史估值不足）當驗收樣本。
"""
from __future__ import annotations

import re
from collections import Counter

MODULES = {"quote": "行情", "fundamental": "基本面", "valuation": "估值", "short": "短線策略", "swing": "波段策略",
           "long": "長期判斷", "ai": "AI 評分"}
ORDER = ["ready", "partial", "insufficient_history", "stale", "not_applicable", "missing_source", "calculation_error"]


def _norm(why: str) -> str:
    """把原因裡的數字、日期、股票各自的值拿掉，相同原因才能合併計數。"""
    why = re.sub(r"（財報補齊中：.*?(會繼續|接著補)）", "（財報補齊中）", why)
    s = re.sub(r"\d{4}-\d{2}(-\d{2})?(-Q\d)?", "某期", why)
    s = re.sub(r"-?\d+(\.\d+)?", "N", s)
    return s.split("；")[0][:80]


def _pick(rows: list[dict], cond, key=None) -> dict | None:
    xs = [r for r in rows if cond(r)]
    if key:
        xs.sort(key=key)
    return xs[0] if xs else None


def build(rows: list[dict], decision: dict) -> dict:
    common = [r for r in rows if r.get("is_common") and r["stock_id"] in decision]
    n = len(common)
    mods = {}
    reasons = Counter()
    for k, name in MODULES.items():
        c = Counter((decision[r["stock_id"]].get("modules") or {}).get(k, {}).get("status", "missing_source") for r in common)
        mods[k] = {"name": name, "counts": {s: c.get(s, 0) for s in ORDER if c.get(s)},
                   "usable": round((c.get("ready", 0) + c.get("partial", 0)) / n * 100, 1) if n else None}
        for r in common:
            m = (decision[r["stock_id"]].get("modules") or {}).get(k) or {}
            if m.get("status") not in ("ready", None) and m.get("why"):
                reasons[(name, _norm(m["why"]))] += 1
    usable_any = sum(1 for r in common if all((decision[r["stock_id"]].get("modules") or {}).get(k, {}).get("status") != "missing_source"
                                              for k in ("quote",)))
    D = lambda r: decision[r["stock_id"]]
    st = lambda r, k: ((D(r).get("modules") or {}).get(k) or {}).get("status")
    by_cap = lambda r: -(r.get("market_cap") or 0)
    samples = {
        "大型權值股": _pick(common, lambda r: True, by_cap),
        "中小型上市股票": _pick(common, lambda r: r.get("market") == "twse" and (r.get("market_cap") or 0) < 1e10 and r.get("base_ok"), by_cap),
        "上櫃股票": _pick(common, lambda r: r.get("market") == "tpex" and r.get("base_ok"), by_cap),
        "新上市股票": _pick(common, lambda r: (r.get("history_days") or 0) < 120, lambda r: -(r.get("history_days") or 0)),
        "金融股": _pick(common, lambda r: r.get("is_financial"), by_cap),
        "EPS 為負": _pick(common, lambda r: (r.get("eps_ttm") or 0) < 0, by_cap),
        "財報缺漏": _pick(common, lambda r: not r.get("fin_period"), by_cap),
        "價格完整但歷史估值不足": _pick(common, lambda r: st(r, "valuation") == "insufficient_history", by_cap),
    }
    return {
        "schema_version": 1, "stocks": n, "usable_quote": usable_any,
        "modules": mods,
        "reasons": [{"module": m, "why": w, "stocks": c} for (m, w), c in reasons.most_common(25)],
        "samples": {k: None if r is None else {"stock_id": r["stock_id"], "name": r.get("name"),
                                               "status": {m: st(r, m) for m in MODULES}} for k, r in samples.items()},
    }


# ---------- V19 全市場資料完整性稽核（data/coverage_audit.json） ----------
# 直接查正式資料庫：每檔普通股的行情、月營收、季報、法人、融資融券各自的起迄日期與筆數，
# 加上快照算出的估值、AI 評分、正式推薦資格，整理成全市場統計、缺口排行榜與逐檔明細。
# 「不符合正式推薦資格」和「沒有資料」分開統計：資料齊全但成交金額太低的股票，研究資訊照常可用。

STATUS8 = {"ready": "完整可用", "partial": "部分可用", "insufficient_history": "歷史不足", "stale": "資料過期",
           "missing_source": "資料來源缺失", "not_applicable": "該指標不適用", "calculation_error": "計算錯誤"}
PX_DAYS = (120, 240, 500, 1000)
FIN_YEARS = (3, 5, 8)


def _span(con, sql: str) -> dict:
    return {r[0]: (r[1], r[2], r[3]) for r in con.execute(sql)}


def _quarters(end: str, n: int) -> list[str]:
    """end＝'2026-06-30' 這種季底日期，往前 n 季的季度代號（含 end）。"""
    y, q = int(end[:4]), (int(end[5:7]) - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append(f"{y}-Q{q}")
        y, q = (y - 1, 4) if q == 1 else (y, q - 1)
    return out


def _months(end: str, n: int) -> list[str]:
    y, m = int(end[:4]), int(end[5:7])
    out = []
    for _ in range(n):
        out.append(f"{y}-{m:02d}")
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return out


SCORE_CAT = {"miss": "資料缺失", "stale": "資料過期（財報、營收或本益比沒更新到應有的期別）",
             "na": "財務指標不適用（去年同期虧損，年增率沒有意義）", "loss": "財務指標不適用（近四季虧損，本益比沒有意義）",
             "hist": "股價歷史不足（新上市或停牌）"}
FLABEL = {"quality": "財務品質", "growth": "成長能力", "value": "估值合理性", "momentum": "股價動能", "trend": "趨勢確認"}


def audit(con, rows: list[dict], decision: dict, today=None) -> dict:
    from . import config
    from .snapshot import expected_fin_end, expected_rev_ym
    today = today or config.today()
    fin_end, rev_ym = expected_fin_end(today), expected_rev_ym(today)
    common = [r for r in rows if r.get("is_common") and r["stock_id"] in decision]
    px = _span(con, "SELECT stock_id, MIN(date), MAX(date), COUNT(*) FROM prices GROUP BY stock_id")
    rv = _span(con, "SELECT stock_id, MIN(ym), MAX(ym), COUNT(*) FROM revenue WHERE revenue IS NOT NULL GROUP BY stock_id")
    fn = _span(con, "SELECT stock_id, MIN(period), MAX(period), COUNT(*) FROM financials WHERE revenue IS NOT NULL OR net_income IS NOT NULL OR eps IS NOT NULL GROUP BY stock_id")
    it = _span(con, "SELECT stock_id, MIN(date), MAX(date), COUNT(*) FROM institutional GROUP BY stock_id")
    mg = _span(con, "SELECT stock_id, MIN(date), MAX(date), COUNT(*) FROM margin GROUP BY stock_id")
    rev_set, fin_set = {}, {}
    for sid, ym in con.execute("SELECT stock_id, ym FROM revenue WHERE revenue IS NOT NULL AND ym >= ?", (_months(rev_ym, 12)[-1],)):
        rev_set.setdefault(sid, set()).add(ym)
    q32 = _quarters(fin_end, 32)
    for sid, p in con.execute("SELECT stock_id, period FROM financials WHERE (revenue IS NOT NULL OR eps IS NOT NULL) AND period >= ?", (q32[-1],)):
        fin_set.setdefault(sid, set()).add(p)
    # V20.4 財報完整率分層：有紀錄 → 必要指標（營收、淨利、EPS）→ 主要科目（再加資產、負債、權益、營業現金流；一般業再加營業利益）
    fin_req, fin_main = {}, {}
    for row in con.execute("SELECT stock_id, period, revenue, operating_income, net_income, eps, total_assets, total_liabilities, equity, ocf "
                           "FROM financials WHERE period >= ?", (_quarters(fin_end, 4)[-1],)):
        sid, p, rev, op, ni, eps, ta, tl, eq, ocf = row
        if None not in (rev, ni, eps):
            fin_req.setdefault(sid, set()).add(p)
        if None not in (ni, eps, ta, tl, eq, ocf):
            fin_main.setdefault(sid, {})[p] = (rev is not None and op is not None)
    last_day = max((v[1] for v in px.values()), default=None)
    recent = [d for (d,) in con.execute("SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 5")]
    chip_cut = recent[-1] if recent else None
    m12, q4 = set(_months(rev_ym, 12)), set(_quarters(fin_end, 4))
    try:
        from . import fin_watch
        fin_watch.ensure(con)
        ann = {sid: per for sid, per in con.execute("SELECT stock_id, period FROM fin_announce WHERE status IN ('pending','waiting_source')")}
    except Exception:
        ann = {}

    per_stock, gaps = [], Counter()
    score_cat, score_fac, score_ex = Counter(), Counter(), {}
    gap_ex: dict = {}
    C = Counter()
    for r in common:
        sid = r["stock_id"]
        M = decision[sid].get("modules") or {}
        p, v, f, i, m = px.get(sid), rv.get(sid), fn.get(sid), it.get(sid), mg.get(sid)
        n_px = p[2] if p else 0
        rev12 = len(m12 & rev_set.get(sid, set()))
        have_q = fin_set.get(sid, set())
        fin4 = q4 <= have_q
        fin_req4 = q4 <= fin_req.get(sid, set())
        fm = fin_main.get(sid, {})
        fin_main4 = fin_req4 and q4 <= set(fm) and (bool(r.get("is_financial")) or all(fm[q] for q in q4))   # 分層：主要科目包含必要指標
        ref = (decision[sid].get("ref_scores") or {}) if not r.get("base_ok") else {f: r.get(f"score_{f}") for f in ("quality", "growth", "value") if r.get(f"score_{f}") is not None}
        fin_factor = all(f in ref for f in ("quality", "growth", "value"))
        fin_years = {y: len(set(q32[:4 * y]) & have_q) >= 4 * y - 1 for y in FIN_YEARS}
        inst_ok = bool(i and chip_cut and i[1] >= chip_cut)
        mg_ok = bool(m and chip_cut and m[1] >= chip_cut)
        val_ok = (M.get("valuation") or {}).get("status") in ("ready", "partial")
        ai_off = (M.get("ai") or {}).get("status") == "ready"
        ai_any = (M.get("ai") or {}).get("status") in ("ready", "partial")
        rb = ((r.get("ai") or {}).get("rec") or {}).get("balanced") or {}
        rec = rb.get("recommendation_eligibility")
        eligible = bool(r.get("base_ok") and r.get("complete") and rec == "normal")
        quote_ok = (M.get("quote") or {}).get("status") == "ready"
        C["quote"] += quote_ok
        for d in PX_DAYS:
            C[f"px{d}"] += n_px >= d
        C["rev12"] += rev12 == 12
        C["fin4"] += fin4
        C["fin_req4"] += fin_req4
        C["fin_main4"] += fin_main4
        C["fin_factor"] += fin_factor
        ai_ref_full = not ai_off and r.get("ref_composite") is not None
        C["ai_ref_full"] += ai_ref_full
        C["ai_full"] += ai_off or ai_ref_full
        if not (ai_off or ai_ref_full):   # 沒有完整綜合分數（正式或母體外）：拆成缺哪個因子、什麼原因
            kinds = {g.split(":")[1] for g in r.get("factor_gaps") or []}
            cats = {SCORE_CAT[k] for k in kinds} or {"必要項目數不夠（有部分資料，但不足以計分）"}
            if not ai_any:
                cats.add("完全無法評分")
            for c_ in cats:
                score_cat[c_] += 1
                score_ex.setdefault(c_, []).append(sid) if len(score_ex.get(c_, [])) < 8 else None
            for g in r.get("factor_gaps") or []:
                f_, k_ = g.split(":")
                score_fac[f"{FLABEL.get(f_, f_)}：{SCORE_CAT[k_]}"] += 1
        for y in FIN_YEARS:
            C[f"fin{y}y"] += fin_years[y]
        C["inst"] += inst_ok
        C["margin"] += mg_ok
        C["val"] += val_ok
        C["ai_official"] += ai_off
        C["ai_any"] += ai_any
        C["eligible"] += eligible
        why = []  # 這檔股票的具體缺口（同一句話會合併計數成排行榜）
        if not p:
            why.append("沒有任何收盤行情")
        elif not quote_ok:
            why.append("最新交易日沒有成交（可能停牌或暫停交易）")
        if n_px and n_px < 120:
            why.append("股價歷史未滿 120 個交易日（新上市或資料起點較晚）")
        if not v:
            why.append("沒有月營收資料")
        elif rev12 < 12:
            why.append("近 12 個月月營收不完整")
        if sid in ann:
            why.append("新一季財報已公告，資料來源尚未同步")
        elif not f:
            why.append("沒有季報資料")
        elif not fin4:
            why.append("近四季財報不完整")
        if not inst_ok:
            why.append("近 5 個交易日官方法人名單沒有這檔（官方名單只列有買賣的股票，通常代表法人沒有交易）")
        if not mg_ok:
            why.append("近 5 個交易日沒有融資融券資料（可能不是信用交易標的）")
        vs = (M.get("valuation") or {}).get("status")
        if vs == "not_applicable":
            why.append("近四季虧損，本益比不適用")
        elif vs == "insufficient_history":
            why.append("歷史本益比天數不足")
        elif vs in ("missing_source", "calculation_error"):
            why.append("估值無法計算")
        if not ai_any:
            why.append("AI 評分無法計算")
        if not eligible:
            ex = r.get("exclude_reasons") or []
            why.append("不符合正式推薦資格：" + ("、".join(ex) if ex else "因子未全部算出" if not r.get("complete") else
                                                 (rb.get("recommendation_cap_reason") or "流動性限制").split("，")[0] if rec in ("excluded", "capped") else "未排名"))
        for w in why:
            gaps[w] += 1
            gap_ex.setdefault(w, []).append(sid) if len(gap_ex.get(w, [])) < 8 else None
        st = [x.get("status") for x in M.values()]
        overall = ("calculation_error" if "calculation_error" in st else "missing_source" if (M.get("quote") or {}).get("status") == "missing_source"
                   else "stale" if "stale" in ((M.get("quote") or {}).get("status"), (M.get("fundamental") or {}).get("status"))
                   else "insufficient_history" if n_px < 120 else "ready" if all(s in ("ready", "not_applicable") for s in st) else "partial")
        C["overall_" + overall] += 1
        per_stock.append({
            "id": sid, "n": r.get("name"), "mk": r.get("market"),
            "px": [p[0], p[1], p[2]] if p else None, "rev": [v[0], v[1], v[2], rev12] if v else None,
            "fin": [f[0], f[1], f[2], int(fin4)] if f else None,
            "inst": [i[0], i[1], i[2]] if i else None, "mg": [m[0], m[1], m[2]] if m else None,
            "st": overall, "elig": int(eligible), "ai": "official" if ai_off else "reference" if ai_any else "none",
            "why": [w for w in why if not w.startswith("不符合正式推薦資格")], "not_elig": next((w for w in why if w.startswith("不符合正式推薦資格")), None),
        })
    n = len(common)
    pct = lambda k: round(C[k] / n * 100, 1) if n else None
    rowsum = lambda k, label: {"key": k, "label": label, "stocks": C[k], "pct": pct(k)}
    mk = Counter(r.get("market") for r in common)
    return {
        "schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
        "source": "正式資料庫 data/stocks.db（證交所、櫃買中心、FinMind）；不含示範資料",
        "as_of": {"prices": last_day, "expected_fin_quarter": _quarters(fin_end, 1)[0], "expected_rev_ym": rev_ym, "chip_cut": chip_cut},
        "universe": {"common": n, "twse": mk.get("twse", 0), "tpex": mk.get("tpex", 0),
                     "other": [{"id": r["stock_id"], "name": r.get("name"), "market": r.get("market")} for r in common if r.get("market") not in ("twse", "tpex")],
                     "note": "分母＝這次快照裡的普通股（ETF、特別股、存託憑證不算）。上市＋上櫃以外的是股票清單市場別還是「emerging（興櫃）」的股票，"
                             "多半是剛從興櫃轉上市櫃、已經有正式行情；股票清單每週更新時會改正市場別。"},
        "summary": [rowsum("quote", "有最新行情"), *[rowsum(f"px{d}", f"至少 {d} 個交易日行情") for d in PX_DAYS],
                    rowsum("rev12", "近 12 個月月營收完整"), rowsum("fin4", "近四季都有財報紀錄（營收或 EPS）"),
                    rowsum("fin_req4", "近四季必要指標齊全（營收、淨利、EPS）"), rowsum("fin_main4", "近四季主要科目齊全（再加資產、負債、權益、營業現金流）"),
                    rowsum("fin_factor", "可計算基本面因子（品質、成長、估值都算得出來）"),
                    *[rowsum(f"fin{y}y", f"有 {y} 年財報歷史") for y in FIN_YEARS],
                    rowsum("inst", "有最新法人買賣資料"), rowsum("margin", "有最新融資融券資料"), rowsum("val", "可計算有效估值"),
                    rowsum("ai_any", "可計算 AI 評分（含參考分數）"), rowsum("ai_official", "有正式 AI 綜合分數（推薦母體內，參與排名）"),
                    rowsum("ai_ref_full", "有母體外綜合分數（同一套公式，不排名）"), rowsum("ai_full", "有完整綜合分數（正式＋母體外）"),
                    rowsum("eligible", "可納入正式推薦")],
        "overall": {STATUS8[k[8:]]: v for k, v in C.items() if k.startswith("overall_")},
        "gaps": [{"why": w, "stocks": c, "examples": gap_ex.get(w, [])} for w, c in gaps.most_common(40)],
        # V20.4 正式評分與推薦資格分開：「評分」看五個因子算不算得出來；「推薦資格」另外看流動性、處置、長期虧損
        "score_tiers": {"official": C["ai_official"], "ref_full": C["ai_ref_full"], "partial": C["ai_any"] - C["ai_official"] - C["ai_ref_full"],
                        "none": n - C["ai_any"], "eligible": C["eligible"],
                        "rule": "正式綜合分數只在推薦母體內排名（成交金額、交易日數、長期虧損、處置股的門檻沒有放寬）；"
                                "母體外但五個因子都算得出來的股票，用同一套權重算綜合分數，只給個股研究參考，不排名、不進 AI 精選。"},
        "score_gaps": [{"why": w, "stocks": c, "examples": score_ex.get(w, [])} for w, c in score_cat.most_common()],
        "score_gaps_by_factor": [{"why": w, "stocks": c} for w, c in score_fac.most_common(20)],
        "stocks": per_stock,
        "notes": ["「不符合正式推薦資格」只代表不進 AI 精選（例如成交金額過低），不等於沒有資料；個股研究資訊照常可用。",
                  "財報分四層：有紀錄、必要指標、主要科目、可計算基本面因子；「有紀錄」不代表所有科目都完整。N 年歷史＝最近 4N 季至少有 4N-1 季。法人、融資融券只從 2026-08 起有資料，不能假設和股價一樣長。"],
    }
