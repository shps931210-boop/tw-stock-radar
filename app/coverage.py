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
        "universe": {"common": n, "twse": mk.get("twse", 0), "tpex": mk.get("tpex", 0)},
        "summary": [rowsum("quote", "有最新行情"), *[rowsum(f"px{d}", f"至少 {d} 個交易日行情") for d in PX_DAYS],
                    rowsum("rev12", "近 12 個月月營收完整"), rowsum("fin4", "近四季財報完整"),
                    *[rowsum(f"fin{y}y", f"有 {y} 年財報歷史") for y in FIN_YEARS],
                    rowsum("inst", "有最新法人買賣資料"), rowsum("margin", "有最新融資融券資料"), rowsum("val", "可計算有效估值"),
                    rowsum("ai_any", "可計算 AI 評分（含參考分數）"), rowsum("ai_official", "有正式 AI 綜合分數"),
                    rowsum("eligible", "可納入正式推薦")],
        "overall": {STATUS8[k[8:]]: v for k, v in C.items() if k.startswith("overall_")},
        "gaps": [{"why": w, "stocks": c, "examples": gap_ex.get(w, [])} for w, c in gaps.most_common(40)],
        "stocks": per_stock,
        "notes": ["「不符合正式推薦資格」只代表不進 AI 精選（例如成交金額過低），不等於沒有資料；個股研究資訊照常可用。",
                  "財報完整以「有營收或 EPS」計；N 年歷史＝最近 4N 季至少有 4N-1 季。法人、融資融券只從 2026-08 起有資料，不能假設和股價一樣長。"],
    }
