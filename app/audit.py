"""對帳報告（data/audit/latest.json）：用另一套獨立寫法，直接從資料庫的原始成交價重算，和網站公布的數字逐筆比對。

績效驗證（選股名單事後報酬）
- 交易日曆：資料庫裡 0050 和加權指數有資料的日期。進場日＝選股日後第一個交易日；第 h 天出場＝進場日算第 1 天的第 h 個交易日。
- 報酬＝出場日原始收盤價 ÷ 進場日原始開盤價 × Π（期間內每次除權息的 前收盤 ÷ 參考價）− 交易成本。
  除權息日要在「進場日之後、出場日當天或之前」才算；前收盤和資料庫差超過 1% 的除權息紀錄不採用（與網站相同的資料檢查）。
- 0050 基準用同樣公式，不扣成本。
- 只比對網站有計入的股票（進場日買得到、有價格）；兩邊差超過 0.01 個百分點就列為不一致。

產業分析（最新資料日 1 天報酬）
- 每檔普通股：今天原始收盤 ÷ 前一個有成交日原始收盤（今天除權息就乘上 前收盤 ÷ 參考價）− 1，和網站的個股漲跌比對。
- 產業等權報酬：只用最新資料日有收盤價的股票重算平均，和公布值比對。
- 法人 5 日估算金額：抽查每個產業成交金額最大的一檔，用原始收盤價 × 買賣超股數重算。
"""
from __future__ import annotations

import numpy as np

from . import config, db, track

TOL = 0.01  # 百分點
EX_TOL = 0.01


def _calendar(con) -> list[str]:
    d = {r[0] for r in con.execute("SELECT date FROM prices WHERE stock_id=?", (track.BENCH_ID,))}
    d |= {r[0] for r in con.execute("SELECT date FROM index_prices")}
    return sorted(d)


def _ex_factor(con, sid: str, after: str, upto: str, raw: dict) -> tuple[float, list[dict]]:
    """(after, upto] 之間的除權息調整倍數，以及採用／不採用的紀錄。raw = {日期: 原始收盤}。"""
    f, used = 1.0, []
    dates = sorted(raw)
    for d, prev_close, ref in con.execute("SELECT date, prev_close, ref_price FROM exrights WHERE stock_id=? AND date>? AND date<=? ORDER BY date", (sid, after, upto)):
        before = [x for x in dates if x < d]
        p = raw.get(before[-1]) if before else None
        ok = bool(p and prev_close and ref and abs(p / prev_close - 1) <= EX_TOL)
        used.append({"date": d, "prev_close": prev_close, "ref_price": ref, "used": ok})
        if ok:
            f *= prev_close / ref
    return f, used


def _px(con, sid: str, lo: str, hi: str) -> dict:
    return {d: (o, c) for d, o, c in con.execute("SELECT date, open, close FROM prices WHERE stock_id=? AND date>=? AND date<=?", (sid, lo, hi))}


def _ret(con, sid, entry, exit_, cost):
    """獨立計算一檔股票的報酬（%）。回傳 (報酬, 證據) 或 (None, 原因)。"""
    lo = con.execute("SELECT MAX(date) FROM prices WHERE stock_id=? AND date<?", (sid, entry)).fetchone()[0] or entry
    px = _px(con, sid, lo, exit_)
    if entry not in px:
        return None, "進場日沒有成交"
    o = px[entry][0] if px[entry][0] and px[entry][0] > 0 else px[entry][1]
    held = [d for d in sorted(px) if entry <= d <= exit_ and px[d][1] and px[d][1] > 0]
    if not (o and o > 0) or not held:
        return None, "沒有有效價格"
    last = held[-1]
    f, ev = _ex_factor(con, sid, entry, last, {d: v[1] for d, v in px.items()})
    r = (px[last][1] / o * f - 1 - cost) * 100
    return r, {"entry": entry, "open_raw": o, "exit": last, "close_raw": px[last][1], "ex_events": ev, "factor": round(f, 6)}


def performance(dbname: str = "stocks.db", max_rows: int = 60) -> dict:
    logs = track.load()
    out = {"checked": 0, "matched": 0, "mismatched": [], "bench": [], "evidence": [], "cohorts": 0}
    if not logs:
        return {**out, "note": "還沒有選股紀錄"}
    ids = sorted({p["stock_id"] for L in logs for p in L["picks"].get("balanced", [])})
    series, idx, bench = track._prices(ids, logs[0]["date"], dbname)
    with db.connect(dbname) as con:
        cal = _calendar(con)
        for L in logs:
            d, group = L["date"], L["picks"].get("balanced", [])
            later = [x for x in cal if x > d]
            for h in track.HORIZONS:
                if len(later) < h:
                    continue
                sysc = track._cohort(group, d, series, idx, bench, h, None)
                if not sysc or not sysc.get("_per"):
                    continue
                out["cohorts"] += 1
                entry, exit_ = later[0], later[h - 1]
                if sysc.get("exit") != exit_ or (sysc.get("entry") and sysc["entry"] != entry):
                    out["mismatched"].append({"date": d, "h": h, "what": "交易日不一致", "system": [sysc.get("entry"), sysc.get("exit")], "audit": [entry, exit_]})
                for sid, sr in sysc["_per"].items():
                    ar, ev = _ret(con, sid, entry, exit_, track.COST)
                    out["checked"] += 1
                    srp = sr * 100
                    if ar is None or abs(ar - srp) > TOL:
                        out["mismatched"].append({"date": d, "h": h, "stock_id": sid, "system": round(srp, 4), "audit": None if ar is None else round(ar, 4),
                                                  "why": ev if isinstance(ev, str) else None, "evidence": ev if isinstance(ev, dict) else None})
                    else:
                        out["matched"] += 1
                        if len(out["evidence"]) < max_rows:
                            out["evidence"].append({"date": d, "h": h, "stock_id": sid, "return_pct": round(ar, 4), **ev})
                br, bev = _ret(con, track.BENCH_ID, entry, exit_, 0.0)
                sb = sysc.get("bench")
                out["bench"].append({"date": d, "h": h, "system": sb, "audit": None if br is None else round(br, 4),
                                     "ok": sb is not None and br is not None and abs(br - sb) <= TOL + 0.005, "evidence": bev})
    out["mismatched"] = out["mismatched"][:50]
    return out


def sectors(snap: dict, dbname: str = "stocks.db") -> dict:
    S = snap.get("sectors") or {}
    day = (snap.get("data_dates") or {}).get("prices")
    common = [s for s in snap.get("stocks", []) if s.get("is_common")]
    out = {"data_date": day, "stocks_checked": 0, "stock_mismatch": [], "industries_checked": 0, "industry_mismatch": [], "flows_checked": 0, "flows_mismatch": [],
           "stale_members": sum(1 for s in common if s.get("date") != day)}
    if not day or not S.get("industries"):
        return {**out, "note": "沒有產業資料"}
    mine: dict[str, float] = {}
    with db.connect(dbname) as con:
        for s in common:
            if s.get("date") != day or s.get("chg_pct") is None:
                continue
            sid = s["stock_id"]
            prev = con.execute("SELECT date, close FROM prices WHERE stock_id=? AND date<? AND close>0 ORDER BY date DESC LIMIT 1", (sid, day)).fetchone()
            cur = con.execute("SELECT close FROM prices WHERE stock_id=? AND date=?", (sid, day)).fetchone()
            if not prev or not cur or not cur[0]:
                continue
            f, _ = _ex_factor(con, sid, prev[0], day, {prev[0]: prev[1], day: cur[0]})
            r = (cur[0] / prev[1] * f - 1) * 100
            mine[sid] = r
            out["stocks_checked"] += 1
            if abs(r - s["chg_pct"]) > TOL:
                out["stock_mismatch"].append({"stock_id": sid, "system": round(s["chg_pct"], 4), "audit": round(r, 4)})
        by = {s["stock_id"]: s for s in common}
        for ind in S["industries"]:
            pub = (ind.get("ret") or {}).get("1d")
            vals = [mine[x] for x in ind.get("members", []) if x in mine]
            if not pub or not vals:
                continue
            out["industries_checked"] += 1
            a = float(np.mean(vals))
            if abs(a - pub["eq"]) > TOL + 0.005 or len(vals) != pub["n"]:
                out["industry_mismatch"].append({"industry": ind["industry"], "system": pub["eq"], "audit": round(a, 4), "n_system": pub["n"], "n_audit": len(vals)})
            # 法人估算金額抽查：產業內今天成交金額最大的一檔
            cand = sorted((by[x] for x in ind.get("members", []) if x in by and by[x].get("foreign_5d_value") is not None),
                          key=lambda s: -(s.get("value_1d") or 0))[:1]
            for s in cand:
                rows = con.execute("SELECT i.foreign_net, p.close FROM institutional i JOIN prices p ON p.stock_id=i.stock_id AND p.date=i.date "
                                   "WHERE i.stock_id=? AND i.date<=? ORDER BY i.date DESC LIMIT 5", (s["stock_id"], s.get("inst_date") or day)).fetchall()
                if len(rows) < 5:
                    continue
                v = sum(fn * c for fn, c in rows)
                out["flows_checked"] += 1
                if abs(v - s["foreign_5d_value"]) > max(1.0, abs(v) * 1e-5):  # 快照數字有四捨五入
                    out["flows_mismatch"].append({"stock_id": s["stock_id"], "system": s["foreign_5d_value"], "audit": v})
    out["stock_mismatch"] = out["stock_mismatch"][:50]
    return out


def build(snap: dict, dbname: str = "stocks.db") -> dict:
    warn, perf, sect = [], None, None
    try:
        perf = performance(dbname)
    except Exception as e:
        warn.append(f"績效對帳失敗：{type(e).__name__}: {e}")
    try:
        sect = sectors(snap, dbname)
    except Exception as e:
        warn.append(f"產業對帳失敗：{type(e).__name__}: {e}")
    ok_p = perf is not None and not perf["mismatched"] and all(b["ok"] for b in perf["bench"] if b["system"] is not None)
    ok_s = sect is not None and not sect["stock_mismatch"] and not sect["industry_mismatch"] and not sect["flows_mismatch"]
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "data_as_of": (snap.get("data_dates") or {}).get("prices"),
            "source": "資料庫原始成交價、除權息參考價、三大法人買賣超（獨立重算）", "tolerance_pct_point": TOL,
            "summary": {"performance_ok": ok_p, "sectors_ok": ok_s,
                        "performance": None if perf is None else {k: perf[k] for k in ("cohorts", "checked", "matched")} | {"mismatched": len(perf["mismatched"])},
                        "sectors": None if sect is None else {k: sect[k] for k in ("stocks_checked", "industries_checked", "flows_checked", "stale_members")}
                        | {"mismatched": len(sect["stock_mismatch"]) + len(sect["industry_mismatch"]) + len(sect["flows_mismatch"])}},
            "method": __doc__, "warnings": warn, "performance": perf, "sectors": sect}
