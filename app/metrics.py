"""從資料庫計算每檔股票的所有指標（定義見 catalog.py）。

約定：比率類一律用「百分比數字」，例如 ROE 15% 存成 15.0。資料不足時為 None。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import db, indicators


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _pct(a, b):
    a, b = _f(a), _f(b)
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def is_common_id(sid: str) -> bool:
    return bool(len(sid) == 4 and sid[0] != "0" and sid.isdigit())


def _turn(a, b):
    """基期為負或零時，成長率沒有意義：回傳（成長率, 標籤）。標籤：虧轉盈、盈轉虧、持續虧損。"""
    a, b = _f(a), _f(b)
    if a is None or b is None:
        return None, None
    if b <= 0:
        return None, ("虧轉盈" if a > 0 else "持續虧損")
    return (a - b) / b * 100, ("盈轉虧" if a <= 0 else None)


def _streak(arr) -> int:
    n = 0
    for v in reversed(list(arr)):
        if v > 0:
            n += 1
        else:
            break
    return n


def price_metrics(px: pd.DataFrame, idx: pd.Series | None) -> dict:
    m: dict = {}
    n = len(px)
    if n < 1:
        return m
    c, h, lo, v = (px[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    val = px["value"].to_numpy(float)
    m["date"] = px["date"].iloc[-1]
    m["close"] = c[-1]
    m["history_days"] = n
    if n < 2:  # 只有最新一天（剛開始回補）：先顯示收盤價，其他指標等歷史資料
        m["avg_value_20"] = val[-1]
        return m
    m["chg_pct"] = _pct(c[-1], c[-2])
    m["avg_value_20"] = float(np.nanmean(val[-20:]))
    m["volume_lots"] = v[-1] / 1000
    for k, d in (("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126), ("ret_12m", 252)):
        m[k] = _pct(c[-1], c[-d - 1]) if n > d else None
    m["mom_12_1"] = _pct(c[-22], c[-253]) if n >= 253 else None
    s = pd.Series(c)
    for k in (5, 10, 20, 60, 120, 240):
        m[f"ma{k}"] = _f(s.rolling(k).mean().iloc[-1]) if n >= k else None
    if n >= 80:
        ma60 = s.rolling(60).mean()
        m["ma60_slope_20"] = _pct(ma60.iloc[-1], ma60.iloc[-21])
    if all(m.get(k) for k in ("ma20", "ma60", "ma120")):
        m["ma_bull"] = m["ma20"] > m["ma60"] > m["ma120"]
    m["dist_ma60"] = _pct(c[-1], m.get("ma60"))
    for d in (20, 60, 120, 252):
        if n > d:
            m[f"brk_{d}"] = bool(c[-1] > np.max(h[-d - 1:-1]))
    m["high_52w"] = float(np.max(h[-252:]))
    m["low_52w"] = float(np.min(lo[-252:]))
    m["vol_ratio"] = _f(v[-1] / np.mean(v[-21:-1])) if n > 21 and np.mean(v[-21:-1]) > 0 else None
    if n >= 20:
        mid, sd = s.rolling(20).mean().iloc[-1], s.rolling(20).std().iloc[-1]
        m["bb_break"] = bool(c[-1] > mid + 2 * sd)
        m["bb_width"] = _f(4 * sd / mid * 100)
    if n >= 35:
        e = indicators.enrich(px[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        m["rsi"] = _f(e["rsi"].iloc[-1])
        m["k"], m["d"] = _f(e["k"].iloc[-1]), _f(e["d"].iloc[-1])
        m["macd_hist"] = _f(e["hist"].iloc[-1])
        # 進場時機訊號（第一版「到達買點」的技術觸發）
        def crossed(a, b, within):
            above = (e[a] > e[b]).to_numpy()[-(within + 1):]
            return bool(above[-1] and not above[:-1].all())
        t = e.iloc[-1]
        m["sig_ma_cross"] = crossed("ma5", "ma20", 3) if n >= 25 else None
        m["sig_kd_cross"] = bool(crossed("k", "d", 2) and t["k"] < 40)
        m["sig_macd_cross"] = crossed("dif", "dea", 3)
        m["sig_breakout"] = bool(m.get("brk_20") and (m.get("vol_ratio") or 0) >= 1.5)
        if n >= 60:
            m["sig_pullback"] = bool(abs(t["close"] / t["ma20"] - 1) <= 0.02 and t["ma20"] > t["ma60"]
                                     and t["close"] >= t["ma20"] and t["close"] > t["open"])
        m["dist_ma20"] = _pct(c[-1], m.get("ma20"))
    m["spark"] = [float(f"{x:.4g}") for x in c[-60:]]
    m["stop_ref"] = _f(min(m.get("ma20") or c[-1], float(np.min(lo[-10:]))))
    # 風險
    r = np.diff(np.log(c[-253:]))
    if len(r) >= 120:
        m["vol_1y"] = float(np.std(r) * math.sqrt(252) * 100)
        m["downside_vol"] = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(252) * 100)
        w = c[-253:]
        m["mdd_1y"] = float(np.min(w / np.maximum.accumulate(w) - 1) * 100)
        if idx is not None and len(idx) > 120:
            sr = pd.Series(np.log(px["close"].to_numpy(float)), index=px["date"]).diff()
            ir = np.log(idx).diff()
            j = pd.concat([sr, ir], axis=1, join="inner").dropna().iloc[-252:]
            if len(j) >= 120 and j.iloc[:, 1].var() > 0:
                m["beta"] = float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())
    return m


def fundamental_metrics(fin: pd.DataFrame, rev: pd.DataFrame, div: pd.DataFrame, val: pd.DataFrame) -> dict:
    m: dict = {}
    # ---- 估值（每日）----
    if len(val):
        last = val.iloc[-1]
        m["val_date"] = last["date"]
        m["per"] = _f(last["per"]) if _f(last["per"]) and last["per"] > 0 else None
        m["pbr"] = _f(last["pbr"]) if _f(last["pbr"]) and last["pbr"] > 0 else None
        m["dy"] = _f(last["dy"])
        m["earnings_yield"] = 100 / m["per"] if m["per"] else None
        m["book_to_price"] = 1 / m["pbr"] if m["pbr"] else None
        pe_hist = val["per"][val["per"] > 0]
        if m["per"] and len(pe_hist) > 120:
            m["per_pctile_1y"] = float((pe_hist.iloc[-250:] < m["per"]).mean() * 100)
    # ---- 月營收 ----
    if len(rev) >= 13:
        r = dict(zip(rev["ym"], rev["revenue"].astype(float)))
        last = pd.Period(rev["ym"].iloc[-1], "M")
        g = lambda p: r.get(str(p))
        m["rev_ym"] = str(last)
        m["rev_yoy"] = _pct(g(last), g(last - 12))
        m["rev_yoy_prev"] = _pct(g(last - 1), g(last - 13))
        m["rev_mom"] = _pct(g(last), g(last - 1))
        cur3 = [g(last - i) for i in range(3)]
        old3 = [g(last - 12 - i) for i in range(3)]
        if None not in cur3 + old3:
            m["rev_yoy_3m"] = _pct(sum(cur3), sum(old3))
        ytd = [g(pd.Period(year=last.year, month=i, freq="M")) for i in range(1, last.month + 1)]
        ytd0 = [g(pd.Period(year=last.year - 1, month=i, freq="M")) for i in range(1, last.month + 1)]
        if None not in ytd + ytd0:
            m["rev_ytd_yoy"] = _pct(sum(ytd), sum(ytd0))
        if m.get("rev_yoy") is not None and m.get("rev_yoy_prev") is not None:
            m["rev_accel"] = m["rev_yoy"] > m["rev_yoy_prev"]
        last12 = [g(last - i) for i in range(12)]
        if None not in last12:
            m["rev_high_12m"] = g(last) >= max(last12)
    # ---- 財報（單季）----
    if len(fin) >= 4:
        f = fin.sort_values("period_end").reset_index(drop=True)
        col = lambda k: f[k].astype(float)
        def ttm(k, end=0):
            x = col(k).iloc[len(f) - 4 - end: len(f) - end] if len(f) - 4 - end >= 0 else None
            return None if x is None or x.isna().any() else float(x.sum())
        def at(k, back=0):
            i = len(f) - 1 - back
            return _f(f[k].iloc[i]) if i >= 0 else None
        m["fin_period"] = f["period"].iloc[-1]
        m["fin_period_end"] = f["period_end"].iloc[-1]
        rev_t, gp_t, op_t, ni_t = ttm("revenue"), ttm("gross_profit"), ttm("operating_income"), ttm("net_income")
        m["eps_ttm"], m["ni_ttm"] = ttm("eps"), ni_t
        m["ocf_ttm"] = ttm("ocf")
        capex = ttm("capex")
        m["fcf_ttm"] = m["ocf_ttm"] - capex if m["ocf_ttm"] is not None and capex is not None else None
        eq, eq4 = at("equity"), at("equity", 4)
        ta, ta4 = at("total_assets"), at("total_assets", 4)
        avg = lambda a, b: (a + b) / 2 if a and b else a
        if ni_t is not None and eq:
            m["roe"] = ni_t / avg(eq, eq4) * 100
            m["roe_basis"] = "平均權益" if eq4 else "期末權益"
        if ni_t is not None and ta:
            m["roa"] = ni_t / avg(ta, ta4) * 100
        if rev_t:
            m["gross_margin"] = gp_t / rev_t * 100 if gp_t is not None else None
            m["op_margin"] = op_t / rev_t * 100 if op_t is not None else None
        if ta and at("total_liabilities") is not None:
            m["debt_ratio"] = at("total_liabilities") / ta * 100
        if ta and eq:
            m["equity_multiplier"] = ta / eq
        if m.get("ocf_ttm") is not None and ta:
            m["ocf_to_assets"] = m["ocf_ttm"] / ta * 100
        m["op_income_q"] = at("operating_income")
        m["ni_q"] = at("net_income")
        if len(f) >= 5:
            m["eps_q_yoy"], m["eps_q_turn"] = _turn(at("eps"), at("eps", 4))
            m["op_q_yoy"], m["op_q_turn"] = _turn(at("operating_income"), at("operating_income", 4))
            gq = lambda b: (at("gross_profit", b) / at("revenue", b) * 100) if at("gross_profit", b) is not None and at("revenue", b) else None
            if gq(0) is not None and gq(4) is not None:
                m["gm_q_change"] = gq(0) - gq(4)
        if len(f) >= 8:
            ni_p, rev_p, gp_p = ttm("net_income", 4), ttm("revenue", 4), ttm("gross_profit", 4)
            m["eps_ttm_yoy"], m["eps_ttm_turn"] = _turn(m["eps_ttm"], ttm("eps", 4))
            m["long_term_loss"] = (ni_t is not None and ni_p is not None and ni_t < 0 and ni_p < 0)
            nis = col("net_income").iloc[-8:]
            if nis.notna().all() and abs(nis.mean()) > 0:
                m["earnings_cv"] = float(nis.std() / abs(nis.mean()))
            # Piotroski F-Score（以近四季 vs 前四季比較）
            ta8 = at("total_assets", 8) or ta4
            roa_p = ni_p / avg(ta4, ta8) * 100 if ni_p is not None and ta4 else None
            cr = lambda b: at("current_assets", b) / at("current_liabilities", b) if at("current_assets", b) and at("current_liabilities", b) else None
            dr = lambda b: at("total_liabilities", b) / at("total_assets", b) if at("total_liabilities", b) is not None and at("total_assets", b) else None
            tests = [
                (m.get("roa"), lambda: m["roa"] > 0),
                (m.get("ocf_ttm"), lambda: m["ocf_ttm"] > 0),
                (m.get("roa") is not None and roa_p is not None or None, lambda: m["roa"] > roa_p),
                (m.get("ocf_ttm") is not None and ni_t is not None or None, lambda: m["ocf_ttm"] > ni_t),
                (dr(0) is not None and dr(4) is not None or None, lambda: dr(0) < dr(4)),
                (cr(0) is not None and cr(4) is not None or None, lambda: cr(0) > cr(4)),
                (at("share_capital") is not None and at("share_capital", 4) is not None or None,
                 lambda: at("share_capital") <= at("share_capital", 4) * 1.001),
                (gp_t is not None and gp_p is not None and rev_t and rev_p or None, lambda: gp_t / rev_t > gp_p / rev_p),
                (rev_t and rev_p and ta and ta4 or None, lambda: rev_t / ta > rev_p / ta4),
            ]
            avail = [t for t in tests if t[0] is not None]
            m["fscore_n"] = len(avail)
            # 標準 F-Score 九項都要能算；缺任何一項就不給分數，避免看起來像完整的 F-Score
            m["fscore"] = sum(1 for _, fn in avail if fn()) if len(avail) == len(tests) else None
        if len(f) >= 12:
            hits = 0
            for end in (0, 4, 8):
                n_ = ttm("net_income", end)
                e_ = at("equity", end)
                if n_ is not None and e_:
                    hits += n_ / e_ * 100 > 15
            m["roe_years_15"] = hits
        if m.get("ocf_ttm") is not None and ni_t:
            m["ocf_to_ni"] = m["ocf_ttm"] / ni_t if ni_t > 0 else None
        if m.get("pbr") and eq:
            m["market_cap"] = eq * m["pbr"]
            if m.get("fcf_ttm") is not None:
                m["fcf_yield"] = m["fcf_ttm"] / m["market_cap"] * 100
    # ---- 股利 ----
    if len(div):
        d = div.sort_values("year")
        cash = d.set_index("year")["cash"].astype(float)
        yrs = 0
        for y in range(int(cash.index.max()), int(cash.index.min()) - 1, -1):
            if cash.get(y, 0) > 0:
                yrs += 1
            else:
                break
        m["div_years"] = yrs
        m["div_cash_last"] = float(cash.iloc[-1])
        if m.get("eps_ttm") and m["eps_ttm"] > 0:
            m["payout"] = cash.iloc[-1] / m["eps_ttm"] * 100
        if len(cash) >= 4 and cash.iloc[-4] > 0:
            m["div_growth_3y"] = ((cash.iloc[-1] / cash.iloc[-4]) ** (1 / 3) - 1) * 100
    return m


CHIP_MAX_LAG = 1  # 法人、融資資料最多可以比股價晚幾個交易日（融資 21:00 才公布，14:40 那次會晚一天）


def _aligned(flow: pd.DataFrame, px: pd.DataFrame, n: int, cols: list[str]):
    """把籌碼資料和股價依交易日合併，取最近 n 個「連續」交易日。
    回傳 (合併後資料, 籌碼最新日期, 落後交易日數)；日期不連續或落後太多時合併資料為 None。"""
    if not len(flow):
        return None, None, None
    pdates = px["date"].tolist()
    last = flow["date"].max()
    lag = len(pdates) - np.searchsorted(pdates, last, side="right")
    if lag > CHIP_MAX_LAG:
        return None, last, lag
    j = px[["date", "close", "volume"]].merge(flow[["date"] + cols], on="date", how="inner").sort_values("date")
    j = j[j["date"] <= last]
    end = pdates.index(last) if last in pdates else None
    if end is None or end + 1 < n or len(j) < n:
        return None, last, lag
    if j["date"].iloc[-n:].tolist() != pdates[end - n + 1:end + 1]:  # 中間缺了某幾天的籌碼資料
        return None, last, lag
    return j.iloc[-n:].reset_index(drop=True), last, lag


def chip_metrics(inst: pd.DataFrame, mg: pd.DataFrame, px: pd.DataFrame) -> dict:
    """籌碼指標一律用「同一天」的股價與籌碼計算；籌碼資料落後或缺日時，該指標顯示為尚未更新（None）。"""
    m: dict = {}
    if not len(px):
        return m
    px = px.sort_values("date").reset_index(drop=True)
    cols = ["foreign_net", "trust_net", "dealer_net"]
    if len(inst):
        i = inst.sort_values("date")
        j5, m["inst_date"], m["inst_lag"] = _aligned(i, px, 5, cols)
        j20, _, _ = _aligned(i, px, 20, cols)
        if j5 is not None:
            m["foreign_5d"] = float(j5["foreign_net"].sum() / 1000)
            m["trust_5d"] = float(j5["trust_net"].sum() / 1000)
            m["foreign_trust_same_day"] = bool(j5["foreign_net"].iloc[-1] > 0 and j5["trust_net"].iloc[-1] > 0)
            vol5 = j5["volume"].sum()
            if vol5:
                m["inst_ratio_5d"] = float(j5[cols].sum().sum() / vol5 * 100)
            full = i[i["date"] <= m["inst_date"]]
            m["foreign_streak"] = _streak(full["foreign_net"])
            m["trust_streak"] = _streak(full["trust_net"])
        if j20 is not None:
            m["inst_20d"] = float(j20[cols].sum().sum() / 1000)
    if len(mg):
        g = mg.sort_values("date")
        j, m["margin_date"], m["margin_lag"] = _aligned(g, px, 20, ["margin_bal"])
        if j is not None:
            m["margin_chg_20d"] = _pct(j["margin_bal"].iloc[-1], j["margin_bal"].iloc[0])
            p20 = _pct(j["close"].iloc[-1], j["close"].iloc[0])
            if m["margin_chg_20d"] is not None and p20 is not None:
                m["price_up_margin_down"] = p20 > 0 and m["margin_chg_20d"] < 0
    return m


def adjust_prices(px: pd.DataFrame, ex: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """依除權息參考價還原歷史股價（向後調整：最新價不變，除權息日以前的價格乘上 參考價/前收盤）。
    證交所公布的前收盤與資料庫不一致（差超過 1%）時，不用這筆，避免用錯資料。"""
    if not len(ex) or len(px) < 2:
        return px, 0
    px = px.sort_values("date").reset_index(drop=True)
    dates = px["date"].tolist()
    factor = np.ones(len(px))
    used = 0
    for _, e in ex.iterrows():
        k = np.searchsorted(dates, e["date"])  # 除權息日（或之後第一個交易日）的位置
        if k == 0 or k >= len(dates):
            continue  # 除權息日在資料範圍外，或還沒發生
        prev = px["close"].iloc[k - 1]
        if not prev or abs(prev / e["prev_close"] - 1) > 0.01:
            continue
        factor[:k] *= e["ref_price"] / e["prev_close"]
        used += 1
    if used:
        px = px.copy()
        for c in ("open", "high", "low", "close"):
            px[c] = px[c] * factor
    return px, used


def compute_all(dbname: str = "stocks.db", stock_ids: list[str] | None = None) -> pd.DataFrame:
    with db.connect(dbname) as con:
        where = ""
        params: tuple = ()
        if stock_ids:
            where = f" WHERE stock_id IN ({','.join('?' * len(stock_ids))})"
            params = tuple(stock_ids)
        cut = con.execute("SELECT date FROM (SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 520) ORDER BY date LIMIT 1").fetchone()
        cut = cut[0] if cut else "1900-01-01"
        q = lambda t, extra="": db.read(con, f"SELECT * FROM {t}{where}{extra}", params)
        px = db.read(con, f"SELECT * FROM prices WHERE date >= ?" + (where.replace(" WHERE", " AND") if where else ""), (cut,) + params)
        stocks, fin, rev, div = q("stocks"), q("financials"), q("revenue"), q("dividends")
        val, inst, mg = q("valuation"), q("institutional"), q("margin")
        disp = set(db.read(con, "SELECT stock_id FROM disposition")["stock_id"])
        idx = db.read(con, "SELECT * FROM index_prices ORDER BY date")
        ex = db.read(con, "SELECT * FROM exrights" + where, params)
    idx_s = idx.set_index("date")["close"] if len(idx) else None
    G = lambda df: {k: g for k, g in df.groupby("stock_id")} if len(df) else {}
    pxg, fing, revg, divg = G(px.sort_values("date")), G(fin), G(rev.sort_values("ym")), G(div)
    valg, instg, mgg, exg = G(val.sort_values("date")), G(inst), G(mg), G(ex)
    empty = pd.DataFrame()
    info = stocks.set_index("stock_id").to_dict("index") if len(stocks) else {}
    rows = []
    for sid, p in pxg.items():
        p = p.reset_index(drop=True)
        m = {"stock_id": sid, "name": info.get(sid, {}).get("name", sid),
             "industry": info.get(sid, {}).get("industry") or "", "market": info.get(sid, {}).get("market") or ""}
        m["is_common"] = is_common_id(sid)
        m["is_financial"] = "金融" in m["industry"]
        m["disposition"] = sid in disp
        adj, m["adj_events"] = adjust_prices(p, exg.get(sid, empty))
        m.update(price_metrics(adj, idx_s))
        m["close"] = float(p["close"].iloc[-1])  # 顯示用的收盤價永遠是實際成交價
        m.update(fundamental_metrics(fing.get(sid, empty), revg.get(sid, empty), divg.get(sid, empty), valg.get(sid, empty)))
        m.update(chip_metrics(instg.get(sid, empty), mgg.get(sid, empty), p))  # 籌碼與量用原始成交資料
        rows.append(m)
    df = pd.DataFrame(rows)
    for c in df.columns:  # 全部為 None 的欄位轉成數值 NaN，布林與文字欄位保留
        if df[c].dtype == object and not df[c].map(lambda x: isinstance(x, (bool, str, list))).any():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
