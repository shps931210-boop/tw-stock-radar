"""技術訊號歷史統計（簡易回測）：每個技術訊號在過去出現後，股價接下來 5、20 個交易日的表現。

範圍與限制（網站上會一起顯示）：
- 只能驗證「技術訊號」本身。體質條件（財報、營收）沒有保存「當時已公布」的歷史版本，
  直接拿現在的財報去回測會產生前視偏誤，所以完整的「今日技術訊號」規則還不能回測。
- 成交假設：訊號在收盤後才確定，所以在「隔天開盤價」買進，持有 h 個交易日後以收盤價賣出。
- 同一檔股票出現訊號後的 h 天內再出現同一訊號不重複計入（冷卻期），避免同一段行情被算很多次。
- 用還原除權息後的股價；扣除一次買賣的交易成本（手續費 0.1425% × 2 ＋ 證交稅 0.3%）。
- 超額報酬 = 個股報酬 − 同期加權指數報酬（加權指數是價格指數，不含股利，對個股略為有利）。
- 只納入 20 日平均成交金額 ≥ 1,000 萬的普通股。基準（所有股票、所有日子）不套用冷卻期。
- 資料庫目前約一年的股價，樣本期間短、只涵蓋單一市場環境，結果僅供參考。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db, indicators, metrics

METHOD = 2  # 計算方式改了就加一，讓快取失效
COST = 0.001425 * 2 + 0.003
HORIZONS = (5, 20)
MIN_VALUE = 1e7


def _crossed(above: pd.Series, within: int) -> pd.Series:
    """今天在上方，且前 within 天內至少有一天不在上方（= within 天內發生黃金交叉）。"""
    prev_all = above.shift(1).astype(float).rolling(within).min() == 1
    return above & ~prev_all


def signals(px: pd.DataFrame) -> pd.DataFrame:
    """和 metrics.price_metrics 相同定義的技術訊號，算出每一天的值。"""
    e = indicators.enrich(px[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
    c, v = e["close"], e["volume"]
    vol_ratio = v / v.shift(1).rolling(20).mean()
    brk20 = c > e["high"].shift(1).rolling(20).max()
    out = pd.DataFrame(index=e.index)
    out["sig_ma_cross"] = _crossed(e["ma5"] > e["ma20"], 3)
    out["sig_kd_cross"] = _crossed(e["k"] > e["d"], 2) & (e["k"] < 40)
    out["sig_macd_cross"] = _crossed(e["dif"] > e["dea"], 3)
    out["sig_breakout"] = brk20 & (vol_ratio >= 1.5)
    out["sig_pullback"] = ((c / e["ma20"] - 1).abs() <= 0.02) & (e["ma20"] > e["ma60"]) & (c >= e["ma20"]) & (c > e["open"])
    out.iloc[:60] = False  # 指標暖機期
    return out


def run(dbname: str = "stocks.db") -> dict:
    from .ai_model import TRIGGERS
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT * FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        idx = db.read(con, "SELECT * FROM index_prices ORDER BY date")
    if not len(px) or len(idx) < 80:
        return {"ready": False, "reason": "歷史股價或加權指數不足 80 個交易日"}
    idx_s = idx.set_index("date")["close"]
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    rec = {k: {h: [] for h in HORIZONS} for k in TRIGGERS}
    base = {h: [] for h in HORIZONS}
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid) or len(p) < 100:
            continue
        p, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        liquid = (p["value"].rolling(20).mean() >= MIN_VALUE).to_numpy()
        sig = signals(p)
        c = p["close"].to_numpy(float)
        o = p["open"].to_numpy(float)
        o = np.where(np.isnan(o) | (o <= 0), c, o)
        ix = idx_s.reindex(p["date"]).to_numpy(float)
        n = len(c)
        for h in HORIZONS:
            fwd = np.full(n, np.nan)   # t 日訊號 → t+1 開盤買、t+h 收盤賣
            fwd[:n - h] = c[h:] / o[1:n - h + 1] - 1 - COST
            ifwd = np.full(n, np.nan)
            ifwd[:n - h] = ix[h:] / ix[:n - h] - 1
            ok = liquid & ~np.isnan(fwd) & ~np.isnan(ifwd)
            ok[:60] = False
            base[h].append(np.c_[fwd[ok], fwd[ok] - ifwd[ok]])
            for k in TRIGGERS:
                hits = np.flatnonzero(ok & sig[k].to_numpy(bool))
                keep, nxt = [], -1
                for t in hits:  # 冷卻期：h 天內不重複計入
                    if t >= nxt:
                        keep.append(t)
                        nxt = t + h
                if keep:
                    rec[k][h].append(np.c_[fwd[keep], fwd[keep] - ifwd[keep]])

    def stats(chunks):
        if not chunks:
            return {"n": 0}
        a = np.vstack(chunks)
        if not len(a):
            return {"n": 0}
        return {"n": int(len(a)), "avg": round(float(a[:, 0].mean() * 100), 2), "median": round(float(np.median(a[:, 0]) * 100), 2),
                "excess": round(float(a[:, 1].mean() * 100), 2), "win": round(float((a[:, 1] > 0).mean() * 100), 1),
                "up": round(float((a[:, 0] > 0).mean() * 100), 1)}

    dates = sorted(px["date"].unique())
    return {
        "ready": True, "method": METHOD, "start": dates[60] if len(dates) > 60 else dates[0], "end": dates[-1], "days": len(dates),
        "cost_pct": round(COST * 100, 3), "horizons": list(HORIZONS),
        "baseline": {h: stats(base[h]) for h in HORIZONS},
        "signals": [{"key": k, "label": TRIGGERS[k], **{f"h{h}": stats(rec[k][h]) for h in HORIZONS}} for k in TRIGGERS],
        "note": "只驗證技術訊號本身，不含體質條件（缺少當時已公布的財報版本）。隔天開盤買進、持有期滿以收盤價賣出，同一檔同一訊號有冷卻期；報酬已扣交易成本、用還原除權息股價；"
                "超額報酬以加權指數（價格指數）為基準。樣本約一年，只涵蓋單一市場環境，僅供參考。",
    }
