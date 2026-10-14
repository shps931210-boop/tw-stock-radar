"""進出場策略引擎：短線、波段兩套規則由程式計算，網站顯示的「今天狀態」和歷史回測用的是同一個函式。

原則：
- 進場價、停損、停利都由規則算出來，AI 只負責解釋，不自己產生價格。
- 訊號在收盤後才確定，所以回測一律「隔天開盤價」進場；漲停鎖死或成交金額太少買不到，另外計數，不算進績效。
- 停損、停利用盤中最高最低價判斷；同一天同時碰到停損與停利，保守假設先停損。開盤就跳空越過停損價，以開盤價出場。
- 報酬已扣交易成本（手續費 0.1425% × 2 ＋ 證交稅 0.3%），用還原除權息股價。超額報酬以 0050 同一段期間（進場日開盤 → 出場日收盤）為基準。
- 同一檔股票持有中不重複進場。

長期策略看的是基本面與估值，判斷寫在 long_status()；它沒有可靠的「當時已公布」估值歷史，所以沒有在這裡回測，
網站會把長期的驗證連到「AI 多因子歷史回測」，並註明兩者規則不完全相同。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import db, metrics, track

METHOD = 1  # 規則或回測方式改了就加一
COST = 0.001425 * 2 + 0.003
MIN_VALUE = 1e7  # 訊號日 20 日平均成交金額至少 1,000 萬

RULES = {
    "short": {
        "name": "短線", "hold": "1–5 個交易日", "max_days": 5, "atr_stop": 1.5, "atr_target": 3.0,
        "setup": "股價在月線（20 日均線）之上，接近 20 日高點",
        "entry": "收盤突破前 20 日最高價，成交量是 20 日均量 1.5 倍以上，且收紅 K",
        "stop": "進場價 − 1.5 × ATR（14 日平均真實波幅）",
        "target": "進場價 + 3 × ATR（風險報酬比 2：1）",
        "exit": "碰到停損或停利；第 5 個交易日收盤仍未出場就出場",
        "invalid": "跌回突破前的整理區間、量能縮回均量以下",
    },
    "swing": {
        "name": "波段", "hold": "2–8 週", "max_days": 40, "atr_stop": 2.0, "atr_trail": 3.0,
        "setup": "多頭趨勢：月線在季線（60 日均線）之上、季線向上、股價在季線之上",
        "entry": "近 3 天曾回檔到月線 2% 以內，今天收盤站上月線且高於前一天最高價（回檔後轉強）",
        "stop": "進場價 − 2 × ATR",
        "target": "沒有固定停利：從進場後最高收盤價回落 3 × ATR 就出場（移動停利）；第一參考目標 = 進場價 + 2 倍風險",
        "exit": "碰到停損、移動停利，或收盤跌破季線；持有滿 40 個交易日出場",
        "invalid": "收盤跌破季線，或季線轉為下彎",
    },
}
LONG_RULE = {
    "name": "長期", "hold": "6 個月以上",
    "setup": "財務品質與成長因子都在全市場前 40%（分數 ≥ 60），AI 均衡型綜合分數 ≥ 60",
    "entry": "估值不貴：目前本益比不高於自己近兩年的中位數，或低於同產業中位數；且股價沒有跌破年線 10% 以上",
    "stop": "不用價格停損，而是「投資論點失效」就重新評估",
    "exit": "財務品質或成長因子掉到 40 分以下、本益比高於自己近兩年的第 80 百分位（估值過熱）、或股價跌破年線 15%",
    "invalid": "連續兩季 EPS 年減、月營收連續 3 個月年減 10% 以上",
}


def frame(px: pd.DataFrame) -> pd.DataFrame:
    """策略需要的欄位（還原除權息股價）。"""
    f = px[["date", "open", "high", "low", "close", "volume", "value"]].reset_index(drop=True).copy()
    o, h, lo, c, v = (f[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    o = o.where(o > 0, c)
    f["open"] = o
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean()
    f["ma20"], f["ma60"], f["ma240"] = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(240).mean()
    f["ma60_up"] = f["ma60"] > f["ma60"].shift(20)
    f["hh20"] = h.shift(1).rolling(20).max()
    f["vol_ratio"] = v / v.shift(1).rolling(20).mean()
    f["liquid"] = f["value"].astype(float).rolling(20).mean() >= MIN_VALUE
    return f


def entries(f: pd.DataFrame, warmup: int = 60) -> dict[str, pd.Series]:
    """每一天收盤後，兩套策略的「進場條件成立」與「觀察條件（setup）成立」。回測用 warmup=60 天暖機。"""
    c, o, h, lo = f["close"], f["open"], f["high"], f["low"]
    short_setup = (c > f["ma20"]) & (c >= f["hh20"] * 0.97)
    short = (c > f["hh20"]) & (f["vol_ratio"] >= 1.5) & (c > o) & (c > f["ma20"])
    trend = (f["ma20"] > f["ma60"]) & f["ma60_up"] & (c > f["ma60"])
    touched = (lo <= f["ma20"] * 1.02).astype(float).rolling(3).max() == 1
    swing_setup = trend & touched
    swing = swing_setup & (c > f["ma20"]) & (c > h.shift(1))
    warm = pd.Series(np.arange(len(f)) >= warmup)
    return {"short": short & warm, "short_setup": short_setup & warm, "swing": swing & warm, "swing_setup": swing_setup & warm}


def levels(mode: str, entry: float, atr: float) -> dict:
    r = RULES[mode]
    stop = entry - r["atr_stop"] * atr
    target = entry + r["atr_target"] * atr if "atr_target" in r else entry + 2 * (entry - stop)
    return {"entry": entry, "stop": stop, "target": target, "risk_pct": (entry - stop) / entry * 100,
            "reward_pct": (target - entry) / entry * 100, "rr": (target - entry) / (entry - stop) if entry > stop else None}


def simulate(f: pd.DataFrame, i: int, mode: str) -> dict | None:
    """第 i 天收盤出現訊號 → 第 i+1 天開盤進場，逐日檢查出場。資料不夠走完持有期就回傳 None。"""
    r, n = RULES[mode], len(f)
    if i + 1 >= n:
        return None
    o, h, lo, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ma60 = f["ma60"].to_numpy(float)
    entry = o[i + 1]
    lv = levels(mode, entry, float(f["atr"].iloc[i]))
    stop, target = lv["stop"], lv["target"] if mode == "short" else None
    peak, low_min = entry, entry
    for j in range(i + 1, n):
        day = j - i  # 進場日算第 1 天
        low_min = min(low_min, lo[j])
        op = o[j] if j > i + 1 else entry
        if op <= stop:
            px, why = op, "停損（跳空）"
        elif lo[j] <= stop:
            px, why = stop, "停損"
        elif target is not None and op >= target:
            px, why = op, "停利"
        elif target is not None and h[j] >= target:
            px, why = target, "停利"
        elif mode == "swing" and c[j] < ma60[j]:
            px, why = c[j], "跌破季線"
        elif day >= r["max_days"]:
            px, why = c[j], "持有期滿"
        else:
            if mode == "swing":  # 移動停利：最高收盤價 − 3 ATR，只往上調
                peak = max(peak, c[j])
                stop = max(stop, peak - r["atr_trail"] * float(f["atr"].iloc[j]))
            continue
        risk = entry - lv["stop"]
        return {"i": i, "exit_i": j, "entry": entry, "exit": px, "why": why, "days": day,
                "ret": px / entry - 1 - COST, "r": (px - entry) / risk if risk > 0 else None, "mae": low_min / entry - 1}
    return None


SHORT_MIN, SWING_MIN = 21, 80  # 短線要 20 日高點與 20 日均量；波段要季線與季線 20 天前的位置


def status(px: pd.DataFrame) -> dict:
    """今天收盤後兩套技術策略的狀態（網站個股頁、決策卡用）。"""
    n = len(px)
    if n < SHORT_MIN:
        return {"ready": False, "status": "insufficient_history", "days": n,
                "why": f"歷史股價只有 {n} 天：短線規則至少要 {SHORT_MIN} 天（20 日高點、均量），波段規則要 {SWING_MIN} 天"}
    f = frame(px)
    e = entries(f, warmup=0)
    t = f.iloc[-1]
    if pd.isna(t["atr"]) or not t["atr"]:
        return {"ready": False, "why": "波動資料不足"}
    out = {"ready": True, "status": "ready" if n >= SWING_MIN else "partial", "days": n,
           "date": t["date"], "atr": float(t["atr"]), "atr_pct": float(t["atr"] / t["close"] * 100)}
    for m in ("short", "swing"):
        if m == "swing" and n < SWING_MIN:
            out[m] = {"state": "na", "status": "insufficient_history",
                      "miss": [f"歷史股價只有 {n} 天，波段規則要看季線和季線方向，至少要 {SWING_MIN} 天"]}
            continue
        on, setup = bool(e[m].iloc[-1]), bool(e[m + "_setup"].iloc[-1])
        lv = levels(m, float(t["close"]), float(t["atr"]))  # 以今天收盤價當參考進場價
        miss = []
        if not on:
            if m == "short":
                if not t["close"] > t["ma20"]: miss.append("股價還在月線之下")
                if not t["close"] > t["hh20"]: miss.append(f"還沒突破前 20 日高點 {t['hh20']:.2f}（差 {(t['hh20'] / t['close'] - 1) * 100:.1f}%）")
                if not (t["vol_ratio"] or 0) >= 1.5: miss.append(f"量能 {t['vol_ratio'] or 0:.1f} 倍均量，未達 1.5 倍")
                if not t["close"] > t["open"]: miss.append("今天不是紅 K")
            else:
                if not t["ma20"] > t["ma60"]: miss.append("月線還在季線之下")
                if not bool(t["ma60_up"]): miss.append("季線沒有向上")
                if not t["close"] > t["ma60"]: miss.append("股價在季線之下")
                if setup and not on: miss.append(f"已回檔到月線附近，等收盤站上月線且高於前一天最高價 {f['high'].iloc[-2]:.2f}")
                elif not miss: miss.append(f"趨勢條件符合，等回檔到月線 {t['ma20']:.2f} 的 2% 以內再轉強（目前距月線 {(t['close'] / t['ma20'] - 1) * 100:+.1f}%）")
        out[m] = {"state": "triggered" if on else "setup" if setup else "none", "miss": miss, **{k: round(v, 2) if v is not None else None for k, v in lv.items()}}
    return out


def long_status(s: dict, pe: dict | None) -> dict:
    """長期策略：基本面與估值。s 是快照裡的個股資料（可帶 ref_scores），pe 是估值結果（valuation.band() 加上交易所本益比）。
    四項檢查：財務品質 ≥ 60、成長能力 ≥ 60、AI 均衡型綜合分數 ≥ 60、本益比不高於自己歷史中位數或同業中位數。
    缺其中幾項時，用拿得到的項目做「部分判斷」（status=partial），最多只到「觀察中」，不會給「條件成立」；
    品質和成長都沒有時才算無法判定。"""
    ranked = bool(s.get("base_ok"))
    ref = (s.get("ref_scores") or {}) if not ranked else {}
    q = s.get("score_quality") if ranked else ref.get("quality")
    g = s.get("score_growth") if ranked else ref.get("growth")
    comp = (s.get("ai") or {}).get("modes", {}).get("balanced", {}).get("score") if ranked and s.get("complete") else None
    det = {x.split("（")[0]: x for x in s.get("missing_detail") or []} if ranked else {}
    unknown = []
    if q is None: unknown.append(det.get("財務品質", "財務品質因子還不能計算（缺財報資料）"))
    if g is None: unknown.append(det.get("成長能力", "成長能力因子還不能計算（缺財報或營收資料）"))
    if comp is None:
        ex = s.get("exclude_reasons") or []
        unknown.append(f"沒有正式 AI 綜合分數：未通過排行榜基本門檻（{'、'.join(ex)}）" if ex else "沒有正式 AI 綜合分數：五個因子還沒有全部算得出來")
    if q is None and g is None:
        return {"state": "na", "status": "missing_source", "miss": unknown}
    miss, warn = [], []
    if q is not None and q < 60: miss.append(f"財務品質因子 {q:.0f} 分，未達 60")
    if g is not None and g < 60: miss.append(f"成長能力因子 {g:.0f} 分，未達 60")
    if comp is not None and comp < 60: miss.append(f"AI 均衡型綜合分數 {comp:.0f}，未達 60")
    good = not miss
    cheap = None
    if pe and pe.get("pe") and (pe.get("p50") or pe.get("peer")):
        cheap = bool((pe.get("p50") and pe["pe"] <= pe["p50"]) or (pe.get("peer") and pe["pe"] <= pe["peer"]))
        if not cheap:
            ref_txt = [f"自己近兩年中位數 {pe['p50']:.1f} 倍" if pe.get("p50") else None, f"同業中位數 {pe['peer']:.1f} 倍" if pe.get("peer") else None]
            miss.append(f"本益比 {pe['pe']:.1f} 倍，高於" + "與".join(x for x in ref_txt if x))
        if pe.get("p80") and pe["pe"] > pe["p80"]: warn.append("本益比高於自己近兩年第 80 百分位，估值偏熱")
        if not pe.get("p50"): warn.append("歷史本益比區間還不能算，只和同業中位數比較")
    elif pe and pe.get("why"):
        unknown.append(pe["why"])
    ma240, close = s.get("ma240"), s.get("close")
    if ma240 and close:
        d = (close / ma240 - 1) * 100
        if d < -10: miss.append(f"股價在年線下方 {abs(d):.0f}%")
        if d < -15: warn.append("股價跌破年線 15% 以上，長期論點需要重新檢查")
    if (q is not None and q < 40) or (g is not None and g < 40): warn.append("財務品質或成長因子低於 40 分，原本的長期論點可能已經不成立")
    full = not unknown and cheap is not None and bool(pe.get("p50"))
    if not good:
        state = "none"
    elif cheap is None:
        state = "na" if full else "setup"
    elif cheap and not [m for m in miss if "年線" in m]:
        state = "triggered" if not unknown else "setup"
    else:
        state = "setup"
    out = {"state": state, "status": "ready" if full or (not unknown and cheap is not None) else "partial", "miss": miss + unknown, "warn": warn}
    if unknown and state != "na":
        out["note"] = "部分判斷：" + "；".join(unknown) + "。拿得到的項目照常檢查，少的項目不當作通過，所以最多只到「觀察中」"
    if not ranked:
        out["basis"] = "ref"
    return out


def backtest(dbname: str = "stocks.db") -> dict:
    """短線、波段兩套規則的歷史回測（和 status() 同一套 entries/levels/simulate）。"""
    with db.connect(dbname) as con:
        px = db.read(con, "SELECT * FROM prices ORDER BY stock_id, date")
        ex = db.read(con, "SELECT * FROM exrights")
        bench, kind = track.benchmark(con)
    if not len(px) or kind != track.BENCH_ID:
        return {"ready": False, "reason": "沒有股價或 0050 資料"}
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    trades = {m: [] for m in RULES}
    funnel = {m: {"setup_days": 0, "signals": 0, "limit_up": 0, "thin": 0, "open_trades": 0} for m in RULES}
    for sid, p in px.groupby("stock_id"):
        if not metrics.is_common_id(sid) or len(p) < 100:
            continue
        p, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        f = frame(p)
        e = entries(f)
        o, c, h, lo = (f[k].to_numpy(float) for k in ("open", "close", "high", "low"))
        val = p["value"].to_numpy(float)
        liquid = f["liquid"].to_numpy(bool)
        b = bench.reindex(f["date"])
        bo, bc = b["open"].to_numpy(float), b["close"].to_numpy(float)
        for m in RULES:
            funnel[m]["setup_days"] += int((e[m + "_setup"] & f["liquid"]).sum())
            nxt = 0
            for i in np.flatnonzero(e[m].to_numpy(bool) & liquid):
                if i < nxt:
                    continue
                funnel[m]["signals"] += 1
                if i + 1 >= len(f):
                    funnel[m]["open_trades"] += 1
                    continue
                if o[i + 1] >= c[i] * (1 + track.LIMIT_UP) and h[i + 1] == lo[i + 1]:
                    funnel[m]["limit_up"] += 1
                    continue
                if val[i + 1] < track.THIN_VALUE:
                    funnel[m]["thin"] += 1
                    continue
                t = simulate(f, i, m)
                if t is None:
                    funnel[m]["open_trades"] += 1
                    continue
                bret = bc[t["exit_i"]] / bo[i + 1] - 1 if bo[i + 1] and not np.isnan(bc[t["exit_i"]]) and not np.isnan(bo[i + 1]) else np.nan
                trades[m].append({**t, "sid": sid, "date": f["date"].iloc[i], "excess": t["ret"] - bret if not np.isnan(bret) else None})
                nxt = t["exit_i"] + 1
    return {"ready": True, "method": METHOD, "cost_pct": round(COST * 100, 3), "start": px["date"].min(), "end": px["date"].max(),
            "modes": {m: {**stats(trades[m]), "funnel": funnel[m]} for m in RULES}}


def stats(ts: list[dict]) -> dict:
    if not ts:
        return {"n": 0}
    r = np.array([t["ret"] for t in ts])
    ex = np.array([t["excess"] for t in ts if t["excess"] is not None])
    rr = np.array([t["r"] for t in ts if t["r"] is not None])
    win, loss = r[r > 0], r[r <= 0]
    why: dict = {}
    for t in ts:
        why[t["why"]] = why.get(t["why"], 0) + 1
    years: dict = {}
    for t in ts:
        years.setdefault(t["date"][:4], []).append(t["ret"])
    pct = lambda x: round(float(x) * 100, 2)
    return {"n": len(ts), "win": round(float((r > 0).mean() * 100), 1), "avg": pct(r.mean()), "median": pct(np.median(r)),
            "p10": pct(np.percentile(r, 10)), "p90": pct(np.percentile(r, 90)),
            "avg_win": pct(win.mean()) if len(win) else None, "avg_loss": pct(loss.mean()) if len(loss) else None,
            "profit_factor": round(float(win.sum() / -loss.sum()), 2) if len(loss) and loss.sum() < 0 else None,
            "avg_r": round(float(rr.mean()), 2) if len(rr) else None, "excess": pct(ex.mean()) if len(ex) else None,
            "beat": round(float((ex > 0).mean() * 100), 1) if len(ex) else None,
            "mae": pct(np.mean([t["mae"] for t in ts])), "days": round(float(np.mean([t["days"] for t in ts])), 1),
            "why": dict(sorted(why.items(), key=lambda x: -x[1])),
            "years": {y: {"n": len(v), "avg": pct(np.mean(v)), "win": round(float((np.array(v) > 0).mean() * 100), 1)} for y, v in sorted(years.items())}}
