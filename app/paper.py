"""AI 模擬交易（第一階段：只模擬、不下單）。

這個模組只讀「已經封存」的正式 AI 精選（history/picks/），用固定規則模擬買賣，從來不會送出任何真實委託；
真實下單的介面（app/broker.py）預設永久關閉。

固定規則（RULES_VERSION，事先寫死，不會依結果調整）
- 起點：2026-10-27 的名單（10/28 開盤第一次進場）。之前的日期不回填成模擬交易；想看以前的名單表現，看「AI 預測回放」。
- 三本帳：穩健型、均衡型、積極型各一本，各自 100 萬元起始資金，最多同時持有 5 檔，每檔目標金額＝前一天收盤總資產 ÷ 5。
- 名單：訊號日的「預設 AI 精選」（rec_picks），用在進場日 09:00 以前產生的最新一版；沒有就那天不下單。
  依排名順序挑還沒持有的股票，補滿空出來的位置。
- 進場：訊號日隔一個交易日的開盤價 ×（1＋0.1% 滑價），不會用訊號日當天或更早的價格。
  買不到（當天沒成交、停牌、漲停鎖死＝開高低同價且漲幅 ≥ 9.5%）就取消，現金留著，原因記下來。
  成交股數最多是當天成交量的 5%，超過的部分算沒成交（部分成交），零股照實記。
- 出場（收盤後判斷，隔一個交易日開盤賣）：收盤價 ≤ 買進價 −10% 停損；收盤價 ≥ 買進價 +20% 停利；持有滿 20 個交易日（進場日算第 1 天）到期。
  只用收盤價判斷，同一天不可能同時碰到停損和停利；盤中高低點不看。
  賣不掉（沒成交、跌停鎖死）就隔天再賣；成交量 5% 上限賣不完的部分隔天繼續賣。
- 成本：手續費 0.1425%（整股最低 20 元、零股最低 1 元），賣出證交稅股票 0.3%、ETF 0.1%，滑價買賣各 0.1%。
- 除權息：持有到除權息日，依交易所公告的參考價差（前一天收盤－參考價）×股數記為現金（近似，股票股利也當現金算）。
- 比較：同樣 100 萬元在第一個進場日開盤買 0050 放著；加權指數用收盤價換算。

每次執行都從起點整本重算（ledger.json），所以任何一筆都能從原始價格和封存名單重現；
另外每天把當天的結果指紋寫進 days.json（只新增不改），之後重算和以前記的不一樣就標示出來，不會悄悄改掉。

執行：python -m app.paper
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

from . import broker, config

RULES_VERSION = "2026-10-27"
START_SIGNAL = "2026-10-27"
MODES = ("stable", "balanced", "aggressive")
MODE_NAMES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}
CASH0 = 1_000_000
MAX_POS = 5
SLIP = 0.001
FEE, MIN_FEE, ODD_MIN_FEE = 0.001425, 20, 1
TAX_STOCK, TAX_ETF = 0.003, 0.001
VOL_CAP = 0.05
STOP, TARGET, HOLD = -0.10, 0.20, 20
LIMIT = 0.095
OPEN_AT = "09:00:00"
BENCH = "0050"
REPLAY_START = "2026-10-08"
REPLAY_TOP = 10
REPLAY_COST = 0.001425 * 2 + 0.003
LIVE_TRADING = broker.LIVE_TRADING   # 第一階段永久關閉；這個模組也沒有任何送單的程式碼

PICKS = config.ROOT / "history" / "picks"
OUT = config.ROOT / "history" / "paper"
HOL = config.ROOT / "history" / "premarket" / "holidays.json"
NOTES = config.ROOT / "history" / "audit" / "closure_notes.json"

REASONS = {"stop": "停損（收盤跌破買進價 10%）", "target": "停利（收盤漲超過買進價 20%）", "expiry": "持有滿 20 個交易日"}
UNFILLED = {"no_trade": "當天沒有成交（停牌或沒有交易）", "limit_up": "漲停鎖死買不到", "limit_down": "跌停鎖死賣不掉",
            "volume": "成交量太少（超過當天成交量 5% 的部分）", "cash": "現金不夠買 1 股", "no_price": "沒有價格資料"}
RULES_TEXT = ("訊號日收盤後的正式 AI 精選，隔一個交易日開盤買（＋0.1% 滑價）；收盤跌 10% 停損、漲 20% 停利、滿 20 個交易日到期，"
              "都在隔一個交易日開盤賣（－0.1% 滑價）。手續費 0.1425%、證交稅 0.3%（ETF 0.1%），成交量最多吃當天 5%，"
              "漲停買不到、跌停賣不掉照實記。只模擬，不會送出任何真實委託。")


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text("utf-8"))
    except (OSError, ValueError):
        return None


def fee(value: float, shares: int) -> int:
    return max(MIN_FEE if shares >= 1000 else ODD_MIN_FEE, math.floor(value * FEE))


def tax(value: float, sid: str) -> int:
    return math.floor(value * (TAX_ETF if str(sid).startswith("00") else TAX_STOCK))


# ---------------- 交易日與封存名單 ----------------
def next_trading(d: str, days: list[str], hol: set[str] | None = None) -> str:
    """資料裡有的就用資料的下一個交易日；資料還沒到的，用休市日曆推算（跳過週末、休市日）。"""
    later = [x for x in days if x > d]
    if later:
        return later[0]
    hol = hol or set()
    x = date.fromisoformat(d)
    while True:
        x += timedelta(days=1)
        if x.weekday() < 5 and x.isoformat() not in hol:
            return x.isoformat()


def _enrich(x: dict, key: str = "rec_picks") -> dict:
    """舊版紀錄的正式精選只存代號和分數；名稱、收盤價、因子分數從同一份紀錄的原始排名補上（同一份封存檔，不是事後資料）。"""
    info = {}
    for rows in (x.get("picks") or {}).values():
        for c in rows:
            info.setdefault(c["stock_id"], c)
    return {m: [{**{k: v for k, v in info.get(c["stock_id"], {}).items() if k in ("name", "close", "industry", "factors")}, **c}
                for c in rows] for m, rows in (x.get(key) or {}).items()}


def signal_for(d: str, entry: str, picks: Path = PICKS) -> dict | None:
    """訊號日 d 在進場日 09:00 以前產生的最新一版名單。之後才產生的版本一律不用（避免用到開盤後的資訊）。"""
    best = None
    for f in sorted(picks.glob(f"{d}*.json")) if picks.exists() else []:
        x = _read(f) or {}
        g = x.get("generated_at") or ""
        if not x.get("rec_picks") or not g or g >= f"{entry}T{OPEN_AT}":
            continue
        if best is None or g > best["generated_at"]:
            best = {"file": f.name, "generated_at": g, "model_version": x.get("model_version"), "model_hash": x.get("model_hash"),
                    "data_dates": x.get("data_dates"), "lists": _enrich(x)}
    return best


# ---------------- 一本帳的模擬 ----------------
def _bar(px, sid, d):
    return (px.get(sid) or {}).get(d)


def _prev_close(px, sid, d, exr):
    e = exr.get((sid, d))
    if e and e[1]:
        return e[1]
    rows = px.get(sid) or {}
    ks = [k for k in rows if k < d]
    return rows[max(ks)][3] if ks else None


def _last_close(px, sid, d):
    rows = px.get(sid) or {}
    ks = [k for k in rows if k <= d and rows[k][3]]
    return rows[max(ks)][3] if ks else None


def simulate(mode: str, days: list[str], px: dict, signals: dict, exr: dict | None = None) -> dict:
    """依固定規則模擬一本帳。days：已經有收盤資料的交易日；signals：{訊號日: signal_for(...)}。只用 d 當天以前的資料做 d 的決定。"""
    exr = exr or {}
    cash = float(CASH0)
    pos: dict[str, dict] = {}       # sid -> 持股
    sell_q: dict[str, str] = {}     # sid -> 出場原因（隔天開盤賣）
    trades, orders, curve = [], [], []
    tid = 0
    first_entry = None
    for i, d in enumerate(days):
        if d <= START_SIGNAL:   # 起點以前不模擬（不回填）
            continue
        prev = days[i - 1] if i else None
        # 1. 除權息：前一天以前就持有的股票，參考價差記為現金
        for sid, p in pos.items():
            e = exr.get((sid, d))
            if e and p["entry_date"] < d and e[0] and e[1] and e[0] > e[1]:
                amt = round((e[0] - e[1]) * p["shares"], 2)
                cash += amt
                p["dividends"] += amt
        # 2. 開盤先賣
        for sid in list(sell_q):
            p, why = pos[sid], sell_q[sid]
            b = _bar(px, sid, d)
            o = {"date": d, "side": "sell", "sid": sid, "name": p["name"], "want": p["shares"], "reason": REASONS[why]}
            if not b or not b[4]:
                orders.append({**o, "filled": 0, "status": "unfilled", "why": UNFILLED["no_trade"]})
                continue
            op, hi, lo, cl, vol = b
            pc = _prev_close(px, sid, d, exr)
            if op == hi == lo and pc and op <= pc * (1 - LIMIT):
                orders.append({**o, "filled": 0, "status": "unfilled", "why": UNFILLED["limit_down"]})
                continue
            q = min(p["shares"], int(vol * VOL_CAP))
            if q <= 0:
                orders.append({**o, "filled": 0, "status": "unfilled", "why": UNFILLED["volume"]})
                continue
            price = max(lo, round(op * (1 - SLIP), 4))
            val = q * price
            f, t = fee(val, q), tax(val, sid)
            cash += val - f - t
            p["proceeds"] += val - f - t
            p["sold"] += q
            p["sell_fee"] += f
            p["sell_tax"] += t
            p["sell_value"] += val
            p["shares"] -= q
            orders.append({**o, "filled": q, "price": round(price, 4), "status": "filled" if p["shares"] == 0 else "partial",
                           "why": None if p["shares"] == 0 else UNFILLED["volume"]})
            if p["shares"] == 0:
                bought = p["buy_value"] + p["buy_fee"]
                net = p["proceeds"] + p["dividends"] - bought
                gross = p["sell_value"] / p["buy_value"] - 1 if p["buy_value"] else None
                trades.append({**{k: v for k, v in p.items() if k not in ("cost", "proceeds", "sold", "shares")},
                               "shares": p["bought"], "exit_date": d, "exit_price": round(p["sell_value"] / p["bought"], 4),
                               "exit_reason": REASONS[why], "exit_kind": why, "exit_signal_date": p["exit_signal_date"],
                               "gross_ret": round(gross * 100, 3) if gross is not None else None,
                               "net_ret": round(net / bought * 100, 3), "pnl": round(net, 0),
                               "bench_ret": _bench_ret(px, p["entry_date"], d), "status": "closed"})
                tr = trades[-1]
                tr["excess"] = round(tr["net_ret"] - tr["bench_ret"], 3) if tr["bench_ret"] is not None else None
                del pos[sid]
                del sell_q[sid]
        # 3. 開盤再買：前一個交易日的訊號
        sig = signals.get(prev) if prev else None
        if sig:
            prev_eq = curve[-1]["equity"] if curve else CASH0
            slots = MAX_POS - len(pos)
            cands = [c for c in (sig["lists"].get(mode) or []) if c["stock_id"] not in pos][:max(slots, 0)]
            for c in cands:
                sid = c["stock_id"]
                o = {"date": d, "side": "buy", "sid": sid, "name": c.get("name"), "signal_date": prev, "rank": c.get("rank")}
                b = _bar(px, sid, d)
                if not b or not b[4]:
                    orders.append({**o, "want": None, "filled": 0, "status": "unfilled", "why": UNFILLED["no_trade"]})
                    continue
                op, hi, lo, cl, vol = b
                pc = _prev_close(px, sid, d, exr)
                if op == hi == lo and pc and op >= pc * (1 + LIMIT):
                    orders.append({**o, "want": None, "filled": 0, "status": "unfilled", "why": UNFILLED["limit_up"]})
                    continue
                price = min(hi, round(op * (1 + SLIP), 4))
                budget = min(prev_eq / MAX_POS, cash)
                want = int(budget / (price * (1 + FEE)))
                while want > 0 and want * price + fee(want * price, want) > cash:
                    want -= 1
                q = min(want, int(vol * VOL_CAP))
                if want <= 0:
                    orders.append({**o, "want": 0, "filled": 0, "status": "unfilled", "why": UNFILLED["cash"]})
                    continue
                if q <= 0:
                    orders.append({**o, "want": want, "filled": 0, "status": "unfilled", "why": UNFILLED["volume"]})
                    continue
                val = q * price
                f = fee(val, q)
                cash -= val + f
                tid += 1
                first_entry = first_entry or d
                pos[sid] = {"id": f"{mode[:1].upper()}{tid:04d}", "mode": mode, "sid": sid, "name": c.get("name"),
                            "industry": c.get("industry"), "signal_date": prev, "pick_file": sig["file"],
                            "pick_generated_at": sig["generated_at"], "model_version": sig.get("model_version"),
                            "model_hash": sig.get("model_hash"), "data_dates": sig.get("data_dates"),
                            "rank": c.get("rank"), "score": c.get("score"), "liq": c.get("liq"),
                            "reason": _why(c), "signal_close": c.get("close"),
                            "entry_date": d, "entry_price": round(price, 4), "entry_open": op, "bought": q, "shares": q,
                            "buy_value": round(val, 2), "buy_fee": f, "cost": val + f, "proceeds": 0.0, "sold": 0,
                            "sell_fee": 0, "sell_tax": 0, "sell_value": 0.0, "dividends": 0.0,
                            "stop_price": round(price * (1 + STOP), 2), "target_price": round(price * (1 + TARGET), 2),
                            "max_close": None, "min_close": None, "fill_note": None if q == want else f"部分成交 {q}/{want} 股（成交量上限）",
                            "exit_signal_date": None}
                orders.append({**o, "want": want, "filled": q, "price": round(price, 4), "status": "filled" if q == want else "partial",
                               "why": None if q == want else UNFILLED["volume"]})
        # 4. 收盤：市值、出場判斷（明天開盤執行）
        mv = 0.0
        for sid, p in pos.items():
            b = _bar(px, sid, d)
            cl = b[3] if b and b[3] else _last_close(px, sid, d)
            p["last_close"], p["last_date"] = cl, d
            if cl is None:
                continue
            mv += p["shares"] * cl
            p["max_close"] = cl if p["max_close"] is None else max(p["max_close"], cl)
            p["min_close"] = cl if p["min_close"] is None else min(p["min_close"], cl)
            k = days.index(d) - days.index(p["entry_date"]) + 1
            p["held_days"] = k
            if sid in sell_q:
                continue
            why = ("stop" if cl <= p["entry_price"] * (1 + STOP) else "target" if cl >= p["entry_price"] * (1 + TARGET)
                   else "expiry" if k >= HOLD else None)
            if why and b:   # 當天沒成交的不用舊價格判斷停損停利，到期照算
                sell_q[sid], p["exit_signal_date"] = why, d
            elif why == "expiry":
                sell_q[sid], p["exit_signal_date"] = why, d
        curve.append({"date": d, "cash": round(cash, 0), "mv": round(mv, 0), "equity": round(cash + mv, 0), "n": len(pos)})
    open_pos = []
    for sid, p in pos.items():
        cl = p.get("last_close")
        val = (p["shares"] * cl) if cl else None
        unreal = (val + p["proceeds"] + p["dividends"] - (p["buy_value"] + p["buy_fee"])) if val is not None else None
        open_pos.append({k: v for k, v in p.items() if k not in ("cost", "proceeds")}
                        | {"market_value": round(val, 0) if val is not None else None,
                           "unreal_ret": round(unreal / (p["buy_value"] + p["buy_fee"]) * 100, 3) if unreal is not None else None,
                           "pending_exit": REASONS.get(sell_q.get(sid)) if sid in sell_q else None, "status": "open"})
    return {"mode": mode, "cash": round(cash, 0), "positions": open_pos, "trades": trades, "orders": orders, "curve": curve,
            "first_entry": first_entry, "pending_sells": [{"sid": s, "reason": REASONS[w]} for s, w in sell_q.items()]}


def _why(c: dict) -> str:
    f = c.get("factors") or {}
    top = sorted(((v, k) for k, v in f.items() if v is not None), reverse=True)[:2]
    nm = {"quality": "品質", "growth": "成長", "value": "價值", "momentum": "動能", "trend": "趨勢"}
    return f"正式精選第 {c.get('rank')} 名，AI 量化評分 {c.get('score')}；" + "、".join(f"{nm.get(k, k)} {v:.0f}" for v, k in top)


def _bench_ret(px, a: str, b: str) -> float | None:
    """0050 同一段期間（進場日開盤 → 出場日開盤）的價格報酬，不扣成本。"""
    r = px.get(BENCH) or {}
    if a in r and b in r and r[a][0] and r[b][0]:
        return round((r[b][0] / r[a][0] - 1) * 100, 3)
    return None


# ---------------- 統計 ----------------
def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h) * 100, 1), round(min(1, c + h) * 100, 1)]


def mdd(vals: list[float]) -> float | None:
    if not vals:
        return None
    peak, m = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        m = min(m, v / peak - 1)
    return round(m * 100, 2)


def window_done(entry: str | None, days: list[str]) -> bool:
    """進場後已經過了 20 個交易日＋賣出那天：這一批的結果（停利／停損／到期）全部都確定了。"""
    return bool(entry) and entry in days and len(days) - 1 - days.index(entry) >= HOLD


def odds(trades: list[dict], days: list[str] | None = None, key: str = "exit_kind", ret: str = "net_ret") -> dict:
    """先停利、先停損、到期出場各占多少（Wilson 95% 區間），扣成本平均報酬（±1.96 標準誤）。少於 30 筆標示樣本不足。
    只算「整段 20 個交易日都已經過完」的進場批次：太早算的話，只有很快停損或停利的交易會先出場，比例會偏掉。
    還在觀察期內就先出場的另外列在 early，不算進比例。"""
    if days is not None:
        early = [t for t in trades if not window_done(t.get("entry_date") or t.get("entry"), days)]
        trades = [t for t in trades if window_done(t.get("entry_date") or t.get("entry"), days)]
    else:
        early = []
    n = len(trades)
    out = {"n": n, "enough": n >= 30, "label": "樣本不足，只能看方向" if n < 30 else "樣本數夠做初步判斷",
           "early": {k: sum(1 for t in early if t.get(key) == k) for k in ("target", "stop", "expiry")},
           "rule": "只算進場後 20 個交易日都已過完的批次；同一天收盤只會觸發一種出場（先看停損，再看停利，最後看到期）；買不到的不算交易，另外列出"}
    for k in ("target", "stop", "expiry"):
        c = sum(1 for t in trades if t.get(key) == k)
        out[k] = {"n": c, "pct": round(c / n * 100, 1) if n else None, "ci95": wilson(c, n)}
    rs = [t[ret] for t in trades if t.get(ret) is not None]
    if rs:
        m = sum(rs) / len(rs)
        sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) if len(rs) > 1 else None
        out["mean_ret"] = round(m, 3)
        out["ci95_ret"] = [round(m - 1.96 * sd / math.sqrt(len(rs)), 3), round(m + 1.96 * sd / math.sqrt(len(rs)), 3)] if sd else None
        out["win_pct"] = round(sum(x > 0 for x in rs) / len(rs) * 100, 1)
    ex = [t["excess"] for t in trades if t.get("excess") is not None]
    out["mean_excess"] = round(sum(ex) / len(ex), 3) if ex else None
    return out


TAGS = {"pick": "選股：比同期 0050 差", "entry": "進場：開盤比訊號日收盤高 3% 以上才買到（追高）",
        "hold": "持有：曾經賺 10% 以上，最後卻虧損出場", "exit": "出場：停損後 10 個交易日內又漲回買進價以上",
        "risk": "風控：虧損超過 12%（停損沒守住，例如跳空）", "cost": "成本：扣成本前賺錢、扣成本後虧錢"}


def review(trades: list[dict], px: dict, days: list[str]) -> list[dict]:
    """每一筆虧損的已出場交易，檢查是哪個環節出問題（可以同時有好幾個）。只做紀錄，不會改策略。"""
    out = []
    for t in trades:
        if t.get("net_ret") is None or t["net_ret"] >= 0:
            continue
        tags, notes = [], {}
        if t.get("bench_ret") is not None and t["net_ret"] < t["bench_ret"]:
            tags.append("pick")
        if t.get("signal_close") and t.get("entry_open") and t["entry_open"] > t["signal_close"] * 1.03:
            tags.append("entry")
        if t.get("max_close") and t["max_close"] >= t["entry_price"] * 1.10:
            tags.append("hold")
        if t.get("exit_kind") == "stop":
            after = [d for d in days if d > t["exit_date"]][:10]
            if len(after) < 10:
                notes["exit"] = "停損後還不到 10 個交易日，之後再判斷"
            elif any(((px.get(t["sid"]) or {}).get(d) or (0, 0, 0, 0))[3] > t["entry_price"] for d in after):
                tags.append("exit")
        if t["net_ret"] < -12:
            tags.append("risk")
        if t.get("gross_ret") is not None and t["gross_ret"] > 0:
            tags.append("cost")
        out.append({"id": t["id"], "mode": t["mode"], "sid": t["sid"], "name": t["name"], "net_ret": t["net_ret"],
                    "exit_reason": t["exit_reason"], "tags": tags, "labels": [TAGS[x] for x in tags], "notes": notes})
    return out


def research_items(ledger: dict | None = None, path: Path | None = None, min_n: int = 3) -> list[dict]:
    """把模擬交易的虧損原因整理成研究清單項目（同一個原因 ≥ 3 筆才列）。只是待驗證的假設，不會自動改正式策略。"""
    L = ledger if ledger is not None else _read(path or OUT / "ledger.json")
    out = []
    for m, B in ((L or {}).get("books") or {}).items():
        cnt: dict[str, int] = {}
        for r in B.get("review", []):
            for x in r["tags"]:
                cnt[x] = cnt.get(x, 0) + 1
        for x, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            if c >= min_n:
                out.append({"kind": "paper", "key": f"{m}:{x}",
                            "text": f"模擬交易{MODE_NAMES.get(m, m)}有 {c} 筆虧損屬於「{TAGS[x]}」：列入研究，用樣本外資料驗證後才考慮調整，不會自動改策略"})
    return out


# ---------------- 整本重算＋每日指紋 ----------------
def build(days: list[str], px: dict, signals: dict, exr: dict | None = None, idx: dict | None = None,
          hol: set[str] | None = None, picks: Path = PICKS) -> dict:
    books = {}
    for m in MODES:
        B = simulate(m, days, px, signals, exr)
        eq = [c["equity"] for c in B["curve"]]
        closed = B["trades"]
        B["kpi"] = {"equity": eq[-1] if eq else CASH0, "cash": B["cash"], "ret": round((eq[-1] / CASH0 - 1) * 100, 3) if eq else 0.0,
                    "mdd": mdd(eq), "n_open": len(B["positions"]), "n_closed": len(closed),
                    "realized": round(sum(t["pnl"] for t in closed), 0),
                    "fees": round(sum(t["buy_fee"] + t["sell_fee"] for t in closed) + sum(p["buy_fee"] + p["sell_fee"] for p in B["positions"]), 0),
                    "taxes": round(sum(t["sell_tax"] for t in closed) + sum(p["sell_tax"] for p in B["positions"]), 0),
                    "unfilled": _count([o for o in B["orders"] if o["status"] != "filled"])}
        B["odds"] = odds(closed, days)
        B["review"] = review(closed, px, days)
        books[m] = B
    # 基準：同一筆錢在第一次進場日開盤買 0050；加權指數用收盤價換算
    first = min((b["first_entry"] for b in books.values() if b["first_entry"]), default=None)
    bench = _bench_curve(days, px, first, exr, idx)
    last = days[-1] if days else None
    plan = None
    if last and last >= START_SIGNAL:
        entry = next_trading(last, [], hol)
        sig = signal_for(last, entry, picks) if last not in signals else signals[last]
        if sig:
            plan = {"signal_date": last, "entry_date": entry, "pick_file": sig["file"], "generated_at": sig["generated_at"],
                    "model_version": sig.get("model_version"),
                    "buys": {m: [{"sid": c["stock_id"], "name": c.get("name"), "rank": c.get("rank"), "score": c.get("score")}
                                 for c in (sig["lists"].get(m) or []) if c["stock_id"] not in {p["sid"] for p in books[m]["positions"]}
                                 ][:max(0, MAX_POS - len(books[m]["positions"]) + len(books[m]["pending_sells"]))] for m in MODES},
                    "sells": {m: books[m]["pending_sells"] for m in MODES}}
    return {"schema_version": 1, "rules_version": RULES_VERSION, "rules": RULES_TEXT, "live_trading": LIVE_TRADING,
            "start_signal": START_SIGNAL, "last_data": last, "first_entry": first, "books": books, "bench": bench, "plan": plan,
            "signals": {d: {k: s[k] for k in ("file", "generated_at", "model_version", "model_hash", "data_dates")} for d, s in signals.items()},
            "note": "這是依固定規則的模擬交易（前瞻紀錄，從 2026-10-28 開盤開始），不是回測，也沒有送出任何真實委託。"}


def _count(orders):
    c: dict[str, int] = {}
    for o in orders:
        c[o["why"] or "?"] = c.get(o["why"] or "?", 0) + 1
    return c


def _bench_curve(days, px, first, exr, idx) -> dict:
    out = {"first": first, "curve": []}
    if not first:
        return out
    r = px.get(BENCH) or {}
    b0 = r.get(first)
    i0 = (idx or {}).get(days[days.index(first) - 1]) if days.index(first) else None
    if b0 and b0[0]:
        price = b0[0] * (1 + SLIP)
        sh = int(CASH0 / (price * (1 + FEE)))
        cash = CASH0 - sh * price - fee(sh * price, sh)
        out["shares"], out["price"] = sh, round(price, 4)
    for d in days[days.index(first):]:
        row = {"date": d}
        if b0 and b0[0]:
            e = exr.get((BENCH, d)) if exr else None
            if e and d > first and e[0] and e[1] and e[0] > e[1]:
                cash += (e[0] - e[1]) * sh
            cl = _last_close(px, BENCH, d)
            row["e0050"] = round(cash + sh * cl, 0) if cl else None
        if i0 and (idx or {}).get(d):
            row["taiex"] = round(CASH0 * idx[d] / i0, 0)
        out["curve"].append(row)
    for k in ("e0050", "taiex"):
        v = [x[k] for x in out["curve"] if x.get(k)]
        out[k] = {"ret": round((v[-1] / CASH0 - 1) * 100, 3) if v else None, "mdd": mdd(v)}
    out["note"] = "0050：第一個進場日開盤用同樣 100 萬元買進（含手續費、滑價）放著；加權指數：從前一天收盤換算，不含成本。"
    return out


def fingerprint(L: dict, d: str) -> dict:
    """某一天每本帳的現金、總資產、持股（指紋），用來檢查之後重算有沒有變。"""
    out = {}
    for m, B in L["books"].items():
        c = next((x for x in B["curve"] if x["date"] == d), None)
        if not c:
            continue
        held = sorted((t["sid"], t["entry_date"]) for t in B["trades"] + B["positions"]
                      if t["entry_date"] <= d and (t.get("exit_date") or "9999") > d)
        out[m] = {"equity": c["equity"], "cash": c["cash"], "held": held,
                  "hash": hashlib.sha256(json.dumps([c["equity"], c["cash"], held]).encode()).hexdigest()[:12]}
    return out


def update_days(L: dict, path: Path) -> dict:
    """days.json 只新增、不改舊的；重算結果和以前記的不同，就記到 mismatches（例如官方修正了價格）。"""
    D = _read(path) or {"rules_version": L["rules_version"], "days": {}, "mismatches": []}
    days = sorted({c["date"] for B in L["books"].values() for c in B["curve"]})
    for d in days:
        fp = fingerprint(L, d)
        old = D["days"].get(d)
        if old is None:
            D["days"][d] = {"recorded_at": config.now().isoformat(timespec="seconds"), "rules_version": L["rules_version"], **fp}
        else:
            for m, v in fp.items():
                if (old.get(m) or {}).get("hash") not in (None, v["hash"]):
                    key = (d, m, v["hash"])
                    if not any((x["date"], x["mode"], x["new_hash"]) == key for x in D["mismatches"]):
                        D["mismatches"].append({"date": d, "mode": m, "old_hash": old[m]["hash"], "new_hash": v["hash"],
                                                "old_equity": old[m]["equity"], "new_equity": v["equity"],
                                                "seen_at": config.now().isoformat(timespec="seconds"),
                                                "rules_changed": old.get("rules_version") != L["rules_version"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(D, ensure_ascii=False, indent=1), "utf-8")
    return D


# ---------------- AI 預測回放：只用封存紀錄 ----------------
def replay(days: list[str], px: dict, idx: dict, picks: Path = PICKS, top: int = REPLAY_TOP, hol: set[str] | None = None,
           names: dict | None = None) -> dict:
    """每個封存日的首次發布版名單（和績效追蹤同一份），前 10 名：當時的分數、模型版本、資料日期，
    之後照同一套模擬規則（停損／停利／20 日到期）看結果，和同期 0050、加權指數比。這是事後對帳，不是當時的模擬交易。"""
    out = []
    files = sorted(f for f in picks.glob("????-??-??.json")) if picks.exists() else []
    for f in files:
        s = f.stem
        if s < REPLAY_START:
            continue
        x = _read(f) or {}
        entry = next_trading(s, days, hol)
        g = x.get("generated_at") or ""
        official = bool(x.get("rec_picks"))
        lists = _enrich(x) if official else (x.get("picks") or {})
        day = {"date": s, "file": f.name, "generated_at": g, "entry": entry, "on_time": bool(g) and g < f"{entry}T{OPEN_AT}",
               "model_version": x.get("model_version"), "model_hash": x.get("model_hash"), "data_dates": x.get("data_dates"),
               "list_kind": "正式精選" if official else "原始排名（當時還沒有正式精選）", "modes": {}}
        for m in MODES:
            rows = []
            for c in (lists.get(m) or [])[:top]:
                rows.append({"sid": c["stock_id"], "name": c.get("name") or (names or {}).get(c["stock_id"]), "rank": c.get("rank"), "score": c.get("score"),
                             "close": c.get("close"), "factors": c.get("factors"), "reason": _why(c), **_outcome(c["stock_id"], s, entry, days, px, idx)})
            done = [r for r in rows if r.get("done")]
            day["modes"][m] = {"picks": rows, "n_done": len(done),
                               "avg_net": round(sum(r["net_ret"] for r in done) / len(done), 3) if done else None}
        out.append(day)
    allx = {m: [r | {"exit_kind": r["kind"]} for d in out if d["on_time"] for r in d["modes"][m]["picks"] if r.get("done") and r.get("kind")] for m in MODES}
    unf = {m: sum(1 for d in out if d["on_time"] for r in d["modes"][m]["picks"] if r.get("unfilled")) for m in MODES}
    return {"schema_version": 1, "days": out[::-1], "odds": {m: odds(v, days) | {"unfilled": unf[m]} for m, v in allx.items()},
            "rule": "每天首次發布、開盤前封存的名單前 10 名；隔一個交易日開盤買，之後照模擬交易規則："
                    "收盤跌 10% 停損、漲 20% 停利、滿 20 個交易日到期，隔天開盤賣；扣成本 0.585%。0050 同一段期間；加權指數用收盤價。",
            "note": "回放只用當時封存的紀錄，結果是事後對帳；同一天的 10 檔、相鄰幾天的名單會重疊，筆數不等於獨立樣本。不是當時的模擬交易，也不是回測。"}


def _outcome(sid, s, entry, days, px, idx) -> dict:
    r = px.get(sid) or {}
    if entry not in days:
        return {"status": "還沒進場", "done": False}
    b = r.get(entry)
    if not b or not b[4]:
        return {"status": "進場日沒有成交，買不到", "done": False, "unfilled": "no_trade"}
    pc = _prev_close(px, sid, entry, {})
    if b[0] == b[1] == b[2] and pc and b[0] >= pc * (1 + LIMIT):
        return {"status": "漲停鎖死，買不到", "done": False, "unfilled": "limit_up"}
    buy = b[0] * (1 + SLIP)
    i0 = days.index(entry)
    kind, k, exit_d = None, 0, None
    for j, d in enumerate(days[i0:]):
        bar = r.get(d)
        if not bar or not bar[3]:
            continue
        k = j + 1
        cl = bar[3]
        kind = "stop" if cl <= buy * (1 + STOP) else "target" if cl >= buy * (1 + TARGET) else "expiry" if k >= HOLD else None
        if kind:
            exit_d = d
            break
    sell_d = next((d for d in days if d > exit_d and r.get(d) and r[d][4]), None) if exit_d else None
    last = max((d for d in r if d <= days[-1] and r[d][3]), default=entry)
    if not sell_d:
        cl = r[last][3]
        net = (cl * (1 - SLIP)) / buy * (1 - REPLAY_COST) - 1
        return {"status": f"進行中（第 {days.index(last) - i0 + 1} 天）" if not exit_d else f"{REASONS[kind]}，等下一個交易日開盤賣",
                "done": False, "net_ret": round(net * 100, 2), "as_of": last,
                "bench_ret": _bench_close(px, entry, last), "taiex_ret": _idx_ret(idx, days, entry, last)}
    sell = r[sell_d][0] * (1 - SLIP)
    net = sell / buy * (1 - REPLAY_COST) - 1
    br = _bench_ret(px, entry, sell_d)
    return {"status": REASONS[kind], "kind": kind, "done": True, "entry": entry, "exit": sell_d, "net_ret": round(net * 100, 2),
            "bench_ret": br, "excess": round(net * 100 - br, 2) if br is not None else None, "taiex_ret": _idx_ret(idx, days, entry, sell_d)}


def _bench_close(px, a, b):
    r = px.get(BENCH) or {}
    if a in r and b in r and r[a][0]:
        return round((r[b][3] / r[a][0] - 1) * 100, 2)
    return None


def _idx_ret(idx, days, a, b):
    i = days.index(a)
    p = idx.get(days[i - 1]) if i else None
    return round((idx[b] / p - 1) * 100, 2) if p and idx.get(b) else None


# ---------------- 讀資料、寫檔 ----------------
def load(dbname: str = "stocks.db", since: str = "2026-09-01"):
    from . import db
    with db.connect(dbname) as con:
        idx = dict(con.execute("SELECT date, close FROM index_prices WHERE date>=? ORDER BY date", (since,)).fetchall())
        days = sorted(idx)
        rows = con.execute("SELECT stock_id, date, open, high, low, close, volume FROM prices WHERE date>=?", (since,)).fetchall()
        exr = {(s, d): (p, r) for s, d, p, r in con.execute("SELECT stock_id, date, prev_close, ref_price FROM exrights WHERE date>=?", (since,))}
        names = dict(con.execute("SELECT stock_id, name FROM stocks").fetchall())
    px: dict = {}
    for s, d, o, h, lo, c, v in rows:
        px.setdefault(s, {})[d] = (o, h, lo, c, v)
    return days, px, exr, idx, names


def holidays() -> set[str]:
    return set((_read(HOL) or {}).get("dates", [])) | set((_read(NOTES) or {}).keys())


def signals_from(days: list[str], hol: set[str], picks: Path = PICKS, names: dict | None = None) -> dict:
    out = {}
    for d in days:
        if d < START_SIGNAL:
            continue
        s = signal_for(d, next_trading(d, days, hol), picks)
        if s:
            for rows in s["lists"].values():
                for c in rows:
                    c["name"] = c.get("name") or (names or {}).get(c["stock_id"])
            out[d] = s
    return out


def site(L: dict, R: dict, D: dict) -> dict:
    """網站版：每本帳的重點、持股、全部交易、委託、資產曲線（取需要的欄位）。"""
    books = {}
    for m, B in L["books"].items():
        books[m] = {"name": MODE_NAMES[m], "kpi": B["kpi"], "odds": B["odds"], "review": B["review"],
                    "positions": B["positions"], "trades": B["trades"][::-1], "orders": B["orders"][::-1][:200],
                    "curve": B["curve"], "pending_sells": B["pending_sells"]}
    return {"generated_at": config.now().isoformat(timespec="seconds"), "rules_version": L["rules_version"], "rules": L["rules"],
            "live_trading": L["live_trading"], "last_data": L["last_data"], "first_entry": L["first_entry"], "start_signal": L["start_signal"],
            "books": books, "bench": L["bench"], "plan": L["plan"], "note": L["note"],
            "integrity": {"days_recorded": len(D["days"]), "mismatches": D["mismatches"][-20:], "n_mismatch": len(D["mismatches"])},
            "cost": "全部在 GitHub Actions 免費額度內執行（每次幾秒），不使用付費資料或服務，每月 NT$0。"}


def run(dbname: str = "stocks.db", out: Path = OUT, site_dir: Path | None = None) -> dict:
    if broker.LIVE_TRADING:   # 雙重保險：真實下單打開時，模擬交易拒絕執行
        raise broker.LiveTradingDisabled("第一階段只做模擬交易")
    days, px, exr, idx, names = load(dbname)
    hol = holidays()
    sig = signals_from(days, hol, names=names)
    L = build(days, px, sig, exr, idx, hol)
    R = replay(days, px, idx, hol=hol, names=names)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ledger.json").write_text(json.dumps(L, ensure_ascii=False, indent=1), "utf-8")
    D = update_days(L, out / "days.json")
    S = site(L, R, D)
    (out / "site.json").write_text(json.dumps(S, ensure_ascii=False), "utf-8")
    (out / "replay.json").write_text(json.dumps(R, ensure_ascii=False), "utf-8")
    return S


def main(argv: list[str]) -> int:
    S = run()
    print(json.dumps({"last_data": S["last_data"], "first_entry": S["first_entry"], "plan": bool(S["plan"]),
                      "books": {m: B["kpi"] for m, B in S["books"].items()}, "mismatch": S["integrity"]["n_mismatch"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
