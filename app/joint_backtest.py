"""AI 量化選股＋進出場規則聯合回測（V20）：AI 名單加上進場訊號、停損停利之後，報酬到底有沒有比「整個月抱著」好？

候選名單：直接讀 AI 多因子歷史回測（history/backtest/ai_pit.json）每個月用「當時已公布資料」算出的前 N 名，
不重新評分，確保三種做法用的是同一份名單。名單在訊號日收盤後才知道，下一個交易日開盤才能動作。

四種做法（每個模型、前 10／20 名各算一次）：
- 整月持有（對照組）：每月第一個交易日開盤等權買進名單，下個月第一個交易日開盤換股；續抱不扣成本。
- 整月持有＋停損：同上，但進場價 − 2 ATR（訊號日的 14 日 ATR）停損；停損後那一份留現金到下個月。
- 名單內等波段訊號：名單公布期間，名單裡的股票出現「波段」進場訊號才在隔天開盤買；出場完全照波段規則
  （停損、移動停利、跌破季線、持有 40 天），可以跨月。
- 名單內等短線訊號：同上，換成「短線」規則（停損 1.5 ATR、停利 3 ATR、最多 5 天）。
進出場價格、訊號、出場判斷都直接用 app/strategy.py（和個股研究卡、策略回測同一套程式）。

資金：分成 N 份，每筆交易用「當時總資產 ÷ N」（現金不夠就用剩下的現金）；空著的份額就是現金，報告會列出平均現金比例。
成本：買進時一次扣來回成本（手續費 ×2 + 證交稅）。買不到（沒成交、開盤鎖漲停、成交金額太小）就留現金，並記錄原因。
停牌：用最後一筆收盤價計算市值，恢復交易才能賣出。
報告一定同時列出：累積報酬、年化、最大回撤、勝率、交易次數、換手率、成本、平均現金比例、跑輸 0050 的月份。

執行：python -m app.joint_backtest（「AI 歷史回測」每週執行，在 pit_backtest 之後），輸出 history/backtest/joint.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ai_model, config, strategy, track
from . import pit_backtest as pb

METHOD = 1
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "joint.json"
PIT = pb.OUT
TOPS = (10, 20)
STOP_ATR = 2.0
VARIANTS = {"hold": "整月持有（對照組）", "hold_stop": "整月持有＋2 ATR 停損", "swing": "名單內等波段訊號", "short": "名單內等短線訊號"}
COST = strategy.COST


def _frames(D: dict, ids: set) -> dict:
    out = {}
    for sid in ids:
        a = D["adj"].get(sid)
        if a is None or len(a) < 30:
            continue
        f = strategy.frame(a.reset_index())
        e = strategy.entries(f)
        f["sig_swing"] = (e["swing"] & f["liquid"]).to_numpy(bool)
        f["sig_short"] = (e["short"] & f["liquid"]).to_numpy(bool)
        out[sid] = f
    return out


def _buyable(D: dict, sid: str, day: str) -> str | None:
    """進場日能不能買：None = 可以；否則回傳原因（和 AI 歷史回測同一套規則）。"""
    r = D["rawi"].get(sid)
    if r is None or day not in r.index:
        return "not_traded"
    row = r.loc[day]
    if not row["volume"] or not row["open"]:
        return "not_traded"
    if (row["value"] or 0) < pb.THIN_VALUE:
        return "thin"
    prev = r[r.index < day]
    if len(prev) and row["open"] == row["high"] == row["low"] and abs(row["open"] - pb._limit_up(float(prev["close"].iloc[-1]))) < 1e-6:
        return "locked"
    return None


def simulate(D: dict, F: dict, lists: list[dict], n: int, variant: str) -> dict:
    """lists：[{"signal", "entry", "next_signal", "ids"}]（依時間排序）。逐日模擬，回傳每日資產與交易紀錄。"""
    dates = [d for d in D["dates"] if lists[0]["entry"] <= d <= lists[-1]["exit"]]
    idx = {sid: {d: i for i, d in enumerate(f["date"])} for sid, f in F.items()}
    cash, pos = 1.0, {}            # pos[sid] = {"sh", "entry", "stop", "trade"}
    curve, trades, skips, nslot = [], [], {"not_traded": 0, "thin": 0, "locked": 0, "no_slot": 0, "no_data": 0}, 0
    pending = []                   # 波段／短線：前一天收盤出現訊號，今天開盤買
    turnover = 0.0
    li = 0
    last_px = {}

    def px(sid, d, k="close"):
        i = idx.get(sid, {}).get(d)
        if i is None:
            return None
        v = float(F[sid][k].iloc[i])
        return v if v > 0 else None

    def equity(d):
        tot = cash
        for sid, p in pos.items():
            c = px(sid, d) or last_px.get(sid, p["entry"])
            last_px[sid] = c
            tot += p["sh"] * c
        return tot

    def close_pos(sid, price, why, d):
        nonlocal cash
        p = pos.pop(sid)
        cash += p["sh"] * price
        t = p["trade"]
        t.update(exit=d, exit_px=price, why=why, ret=round((p["sh"] * price) / t["alloc"] - 1, 5))
        trades.append(t)

    def open_pos(sid, d, eq, stop=None, mode=None, sig_i=None):
        nonlocal cash, turnover
        why = _buyable(D, sid, d)
        if why:
            skips[why] += 1
            return False
        o = px(sid, d, "open")
        if not o:
            skips["no_data"] += 1
            return False
        alloc = min(eq / n, cash)
        if alloc <= 1e-9:
            skips["no_slot"] += 1
            return False
        cash -= alloc
        turnover += alloc
        pos[sid] = {"sh": alloc * (1 - COST) / o, "entry": o, "stop": stop, "mode": mode, "sig_i": sig_i, "peak": o,
                    "trade": {"sid": sid, "entry": d, "entry_px": o, "alloc": alloc}}
        return True

    for k, d in enumerate(dates):
        # 1. 換月（只有整月持有兩種做法）：開盤賣出不在新名單的、買進新名單裡還沒持有的
        while li + 1 < len(lists) and d >= lists[li + 1]["entry"]:
            li += 1
        cur = lists[li] if d >= lists[li]["entry"] else None
        if variant in ("hold", "hold_stop") and cur and d == cur["entry"]:
            new = cur["ids"][:n]
            for sid in [s for s in pos if s not in new]:
                o = px(sid, d, "open")
                if o:
                    close_pos(sid, o, "換股", d)
                else:  # 換股日停牌或已下市：用最後收盤價計算，之後留現金（和 AI 歷史回測相同）
                    close_pos(sid, last_px.get(sid, pos[sid]["entry"]), "停牌／下市（以最後收盤價計）", d)
            eq = equity(d)
            for sid in new:
                if sid in pos:
                    continue
                stop = None
                if variant == "hold_stop":
                    f = F.get(sid)
                    si = idx.get(sid, {}).get(cur["signal"])
                    atr = float(f["atr"].iloc[si]) if f is not None and si is not None and not np.isnan(f["atr"].iloc[si]) else None
                    o = px(sid, d, "open")
                    stop = o - STOP_ATR * atr if atr and o else None
                open_pos(sid, d, eq, stop=stop)
        # 2. 波段／短線：昨天收盤的訊號今天開盤買
        if variant in ("swing", "short") and pending:
            eq = equity(dates[k - 1]) if k else 1.0
            for sid, si in pending:
                if sid in pos:
                    continue
                if len(pos) >= n:
                    skips["no_slot"] += 1
                    continue
                f = F[sid]
                o = px(sid, d, "open")
                if not o:
                    skips["no_data"] += 1
                    continue
                lv = strategy.levels(variant, o, float(f["atr"].iloc[si]))
                if open_pos(sid, d, eq, stop=lv["stop"], mode=variant, sig_i=si):
                    pos[sid]["target"] = lv["target"] if variant == "short" else None
                    pos[sid]["day"] = 0
            pending = []
        # 3. 盤中出場：停損、停利（跳空用開盤價）；波段的季線、持有期限用收盤價
        for sid in list(pos):
            p = pos[sid]
            i = idx.get(sid, {}).get(d)
            if i is None:
                continue  # 停牌
            f = F[sid]
            o, h, lo, c = (float(f[x].iloc[i]) for x in ("open", "high", "low", "close"))
            if p["mode"]:
                p["day"] += 1
            stop, tgt = p.get("stop"), p.get("target")
            if stop and p["trade"]["entry"] != d and o <= stop:
                close_pos(sid, o, "停損（跳空）", d)
            elif stop and lo <= stop:
                close_pos(sid, stop, "停損", d)
            elif tgt and p["trade"]["entry"] != d and o >= tgt:
                close_pos(sid, o, "停利", d)
            elif tgt and h >= tgt:
                close_pos(sid, tgt, "停利", d)
            elif p["mode"] == "swing" and c < float(f["ma60"].iloc[i]):
                close_pos(sid, c, "跌破季線", d)
            elif p["mode"] and p["day"] >= strategy.RULES[p["mode"]]["max_days"]:
                close_pos(sid, c, "持有期滿", d)
            elif p["mode"] == "swing":  # 移動停利：最高收盤 − 3 ATR，只往上調
                p["peak"] = max(p["peak"], c)
                p["stop"] = max(stop, p["peak"] - strategy.RULES["swing"]["atr_trail"] * float(f["atr"].iloc[i]))
        # 4. 收盤後：名單內的股票出現訊號，明天開盤買
        if variant in ("swing", "short"):
            act = lists[li] if lists[li]["signal"] <= d else None
            if li + 1 < len(lists) and d >= lists[li + 1]["signal"]:
                act = lists[li + 1]  # 新名單在訊號日收盤就公布了
            if act and d < act.get("next_signal", "9999"):
                for sid in act["ids"][:n]:
                    i = idx.get(sid, {}).get(d)
                    if i is not None and sid not in pos and F[sid][f"sig_{variant}"].iloc[i]:
                        pending.append((sid, i))
        eq = equity(d)
        curve.append((d, eq, (eq - cash) / eq if eq else 0))
    # 期末還沒出場的部位：用最後收盤價計算（列為未平倉）
    open_n = len(pos)
    for sid in list(pos):
        close_pos(sid, last_px.get(sid, pos[sid]["entry"]), "回測結束（未平倉）", dates[-1])
    return {"curve": curve, "trades": trades, "skips": skips, "turnover": turnover, "open_at_end": open_n}


def _summary(sim: dict, lists: list[dict], D: dict) -> dict:
    c = pd.Series({d: e for d, e, _ in sim["curve"]})
    inv = np.array([x for _, _, x in sim["curve"]])
    years = len(c) / 245
    peak = c.cummax()
    rets = [t["ret"] for t in sim["trades"]]
    # 每月報酬（換月日開盤之間，用前一日收盤資產近似）和 0050 比較
    months = []
    for L in lists:
        a = c[(c.index >= L["entry"]) & (c.index < L["exit"])]
        if len(a) < 2:
            continue
        b = pb._bench(D, L["entry"], L["exit"])
        start = c[c.index < L["entry"]]
        base = float(start.iloc[-1]) if len(start) else 1.0
        end = c[c.index < L["exit"]]
        r = float(end.iloc[-1]) / base - 1
        months.append({"entry": L["entry"], "ret": round(r * 100, 2), "bench": None if b is None else round(b * 100, 2),
                       "excess": None if b is None else round((r - b) * 100, 2)})
    ex = [m["excess"] for m in months if m["excess"] is not None]
    why = {}
    for t in sim["trades"]:
        why[t["why"]] = why.get(t["why"], 0) + 1
    return {"total": round(float(c.iloc[-1] - 1) * 100, 2), "cagr": round(float(c.iloc[-1] ** (1 / years) - 1) * 100, 2) if years > 0.5 else None,
            "mdd": round(float((c / peak - 1).min()) * 100, 2),
            "calmar": round(float(c.iloc[-1] ** (1 / years) - 1) / abs(float((c / peak - 1).min())), 2) if years > 0.5 and (c / peak - 1).min() < 0 else None,
            "vol_ann": round(float(c.pct_change().std() * np.sqrt(245)) * 100, 2),
            "trades": len(rets), "win": round(float(np.mean([r > 0 for r in rets])) * 100, 1) if rets else None,
            "avg_trade": round(float(np.mean(rets)) * 100, 2) if rets else None,
            "median_trade": round(float(np.median(rets)) * 100, 2) if rets else None,
            "turnover": round(sim["turnover"] / float(c.mean()) / years, 2) if years else None,
            "cost_total": round(sim["turnover"] * COST * 100, 2),
            "invested": round(float(inv.mean()) * 100, 1), "cash": round(float(1 - inv.mean()) * 100, 1),
            "months": len(months), "beat": round(float(np.mean([x > 0 for x in ex])) * 100, 1) if ex else None,
            "avg_excess": round(float(np.mean(ex)), 2) if ex else None,
            "worst_excess": min(ex) if ex else None,
            "lagging": [m for m in months if m["excess"] is not None and m["excess"] < 0],
            "skips": sim["skips"], "exits": why, "open_at_end": sim["open_at_end"], "monthly": months,
            "curve": [[d, round(e, 4)] for d, e, _ in sim["curve"][::5]]}


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    P = json.loads(PIT.read_text("utf-8"))
    D = pb.load(dbname)
    log(f"資料讀取 {time.time() - t0:.0f} 秒")
    tl = P["timeline"]
    out_modes = {}
    for m, M in P["modes"].items():
        picks = M["top"]["20"]["picks"]
        lists = []
        for j, (t, pk) in enumerate(zip(tl, picks)):
            lists.append({"signal": t["signal"], "entry": t["entry"], "exit": t["exit"],
                          "next_signal": tl[j + 1]["signal"] if j + 1 < len(tl) else t["exit"],
                          "ids": [l["id"] for l in (pk.get("legs") or [])]})
        F = _frames(D, {sid for L in lists for sid in L["ids"]})
        out_modes[m] = {"name": M["name"], "top": {}}
        for n in TOPS:
            out_modes[m]["top"][str(n)] = {v: _summary(simulate(D, F, lists, n, v), lists, D) for v in VARIANTS}
            log(f"{M['name']} 前 {n} 名完成（{time.time() - t0:.0f} 秒）")
    bench = pd.Series({d: float(D["adj"][pb.BENCH].loc[d, "close"]) for d in D["dates"]
                       if tl[0]["entry"] <= d <= tl[-1]["exit"] and d in D["adj"][pb.BENCH].index})
    bc = bench / float(D["adj"][pb.BENCH].loc[tl[0]["entry"], "open"])
    return {"schema_version": 1, "method": METHOD, "model_version": P.get("model_version"), "model_hash": P.get("model_hash"),
            "generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "data_end": P.get("data_end"),
            "start": tl[0]["entry"], "end": tl[-1]["exit"], "cost_pct": round(COST * 100, 3), "bench": pb.BENCH,
            "variants": VARIANTS, "stop_atr": STOP_ATR, "modes": out_modes,
            "bench_stats": {"total": round(float(bc.iloc[-1] - 1) * 100, 2), "mdd": round(float((bc / bc.cummax() - 1).min()) * 100, 2),
                            "curve": [[d, round(float(v), 4)] for d, v in list(bc.items())[::5]]},
            "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> int:
    if not PIT.exists():
        print("還沒有 AI 歷史回測結果（history/backtest/ai_pit.json），先執行 python -m app.pit_backtest")
        return 1
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"0050：{res['bench_stats']['total']}%，最大回撤 {res['bench_stats']['mdd']}%")
    for m, x in res["modes"].items():
        for n, V in x["top"].items():
            for v, s in V.items():
                print(f"{x['name']} 前 {n} 名 {VARIANTS[v]}：累積 {s['total']}%，回撤 {s['mdd']}%，交易 {s['trades']} 筆，勝率 {s['win']}%，"
                      f"現金 {s['cash']}%，換手 {s['turnover']} 倍，每月超額 {s['avg_excess']}%，贏 0050 月份 {s['beat']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
