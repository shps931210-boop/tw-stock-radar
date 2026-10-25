"""因子有效性研究（V18.2）：先量清楚現在的模型有沒有預測能力，再談優化。

全部沿用 app/pit_backtest 的「不偷看未來」資料重建（每個月最後一個交易日收盤後，只用當時已公布的資料評分），
每個月算一次，檢查：
- Rank IC：因子分數排名和之後報酬排名的相關係數（Spearman）。正值代表分數高的股票之後表現較好；
  台股月資料 0.03～0.05 已算有用，t 值大於 2 才比較不像運氣。
- 產業中性 IC：分數和報酬都先扣掉同產業平均再算，看扣掉「押對產業」之後還有沒有選股能力。
- 五分組：依分數分五組，每組等權平均報酬；第 5 組（分數最高）減第 1 組是多空價差。
- 產業集中度上限：前 N 名每個產業最多 20%／30%／40%，和不設上限比較。只回報，不直接改模型。
- 多空環境：訊號日加權指數在 120 日均線之上算多頭、之下算空頭（訊號日當下就知道，不是事後分類）。
- 期間：5 個交易日、1 個月（到下個月第一個交易日）、3 個月。3 個月的區間互相重疊，t 值用有效樣本數 ÷ 3 修正。

報酬：進場日開盤到出場日開盤、還原除權息；開盤鎖漲停、沒成交、成交金額太小的買不到，不列入。
基準：0050；另列「單純動能（只看過去一年漲幅排名）」作為最簡單的對照組。
現在的三種模型是固定的比較基準，這份研究不會拿來直接調權重（樣本只有十幾個月，直接照結果調會過度擬合）。

執行：python -m app.factor_research（GitHub Actions「AI 歷史回測」每週執行），輸出 history/backtest/factors.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ai_model, config, pit_backtest as pb

METHOD = 1
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "factors.json"
FACTORS = {"quality": "品質", "growth": "成長", "value": "估值", "momentum": "動能", "trend": "趨勢"}
HORIZONS = {"5d": ("5 個交易日", None), "1m": ("1 個月", 1), "3m": ("3 個月", 3)}
CAPS = (None, 0.2, 0.3, 0.4)
TOPS = (10, 20)
REGIME_MA = 120
MIN_N = 30          # 一個月至少要有 30 檔有分數、有報酬才算 IC
MIN_IND = 3         # 產業中性：同產業至少 3 檔才扣產業平均，不足的歸到「其他」


def _ic(score: pd.Series, ret: pd.Series) -> float | None:
    ok = score.notna() & ret.notna()
    if ok.sum() < MIN_N:
        return None
    a, b = score[ok].rank(), ret[ok].rank()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _demean(x: pd.Series, ind: pd.Series) -> pd.Series:
    return x - x.groupby(ind).transform("mean")


def _ind_groups(ind: pd.Series) -> pd.Series:
    ind = ind.fillna("").replace("", "其他")
    cnt = ind.map(ind.value_counts())
    return ind.where(cnt >= MIN_IND, "其他")


def t_pvalue(t: float, df: int) -> float:
    """雙尾 t 檢定 p 值（Student t 分布，數值積分；不需要 scipy）。"""
    import math
    if df < 1 or t is None or math.isnan(t):
        return float("nan")
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    a, n = abs(t), 2000
    h = a / n
    area = f(0) + f(a) + sum((4 if k % 2 else 2) * f(k * h) for k in range(1, n))
    return max(0.0, min(1.0, 1 - 2 * area * h / 3))


def _nw_se(x: np.ndarray, lag: int) -> float:
    """Newey-West 標準誤：重疊期間（3 個月）的報酬前後相關，要用這個，不能直接 ÷ √n。"""
    n = len(x)
    e = x - x.mean()
    s = float(e @ e) / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * float(e[L:] @ e[:-L]) / n
    return float(np.sqrt(max(s, 1e-12) / n))


def _boot_ci(x: np.ndarray, block: int, reps: int = 2000, seed: int = 7) -> tuple[float, float]:
    """平均 IC 的 95% 信賴區間（區塊 bootstrap；重疊期間用相同長度的區塊抽樣）。"""
    rng = np.random.default_rng(seed)
    n, b = len(x), max(1, block)
    starts = np.arange(n - b + 1)
    k = int(np.ceil(n / b))
    means = [np.concatenate([x[s:s + b] for s in rng.choice(starts, k)])[:n].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summ(ics: list, months_eff: float = 1) -> dict:
    x = np.array([v for v in ics if v is not None], float)
    if len(x) < 3:
        return {"n": int(len(x)), "ic": None}
    sd = float(x.std(ddof=1))
    lag = int(months_eff) - 1
    se = _nw_se(x, lag) if lag > 0 else sd / np.sqrt(len(x))
    t = float(x.mean()) / se if se else None
    lo, hi = _boot_ci(x, int(months_eff))
    return {"n": int(len(x)), "ic": round(float(x.mean()), 4), "ic_sd": round(sd, 4),
            "ir": round(float(x.mean()) / sd, 2) if sd else None,
            "t": round(t, 2) if t is not None else None,
            "p": round(t_pvalue(t, len(x) - 1), 4) if t is not None else None,
            "ci": [round(lo, 4), round(hi, 4)],
            "pos": round(float((x > 0).mean()) * 100, 1)}


def holm(ps: dict) -> dict:
    """Holm 多重檢定修正：同時檢驗很多個因子時，避免「試越多次、越容易碰巧顯著」。"""
    items = sorted((p, k) for k, p in ps.items() if p is not None and not np.isnan(p))
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 4)
    return out


def _quint(score: pd.Series, ret: pd.Series) -> list | None:
    ok = score.notna() & ret.notna()
    if ok.sum() < MIN_N:
        return None
    q = pd.qcut(score[ok].rank(method="first"), 5, labels=False)
    return [float(ret[ok][q == k].mean()) for k in range(5)]


BANDS = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]   # AI 分數區間（分數是全市場相對排名，不是機率）


def _bands(score: pd.Series, ret: pd.Series, bench: float | None) -> list | None:
    """每個分數區間：平均超額（相對 0050）、跑贏 0050 的比例、檔數。用來檢查「分數越高、之後表現越好」是否成立。"""
    ok = score.notna() & ret.notna()
    if ok.sum() < MIN_N or bench is None:
        return None
    sc, ex = score[ok], ret[ok] - bench
    out = []
    for lo, hi in BANDS:
        m = (sc >= lo) & (sc < hi)
        out.append(None if m.sum() == 0 else {"ex": float(ex[m].mean()), "hit": float((ex[m] > 0).mean()), "n": int(m.sum())})
    return out


def band_summary(rows: list) -> list:
    """把每月的分數區間結果合併：平均每期超額、t 值（每期一個數字）、跑贏比例、總檔次。"""
    out = []
    for i, (lo, hi) in enumerate(BANDS):
        xs = [r[i] for r in rows if r and r[i]]
        ex = np.array([x["ex"] for x in xs])
        sd = float(ex.std(ddof=1)) if len(ex) > 1 else 0.0
        out.append({"band": f"{lo}–{min(hi, 100)}" if hi <= 100 else f"{lo} 以上", "periods": len(xs),
                    "n": int(sum(x["n"] for x in xs)),
                    "excess": round(float(ex.mean()) * 100, 2) if len(ex) else None,
                    "t": round(float(ex.mean() / (sd / np.sqrt(len(ex)))), 2) if len(ex) > 1 and sd else None,
                    "hit": round(float(np.mean([x["hit"] for x in xs])) * 100, 1) if xs else None})
    return out


def _cap_pick(ranked: list[tuple[str, str]], n: int, cap: float | None) -> list[str]:
    """依排名往下挑，每個產業最多 n×cap 檔（無產業資料的各自算一類）。"""
    lim = n if cap is None else max(1, int(round(n * cap)))
    out, used = [], {}
    for sid, ind in ranked:
        k = ind or sid
        if used.get(k, 0) >= lim:
            continue
        used[k] = used.get(k, 0) + 1
        out.append(sid)
        if len(out) == n:
            break
    return out


def _regime(D: dict, sig: str) -> str | None:
    s = D["idx_s"]
    if s is None:
        return None
    s = s[s.index <= sig].astype(float)
    if len(s) < REGIME_MA:
        return None
    return "bull" if s.iloc[-1] >= s.iloc[-REGIME_MA:].mean() else "bear"


def run(dbname: str = "stocks.db", start: str | None = None, limit: int | None = None, log=print) -> dict:
    t0 = time.time()
    cfg = config.load()
    D = pb.load(dbname)
    start = start or pb.auto_start(D)
    dates = D["dates"]
    pos = {d: i for i, d in enumerate(dates)}
    rb = pb.rebalance_dates(dates, start)
    periods = list(zip(rb, rb[1:]))
    if limit:
        periods = periods[-limit:]
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(periods)} 個完整月份")
    modes = list(ai_model.MODES)
    keys = [*FACTORS, *modes]
    ic = {h: {k: {"raw": [], "neu": []} for k in keys} for h in HORIZONS}
    qs = {h: {k: [] for k in keys} for h in HORIZONS}
    reg_ic = {k: {"bull": [], "bear": []} for k in keys}
    bands = {m: [] for m in modes}
    port = {(m, n, c): {"rets": [], "hit": [], "ind_max": [], "invested": [], "skip": {}, "prev": set()} for m in [*modes, "momentum"] for n in TOPS for c in CAPS}
    bench, months = [], []
    for j, ((sig, entry), (_, exit1)) in enumerate(periods):
        rows = pb.scored_rows(D, sig, cfg)
        df = pd.DataFrame([{"id": r["stock_id"], "ind": r.get("industry") or "", "base": bool(r.get("base_ok")),
                            **{f: r.get(f"score_{f}") for f in FACTORS},
                            **{m: r["ai"]["modes"][m]["score"] for m in modes},
                            **{f"rk_{m}": r["ai"]["modes"][m]["rank"] for m in modes}} for r in rows]).set_index("id")
        df[list(FACTORS) + modes] = df[list(FACTORS) + modes].apply(pd.to_numeric, errors="coerce")
        df["g"] = _ind_groups(df["ind"])
        k1 = rb.index((sig, entry))
        ex = {"5d": dates[pos[entry] + 5] if pos[entry] + 5 < len(dates) else None, "1m": exit1,
              "3m": rb[k1 + 3][1] if k1 + 3 < len(rb) else None}
        rets = {}
        for h, e in ex.items():
            hr = {sid: (pb.hold_return(D, sid, entry, e) if e else (None, "no_exit")) for sid in df.index}
            rets[h] = pd.Series({sid: x[0] for sid, x in hr.items()}, dtype=float)
            if h == "1m":
                st1 = {sid: x[1] for sid, x in hr.items()}
        b = pb._bench(D, entry, exit1)
        reg = _regime(D, sig)
        bench.append(b)
        months.append({"signal": sig, "entry": entry, "exit": exit1, "regime": reg, "bench": None if b is None else round(b * 100, 2),
                       "stocks": int(len(df)), "base_ok": int(df["base"].sum())})
        for h in HORIZONS:
            r = rets[h]
            if r.notna().sum() < MIN_N:
                continue
            rn = _demean(r, df["g"])
            for k in keys:
                # 因子：通過基本門檻的股票；模型分數：通過該模型硬性門檻（有排名）的股票
                mask = df["base"] if k in FACTORS else df[f"rk_{k}"].notna()
                s = df[k].where(mask)
                v = _ic(s, r)
                ic[h][k]["raw"].append(v)
                ic[h][k]["neu"].append(_ic(_demean(s, df["g"]), rn))
                qs[h][k].append(_quint(s, r))
                if h == "1m" and reg:
                    reg_ic[k][reg].append(v)
                if h == "1m" and k in bands:
                    bands[k].append(_bands(s, r, b))
        # 前 N 名組合（1 個月持有，和 AI 歷史回測同一套交易規則）
        r1 = rets["1m"]
        ranked = {m: [(sid, df.at[sid, "ind"]) for sid in df[df[f"rk_{m}"].notna()].sort_values(f"rk_{m}").index] for m in modes}
        mom = df[df["base"] & df["momentum"].notna()].sort_values("momentum", ascending=False)
        ranked["momentum"] = [(sid, df.at[sid, "ind"]) for sid in mom.index]
        for (m, n, c), P in port.items():
            pick = _cap_pick(ranked[m], n, c)
            w = 1 / n
            got = [(sid, r1.get(sid)) for sid in pick]
            gross = sum(x * w for _, x in got if x is not None and not np.isnan(x))
            cost = sum(pb.COST * w for sid, x in got if x is not None and not np.isnan(x) and sid not in P["prev"])
            P["rets"].append(gross - cost)
            ok = [x for _, x in got if x is not None and not np.isnan(x)]
            P["invested"].append(len([1 for _, x in got if x is not None and not np.isnan(x)]) / n)
            for sid, x in got:
                if x is None or np.isnan(x):
                    P["skip"][st1.get(sid, "not_traded")] = P["skip"].get(st1.get(sid, "not_traded"), 0) + 1
                elif st1.get(sid) == "stale":
                    P["skip"]["stale"] = P["skip"].get("stale", 0) + 1
            P["hit"].append(float(np.mean([x > b for x in ok])) if ok and b is not None else None)
            inds = pd.Series([df.at[sid, "ind"] or sid for sid in pick])
            P["ind_max"].append(float(inds.value_counts().iloc[0] / n) if len(inds) else None)
            P["prev"] = set(pick)
        log(f"{entry}：完成（{time.time() - t0:.0f} 秒）")

    regs = [mo["regime"] for mo in months]

    def factor_block(k: str) -> dict:
        out = {}
        for h, (label, mlen) in HORIZONS.items():
            eff = mlen if mlen and mlen > 1 else 1
            q = [x for x in qs[h][k] if x]
            avg = [round(float(np.mean([x[i] for x in q])) * 100, 2) for i in range(5)] if q else None
            out[h] = {"raw": _summ(ic[h][k]["raw"], eff), "neutral": _summ(ic[h][k]["neu"], eff), "quint": avg,
                      "spread": round(avg[4] - avg[0], 2) if avg else None,
                      "monthly": [None if v is None else round(v, 3) for v in ic[h][k]["raw"]] if h == "1m" else None}
        out["regime"] = {g: _summ(reg_ic[k][g]) for g in ("bull", "bear")}
        if k in bands:   # 分數區間校驗（1 個月持有，超額相對 0050，未扣成本）
            out["bands"] = band_summary(bands[k])
        return out

    def port_block(P: dict) -> dict:
        st = pb._stats(P["rets"], bench)
        ex = {g: [x - b for x, b, rg in zip(P["rets"], bench, regs) if b is not None and rg == g] for g in ("bull", "bear")}
        st["by_regime"] = {g: {"n": len(v), "avg_excess": round(float(np.mean(v)) * 100, 2) if v else None} for g, v in ex.items()}
        h = [x for x in P["hit"] if x is not None]
        st["hit"] = round(float(np.mean(h)) * 100, 1) if h else None
        im = [x for x in P["ind_max"] if x is not None]
        st["ind_max"] = round(float(np.mean(im)) * 100, 1) if im else None
        # 每期真正投入的資金：買不到的那一份留現金（報酬 0），這裡揭露平均投入比例與原因
        st["invested"] = round(float(np.mean(P["invested"])) * 100, 1) if P["invested"] else None
        st["skip"] = P["skip"]
        st["monthly"] = [round(x * 100, 2) for x in P["rets"]]
        return st

    res_f = {k: {"name": FACTORS[k], **factor_block(k)} for k in FACTORS}
    res_m = {m: {"name": ai_model.MODES[m]["name"], **factor_block(m)} for m in modes}
    for h in HORIZONS:  # 同一個期間一起檢驗 8 個項目，做 Holm 修正
        for kind in ("raw", "neutral"):
            adj = holm({k: x[h][kind].get("p") for k, x in {**res_f, **res_m}.items()})
            for k, x in {**res_f, **res_m}.items():
                x[h][kind]["p_holm"] = adj.get(k)
    return {"schema_version": 2, "method": METHOD, "model_version": ai_model.MODEL_VERSION, "model_hash": pb.model_hash(),
            "generated_at": config.now().isoformat(timespec="seconds") + "+08:00", "start": start,
            "data_end": dates[-1] if dates else None, "bench": pb.BENCH, "cost_pct": round(pb.COST * 100, 3),
            "regime_rule": f"訊號日加權指數 ≥ {REGIME_MA} 日均線為多頭", "months": months,
            "horizons": {h: v[0] for h, v in HORIZONS.items()},
            "factors": res_f, "models": res_m,
            "portfolios": {m: {"name": ai_model.MODES[m]["name"] if m in ai_model.MODES else "單純動能（對照組）",
                               **{str(n): {("none" if c is None else str(int(c * 100))): port_block(port[(m, n, c)]) for c in CAPS} for n in TOPS}}
                           for m in [*modes, "momentum"]},
            "bench_stats": pb._stats([b or 0 for b in bench], bench) if bench else None,
            "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    res = run(limit=limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), "utf-8")
    for grp in ("factors", "models"):
        for k, x in res[grp].items():
            a, b = x["1m"]["raw"], x["1m"]["neutral"]
            print(f"{x['name']}：1 個月 Rank IC {a.get('ic')}（t {a.get('t')}，正值月份 {a.get('pos')}%）；產業中性 {b.get('ic')}；五分組價差 {x['1m']['spread']}%")
    for m, x in res["portfolios"].items():
        t = x["10"]
        print(f"{x['name']} 前 10 名 平均超額：不設限 {t['none']['avg_excess']}%，20% {t['20']['avg_excess']}%，30% {t['30']['avg_excess']}%，40% {t['40']['avg_excess']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
