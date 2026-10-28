"""V21 選股品質總檢討（事前登記的研究）→ history/backtest/selection.json。

回答使用者 10/28 的三個問題：AI 高分股是不是真的比較會漲？為什麼常常輸 0050？積極型怎麼改才有證據？

事前固定（2026-10-28 寫下，看結果前定案，之後不能看結果再加減）：
- 期間：2019-08 起，每 20 個交易日一個訊號日（彼此不重疊），訊號日收盤後的分數，隔一個交易日開盤買，
  再過 20 個交易日開盤賣（還原除權息），和網站、到期對帳同一套 hold_return；買不到的那一份留現金。
- 成本：新買進的部位才扣 0.585%（和既有回測一致）。基準 0050（同一段開盤到開盤）。
- 開發段：訊號日 < 2025-01-01；保留段：訊號日 ≥ 2025-01-01，只在最後看一次。
  說明：保留段的月份過去其他研究（risk、race）看過，但本研究的候選規則是第一次跑，規則在跑之前寫定。
- 研究一（高分股有沒有比較會漲）：三種模式的分數分十組，每組 20 日超額報酬；前 5／10／20／50 名；Rank IC；
  依市場多空、流動性分層。
- 研究二（為什麼輸 0050）：把「前 10 名 − 0050」拆成「全部可排名股票等權平均 − 0050」（大盤結構，例如台積電權重）
  和「前 10 名 − 等權平均」（選股能力）。
- 研究三（五大因子）：單一因子前 10% 減等權平均、Rank IC。
- 研究四（產業集中）：前 10 名同一產業 ≥ 30% 的期間 vs 其他期間。
- 研究五（候選規則，主要對象積極型，穩健／均衡同時列出但只當參考）：見 CHALLENGERS。
  門檻（全部要過）：開發段逐期差異 t ≥ T_NEED（8 個候選的 Bonferroni，單尾 5%）、開發段前後兩半都 > 0、
  保留段平均差異 > 0、最大回撤不比原版差超過 5 個百分點。過了也只進影子追蹤，正式模型要使用者同意才換。
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ai_model, config
from . import pit_backtest as pb
from .factor_research import REGIME_MA, _cap_pick

STUDY_VERSION = "2026-10-28"
OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "selection.json"
START = "2019-08-01"
STEP = 20
HOLDOUT_FROM = "2025-01-01"
T_NEED = 2.6
MDD_GAP = 5.0
FACTORS = {"quality": "品質", "growth": "成長", "value": "價值", "momentum": "動能", "trend": "趨勢"}
TOPS = (5, 10, 20, 50)
BASE_N = 10
BUFFER_RANK = 30      # BUF：已持有的股票排名還在前 30 名就續抱
BIG_N = 150           # BIG：只在市值前 150 大裡挑
IND_CAP2 = 0.2        # IND2：同產業最多 2 檔（10 檔的 20%）
GROWTH_MIN = 60       # GROW：成長分數至少 60
CHALLENGERS = {
    "T5": "只買前 5 名",
    "T20": "買前 20 名",
    "BUF": f"前 10 名，但已持有的股票排名還在前 {BUFFER_RANK} 名就續抱（降低換手成本）",
    "BIG": f"只在市值前 {BIG_N} 大的股票裡挑前 10 名",
    "TREND": "只挑股價在季線之上且季線上揚的股票",
    "IND2": "同一產業最多 2 檔",
    "GROW": f"成長分數至少 {GROWTH_MIN}",
    "BUFBIG": "BUF＋BIG",
}
COLS = ["industry", "market_cap", "avg_value_20", "vol_1y", "ret_3m", "dist_ma20", "dist_ma60", "ma60_slope_20", "base_ok"]

_D = _CFG = None


def signal_dates(dates: list[str], start: str = START, step: int = STEP) -> list[tuple[str, str, str]]:
    """（訊號日, 進場日, 出場日），每 step 個交易日一期，彼此不重疊。"""
    i = next((k for k, d in enumerate(dates) if d >= start), None)
    out = []
    if i is None:
        return out
    i = max(i, 1) - 1
    while i + 1 + step < len(dates):
        out.append((dates[i], dates[i + 1], dates[i + 1 + step]))
        i += step
    return out


def _regime(D: dict, sig: str) -> str | None:
    s = D["idx_s"]
    if s is None:
        return None
    s = s[s.index <= sig].astype(float)
    if len(s) < REGIME_MA:
        return None
    return "bull" if s.iloc[-1] >= s.iloc[-REGIME_MA:].mean() else "bear"


def _period(p: tuple[str, str, str]) -> dict:
    sig, entry, exit_ = p
    D = _D
    rows = pb.scored_rows(D, sig, _CFG)
    recs = []
    for r in rows:
        x = {"id": r["stock_id"], **{c: r.get(c) for c in COLS}, **{f: r.get(f"score_{f}") for f in FACTORS}}
        for m in ai_model.MODES:
            x[f"rk_{m}"] = r["ai"]["modes"][m]["rank"]
            x[f"sc_{m}"] = r["ai"]["modes"][m]["score"]
        if not any(x[f"rk_{m}"] for m in ai_model.MODES):
            continue
        ret, st = pb.hold_return(D, r["stock_id"], entry, exit_)
        x["ret"], x["status"] = ret, st
        recs.append(x)
    return {"signal": sig, "entry": entry, "exit": exit_, "bench": pb._bench(D, entry, exit_), "regime": _regime(D, sig),
            "df": pd.DataFrame(recs).set_index("id") if recs else pd.DataFrame()}


def collect(dbname: str = "stocks.db", workers: int | None = None, log=print) -> list[dict]:
    global _D, _CFG
    t0 = time.time()
    _D, _CFG = pb.load(dbname), config.load()
    ps = signal_dates(_D["dates"])
    log(f"資料讀取 {time.time() - t0:.0f} 秒；{len(ps)} 期")
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    if workers > 1 and sys.platform != "win32":
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            return list(ex.map(_period, ps))
    return [_period(p) for p in ps]


# ---------------- 挑股規則 ----------------
def _ranked(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    return df[df[f"rk_{mode}"].notna()].sort_values(f"rk_{mode}")


def pick(df: pd.DataFrame, mode: str, rule: str, prev: list[str] | None = None) -> list[str]:
    ok = _ranked(df, mode)
    n = {"T5": 5, "T20": 20}.get(rule, BASE_N)
    if rule in ("BIG", "BUFBIG"):
        big = df["market_cap"].dropna().nlargest(BIG_N).index
        ok = ok[ok.index.isin(big)]
    if rule == "TREND":
        ok = ok[(ok["dist_ma60"] > 0) & (ok["ma60_slope_20"] > 0)]
    if rule == "GROW":
        ok = ok[ok["growth"] >= GROWTH_MIN]
    if rule == "IND2":
        return _cap_pick([(s, ok.at[s, "industry"]) for s in ok.index], n, IND_CAP2)
    if rule in ("BUF", "BUFBIG") and prev:
        top = list(ok.index[:BUFFER_RANK])
        keep = [s for s in prev if s in top][:n]
        return keep + [s for s in ok.index if s not in keep][: n - len(keep)]
    return list(ok.index[:n])


def _net(periods: list[dict], picks: list[list[str]]) -> list[float]:
    """每期成本後報酬：等權，買不到的留現金；只對新買進的部位扣成本。"""
    out, prev = [], set()
    for P, ids in zip(periods, picks):
        if not ids:
            out.append(0.0)
            prev = set()
            continue
        r = P["df"]["ret"].reindex(ids)
        got = r.notna()
        new = [s for s in ids if s not in prev and got.get(s, False)]
        out.append(float(r.fillna(0).sum() / len(ids) - pb.COST * len(new) / len(ids)))
        prev = set(ids)
    return out


def run_rule(periods: list[dict], mode: str, rule: str) -> tuple[list[float], list[list[str]]]:
    picks, prev = [], None
    for P in periods:
        ids = pick(P["df"], mode, rule, prev) if len(P["df"]) else []
        picks.append(ids)
        prev = ids
    return _net(periods, picks), picks


# ---------------- 統計 ----------------
def _t(x: np.ndarray) -> float | None:
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def _mdd(r: list[float]) -> float:
    c = np.cumprod(1 + np.array(r, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], c]))[1:]
    return round(float((c / peak - 1).min()) * 100, 2) if len(c) else 0.0


def _pct(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x) * 100, 2)


def summary(r: list[float], bench: list[float]) -> dict:
    r, b = np.array(r, float), np.array(bench, float)
    ex = r - b
    return {"periods": len(r), "total": _pct(np.prod(1 + r) - 1), "bench_total": _pct(np.prod(1 + b) - 1),
            "avg_excess": _pct(ex.mean()), "t_excess": None if _t(ex) is None else round(_t(ex), 2),
            "beat_0050": round(float((ex > 0).mean()) * 100, 1), "mdd": _mdd(list(r)),
            "big_loss": round(float((r < -0.10).mean()) * 100, 1)}


def deciles(periods: list[dict], mode: str) -> dict:
    """分數十組（第 10 組最高）每組 20 日平均超額（未扣成本，十組一樣，比的是排序能力）。"""
    rows, ics = [], []
    for P in periods:
        df = _ranked(P["df"], mode)
        df = df[df["ret"].notna()]
        if len(df) < 50 or P["bench"] is None:
            continue
        ex = df["ret"] - P["bench"]
        q = pd.qcut(df[f"sc_{mode}"].rank(method="first"), 10, labels=False) + 1
        rows.append(ex.groupby(q).mean())
        ics.append(float(df[f"sc_{mode}"].rank().corr(df["ret"].rank())))
    M = pd.DataFrame(rows)
    ic = np.array(ics)
    top_bottom = (M[10] - M[1]).to_numpy()
    return {"decile_excess": [_pct(M[k].mean()) for k in range(1, 11)],
            "decile_beat_bench": [round(float((M[k] > 0).mean()) * 100, 1) for k in range(1, 11)],
            "top_minus_bottom": _pct(top_bottom.mean()), "t_top_minus_bottom": None if _t(top_bottom) is None else round(_t(top_bottom), 2),
            "ic": round(float(ic.mean()), 4), "ic_t": None if _t(ic) is None else round(_t(ic), 2), "ic_pos": round(float((ic > 0).mean()) * 100, 1),
            "monotonic": bool(all(M[k].mean() <= M[k + 1].mean() for k in range(1, 10)))}


def decompose(periods: list[dict], mode: str, n: int = BASE_N) -> dict:
    """前 n 名 − 0050 ＝（等權平均 − 0050）＋（前 n 名 − 等權平均）。都未扣成本。"""
    a, b, c = [], [], []
    for P in periods:
        df = _ranked(P["df"], mode)
        df = df[df["ret"].notna()]
        if not len(df) or P["bench"] is None:
            continue
        ew, top = df["ret"].mean(), df["ret"].iloc[:n].mean()
        a.append(top - P["bench"]); b.append(ew - P["bench"]); c.append(top - ew)
    a, b, c = map(np.array, (a, b, c))
    return {"top_vs_0050": _pct(a.mean()), "universe_vs_0050": _pct(b.mean()), "t_universe_vs_0050": round(_t(b), 2) if _t(b) else None,
            "universe_beat_0050": round(float((b > 0).mean()) * 100, 1), "selection": _pct(c.mean()),
            "t_selection": round(_t(c), 2) if _t(c) else None, "selection_pos": round(float((c > 0).mean()) * 100, 1)}


def factor_table(periods: list[dict]) -> dict:
    out = {}
    for f, name in FACTORS.items():
        sp, ics = [], []
        for P in periods:
            df = P["df"][P["df"]["rk_balanced"].notna() & P["df"]["ret"].notna() & P["df"][f].notna()]
            if len(df) < 50:
                continue
            k = max(1, len(df) // 10)
            sp.append(df.nlargest(k, f)["ret"].mean() - df["ret"].mean())
            ics.append(float(df[f].rank().corr(df["ret"].rank())))
        sp, ics = np.array(sp), np.array(ics)
        h = len(sp) // 2
        out[f] = {"name": name, "top10pct_vs_universe": _pct(sp.mean()), "t": round(_t(sp), 2) if _t(sp) else None,
                  "ic": round(float(ics.mean()), 4), "ic_t": round(_t(ics), 2) if _t(ics) else None,
                  "first_half": _pct(sp[:h].mean()), "second_half": _pct(sp[h:].mean())}
    return out


def by_layer(periods: list[dict], mode: str) -> dict:
    """前 10 名減等權平均（選股能力），依多空與流動性分層。"""
    reg, liq = {"bull": [], "bear": []}, {"高流動性": [], "中流動性": [], "低流動性": []}
    for P in periods:
        df = _ranked(P["df"], mode)
        df = df[df["ret"].notna()]
        if not len(df):
            continue
        if P["regime"] in reg:
            reg[P["regime"]].append(df["ret"].iloc[:BASE_N].mean() - df["ret"].mean())
        tier = pd.qcut(df["avg_value_20"].rank(method="first"), 3, labels=["低流動性", "中流動性", "高流動性"])
        for t in liq:
            sub = df[tier == t]
            if len(sub) >= BASE_N:
                liq[t].append(float(sub[f"sc_{mode}"].rank().corr(sub["ret"].rank())))
    return {"selection_by_regime": {k: {"mean": _pct(np.mean(v)) if v else None, "n": len(v), "t": round(_t(np.array(v)), 2) if len(v) > 2 and _t(np.array(v)) else None} for k, v in reg.items()},
            "ic_by_liquidity": {k: {"ic": round(float(np.mean(v)), 4) if v else None, "t": round(_t(np.array(v)), 2) if len(v) > 2 and _t(np.array(v)) else None} for k, v in liq.items()}}


def concentration(periods: list[dict], mode: str) -> dict:
    hi, lo = [], []
    for P in periods:
        df = _ranked(P["df"], mode)
        top = df.iloc[:BASE_N]
        if not len(top) or P["bench"] is None:
            continue
        share = top["industry"].fillna("").value_counts().iloc[0] / len(top)
        ex = top["ret"].fillna(0).mean() - P["bench"]
        (hi if share >= 0.3 else lo).append(ex)
    d = None
    if len(hi) > 2 and len(lo) > 2:
        d = float(np.mean(hi) - np.mean(lo))
        se = np.sqrt(np.var(hi, ddof=1) / len(hi) + np.var(lo, ddof=1) / len(lo))
    return {"concentrated_periods": len(hi), "other_periods": len(lo), "concentrated_excess": _pct(np.mean(hi)) if hi else None,
            "other_excess": _pct(np.mean(lo)) if lo else None, "diff": _pct(d), "t": round(d / se, 2) if d is not None and se else None}


def challengers(periods: list[dict], mode: str) -> dict:
    bench = [P["bench"] or 0.0 for P in periods]
    dev = np.array([P["signal"] < HOLDOUT_FROM for P in periods])
    base, base_picks = run_rule(periods, mode, "BASE")
    turnover = lambda pk: round(float(np.mean([len(set(b) - set(a)) / max(len(b), 1) for a, b in zip(pk, pk[1:])])) * 100, 1)
    out = {"BASE": {"rule": "原版前 10 名", **summary(base, bench), "turnover": turnover(base_picks),
                    "dev": summary(list(np.array(base)[dev]), list(np.array(bench)[dev])),
                    "holdout": summary(list(np.array(base)[~dev]), list(np.array(bench)[~dev]))}}
    for k, label in CHALLENGERS.items():
        r, pk = run_rule(periods, mode, k)
        diff = np.array(r) - np.array(base)
        dd = diff[dev]
        h = len(dd) // 2
        t = _t(dd)
        s = summary(r, bench)
        v = {"dev_diff": _pct(dd.mean()), "dev_t": None if t is None else round(t, 2), "first_half": _pct(dd[:h].mean()),
             "second_half": _pct(dd[h:].mean()), "holdout_diff": _pct(diff[~dev].mean()) if (~dev).any() else None,
             "mdd_gap": round(s["mdd"] - out["BASE"]["mdd"], 2)}
        v["passed"] = bool(t is not None and t >= T_NEED and dd[:h].mean() > 0 and dd[h:].mean() > 0
                           and (~dev).any() and diff[~dev].mean() > 0 and v["mdd_gap"] >= -MDD_GAP)
        out[k] = {"rule": label, **s, "turnover": turnover(pk),
                  "dev": summary(list(np.array(r)[dev]), list(np.array(bench)[dev])),
                  "holdout": summary(list(np.array(r)[~dev]), list(np.array(bench)[~dev])), "vs_base": v}
    return out


def analyse(periods: list[dict]) -> dict:
    periods = [P for P in periods if len(P["df"]) and P["bench"] is not None]
    res = {"modes": {}}
    for m in ai_model.MODES:
        res["modes"][m] = {"name": ai_model.MODES[m]["name"], "deciles": deciles(periods, m), "decompose": decompose(periods, m),
                           "layers": by_layer(periods, m), "concentration": concentration(periods, m),
                           "tops": {str(n): summary(run_rule_n(periods, m, n), [P["bench"] for P in periods]) for n in TOPS},
                           "challengers": challengers(periods, m)}
    res["factors"] = factor_table(periods)
    res["passed"] = [f"{m}:{k}" for m in res["modes"] for k, v in res["modes"][m]["challengers"].items() if v.get("vs_base", {}).get("passed")]
    res["periods"] = len(periods)
    res["dev_periods"] = sum(P["signal"] < HOLDOUT_FROM for P in periods)
    res["first"], res["last"] = periods[0]["signal"], periods[-1]["signal"]
    return res


def run_rule_n(periods: list[dict], mode: str, n: int) -> list[float]:
    picks = [list(_ranked(P["df"], mode).index[:n]) for P in periods]
    return _net(periods, picks)


CACHE = config.DATA_DIR / "selection_periods.pkl"   # 給 academic_study 共用同一份樣本（不進 git）


def run(dbname: str = "stocks.db", log=print) -> dict:
    t0 = time.time()
    periods = collect(dbname, log=log)
    try:
        import pickle
        CACHE.write_bytes(pickle.dumps(periods))
    except OSError as e:
        log(f"樣本快取寫入失敗：{e}")
    res = analyse(periods)
    res.update({"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds") + "+08:00",
                "step": STEP, "holdout_from": HOLDOUT_FROM, "t_need": T_NEED, "mdd_gap": MDD_GAP, "cost_pct": round(pb.COST * 100, 3),
                "challenger_rules": CHALLENGERS, "data_end": _D["dates"][-1], "seconds": round(time.time() - t0),
                "note": "歷史回測（研究），不是前瞻績效；正式模型不因本研究自動改變。"})
    return res


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({m: v["challengers"]["BASE"] for m, v in res["modes"].items()}, ensure_ascii=False))
    print("通過：", res["passed"] or "沒有")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
