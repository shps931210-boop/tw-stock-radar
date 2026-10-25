"""V14 投資組合轉化研究（只做研究，不改正式模型）：同一份 LightGBM 排序，換不同的「選股與持股方式」，看能不能變成跑贏 0050 的組合。

事前固定（2026-10-25 定案，看結果前寫好，之後不能加減）：
- 排序來源固定：lgbm_study 的基準 LightGBM（base）；V21 前 10 名只當參考。
- 所有方式共用：同一組測試日、同一個 20 日持有期間（隔天開盤買、第 20 個交易日後開盤賣）、同一個來回成本 track.COST、同一個股票母體、和 0050 比。
- 基準方式：前 10 名等權重（top10）。其他方式都和它比，多重比較用 Bonferroni：|t| ≥ T_NEED 才算有證據。
- 保留期（2024-09 之後）已經在之前的研究看過很多次，不再當成獨立證據；真正沒看過的只有之後的真實前瞻封存。
  所以另外做「逐期樣本外選方式」：每一期只用之前已經到期的期間，挑平均成績最好的方式來用，看這個挑法本身的成績。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_race as mr
from . import track

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "portfolio_study.json"
STUDY_VERSION = "2026-10-25"
SCORE = "base"
ENSEMBLE = ("base", "w_25", "w_50", "w_100", "t_excess", "t_riskadj")
MIN_VALUE = 3e7          # 流動性 D 級以上（E、F 級不交易），和 app/liquidity.py 一致
IND_MAX = 2
BUFFER = 30
VOL_CUT = 0.8
MIN_PAST = 12            # 逐期樣本外挑選：至少要有 12 期已到期的紀錄才開始挑
COST_MULT = (0.0, 1.0, 2.0, 3.0)
VARIANTS = {
    "top10": "前 10 名等權重（基準）",
    "top5": "前 5 名等權重",
    "top15": "前 15 名等權重",
    "top20": "前 20 名等權重",
    "uncert": "前 10 名，模型意見越一致權重越高（不確定性加權）",
    "liq_w": "前 10 名，成交金額越大權重越高（流動性加權）",
    "inv_vol": "前 10 名，波動越小權重越高（波動率風控）",
    "vol_cut": "先排除波動最大 20% 的股票，再取前 10 名",
    "ind_cap": "前 10 名，同一產業最多 2 檔",
    "eligible": "模型排全部股票，只從可交易（流動性 D 級以上）的股票取前 10 名",
    "buffer": "前 10 名，但原本持股只要還在前 30 名就續抱（降低換股）",
    "hold60": "前 10 名，每 60 個交易日才換股",
    "entry_close": "前 10 名，改用隔天收盤買、第 20 個交易日後收盤賣",
    "entry_t2": "前 10 名，晚一天進場（第 2 天開盤買）",
}
T_NEED = 2.89            # 雙尾 5% ÷ 13 個比較（Bonferroni）


def _w_equal(top: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / len(top), index=top["sid"].values)


def _norm(w: np.ndarray, sids) -> pd.Series:
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones(len(w))
    return pd.Series(w / w.sum(), index=list(sids))


def select(g: pd.DataFrame, v: str, prev: pd.Series | None) -> pd.Series:
    """一個訊號日要持有哪些股票、各多少權重（回傳 sid → 權重）。"""
    g = g[g[SCORE].notna()].sort_values(SCORE, ascending=False)
    if v in ("top5", "top15", "top20"):
        return _w_equal(g.head(int(v[3:])))
    if v == "uncert":
        top = g.head(10)
        cols = [c for c in ENSEMBLE if c in g]
        pr = g[cols].rank(pct=True).loc[top.index]
        return _norm(1 / (0.05 + pr.std(axis=1).to_numpy()), top["sid"])
    if v == "liq_w":
        top = g.head(10)
        return _norm(np.exp(top["avg_value_20"].to_numpy() / 2), top["sid"])
    if v == "inv_vol":
        top = g.head(10)
        vol = top["vol_1y"].fillna(top["vol_1y"].median()).clip(lower=g["vol_1y"].quantile(0.05))
        return _norm(1 / vol.to_numpy(), top["sid"])
    if v == "vol_cut":
        return _w_equal(g[~(g["vol_1y"] > g["vol_1y"].quantile(VOL_CUT))].head(10))
    if v == "ind_cap":
        keep, cnt = [], {}
        for i, r in g.iterrows():
            k = r["industry"] or "?"
            if cnt.get(k, 0) < IND_MAX:
                keep.append(i)
                cnt[k] = cnt.get(k, 0) + 1
            if len(keep) == 10:
                break
        return _w_equal(g.loc[keep])
    if v == "eligible":
        return _w_equal(g[g["avg_value_20"] >= math.log(MIN_VALUE)].head(10))
    if v == "buffer":
        top30 = list(g.head(BUFFER)["sid"])
        held = [s for s in (prev.index if prev is not None else []) if s in top30][:10]
        fill = [s for s in g["sid"] if s not in held][: 10 - len(held)]
        return pd.Series(0.1, index=held + fill)
    return _w_equal(g.head(10))


def _turnover(w: pd.Series, prev: pd.Series | None) -> float:
    if prev is None:
        return 1.0
    a = w.reindex(w.index.union(prev.index), fill_value=0) - prev.reindex(w.index.union(prev.index), fill_value=0)
    return float(a.abs().sum() / 2)


def entry_labels(D: dict, dates: list[str], picks: dict, rule: str) -> dict:
    """換進場價規則時重算報酬（只算被選到的股票）。回傳 {(訊號日, sid): 超額報酬}。"""
    from . import pit_backtest as pb
    out = {}
    pos = {d: i for i, d in enumerate(dates)}
    for sig, sids in picks.items():
        i = pos[sig]
        if rule == "entry_t2":
            e, x = i + 2, i + 2 + mr.H
            if x >= len(dates):
                continue
            b = pb._bench(D, dates[e], dates[x])
            for s in sids:
                r, _ = pb.hold_return(D, s, dates[e], dates[x])
                out[(sig, s)] = r - b if r is not None and b is not None else None
        else:   # entry_close：隔天收盤買、第 1+H 天收盤賣
            e, x = i + 1, i + 1 + mr.H
            if x >= len(dates):
                continue
            ba = D["adj"].get(pb.BENCH)
            b = float(ba.loc[dates[x], "close"] / ba.loc[dates[e], "close"] - 1) if ba is not None and dates[e] in ba.index and dates[x] in ba.index else None
            for s in sids:
                a = D["adj"].get(s)
                if a is None or dates[e] not in a.index or not a.loc[dates[e], "close"]:
                    out[(sig, s)] = None
                    continue
                held = a[(a.index >= dates[e]) & (a.index <= dates[x])]
                r = float(held["close"].iloc[-1] / a.loc[dates[e], "close"] - 1)
                out[(sig, s)] = r - b if b is not None else None
    return out


def book(preds: pd.DataFrame, v: str, dates_used: list[str], B: dict, alt: dict | None = None, cost_mult: float = 1.0) -> dict:
    """依時間逐期持有；每期的扣成本超額報酬（相對 0050）。"""
    y = "y60" if v == "hold60" else f"y{mr.H}"
    rows, prev = [], None
    G = {d: g for d, g in preds.groupby("date")}
    for d in dates_used:
        g = G.get(d)
        if g is None or g[SCORE].notna().sum() < 20:
            continue
        w = select(g, "top10" if v.startswith("entry_") else v, prev)
        r = g.set_index("sid")[y].reindex(w.index)
        if alt is not None:
            r = pd.Series([alt.get((d, s)) for s in w.index], index=w.index, dtype=float)
        r = r.fillna(r.mean() if r.notna().any() else 0.0)   # 買不到的股票：當成持有其他持股的平均（和 model_race 等權重平均一致）
        to = _turnover(w, prev)
        net = float((w * r).sum()) - track.COST * cost_mult * to
        b, tx = B.get(d, (None, None))
        rows.append({"date": d, "net": net, "turnover": to, "n": int(len(w)), "abs": net + b if b is not None else None, "bench": b, "taiex": tx,
                     "maxw": float(w.max())})
        prev = w
    if not rows:
        return {"periods": 0}
    x = np.array([r["net"] for r in rows])
    ab = [r["abs"] for r in rows if r["abs"] is not None]
    bb = [r["bench"] for r in rows if r["bench"] is not None]
    tx = [r["taiex"] for r in rows if r["taiex"] is not None]
    t = mr._t(x)
    yrs = len(rows) * (60 if v == "hold60" else mr.H) / 245
    tot = float(np.prod([1 + a for a in ab]) - 1) if ab else None
    btot = float(np.prod([1 + a for a in bb]) - 1) if bb else None
    return {"periods": len(rows), "net_excess": round(float(x.mean()) * 100, 2), "t": round(t, 2) if t is not None else None,
            "beat_0050": round(float((x > 0).mean()) * 100, 1), "turnover": round(float(np.mean([r["turnover"] for r in rows])) * 100, 1),
            "total": round(tot * 100, 1) if tot is not None else None, "bench_total": round(btot * 100, 1) if btot is not None else None,
            "taiex_total": round(float(np.prod([1 + a for a in tx]) - 1) * 100, 1) if tx else None,
            "cagr": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if tot is not None and yrs > 0 else None,
            "bench_cagr": round(((1 + btot) ** (1 / yrs) - 1) * 100, 2) if btot is not None and yrs > 0 else None,
            "mdd": mr._mdd(ab), "bench_mdd": mr._mdd(bb), "years": round(yrs, 1),
            "series": [round(v * 100, 3) for v in x], "dates": [r["date"] for r in rows]}


def _diff(a: dict, b: dict) -> dict:
    da, db_ = dict(zip(a.get("dates", []), a.get("series", []))), dict(zip(b.get("dates", []), b.get("series", [])))
    common = [d for d in da if d in db_]
    if len(common) < 3:
        return {"diff": None, "t": None, "n": len(common)}
    d = np.array([da[k] - db_[k] for k in common])
    t = mr._t(d)
    return {"diff": round(float(d.mean()), 2), "t": round(t, 2) if t is not None else None, "n": len(common)}


def walk_select(R: dict, names: list[str]) -> dict:
    """逐期樣本外挑方式：第 k 期只看第 k 期以前已經到期（前一期以前）的成績，挑平均最高的方式。"""
    ser = {v: dict(zip(R[v]["dates"], R[v]["series"])) for v in names}
    dates = R["top10"]["dates"]
    out, picks = [], []
    for k, d in enumerate(dates):
        past = dates[: max(0, k - 1)]   # 上一期（還沒到期）不能用
        if len(past) < MIN_PAST:
            continue
        best = max(names, key=lambda v: np.mean([ser[v][p] for p in past if p in ser[v]] or [-1e9]))
        if d in ser[best]:
            out.append(ser[best][d] - ser["top10"][d])
            picks.append(best)
    x = np.array(out)
    t = mr._t(x / 100) if len(x) else None
    return {"periods": len(x), "vs_top10": round(float(x.mean()), 2) if len(x) else None, "t": round(t, 2) if t is not None else None,
            "picked": {v: picks.count(v) for v in sorted(set(picks), key=picks.count, reverse=True)}}


def run(preds: pd.DataFrame, tests: list[str], D: dict, log=print, features: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    dates = D["dates"]
    B = {d: mr._bench(D, dates, d) for d in tests}
    use = tests[0::2]                     # 不重疊的 20 日期間（訊號日約每 10 個交易日一個）
    use60 = tests[0::6]
    B60 = {d: _bench_h(D, dates, d, 60) for d in use60}   # 60 日持有要用 60 日的 0050、加權指數報酬
    G = {d: g for d, g in preds.groupby("date")}
    base_picks = {d: list(select(G[d], "top10", None).index) for d in use if d in G}
    R = {}
    for v in VARIANTS:
        alt = entry_labels(D, dates, base_picks, v) if v.startswith("entry_") else None
        R[v] = book(preds, v, use60, B60, alt) if v == "hold60" else book(preds, v, use, B, alt)
        log(f"  {v}: 每期 {R[v].get('net_excess')}%（t {R[v].get('t')}）")
    R["v21_top10"] = book(preds.assign(**{SCORE: preds["v21"]}), "top10", use, B)
    cost = {v: {f"x{m:g}": book(preds, v, use, B, cost_mult=m).get("net_excess") for m in COST_MULT} for v in VARIANTS if v not in ("hold60",) and not v.startswith("entry_")}
    half = use[len(use) // 2]
    halves = {v: {"前半": _sub(R[v], lambda d: d < half), "後半": _sub(R[v], lambda d: d >= half)} for v in VARIANTS if v != "hold60"}
    comp = [v for v in VARIANTS if v not in ("top10", "hold60")]
    for v in comp:
        R[v]["vs_top10"] = _diff(R[v], R["top10"])
        ok_t = (R[v]["vs_top10"]["t"] or 0) >= T_NEED
        ok_mdd = (R[v].get("mdd") or -1e9) >= (R["top10"].get("mdd") or -1e9) - mr.MDD_GAP
        R[v]["verdict"] = {"passed": bool(ok_t and ok_mdd),
                           "label": "有足夠證據勝過前 10 名等權重" if ok_t and ok_mdd else ("方向較好但證據不足" if (R[v]["vs_top10"]["diff"] or 0) > 0 else "沒有比較好")}
    R["hold60"]["note"] = "每期 60 個交易日，和其他方式的期間長度不同：比年化報酬和總報酬，不做 t 比較"
    sel = walk_select(R, [v for v in VARIANTS if v != "hold60"])
    from . import robustness as rb
    reg = rb.regime4(D.get("idx_s"))
    regimes = {v: _by_regime(R[v], reg) for v in ("top10", "top5", "top20", "buffer", "inv_vol", "ind_cap", "v21_top10")}
    loss = loss_attribution(preds, use, features)
    passed = [v for v in comp if R[v].get("verdict", {}).get("passed")]
    best = max((v for v in VARIANTS if v != "hold60"), key=lambda v: R[v].get("net_excess") or -1e9)
    res = {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": __import__("app.config", fromlist=["now"]).now().isoformat(timespec="seconds"), "score": SCORE, "horizon": mr.H, "cost_pct": round(track.COST * 100, 3), "t_need": T_NEED,
           "first": use[0], "last": use[-1], "periods": len(use), "split": half, "names": {**VARIANTS, "v21_top10": "V21 前 10 名等權重（正式模型，參考）"},
           "results": {k: {kk: vv for kk, vv in r.items()} for k, r in R.items()}, "cost": cost, "halves": halves, "walk_select": sel, "regimes": regimes, "loss_attribution": loss,
           "passed": passed, "best_in_sample": best, "seconds": round(time.time() - t0),
           "conclusion": ("、".join(VARIANTS[v] for v in passed) + " 通過門檻，下一步放進影子預測做前瞻比較；正式精選不變") if passed else
           "沒有任何選股方式在事前門檻下勝過前 10 名等權重；正式精選的組合方式不變",
           "rules": f"14 種方式同一組測試日、成本、母體；和前 10 名等權重比，|t| ≥ {T_NEED}（Bonferroni）且最大回撤不能差超過 {mr.MDD_GAP:g} 個百分點才算通過。"
                    "2024-09 之後的保留期之前已經看過，不當作獨立證據；「逐期樣本外挑選」只用每一期之前的資料決定要用哪一種方式。"}
    log(f"投資組合研究：{res['conclusion']}（{res['seconds']} 秒）")
    return res


REG_NAME = {"up_calm": "多頭・平穩", "up_wild": "多頭・波動大", "down_calm": "空頭・平穩", "down_wild": "空頭・波動大"}
BIG_LOSS = -0.10
LOSS_FEATS = {"vol_1y": "一年波動", "ret_3m": "近 3 個月漲幅", "mom_12_1": "12-1 個月動能", "earnings_yield": "盈餘殖利率", "dist_ma60": "離季線距離",
              "rsi": "RSI", "avg_value_20": "成交金額", "market_cap": "市值", "rev_yoy": "月營收年增", "beta": "Beta"}


def _by_regime(r: dict, reg: dict) -> dict:
    """同一個選股方式在四種市場環境（訊號日當天已知）的每期扣成本超額。"""
    acc: dict = {}
    for d, v in zip(r.get("dates", []), r.get("series", [])):
        k = reg.get(d)
        if k:
            acc.setdefault(k, []).append(v)
    out = {}
    for k, xs in acc.items():
        x = np.array(xs)
        t = mr._t(x / 100)
        out[k] = {"name": REG_NAME.get(k, k), "periods": len(x), "net_excess": round(float(x.mean()), 2), "t": round(t, 2) if t is not None else None}
    return out


def loss_attribution(preds: pd.DataFrame, dates: list[str], features: pd.DataFrame | None = None) -> dict:
    """重大損失歸因（研究模板）：前 10 名裡 20 日落後 0050 超過 10% 的持股，和其他前 10 名持股比，選股當天的特徵有什麼不同。
    特徵用當天全市場百分位（訊號日已知），差距用 t 檢定；只描述共同特徵，不代表因果。"""
    y = f"y{mr.H}"
    P = preds[preds["date"].isin(dates)]
    if features is not None and len(features):
        cols = [c for c in LOSS_FEATS if c in features and c not in P]
        P = P.merge(features[["date", "sid", *cols]], on=["date", "sid"], how="left")
    feats = [c for c in LOSS_FEATS if c in P]
    for c in feats:
        P[f"p_{c}"] = P.groupby("date")[c].rank(pct=True)
    top = pd.concat([g.nlargest(10, SCORE) for _, g in P.groupby("date")]) if len(P) else P
    top = top[top[y].notna()]
    bad = top[y] < BIG_LOSS
    out = {"picks": int(len(top)), "big_loss": int(bad.sum()), "share": round(float(bad.mean()) * 100, 1) if len(top) else None, "features": []}
    if bad.sum() < 20:
        out["note"] = "重大損失樣本不到 20 筆，不分析"
        return out
    for c in feats:
        a, b = top.loc[bad, f"p_{c}"].dropna(), top.loc[~bad, f"p_{c}"].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (a.mean() - b.mean()) / se if se > 0 else None
        out["features"].append({"feature": c, "name": LOSS_FEATS[c], "loss_pct": round(float(a.mean()) * 100, 1), "other_pct": round(float(b.mean()) * 100, 1),
                                "t": round(float(t), 2) if t is not None else None})
    out["features"].sort(key=lambda r: -abs(r["t"] or 0))
    sig = [r for r in out["features"] if abs(r["t"] or 0) >= 2.81]   # 雙尾 5% ÷ 10 個特徵
    out["finding"] = ("重大損失的持股在選股當天：" + "；".join(f"{r['name']}的全市場百分位平均 {r['loss_pct']}（其他持股 {r['other_pct']}，t {r['t']}）" for r in sig[:3])
                      + "。這是共同特徵，要再用樣本外排除規則驗證，才能說能降低損失") if sig else "重大損失的持股和其他持股，在選股當天的特徵沒有顯著差異"
    return out


def _bench_h(D: dict, dates: list[str], sig: str, h: int) -> tuple[float | None, float | None]:
    from . import pit_backtest as pb
    pos = dates.index(sig)
    if pos + 1 + h >= len(dates):
        return None, None
    s = D.get("idx_s")
    tx = float(s[dates[pos + h]] / s[sig] - 1) if s is not None and sig in s.index and dates[pos + h] in s.index else None
    return pb._bench(D, dates[pos + 1], dates[pos + 1 + h]), tx


def _sub(r: dict, keep) -> dict:
    x = np.array([s for d, s in zip(r.get("dates", []), r.get("series", [])) if keep(d)])
    if not len(x):
        return {}
    t = mr._t(x / 100)
    return {"periods": len(x), "net_excess": round(float(x.mean()), 2), "t": round(t, 2) if t is not None else None}


def main(argv: list[str]) -> int:
    from . import lgbm_study as ls, pit_backtest as pb
    path = Path(argv[0]) if argv else ls.PREDS
    if not path.exists():
        print("沒有 LightGBM 預測（先跑 lgbm_study），跳過")
        return 0
    k = pd.read_pickle(path)
    panel = None   # 重大損失歸因要用到的特徵（研究資料集）；沒有就只用預測檔裡的欄位
    pp = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "ml_panel.pkl"
    if pp.exists():
        panel = pd.read_pickle(pp)
        panel = panel["panel"] if isinstance(panel, dict) else panel
    res = run(k["preds"], k["tests"], pb.load(), features=panel)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
