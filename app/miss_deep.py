"""V22 失準根因（事前登記：docs/prereg_v22.md 第 1 節）→ history/research/miss_deep.json。

對每一份到期對帳檔的正式名單、以及模擬交易目前的持倉，逐檔列出「當時」看得到的東西：
為什麼高分（五因子）、營收與財報有沒有支持、價格是否過熱、進場是否跳空、產業與相關性、訊號日以前的新聞與處置、
持有期間的結果。再把同一天全部可排名股票依同一個特徵分組，看「這個特徵當天是否和下跌有關」（只有一天，只是描述）。

這裡不設計排除名單。每一個看起來像原因的特徵，都在 app/perf_study.py 用 87 期歷史另外驗證。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import pit_backtest as pb

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "history" / "audit" / "maturity"
PICKS = ROOT / "history" / "picks"
PAPER = ROOT / "history" / "paper" / "site.json"
OUT = ROOT / "history" / "research" / "miss_deep.json"
STUDY_VERSION = "2026-11-01"
GAP = 0.03           # 和 perf_study C1 同一個門檻
BIG_LOSS = -0.10
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}


def _px(D, sid, d, k="close"):
    a = D["adj"].get(sid)
    if a is None or d not in a.index:
        return None
    v = float(a.at[d, k])
    return v if v > 0 and not np.isnan(v) else None


def price_state(D: dict, sid: str, sig: str, entry: str) -> dict:
    """訊號日收盤時的價格狀態＋進場日開盤跳空（開盤時已知）。"""
    dates = D["dates"]
    i = dates.index(sig)
    c = [_px(D, sid, d) for d in dates[max(0, i - 250):i + 1]]
    c = pd.Series([x for x in c if x is not None], dtype=float)
    out = {"pre5": None, "pre20": None, "dist_ma20": None, "rsi14": None, "vol20": None, "from_hi250": None, "gap_entry": None}
    if len(c) >= 21:
        out["pre5"] = round((c.iloc[-1] / c.iloc[-6] - 1) * 100, 2)
        out["pre20"] = round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2)
        out["dist_ma20"] = round((c.iloc[-1] / c.iloc[-20:].mean() - 1) * 100, 2)
        r = c.pct_change().dropna()
        out["vol20"] = round(float(r.iloc[-20:].std() * np.sqrt(250) * 100), 1)
        d = c.diff().iloc[-14:]
        up, dn = d.clip(lower=0).mean(), -d.clip(upper=0).mean()
        out["rsi14"] = round(float(100 - 100 / (1 + up / dn)) if dn > 0 else 100.0, 1)
        out["from_hi250"] = round((c.iloc[-1] / c.max() - 1) * 100, 2)
    o, pc = _px(D, sid, entry, "open"), _px(D, sid, sig)
    if o and pc:
        out["gap_entry"] = round((o / pc - 1) * 100, 2)
    return out


def fundamentals(con, sid: str, sig: str) -> dict:
    """訊號日當時已公布的月營收（次月 10 日前）與本益比。"""
    if con is None:
        return {}
    y, m = int(sig[:4]), int(sig[5:7])
    day = int(sig[8:10])
    k = y * 12 + m - 1 - (1 if day > 10 else 2)          # 最新已公布月份
    ym = f"{k // 12}-{k % 12 + 1:02d}"
    rows = dict(con.execute("SELECT ym, revenue FROM revenue WHERE stock_id=? AND ym<=? ORDER BY ym DESC LIMIT 15", (sid, ym)).fetchall())
    def yoy(a):
        b = f"{int(a[:4]) - 1}{a[4:]}"
        return None if not rows.get(a) or not rows.get(b) else round((rows[a] / rows[b] - 1) * 100, 1)
    last = max(rows) if rows else None
    per = con.execute("SELECT per FROM valuation WHERE stock_id=? AND date<=? AND per>0 ORDER BY date DESC LIMIT 750", (sid, sig)).fetchall()
    per = [x[0] for x in per]
    disp = con.execute("SELECT period, reason FROM disposition WHERE stock_id=? AND updated<=?", (sid, f"{sig}T15:00:00")).fetchall()
    return {"rev_month": last, "rev_yoy": yoy(last) if last else None,
            "per": per[0] if per else None,
            "per_pct_3y": round(float((np.array(per) <= per[0]).mean()) * 100, 0) if len(per) >= 120 else None,
            "disposition": [{"period": p, "reason": r} for p, r in disp][:2]}


def news(con, sid: str, sig: str) -> list[dict]:
    if con is None:
        return []
    try:
        rows = con.execute("SELECT date, title, source FROM news_items WHERE stock_id=? AND date<=? AND (fetched_at IS NULL OR fetched_at<=?) "
                           "ORDER BY date DESC LIMIT 6", (sid, sig, f"{sig}T15:00:00")).fetchall()
    except sqlite3.Error:
        return []
    return [{"date": d, "title": t, "source": s} for d, t, s in rows]


_XS: dict = {}


def cross_section(D: dict, uni: list[str], sig: str, entry: str, exit_: str) -> dict:
    if (sig, exit_) not in _XS:
        _XS[(sig, exit_)] = _cross_section(D, uni, sig, entry, exit_)
    return _XS[(sig, exit_)]


def _cross_section(D: dict, uni: list[str], sig: str, entry: str, exit_: str) -> dict:
    """同一天全部可排名股票：依各特徵分三組，持有期間平均報酬（只有一天，只是描述）。"""
    rows = []
    for sid in uni:
        r, st = pb.hold_return(D, sid, entry, exit_)
        if r is None:
            continue
        ps = price_state(D, sid, sig, entry)
        rows.append({"id": sid, "ret": r, **ps})
    df = pd.DataFrame(rows)
    out = {"n": len(df)}
    if df.empty:
        return out
    for k in ("pre20", "dist_ma20", "vol20", "gap_entry", "rsi14"):
        d = df[["ret", k]].dropna()
        if len(d) < 60:
            continue
        q = pd.qcut(d[k].rank(method="first"), 3, labels=["低", "中", "高"])
        out[k] = {g: round(float(d.loc[q == g, "ret"].mean()) * 100, 2) for g in ("低", "中", "高")}
        out[k]["rank_corr"] = round(float(d[k].rank().corr(d["ret"].rank())), 3)
    return out


def corr60(D: dict, ids: list[str], sig: str) -> float | None:
    dates = D["dates"]
    i = dates.index(sig)
    R = pd.DataFrame({s: [_px(D, s, d) for d in dates[max(0, i - 60):i + 1]] for s in ids}).astype(float).pct_change().dropna(how="all")
    C = R.corr().values
    v = C[np.triu_indices_from(C, 1)]
    return round(float(np.nanmean(v)), 3) if len(v) else None


def deep_batch(D: dict, rec: dict, picks: dict, con, mode: str) -> dict:
    sig, entry, exit_ = rec["signal_date"], rec["entry"], rec["exit"]
    lst = ((picks.get("rec_picks") or picks.get("picks") or {}).get(mode) or [])[:10]
    full = {x["stock_id"]: x for x in (picks.get("picks") or {}).get(mode) or []}
    b = pb._bench(D, entry, exit_)
    dates = D["dates"]
    win = dates[dates.index(entry):dates.index(exit_) + 1]
    info = D.get("info") or {}
    rows = []
    for x in lst:
        sid = x["stock_id"]
        f = full.get(sid) or x
        r, st = pb.hold_return(D, sid, entry, exit_)
        p0 = _px(D, sid, entry, "open")
        path = [None if p0 is None or _px(D, sid, d) is None else round((_px(D, sid, d) / p0 - 1) * 100, 2) for d in win]
        row = {"id": sid, "name": f.get("name") or (info.get(sid) or {}).get("name"), "industry": f.get("industry") or (info.get(sid) or {}).get("industry"),
               "rank": x.get("rank"), "score": x.get("score"), "factors": f.get("factors"),
               **price_state(D, sid, sig, entry), **fundamentals(con, sid, sig),
               "ret": None if r is None else round(r * 100, 2),
               "net": None if r is None else round((r - pb.COST) * 100, 2),
               "excess": None if r is None or b is None else round((r - pb.COST - b) * 100, 2),
               "path": path, "worst_close": min([p for p in path if p is not None], default=None),
               "news_before_signal": news(con, sid, sig)}
        row["big_loss"] = row["net"] is not None and row["net"] <= BIG_LOSS * 100
        row["flags"] = flags(row)
        rows.append(row)
    cnt = Counter(r["industry"] or "" for r in rows)
    top_ind, n_ind = cnt.most_common(1)[0] if cnt else (None, 0)
    cl = [r for r in rows if r["industry"] == top_ind]
    nets = [r["net"] for r in rows if r["net"] is not None]
    return {"signal_date": sig, "entry": entry, "exit": exit_, "horizon": rec["horizon"], "mode": mode, "name": MODES[mode],
            "bench_0050": None if b is None else round(b * 100, 2),
            "mean_net": round(float(np.mean(nets)), 2) if nets else None,
            "stocks": rows,
            "concentration": {"industry": top_ind, "n": n_ind, "cluster_mean_net": round(float(np.mean([r["net"] for r in cl if r["net"] is not None])), 2) if cl else None,
                              "others_mean_net": round(float(np.mean([r["net"] for r in rows if r["industry"] != top_ind and r["net"] is not None])), 2)
                              if any(r["industry"] != top_ind for r in rows) else None,
                              "avg_corr_60d": corr60(D, [r["id"] for r in rows], sig)},
            "flag_counts": dict(Counter(f for r in rows for f in r["flags"])),
            "big_losses": [r["id"] for r in rows if r["big_loss"]],
            "cross_section": cross_section(D, picks.get("universe_ids") or [], sig, entry, exit_)}


def flags(r: dict) -> list[str]:
    """事前看得到的警訊（門檻和 perf_study 一致；只是標記，不是排除規則）。"""
    out = []
    if r.get("gap_entry") is not None and r["gap_entry"] >= GAP * 100:
        out.append("進場跳空≥3%")
    if r.get("pre20") is not None and r["pre20"] >= 20:
        out.append("訊號前20日漲≥20%")
    if r.get("dist_ma20") is not None and r["dist_ma20"] >= 10:
        out.append("高於20日均線≥10%")
    if r.get("rsi14") is not None and r["rsi14"] >= 75:
        out.append("RSI≥75")
    if r.get("vol20") is not None and r["vol20"] >= 60:
        out.append("20日波動≥60%")
    if r.get("rev_yoy") is not None and r["rev_yoy"] < 0:
        out.append("最新月營收年減")
    if r.get("per_pct_3y") is not None and r["per_pct_3y"] >= 90:
        out.append("本益比在自己3年前10%高")
    if r.get("disposition"):
        out.append("處置股")
    return out


def paper_view(D: dict, con) -> dict:
    """模擬交易目前持倉：進場跳空與帳面損益。"""
    if not PAPER.exists():
        return {}
    p = json.loads(PAPER.read_text("utf-8"))
    out = {}
    for m, bk in (p.get("books") or {}).items():
        pos = []
        for x in bk.get("positions") or []:
            gap = None if not x.get("signal_close") or not x.get("entry_open") else round((x["entry_open"] / x["signal_close"] - 1) * 100, 2)
            pos.append({"id": x["sid"], "name": x.get("name"), "industry": x.get("industry"), "signal_date": x.get("signal_date"),
                        "entry_date": x.get("entry_date"), "gap_entry": gap, "unreal_ret": x.get("unreal_ret"), "status": x.get("status"),
                        "reason": x.get("reason")})
        g = [q["gap_entry"] for q in pos if q["gap_entry"] is not None]
        out[m] = {"ret": (bk.get("kpi") or {}).get("ret"), "positions": pos,
                  "gap_ge3": sum(1 for v in g if v >= GAP * 100), "mean_gap": round(float(np.mean(g)), 2) if g else None,
                  "closed": (bk.get("kpi") or {}).get("n_closed")}
    out["bench"] = {k: (p.get("bench") or {}).get(k, {}).get("ret") if isinstance((p.get("bench") or {}).get(k), dict) else None for k in ("e0050", "taiex")}
    return out


def run(dbname: str = "stocks.db", D: dict | None = None, focus: str | None = None) -> dict:
    D = D or pb.load(dbname)
    con = sqlite3.connect(config.DATA_DIR / dbname) if (config.DATA_DIR / dbname).exists() else None
    batches = []
    files = sorted(MAT.glob("????-??-??_h*.json"))
    for p in files:
        rec = json.loads(p.read_text("utf-8"))
        kf = PICKS / ((rec.get("official") or {}).get("file") or f"{rec['signal_date']}.json")
        if not kf.exists():
            continue
        x = json.loads(kf.read_text("utf-8"))
        for m in MODES:
            batches.append(deep_batch(D, rec, x, con, m))
    # 三種模式在所有到期批次的警訊比例（重疊批次，只是描述）
    by_mode = {}
    for m in MODES:
        S = [s for b in batches if b["mode"] == m for s in b["stocks"] if s["net"] is not None]
        if not S:
            continue
        fl = Counter(f for s in S for f in s["flags"])
        bl = [s for s in S if s["big_loss"]]
        by_mode[m] = {"stocks": len(S), "big_loss_pct": round(len(bl) / len(S) * 100, 1),
                      "flag_pct": {k: round(v / len(S) * 100, 1) for k, v in fl.most_common()},
                      "big_loss_flag_pct": {k: round(v / max(1, len(bl)) * 100, 1) for k, v in Counter(f for s in bl for f in s["flags"]).most_common()},
                      "mean_gap": round(float(np.nanmean([s["gap_entry"] for s in S if s["gap_entry"] is not None])), 2),
                      "mean_pre20": round(float(np.nanmean([s["pre20"] for s in S if s["pre20"] is not None])), 2)}
    return {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
            "batches": batches, "by_mode": by_mode, "paper": paper_view(D, con),
            "limits": ["5 日批次每天一批、持有期重疊，不是獨立樣本", "警訊只是標記；能不能用要看 perf_study 的 87 期歷史",
                       "新聞只有 2026-10-11 以後；處置股紀錄只有目前抓到的", "產業用證交所分類，PCB／銅箔基板／載板都在「電子零組件業」"]}


def main(argv: list[str]) -> int:
    import pickle
    D = pickle.loads(Path(argv[0]).read_bytes()) if argv else None
    res = run(D=D)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps(res["by_mode"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
