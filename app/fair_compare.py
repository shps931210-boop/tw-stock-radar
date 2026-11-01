"""V22 同時段公平比較 → history/audit/fair_compare.json（網站 data/predictions/maturity/fair.json）。

每一批到期的正式名單，用兩種口徑各算一次，同一個口徑裡選股、0050、加權指數用完全相同的時間點：
1. 開盤到開盤（正式口徑）：進場日開盤 → 出場日開盤。選股與 0050 用還原後開盤價；加權指數用證交所的開盤指數（app/taiex_ohlc.py）。
   沒有加權指數開盤價的日子照實寫「沒有開盤價」，不拿收盤價代替。
2. 收盤到收盤（參考口徑）：訊號日收盤 → 出場前一個交易日收盤。選股、0050、加權指數都用收盤價。
   選股是在訊號日收盤後才產生，這個口徑實際上買不到，只用來和只有收盤價的指數對照。
兩種口徑的選股報酬都扣同樣的來回成本 0.585%；0050、加權指數不扣（基準）。

每批、每個模式列出：成本後報酬、0050、加權指數、相對 0050、相對加權指數、上漲命中率、跑贏 0050／加權指數的比例、
持有期間等權組合的最大回撤、單檔重大虧損（成本後 ≤ −10%）比例與最差一檔。
不改任何到期對帳檔；不合格批次（track.perf_eligible）不列入平均。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from . import config
from . import pit_backtest as pb

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "history" / "audit" / "maturity"
OUT = ROOT / "history" / "audit" / "fair_compare.json"
COST = pb.COST
BIG = -10.0
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}


def _px(D, sid, d, k):
    a = D["adj"].get(sid)
    if a is None or d not in a.index:
        return None
    v = float(a.at[d, k])
    return v if v > 0 and not np.isnan(v) else None


def taiex_ohlc(dbname: str = "stocks.db") -> dict:
    p = config.DATA_DIR / dbname
    if not p.exists():
        return {}
    con = sqlite3.connect(p)
    try:
        return {d: (o, c) for d, o, c in con.execute("SELECT date, open, close FROM index_ohlc")}
    except sqlite3.Error:
        return {}
    finally:
        con.close()


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def basis(D: dict, ids: list[str], a: str, b: str, kind: str, tx: dict, path_days: list[str]) -> dict:
    """kind = 'open'（a 開盤 → b 開盤）或 'close'（a 收盤 → b 收盤）。"""
    nets = {}
    for sid in ids:
        p0, p1 = _px(D, sid, a, kind), _px(D, sid, b, kind)
        if p0 and p1:
            nets[sid] = (p1 / p0 - 1 - COST) * 100
    b0, b1 = _px(D, pb.BENCH, a, kind), _px(D, pb.BENCH, b, kind)
    e50 = None if not b0 or not b1 else (b1 / b0 - 1) * 100
    j = 0 if kind == "open" else 1
    t0, t1 = (tx.get(a) or (None, None))[j], (tx.get(b) or (None, None))[j]
    if kind == "close":     # 收盤用 index_prices（D["idx_s"]），比 index_ohlc 完整
        s = D.get("idx_s")
        t0 = float(s[a]) if s is not None and a in s.index else t0
        t1 = float(s[b]) if s is not None and b in s.index else t1
    etx = None if not t0 or not t1 else (t1 / t0 - 1) * 100
    v = list(nets.values())
    # 持有期間等權組合（每檔從起點價格算）的每日收盤淨值 → 最大回撤
    mdd = None
    if nets and path_days:
        curve = []
        for d in path_days:
            xs = [(_px(D, sid, d, "close") or np.nan) / _px(D, sid, a, kind) for sid in nets]
            xs = [x for x in xs if not np.isnan(x)]
            if xs:
                curve.append(np.mean(xs))
        if curve:
            c = np.array([1.0] + curve)
            mdd = float((c / np.maximum.accumulate(c) - 1).min() * 100)
    return {"net": _r(np.mean(v)) if v else None, "n": len(v), "e0050": _r(e50), "taiex": _r(etx),
            "ex_0050": _r(np.mean(v) - e50) if v and e50 is not None else None,
            "ex_taiex": _r(np.mean(v) - etx) if v and etx is not None else None,
            "up_hit": _r(np.mean([x > 0 for x in v]) * 100, 1) if v else None,
            "beat_0050_hit": _r(np.mean([x > e50 for x in v]) * 100, 1) if v and e50 is not None else None,
            "beat_taiex_hit": _r(np.mean([x > etx for x in v]) * 100, 1) if v and etx is not None else None,
            "mdd": _r(mdd), "big_loss_pct": _r(np.mean([x <= BIG for x in v]) * 100, 1) if v else None,
            "worst": _r(min(v)) if v else None,
            "taiex_missing": etx is None}


def batch(D: dict, r: dict, tx: dict) -> dict | None:
    from . import maturity_report as mr
    g = mr.gate(r)
    off = g.get("official") or {}
    if not off.get("modes"):
        return None
    dates = D["dates"]
    sig, entry, exit_ = r["signal_date"], r["entry"], r["exit"]
    if exit_ not in dates:
        return None
    last = dates[dates.index(exit_) - 1]
    hold_oo = dates[dates.index(entry):dates.index(exit_)]
    hold_cc = dates[dates.index(sig) + 1:dates.index(last) + 1]
    out = {"file": f"{sig}_h{r['horizon']}.json", "signal_date": sig, "horizon": r["horizon"], "eligible": off.get("eligible", True),
           "open_window": [entry, exit_], "close_window": [sig, last], "modes": {}}
    for m, c in off["modes"].items():
        ids = [s["id"] for s in c["stocks"]]
        out["modes"][m] = {"name": MODES.get(m, m), "open": basis(D, ids, entry, exit_, "open", tx, hold_oo),
                           "close": basis(D, ids, sig, last, "close", tx, hold_cc)}
    return out


def aggregate(batches: list[dict]) -> dict:
    agg = {}
    for m in MODES:
        for k in ("open", "close"):
            rows = [b["modes"][m][k] for b in batches if b["eligible"] and m in b["modes"]]
            def avg(key):
                v = [x[key] for x in rows if x.get(key) is not None]
                return _r(np.mean(v)) if v else None
            agg.setdefault(m, {})[k] = {"batches": len(rows), **{key: avg(key) for key in
                                        ("net", "e0050", "taiex", "ex_0050", "ex_taiex", "up_hit", "beat_0050_hit", "beat_taiex_hit", "mdd", "big_loss_pct")},
                                        "taiex_batches": sum(1 for x in rows if x.get("taiex") is not None)}
    return agg


def run(dbname: str = "stocks.db", D: dict | None = None, mat_dir: Path = MAT) -> dict:
    D = D or pb.load(dbname)
    tx = taiex_ohlc(dbname)
    bs = []
    for p in sorted(mat_dir.glob("????-??-??_h*.json")):
        b = batch(D, json.loads(p.read_text("utf-8")), tx)
        if b:
            bs.append(b)
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
            "taiex_open_days": len(tx), "batches": bs[::-1], "average": aggregate(bs),
            "rule": "開盤到開盤（正式口徑）：進場日開盤→出場日開盤，選股、0050、加權指數同一段；加權指數用證交所開盤指數，沒有就照實寫沒有。"
                    "收盤到收盤（參考口徑）：訊號日收盤→出場前一天收盤，三者都用收盤價；選股在收盤後才產生，這個口徑實際上買不到。"
                    "選股都扣來回成本 0.585%，基準不扣。重大虧損＝單檔成本後 ≤ −10%。",
            "note": "相鄰批次持有期重疊（每天一批），平均不是獨立樣本；5 日結果雜訊大，不能代表模型長期能力。"}


def main(argv: list[str]) -> int:
    import pickle
    D = pickle.loads(Path(argv[0]).read_bytes()) if argv else None
    res = run(D=D)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(res["average"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
