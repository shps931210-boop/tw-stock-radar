"""V21.2 失準歸因：到期的正式名單逐檔拆解「哪裡錯」，再用歷史資料檢查同一類錯誤是否反覆出現 → history/research/miss_review.json。

事前固定（2026-10-29 寫下）：
- 對象：每一份到期對帳檔（history/audit/maturity/）的正式三種模式前 10 名；重點案例是 10/16 穩健型 5 日（−3.64%）。
- 每檔記錄：當時分數與五因子、產業、訊號日前 20 日漲幅、距 20 日均線、持有期間每天的價格路徑（相對進場開盤）、
  同產業（當天可排名股票等權）同期報酬、個股相對同產業、訊號日以前已經抓到的新聞與事件（fetched_at 在訊號日 15:00 以前）。
- 每檔的初步分類（只是描述，不是已證實的原因）：
  產業拖累：個股相對同產業差距在 ±1.5 個百分點以內，而且產業本身落後 0050；
  個股選錯：落後同產業超過 1.5 個百分點；
  追高後回落：訊號日前 20 日已漲超過 15%，持有期間下跌；
  其他／表現正常。
- 集中度：前 10 名同產業最多幾檔、訊號日前 60 日報酬的平均兩兩相關。
- 歷史重複檢查（selection_study 的 87 期、20 日、開盤進出、扣成本）：
  1) 穩健型前 10 名有 3 檔以上同產業的期數，其相對 0050 是否比其他期數差（兩樣本 t）；
  2) 前 10 名裡「同產業 3 檔以上」那一群股票，和同期其他入選股票的差距；
  3) 入選時距 20 日均線最高三分之一的股票，和其他入選股票的差距。
  只回答「是否反覆出現」，不在本研究裡設計排除規則；要改規則必須另外事前登記並通過模型層級門檻。
"""
from __future__ import annotations

import json
import pickle
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
OUT = ROOT / "history" / "research" / "miss_review.json"
STUDY_VERSION = "2026-10-29"
RES_BAND = 1.5     # 百分點
HOT_PRE20 = 0.15
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}


def _close(D, sid, d):
    a = D["adj"].get(sid)
    return float(a.at[d, "close"]) if a is not None and d in a.index and a.at[d, "close"] else None


def _open(D, sid, d):
    a = D["adj"].get(sid)
    return float(a.at[d, "open"]) if a is not None and d in a.index and a.at[d, "open"] else None


def _ret(D, sid, a, b):
    x, y = _open(D, sid, a), _open(D, sid, b)
    return None if x is None or y is None else y / x - 1


def _news(con, sid: str, sig: str) -> list[dict]:
    if con is None:
        return []
    try:
        rows = con.execute("SELECT date, title, source FROM news_items WHERE stock_id=? AND date<=? AND (fetched_at IS NULL OR fetched_at<=?) "
                           "ORDER BY date DESC LIMIT 5", (sid, sig, f"{sig}T15:00:00")).fetchall()
        ev = con.execute("SELECT event_date, event_type, title FROM events WHERE stock_id=? AND first_seen<=? ORDER BY event_date DESC LIMIT 3",
                         (sid, f"{sig}T15:00:00")).fetchall()
    except sqlite3.Error:
        return []
    return [{"date": d, "title": t, "source": s} for d, t, s in rows] + [{"date": d, "title": f"[{k}] {t}", "source": "events"} for d, k, t in ev]


def batch(D: dict, rec: dict, picks: dict, con=None) -> dict:
    """一份到期對帳檔的正式名單逐檔拆解。"""
    dates = D["dates"]
    sig, entry, exit_ = rec["signal_date"], rec["entry"], rec["exit"]
    i = dates.index(sig)
    b = pb._bench(D, entry, exit_)
    win = dates[dates.index(entry):dates.index(exit_) + 1]
    info = D.get("info") or {}
    uni = picks.get("universe_ids") or []
    ind_ret = {}
    for sid in uni:
        r = _ret(D, sid, entry, exit_)
        if r is not None:
            ind_ret.setdefault((info.get(sid) or {}).get("industry") or "", []).append(r)
    out = {"signal_date": sig, "entry": entry, "exit": exit_, "horizon": rec["horizon"], "bench_0050": None if b is None else round(b * 100, 2), "modes": {}}
    for m, label in MODES.items():
        lst = (picks.get("picks") or {}).get(m) or []
        lst = lst[:10]
        if not lst:
            continue
        rows, cnt = [], Counter(x.get("industry") or "" for x in lst)
        for x in lst:
            sid, ind = x["stock_id"], x.get("industry") or ""
            r = _ret(D, sid, entry, exit_)
            c0, c20 = _close(D, sid, sig), _close(D, sid, dates[i - 20]) if i >= 20 else None
            ma20 = np.mean([v for v in (_close(D, sid, d) for d in dates[i - 19:i + 1]) if v]) if i >= 19 else None
            p0 = _open(D, sid, entry)
            path = [None if p0 is None or _close(D, sid, d) is None else round((_close(D, sid, d) / p0 - 1) * 100, 2) for d in win]
            peers = [v for v in ind_ret.get(ind, [])]
            ir = float(np.mean(peers)) if len(peers) >= 3 else None
            row = {"id": sid, "name": x.get("name"), "industry": ind, "rank": x.get("rank"), "score": x.get("score"), "factors": x.get("factors"),
                   "pre20": None if not c0 or not c20 else round((c0 / c20 - 1) * 100, 2),
                   "dist_ma20": None if not c0 or not ma20 else round((c0 / ma20 - 1) * 100, 2),
                   "ret": None if r is None else round(r * 100, 2),
                   "excess_net": None if r is None or b is None else round((r - pb.COST - b) * 100, 2),
                   "industry_ret": None if ir is None else round(ir * 100, 2), "industry_n": len(peers),
                   "vs_industry": None if r is None or ir is None else round((r - ir) * 100, 2),
                   "path": path, "worst_in_hold": min([p for p in path if p is not None], default=None),
                   "news_before_signal": _news(con, sid, sig)}
            row["kind"] = classify(row, b)
            rows.append(row)
        sids = [x["stock_id"] for x in lst]
        R = pd.DataFrame({s: [_close(D, s, d) for d in dates[max(0, i - 60):i + 1]] for s in sids}).pct_change().dropna(how="all")
        C = R.corr().values
        corr = float(np.nanmean(C[np.triu_indices_from(C, 1)])) if len(sids) > 1 else None
        ok = [r for r in rows if r["excess_net"] is not None]
        out["modes"][m] = {"name": label, "mean_excess_net": round(float(np.mean([r["excess_net"] for r in ok])), 2) if ok else None,
                           "stocks": rows, "kinds": dict(Counter(r["kind"] for r in rows)),
                           "max_same_industry": cnt.most_common(1)[0] if cnt else None, "avg_corr_60d": None if corr is None else round(corr, 3)}
    return out


def classify(row: dict, b: float | None) -> str:
    if row["ret"] is None:
        return "買不到"
    if row["pre20"] is not None and row["pre20"] > HOT_PRE20 * 100 and row["ret"] < 0:
        return "追高後回落"
    if row["vs_industry"] is not None and row["vs_industry"] < -RES_BAND:
        return "個股選錯"
    if row["vs_industry"] is not None and abs(row["vs_industry"]) <= RES_BAND and b is not None and row["industry_ret"] is not None \
            and row["industry_ret"] < b * 100:
        return "產業拖累"
    if row["excess_net"] is not None and row["excess_net"] >= 0:
        return "表現正常"
    return "其他"


def _t(a, b=None):
    a = np.asarray(a, float)
    if b is None:
        return None if len(a) < 3 or a.std(ddof=1) == 0 else float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))
    b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return None if se == 0 else float((a.mean() - b.mean()) / se)


def recurrence(periods: list[dict], mode: str = "stable") -> dict:
    """歷史上同一類錯誤是否反覆出現（只描述，不設計規則）。"""
    conc, other, in_cl, out_cl, hot, cool, ship, rest = [], [], [], [], [], [], [], []
    for P in periods:
        df = P["df"]
        if P.get("bench") is None or f"rk_{mode}" not in df:
            continue
        top = df[df["ret"].notna()].sort_values(f"rk_{mode}").head(10)
        if len(top) < 5:
            continue
        ex = top["ret"] - pb.COST - P["bench"]
        cnt = top["industry"].fillna("").value_counts()
        (conc if cnt.iloc[0] >= 3 else other).append(float(ex.mean()))
        big = set(cnt[cnt >= 3].index)
        msk = top["industry"].fillna("").isin(big)
        in_cl += list(ex[msk]); out_cl += list(ex[~msk])
        if "dist_ma20" in top and top["dist_ma20"].notna().sum() >= 6:
            q = top["dist_ma20"].quantile(2 / 3)
            hot += list(ex[top["dist_ma20"] >= q]); cool += list(ex[top["dist_ma20"] < q])
        s = top["industry"].fillna("") == "航運業"
        ship += list(ex[s]); rest += list(ex[~s])
    pct = lambda v: None if not len(v) else round(float(np.mean(v)) * 100, 2)
    return {
        "mode": mode, "periods": len(conc) + len(other),
        "concentrated_periods": {"n": len(conc), "share_pct": round(len(conc) / max(1, len(conc) + len(other)) * 100, 1),
                                 "mean_excess": pct(conc), "others_mean_excess": pct(other), "t": _r(_t(conc, other))},
        "cluster_stocks": {"n": len(in_cl), "mean_excess": pct(in_cl), "others": pct(out_cl), "t": _r(_t(in_cl, out_cl))},
        "far_above_ma20": {"n": len(hot), "mean_excess": pct(hot), "others": pct(cool), "t": _r(_t(hot, cool))},
        "shipping": {"n": len(ship), "mean_excess": pct(ship), "others": pct(rest), "t": _r(_t(ship, rest))},
        "note": "20 日、開盤進出、扣成本、相對 0050；t 是兩群平均差的 Welch t。只描述是否反覆出現，不據此設計排除規則。",
    }


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def run(dbname: str = "stocks.db", D: dict | None = None, periods: list[dict] | None = None) -> dict:
    D = D or pb.load(dbname)
    if periods is None:
        from . import academic_study as ac
        periods = pickle.loads(ac.CACHE.read_bytes()) if ac.CACHE.exists() else []
    con = sqlite3.connect(config.DATA_DIR / dbname) if (config.DATA_DIR / dbname).exists() else None
    batches = []
    for p in sorted(MAT.glob("????-??-??_h*.json")):
        rec = json.loads(p.read_text("utf-8"))
        kf = PICKS / ((rec.get("official") or {}).get("file") or f"{rec['signal_date']}.json")
        if kf.exists():
            batches.append(batch(D, rec, json.loads(kf.read_text("utf-8")), con))
    kinds = Counter()
    for bt in batches:
        for m in bt["modes"].values():
            kinds.update(m["kinds"])
    return {"schema_version": 1, "study_version": STUDY_VERSION, "generated_at": config.now().isoformat(timespec="seconds"),
            "rules": {"res_band_pp": RES_BAND, "hot_pre20": HOT_PRE20},
            "batches": batches, "kinds_all": dict(kinds),
            "recurrence": {m: recurrence(periods, m) for m in MODES} if periods else None,
            "limits": ["分類只是描述，不是已證實的原因", "5 日批次彼此高度重疊（每天一批、持有 5 天），不是獨立樣本",
                       "新聞只有 2026-10-11 以後的資料，沒有辦法回測新聞是否能事先避開"]}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), "utf-8")
    print(json.dumps({"kinds": res["kinds_all"], "recurrence": res["recurrence"]}, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
