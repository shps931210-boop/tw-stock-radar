"""到期對帳明細（V20.4）：每一批封存的預測到期後，逐檔列出實際結果。

對象
1. 影子預測（history/predictions/日期.json）：V21、Ridge、LightGBM 各自的前 10 名，加上全部可排名股票的 Rank IC。
2. 正式 AI 精選（history/picks/日期.json，首次發布版）：穩健、均衡、積極各前 10 名。這一天還沒有正式精選時用原始排名，照實標示。

規則和對帳統計（ml_predict.scorecard、track）相同
- 訊號日收盤後封存，隔一個交易日開盤進場；進場後第 h 個交易日開盤出場（還原除權息）。
- 扣交易成本 0.585%（手續費 0.1425% × 2 ＋ 證交稅 0.3%）。基準 0050 同一段期間（開盤到開盤）；加權指數只有收盤價，用「訊號日收盤 → 出場前一天收盤」近似。
- 買不到（沒成交、成交金額太少、漲停鎖死）的照實列出，不算進平均；出場日停牌用最後收盤價並標示。
- 每一批只寫一次（history/audit/maturity/訊號日_h天數.json），之後不改；預測內容直接從封存檔讀，不重新計算。

用法：python -m app.maturity_report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, track

PRED = config.ROOT / "history" / "predictions"
PICKS = config.ROOT / "history" / "picks"
OUT = config.ROOT / "history" / "audit" / "maturity"
COST = track.COST
TOP = 10
HORIZONS = ("5", "20", "60")
SHADOW = {"v21": "V21（正式模型的分數）", "ridge": "Ridge（候選）", "lgbm": "LightGBM（候選）"}
MODES = {"stable": "穩健型", "balanced": "均衡型", "aggressive": "積極型"}
STATUS = {"ok": "正常出場", "stale": "出場日停牌，用最後收盤價", "not_traded": "進場日沒有成交，買不到", "thin": "進場日成交金額太少，買不到",
          "locked": "漲停鎖死，買不到"}


def _cohort(D: dict, ids: list[str], entry: str, exit_: str, bench: float | None, names: dict, extra: dict | None = None) -> dict:
    from . import pit_backtest as pb
    rows = []
    for sid in ids:
        r, st = pb.hold_return(D, sid, entry, exit_)
        net = None if r is None else r - COST
        rows.append({"id": sid, "name": names.get(sid), "status": st, "status_text": STATUS.get(st, st),
                     "ret": None if r is None else round(r * 100, 2), "net": None if net is None else round(net * 100, 2),
                     "excess": None if net is None or bench is None else round((net - bench) * 100, 2), **((extra or {}).get(sid) or {})})
    ok = [x for x in rows if x["net"] is not None]
    nets = [x["net"] for x in ok]
    ex = [x["excess"] for x in ok if x["excess"] is not None]
    return {"stocks": rows, "n": len(rows), "filled": len(ok), "unfilled": [x["id"] for x in rows if x["net"] is None],
            "stale": [x["id"] for x in rows if x["status"] == "stale"],
            "mean_net": round(float(np.mean(nets)), 2) if nets else None,
            "up_hit": round(float(np.mean([x > 0 for x in nets])) * 100, 1) if nets else None,
            "beat_0050": round(float(np.mean([x > 0 for x in ex])) * 100, 1) if ex else None,
            "mean_excess": round(float(np.mean(ex)), 2) if ex else None,
            "worst": sorted(ok, key=lambda x: x["net"])[:3]}


def _taiex(D: dict, sig: str, h: str) -> dict:
    """加權指數（只有收盤價）：訊號日收盤 → 出場前一天收盤；附 0050 同一段收盤報酬。
    2026-10-28 更正：之前只寫加權指數，卻和 0050 開盤到開盤並列，兩者期間不同不能直接比。"""
    from . import pit_backtest as pb
    dates = D["dates"]
    if sig not in dates or dates.index(sig) + int(h) >= len(dates):
        return {"taiex": None, "window": None, "bench_0050_close": None}
    last = dates[dates.index(sig) + int(h)]
    idx = D.get("idx_s")
    tx = round(float(idx[last] / idx[sig] - 1) * 100, 2) if idx is not None and sig in idx.index and last in idx.index else None
    c = pb.close_return(D, pb.BENCH, sig, last)
    return {"taiex": tx, "window": [sig, last], "bench_0050_close": None if c is None else round(c * 100, 2)}


def corrections(D: dict, out: Path = OUT) -> list[dict]:
    """已封存的到期檔不改；缺少 0050 收盤口徑的舊檔，補一筆更正紀錄到 corrections.json（只新增）。"""
    f = out / "corrections.json"
    C = json.loads(f.read_text("utf-8")) if f.exists() else {"note": "到期對帳檔寫過就不改；發現口徑問題時在這裡附加更正，原檔保留。", "items": []}
    done = {(x["file"], x["code"]) for x in C["items"]}
    added = []
    for p in sorted(out.glob("????-??-??_h*.json")):
        r = json.loads(p.read_text("utf-8"))
        if "bench_0050_close" in r or (p.name, "taiex_window") in done:
            continue
        t = _taiex(D, r["signal_date"], str(r["horizon"]))
        added.append({"file": p.name, "code": "taiex_window", "at": config.now().isoformat(timespec="seconds"),
                      "issue": "加權指數是「訊號日收盤→出場前一天收盤」，0050 是「進場日開盤→出場日開盤」，期間不同卻並列。",
                      "original": {"taiex": r.get("taiex"), "bench_0050": r.get("bench_0050")},
                      "added": {"taiex_window": t["window"], "bench_0050_close": t["bench_0050_close"]},
                      "effect": "選股的成本後報酬、相對 0050 超額都不受影響；只有「和加權指數比較」要改用同一段收盤口徑。"})
    if added:
        C["items"] += added
        f.write_text(json.dumps(C, ensure_ascii=False, indent=1), "utf-8")
    return C["items"]


def build_one(D: dict, sig: str, h: str, pred: Path = PRED, picks: Path = PICKS) -> dict | None:
    """一批（訊號日 sig、期間 h）到期了就回傳對帳明細；還沒到期回傳 None。"""
    from . import pit_backtest as pb
    dates = D["dates"]
    w = pb.window(dates, sig, h)
    if w is None:
        return None
    entry, exit_ = w
    b = pb._bench(D, entry, exit_)
    tx = _taiex(D, sig, h)
    names = {k: (v or {}).get("name") for k, v in (D.get("info") or {}).items()}
    out = {"schema_version": 1, "signal_date": sig, "horizon": int(h), "entry": entry, "exit": exit_,
           "bench_0050": None if b is None else round(b * 100, 2), "taiex": tx["taiex"], "taiex_window": tx["window"],
           "bench_0050_close": tx["bench_0050_close"], "cost_pct": round(COST * 100, 3),
           "rule": "訊號日收盤後封存，隔一個交易日開盤買，第 h 個交易日後開盤賣（還原除權息），扣成本 0.585%。0050 同一段期間（開盤到開盤）；"
                   "加權指數只有收盤價，用訊號日收盤到出場前一天收盤，並附上 0050 同一段收盤口徑，兩者才能互相比較。買不到的不算進平均。",
           "created_at": config.now().isoformat(timespec="seconds")}
    # 1. 影子預測
    pf = pred / f"{sig}.json"
    if pf.exists():
        rec = json.loads(pf.read_text("utf-8"))
        out["shadow"] = {"file": pf.name, "generated_at": rec.get("generated_at"), "model_hash": rec.get("model_hash"),
                         "model_version": rec.get("model_version"), "on_time": track.on_time(rec, entry), "n": rec.get("n"), "models": {}}
        y = {}
        for s in rec["stocks"]:
            r, _ = pb.hold_return(D, s["id"], entry, exit_)
            if r is not None and b is not None:
                y[s["id"]] = r - b
        Y = pd.Series(y, dtype=float)
        for m, label in SHADOW.items():
            p = pd.Series({s["id"]: (s.get(h) or {}).get(m) for s in rec["stocks"]}, dtype=float).dropna()
            if len(p) < 30:
                continue
            top = list(p.nlargest(TOP).index)
            ext = {sid: {"pred": round(float(p[sid]), 4), "rank": k + 1} for k, sid in enumerate(top)}
            q = p.reindex(Y.index).dropna()
            ic = float(q.rank().corr(Y.reindex(q.index).rank())) if len(q) >= 30 else None
            out["shadow"]["models"][m] = {"name": label, "ic": None if ic is None else round(ic, 3), "ic_n": len(q),
                                          **_cohort(D, top, entry, exit_, b, names, ext)}
    # 2. 正式 AI 精選（首次發布版）
    kf = picks / f"{sig}.json"
    if kf.exists():
        x = json.loads(kf.read_text("utf-8"))
        official = bool(x.get("rec_picks"))
        lists = x.get("rec_picks") or x.get("picks") or {}
        out["official"] = {"file": kf.name, "generated_at": x.get("generated_at"), "model_version": x.get("model_version"),
                           "model_hash": x.get("model_hash"), "on_time": track.on_time(x, entry),
                           "list_kind": "正式精選" if official else "原始排名（當時還沒有正式精選）", "modes": {}}
        for m, label in MODES.items():
            rows = (lists.get(m) or [])[:TOP]
            if not rows:
                continue
            ext = {c["stock_id"]: {"rank": c.get("rank"), "score": c.get("score")} for c in rows}
            out["official"]["modes"][m] = {"name": label, **_cohort(D, [c["stock_id"] for c in rows], entry, exit_, b, names, ext)}
    return out


def run(dbname: str = "stocks.db", pred: Path = PRED, picks: Path = PICKS, out: Path = OUT, D: dict | None = None) -> dict:
    """找出所有已到期、還沒寫過的批次，各寫一次；最後更新索引。"""
    from . import pit_backtest as pb
    D = D or pb.load(dbname)
    out.mkdir(parents=True, exist_ok=True)
    sigs = sorted({f.stem for f in pred.glob("????-??-??.json")} | {f.stem for f in picks.glob("????-??-??.json")})
    new = []
    for sig in sigs:
        for h in HORIZONS:
            f = out / f"{sig}_h{h}.json"
            if f.exists():
                continue   # 寫過就不改
            r = build_one(D, sig, h, pred, picks)
            if r:
                with open(f, "x", encoding="utf-8") as fh:
                    json.dump(r, fh, ensure_ascii=False, indent=1)
                new.append(f.name)
    fixes = {x["file"]: x["added"] for x in corrections(D, out) if x["code"] == "taiex_window"}
    idx = []
    for f in sorted(out.glob("????-??-??_h*.json")):
        r = json.loads(f.read_text("utf-8"))
        r = {**r, **fixes.get(f.name, {})}
        sh = (r.get("shadow") or {}).get("models") or {}
        of = (r.get("official") or {}).get("modes") or {}
        idx.append({"file": f.name, "signal_date": r["signal_date"], "horizon": r["horizon"], "exit": r["exit"],
                    "bench_0050": r["bench_0050"], "taiex": r["taiex"], "taiex_window": r.get("taiex_window"),
                    "bench_0050_close": r.get("bench_0050_close"), "corrected": f.name in fixes,
                    "shadow": {m: {k: v.get(k) for k in ("ic", "mean_net", "mean_excess", "up_hit", "beat_0050", "filled")} for m, v in sh.items()},
                    "shadow_on_time": (r.get("shadow") or {}).get("on_time"), "shadow_model": (r.get("shadow") or {}).get("model_hash"),
                    "official": {m: {k: v.get(k) for k in ("mean_net", "mean_excess", "up_hit", "beat_0050", "filled")} for m, v in of.items()},
                    "official_kind": (r.get("official") or {}).get("list_kind"), "official_on_time": (r.get("official") or {}).get("on_time")})
    I = {"generated_at": config.now().isoformat(timespec="seconds"), "new": new, "batches": idx[::-1],
         "note": "每一批到期後只寫一次，不會事後修改。同一個期間的相鄰批次持有期重疊，不是獨立樣本；一兩批的短期結果不能代表模型長期有沒有優勢。"}
    (out / "index.json").write_text(json.dumps(I, ensure_ascii=False, indent=1), "utf-8")
    return I


if __name__ == "__main__":
    I = run()
    print(json.dumps({"new": I["new"], "batches": len(I["batches"])}, ensure_ascii=False))
    sys.exit(0)
