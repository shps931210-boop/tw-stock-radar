"""資料與回測稽核（V28）：AI 有沒有偷看到未來？ → history/backtest/audit.json。

每週六和 AI 歷史回測一起跑，每一項都是「實際檢查」，不是說明文字：
1. 未來資料不變性：把訊號日之後的股價、除權息、指數、月營收、季報、股利全部亂改，重算訊號日的特徵與分數，必須一模一樣。
2. 公告時間規則：月營收、季報（一般／金融業）最早可用日，和法規期限比對。
3. Purge：walk-forward 的每個測試日，訓練資料的標籤都在測試訊號日之前到期。
4. 選樣：特徵百分位用訊號日當天完整母體（標籤篩選前）。
5. 股票母體：資料庫裡有沒有期間內下市、停止交易的股票（沒有就有存活者偏誤）。
6. 產業分類、普通股判斷是不是當時的資料。
7. 封存預測：每一份都在進場日開盤前產生、沒有事後補算。
找到問題就如實列在 defects，網站照樣公布。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import pit_backtest as pb

OUT = Path(__file__).resolve().parent.parent / "history" / "backtest" / "audit.json"
PRED_DIR = Path(__file__).resolve().parent.parent / "history" / "predictions"
SAMPLES = 3          # 抽幾個訊號日做未來資料不變性檢查
SKIP = {"stock_id", "name", "industry", "market", "adj_events", "exclude_reasons", "ai", "strategies"}


def _mangle(df: pd.DataFrame, mask: pd.Series, rng) -> pd.DataFrame:
    """把 mask 為真的列（未來資料）的數字欄位亂乘一個倍數。"""
    if not len(df) or not mask.any():
        return df
    df = df.copy()
    for c in df.columns:
        if c not in ("stock_id", "date", "ym", "period", "period_end", "year") and pd.api.types.is_numeric_dtype(df[c]):
            df.loc[mask, c] = df.loc[mask, c] * rng.uniform(0.3, 3.0, int(mask.sum()))
    return df


def future_copy(D: dict, sig: str, seed: int = 7) -> dict:
    """D 的副本：訊號日之後才知道的資料全部被改掉（之前的資料原封不動）。"""
    rng = np.random.default_rng(seed)
    y = int(sig[:4])
    info = D["info"]
    fin_mask = lambda f, sid: f["period"].map(lambda p: pb.fin_available(p, "金融" in (info.get(sid, {}).get("industry") or "")) > sig)
    E = dict(D)
    E["px"] = {s: _mangle(p, p["date"] > sig, rng) for s, p in D["px"].items()}
    E["ex"] = {s: _mangle(p, p["date"] > sig, rng) for s, p in D["ex"].items()}
    E["rev"] = {s: _mangle(p, p["ym"].map(pb.rev_available) > sig, rng) for s, p in D["rev"].items()}
    E["fin"] = {s: _mangle(p, fin_mask(p, s), rng) for s, p in D["fin"].items()}
    E["div"] = {s: _mangle(p, p["year"] >= y, rng) for s, p in D["div"].items()}
    if D["idx_s"] is not None:
        s = D["idx_s"].copy().astype(float)
        m = s.index > sig
        s[m] = s[m] * rng.uniform(0.3, 3.0, int(m.sum()))
        E["idx_s"] = s
    return E


def _flat(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        for k, v in r.items():
            if k in SKIP or isinstance(v, (list, dict)):
                continue
            out[(r["stock_id"], k)] = v
        for m, x in (r.get("ai") or {}).get("modes", {}).items():
            out[(r["stock_id"], f"ai.{m}.score")] = x.get("score")
            out[(r["stock_id"], f"ai.{m}.rank")] = x.get("rank")
    return out


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return a == b


def invariance(D: dict, sig: str, cfg: dict) -> dict:
    a = _flat(pb.scored_rows(D, sig, cfg))
    b = _flat(pb.scored_rows(future_copy(D, sig), sig, cfg))
    keys = set(a) | set(b)
    bad = sorted(k for k in keys if not _same(a.get(k), b.get(k)))
    return {"signal": sig, "values": len(keys), "changed": len(bad), "examples": [f"{s}.{k}" for s, k in bad[:8]], "ok": not bad}


def availability() -> list[dict]:
    """公告時間規則（法定期限的隔天才能用）。"""
    cases = [("月營收 2025-07", pb.rev_available("2025-07"), "2025-08-11", "每月 10 日前公告"),
             ("一般公司 Q1", pb.fin_available("2025-Q1"), "2025-05-16", "5/15 前公告"),
             ("一般公司 Q2", pb.fin_available("2025-Q2"), "2025-08-15", "8/14 前公告"),
             ("一般公司 Q3", pb.fin_available("2025-Q3"), "2025-11-15", "11/14 前公告"),
             ("一般公司年報", pb.fin_available("2024-Q4"), "2025-04-01", "3/31 前公告"),
             ("金融業 Q2", pb.fin_available("2025-Q2", True), "2025-09-01", "8/31 前公告")]
    return [{"item": n, "used_from": got, "rule_from": want, "rule": r, "ok": got >= want} for n, got, want, r in cases]


def purge_check(sigs: list[str], dates: list[str], h: int, min_train: int) -> dict:
    """重做 walk-forward 的訓練日挑選，確認每個訓練標籤都在測試訊號日之前到期。"""
    pos = {d: i for i, d in enumerate(dates)}
    ex = {s: pos[s] + 1 + h for s in sigs}
    bad, tests = [], 0
    for t in sigs:
        tr = [s for s in sigs if ex[s] < pos[t]]
        if len(tr) < min_train:
            continue
        tests += 1
        if any(ex[s] >= pos[t] for s in tr) or t in tr:
            bad.append(t)
    return {"horizon": h, "tests": tests, "violations": bad, "ok": not bad}


def universe(D: dict) -> dict:
    end = D["dates"][-1] if D["dates"] else None
    from . import metrics
    last = {s: p["date"].iloc[-1] for s, p in D["px"].items() if len(p) and metrics.is_common_id(s)}
    gone = [s for s, d in last.items() if end and d < (pd.Timestamp(end) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")]
    return {"stocks_with_prices": len(last), "stopped_trading_in_period": len(gone), "examples": sorted(gone)[:10],
            "ok": len(gone) > 0,
            "note": f"資料庫有 {len(gone)} 檔期間內停止交易的普通股，回測會把它們算進當時的母體；但無法確認是否包含所有下市股票（下市股票的歷史若沒補到，仍有部分存活者偏誤）" if gone else "資料庫只有目前仍在交易的股票：回測有存活者偏誤（下市股票被排除，報酬可能被高估）"}


def sealed(dates: list[str] | None = None) -> dict:
    """進場日優先用實際交易日曆（資料庫裡訊號日之後的第一個交易日）；還沒有資料時才用休市日曆推估。
    2026-10-29 更正：10/25 的稽核把 10/22 的進場日推成 10/23，但 10/23 颱風臨時休市（休市日曆沒有），實際是 10/27。"""
    files = sorted(PRED_DIR.glob("????-??-??.json")) if PRED_DIR.exists() else []
    from . import premarket, track
    H = premarket.holidays()
    rows = []
    for f in files:
        r = json.loads(f.read_text("utf-8"))
        nxt = [d for d in (dates or []) if d > r["signal_date"]]
        entry = nxt[0] if nxt else str(premarket.next_trading_day(date.fromisoformat(r["signal_date"]), H))
        ok = track.on_time({"generated_at": r.get("generated_at")}, entry)
        rows.append({"signal": r["signal_date"], "generated_at": r.get("generated_at"), "entry": entry, "on_time": ok,
                     "trained_before_signal": (r.get("time_check") or {}).get("trained_before_signal")})
    late = [x for x in rows if x["on_time"] is False or x["trained_before_signal"] is False]
    return {"files": len(rows), "late_or_leaky": late, "ok": not late, "rows": rows[-10:]}


def run(dbname: str = "stocks.db", log=print) -> dict:
    from . import ml_research as ml
    t0 = time.time()
    cfg = config.load()
    D = pb.load(dbname)
    start = pb.auto_start(D)
    sigs = ml.signal_dates(D["dates"], start)
    pick = [sigs[int(i)] for i in np.linspace(0, len(sigs) - 1, min(SAMPLES, len(sigs)))] if sigs else []
    inv = []
    for s in pick:
        inv.append(invariance(D, s, cfg))
        log(f"未來資料不變性 {s}：{inv[-1]['values']} 個數值，改變 {inv[-1]['changed']} 個（{time.time() - t0:.0f} 秒）")
    checks = [
        {"id": "future_invariance", "name": "改掉訊號日之後的所有資料，特徵與分數不變", "ok": all(x["ok"] for x in inv) and bool(inv), "detail": inv},
        {"id": "availability", "name": "月營收、季報只在法定公告期限後使用", "ok": all(x["ok"] for x in availability()), "detail": availability()},
        {"id": "purge", "name": "walk-forward 訓練標籤都在測試日之前到期", "ok": None, "detail": [purge_check(sigs, D["dates"], h, ml.MIN_TRAIN) for h in ml.HORIZONS]},
        {"id": "selection", "name": "特徵百分位用標籤篩選前的當日母體（METHOD ≥ 2）", "ok": ml.METHOD >= 2, "detail": {"method": ml.METHOD}},
        {"id": "universe", "name": "母體包含期間內下市或停止交易的股票", "ok": None, "detail": universe(D)},
        {"id": "sealed", "name": "每日封存預測都在進場日開盤前產生、訓練資料早於訊號日", "ok": None, "detail": sealed(D.get("dates"))},
    ]
    checks[2]["ok"] = all(x["ok"] for x in checks[2]["detail"])
    checks[4]["ok"] = checks[4]["detail"]["ok"]
    checks[5]["ok"] = checks[5]["detail"]["ok"]
    defects = [
        {"item": "產業分類用目前的資料", "impact": "公司若在期間內換產業，歷史上會被放在現在的產業；影響產業內比較與產業特徵，對排序影響推估很小", "status": "未解決"},
        {"item": "普通股、ETF 判斷用目前的代號規則", "impact": "影響很小", "status": "未解決"},
        {"item": "金融業 Q1、Q3 公告期限用一般公司的規則", "impact": "若金融業實際期限較晚，可能提早使用最多約半個月；金融業佔母體約 2%", "status": "待確認法規"},
        {"item": "回測的處置股票、全額交割狀態沒有歷史資料（一律當成正常）", "impact": "可能把當時不易交易的股票算進名單", "status": "未解決"},
    ]
    defects.insert(0, {"item": "股票母體只含目前仍在交易的股票" if not checks[4]["ok"] else "下市股票可能沒有全部收錄",
                       "impact": "存活者偏誤：回測報酬可能被高估（台股每年約有數十檔下市）", "status": "未解決" if not checks[4]["ok"] else "部分解決"})
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "data_end": D["dates"][-1] if D["dates"] else None,
            "checks": checks, "passed": sum(bool(c["ok"]) for c in checks), "total": len(checks), "defects": defects,
            "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    for c in res["checks"]:
        print(("通過" if c["ok"] else "未通過"), c["name"])
    return 0 if all(c["ok"] for c in res["checks"] if c["id"] in ("future_invariance", "availability", "purge", "selection")) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
