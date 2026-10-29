"""V21.2 正式績效對帳：每一份到期對帳檔（history/audit/maturity/）都用「另一套獨立寫法」重算一次 → history/audit/reconcile.json。

為什麼要有：網站、每日報告、週報、研究報告都引用到期對帳檔的數字（例如 10/16 穩健型 5 日相對 0050 −3.64%）。
這裡不呼叫 maturity_report／pit_backtest 的報酬函式，直接用還原後開盤價重算，兩邊對不上就列出來，並讓測試失敗。

另外列出「績效口徑表」：正式推薦、影子模型、歷史回測、模擬交易、因子研究各自怎麼算，哪些不能直接比。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import config

ROOT = Path(__file__).resolve().parent.parent
MAT = ROOT / "history" / "audit" / "maturity"
PICKS = ROOT / "history" / "picks"
PRED = ROOT / "history" / "predictions"
OUT = ROOT / "history" / "audit" / "reconcile.json"
COST = 0.00585
BENCH = "0050"
TOL = 0.011          # 百分點：檔案存兩位小數，四捨五入誤差最多 0.005，兩次相減最多 0.01

DEFINITIONS = {
    "official": {"name": "正式推薦績效", "source": "history/picks/（首次發布版）→ history/audit/maturity/ 的 official",
                 "rule": "訊號日收盤後封存；隔一個交易日開盤買、第 h 個交易日後開盤賣（還原除權息）；扣來回成本 0.585%；前 10 名等權；買不到的不算",
                 "bench": "0050 同一段開盤到開盤（還原除權息）"},
    "shadow": {"name": "影子模型績效", "source": "history/predictions/ → maturity 的 shadow／academic",
               "rule": "和正式推薦相同的進出場與成本，名單是候選模型分數的前 10 名", "bench": "同上"},
    "scorecard": {"name": "影子預測成績單", "source": "history/predictions/scorecard.json",
                  "rule": "top_excess＝前 20 名、未扣成本；top10_net_excess＝前 10 名、扣成本（和到期對帳同口徑）",
                  "bench": "0050 開盤到開盤"},
    "backtest": {"name": "歷史回測", "source": "history/backtest/",
                 "rule": "用當時已公布的資料重建；20 日不重疊期間；不是前瞻績效", "bench": "0050 開盤到開盤"},
    "paper": {"name": "AI 模擬交易", "source": "history/paper/",
              "rule": "每本帳 100 萬、最多 5 檔；開盤＋0.1% 滑價；停損 −10%、停利 +20%、最長 20 日；實際手續費與證交稅",
              "bench": "0050 同期間；持有期間與名單規則都和正式推薦不同，不能直接比"},
    "factor": {"name": "因子研究", "source": "history/backtest/selection.json、academic.json、twolayer.json",
               "rule": "單因子十分位或 Rank IC；有的未扣成本（會註明）", "bench": "0050 或全體可排名股票"},
    "taiex": {"name": "加權指數", "rule": "只有收盤價：訊號日收盤 → 出場前一天收盤；要和 0050 比時用 bench_0050_close（同一段收盤）",
              "note": "和 0050 開盤到開盤的期間不同，不能直接相減"},
}


def _open(D: dict, sid: str, d: str) -> float | None:
    a = D["adj"].get(sid)
    if a is None or d not in a.index:
        return None
    v = float(a.loc[d, "open"])
    return v if v > 0 and not np.isnan(v) else None


def check_file(D: dict, r: dict, picks_dir: Path = PICKS) -> list[str]:
    """一份到期對帳檔的獨立重算。回傳不一致的地方（空清單＝全部一致）。"""
    bad = []
    dates = D["dates"]
    sig, h = r["signal_date"], int(r["horizon"])
    if sig not in dates:
        return [f"{sig} 不在交易日曆"]
    i = dates.index(sig)
    if i + 1 + h >= len(dates):
        return [f"{sig} h{h} 還沒到期，卻有對帳檔"]
    entry, exit_ = dates[i + 1], dates[i + 1 + h]
    if (r["entry"], r["exit"]) != (entry, exit_):
        bad.append(f"進出場日 {r['entry']}→{r['exit']}，應為 {entry}→{exit_}")
    b0, b1 = _open(D, BENCH, entry), _open(D, BENCH, exit_)
    b = None if b0 is None or b1 is None else b1 / b0 - 1
    if b is not None and r.get("bench_0050") is not None and abs(b * 100 - r["bench_0050"]) > TOL:
        bad.append(f"0050 報酬 {r['bench_0050']}，重算 {b * 100:.2f}")
    if abs(r.get("cost_pct", 0) - COST * 100) > 1e-9:
        bad.append(f"成本 {r.get('cost_pct')}，應為 {COST * 100}")

    def cohort(label, c):
        ex = []
        for s in c["stocks"]:
            if s.get("net") is None:
                continue
            p0, p1 = _open(D, s["id"], entry), _open(D, s["id"], exit_)
            if s.get("status") == "ok":
                if p0 is None or p1 is None:
                    bad.append(f"{label} {s['id']} 標正常出場，但缺開盤價")
                    continue
                ret = p1 / p0 - 1
                if abs(ret * 100 - s["ret"]) > TOL:
                    bad.append(f"{label} {s['id']} 報酬 {s['ret']}，重算 {ret * 100:.2f}")
            if abs(s["ret"] - COST * 100 - s["net"]) > TOL:
                bad.append(f"{label} {s['id']} 扣成本 {s['net']} ≠ {s['ret']} − {COST * 100}")
            if s.get("excess") is not None and b is not None:
                if abs(s["net"] - b * 100 - s["excess"]) > TOL:
                    bad.append(f"{label} {s['id']} 超額 {s['excess']}，重算 {s['net'] - b * 100:.2f}")
                ex.append(s["net"] - b * 100)
        if ex and c.get("mean_excess") is not None and abs(np.mean(ex) - c["mean_excess"]) > TOL:
            bad.append(f"{label} 平均超額 {c['mean_excess']}，重算 {np.mean(ex):.2f}")
        return round(float(np.mean(ex)), 2) if ex else None

    off = r.get("official") or {}
    kf = picks_dir / (off.get("file") or f"{sig}.json")
    if off and kf.exists():   # 名單要和首次發布版的前 10 名完全一樣、順序一樣
        x = json.loads(kf.read_text("utf-8"))
        lists = x.get("rec_picks") or x.get("picks") or {}
        for m, c in (off.get("modes") or {}).items():
            want = [y["stock_id"] for y in (lists.get(m) or [])[:10]]
            got = [s["id"] for s in c["stocks"]]
            if want != got:
                bad.append(f"正式 {m} 名單和 {kf.name} 的前 10 名不同")
    for m, c in (off.get("modes") or {}).items():
        cohort(f"正式 {m}", c)
    for m, c in ((r.get("shadow") or {}).get("models") or {}).items():
        cohort(f"影子 {m}", c)
    for m, c in ((r.get("academic") or {}).get("lists") or {}).items():
        cohort(f"學術 {m}", c)
    return bad


def cross_sources(mat_dir: Path = MAT, pred: Path = PRED) -> list[dict]:
    """同一批、同口徑的數字在不同檔案要一樣；口徑不同的列出差在哪裡（不算錯）。"""
    out = []
    sc = json.loads((pred / "scorecard.json").read_text("utf-8")) if (pred / "scorecard.json").exists() else {}
    for p in sorted(mat_dir.glob("????-??-??_h*.json")):
        r = json.loads(p.read_text("utf-8"))
        v = ((r.get("shadow") or {}).get("models") or {}).get("v21")
        if not v:
            continue
        h = str(r["horizon"])
        day = next((d for d in sc.get("daily", []) if d.get("sig") == r["signal_date"] and str(d.get("h")) == h), None)
        row = {"batch": p.stem, "maturity_v21_top10_net": v.get("mean_excess")}
        if day:
            t = (day.get("top") or {}).get("v21")
            row["scorecard_top20_gross"] = None if t is None else round(t * 100, 2)
            t = (day.get("top10") or {}).get("v21")
            row["scorecard_top10_net"] = None if t is None else round(t * 100, 2)
        row["note"] = "到期對帳＝前 10 名、扣成本；成績單 top_excess＝前 20 名、未扣成本（口徑不同，不能直接比）；top10_net_excess 才和到期對帳同口徑（成績單遇到買不到的會往下補第 11 名，到期對帳則不補；有買不到的股票時兩者可能有小差異）"
        if row.get("scorecard_top10_net") is not None and row["maturity_v21_top10_net"] is not None:
            row["same_basis_match"] = abs(row["scorecard_top10_net"] - row["maturity_v21_top10_net"]) <= TOL
        out.append(row)
    return out


def run(dbname: str = "stocks.db", D: dict | None = None, mat_dir: Path = MAT, picks_dir: Path = PICKS) -> dict:
    from . import pit_backtest as pb
    D = D or pb.load(dbname)
    files = []
    for p in sorted(mat_dir.glob("????-??-??_h*.json")):
        r = json.loads(p.read_text("utf-8"))
        bad = check_file(D, r, picks_dir)
        summ = {m: c.get("mean_excess") for m, c in ((r.get("official") or {}).get("modes") or {}).items()}
        summ.update({f"shadow_{m}": c.get("mean_excess") for m, c in ((r.get("shadow") or {}).get("models") or {}).items()})
        files.append({"file": p.name, "entry": r["entry"], "exit": r["exit"], "ok": not bad, "problems": bad, "mean_excess": summ})
    res = {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"),
           "method": "不經過 maturity_report／pit_backtest 的報酬函式，直接用還原後開盤價重算報酬、扣成本、0050 超額與名單（首次發布版前 10 名）",
           "tolerance_pp": TOL, "files": files, "all_ok": all(f["ok"] for f in files), "cross_sources": cross_sources(mat_dir),
           "definitions": DEFINITIONS}
    return res


def main(argv: list[str]) -> int:
    res = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({"all_ok": res["all_ok"], "files": [(f["file"], f["ok"], f["problems"][:3]) for f in res["files"]]}, ensure_ascii=False))
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
