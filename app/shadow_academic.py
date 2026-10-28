"""學術候選的前瞻影子紀錄（V21.1）：均衡型分數＋月營收驚喜（MSURGE）。

academic_study（2026-10-28 事前登記）的結果：MSURGE 在因子層級達到 E2，但「均衡型＋MSURGE」開發段 t 2.17，
沒過模型層級門檻 2.9。所以只做前瞻影子紀錄，累積沒人看過的新資料（E3 需要的證據），不影響正式排行榜。

每天行情齊全後封存一次 history/predictions/academic/{日期}.json（寫過就不改）：
同一天的均衡型原始前 10 名，和「0.75×均衡型分數百分位＋0.25×MSURGE 百分位」前 10 名，規則和研究完全相同。
到期後由 maturity_report 用同一套開盤進出、扣成本規則對帳。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import config, db
from . import academic_study as ac

OUT = config.ROOT / "history" / "predictions" / "academic"
CANDIDATE = "balanced_msurge"
LABEL = "均衡型＋月營收驚喜（學術候選，影子）"
TOP = 10


def build(rows: list[dict], rev: dict[str, pd.DataFrame], sig: str) -> dict:
    df = pd.DataFrame([{"id": r["stock_id"], "name": r.get("name"), "industry": r.get("industry"),
                        "score": r["ai"]["modes"]["balanced"]["score"], "rank": r["ai"]["modes"]["balanced"]["rank"]}
                       for r in rows if (r.get("ai") or {}).get("modes", {}).get("balanced", {}).get("rank")]).set_index("id")
    df["msurge"] = [ac._rev_feats(rev.get(s), sig).get("msurge") for s in df.index]
    blend = (1 - ac.BLEND) * df["score"].rank(pct=True) + ac.BLEND * df["msurge"].rank(pct=True).fillna(0.5)
    base = df.sort_values("rank").index[:TOP]
    cand = blend.sort_values(ascending=False).index[:TOP]
    row = lambda s, k: {"stock_id": s, "name": df.at[s, "name"], "rank": k + 1, "balanced_score": df.at[s, "score"],
                        "msurge": None if pd.isna(df.at[s, "msurge"]) else round(float(df.at[s, "msurge"]), 3)}
    return {"schema_version": 1, "signal_date": sig, "generated_at": config.now().isoformat(timespec="seconds"),
            "status": "shadow", "candidate": CANDIDATE, "label": LABEL, "study": "history/backtest/academic.json（事前登記 2026-10-28）",
            "rule": f"{1 - ac.BLEND:.2f}×均衡型分數百分位＋{ac.BLEND:.2f}×MSURGE 百分位；沒有 MSURGE 的股票給中位數",
            "universe_n": int(len(df)), "msurge_coverage": round(float(df["msurge"].notna().mean()) * 100, 1),
            "lists": {"balanced": [row(s, k) for k, s in enumerate(base)], CANDIDATE: [row(s, k) for k, s in enumerate(cand)]},
            "overlap": int(len(set(base) & set(cand)))}


def seal(dbname: str = "stocks.db", out: Path = OUT, snap_path: Path | None = None) -> dict | None:
    from .ml_predict import completeness
    with db.connect(dbname) as con:
        last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        comp = completeness(con, last) if last else {"ok": False, "why": "資料庫沒有股價"}
        if not comp["ok"]:
            print(f"{last} 行情不完整，學術候選暫不封存：{comp['why']}")
            return None
        if (out / f"{last}.json").exists():
            print(f"{last} 學術候選已封存，不覆寫")
            return None
        R = db.read(con, "SELECT * FROM revenue ORDER BY stock_id, ym")
    snap = json.loads((snap_path or config.DATA_DIR / "snapshot.json").read_text("utf-8"))
    rows = snap.get("stocks") or []
    if not rows or max((r.get("date") or "") for r in rows) < last:
        print("快照日期和行情不一致，學術候選暫不封存")
        return None
    rev = {s: g for s, g in R.groupby("stock_id")}
    rec = build(rows, rev, last)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{last}.json", "x", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print(f"已封存 {last} 學術候選：重疊 {rec['overlap']}/10，MSURGE 覆蓋 {rec['msurge_coverage']}%")
    return rec


if __name__ == "__main__":
    seal()
    sys.exit(0)
