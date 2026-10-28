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


V_PARTS = ("score_quality", "score_growth", "score_value", "msurge")   # 和 twolayer_study 同一個定義
E_PARTS = ("score_trend", "st_reversal")
SITE = config.DATA_DIR / "twolayer.json"


def layers(rows: list[dict], rev: dict[str, pd.DataFrame], sig: str) -> pd.DataFrame:
    """投資價值 V、進場機會 E（均衡型可排名母體內的百分位等權平均，0–100）。"""
    df = pd.DataFrame([{"id": r["stock_id"], **{k: r.get(k) for k in V_PARTS[:3] + E_PARTS[:1]},
                        "st_reversal": None if r.get("ret_20d") is None else -float(r["ret_20d"])}
                       for r in rows if (r.get("ai") or {}).get("modes", {}).get("balanced", {}).get("rank")]).set_index("id")
    df["msurge"] = [ac._rev_feats(rev.get(s), sig).get("msurge") for s in df.index]
    pct = lambda cols: pd.concat([df[c].astype(float).rank(pct=True) for c in cols], axis=1).mean(axis=1, skipna=True) * 100
    df["V"], df["E"] = pct(V_PARTS), pct(E_PARTS)
    return df


def site(rows: list[dict], rev: dict[str, pd.DataFrame], sig: str, path: Path = SITE) -> dict:
    """網站用：每檔的兩層分數＋研究證據摘要。每次更新都重算（不是封存）。"""
    L = layers(rows, rev, sig)
    st = {}
    f = config.ROOT / "history" / "backtest" / "twolayer.json"
    if f.exists():
        T = json.loads(f.read_text("utf-8"))
        st = {"ic_v": T["ic"]["V"]["20"], "ic_e": T["ic"]["E"]["20"], "timing20": T["timing"]["20"], "periods": T.get("periods"),
              "study_version": T.get("study_version")}
    out = {"date": sig, "generated_at": config.now().isoformat(timespec="seconds"), "evidence": st,
           "status": "研究中：歷史資料驗證排序有效，前瞻驗證累積中；不是買賣訊號，也不影響正式排行榜",
           "v_parts": ["品質", "成長", "價值", "月營收驚喜"], "e_parts": ["趨勢", "短期回檔"],
           "stocks": {s: [None if pd.isna(v) else round(float(v)), None if pd.isna(e) else round(float(e))] for s, v, e in zip(L.index, L["V"], L["E"])}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    return out


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
            "overlap": int(len(set(base) & set(cand))),
            "layers": {k: [None if pd.isna(v) else round(float(v), 1), None if pd.isna(e) else round(float(e), 1)]
                       for k, v, e in (lambda L: zip(L.index, L["V"], L["E"]))(layers(rows, rev, sig))}}


def seal(dbname: str = "stocks.db", out: Path = OUT, snap_path: Path | None = None) -> dict | None:
    from .ml_predict import completeness
    with db.connect(dbname) as con:
        last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        comp = completeness(con, last) if last else {"ok": False, "why": "資料庫沒有股價"}
        if not comp["ok"]:
            print(f"{last} 行情不完整，學術候選暫不封存：{comp['why']}")
            return None
        R = db.read(con, "SELECT * FROM revenue ORDER BY stock_id, ym")
    snap = json.loads((snap_path or config.DATA_DIR / "snapshot.json").read_text("utf-8"))
    rows = snap.get("stocks") or []
    if not rows or max((r.get("date") or "") for r in rows) < last:
        print("快照日期和行情不一致，學術候選暫不封存")
        return None
    rev = {s: g for s, g in R.groupby("stock_id")}
    try:
        site(rows, rev, last)
    except Exception as e:  # 網站說明失敗不影響封存
        print("兩層分數網站資料失敗：", e)
    if (out / f"{last}.json").exists():
        print(f"{last} 學術候選已封存，不覆寫")
        return None
    rec = build(rows, rev, last)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{last}.json", "x", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print(f"已封存 {last} 學術候選：重疊 {rec['overlap']}/10，MSURGE 覆蓋 {rec['msurge_coverage']}%")
    return rec


if __name__ == "__main__":
    seal()
    sys.exit(0)
