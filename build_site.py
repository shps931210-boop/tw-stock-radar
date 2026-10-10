"""把真實資料輸出成不需要後端的靜態網站（site/），給 GitHub Pages 等靜態主機使用。

    python -m app.update      # 先更新資料
    python build_site.py      # 產生 site/index.html、site/data/*.json

沒有成功取得真實行情時，只輸出失敗原因（site/data/status.json），不會用示範資料頂替。
"""
import json
import shutil
import sys
from pathlib import Path

from app import config, db
from app.server import stock_detail

def quality(snap: dict) -> dict:
    """資料品質摘要（data/quality.json）：完整度、排名數、缺資料原因，方便遠端檢查。"""
    from collections import Counter
    st = snap["stocks"]
    base = [r for r in st if r.get("base_ok")]
    pick = ("close", "date", "adj_events", "data_completeness", "missing_detail", "stale_notes", "fscore", "fscore_n",
            "inst_date", "inst_lag", "margin_lag", "fin_period", "rev_ym", "val_date", "composite", "verdict")
    return {
        "generated_at": snap.get("generated_at"), "rankings_blocked": snap.get("rankings_blocked"),
        "adjusted_markets": snap.get("adjusted_markets"), "freshness_rule": snap.get("freshness_rule"),
        "stocks": len(st), "base_ok": len(base), "complete": sum(bool(r.get("complete")) for r in base),
        "ranked": {m: sum(1 for r in st if r["ai"]["modes"][m]["rank"]) for m in snap.get("ai_modes", {})},
        "signals": Counter(r.get("timing", {}).get("status") for r in st),
        "adjusted_stocks": sum(1 for r in st if (r.get("adj_events") or 0) > 0),
        "chips_ok": sum(r.get("inst_ratio_5d") is not None for r in base),
        "missing": Counter(m.split("（")[0] for r in base for m in r.get("missing_detail") or []).most_common(),
        "stale": Counter(n.split(" ")[0] for r in base for n in r.get("stale_notes") or []).most_common(),
        "samples": {r["stock_id"]: {k: r.get(k) for k in pick} | {"ai": r["ai"]["modes"].get("balanced"), "summary": r["ai"].get("summary")}
                    for r in st if r["stock_id"] in ("2330", "2303", "2317", "2454", "6488")},
    }


root = Path(__file__).parent
out = root / "site"
shutil.rmtree(out, ignore_errors=True)
(out / "data" / "stock").mkdir(parents=True)

html = (root / "static" / "index.html").read_text("utf-8")
html = html.replace("<script>\nconst STATIC", "<script>window.STATIC_SITE = true;</script>\n<script>\nconst STATIC", 1)
(out / "index.html").write_text(html, "utf-8")

errors, last = [], None
if (config.DATA_DIR / "stocks.db").exists():
    with db.connect() as con:
        errors = (db.get_meta(con, "last_errors") or "").splitlines()
        last = db.get_meta(con, "last_update")
(out / "data" / "status.json").write_text(json.dumps(
    {"no_data": True, "last_attempt": last, "errors": errors[:15],
     "detail": "還沒有成功取得真實行情，所以沒有選股結果。下一次排程更新成功後會自動出現。"}, ensure_ascii=False), "utf-8")

if (config.DATA_DIR / "probe.json").exists():  # 連線診斷結果（python -m app.probe）
    shutil.copy(config.DATA_DIR / "probe.json", out / "data" / "probe.json")

snap_path = config.DATA_DIR / "snapshot.json"
if snap_path.exists():
    shutil.copy(snap_path, out / "data" / "snapshot.json")
    snap = json.loads(snap_path.read_text("utf-8"))
    ids = [s["stock_id"] for s in snap["stocks"] if s.get("is_common")]
    for sid in ids:
        (out / "data" / "stock" / f"{sid}.json").write_text(
            json.dumps(stock_detail(sid, "stocks.db"), ensure_ascii=False, separators=(",", ":")), "utf-8")
    (out / "data" / "quality.json").write_text(json.dumps(quality(snap), ensure_ascii=False, indent=1), "utf-8")
    print(f"已產生 site/（{len(ids)} 檔個股資料）")
else:
    print("沒有真實資料快照，site/ 只會顯示資料取得失敗")
    if "--strict" in sys.argv:
        sys.exit(1)  # GitHub Actions：不發布，網站維持上一次成功的版本
