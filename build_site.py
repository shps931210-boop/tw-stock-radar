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

snap_path = config.DATA_DIR / "snapshot.json"
if snap_path.exists():
    shutil.copy(snap_path, out / "data" / "snapshot.json")
    snap = json.loads(snap_path.read_text("utf-8"))
    ids = [s["stock_id"] for s in snap["stocks"] if s.get("is_common")]
    for sid in ids:
        (out / "data" / "stock" / f"{sid}.json").write_text(
            json.dumps(stock_detail(sid, "stocks.db"), ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"已產生 site/（{len(ids)} 檔個股資料）")
else:
    print("沒有真實資料快照，site/ 只會顯示資料取得失敗")
    if "--strict" in sys.argv:
        sys.exit(1)  # GitHub Actions：不發布，網站維持上一次成功的版本
