"""產生可單獨打開的 demo.html（內嵌示範資料，不需要後端）。先執行 python -m app.update --demo。"""
import json
from pathlib import Path

from app.server import stock_detail

root = Path(__file__).parent
snap = json.loads((root / "data" / "snapshot_demo.json").read_text("utf-8"))
details = {s["stock_id"]: stock_detail(s["stock_id"], "demo.db") for s in snap["stocks"]}
data = json.dumps({"snapshot": snap, "details": details}, ensure_ascii=False, separators=(",", ":"))
html = (root / "static" / "index.html").read_text("utf-8")
html = html.replace("<script>\nconst STATIC", f"<script>window.STATIC_DATA = {data};</script>\n<script>\nconst STATIC", 1)
(root / "demo.html").write_text(html, "utf-8")
print(f"已產生 demo.html（{len(html) // 1024} KB）")
