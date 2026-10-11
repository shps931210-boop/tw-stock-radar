"""資料庫備份檢查（GitHub Actions 用）。

python -m app.backup verify 下載回來的.gz   還原測試：解壓縮、完整性檢查、比對筆數和最新日期
python -m app.backup status 備份結果 還原結果   寫 data/backup_status.json（網站可信度頁、app.check 會讀）
"""
from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from . import config, db

VERIFY = config.DATA_DIR / "backup_verify.json"
STATUS = config.DATA_DIR / "backup_status.json"


def _facts(path: Path) -> dict:
    con = sqlite3.connect(path)
    try:
        return {"integrity": con.execute("PRAGMA integrity_check").fetchone()[0],
                "rows": con.execute("SELECT COUNT(*) FROM prices").fetchone()[0],
                "max_date": con.execute("SELECT MAX(date) FROM prices").fetchone()[0]}
    finally:
        con.close()


def verify(gz: str) -> int:
    """把剛上傳的備份下載回來還原，確認真的能用：完整性 ok、股價筆數與最新日期和目前資料庫一致。"""
    VERIFY.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as d:
        restored = Path(d) / "restored.db"
        with gzip.open(gz, "rb") as f, open(restored, "wb") as g:
            shutil.copyfileobj(f, g)
        got = _facts(restored)
    local = _facts(db.path("stocks.db"))
    problems = []
    if got["integrity"] != "ok":
        problems.append(f"完整性檢查：{got['integrity']}")
    if got["rows"] != local["rows"] or got["max_date"] != local["max_date"]:
        problems.append(f"還原後股價 {got['rows']} 筆、到 {got['max_date']}；目前資料庫 {local['rows']} 筆、到 {local['max_date']}")
    out = {**got, "local_rows": local["rows"], "size_mb": round(Path(gz).stat().st_size / 1e6, 1), "problems": problems}
    VERIFY.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print("✅ 還原測試通過" if not problems else "❌ " + "；".join(problems), out)
    return 1 if problems else 0


def status(backup: str, restore: str) -> int:
    """記錄這次備份與還原測試的結果；最後一次成功的時間存在資料庫裡，快取被清掉也還在。"""
    v = json.loads(VERIFY.read_text("utf-8")) if VERIFY.exists() else {}
    ok = backup == "success" and restore == "success" and not v.get("problems")
    now = datetime.now().isoformat(timespec="minutes")  # Actions 已設定 TZ=Asia/Taipei
    with db.connect() as con:
        if ok:
            db.set_meta(con, "backup_last_ok", now)
            con.commit()
        last_ok = db.get_meta(con, "backup_last_ok")
    out = {"checked_at": now, "ok": ok, "backup": backup, "restore": restore, "last_ok": last_ok,
           "rows": v.get("rows"), "max_date": v.get("max_date"), "size_mb": v.get("size_mb"), "problems": v.get("problems") or []}
    if backup != "success":
        out["problems"].insert(0, "上傳備份失敗")
    elif restore != "success" and not v.get("problems"):
        out["problems"].insert(0, "下載備份做還原測試失敗")
    STATUS.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    sys.exit(verify(*args) if cmd == "verify" else status(*args))
