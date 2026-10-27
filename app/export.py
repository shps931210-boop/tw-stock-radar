"""程式碼與成果打包（V20.2）：產生 tw-stock-screener.zip，裡面附 MANIFEST.json。

以前只用 zip 檔名分辨版本，兩次打包內容一樣時看不出來；history/audit（覆蓋率、封存日曆、網站驗收）與 history/fin（財報同步）也沒有放進去。
MANIFEST.json 記錄：程式 commit、打包時間、最新資料日、對帳統計日期、最近一次網站驗收、每個檔案的 SHA-256，
拿兩份 zip 的 MANIFEST 一比就知道差在哪裡（程式沒變、只有每日紀錄變了，也看得出來）。

用法：python -m app.export <輸出 zip 路徑>
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone, timedelta

from . import config

HIST = ("history/backtest/", "history/models/", "history/predictions/", "history/premarket/", "history/research/", "history/learning/",
        "history/audit/", "history/fin/", "history/paper/")
SKIP = ("data/.gitkeep", "package-lock.json", "history/backtest/joint_trades.json")
CODE = ("app/", "static/", "tests/", "worker/", "supabase/", ".github/", "build_site.py", "build_demo.py", "requirements.txt", "config", "docs/", "README.md")


def files() -> list[str]:
    fs = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=config.ROOT).stdout.split()
    return [f for f in fs if (not f.startswith("history/") or f.startswith(HIST)) and f not in SKIP and not f.startswith("history/premarket/events/")]


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()


def _read(p):
    try:
        return json.loads((config.ROOT / p).read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def manifest(fs: list[str], blobs: dict[str, bytes]) -> dict:
    code = [f for f in fs if f.startswith(CODE)]
    code_hash = hashlib.sha256("".join(f + hashlib.sha256(blobs[f]).hexdigest() for f in sorted(code)).encode()).hexdigest()
    sc, cal, e2e = _read("history/predictions/scorecard.json"), _read("history/audit/seal_calendar.json"), _read("history/audit/home_e2e.json")
    last_code = _git("log", "-1", "--format=%h %cI %s", "--", *[c.rstrip("/") for c in CODE])
    return {
        "schema_version": 1,
        "packed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"), "commit_time": _git("log", "-1", "--format=%cI"),
        "last_code_commit": last_code,
        "code_sha256": code_hash,
        "note": "code_sha256 只算程式檔；兩份 zip 的 code_sha256 相同＝程式完全沒改，只有每日紀錄（history/）可能不同",
        "data": {"last_data_date": (cal.get("summary") or {}).get("last_data"), "scorecard_generated_at": sc.get("generated_at"),
                 "scorecard_last": sc.get("last"), "matured": {h: (v or {}).get("matured") for h, v in (sc.get("horizons") or {}).items()},
                 "seal_calendar": {k: (cal.get("summary") or {}).get(k) for k in ("expected", "sealed", "missing", "next_due")},
                 "site_e2e": {k: e2e.get(k) for k in ("at", "passed", "total", "as_of")}},
        "files": {f: hashlib.sha256(blobs[f]).hexdigest()[:16] for f in fs},
    }


def build(out: str) -> tuple[bytes, dict, dict]:
    fs = files()
    blobs = {f: (config.ROOT / f).read_bytes() for f in fs}
    m = manifest(fs, blobs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("tw-stock-screener/MANIFEST.json", json.dumps(m, ensure_ascii=False, indent=1))
        for f in fs:
            z.writestr("tw-stock-screener/" + f, blobs[f])
    open(out, "wb").write(buf.getvalue())
    return buf.getvalue(), m, blobs


if __name__ == "__main__":
    data, m, _ = build(sys.argv[1] if len(sys.argv) > 1 else "tw-stock-screener.zip")
    print(json.dumps({k: m[k] for k in ("commit", "last_code_commit", "code_sha256", "data")}, ensure_ascii=False, indent=1), len(m["files"]), "files")
