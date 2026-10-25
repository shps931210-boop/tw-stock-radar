"""盤中暫時行情的存放處（V12）：data/intraday/quotes.json，和正式盤後資料（資料庫 prices 表）完全分開。

- 這裡的資料不會寫進資料庫，AI 預測、封存、對帳、訓練都只讀資料庫的正式收盤價。
- 寫入是「先寫暫存檔再改名」，同時有很多人讀也不會讀到寫一半的檔案。
- 讀取時檢查是否過期；過期或沒有盤中資料就退回盤後正式價格，並清楚標示價格時間。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .. import config
from .providers import Quote, parse_ts

DIR = config.DATA_DIR / "intraday"
FILE = DIR / "quotes.json"
STALE_SECONDS = 120     # 盤中超過 2 分鐘沒有新行情就標示過期

LABEL = {"intraday": "盤中", "simulated": "模擬（不是真實行情）", "official": "盤後"}


def write(quotes: dict[str, Quote], status: dict, path: Path = FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": 1, "kind": "intraday", "written_at": config.now().isoformat(timespec="seconds"), "status": status,
           "quotes": {s: q.to_dict() for s, q in sorted(quotes.items())}}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".quotes-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def read(path: Path = FILE) -> dict | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def view(sid: str, official: Quote | None, path: Path = FILE, now=None, stale: int = STALE_SECONDS) -> dict:
    """網站要顯示的一檔價格：有新鮮的盤中行情就用盤中，否則退回盤後正式價格。一定附上價格時間與來源。"""
    now = now or config.now()
    doc = read(path)
    q = (doc or {}).get("quotes", {}).get(sid)
    if q:
        age = (now - parse_ts(q["ts"])).total_seconds()
        fresh = age <= stale and (doc.get("status") or {}).get("connected", False)
        if fresh:
            return {**q, "label": LABEL[q["kind"]], "stale": False, "age_seconds": round(age)}
        if official is None:
            return {**q, "label": LABEL[q["kind"]] + "・已過期", "stale": True, "age_seconds": round(age)}
    if official is None:
        return {"sid": sid, "price": None, "label": "沒有資料", "stale": True}
    return {**official.to_dict(), "label": LABEL["official"], "stale": False,
            "note": "盤中行情未連線或已過期，顯示最後正式收盤價" if q else None}
