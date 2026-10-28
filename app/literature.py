"""學術證據庫（history/research/literature.json）的 DOI 核對：用 Crossref 公開 API 比對題名、年份，順便看有沒有撤稿／更正紀錄。

只在 GitHub Actions（每週回測）執行；免費、不需要金鑰。每篇之間停 1 秒，遵守 Crossref 的禮貌使用規範。
結果寫回每篇的 doi_check，不改其他欄位。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

from . import config

LIB = Path(__file__).resolve().parent.parent / "history" / "research" / "literature.json"
API = "https://api.crossref.org/works/"
HEADERS = {"User-Agent": "tw-stock-radar/1.0 (https://github.com/shps931210-boop/tw-stock-radar)"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def compare(item: dict, msg: dict) -> dict:
    title = (msg.get("title") or [""])[0]
    year = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0][0]
    a, b = _norm(item["title"]), _norm(title)
    upd = [u.get("type") for u in msg.get("updated-by", []) or []] + [r.get("type") for r in (msg.get("relation") or {}).get("is-retracted-by", [])]
    return {"ok": bool(a and b and (a in b or b in a)), "crossref_title": title, "crossref_year": year,
            "year_match": year == item.get("year"), "journal": (msg.get("container-title") or [""])[0],
            "updates": [u for u in upd if u], "checked_at": config.now().isoformat(timespec="seconds")}


def verify(path: Path = LIB, sleep: float = 1.0, get=requests.get) -> dict:
    L = json.loads(path.read_text("utf-8"))
    for it in L["items"]:
        if not it.get("doi"):
            it["doi_check"] = {"ok": None, "note": "沒有 DOI（學位論文、工作論文或期刊網頁），需人工核對"}
            continue
        try:
            r = get(API + it["doi"], headers=HEADERS, timeout=30)
            it["doi_check"] = compare(it, r.json()["message"]) if r.status_code == 200 else {"ok": False, "note": f"Crossref 回應 {r.status_code}"}
        except Exception as e:  # 網路問題不擋研究流程
            it["doi_check"] = {"ok": None, "note": f"查詢失敗：{type(e).__name__}"}
        time.sleep(sleep)
    L["doi_checked_at"] = config.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(L, ensure_ascii=False, indent=1), "utf-8")
    return {"ok": sum(1 for x in L["items"] if (x.get("doi_check") or {}).get("ok")),
            "failed": [x["id"] for x in L["items"] if (x.get("doi_check") or {}).get("ok") is False]}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
    sys.exit(0)
