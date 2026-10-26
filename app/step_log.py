"""非必要步驟的執行紀錄（history/audit/steps.json）。

盤後更新裡有些步驟設成「失敗不影響網站」（影子預測、新聞、國際市場、盤前情報、連線診斷…），
以前失敗只留在 Actions 紀錄裡，網站和通知信都看不到。現在每次執行都記下每個步驟的結果；
同一個步驟連續失敗 2 次以上，app.check 會讓這次執行標成失敗並寄信，網站「資料來源與方法」也會顯示。
用法：python -m app.step_log "影子預測=success" "新聞=failure" ...（skipped 表示這個模式不跑，不算失敗）
"""
from __future__ import annotations

import json
import os
import sys

from . import config

PATH = config.ROOT / "history" / "audit" / "steps.json"
KEEP = 120
ALERT = 2


def load(path=None) -> dict:
    p = path or PATH
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return {"runs": []}


def record(pairs: list[str], path=None) -> dict:
    p = path or PATH
    d = load(p)
    steps = {}
    for x in pairs:
        k, _, v = x.partition("=")
        if k and v:
            steps[k] = v
    rid = os.environ.get("GITHUB_RUN_ID")
    d["runs"] = [r for r in d.get("runs", []) if not rid or r.get("run_id") != rid]
    d["runs"].append({"at": config.now().isoformat(timespec="seconds"), "run_id": rid, "mode": os.environ.get("MODE"), "steps": steps})
    d["runs"] = d["runs"][-KEEP:]
    d["status"] = status(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
    return d


def status(d: dict) -> dict:
    """每個步驟：最近一次結果、連續失敗次數、最後成功時間（skipped 不計入）。"""
    out = {}
    for r in d.get("runs", []):
        for k, v in r["steps"].items():
            if v in ("skipped", ""):
                continue
            s = out.setdefault(k, {"streak": 0, "last_ok": None, "last": None, "runs": 0, "fails": 0})
            s["runs"] += 1
            s["last"] = v
            if v == "success":
                s["streak"], s["last_ok"] = 0, r["at"]
            else:
                s["streak"] += 1
                s["fails"] += 1
    return out


def alerts(path=None) -> list[str]:
    st = status(load(path))
    return [f"「{k}」連續失敗 {s['streak']} 次（最後成功：{(s['last_ok'] or '從未')[:16].replace('T', ' ')}）" for k, s in st.items() if s["streak"] >= ALERT]


if __name__ == "__main__":
    d = record(sys.argv[1:])
    for k, s in d["status"].items():
        print(f"{k}: {s['last']}（連續失敗 {s['streak']}）")
