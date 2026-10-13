"""會員通知後端（盤後更新的最後一步執行）。

使用者沒打開網站，也會依自選股產生通知。只有這支程式能寫入 user_notifications 和 alert_processing_state；
網站上的會員只能讀取自己的通知、標為已讀、修改自己的通知規則（supabase/schema.sql）。

需要 GitHub Secret「SUPABASE_SECRET_KEY」（Supabase 的 secret key，sb_secret_ 開頭）。這把金鑰只在 Actions 的這一步使用，
不會寫進網站。沒有設定就跳過，網站改回在瀏覽器裡產生、只存在那台裝置的通知。

判斷規則和網站上的 alertEvents（static/index.html <alert-core>）相同，tests/alert_cases.json 用同一組案例測兩邊。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

from . import config

TYPES = {"signal": {}, "top": {}, "score": {"min": 10}, "risk": {}, "exdiv": {"days": 3}}
TOP_N = 20


def _f1(x: float) -> str:
    return f"{x:.1f}"


def events(snap: dict, watch_ids: list[str], rules: dict, state: dict | None, today: str) -> tuple[list[dict], dict]:
    """回傳 (通知清單, 新的比較基準)。純函式，和網站的 alertEvents 結果一致。"""
    date_ = (snap.get("data_dates") or {}).get("prices")
    by = {s["stock_id"]: s for s in snap.get("stocks", [])}
    on = lambda k: (rules.get(k) or {}).get("enabled") is not False
    prm = lambda k: {**TYPES[k], **((rules.get(k) or {}).get("params") or {})}
    out, base = [], {}
    prev = (state.get("base") or {}) if state and state.get("date") and date_ and state["date"] < date_ else None
    nm = lambda s: f"{s['stock_id']} {s['name']}"
    for sid in watch_ids:
        s = by.get(sid)
        if not s or not date_:
            continue
        m = ((s.get("ai") or {}).get("modes") or {}).get("balanced") or {}
        ex = s.get("exclude_reasons") or []
        base[sid] = {"s": m.get("score"), "r": m.get("rank"), "ok": bool(s.get("base_ok")), "ex": ex}
        if state and state.get("date") == date_:
            base[sid] = (state.get("base") or {}).get(sid) or base[sid]
        t = s.get("timing") or {}
        if on("signal") and t.get("status") == "buy" and s.get("date") == date_:
            out.append({"key": f"signal:{sid}:{date_}", "type": "signal", "stock_id": sid, "data_date": date_, "title": f"{nm(s)} 出現技術訊號",
                        "body": "、".join((snap.get("triggers") or {}).get(k, k) for k in t.get("triggers") or []) + "。技術型態提醒，歷史回測並未證明能勝過大盤。"})
        p = (prev or {}).get(sid)
        if not p:
            continue
        r, sc = m.get("rank"), m.get("score")
        if on("top") and r is not None and r <= TOP_N and (p.get("r") is None or p["r"] > TOP_N):
            was = "沒有排名" if p.get("r") is None else f"第 {p['r']} 名"
            out.append({"key": f"top:{sid}:{date_}", "type": "top", "stock_id": sid, "data_date": date_,
                        "title": f"{nm(s)} 進入 AI 均衡型前 20 名", "body": f"第 {r} 名（上次{was}）。"})
        if on("score") and sc is not None and p.get("s") is not None and abs(sc - p["s"]) >= prm("score")["min"]:
            out.append({"key": f"score:{sid}:{date_}", "type": "score", "stock_id": sid, "data_date": date_,
                        "title": f"{nm(s)} AI 分數{'上升' if sc > p['s'] else '下降'} {_f1(abs(sc - p['s']))} 分",
                        "body": f"{_f1(p['s'])} → {_f1(sc)}（全市場百分位，不是上漲機率）。"})
        added = [x for x in ex if x not in (p.get("ex") or [])]
        dropped_gate = p.get("ok") and not s.get("base_ok")
        if on("risk") and (dropped_gate or added):
            out.append({"key": f"risk:{sid}:{date_}", "type": "risk", "stock_id": sid, "data_date": date_,
                        "title": f"{nm(s)} {'不再通過基本門檻' if dropped_gate else '出現新的排除原因'}",
                        "body": "、".join(added or ex) or "請到個股頁查看。"})
    if on("exdiv"):
        lim = (date.fromisoformat(today) + timedelta(days=prm("exdiv")["days"])).isoformat()
        for e in (snap.get("calendar") or {}).get("events") or []:
            if e.get("event_type") == "dividend" and e.get("status") == "confirmed" and e.get("stock_id") in watch_ids and today <= e["event_date"] <= lim:
                name = e.get("name") or (by.get(e["stock_id"]) or {}).get("name") or ""
                out.append({"key": f"exdiv:{e['stock_id']}:{e['event_date']}", "type": "exdiv", "stock_id": e["stock_id"], "data_date": date_,
                            "title": f"{e['stock_id']} {name} {e['event_date']} {e['title'].split('（')[0]}",
                            "body": f"{e['title']}。前一個交易日收盤前買進才參加；金額以公司公告為準。"})
    return out, {"date": date_, "base": base}


def _check(r) -> None:
    """失敗時把 Supabase 回的錯誤原因（例如 permission denied）一起記下來，方便判斷是金鑰還是權限的問題。"""
    if getattr(r, "status_code", 200) >= 400:
        try:
            body = r.json()
            why = " ".join(str(body.get(k) or "") for k in ("code", "message", "hint")).strip()
        except Exception:
            why = (r.text or "")[:120]
        raise requests.HTTPError(f"{r.status_code} {why}".strip(), response=r)


def key_kind(key: str) -> str:
    """只回報金鑰種類，不回報內容。"""
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("sb_publishable_"):
        return "publishable（貼錯了，要用 sb_secret_ 開頭的那一把）"
    if key.startswith("eyJ"):
        return "舊版 JWT"
    return "無法辨識"


class Supa:
    """用 secret key 呼叫 Supabase REST（只在 Actions 裡用）。"""

    def __init__(self, url: str, key: str, session=None):
        self.url = url.rstrip("/") + "/rest/v1/"
        self.s = session or requests.Session()
        self.s.headers.update({"apikey": key, "Content-Type": "application/json"})
        if key.startswith("eyJ"):  # 舊版 service_role JWT 也要放 Authorization
            self.s.headers["Authorization"] = f"Bearer {key}"

    def get(self, table: str, select: str) -> list[dict]:
        out, step = [], 1000
        while True:
            r = self.s.get(self.url + table, params={"select": select, "order": "user_id", "limit": step, "offset": len(out)}, timeout=30)
            _check(r)
            rows = r.json()
            out += rows
            if len(rows) < step:
                return out

    def insert(self, table: str, rows: list[dict], on_conflict: str, merge: bool) -> None:
        if not rows:
            return
        pref = "resolution=merge-duplicates" if merge else "resolution=ignore-duplicates"
        r = self.s.post(self.url + table, params={"on_conflict": on_conflict}, data=json.dumps(rows, ensure_ascii=False).encode(),
                        headers={"Prefer": f"{pref},return=minimal"}, timeout=60)
        _check(r)


def run(snap: dict, supa: Supa, today: str | None = None) -> dict:
    today = today or config.today().isoformat()
    users = supa.get("user_state", "user_id,watch")
    rules: dict[str, dict] = {}
    for r in supa.get("alert_rules", "user_id,rule_type,enabled,params"):
        rules.setdefault(r["user_id"], {})[r["rule_type"]] = {"enabled": r["enabled"], "params": r.get("params") or {}}
    states = {r["user_id"]: {"date": r.get("data_date"), "base": r.get("baseline") or {}} for r in supa.get("alert_processing_state", "user_id,data_date,baseline")}
    notes, new_states, n_users = [], [], 0
    now = config.now().isoformat(timespec="seconds") + "+08:00"
    for u in users:
        watch = list((u.get("watch") or {}).keys())
        if not watch:
            continue
        n_users += 1
        ns, st = events(snap, watch, rules.get(u["user_id"], {}), states.get(u["user_id"]), today)
        notes += [{"user_id": u["user_id"], "event_key": x["key"], "rule_type": x["type"], "stock_id": x["stock_id"],
                   "title": x["title"][:200], "body": (x["body"] or "")[:1000], "data_date": x["data_date"]} for x in ns]
        if st["date"]:
            new_states.append({"user_id": u["user_id"], "data_date": st["date"], "baseline": st["base"], "updated_at": now})
    for i in range(0, len(notes), 500):
        supa.insert("user_notifications", notes[i:i + 500], "user_id,event_key", merge=False)
    supa.insert("alert_processing_state", new_states, "user_id", merge=True)
    return {"ok": True, "ran_at": now, "data_date": (snap.get("data_dates") or {}).get("prices"), "users": n_users, "candidates": len(notes)}


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("site/data/alerts/engine.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        res = {"ok": False, "enabled": False, "reason": "還沒有設定 SUPABASE_SECRET_KEY，通知只在瀏覽器裡產生"}
    else:
        try:
            snap = json.loads((config.DATA_DIR / "snapshot.json").read_text("utf-8"))
            res = {"enabled": True, **run(snap, Supa(url, key))}
        except Exception as e:  # 失敗只記錄，不影響網站發布；不寫出金鑰或會員資料
            res = {"ok": False, "enabled": True, "ran_at": config.now().isoformat(timespec="seconds") + "+08:00",
                   "key_kind": key_kind(key), "reason": f"{type(e).__name__}: {str(e)[:200]}".replace(key, "***")}
    out.write_text(json.dumps(res, ensure_ascii=False), "utf-8")
    print("會員通知：", json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
