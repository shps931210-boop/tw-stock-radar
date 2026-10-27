"""預測封存日曆（V20.1）：從第一次封存開始，每一個應該有預測的交易日都列出來。

每一天回答四件事：
1. 這天是不是交易日？不是的話為什麼（週末、證交所休市日曆、非例行休市如颱風假）。
2. 影子預測（history/predictions/）有沒有封存？在隔天開盤前封存的嗎？用的是哪一版模型？
3. 正式精選紀錄（history/picks/）有沒有留下？
4. 5／20／60 日預測什麼時候到期、到期了沒、成績單（scorecard.json）有沒有算進去。

沒有封存的交易日照實標「缺漏」並寫原因，不補做、不回填。輸出 history/audit/seal_calendar.json（網站 data/predictions/calendar.json）。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from . import config

PRED = config.ROOT / "history" / "predictions"
PICKS = config.ROOT / "history" / "picks"
HOL = config.ROOT / "history" / "premarket" / "holidays.json"
NOTES = config.ROOT / "history" / "audit" / "closure_notes.json"   # 非例行休市的原因（人工註記，只寫官方公告過的事實）
OUT = config.ROOT / "history" / "audit" / "seal_calendar.json"
HORIZONS = (5, 20, 60)
OPEN_AT = "09:00"


def _read(p: Path):
    try:
        return json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def build(today: date | None = None, data_days: list[str] | None = None, pred: Path = PRED, picks: Path = PICKS,
          hol: Path = HOL, notes: Path = NOTES) -> dict:
    """data_days：資料庫實際有行情的日期（有資料庫時傳入，用來找出颱風假這類非例行休市）；沒有時只用休市日曆。"""
    today = today or config.today()
    H = set((_read(hol) or {}).get("dates", []))
    N = _read(notes) or {}
    files = sorted(pred.glob("????-??-??.json")) if pred.exists() else []
    if not files:
        return {"days": [], "summary": {"expected": 0}}
    first = date.fromisoformat(files[0].stem)
    sealed = {f.stem: _read(f) or {} for f in files}
    sc = _read(pred / "scorecard.json") or {}
    have = set(data_days or [])
    last_data = max(have) if have else max(sealed)
    pk = {p.name.split(".")[0] for p in picks.glob("????-??-??*.json")} if picks.exists() else set()

    def closed(d: date) -> str | None:
        s = d.isoformat()
        if d.weekday() >= 5:
            return "週末"
        if s in H:
            return N.get(s) or "證交所休市日曆"
        if have and s <= last_data and s not in have:
            return N.get(s) or "非例行休市：官方當天沒有行情（例如颱風停止交易）"
        if s in N:
            return N[s]
        return None

    def trading_after(d: date, n: int) -> date:
        x = d
        while n:
            x += timedelta(days=1)
            if not closed(x):
                n -= 1
        return x

    rows = []
    d = first
    while d <= today:
        s = d.isoformat()
        why = closed(d)
        if why and why != "週末":
            rows.append({"date": s, "trading": False, "reason": why})
        elif not why:
            rec = sealed.get(s)
            entry = trading_after(d, 1)
            r = {"date": s, "trading": True, "entry": entry.isoformat(), "picks": s in pk,
                 "pre_open": (pred / "pre_open" / f"{s}.json").exists()}  # 這一天開盤前有沒有封存盤前版（V26）
            if rec:
                g = rec.get("generated_at") or ""
                r.update(sealed=True, generated_at=g, on_time=g[:16] < f"{entry.isoformat()}T{OPEN_AT}" if g else None,
                         model_version=rec.get("model_version"), n=rec.get("n"), status=rec.get("status"),
                         data_ok=(rec.get("data_complete") or {}).get("ok"))
            elif s > last_data:
                r.update(sealed=False, reason="今天的收盤資料還沒更新（完整更新後才會封存）" if s == today.isoformat() else "資料還沒更新到這一天")
            else:
                r.update(sealed=False, reason="缺漏：這個交易日沒有封存的預測，不會事後補做")
            mat = {}
            for h in HORIZONS:
                due = trading_after(entry, h)  # 進場日後第 h 個交易日開盤出場；那天收盤資料進來後才能對帳
                mat[str(h)] = {"due": due.isoformat(), "matured": bool(rec) and due.isoformat() <= last_data}
            r["maturity"] = mat
            rows.append(r)
        d += timedelta(days=1)
    tr = [r for r in rows if r["trading"] and r["date"] <= last_data]
    nxt = sorted((m["due"], h) for r in rows if r.get("sealed") for h, m in r["maturity"].items() if not m["matured"])
    summary = {
        "first": first.isoformat(), "last_data": last_data, "today": today.isoformat(),
        "expected": len(tr), "sealed": sum(bool(r.get("sealed")) for r in tr), "on_time": sum(bool(r.get("on_time")) for r in tr),
        "missing": [r["date"] for r in tr if not r.get("sealed")],
        "picks": sum(bool(r.get("picks")) for r in tr),
        "pre_open": sum(bool(r.get("pre_open")) for r in tr),
        "closed": [{"date": r["date"], "reason": r["reason"]} for r in rows if not r["trading"]],
        "matured": {str(h): sum(1 for r in tr if r.get("maturity", {}).get(str(h), {}).get("matured")) for h in HORIZONS},
        "scorecard": {"generated_at": sc.get("generated_at"), "days": sc.get("days"), "last": sc.get("last"),
                      "matured": {h: (v or {}).get("matured", 0) for h, v in (sc.get("horizons") or {}).items()}},
        "next_due": {"date": nxt[0][0], "horizon": nxt[0][1]} if nxt else None,
    }
    # 成績單有沒有跟上：日曆算出已到期、成績單卻還是 0，代表對帳步驟沒跑或失敗
    lag = {h: summary["matured"][h] - int(summary["scorecard"]["matured"].get(h) or 0) for h in summary["matured"]}
    summary["scorecard_lag"] = {h: v for h, v in lag.items() if v > 0}
    summary["scorecard_stale"] = bool(sc.get("last")) and sc.get("last") < max((r["date"] for r in tr if r.get("sealed")), default="")
    return {"schema_version": 1, "generated_at": config.now().isoformat(timespec="seconds"), "summary": summary, "days": rows,
            "rule": "訊號日收盤後封存，隔一個交易日開盤進場，第 h 個交易日後開盤出場；出場日收盤資料進來後才算到期。"
                    "缺漏的交易日不補做，週末不列出。"}


def write(data_days: list[str] | None = None, out: Path = OUT) -> dict:
    cal = build(data_days=data_days)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, ensure_ascii=False, indent=1), "utf-8")
    return cal


def db_days(dbname: str = "stocks.db", since: str = "2026-01-01") -> list[str] | None:
    try:
        from . import db
        with db.connect(dbname) as con:
            return [r[0] for r in con.execute("SELECT DISTINCT date FROM index_prices WHERE date>=? ORDER BY date", (since,))]
    except Exception:  # noqa: BLE001  沒有資料庫時只用休市日曆
        return None


if __name__ == "__main__":
    c = write(db_days())
    print(json.dumps(c["summary"], ensure_ascii=False, indent=1))
