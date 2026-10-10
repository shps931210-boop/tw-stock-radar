"""網站後端：python -m app.server  然後打開 http://127.0.0.1:8000

啟動後會依 config.json 的 schedule（預設平日 14:45、21:45）自動更新資料，也可以在網頁上按「立即更新」。
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from . import config, db, update

app = FastAPI(title="股栗子")
STATIC = config.ROOT / "static"
JOB = {"running": False, "stage": "", "done": 0, "total": 0, "msg": "", "started": None, "finished": None, "errors": []}
_lock = threading.Lock()
_last_manual = 0.0
MANUAL_COOLDOWN = 600  # 手動更新至少間隔 10 分鐘


def _require_admin(request: Request, token: str | None) -> None:
    """更新資料只限管理員：有設定環境變數 ADMIN_TOKEN 時要帶 X-Admin-Token；沒設定時只接受本機連線。"""
    want = os.environ.get("ADMIN_TOKEN")
    if want:
        if not token or not hmac.compare_digest(token, want):
            raise HTTPException(403, "需要管理員權限才能更新資料")
    elif (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "只有本機可以觸發更新；公開部署請設定 ADMIN_TOKEN")


def _snapshot_path(demo: bool):
    """只有明確要求 ?demo=1 才給示範資料；真實資料不存在時絕不自動改用示範資料。"""
    if demo:
        return config.DATA_DIR / "snapshot_demo.json", True
    return config.DATA_DIR / "snapshot.json", False


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/snapshot")
def snapshot(demo: bool = False):
    p, _ = _snapshot_path(demo)
    if p.exists():
        return FileResponse(p, media_type="application/json")
    errors, last = [], None
    if (config.DATA_DIR / "stocks.db").exists():
        with db.connect() as con:
            errors = (db.get_meta(con, "last_errors") or "").splitlines()
            last = db.get_meta(con, "last_update")
    return JSONResponse(status_code=503, content={
        "no_data": True, "running": JOB["running"], "last_attempt": last, "errors": errors[:15] or JOB["errors"][:15],
        "detail": "還沒有成功取得真實行情，所以沒有選股結果。" + ("" if JOB["running"] else "請按「立即更新資料」或執行 python -m app.update。")})


def _rows(con, sql, params):
    return db.read(con, sql, params).to_dict("records")


def stock_detail(sid: str, dbname: str) -> dict:
    with db.connect(dbname) as con:
        px = _rows(con, "SELECT date,open,high,low,close,volume,value FROM prices WHERE stock_id=? ORDER BY date DESC LIMIT 520", (sid,))
        if not px:
            raise HTTPException(404, f"資料庫裡沒有 {sid}")
        out = {
            "prices": px[::-1],
            "index": _rows(con, "SELECT date, close FROM index_prices ORDER BY date DESC LIMIT 520", ())[::-1],
            "revenue": _rows(con, "SELECT ym, revenue FROM revenue WHERE stock_id=? ORDER BY ym DESC LIMIT 48", (sid,))[::-1],
            "financials": _rows(con, "SELECT * FROM financials WHERE stock_id=? ORDER BY period_end DESC LIMIT 12", (sid,))[::-1],
            "institutional": _rows(con, "SELECT date, foreign_net, trust_net, dealer_net FROM institutional WHERE stock_id=? ORDER BY date DESC LIMIT 60", (sid,))[::-1],
            "margin": _rows(con, "SELECT date, margin_bal, short_bal FROM margin WHERE stock_id=? ORDER BY date DESC LIMIT 60", (sid,))[::-1],
            "valuation": _rows(con, "SELECT date, per, pbr, dy FROM valuation WHERE stock_id=? ORDER BY date DESC LIMIT 260", (sid,))[::-1],
            "dividends": _rows(con, "SELECT year, cash, stock FROM dividends WHERE stock_id=? ORDER BY year", (sid,)),
            "news": _rows(con, "SELECT date, title, source, link FROM news WHERE stock_id=? ORDER BY date DESC LIMIT 15", (sid,)),
        }
    return json.loads(json.dumps(out, default=str).replace("NaN", "null"))


@app.get("/api/stock/{sid}")
def stock(sid: str, demo: bool = False):
    _, is_demo = _snapshot_path(demo)
    if not is_demo:
        update.fetch_news(sid)
    return JSONResponse(stock_detail(sid, "demo.db" if is_demo else "stocks.db"))


@app.get("/api/status")
def status():
    return JOB


def _run_update(daily=True, fundamentals=True):
    if not _lock.acquire(blocking=False):
        return False

    def progress(stage, done, total, msg=""):
        JOB.update(stage=stage, done=done, total=total, msg=msg)

    def work():
        try:
            JOB.update(running=True, started=config.now().isoformat(timespec="seconds"), errors=[])
            JOB["errors"] = update.run(daily, fundamentals, progress)
        except Exception as e:
            JOB["errors"] = [str(e)]
        finally:
            JOB.update(running=False, stage="完成", finished=config.now().isoformat(timespec="seconds"))
            _lock.release()

    threading.Thread(target=work, daemon=True).start()
    return True


@app.post("/api/update")
def run_update(request: Request, fundamentals: bool = True, x_admin_token: str | None = Header(None)):
    global _last_manual
    _require_admin(request, x_admin_token)
    if time.time() - _last_manual < MANUAL_COOLDOWN:
        raise HTTPException(429, "剛剛更新過，請 10 分鐘後再試")
    if not _run_update(True, fundamentals):
        raise HTTPException(409, "更新正在進行中")
    _last_manual = time.time()
    print(f"[{config.now():%Y-%m-%d %H:%M:%S}] 手動更新 from {request.client.host if request.client else '?'}", flush=True)
    return {"started": True}


def _scheduler():
    done_slots = set()
    while True:
        now = config.now()
        slot = now.strftime("%H:%M")
        if now.weekday() < 5 and slot in config.load().get("schedule", []):
            key = now.strftime("%Y-%m-%d ") + slot
            if key not in done_slots and _run_update():
                done_slots.add(key)
        time.sleep(20)


@app.on_event("startup")
def _start_scheduler():
    threading.Thread(target=_scheduler, daemon=True).start()


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
