"""V22 加權指數開盤價（證交所 MI_5MINS_HIST，一次一個月的開高低收）→ 資料表 index_ohlc。

為什麼要有：選股和 0050 都是「進場日開盤 → 出場日開盤」，加權指數原本只有收盤價，只能用不同時段近似比較
（10/30 那批 0050 開盤到開盤 −3.68%，收盤到收盤 0.00%，差別全在 10/30 開盤跳空）。
2026-11-01 第一次抓回來核對：開盤指數是 9:00 第一次計算，很多股票還沒成交、用前一天收盤代入，和前一天收盤平均只差 0.42%，
不等於可成交的開盤價。所以 fair_compare 只把它列參考，相對加權超額用收盤口徑；收盤欄位和既有加權指數收盤逐日核對。

每次執行：抓本月與上個月，再往前回補最多 BACKFILL 個月（從 2019-06 開始），已經有的月份不重抓。
失敗不影響其他步驟；比較時缺開盤價就照實標示「沒有開盤價」，不拿收盤價代替。
用法：python -m app.taiex_ohlc
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date

import requests

from . import config, db
from .sources import http
from .sources.twse import BASE, HEADERS, num

START = "2019-06"
BACKFILL = 12
INTERVAL = 3.5


def roc(s) -> str | None:
    """'115/10/01' 或 '1151001' → '2026-10-01'"""
    p = str(s or "").strip().replace("/", "")
    if len(p) == 7 and p.isdigit():
        return f"{int(p[:3]) + 1911}-{p[3:5]}-{p[5:]}"
    return None


def parse(js: dict) -> list[dict]:
    """MI_5MINS_HIST 回應 → [{date, open, high, low, close}]；欄位用名稱找，不靠位置。"""
    if js.get("stat") != "OK":
        return []
    f = js.get("fields") or []
    def ix(key):
        return next(i for i, x in enumerate(f) if key in str(x))
    i = {"date": ix("日期"), "open": ix("開盤"), "high": ix("最高"), "low": ix("最低"), "close": ix("收盤")}
    out = []
    for r in js.get("data") or []:
        d = roc(r[i["date"]])
        row = {"date": d, **{k: num(r[i[k]]) for k in ("open", "high", "low", "close")}}
        if d and row["open"] and row["close"]:
            out.append(row)
    return out


def months(today: date, have: set[str]) -> list[str]:
    cur = today.year * 12 + today.month - 1
    first = int(START[:4]) * 12 + int(START[5:]) - 1
    recent = [cur, cur - 1]
    old = [k for k in range(cur - 2, first - 1, -1) if f"{k // 12}-{k % 12 + 1:02d}" not in have][:BACKFILL]
    return [f"{k // 12}-{k % 12 + 1:02d}" for k in recent + old]


def fill_index_prices(con) -> list[str]:
    """2026-11-19：每日行情走 OpenAPI 備援的那天（例如 11/16、10/07）股價有了，加權指數收盤卻沒有，
    0050／加權指數比較和封存日曆會少一天。用同一個官方來源（證交所 MI_5MINS_HIST）的收盤補上，只補缺的、不覆寫既有的。"""
    cand = [r[0] for r in con.execute("SELECT date FROM index_ohlc o WHERE close IS NOT NULL"
                                      " AND NOT EXISTS (SELECT 1 FROM index_prices p WHERE p.date=o.date) ORDER BY date")]
    # 只補股價那天確實有資料的日子（prices 主鍵是 stock_id+date，用 0050 查才走得到索引）
    days = [d for d in cand if con.execute("SELECT 1 FROM prices WHERE stock_id='0050' AND date=?", (d,)).fetchone()]
    if days:
        q = ",".join("?" * len(days))
        con.execute(f"INSERT OR IGNORE INTO index_prices (date, close) SELECT date, close FROM index_ohlc WHERE date IN ({q})", days)
        con.commit()
    return days


def run(session: requests.Session | None = None) -> dict:
    s = session or requests.Session()
    s.headers.update(HEADERS)
    with db.connect() as con:
        have = {r[0] for r in con.execute("SELECT DISTINCT substr(date,1,7) FROM index_ohlc")}
        todo = months(config.today(), have)
        got, errors = 0, []
        for n, ym in enumerate(todo):
            if n:
                time.sleep(INTERVAL)
            try:
                js = http.get(s, BASE + "/rwd/zh/TAIEX/MI_5MINS_HIST", params={"date": ym.replace("-", "") + "01", "response": "json"}).json()
                rows = parse(js)
            except (requests.RequestException, ValueError, StopIteration) as e:
                errors.append(f"{ym}: {http.describe(e) if isinstance(e, Exception) else e}")
                continue
            if rows:
                db.upsert(con, "index_ohlc", rows)
                con.commit()
                got += len(rows)
        filled = fill_index_prices(con)
        span = con.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM index_ohlc").fetchone()
        # 和既有的加權指數收盤價核對（同一天收盤要一樣）
        bad = con.execute("SELECT COUNT(*) FROM index_ohlc o JOIN index_prices p USING(date) WHERE ABS(o.close - p.close) > 0.05").fetchone()[0]
    return {"months": todo, "rows": got, "errors": errors, "first": span[0], "last": span[1], "days": span[2], "close_mismatch": bad,
            "filled_close": filled}


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, ensure_ascii=False))
    sys.exit(0 if not r["errors"] or r["rows"] else 1)
