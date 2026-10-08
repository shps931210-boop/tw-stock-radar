"""資料更新：把各來源的最新資料補進資料庫，再重算選股快照。

用法：
    python -m app.update            # 增量更新（第一次會回補約一年半股價，約需 40~60 分鐘）
    python -m app.update --demo     # 重建示範資料庫
    python -m app.update --daily-only / --fundamentals-only
"""
from __future__ import annotations

import argparse
import re
import traceback
from datetime import date, datetime, timedelta

import requests

from . import config, db
from .sources import http
from .sources.finmind import FinMindError, FinMindSource
from .sources.tpex import TpexSource
from .sources.twse import TwseSource, num

Progress = callable


def _say(progress, stage: str, done: int = 0, total: int = 0, msg: str = ""):
    if progress:
        progress(stage, done, total, msg)
    else:
        print(f"[{stage}] {done}/{total} {msg}", flush=True)


def is_common_stock(sid: str) -> bool:
    return bool(re.fullmatch(r"[1-9]\d{3}", sid))


def is_tracked(sid: str) -> bool:
    """只存普通股與 ETF（00 開頭），權證、可轉債等幾萬檔不存，資料庫與網站才不會過大。"""
    return is_common_stock(sid) or bool(re.fullmatch(r"00\d{2,4}[A-Z]?", sid))


def _need(con, source: str, d: date) -> bool:
    """這一天這個資料是否還需要抓。抓過但沒資料的日子，3 天後就當作休市不再重試。"""
    row = con.execute("SELECT ok FROM fetch_log WHERE source=? AND key=?", (source, d.isoformat())).fetchone()
    if row is None:
        return True
    return row[0] == 0 and (config.today() - d).days <= 3


def update_daily(cfg: dict, progress=None, errors: list | None = None) -> None:
    errors = errors if errors is not None else []
    srcs = []
    if "twse" in cfg["markets"]:
        srcs.append(TwseSource(cfg["twse_interval_sec"]))
    if "tpex" in cfg["markets"]:
        srcs.append(TpexSource(cfg["twse_interval_sec"]))
    now = config.now()
    last = config.today() if now.hour >= 14 else config.today() - timedelta(days=1)
    days = [last - timedelta(days=i) for i in range(cfg["price_backfill_days"])]
    days = [d for d in days if d.weekday() < 5]
    flow_cut = config.today() - timedelta(days=cfg["flow_backfill_days"])
    jobs = [("quotes", "daily_quotes", None), ("inst", "institutional", "institutional"),
            ("margin", "margin", "margin"), ("val", "valuation", "valuation")]

    with db.connect() as con:
        # 清掉以前存進來的權證等非追蹤代號
        junk = [r[0] for r in con.execute("SELECT stock_id FROM stocks") if not is_tracked(r[0])]
        for i in range(0, len(junk), 500):
            q = ",".join("?" * len(junk[i:i + 500]))
            for t in ("prices", "valuation", "institutional", "margin", "stocks"):
                con.execute(f"DELETE FROM {t} WHERE stock_id IN ({q})", junk[i:i + 500])
        con.commit()
        todo = [(s, d, j) for s in srcs for j in jobs for d in days
                if (j[0] == "quotes" or d >= flow_cut) and _need(con, f"{s.market}_{j[0]}", d)]
        dead = set()  # 解析失敗的 (市場, 資料) 本次不再重試
        for n, (s, d, (key, fn, table)) in enumerate(todo, 1):
            tag = f"{s.market}_{key}"
            if tag in dead:
                continue
            _say(progress, "每日行情與籌碼", n, len(todo), f"{s.market} {key} {d}")
            try:
                res = getattr(s, fn)(d)
            except KeyError as e:  # 欄位對不上：整個來源本次停用，避免每天都重複同樣錯誤
                errors.append(f"{tag}: {http.describe(e)}")
                dead.add(tag)
                continue
            except ValueError as e:  # 單日回應不是 JSON（偶發、被限流）：只略過這一天，下次再補
                errors.append(f"{tag} {d}: {http.describe(e)}")
                continue
            except requests.RequestException as e:
                errors.append(f"{tag} {d}: {http.describe(e)}")
                continue
            ds = d.isoformat()
            if key == "quotes":
                rows, taiex = res
                rows = [r for r in rows if is_tracked(r["stock_id"])]
                if taiex:
                    db.upsert(con, "index_prices", [{"date": ds, "close": taiex}])
                for r in rows:
                    con.execute("INSERT OR IGNORE INTO stocks (stock_id, name, market, industry) VALUES (?,?,?,'')",
                                (r["stock_id"], r.pop("name"), s.market))
                    r["date"] = ds
                db.upsert(con, "prices", rows)
            else:
                rows = res
                for r in rows:
                    r["date"] = ds
                db.upsert(con, table, rows)
            db.log_fetch(con, tag, ds, bool(rows))
            con.commit()
        _openapi_fallback(con, cfg, errors)
        db.set_meta(con, "updated_prices", config.now().isoformat(timespec="seconds"))
        # 處置股票
        if srcs and srcs[0].market == "twse":
            try:
                rows = srcs[0].disposition()
                con.execute("DELETE FROM disposition")
                db.upsert(con, "disposition", [{**r, "updated": config.today().isoformat()} for r in rows])
            except Exception as e:
                errors.append(f"處置股票: {http.describe(e)}")


def _openapi_fallback(con, cfg: dict, errors: list) -> None:
    """主要來源沒抓到最新一天時，改用官方 OpenAPI 補上最近一個交易日（只有一天，不能回補歷史）。"""
    from .sources import tpex as P
    from .sources import twse as T
    plan = [("twse", T.openapi_quotes, T.openapi_valuation)]
    if "tpex" in cfg["markets"]:
        plan.append(("tpex", P.openapi_quotes, None))
    for mkt, quotes, valuation in plan:
        have = con.execute("SELECT MAX(p.date) FROM prices p JOIN stocks s USING(stock_id) WHERE s.market=?", (mkt,)).fetchone()[0]
        try:
            day, rows = quotes()
        except Exception as e:
            errors.append(f"{mkt} OpenAPI 行情: {http.describe(e)}")
            continue
        if day:  # 官方實際公布到哪一天：用來判斷網站資料是否過期（自動涵蓋國定假日與颱風假）
            db.set_meta(con, f"official_latest_{mkt}", day)
        rows = [r for r in rows if is_tracked(r["stock_id"])]
        if not day or not rows or (have and day <= have):
            continue
        for r in rows:
            con.execute("INSERT OR IGNORE INTO stocks (stock_id, name, market, industry) VALUES (?,?,?,'')",
                        (r["stock_id"], r.pop("name"), mkt))
            r["date"] = day
        db.upsert(con, "prices", rows)
        db.log_fetch(con, f"{mkt}_quotes", day, True)
        db.log_fetch(con, f"{mkt}_openapi", day, True)
        if valuation:
            try:
                vday, vrows = valuation()
                db.upsert(con, "valuation", [{**r, "date": vday or day} for r in vrows])
                db.log_fetch(con, f"{mkt}_val", vday or day, bool(vrows))
            except Exception as e:
                errors.append(f"{mkt} OpenAPI 本益比: {http.describe(e)}")
        con.commit()


def update_revenue_bulk(cfg: dict, errors: list) -> None:
    """證交所 OpenAPI 一次取得全部上市公司最新月營收（不耗 FinMind 額度）。"""
    urls = ["https://openapi.twse.com.tw/v1/opendata/t187ap05_L"]
    if "tpex" in cfg["markets"]:
        urls.append("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O")  # 上櫃，尚未核對
    with db.connect() as con:
        for u in urls:
            try:
                data = http.get(requests.Session(), u, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).json()
                rows = []
                for x in data:
                    ym = str(x.get("資料年月", ""))
                    if len(ym) < 4:
                        continue
                    y, m = int(ym[:-2]) + 1911, int(ym[-2:])
                    rev = num(x.get("營業收入-當月營收"))
                    if rev is not None:
                        rows.append({"stock_id": str(x["公司代號"]).strip(), "ym": f"{y}-{m:02d}", "revenue": rev * 1000})
                        prev = num(x.get("營業收入-去年當月營收"))
                        if prev is not None:
                            rows.append({"stock_id": str(x["公司代號"]).strip(), "ym": f"{y - 1}-{m:02d}", "revenue": prev * 1000})
                db.upsert(con, "revenue", rows)
                for x in data:
                    if x.get("產業別"):
                        con.execute("UPDATE stocks SET industry=? WHERE stock_id=? AND (industry IS NULL OR industry='')",
                                    (x["產業別"], str(x["公司代號"]).strip()))
            except Exception as e:
                errors.append(f"月營收批次 {u.split('/')[-1]}: {http.describe(e)}")
        db.set_meta(con, "updated_revenue", config.now().isoformat(timespec="seconds"))


def update_fundamentals(cfg: dict, progress=None, errors: list | None = None) -> None:
    errors = errors if errors is not None else []
    fm = FinMindSource(cfg.get("finmind_token", ""), cfg["finmind_interval_sec"])
    with db.connect() as con:
        # 產業別與股票清單，一週更新一次
        t = db.fetched_at(con, "finmind_info", "all")
        if not t or config.now() - t > timedelta(days=7):
            try:
                for r in fm.stock_info():
                    con.execute("INSERT INTO stocks VALUES (?,?,?,?) ON CONFLICT(stock_id) DO UPDATE SET "
                                "industry=excluded.industry, name=excluded.name", (r["stock_id"], r["name"], r["market"], r["industry"]))
                db.log_fetch(con, "finmind_info", "all")
            except Exception as e:
                errors.append(f"股票清單: {http.describe(e)}")
        # 依近 20 日成交金額挑出要抓財報的股票
        liquid = db.read(con, """SELECT stock_id, AVG(value) v FROM prices
            WHERE date >= (SELECT date FROM (SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT 20) ORDER BY date LIMIT 1)
            GROUP BY stock_id ORDER BY v DESC""")
        ids = [s for s in liquid["stock_id"] if is_common_stock(s)][: cfg["fundamental_top_n"]]
        ids = list(dict.fromkeys(cfg.get("watchlist", []) + ids))
        plan = [("fm_revenue", 1 if config.today().day <= 12 else 7, fm.month_revenue, "revenue"),
                ("fm_financials", 20, fm.financials, "financials"),
                ("fm_dividends", 45, fm.dividends, "dividends")]
        todo = []
        for sid in ids:
            for src, age, fn, table in plan:
                t = db.fetched_at(con, src, sid)
                if not t or config.now() - t > timedelta(days=age):
                    todo.append((sid, src, fn, table))
        for n, (sid, src, fn, table) in enumerate(todo, 1):
            _say(progress, "財報與營收", n, len(todo), f"{sid} {src[3:]}")
            try:
                db.upsert(con, table, fn(sid))
                db.log_fetch(con, src, sid)
                con.commit()
            except FinMindError as e:
                errors.append(str(e))
                if "用完" in str(e):
                    break  # 額度用完，下次排程再繼續
            except Exception as e:
                errors.append(f"{sid} {src}: {http.describe(e)}")
        now = config.now().isoformat(timespec="seconds")
        for k in ("revenue", "financials", "dividends"):
            db.set_meta(con, f"updated_{k}", now)


def fetch_news(sid: str, cfg: dict | None = None) -> None:
    cfg = cfg or config.load()
    with db.connect() as con:
        t = db.fetched_at(con, "fm_news", sid)
        if t and config.now() - t < timedelta(hours=3):
            return
        try:
            db.upsert(con, "news", FinMindSource(cfg.get("finmind_token", "")).news(sid))
            db.log_fetch(con, "fm_news", sid)
        except Exception:
            db.log_fetch(con, "fm_news", sid, ok=False)


def run(daily: bool = True, fundamentals: bool = True, progress=None) -> list[str]:
    """分三階段，每階段結束都重算快照，讓網站盡早顯示真實資料：
    ① 最新一個交易日收盤價（OpenAPI，幾秒鐘）→ ② 回補歷史股價與籌碼 → ③ 月營收、財報、股利。"""
    cfg = config.load()
    errors: list[str] = []
    stages = []
    if daily:
        stages.append(("最新收盤價", lambda: _latest_only(cfg, errors)))
        stages.append(("回補歷史股價與籌碼", lambda: (update_daily(cfg, progress, errors), update_revenue_bulk(cfg, errors))))
    if fundamentals:
        stages.append(("財報與營收", lambda: update_fundamentals(cfg, progress, errors)))
    for i, (name, fn) in enumerate(stages, 1):
        _say(progress, f"第 {i}/{len(stages)} 階段：{name}", 0, 0)
        try:
            fn()
        except Exception as e:
            errors.append(f"{name}中斷：{http.describe(e)}")
            traceback.print_exc()
        _publish(errors, progress, final=i == len(stages))
    return errors


def _latest_only(cfg: dict, errors: list) -> None:
    with db.connect() as con:
        _openapi_fallback(con, cfg, errors)


def _group(errors: list) -> list:
    """同一個來源、同一種錯誤只列一行，附上發生天數，例如「tpex_quotes（29 天）：HTTP 403」。"""
    out, seen = [], {}
    for e in errors:
        key = re.sub(r" \d{4}-\d{2}-\d{2}", "", e)
        if key in seen:
            seen[key][1] += 1
        else:
            seen[key] = [len(out), 1]
            out.append(key)
    return [m if seen[m][1] == 1 else m.replace(":", f"（{seen[m][1]} 次）:", 1) for m in out]


def _publish(errors: list, progress=None, final: bool = False) -> None:
    """寫入更新狀態；有任何真實行情就重算快照（沒有就不產生，避免顯示假的或空的排行榜）。"""
    with db.connect() as con:
        has_prices = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] > 0
        msgs = (["沒有取得任何真實行情，未產生選股結果"] if not has_prices else []) + _group(errors)
        db.set_meta(con, "last_update", config.now().isoformat(timespec="seconds"))
        db.set_meta(con, "source", "live")
        db.set_meta(con, "update_state", "done" if final else "running")
        db.set_meta(con, "last_errors", "\n".join(msgs[:30]))
    if has_prices:
        _say(progress, "重新計算指標", 0, 0)
        from . import snapshot
        snapshot.build()


def main():
    ap = argparse.ArgumentParser(description="更新台股資料庫")
    ap.add_argument("--demo", action="store_true", help="重建示範資料庫")
    ap.add_argument("--daily-only", action="store_true")
    ap.add_argument("--fundamentals-only", action="store_true")
    a = ap.parse_args()
    if a.demo:
        from .sources import demo
        from . import snapshot
        print("建立示範資料庫…", demo.build())
        snapshot.build(demo=True)
        print("完成")
        return
    errs = run(daily=not a.fundamentals_only, fundamentals=not a.daily_only)
    print(f"\n完成，{len(errs)} 個警告")
    for e in errs[:30]:
        print(" -", e)


if __name__ == "__main__":
    main()
