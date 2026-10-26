"""資料更新：把各來源的最新資料補進資料庫，再重算選股快照。

用法：
    python -m app.update            # 增量更新（第一次會回補約一年半股價，約需 40~60 分鐘）
    python -m app.update --demo     # 重建示範資料庫
    python -m app.update --daily-only / --fundamentals-only / --backfill
"""
from __future__ import annotations

import json
import os
import argparse
import re
import traceback
from datetime import date, datetime, timedelta

import requests

from . import config, db, events, fin_watch
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


def _span(cfg: dict, last: date) -> int:
    """要回補幾個日曆天：price_backfill_days 與 history_start（研究用的長期歷史起點）取較長者。"""
    n = cfg["price_backfill_days"]
    if cfg.get("history_start") and n > FAST_DAYS:  # 快速更新只看最近幾天
        n = max(n, (last - date.fromisoformat(cfg["history_start"])).days + 1)
    return n


def update_history(cfg: dict, progress=None, errors: list | None = None) -> None:
    """夜間補齊：只補較舊的歷史收盤行情（不含法人、融資等籌碼，那些只保留最近幾十天），
    每次最多 history_dates_backfill 個交易日，分幾個晚上把股價往前補到 history_start。"""
    errors = errors if errors is not None else []
    last = config.today() - timedelta(days=cfg.get("recent_backfill_days", 420))
    days = [d for d in (last - timedelta(days=i) for i in range(_span(cfg, last))) if d.weekday() < 5]
    srcs = [TwseSource(cfg["twse_interval_sec"])] + ([TpexSource(cfg["twse_interval_sec"])] if "tpex" in cfg["markets"] else [])
    cap = cfg.get("history_dates_backfill", 120)
    with db.connect() as con:
        todo = [(s, d, ("quotes", "daily_quotes", None)) for s in srcs
                for d in [d for d in days if _need(con, f"{s.market}_quotes", d)][:cap]]
        _fetch(con, todo, progress, errors, "歷史股價回補")
        _unmark_false_holidays(con, [d.isoformat() for _, d, _ in todo], errors)
        db.set_meta(con, "history_backfill", json.dumps(_history_state(con, cfg), ensure_ascii=False))
        con.commit()


def _unmark_false_holidays(con, dates: list[str], errors: list) -> None:
    """抓到空資料的平日會被記成「休市」以後不再重抓。回補很舊的日期時要防止端點其實不支援而誤判：
    ① 上市那天有行情、上櫃卻是空的 → 上櫃不算休市，下次再試；
    ② 上市連續 12 個平日以上都是空的（農曆年最多約 7 個）→ 視為端點不支援，全部取消標記。"""
    dates = sorted(set(dates))
    if not dates:
        return
    q = ",".join("?" * len(dates))
    have = {r[0] for r in con.execute(f"SELECT DISTINCT date FROM prices WHERE date IN ({q})", dates)}
    empty = {r[0] for r in con.execute(f"SELECT key FROM fetch_log WHERE source='twse_quotes' AND ok=0 AND key IN ({q})", dates)}
    bad = []
    run, longest = [], []
    for d in dates:
        if d in empty:
            run.append(d)
            longest = run if len(run) > len(longest) else longest
        elif d in have:
            run = []
    if len(longest) >= 12:
        bad = [d for d in dates if d in empty]
        errors.append(f"twse_quotes 歷史回補：{longest[0]}～{longest[-1]} 連續 {len(longest)} 個平日沒有資料，可能端點不支援這麼舊的日期，下次再試")
    tp = [r[0] for r in con.execute(f"SELECT key FROM fetch_log WHERE source='tpex_quotes' AND ok=0 AND key IN ({q})", dates)]
    tp = [d for d in tp if d in have]
    if tp:
        errors.append(f"tpex_quotes 歷史回補：{len(tp)} 天上市有行情但上櫃沒有資料（{tp[0]}～{tp[-1]}），下次再試")
    for src, ks in (("twse_quotes", bad), ("tpex_quotes", tp)):
        for k in ks:
            con.execute("DELETE FROM fetch_log WHERE source=? AND key=?", (src, k))
    con.commit()


def _history_state(con, cfg: dict) -> dict:
    """歷史股價補到哪裡：網站「資料盤點」會顯示。"""
    first = con.execute("SELECT MIN(date) FROM prices").fetchone()[0]
    left = {}
    last = config.today() - timedelta(days=cfg.get("recent_backfill_days", 420))
    days = [d for d in (last - timedelta(days=i) for i in range(_span(cfg, last))) if d.weekday() < 5]
    for m in cfg["markets"]:
        done = {r[0] for r in con.execute("SELECT key FROM fetch_log WHERE source=?", (f"{m}_quotes",))}
        left[m] = sum(1 for d in days if d.isoformat() not in done)
    return {"target": cfg.get("history_start"), "first_date": first, "left_days": left,
            "at": config.now().isoformat(timespec="seconds")}


def update_daily(cfg: dict, progress=None, errors: list | None = None) -> None:
    errors = errors if errors is not None else []
    srcs = []
    if "twse" in cfg["markets"]:
        srcs.append(TwseSource(cfg["twse_interval_sec"]))
    if "tpex" in cfg["markets"]:
        srcs.append(TpexSource(cfg["twse_interval_sec"]))
    now = config.now()
    last = config.today() if now.hour >= 14 else config.today() - timedelta(days=1)
    days = [last - timedelta(days=i) for i in range(_span(cfg, last))]
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
        # 較舊的歷史股價（超過約一年半）每次只補一部分，分幾天慢慢補齊，避免單次執行超過時間限制
        old_cut = last - timedelta(days=cfg.get("recent_backfill_days", 420))
        cap = cfg.get("history_dates_per_run", 60)
        left = {s.market: cap for s in srcs}
        keep = []
        for t in todo:
            if t[1] < old_cut:
                if left[t[0].market] <= 0:
                    continue
                left[t[0].market] -= 1
            keep.append(t)
        todo = keep
        _fetch(con, todo, progress, errors)
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


def _fetch(con, todo: list, progress, errors: list, label: str = "每日行情與籌碼") -> None:
    dead = set()  # 解析失敗的 (市場, 資料) 本次不再重試
    for n, (s, d, (key, fn, table)) in enumerate(todo, 1):
        tag = f"{s.market}_{key}"
        if tag in dead:
            continue
        _say(progress, label, n, len(todo), f"{s.market} {key} {d}")
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
            if getattr(s, "indices", None):
                db.upsert(con, "sector_index", [{"date": ds, "name": k, "close": v} for k, v in s.indices.items()])
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


def _openapi_fallback(con, cfg: dict, errors: list) -> None:
    """用官方 OpenAPI 補上最近一個交易日（只有一天，不能回補歷史）。

    股價與本益比分開判斷：股價那天已經有了，也照樣同步本益比；
    當天只存了部分股票時（例如主要來源中途失敗），用 OpenAPI 的完整清單補齊。
    """
    from .sources import tpex as P
    from .sources import twse as T
    plan = [("twse", T.openapi_quotes, T.openapi_valuation)]
    if "tpex" in cfg["markets"]:
        plan.append(("tpex", P.openapi_quotes, None))
    for mkt, quotes, valuation in plan:
        try:
            day, rows = quotes()
        except Exception as e:
            errors.append(f"{mkt} OpenAPI 行情: {http.describe(e)}")
            day, rows = None, []
        if day:  # 官方實際公布到哪一天：用來判斷網站資料是否過期（自動涵蓋國定假日與颱風假）
            db.set_meta(con, f"official_latest_{mkt}", day)
        rows = [r for r in rows if is_tracked(r["stock_id"])]
        if day and rows:
            _sync_quotes(con, mkt, day, rows, errors)
        if mkt == "tpex":
            try:  # 櫃買指數：只有最近一日，欄位未驗證；失敗只記在 meta，不算更新錯誤
                iday, val, keys = P.openapi_otc_index()
                db.set_meta(con, "otc_index_probe", f"{iday or '-'} {val or '-'} 欄位：{','.join(keys)[:300]}")
                if iday and val:
                    db.upsert(con, "sector_index", [{"date": iday, "name": "櫃買指數", "close": val}])
            except Exception as e:
                db.set_meta(con, "otc_index_probe", f"失敗：{http.describe(e)}"[:300])
        if valuation:
            try:
                vday, vrows = valuation()
                vday = vday or day
                vrows = [r for r in vrows if is_tracked(r["stock_id"])]
                have = con.execute("SELECT COUNT(*) FROM valuation v JOIN stocks s USING(stock_id) WHERE s.market=? AND v.date=?",
                                   (mkt, vday)).fetchone()[0] if vday else 0
                if vday and len(vrows) > have:
                    db.upsert(con, "valuation", [{**r, "date": vday} for r in vrows])
                    db.log_fetch(con, f"{mkt}_val", vday, True)
            except Exception as e:
                errors.append(f"{mkt} OpenAPI 本益比: {http.describe(e)}")
        con.commit()


def _prev_count(con, mkt: str, day: str) -> int:
    """前一個交易日這個市場有幾檔行情，用來檢查今天是否完整。"""
    r = con.execute("SELECT COUNT(*) FROM prices p JOIN stocks s USING(stock_id) WHERE s.market=? AND p.date="
                    "(SELECT MAX(p2.date) FROM prices p2 JOIN stocks s2 USING(stock_id) WHERE s2.market=? AND p2.date<?)",
                    (mkt, mkt, day)).fetchone()
    return r[0] if r else 0


def _sync_quotes(con, mkt: str, day: str, rows: list[dict], errors: list) -> None:
    have = con.execute("SELECT MAX(p.date) FROM prices p JOIN stocks s USING(stock_id) WHERE s.market=?", (mkt,)).fetchone()[0]
    if have and day < have:
        return  # OpenAPI 比資料庫舊（例如快取），不覆蓋
    n_have = con.execute("SELECT COUNT(*) FROM prices p JOIN stocks s USING(stock_id) WHERE s.market=? AND p.date=?",
                         (mkt, day)).fetchone()[0]
    if n_have >= len(rows):
        return  # 這一天已經完整
    for r in rows:
        con.execute("INSERT OR IGNORE INTO stocks (stock_id, name, market, industry) VALUES (?,?,?,'')",
                    (r["stock_id"], r.pop("name"), mkt))
        r["date"] = day
    if n_have:  # 主要來源已存的股票不覆蓋，只補缺的
        got = {x[0] for x in con.execute("SELECT stock_id FROM prices WHERE date=?", (day,))}
        rows = [r for r in rows if r["stock_id"] not in got]
    db.upsert(con, "prices", rows)
    total = n_have + len(rows)
    prev = _prev_count(con, mkt, day)
    if prev and total < prev * 0.9:  # 比前一天少一成以上，視為不完整：保留資料但不標記成功，下次會再補
        errors.append(f"{mkt} OpenAPI 行情 {day}: 只有 {total} 檔，前一個交易日有 {prev} 檔，資料可能不完整")
        return
    db.log_fetch(con, f"{mkt}_quotes", day, True)
    db.log_fetch(con, f"{mkt}_openapi", day, True)


def update_exrights(cfg: dict, errors: list) -> None:
    """除權息參考價：用來把歷史股價還原，讓均線、報酬率、波動率不會因為除息缺口失真。
    每個月抓一次；本月與上個月每次都重抓（可能有新公告）。"""
    srcs = [TwseSource(cfg["twse_interval_sec"])] + ([TpexSource(cfg["twse_interval_sec"])] if "tpex" in cfg["markets"] else [])
    today = config.today()
    first = (today - timedelta(days=_span(cfg, today) + 31)).replace(day=1)
    months = []
    m = first
    while m <= today:
        nxt = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
        months.append((m, min(nxt - timedelta(days=1), today)))
        m = nxt
    with db.connect() as con:
        for s in srcs:
            tag = f"{s.market}_exrights"
            for a, b in months:
                key = a.strftime("%Y-%m")
                recent = (today - a).days < 62
                if not recent and con.execute("SELECT 1 FROM fetch_log WHERE source=? AND key=? AND ok=1", (tag, key)).fetchone():
                    continue
                try:
                    rows = [r for r in s.ex_rights(a, b) if is_tracked(r["stock_id"])]
                except Exception as e:
                    errors.append(f"{tag}: {http.describe(e)}")
                    break  # 這個市場的端點不能用，本次不再試其他月份
                db.upsert(con, "exrights", rows)
                db.log_fetch(con, tag, key, True)
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


def _fin_requests() -> tuple[list[str], object]:
    """會員在個股頁按「優先補抓財報」的請求（Supabase fin_requests 表）。只在 Actions 裡用 secret key 讀，讀不到就略過。"""
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        return [], None
    try:
        from .alerts import Supa
        supa = Supa(url, key)
        r = supa.s.get(supa.url + "fin_requests", params={"select": "stock_id,requested_at", "order": "requested_at", "limit": 2000}, timeout=30)
        if r.status_code >= 400:
            return [], None
        return list(dict.fromkeys(x["stock_id"] for x in r.json())), supa
    except Exception:
        return [], None


def _clear_requests(supa, sids: list[str]) -> None:
    if supa and sids:
        try:
            supa.s.delete(supa.url + "fin_requests", params={"stock_id": f"in.({','.join(sids)})"}, timeout=30)
        except Exception:
            pass


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
        common = set(ids)
        req, supa = _fin_requests()
        req = [s for s in req if s in common]  # 只接受普通股代號；會員請求的股票排最前面
        ids = list(dict.fromkeys(req + cfg.get("watchlist", []) + ids))
        # 財報一律從 fin_history_start 抓起（現金流量表要靠前一季換算單季，窗口短會把最早一季覆蓋成空值）；
        # 月營收定期重抓只抓近幾年，第一次抓或長期歷史補齊時才從 fin_history_start 抓起
        fs = cfg.get("fin_history_start")
        rev_long = (lambda sid: fm.month_revenue(sid, start=fs)) if fs else fm.month_revenue
        fin = (lambda sid: fm.financials(sid, start=fs)) if fs else fm.financials
        plan = [("fm_revenue", 1 if config.today().day <= 12 else 7, fm.month_revenue, "revenue"),
                ("fm_financials", 20, fin, "financials"),
                ("fm_dividends", 45, fm.dividends, "dividends")]
        deep = {"fm_revenue": ("fm_revenue_hist", rev_long), "fm_financials": ("fm_financials_hist", fin)} if fs else {}
        # 補齊佇列：從來沒抓過的股票優先（依成交金額排序），再來才是定期重抓；額度用完就停，下次從還沒補到的繼續
        # 第三優先：以前只抓了近幾年的股票，把財報與月營收往前補到 fin_history_start（長期回測與模型研究用），每檔只補一次
        # V19：官方已公告、資料庫還沒有的新一季財報最優先（正式推薦與候選 → 會員申請與追蹤清單 → 成交金額）
        fin_watch.run_detect(errors, con)  # 用同一條連線，避免和上面還沒提交的寫入互鎖
        hot = fin_watch.hot_list(con, req + cfg.get("watchlist", []) + ids)
        hot_ids = {sid for sid, _ in hot}
        new, due, hist = [], [], []
        for sid in ids:
            for src, age, fn, table in plan:
                t = db.fetched_at(con, src, sid)
                if src == "fm_financials":
                    if sid in hot_ids:
                        continue  # 已排進新財報佇列
                    age = fin_watch.fin_age(con, sid, config.today(), age)  # 財報季內缺最新季：5 天（偵測失敗時的保底）
                if not t:
                    new.append((sid, src, deep[src][1] if src in deep else fn, table))
                elif config.now() - t > timedelta(days=age):
                    due.append((sid, src, fn, table))
                    if fn is deep.get(src, (0, None))[1]:
                        continue  # 定期重抓本來就從長期起點抓起
                if t and src in deep and not db.fetched_at(con, deep[src][0], sid):
                    hist.append((sid, deep[src][0], deep[src][1], table))
        new.sort(key=lambda x: x[1] == "fm_dividends")  # 先補財報與營收（研究卡需要），股利最後；同類依原順序（會員申請→成交金額）
        todo = [(sid, "fm_financials", fin, "financials") for sid, _ in hot] + new + due + hist
        hot_period = dict(hot)
        # 補齊進度（網站「資料來源與方法」和個股研究卡會顯示）：這次要補多少、補了多少、失敗原因、是否因額度用完而停下
        Q = {"run_at": config.now().isoformat(timespec="seconds"), "stocks": len(ids), "hot": len(hot), "new": len(new), "due": len(due), "hist": len(hist),
             "done": 0, "failed": 0, "quota_hit": False, "errors": [], "requested": len(req)}
        db.set_meta(con, "fundamental_queue", json.dumps(Q, ensure_ascii=False))
        for n, (sid, src, fn, table) in enumerate(todo, 1):
            _say(progress, "財報與營收", n, len(todo), f"{sid} {src[3:]}")
            try:
                data = fn(sid)
                if table == "financials":
                    data = fin_watch.guard(con, sid, data)  # 空白欄位不蓋掉舊數字；和官方不一致的舊版數字不寫入
                    fin_watch.compare(con, sid, data)  # 更正紀錄、每季第一次取得時間
                db.upsert(con, table, data)
                db.log_fetch(con, src, sid)
                if table == "financials" and sid in hot_period:
                    Q["hot_synced"] = Q.get("hot_synced", 0) + fin_watch.after_fetch(con, sid, hot_period.pop(sid))
                if src in deep and fn is deep[src][1]:  # 第一次就從長期起點抓起，不必再補
                    db.log_fetch(con, deep[src][0], sid)
                con.commit()
                Q["done"] += 1
            except FinMindError as e:
                errors.append(str(e))
                Q["failed"] += 1
                Q["errors"] = (Q["errors"] + [str(e)[:120]])[-5:]
                if "用完" in str(e):
                    Q["quota_hit"] = True
                    break  # 額度用完，下次排程再繼續
            except Exception as e:
                errors.append(f"{sid} {src}: {http.describe(e)}")
                Q["failed"] += 1
                Q["errors"] = (Q["errors"] + [f"{sid} {src[3:]}：{http.describe(e)}"[:120]])[-5:]
        # 補完這次之後，還有幾檔普通股從來沒有財報（下一次完整更新優先補）
        pending = [sid for sid in ids if not db.fetched_at(con, "fm_financials", sid)]
        Q["never_fin"] = len(pending)
        Q["hist_left"] = sum(1 for sid in ids if deep and not db.fetched_at(con, "fm_financials_hist", sid))  # 財報還沒往前補到長期起點的股票
        Q["pending"] = pending  # 還沒補到財報的股票，依補抓順序（個股頁顯示「第幾位」）
        has_fin = {r[0] for r in con.execute("SELECT DISTINCT stock_id FROM financials")}
        Q["no_data"] = [sid for sid in ids if db.fetched_at(con, "fm_financials", sid) and sid not in has_fin][:500]  # 抓過但資料來源沒有財報
        done_req = [sid for sid in req if db.fetched_at(con, "fm_financials", sid)]
        Q["requested_done"] = done_req
        _clear_requests(supa, done_req)
        Q["finished_at"] = config.now().isoformat(timespec="seconds")
        db.set_meta(con, "fundamental_queue", json.dumps(Q, ensure_ascii=False))
        now = config.now().isoformat(timespec="seconds")
        for k in ("revenue", "financials", "dividends"):
            db.set_meta(con, f"updated_{k}", now)


# 快速收盤更新只看最近幾天：今天的行情不必等三年前的歷史資料回補完
FAST_DAYS = 10


def run(daily: bool = True, fundamentals: bool = True, progress=None, fast: bool = False, history: bool = False) -> list[str]:
    """分三階段，每階段結束都重算快照，讓網站盡早顯示真實資料：
    ① 最新一個交易日收盤價（OpenAPI，幾秒鐘）→ ② 回補歷史股價與籌碼 → ③ 月營收、財報、股利。
    fast=True（盤後快速更新）：只抓最近 FAST_DAYS 天的行情與籌碼、除權息、月營收，重算一次就發布；
    較舊的歷史股價、財報與股利留給晚上的完整更新。
    history=True（夜間補齊）：先把更舊的歷史收盤行情往前補一段，再補財報。"""
    cfg = config.load()
    errors: list[str] = []
    stages = []
    if fast:
        near = {**cfg, "price_backfill_days": FAST_DAYS, "flow_backfill_days": FAST_DAYS, "history_dates_per_run": 0}
        stages.append(("快速收盤更新", lambda: (_latest_only(cfg, errors), fin_watch.run_detect(errors), update_daily(near, progress, errors), update_exrights(near, errors),
                                                update_revenue_bulk(cfg, errors), events.update(cfg, errors))))
    elif daily:
        stages.append(("最新收盤價", lambda: _latest_only(cfg, errors)))
        stages.append(("回補歷史股價與籌碼", lambda: (update_daily(cfg, progress, errors), update_exrights(cfg, errors),
                                                          update_revenue_bulk(cfg, errors), events.update(cfg, errors))))
    if history:
        stages.append(("回補長期歷史股價", lambda: (update_history(cfg, progress, errors), update_exrights(cfg, errors))))
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
        from . import snapshot, track
        snap = snapshot.build()
        if final and (saved := track.record(snap)):
            _say(progress, f"已保存今日選股紀錄 {saved}", 0, 0)


def main():
    ap = argparse.ArgumentParser(description="更新台股資料庫")
    ap.add_argument("--demo", action="store_true", help="重建示範資料庫")
    ap.add_argument("--daily-only", action="store_true")
    ap.add_argument("--fundamentals-only", action="store_true")
    ap.add_argument("--fast", action="store_true", help="盤後快速更新：只抓最近幾天，不回補歷史、不抓財報")
    ap.add_argument("--backfill", action="store_true", help="夜間補齊：往前回補長期歷史股價，再補財報")
    a = ap.parse_args()
    if a.backfill:
        a.fundamentals_only = True
    if a.demo:
        from .sources import demo
        from . import snapshot
        print("建立示範資料庫…", demo.build())
        snapshot.build(demo=True)
        print("完成")
        return
    with db.connect() as con:
        db.set_meta(con, "update_mode", "fast" if a.fast else "backfill" if a.fundamentals_only else "full")
    errs = run(daily=not a.fundamentals_only, fundamentals=not a.daily_only and not a.fast, fast=a.fast, history=a.backfill)
    print(f"\n完成，{len(errs)} 個警告")
    for e in errs[:30]:
        print(" -", e)


if __name__ == "__main__":
    main()
