"""V26 AI 盤前情報（Pre-Market Intelligence）：隔夜與週末事件收集、開盤前風險重評、盤後／盤前兩版預測分開封存。

時間都是台灣時間。排程（.github/workflows/update.yml，MODE=premarket）：
- 交易日 06:20 收集隔夜美股、匯率、油價與重大訊息；08:15 再收集一次並封存「盤前版」。
- 交易日 11:30 再檢查一次：盤前版封存後出現的高重要性事件，追加到警示檔（不改盤前版）。
- 週六、週日各收集一次，事件累積到下一個交易日；週一 08:15 一次整合週五收盤後到週一早上的全部資訊。

重要原則
1. 盤前版只更新「風險標示」，不改 AI 分數：還沒有經過樣本外驗證的事件模型之前，不把新聞好壞換算成報酬。
2. 兩個版本都不可覆寫：盤後版在 history/predictions/日期.json（ml_predict），盤前版在 history/predictions/pre_open/日期.json。
3. 不用未來資訊：只收發布時間早於封存時間的事件；09:00 開盤後才封存的版本標成「開盤後才完成」，不列入績效。
4. 休市日不產生盤前版（證交所休市日曆；抓不到時只排除週末，並在網站警告）。
5. 績效只算「發布後買得到的價格」：開盤跳空（前一日收盤 → 當天開盤）另外統計，不算成策略報酬。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config, db
from . import news as N

ROOT = Path(__file__).resolve().parent.parent
PM_DIR = ROOT / "history" / "premarket"
PRE_DIR = ROOT / "history" / "predictions" / "pre_open"
HOLIDAYS = PM_DIR / "holidays.json"
CLOSE = "13:30"               # 台股收盤
SEAL_FROM, PUBLISH_BY, OPEN = "07:45", "08:40", "09:00"
HOLIDAY_URL = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
SEMI = ("半導體業", "電子零組件業", "電腦及週邊設備業", "其他電子業", "光電業", "通信網路業", "電子通路業", "資訊服務業")


# ---------------- 交易日曆 ----------------
def _roc(v) -> str | None:
    s = re.sub(r"\D", "", str(v or ""))
    if len(s) == 7:
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:7]}"
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def parse_holidays(rows: list[dict]) -> list[str]:
    """證交所休市日曆。「開始交易日」「最後交易日」這類列不是休市，要排除。欄位名稱寬鬆比對。"""
    out = []
    for r in rows:
        name = str(r.get("Name") or r.get("名稱") or "")
        if any(w in name for w in ("開始交易", "最後交易", "結算")):
            continue
        d = _roc(r.get("Date") or r.get("日期"))
        if d:
            out.append(d)
    return sorted(set(out))


def holidays(refresh: bool = True) -> dict:
    """{dates, ok, fetched_at}。每 20 天更新一次，失敗沿用快取；完全沒有就只排除週末（ok=False）。"""
    cache = json.loads(HOLIDAYS.read_text("utf-8")) if HOLIDAYS.exists() else None
    fresh = cache and cache.get("fetched_at", "")[:10] >= (config.today() - timedelta(days=20)).isoformat()
    if refresh and not fresh:
        try:
            import requests
            from .sources.twse import HEADERS
            rows = requests.get(HOLIDAY_URL, headers=HEADERS, timeout=30).json()
            ds = parse_holidays(rows)
            if ds:
                cache = {"source": "臺灣證券交易所 OpenAPI 休市日曆", "fetched_at": config.now().isoformat(timespec="seconds"), "dates": ds, "ok": True}
                PM_DIR.mkdir(parents=True, exist_ok=True)
                HOLIDAYS.write_text(json.dumps(cache, ensure_ascii=False, indent=1), "utf-8")
        except Exception as e:
            print("休市日曆取得失敗：", e)
    return cache or {"dates": [], "ok": False, "fetched_at": None, "source": None}


def is_trading_day(d: date, H: dict) -> bool:
    return d.weekday() < 5 and d.isoformat() not in set(H.get("dates", []))


def next_trading_day(d: date, H: dict) -> date:
    x = d + timedelta(days=1)
    while not is_trading_day(x, H):
        x += timedelta(days=1)
    return x


def target_day(now: datetime, H: dict) -> date:
    """這次收集的資訊要用在哪一個交易日：開盤前算今天，開盤後（或休市日）算下一個交易日。"""
    d = now.date()
    if is_trading_day(d, H) and now.strftime("%H:%M") < OPEN:
        return d
    return next_trading_day(d, H)


# ---------------- 事件 ----------------
def _ts(d: str | None, t: str | None) -> str | None:
    if not d:
        return None
    return f"{d}T{(t or '00:00')[:5]}:00"


def importance(it: dict) -> tuple[int, str]:
    if it["kind"] == "announcement":
        return N.classify(it["title"])
    lv, cat = N.classify(it["title"])
    return min(lv, 2), cat  # 媒體新聞最多算中等重要（不是官方資訊）


def collect_events(con, since: str, until: str, known: dict) -> list[dict]:
    """since～until（含）之間發布的公告與新聞。first_seen_at 沿用第一次看到的時間。"""
    con.executescript(N.SCHEMA)
    ind = dict(con.execute("SELECT stock_id, industry FROM stocks").fetchall())
    rows = con.execute("SELECT id, kind, stock_id, date, time, title, source, link, fetched_at FROM news_items WHERE date >= ? AND date <= ?",
                       (since[:10], until[:10])).fetchall()
    now = config.now().isoformat(timespec="seconds")
    out = []
    for i, kind, sid, d, t, title, src, link, fetched in rows:
        pub = _ts(d, t)
        if not pub or pub <= since or pub > until:
            continue
        lv, cat = importance({"kind": kind, "title": title})
        tone = N.tone(title)
        out.append({"id": i, "kind": kind, "title": re.sub(r"\s+", " ", N.clean_title(title, src)).strip(), "source": src or ("公開資訊觀測站" if kind == "announcement" else ""),
                    "link": link, "published_at": pub, "first_seen_at": (known.get(i) or {}).get("first_seen_at") or fetched or now,
                    "fetched_at": now, "data_asof": pub, "affected_tickers": [sid] if sid else [],
                    "affected_industries": [ind[sid]] if ind.get(sid) else [], "event_type": cat, "importance": lv,
                    "direction": {"positive": "正面", "negative": "負面"}.get(tone, "不確定"),
                    "confidence": "官方公告" if kind == "announcement" else "媒體報導"})
    return out


def dedupe(evs: list[dict]) -> int:
    """同一檔股票、標題前 40 字相同的消息視為同一則：保留最早看到的，其他標記 duplicate_of。回傳重複則數。"""
    first, n = {}, 0
    for e in sorted(evs, key=lambda e: (e["first_seen_at"] or "", e["published_at"])):
        k = (tuple(e.get("affected_tickers") or []), N.clean_title(e["title"])[:40])
        if k in first and first[k] != e["id"]:
            e["duplicate_of"] = first[k]
            n += 1
        else:
            first[k] = e["id"]
            e.pop("duplicate_of", None)
    return n


def check_quality(now: datetime, G: dict, H: dict, news_st: dict | None, risk: dict, dups: int, evs: list[dict]) -> dict:
    """盤前情報品質檢查：各來源是否成功、資料日期、去重結果。任何一項失敗，網站就標示「資料不完整」。"""
    ann = (news_st or {}).get("announcements") or {}
    src = {
        "calendar": {"label": "休市日曆", "ok": bool(H.get("ok"))},
        "announcements": {"label": "公司公告", "ok": bool(ann.get("twse") is not None or ann.get("tpex") is not None),
                          "detail": f"上市 {ann.get('twse', '—')} 則、上櫃 {ann.get('tpex', '—')} 則"},
        "news_at": {"label": "公告／新聞最後更新", "ok": bool((news_st or {}).get("at")) and (news_st or {}).get("at", "") >= (now - timedelta(hours=26)).isoformat(),
                    "detail": (news_st or {}).get("at")},
        "global": {"label": "國際市場", "ok": bool(G.get("markets")) and not risk.get("stale"),
                   "detail": "、".join(risk.get("stale") or []) + ("已過期" if risk.get("stale") else "") if G.get("markets") else "沒有資料"},
    }
    ok = sum(x["ok"] for x in src.values())
    return {"checked_at": now.isoformat(timespec="seconds"), "sources": src, "source_ok": f"{ok}/{len(src)}",
            "state": "ok" if ok == len(src) else "degraded", "duplicates_removed": dups,
            "latest_event_at": max((e["published_at"] for e in evs), default=None)}


def _novel(ev: list[dict], earlier_titles: set[str]) -> None:
    for e in ev:
        e["novel"] = N.clean_title(e["title"])[:40] not in earlier_titles


def global_snapshot(path: Path | None = None) -> dict:
    p = path or (config.DATA_DIR / "global_markets.json")
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def expected_us_session(now: datetime) -> str:
    """美股最近一個應該已經收盤的交易日（美東時間收盤＝台灣隔天 04:00～05:00）。美國國定假日不處理，會被標成「可能過期」。"""
    d = (now - timedelta(hours=6)).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def market_risk(G: dict, now: datetime, events: list[dict]) -> dict:
    """隔夜市場風險（規則判讀）。每一條理由都附數字與資料日期；過期的資料不採用。"""
    exp = expected_us_session(now)
    M = {m["symbol"]: m for m in G.get("markets", [])}
    score, why, stale = 0, [], []
    for sym, nm, lo2, lo1, hi in (("^SOX", "費城半導體", -3.0, -1.5, 1.5), ("TSM", "台積電 ADR", -3.0, -1.5, 1.5), ("^IXIC", "那斯達克", -2.0, -1.0, 1.0)):
        m = M.get(sym)
        if not m or m.get("chg_pct") is None:
            continue
        if m["date"] < exp:
            stale.append(f"{nm}只到 {m['date']}")
            continue
        c = m["chg_pct"]
        if c <= lo2:
            score += 2; why.append(f"{nm} {c:+.2f}%（{m['date']}）大跌")
        elif c <= lo1:
            score += 1; why.append(f"{nm} {c:+.2f}%（{m['date']}）下跌")
        elif c >= hi:
            score -= 1; why.append(f"{nm} {c:+.2f}%（{m['date']}）上漲")
    fx = G.get("fx_usd")
    if fx and fx.get("chg") is not None and fx["chg"] >= 0.15:
        score += 1; why.append(f"新台幣貶值 {fx['chg']:+.3f} 元（{fx['date']}）")
    oil = G.get("oil")
    if oil and oil.get("date") and oil["date"] < exp:
        stale.append(f"WTI 原油只到 {oil['date']}")
    elif oil and oil.get("chg_pct") is not None and abs(oil["chg_pct"]) >= 4:
        score += 1; why.append(f"油價{'大漲' if oil['chg_pct'] > 0 else '大跌'} {oil['chg_pct']:+.2f}%（{oil['date']}）")
    bad = [e for e in events if e["importance"] >= 3 and e["direction"] == "負面"]
    if len(bad) >= 3:
        score += 1; why.append(f"收盤後有 {len(bad)} 則高重要性負面公告")
    if not M:
        stale.append("沒有取得國際市場資料")
    level = "高" if score >= 3 else "偏高" if score == 2 else "偏低" if score <= -1 else "中性"
    return {"level": level, "score": score, "reasons": why or ["隔夜市場沒有明顯異常"], "stale": stale, "us_session_expected": exp,
            "method": "規則判讀：費半、台積電 ADR、那斯達克漲跌幅門檻＋匯率、油價、高重要性負面公告數；不是預測"}


def stock_flags(watch: dict[str, dict], events: list[dict], risk: dict, G: dict) -> list[dict]:
    """盤後名單裡的股票，有沒有收盤後的新事件或產業風險。只標示，不改分數。"""
    by = {}
    for e in events:
        for sid in e["affected_tickers"]:
            by.setdefault(sid, []).append(e)
    sox = next((m for m in G.get("markets", []) if m["symbol"] == "^SOX" and m["date"] >= risk["us_session_expected"]), None)
    out = []
    for sid, info in watch.items():
        ev = sorted(by.get(sid, []), key=lambda e: (-e["importance"], e["published_at"]))
        neg = [e for e in ev if e["direction"] == "負面" and e["importance"] >= 2]
        pos = [e for e in ev if e["direction"] == "正面" and e["importance"] >= 2]
        flag, why = None, []
        if neg:
            flag = "風險上升"; why.append(f"收盤後負面{neg[0]['confidence']}：{neg[0]['title'][:40]}")
        elif any(e["importance"] >= 3 for e in ev):
            flag = "重大訊息待確認"; why.append(f"收盤後重大訊息（{ev[0]['event_type']}）：{ev[0]['title'][:40]}")
        elif pos:
            flag = "有新消息"; why.append(f"收盤後正面消息：{pos[0]['title'][:40]}（不調整分數）")
        if sox and sox["chg_pct"] <= -3 and info.get("industry") in SEMI:
            flag = flag or "產業風險"; why.append(f"費半 {sox['chg_pct']:+.2f}%，所屬產業（{info['industry']}）受影響機率較高")
        if flag:
            out.append({"id": sid, "name": info.get("name"), "industry": info.get("industry"), "lists": info.get("lists", []), "flag": flag,
                        "reasons": why, "event_ids": [e["id"] for e in ev[:3]]})
    order = {"風險上升": 0, "重大訊息待確認": 1, "產業風險": 2, "有新消息": 3}
    return sorted(out, key=lambda x: order[x["flag"]])


def _watchlist(snap: dict, n: int = 20) -> dict[str, dict]:
    out = {}
    for s in snap.get("stocks", []):
        for k, m in ((s.get("ai") or {}).get("modes") or {}).items():
            if m.get("rank") and m["rank"] <= n:
                x = out.setdefault(s["stock_id"], {"name": s.get("name"), "industry": s.get("industry"), "lists": []})
                x["lists"].append(f"{(snap.get('ai_modes') or {}).get(k, {}).get('name', k)}第 {m['rank']} 名")
    return out


# ---------------- 主流程 ----------------
def run(now: datetime | None = None, dbname: str = "stocks.db", snap: dict | None = None, G: dict | None = None,
        H: dict | None = None, pre_dir: Path = PRE_DIR, pm_dir: Path = PM_DIR) -> dict:
    now = now or config.now()
    H = H if H is not None else holidays()
    if snap is None:
        p = config.DATA_DIR / "snapshot.json"
        snap = json.loads(p.read_text("utf-8")) if p.exists() else {}
    G = G if G is not None else global_snapshot()
    tgt = target_day(now, H)
    with db.connect(dbname) as con:
        last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        since = f"{last}T{CLOSE}:00"  # 最後一個有行情的交易日收盤之後
        until = now.isoformat(timespec="seconds")
        evp = pm_dir / "events" / f"{tgt}.json"
        known = {e["id"]: e for e in json.loads(evp.read_text("utf-8"))["events"]} if evp.exists() else {}
        events = collect_events(con, since, until, known)
        old = con.execute("SELECT title FROM news_items WHERE date < ? AND date >= ?",
                          (last, (date.fromisoformat(last) - timedelta(days=7)).isoformat())).fetchall() if last else []
        try:
            news_st = json.loads(db.get_meta(con, "news_status") or "null")
        except ValueError:
            news_st = None
    _novel(events, {N.clean_title(t)[:40] for (t,) in old})
    merged = {**known, **{e["id"]: e for e in events}}   # 只增不減：之前看到的事件保留
    allev = sorted(merged.values(), key=lambda e: (-e["importance"], e["published_at"]))
    dups = dedupe(allev)
    evs = [e for e in allev if not e.get("duplicate_of")]   # 同一則消息只算一次（週末重複報導、多個來源）
    evp.parent.mkdir(parents=True, exist_ok=True)
    evp.write_text(json.dumps({"target_date": str(tgt), "since": since, "updated_at": until, "events": allev}, ensure_ascii=False, indent=1), "utf-8")
    risk = market_risk(G, now, evs)
    flags = stock_flags(_watchlist(snap), evs, risk, G)
    quality = check_quality(now, G, H, news_st, risk, dups, evs)
    hm = now.strftime("%H:%M")
    today_trading = is_trading_day(now.date(), H)
    status = {"mode": "collect", "target_date": str(tgt), "since": since, "events": len(evs), "high": sum(e["importance"] >= 3 for e in evs)}
    rec_path = pre_dir / f"{tgt}.json"
    if today_trading and str(tgt) == now.date().isoformat() and SEAL_FROM <= hm < OPEN and last and last < str(tgt):
        if rec_path.exists():
            status["mode"] = "already_sealed"
        else:
            status["mode"] = "sealed"
            seal(rec_path, now, tgt, last, evs, risk, flags, G, snap, sorted(_watchlist(snap)), quality)
    cur_path = pre_dir / f"{now.date()}.json"   # 今天的盤前版（盤中檢查時，目標日已經是下一個交易日）
    if cur_path.exists():  # 封存後才出現的高重要性事件：追加警示（不改盤前版）
        sealed = json.loads(cur_path.read_text("utf-8"))
        tgt_s = sealed["target_date"]
        seen = {e["id"] for e in sealed.get("events", [])}
        new = [e for e in evs if e["id"] not in seen and e["importance"] >= 3 and e["published_at"] > sealed["created_at"]]
        if new:
            ap = pre_dir / f"{tgt_s}.alerts.json"
            cur = json.loads(ap.read_text("utf-8")) if ap.exists() else {"target_date": tgt_s, "alerts": []}
            have = {a["id"] for a in cur["alerts"]}
            cur["alerts"] += [{"id": e["id"], "detected_at": until, "title": e["title"], "tickers": e["affected_tickers"], "event_type": e["event_type"],
                               "direction": e["direction"], "note": "盤前研究封存後才出現的重大事件，原盤前研究可能需要重新評估"} for e in new if e["id"] not in have]
            ap.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
            status["alerts"] = len(cur["alerts"])
    out = {"schema_version": 1, "generated_at": until, "quality": quality,
           "upcoming_holidays": [d for d in H.get("dates", []) if now.date().isoformat() <= d <= (now.date() + timedelta(days=40)).isoformat()], "target_date": str(tgt), "today": now.date().isoformat(), "today_trading": today_trading,
           "calendar_ok": bool(H.get("ok")), "last_close": last, "status": status, "risk": risk, "flags": flags,
           "events": evs[:40], "event_count": len(evs), "global": G}
    return out


def seal(path: Path, now: datetime, tgt: date, last: str, evs: list[dict], risk: dict, flags: list[dict], G: dict, snap: dict,
         watch: list[str] | None = None, quality: dict | None = None) -> dict:
    """封存盤前版（不可覆寫）。分數沿用盤後版，只加風險標示。"""
    base_p = path.parent.parent / f"{last}.json"
    base = json.loads(base_p.read_text("utf-8")) if base_p.exists() else {}
    hm = now.strftime("%H:%M")
    ts = now.isoformat(timespec="seconds")
    rec = {"schema_version": 1, "prediction_type": "pre_open", "signal_date": last, "target_date": str(tgt), "created_at": ts, "published_at": ts,
           "feature_cutoff": f"{ts}（只用這個時間以前發布的事件與已收盤的國際市場）",
           "status": "on_time" if hm < PUBLISH_BY else "late",
           "base": {"file": f"{last}.json" if base else None, "model_version": base.get("model_version") or base.get("model_hash"),
                    "ai_model": (snap.get("model") or {}).get("version")},
           "data_version": {"prices": last, "us_session": max((m["date"] for m in G.get("markets", [])), default=None),
                            "fx_date": (G.get("fx_usd") or {}).get("date"), "events_until": ts},
           "scores_changed": False, "adjustment": "只更新風險標示，AI 分數沿用盤後版（還沒有經過樣本外驗證的事件模型）",
           "risk": risk, "flags": flags, "watchlist": watch or [], "events": evs[:80], "quality": quality}
    if any(e["published_at"] > ts for e in evs):
        raise ValueError("盤前版含有封存時間之後才發布的事件")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # "x"：檔案已存在就失敗，絕不覆寫
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
    return rec


# ---------------- 績效研究（A 盤後／B 加隔夜市場／C 加事件） ----------------
def evaluate(D: dict, pre_dir: Path = PRE_DIR) -> dict:
    """到期的盤前版：
    - 開盤跳空（前一日收盤 → 目標日開盤）只當研究數字，不算策略報酬。
    - 可交易報酬一律從目標日開盤（盤前版發布之後）開始算：當日開盤→收盤、5 日、20 日（減 0050）。
    - B：隔夜風險等級和 0050 跳空、當日表現的關係。C：被標「風險上升」的股票 vs 盤後名單其他股票。"""
    from . import pit_backtest as pb
    import numpy as np
    files = sorted(p for p in pre_dir.glob("????-??-??.json")) if pre_dir.exists() else []
    dates = D["dates"]
    pos = {d: i for i, d in enumerate(dates)}
    lv = {}
    G3 = {k: {"gap": [], "day": [], "5": [], "20": []} for k in ("A", "C", "flagged")}
    used = skipped = 0
    for f in files:
        r = json.loads(f.read_text("utf-8"))
        t, s = r["target_date"], r["signal_date"]
        if r.get("status") not in ("on_time", "late") or t not in pos or s not in pos:
            skipped += 1  # 開盤後才完成的版本不列入（可能用到開盤後的資訊）
            continue
        used += 1
        b = D["adj"].get(pb.BENCH)
        if b is not None and t in b.index and s in b.index:
            x = lv.setdefault(r["risk"]["level"], {"gap": [], "day": [], "n": 0})
            x["n"] += 1
            x["gap"].append(float(b.loc[t, "open"] / b.loc[s, "close"] - 1))
            x["day"].append(float(b.loc[t, "close"] / b.loc[t, "open"] - 1))
        flagged = {x["id"] for x in r["flags"] if x["flag"] == "風險上升"}
        for sid in r.get("watchlist", []):
            a = D["adj"].get(sid)
            if a is None or t not in a.index or s not in a.index:
                continue
            v = {"gap": float(a.loc[t, "open"] / a.loc[s, "close"] - 1), "day": float(a.loc[t, "close"] / a.loc[t, "open"] - 1)}
            for h in ("5", "20"):
                e = pos[t] + int(h)
                if e < len(dates):
                    ret, _ = pb.hold_return(D, sid, t, dates[e])   # 目標日開盤買進：跳空不算進來
                    bm = pb._bench(D, t, dates[e])
                    if ret is not None and bm is not None:
                        v[h] = ret - bm
            for g in ("A",) + (("flagged",) if sid in flagged else ("C",)):
                for k, x in v.items():
                    G3[g][k].append(x)
    avg = lambda xs: round(float(np.mean(xs)) * 100, 2) if xs else None
    return {"generated_at": config.now().isoformat(timespec="seconds"), "files": len(files), "used": used, "skipped": skipped,
            "by_risk_level": {k: {"n": v["n"], "gap_0050": avg(v["gap"]), "open_to_close_0050": avg(v["day"])} for k, v in lv.items()},
            "groups": {k: {"label": {"A": "A：盤後名單全部", "C": "C：排除盤前「風險上升」", "flagged": "被標「風險上升」的股票"}[k],
                           "n": len(v["gap"]), "gap": avg(v["gap"]), "open_to_close": avg(v["day"]), "excess_5d_from_open": avg(v["5"]),
                           "excess_20d_from_open": avg(v["20"]), "n_20d": len(v["20"])} for k, v in G3.items()},
            "note": "跳空只是研究數字，不算策略報酬；可交易報酬都從目標日開盤（盤前版發布後）開始算，減同期間 0050，未扣成本。"
                    "到期天數少於 60 天前，差異多半只是運氣。"}


def site_data() -> dict | None:
    """網站用：最新一次收集＋最近一份盤前版（摘要）＋封存後警示＋對帳。"""
    p = PM_DIR / "latest.json"
    if not p.exists():
        return None
    out = {"latest": json.loads(p.read_text("utf-8"))}
    files = sorted(PRE_DIR.glob("????-??-??.json")) if PRE_DIR.exists() else []
    if files:
        r = json.loads(files[-1].read_text("utf-8"))
        out["sealed"] = {k: r.get(k) for k in ("target_date", "signal_date", "created_at", "status", "scores_changed", "adjustment", "risk", "flags", "base", "quality")}
        out["sealed"]["events"] = len(r.get("events", []))
        ap = PRE_DIR / f"{r['target_date']}.alerts.json"
        out["alerts"] = json.loads(ap.read_text("utf-8"))["alerts"] if ap.exists() else []
    sc = PM_DIR / "scorecard.json"
    out["scorecard"] = json.loads(sc.read_text("utf-8")) if sc.exists() else None
    out["schedule"] = {"premarket": ["06:20", "08:15"], "intraday": "11:30", "weekend": ["週六 10:00", "週日 20:00"], "publish_by": PUBLISH_BY}
    return out


def main(argv: list[str]) -> int:
    if "--score" in argv:
        from . import pit_backtest as pb
        res = evaluate(pb.load())
        PM_DIR.mkdir(parents=True, exist_ok=True)
        (PM_DIR / "scorecard.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        print("盤前情報對帳：", json.dumps({k: res[k] for k in ("files", "used")}, ensure_ascii=False))
        return 0
    out = run()
    PM_DIR.mkdir(parents=True, exist_ok=True)
    (PM_DIR / "latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    st = out["status"]
    print(f"盤前情報：目標交易日 {out['target_date']}，{st['mode']}，事件 {st['events']} 則（高重要性 {st['high']}），隔夜風險 {out['risk']['level']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
