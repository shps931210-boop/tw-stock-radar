"""每日 AI 量化選股紀錄、入選後股價追蹤、選股績效驗證。

紀錄（history/picks/）
- 日期.json：每個資料日第一次完整更新後的選股結果（首次發布版）。
- 日期.v2.json、v3…：同一天稍後的更新讓名單改變時另存「補正版」（例如 21:00 補上法人資料後）。
  所有檔案都不覆寫，GitHub Actions 會 commit 進 repository。績效驗證一律用首次發布版，避免事後挑選。

績效驗證的計算方式（固定規則，所有紀錄一致）
- 進場：選股日的「下一個交易日開盤價」（名單在收盤後才產生，當天收盤價買不到）。
- 出場：選股日之後第 5、20、60 個交易日的收盤價（進場日算第 1 天，所以「5 天」= 進場日開盤到第 5 個交易日收盤）；
  還沒到期的期間不計入。回測（app/backtest.py）用同一個定義。
- 扣交易成本 0.585%（手續費 0.1425% × 2 ＋ 證交稅 0.3%），用還原除權息股價。
- 基準：元大台灣50（0050，還原除權息，含股利）在「同一段期間」的報酬：進場日開盤 → 出場日收盤，
  和個股完全相同。基準固定是 0050，不會換成別的指數：進場日或出場日缺 0050 價格時，該筆標示「無法比較」，
  只列報酬、不算超額報酬，也不併入績效驗證的超額報酬平均（另外列出筆數）。基準不扣交易成本，對 AI 量化選股比較嚴格。
- 正式樣本外：每筆紀錄都存模型設定指紋（model_hash＝模型程式＋選股設定的雜湊），而且名單要在進場日開盤（09:00）
  之前就產生，才算進績效驗證；事後才產生的名單（例如補跑舊日期）另外標示「名單產生太晚」，不計入。
- 可交易性：進場日沒有成交（停牌、下市）、漲停鎖死（開高低同價且漲幅 ≥ 9.5%）、
  進場日成交金額太少的股票，實際上買不到，不計入，並分類列出檔數與比例。
- 已買進但出場日沒有成交的股票仍然計入，用出場日以前最後一個成交價估算，避免只留下活著的股票；
  分成「之後已恢復交易（停牌）」和「之後都沒有成交（停牌中或下市）」兩類，並另外列出只算「有實際出場價」股票的結果，
  因為最後成交價不一定賣得到。進場日或持有期間沒有任何有效價格的股票（資料異常）不計入並列出。
- 滑價敏感度：另外列出每筆交易再多 0.2%、0.5% 成本時的結果。
- 全市場對照：當天所有可排名股票等權重持有的表現，用來比較「前 20 名」是否真的比整體好。
- 等權重組合的最大回撤：持有期間每天的組合淨值相對前高的最大跌幅。
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from pathlib import Path

from . import config, db, metrics

HISTORY = config.ROOT / "history" / "picks"
TOP_N = 20
COST = 0.001425 * 2 + 0.003
HORIZONS = (1, 5, 20, 60)
FACTORS = ("quality", "growth", "value", "momentum", "trend")
BENCH_ID = "0050"
LIMIT_UP = 0.095
THIN_VALUE = 3e6       # 進場日成交金額低於 300 萬元，視為買不到
SLIPS = (0.002, 0.005)
SKIP_LABELS = {"not_traded": "進場日沒有成交（停牌、下市）", "limit_up": "漲停鎖死買不到", "thin": "成交量太少",
               "no_price": "沒有有效價格（資料異常）"}
NO_BENCH = "unavailable"   # 缺 0050 價格：無法比較
OPEN_AT = "09:00:00"      # 名單要在進場日開盤前產生才算數
STALE_LABELS = {"resumed": "出場日停牌、之後已恢復交易", "no_later": "出場日後都沒有成交（停牌中或下市）"}


def benchmark(con, first: str | None = None) -> tuple[pd.DataFrame, str]:
    """基準的開盤、收盤價（還原除權息）。有 0050 用 0050，否則用加權指數收盤價。回傳 (價格, 種類)。"""
    q = "SELECT * FROM prices WHERE stock_id=?" + (" AND date >= ?" if first else "") + " ORDER BY date"
    px = db.read(con, q, (BENCH_ID, first) if first else (BENCH_ID,))
    if len(px) >= 2:
        ex = db.read(con, "SELECT * FROM exrights WHERE stock_id=?", (BENCH_ID,))
        adj, _ = metrics.adjust_prices(px.reset_index(drop=True), ex)
        b = adj.set_index("date")[["open", "close"]].astype(float)
        b["open"] = b["open"].where(b["open"] > 0, b["close"])
        raw = px.set_index("date")[["open", "close"]].astype(float)  # 未還原：價格報酬（不含配息）
        b["open_raw"] = raw["open"].where(raw["open"] > 0, raw["close"])
        b["close_raw"] = raw["close"]
        return b, BENCH_ID
    return pd.DataFrame(columns=["open", "close", "open_raw", "close_raw"], dtype=float), "taiex"


def model_hash() -> str:
    """模型設定指紋：AI 模型程式邏輯 + 選股相關設定（12 碼）。
    只看程式結構：註解和文字說明（字串）改了不會改變指紋，避免修改用語就把績效紀錄切成兩段；計算邏輯、數字、設定改了才會變。"""
    from . import ai_model
    cfg = config.load()
    keys = {k: cfg.get(k) for k in ("composite_weights", "base_filter", "buy_point")}
    src = _logic(Path(ai_model.__file__).read_text("utf-8")).encode() + json.dumps(keys, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(src).hexdigest()[:12]


def _logic(code: str) -> str:
    """把程式裡的文字字串換成空字串後的語法樹（字典的鍵、比較用的代碼也是字串，所以保留短的英數字串）。"""
    import ast
    import re as _re
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and not _re.fullmatch(r"[A-Za-z0-9_.\-]{0,40}", n.value):
            n.value = ""
    return ast.dump(tree, include_attributes=False)


def on_time(log: dict, entry: str | None) -> bool | None:
    """名單是否在進場日開盤前就產生（才可能真的照著買）。沒有產生時間的舊紀錄回傳 None。"""
    g = log.get("generated_at")
    if not g or not entry:
        return None
    return g[:19] < f"{entry}T{OPEN_AT}"


def _snapshot_picks(snap: dict) -> tuple[dict, list]:
    st = [s for s in snap["stocks"] if s.get("is_common")]
    picks = {}
    for mode in snap.get("ai_modes", {}):
        ranked = sorted((s for s in st if s["ai"]["modes"][mode]["rank"]), key=lambda s: s["ai"]["modes"][mode]["rank"])[:TOP_N]
        picks[mode] = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"][mode]["rank"],
                        "score": s["ai"]["modes"][mode]["score"], "close": s.get("close"), "industry": s.get("industry"),
                        "factors": {f: s.get(f"score_{f}") for f in FACTORS}} for s in ranked]
    for mode in picks:   # V27：附上流動性等級與推薦資格，方便日後比較原始名單和預設精選名單
        for p in picks[mode]:
            s = next(x for x in st if x["stock_id"] == p["stock_id"])
            rec = ((s["ai"].get("rec") or {}).get(mode) or {})
            p["liq"] = s.get("liquidity_grade")
            p["rec"] = rec.get("recommendation_eligibility")
    sigs = [{"stock_id": s["stock_id"], "name": s["name"], "close": s.get("close"), "triggers": s["timing"]["triggers"],
             "score": s["ai"]["modes"]["balanced"]["score"]} for s in st if s.get("timing", {}).get("status") == "buy"]
    return picks, sigs


def eligible(s: dict, mode: str) -> bool:
    """正式推薦資格（V13 再加一道保險）：要有推薦名次、資格是 normal／capped、流動性不是 E／F／未知、資料完整、不是處置股。
    不合格的股票只留在研究用的原始排名，正式名單不足 TOP_N 檔就照實際數量，不拿不合格的補。"""
    rec = ((s.get("ai") or {}).get("rec") or {}).get(mode) or {}
    return (bool(rec.get("rank")) and rec.get("recommendation_eligibility") in ("normal", "capped")
            and s.get("liquidity_grade") not in ("E", "F", "?") and s.get("complete", True) is not False and not s.get("disposition"))


def _rec_picks(snap: dict) -> dict:
    """V27 預設 AI 精選名單（流動性限制後）。和原始名單分開記錄，原始名單與績效計算不受影響。"""
    st = [s for s in snap["stocks"] if s.get("is_common")]
    rec_picks = {}
    for mode in snap.get("ai_modes", {}):
        ranked = sorted((s for s in st if eligible(s, mode)), key=lambda s: s["ai"]["rec"][mode]["rank"])[:TOP_N]
        rec_picks[mode] = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["rec"][mode]["rank"], "raw_rank": s["ai"]["modes"][mode]["rank"],
                            "score": s["ai"]["modes"][mode]["score"], "liq": s.get("liquidity_grade"), "close": s.get("close"), "industry": s.get("industry"),
                            "factors": {f: s.get(f"score_{f}") for f in FACTORS},
                            "rec": s["ai"]["rec"][mode]["recommendation_eligibility"]} for s in ranked]
    return rec_picks


def _versions(day: str) -> list:
    return sorted(HISTORY.glob(f"{day}*.json"), key=lambda p: int((re.search(r"\.v(\d+)\.json$", p.name) or [0, 1])[1]))


def record(snap: dict) -> str | None:
    """寫入當天的選股紀錄。第一次寫首次發布版；之後名單有變才另存補正版。不能用的資料回傳 None。"""
    if snap.get("source") != "live" or snap.get("rankings_blocked"):
        return None
    day = (snap.get("data_dates") or {}).get("prices")
    if not day:
        return None
    from .ai_model import MODEL_VERSION
    picks, sigs = _snapshot_picks(snap)
    universe = sorted(s["stock_id"] for s in snap["stocks"] if s.get("is_common") and s["ai"]["modes"]["balanced"]["rank"])
    out = {"date": day, "model_version": MODEL_VERSION, "model_hash": model_hash(), "generated_at": snap.get("generated_at"),
           "data_dates": snap.get("data_dates"), "taiex": (snap.get("market") or {}).get("taiex"),
           "universe": snap.get("universe"), "picks": picks, "signals": sigs, "universe_ids": universe}
    if any(s.get("liquidity_grade") for s in snap["stocks"]):
        out["rec_picks"] = _rec_picks(snap)
        out["lists_note"] = {"picks": "原始模型排名（研究與績效追蹤用，包含流動性不合格、不列入正式精選的股票，見各股 rec 欄位）",
                             "rec_picks": "預設 AI 精選＝正式推薦名單（排除流動性 E／F 級與資料不足，D 級不進前 10 名）"}
    old = _versions(day)
    if old:
        reasons = diff_reasons(json.loads(old[-1].read_text("utf-8")), out)
        if not reasons:
            return None  # 名單、分數、因子、模型、資料日期都沒變，不用另存
        path = HISTORY / f"{day}.v{len(old) + 1}.json"
        out["reasons"] = reasons
    else:
        path = HISTORY / f"{day}.json"
    out = {"version": len(old) + 1, "kind": "首次發布" if not old else "補正版", **out}
    HISTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    return str(path)


def _sig(log: dict) -> dict:
    r1 = lambda x: None if x is None else round(float(x), 1)
    return {
        "picks": {m: [(p["stock_id"], p.get("rank"), r1(p.get("score"))) for p in g] for m, g in log.get("picks", {}).items()},
        "factors": {m: [tuple(r1((p.get("factors") or {}).get(f)) for f in FACTORS) for p in g] for m, g in log.get("picks", {}).items()},
        "signals": [(x["stock_id"], r1(x.get("score"))) for x in log.get("signals", [])],
    }


def diff_reasons(last: dict, new: dict) -> list[str]:
    """補正版要記錄的差異：名單、排名、分數（差 0.1 以上）、因子分數、模型版本、資料日期。"""
    a, b = _sig(last), _sig(new)
    ids = lambda x: {m: [p[0] for p in g] for m, g in x["picks"].items()}
    out = []
    if ids(a) != ids(b) or [x[0] for x in a["signals"]] != [x[0] for x in b["signals"]]:
        out.append("名單")
    elif a["picks"] != b["picks"] or a["signals"] != b["signals"]:
        out.append("排名或分數")
    if a["factors"] != b["factors"]:
        out.append("因子分數")
    if last.get("model_version") != new.get("model_version") or last.get("model_hash", new.get("model_hash")) != new.get("model_hash"):
        out.append("模型版本")
    if (last.get("data_dates") or {}) != (new.get("data_dates") or {}):
        out.append("資料日期")
    return out


def load(first_only: bool = True) -> list[dict]:
    if not HISTORY.exists():
        return []
    files = sorted(HISTORY.glob("*.json"))
    if first_only:
        files = [p for p in files if not re.search(r"\.v\d+\.json$", p.name)]
    return [json.loads(p.read_text("utf-8")) for p in files]


def _prices(ids: list[str], first: str, dbname: str):
    with db.connect(dbname) as con:
        q = ",".join("?" * len(ids))
        # 從選股日前幾天開始讀，進場日才有前一天收盤價可以判斷漲停
        start = (pd.Timestamp(first) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        px = db.read(con, f"SELECT * FROM prices WHERE stock_id IN ({q}) AND date >= ? ORDER BY date", tuple(ids) + (start,)) if ids else pd.DataFrame()
        ex = db.read(con, f"SELECT * FROM exrights WHERE stock_id IN ({q})", tuple(ids)) if ids else pd.DataFrame()
        idx = db.read(con, "SELECT * FROM index_prices WHERE date >= ? ORDER BY date", (first,))
        bench = benchmark(con, start)
    exg = {k: g for k, g in ex.groupby("stock_id")} if len(ex) else {}
    series = {}
    for sid, p in (px.groupby("stock_id") if len(px) else []):
        adj, _ = metrics.adjust_prices(p.reset_index(drop=True), exg.get(sid, pd.DataFrame()))
        adj = adj.set_index("date")[["open", "high", "low", "close", "value"]].astype(float)
        adj["prev"] = adj["close"].shift(1)
        series[sid] = adj
    idx = idx.set_index("date")["close"] if len(idx) else pd.Series(dtype=float)
    # 交易日曆：加權指數偶爾漏存某一天，所以和 0050 的交易日合併，避免持有天數算錯
    cal = sorted(set(idx.index) | {d for d in bench[0].index if d >= first})
    return series, idx.reindex(cal), bench


def untradable(s: pd.DataFrame, entry: str) -> str | None:
    """進場日買不買得到：沒有成交、漲停鎖死、成交金額太少。買得到回傳 None。"""
    if s is None or entry not in s.index:
        return "not_traded"
    o, hi, lo, prev, val = (s.at[entry, k] for k in ("open", "high", "low", "prev", "value"))
    if pd.notna(prev) and prev > 0 and pd.notna(o) and o >= prev * (1 + LIMIT_UP) and hi == lo == o:
        return "limit_up"
    if pd.notna(val) and val < THIN_VALUE:
        return "thin"
    return None


def _cohort(group: list[dict], day: str, series: dict, idx: pd.Series, bench, h: int | None, obs: str):
    """一組股票從 day 隔日開盤進場、持有 h 天（None = 持有到觀察日）的結果。"""
    dates = idx.index[idx.index > day]
    if not len(dates):
        return None
    entry = dates[0]
    if h is None:
        exit_ = obs
    elif len(dates) >= h:
        exit_ = dates[h - 1]
    else:
        return None  # 還沒到期
    if exit_ < entry or day not in idx.index:
        return None
    bp, kind = bench
    b, basis, b_price = None, NO_BENCH, None  # 基準固定是 0050；缺價格就標示無法比較，不換別的指數
    if kind == BENCH_ID and entry in bp.index and exit_ in bp.index:
        o, c = bp.at[entry, "open"], bp.at[exit_, "close"]
        if pd.notna(o) and o > 0 and pd.notna(c) and c > 0:
            b, basis = float(c / o - 1), BENCH_ID
            ro, rc = bp.at[entry, "open_raw"], bp.at[exit_, "close_raw"]
            b_price = float(rc / ro - 1) if pd.notna(ro) and ro > 0 and pd.notna(rc) and rc > 0 else None
    path_dates = dates[(dates >= entry) & (dates <= exit_)]
    rets, real, paths, per = [], [], [], {}
    skip = {k: 0 for k in SKIP_LABELS}
    stale = {k: 0 for k in STALE_LABELS}
    for p in group:
        s = series.get(p["stock_id"])
        why = untradable(s, entry)
        if why:
            skip[why] += 1
            continue
        o = s.at[entry, "open"]
        if not (pd.notna(o) and o > 0):
            o = s.at[entry, "close"]
        held = s["close"].loc[entry:exit_]
        held = held[held > 0].dropna()
        if not (pd.notna(o) and o > 0) or held.empty:
            skip["no_price"] += 1  # 沒有有效的進場價或結算價：不計入，列在資料品質統計
            continue
        r = float(held.iloc[-1] / o - 1 - COST)
        if held.index[-1] != exit_:  # 出場日沒有成交：用最後成交價估算
            stale["resumed" if (s.index > exit_).any() else "no_later"] += 1
        else:
            real.append(r)
        rets.append(r)
        per[p["stock_id"]] = r
        paths.append(s["close"].where(s["close"] > 0).reindex(path_dates).ffill() / o)
    n_skip = sum(skip.values())
    base = {"n": len(group), "missing": n_skip, "skip": skip, "stale_exit": sum(stale.values()), "stale": stale,
            "missing_pct": _p(n_skip / len(group)) if group else None, "basis": basis, "exit": exit_, "_per": per}
    if not rets:
        return {**base, "priced": 0}
    nav = pd.concat(paths, axis=1).mean(axis=1)
    mdd = float((nav / nav.cummax() - 1).min()) if len(nav) else 0.0
    r = np.array(rets)
    out = {**base, "priced": len(r), "entry": entry, "held": len(path_dates),
           "avg": _p(r.mean()), "median": _p(np.median(r)), "win": _p((r > 0).mean()), "mdd": _p(mdd),
           "est_pct": _p(1 - len(real) / len(r)), "real_avg": _p(np.mean(real)) if real else None, "real_n": len(real),
           "bench": None, "excess": None, "beat": None, "real_excess": None, "slip": None}
    if b is None:
        return out
    return {**out, "bench": _p(b), "bench_price": _p(b_price) if b_price is not None else None, "excess": _p(r.mean() - b), "beat": _p((r > b).mean()),
            "real_excess": _p(np.mean(real) - b) if real else None,
            "slip": {f"{x * 100:g}": {"excess": _p(r.mean() - x - b), "beat": _p((r - x > b).mean())} for x in SLIPS}}


def attribution(per: dict, uni: dict, ind: dict) -> dict | None:
    """簡化產業歸因（等權）：名單平均報酬 − 全市場（當天可排名股票）平均報酬 = 產業配置 + 同產業選股。
    產業配置 = 名單裡每檔股票「所屬產業的全市場平均報酬」的平均 − 全市場平均（選到強勢產業的貢獻）；
    同產業選股 = 每檔股票報酬 − 所屬產業平均報酬，再平均（在產業內挑得好不好）。等權重，兩項相加剛好等於總差異。"""
    if not per or len(uni) < 20:
        return None
    by = {}
    for sid, r in uni.items():
        by.setdefault(ind.get(sid) or "未分類", []).append(r)
    avg = {k: float(np.mean(v)) for k, v in by.items()}
    u = float(np.mean(list(uni.values())))
    keep = [(sid, r) for sid, r in per.items() if sid in uni]
    if not keep:
        return None
    alloc = float(np.mean([avg[ind.get(sid) or "未分類"] - u for sid, _ in keep]))
    sel = float(np.mean([r - avg[ind.get(sid) or "未分類"] for sid, r in keep]))
    small = sum(1 for sid, _ in keep if len(by[ind.get(sid) or "未分類"]) < 5)
    return {"universe": _p(u), "allocation": _p(alloc), "selection": _p(sel), "total": _p(alloc + sel),
            "n": len(keep), "small_industry": small}


def _card(rows: list[dict], names: list[str]) -> dict:
    """績效驗證：所有已到期、正式樣本外的選股日，各期間的平均。
    只合併有 0050 基準、而且名單在進場日開盤前就產生的紀錄；缺 0050（無法比較）和名單產生太晚的筆數另外列出。"""
    card = {}
    for name in names:
        card[name] = {}
        for h in HORIZONS:
            done = [(r, r["fixed"][name][h]) for r in rows if (r["fixed"].get(name) or {}).get(h) and r["fixed"][name][h].get("priced")]
            late = [(r, x) for r, x in done if r.get("on_time") is False]
            timely = [(r, x) for r, x in done if r.get("on_time") is not False]
            cs = [(r, x) for r, x in timely if x["basis"] == BENCH_ID]
            no_bench = len(timely) - len(cs)
            if not cs:
                if done:
                    card[name][h] = {"cohorts": 0, "no_bench": no_bench, "late": len(late)}
                continue
            c = [x for _, x in cs]
            m = lambda k: float(np.mean([x[k] for x in c]))
            uni = [x["avg"] - r["fixed"]["universe"][h]["avg"] for r, x in cs
                   if name != "universe" and ((r["fixed"].get("universe") or {}).get(h) or {}).get("priced")]
            real = [x["real_excess"] for x in c if x.get("real_excess") is not None]
            priced = sum(x["priced"] for x in c)
            total = sum(x["n"] for x in c)
            card[name][h] = {"cohorts": len(c), "avg": round(m("avg"), 2), "excess": round(m("excess"), 2),
                             "beat_cohorts": _p(np.mean([x["excess"] > 0 for x in c])), "win": round(m("win"), 2),
                             "worst_mdd": min(x["mdd"] for x in c), "basis": sorted({x["basis"] for x in c}),
                             "slip": {k: round(float(np.mean([x["slip"][k]["excess"] for x in c])), 2) for k in c[0]["slip"]},
                             "vs_universe": round(float(np.mean(uni)), 2) if uni else None, "vs_universe_n": len(uni),
                             "missing_pct": _p(sum(x["missing"] for x in c) / total) if total else None,
                             "skip": {k: sum(x["skip"].get(k, 0) for x in c) for k in SKIP_LABELS},
                             "stale_exit": sum(x["stale_exit"] for x in c),
                             "stale": {k: sum(x["stale"].get(k, 0) for x in c) for k in STALE_LABELS},
                             "est_pct": _p(sum(x["stale_exit"] for x in c) / priced) if priced else None,
                             "real_excess": round(float(np.mean(real)), 2) if real else None,
                             "real_avg": round(float(np.mean([x["real_avg"] for x in c if x.get("real_avg") is not None])), 2)
                             if any(x.get("real_avg") is not None for x in c) else None,
                             "no_bench": no_bench, "late": len(late),
                             "bench": round(m("bench"), 2), "bench_price": round(float(np.mean(bp)), 2) if (bp := [x["bench_price"] for x in c if x.get("bench_price") is not None]) else None,
                             "attr": _attr_mean([x.get("attr") for x in c])}
    return card


def _attr_mean(xs: list) -> dict | None:
    xs = [x for x in xs if x]
    if not xs:
        return None
    return {**{k: round(float(np.mean([x[k] for x in xs])), 2) for k in ("universe", "allocation", "selection", "total")},
            "cohorts": len(xs), "small_industry": sum(x["small_industry"] for x in xs), "n": sum(x["n"] for x in xs)}


def performance(dbname: str = "stocks.db") -> dict:
    logs = load()
    if not logs:
        return {"days": 0, "logs": [], "scorecard": {}}

    def groups(L):
        g = list(L["picks"].items()) + [("rec_" + m, x) for m, x in (L.get("rec_picks") or {}).items()] + [("signals", L["signals"])]
        if L.get("universe_ids"):
            g.append(("universe", [{"stock_id": x} for x in L["universe_ids"]]))
        return g
    ids = sorted({p["stock_id"] for L in logs for _, g in groups(L) for p in g})
    series, idx, bench = _prices(ids, logs[0]["date"], dbname)
    obs = idx.index[-1] if len(idx) else None
    rows = []
    for L in logs:
        d = L["date"]
        nxt = idx.index[idx.index > d]
        row = {"date": d, "model_version": L.get("model_version"), "model_hash": L.get("model_hash"),
               "generated_at": L.get("generated_at"), "on_time": on_time(L, nxt[0] if len(nxt) else None),
               "track": {}, "fixed": {}}
        for name, g in groups(L):
            row["track"][name] = _cohort(g, d, series, idx, bench, None, obs) if obs else None
            row["fixed"][name] = {h: _cohort(g, d, series, idx, bench, h, obs) for h in HORIZONS}
        rows.append(row)
    names = list(dict.fromkeys(k for L in logs for k, _ in groups(L)))
    with db.connect(dbname) as con:  # 產業別（目前的分類；歷史上改過分類的股票會用新分類）
        q = ",".join("?" * len(ids))
        ind = dict(con.execute(f"SELECT stock_id, industry FROM stocks WHERE stock_id IN ({q})", tuple(ids)).fetchall()) if ids else {}
    for r in rows:
        for h in HORIZONS:
            u = (r["fixed"].get("universe") or {}).get(h)
            for name in r["fixed"]:
                c = r["fixed"][name].get(h)
                if c and name != "universe" and u and u.get("priced"):
                    c["attr"] = attribution(c.get("_per", {}), u.get("_per", {}), ind)
    by_model = {}
    for mv in dict.fromkeys(r["model_version"] or "v1.0" for r in rows):
        sub = [r for r in rows if (r["model_version"] or "v1.0") == mv]
        hashes = list(dict.fromkeys(r["model_hash"] for r in sub if r["model_hash"]))
        by_model[mv] = {"days": len(sub), "first": sub[0]["date"], "last": sub[-1]["date"], "hashes": hashes,
                        "hash_changed": len(hashes) > 1, "scorecard": _card(sub, names)}
    industry = _by_industry(logs, rows)
    for r in rows:  # 個股明細只用來算產業，不放進網站資料
        for c in [r["track"].get(k) for k in r["track"]] + [x for g in r["fixed"].values() for x in g.values()]:
            if c:
                c.pop("_per", None)
    # 0050 資料完整度：觀察期間每個交易日都要有 0050 的開盤、收盤價
    span = idx.index[idx.index >= logs[0]["date"]]
    bp = bench[0]
    gaps = [d for d in span if d not in bp.index or not (bp.at[d, "open"] > 0 and bp.at[d, "close"] > 0)] if bench[1] == BENCH_ID else list(span)
    return {"days": len(logs), "first": logs[0]["date"], "observed": obs, "horizons": list(HORIZONS),
            "cost_pct": round(COST * 100, 3), "bench": BENCH_ID, "bench_ok": bench[1] == BENCH_ID, "basis": BENCH_ID,
            "model_hash": model_hash(), "open_at": OPEN_AT[:5],
            "late_days": [r["date"] for r in rows if r["on_time"] is False],
            "bench_coverage": {"days": len(span), "missing": len(gaps), "missing_dates": gaps[:10]},
            "slips": [f"{x * 100:g}" for x in SLIPS], "skip_labels": SKIP_LABELS, "stale_labels": STALE_LABELS,
            "thin_value": THIN_VALUE, "logs": rows[::-1], "scorecard": _card(rows, names), "by_model": by_model,
            "by_industry": industry}


def _by_industry(logs: list[dict], rows: list[dict]) -> dict:
    """AI 均衡型入選股票依產業分組：每一檔、每一次入選的超額報酬平均（最短的已到期期間，和最長的已到期期間）。"""
    out = {}
    for h in HORIZONS:
        acc = {}
        for L, r in zip(logs, rows):
            c = (r["fixed"].get("balanced") or {}).get(h)
            if not c or not c.get("priced") or c["basis"] != BENCH_ID or r.get("on_time") is False:
                continue
            ind = {p["stock_id"]: p.get("industry") for p in L["picks"].get("balanced", [])}
            for sid, ret in c["_per"].items():
                acc.setdefault(ind.get(sid) or "未分類", []).append(ret * 100 - c["bench"])
        if acc:
            out[h] = sorted(({"industry": k, "n": len(v), "excess": round(float(np.mean(v)), 2),
                              "beat": round(float(np.mean([x > 0 for x in v]) * 100), 1)} for k, v in acc.items()),
                            key=lambda x: -x["n"])
    return out


def changes(snap: dict) -> dict | None:
    """和上一個選股日相比：均衡型前 20 名的新進榜、掉出榜、排名變化，以及分數變化最大的因子。"""
    day = (snap.get("data_dates") or {}).get("prices")
    prev = [L for L in load(first_only=False) if L["date"] < (day or "")]
    if not prev:
        return None
    last = max(prev, key=lambda L: (L["date"], L.get("version", 1)))
    before = {p["stock_id"]: p for p in last["picks"].get("balanced", [])}
    by = {s["stock_id"]: s for s in snap["stocks"]}
    now = sorted((s for s in snap["stocks"] if s.get("is_common") and s["ai"]["modes"]["balanced"]["rank"]),
                 key=lambda s: s["ai"]["modes"]["balanced"]["rank"])[:TOP_N]
    now_ids = [s["stock_id"] for s in now]

    def driver(sid):
        old = (before.get(sid) or {}).get("factors") or {}
        s = by.get(sid) or {}
        diffs = {f: (s.get(f"score_{f}") or 0) - old[f] for f in FACTORS if old.get(f) is not None and s.get(f"score_{f}") is not None}
        if not diffs:
            return None
        f = max(diffs, key=lambda k: abs(diffs[k]))
        return {"factor": f, "change": round(diffs[f], 1)} if abs(diffs[f]) >= 3 else None

    new = [{"stock_id": s["stock_id"], "name": s["name"], "rank": s["ai"]["modes"]["balanced"]["rank"], "driver": driver(s["stock_id"])}
           for s in now if s["stock_id"] not in before]
    dropped = []
    for sid, p in before.items():
        if sid in now_ids:
            continue
        s = by.get(sid) or {}
        r = s.get("ai", {}).get("modes", {}).get("balanced", {})
        why = ("資料不足，暫不排名" if s and not s.get("complete") else "未通過基本門檻" if s and not s.get("base_ok")
               else f"排名降到第 {r['rank']} 名" if r.get("rank") else "不在排行榜")
        dropped.append({"stock_id": sid, "name": p["name"], "was": p["rank"], "why": why, "driver": driver(sid)})
    moved = []
    for s in now:
        p = before.get(s["stock_id"])
        if p and p["rank"] != s["ai"]["modes"]["balanced"]["rank"]:
            moved.append({"stock_id": s["stock_id"], "name": s["name"], "was": p["rank"], "rank": s["ai"]["modes"]["balanced"]["rank"],
                          "driver": driver(s["stock_id"])})
    moved.sort(key=lambda x: abs(x["was"] - x["rank"]), reverse=True)
    return {"since": last["date"], "new": new, "dropped": dropped, "moved": moved[:8]}


def _p(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x) * 100, 2)


def stock_history(dbname: str = "stocks.db", h: int = 20) -> dict:
    """V13 個股頁「AI 紀錄」：這檔股票哪幾天進過正式精選（前 20 名）、名次，以及之後的實際表現。
    規則和績效中心一樣：選股日隔天開盤買、第 h 個交易日收盤出場、扣成本、和 0050 同期比；還沒滿 h 天的標「未到期」，只列目前報酬。"""
    logs = [L for L in load() if L.get("rec_picks")]
    if not logs:
        return {}
    ids = sorted({p["stock_id"] for L in logs for g in L["rec_picks"].values() for p in g})
    series, idx, bench = _prices(ids, logs[0]["date"], dbname)
    cal, b = list(idx.index), bench[0]
    out: dict = {}
    for L in logs:
        d = L["date"]
        nxt = [x for x in cal if x > d]
        if not nxt:
            continue
        entry = nxt[0]
        for mode, g in L["rec_picks"].items():
            for p in g:
                s = series.get(p["stock_id"])
                row = {"date": d, "mode": mode, "rank": p.get("rank"), "raw_rank": p.get("raw_rank"), "score": p.get("score")}
                if s is None or untradable(s, entry):
                    row["status"] = "買不到"
                else:
                    held = [x for x in nxt[:h] if x in s.index]
                    last = held[-1]
                    o = float(s.at[entry, "open"])
                    r = float(s.at[last, "close"] / o - 1) - COST
                    br = float(b.at[last, "close"] / b.at[entry, "open"] - 1) if entry in b.index and last in b.index else None
                    row.update({"days": len(held), "ret": round(r * 100, 2), "excess": None if br is None else round((r - br) * 100, 2),
                                "status": "已到期" if len(nxt) >= h else "未到期"})
                out.setdefault(p["stock_id"], []).append(row)
    return out
