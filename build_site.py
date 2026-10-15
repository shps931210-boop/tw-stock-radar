"""把真實資料輸出成不需要後端的靜態網站（site/），給 GitHub Pages 等靜態主機使用。

    python -m app.update      # 先更新資料
    python build_site.py      # 產生 site/index.html、site/data/*.json

沒有成功取得真實行情時，只輸出失敗原因（site/data/status.json），不會用示範資料頂替。
"""
import base64
import json
import os
import shutil
import sys
from pathlib import Path

from app import config, db
from app import profile
from app.server import stock_detail

def quality(snap: dict) -> dict:
    """資料品質摘要（data/quality.json）：完整度、排名數、缺資料原因，方便遠端檢查。"""
    from collections import Counter
    st = snap["stocks"]
    base = [r for r in st if r.get("base_ok")]
    pick = ("close", "date", "adj_events", "data_completeness", "missing_detail", "stale_notes", "fscore", "fscore_n",
            "inst_date", "inst_lag", "margin_lag", "fin_period", "rev_ym", "val_date", "composite", "verdict")
    return {
        "generated_at": snap.get("generated_at"), "rankings_blocked": snap.get("rankings_blocked"),
        "adjusted_markets": snap.get("adjusted_markets"), "freshness_rule": snap.get("freshness_rule"), "backup": snap.get("backup"),
        "bench": {"track": (snap.get("track") or {}).get("bench"), "backtest": (snap.get("backtest") or {}).get("bench")},
        "scorecard": (snap.get("track") or {}).get("scorecard"),
        "stocks": len(st), "base_ok": len(base), "complete": sum(bool(r.get("complete")) for r in base),
        "ranked": {m: sum(1 for r in st if r["ai"]["modes"][m]["rank"]) for m in snap.get("ai_modes", {})},
        "signals": Counter(r.get("timing", {}).get("status") for r in st),
        "adjusted_stocks": sum(1 for r in st if (r.get("adj_events") or 0) > 0),
        "chips_ok": sum(r.get("inst_ratio_5d") is not None for r in base),
        "missing": Counter(m.split("（")[0] for r in base for m in r.get("missing_detail") or []).most_common(),
        "stale": Counter(n.split(" ")[0] for r in base for n in r.get("stale_notes") or []).most_common(),
        "samples": {r["stock_id"]: {k: r.get(k) for k in pick} | {"ai": r["ai"]["modes"].get("balanced"), "summary": r["ai"].get("summary"),
                                                         "profile": profile.build(r)}
                    for r in st if r["stock_id"] in ("2330", "2303", "2317", "2454", "6488")},
    }


def supabase_config() -> dict | None:
    """會員登入設定（GitHub repository variables：SUPABASE_URL、SUPABASE_KEY）。只接受公開金鑰；放錯成私密金鑰就不啟用。"""
    url, key = os.environ.get("SUPABASE_URL", "").strip(), os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    if not url.startswith("https://"):
        print("SUPABASE_URL 要是 https:// 開頭，先不啟用會員功能")
        return None
    secret = key.startswith("sb_secret_")
    if key.count(".") == 2:  # 舊版 JWT 金鑰：看 role 是不是 service_role
        try:
            part = key.split(".")[1]
            secret = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("role") == "service_role"
        except ValueError:
            return None
    if secret:
        print("⚠️ SUPABASE_KEY 是私密金鑰，不能放在網站上。請改成 publishable key，並到 Supabase 重新產生私密金鑰。會員功能先不啟用。")
        return None
    return {"url": url.rstrip("/"), "key": key}


def public_data(snap: dict) -> dict:
    """公開 JSON：產業熱力圖、選股成績單。每個檔案都有 schema_version、generated_at、data_as_of、source、warnings。"""
    tr = snap.get("track") or {}
    head = {"schema_version": 1, "generated_at": snap.get("generated_at"), "data_as_of": (snap.get("data_dates") or {}).get("prices")}
    defs = {"entry": "選股日隔天開盤價", "exit": "選股日之後第 N 個交易日收盤價（進場日算第 1 天）",
            "cost_pct": tr.get("cost_pct"), "bench": "0050 還原除權息（總報酬），同一段期間：進場日開盤 → 出場日收盤",
            "on_time": f"名單要在進場日 {tr.get('open_at', '09:00')} 前產生才計入", "lists": "只用首次發布版名單"}
    warn = (["紀錄天數還少，結果容易受單一事件影響"] if (tr.get("days") or 0) < 20 else []) + (["有選股日缺 0050 價格，標示無法比較"] if (tr.get("bench_coverage") or {}).get("missing") else [])
    return {
        "industry/latest.json": snap.get("sectors") or {**head, "industries": [], "warnings": ["沒有產業資料"]},
        "performance/summary.json": {**head, "source": "history/picks 首次發布版名單 + 證交所／櫃買中心收盤行情", "definitions": defs, "warnings": warn,
                                     "coverage": {"days": tr.get("days"), "first": tr.get("first"), "observed": tr.get("observed"), "late_days": tr.get("late_days")},
                                     "horizons": tr.get("horizons"), "model_hash": tr.get("model_hash"),
                                     "scorecard": tr.get("scorecard"), "by_model": tr.get("by_model"), "by_industry": tr.get("by_industry")},
        "events/calendar.json": snap.get("calendar") or {**head, "events": [], "warnings": ["沒有事件資料"]},
        "performance/cohorts.json": {**head, "source": "history/picks", "definitions": defs, "warnings": warn, "cohorts": tr.get("logs", [])},
    }


root = Path(__file__).parent
out = root / "site"
shutil.rmtree(out, ignore_errors=True)
(out / "data" / "stock").mkdir(parents=True)

html = (root / "static" / "index.html").read_text("utf-8")
chat_cfg = json.loads((root / "chat.json").read_text("utf-8")) if (root / "chat.json").exists() else {}  # AI 研究助理網址與模型（部署時寫入）
chat = chat_cfg.get("url")
chat_provider = chat_cfg.get("provider") if chat_cfg.get("provider") in ("claude", "free") else "claude"
html = html.replace("<script>\nconst STATIC", f"<script>window.STATIC_SITE = true; window.CHAT_URL = {json.dumps(chat)}; window.CHAT_PROVIDER = {json.dumps(chat_provider)}; window.SUPA = {json.dumps(supabase_config())};</script>\n<script>\nconst STATIC", 1)
(out / "index.html").write_text(html, "utf-8")
shutil.copy(root / "static" / "portfolio.js", out / "portfolio.js")  # 我的投資組合：計算模組（網站與測試共用）

errors, last = [], None
if (config.DATA_DIR / "stocks.db").exists():
    with db.connect() as con:
        errors = (db.get_meta(con, "last_errors") or "").splitlines()
        last = db.get_meta(con, "last_update")
(out / "data" / "status.json").write_text(json.dumps(
    {"no_data": True, "last_attempt": last, "errors": errors[:15],
     "detail": "還沒有成功取得真實行情，所以沒有選股結果。下一次排程更新成功後會自動出現。"}, ensure_ascii=False), "utf-8")

if (config.DATA_DIR / "global_markets.json").exists():  # 國際市場（python -m app.global_mkt）
    (out / "data" / "global").mkdir(parents=True, exist_ok=True)
    shutil.copy(config.DATA_DIR / "global_markets.json", out / "data" / "global" / "markets.json")
pit = root / "history" / "backtest" / "ai_pit.json"
if pit.exists():  # AI 多因子歷史回測（python -m app.pit_backtest，每週由「AI 歷史回測」執行）
    (out / "data" / "backtest").mkdir(parents=True, exist_ok=True)
    shutil.copy(pit, out / "data" / "backtest" / "ai_pit.json")
for extra in ("joint.json",):  # AI 量化選股＋進出場聯合回測（python -m app.joint_backtest）
    if (root / "history" / "backtest" / extra).exists():
        (out / "data" / "backtest").mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "history" / "backtest" / extra, out / "data" / "backtest" / extra)
fac = root / "history" / "backtest" / "factors.json"
if fac.exists():  # 因子有效性檢驗（python -m app.factor_research，每週由「AI 歷史回測」執行）
    (out / "data" / "backtest").mkdir(parents=True, exist_ok=True)
    shutil.copy(fac, out / "data" / "backtest" / "factors.json")
# 我的投資組合用：AI 每日名單的入選紀錄索引（只用每天第一次發布的名單）、0050 還原股價（比較用）
try:
    idx: dict = {}
    for f in sorted((root / "history" / "picks").glob("????-??-??.json")):
        lg = json.loads(f.read_text("utf-8"))
        for mode, xs in (lg.get("picks") or {}).items():
            for x in xs:
                idx.setdefault(x["stock_id"], []).append([lg["date"], mode, x.get("rank"), x.get("score"), x.get("close")])
    (out / "data" / "picks_index.json").write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), "utf-8")
    if (config.DATA_DIR / "stocks.db").exists():
        from app import db as _db, track as _track
        with _db.connect("stocks.db") as con:
            b, kind = _track.benchmark(con)
        if kind == _track.BENCH_ID and len(b):
            (out / "data" / "bench").mkdir(parents=True, exist_ok=True)
            (out / "data" / "bench" / "0050.json").write_text(json.dumps(
                {"id": "0050", "note": "還原除權息收盤價（含配息再投入）", "prices": [[d, round(float(c), 4), round(float(r), 2)] for d, c, r in zip(b.index, b["close"], b["close_raw"])]},
                ensure_ascii=False, separators=(",", ":")), "utf-8")
except Exception as e:
    print("投資組合輔助資料失敗：", e)
if (config.DATA_DIR / "coverage.json").exists():  # 個股研究資料覆蓋率（app/coverage.py）
    shutil.copy(config.DATA_DIR / "coverage.json", out / "data" / "coverage.json")
if (config.DATA_DIR / "probe.json").exists():  # 連線診斷結果（python -m app.probe）
    shutil.copy(config.DATA_DIR / "probe.json", out / "data" / "probe.json")

snap_path = config.DATA_DIR / "snapshot.json"
backup_path = config.DATA_DIR / "backup_status.json"  # python -m app.backup status 寫的
if snap_path.exists():
    snap = json.loads(snap_path.read_text("utf-8"))
    snap["backup"] = json.loads(backup_path.read_text("utf-8")) if backup_path.exists() else None
    (out / "data" / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), "utf-8")
    rows = {s["stock_id"]: s for s in snap["stocks"] if s.get("is_common")}
    ids = list(rows)
    dec_path = config.DATA_DIR / "decision.json"  # 個股研究決策卡（app/snapshot.py _decision）
    dec = json.loads(dec_path.read_text("utf-8")) if dec_path.exists() else {}
    for sid in ids:
        (out / "data" / "stock" / f"{sid}.json").write_text(
            json.dumps({**stock_detail(sid, "stocks.db", rows[sid]), "decision": dec.get(sid)}, ensure_ascii=False, separators=(",", ":")), "utf-8")
    (out / "data" / "quality.json").write_text(json.dumps(quality(snap), ensure_ascii=False, indent=1), "utf-8")
    for path, data in public_data(snap).items():  # 給其他程式讀的公開資料（不含任何會員資料）
        (out / "data" / path).parent.mkdir(parents=True, exist_ok=True)
        (out / "data" / path).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), "utf-8")
    try:  # AI 研究助理用的精簡全市場資料（worker/ 讀這份，不讀整份 snapshot）
        from app import ai_data
        (out / "data" / "ai").mkdir(parents=True, exist_ok=True)
        (out / "data" / "ai" / "universe.json").write_text(json.dumps(ai_data.build(snap), ensure_ascii=False, separators=(",", ":")), "utf-8")
    except Exception as e:
        print("AI 資料失敗：", e)
    try:  # 市場新聞中心：公告與新聞，附來源與連結
        from app import news
        (out / "data" / "news").mkdir(parents=True, exist_ok=True)
        (out / "data" / "news" / "latest.json").write_text(json.dumps(news.latest(snap), ensure_ascii=False, separators=(",", ":")), "utf-8")
    except Exception as e:
        print("新聞中心資料失敗：", e)
    try:  # 對帳報告：用獨立寫法從原始成交價重算績效與產業數字，逐筆比對
        from app import audit
        (out / "data" / "audit").mkdir(parents=True, exist_ok=True)
        (out / "data" / "audit" / "latest.json").write_text(json.dumps(audit.build(snap), ensure_ascii=False, separators=(",", ":")), "utf-8")
    except Exception as e:
        print("對帳報告失敗：", e)
    print(f"已產生 site/（{len(ids)} 檔個股資料）")
else:
    print("沒有真實資料快照，site/ 只會顯示資料取得失敗")
    if "--strict" in sys.argv:
        sys.exit(1)  # GitHub Actions：不發布，網站維持上一次成功的版本

# 資料版本：每個頁面的「重新整理」先讀這份，比對 data_version，有新版才重新下載資料
from app import version  # noqa: E402
files = {str(f.relative_to(out / "data")): f.read_bytes() for f in sorted((out / "data").rglob("*.json")) if f.name not in ("status.json", "version.json")}
ver = version.build(json.loads(snap_path.read_text("utf-8")) if snap_path.exists() else None, files)
(out / "data" / "version.json").write_text(json.dumps(ver, ensure_ascii=False, indent=1), "utf-8")
print(f"資料版本 {ver['data_version']}：{ver['data_state_label']}・交易日 {ver['trading_date']}")
