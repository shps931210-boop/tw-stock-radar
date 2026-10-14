"""網站資料版本（site/data/version.json）：讓每個頁面的「重新整理」可以先問「有沒有新版」，有才重新下載。

時間欄位分開，不互相冒充：
- trading_date：行情對應的交易日（證交所／櫃買中心的收盤資料日）。
- fetched_at：後端最後一次向資料來源抓資料的時間。
- calculated_at：指標、評分、榜單的計算時間。
- published_at：這份網站資料產生的時間（GitHub Pages 實際上線還要再等幾分鐘）。
所有時間都是 ISO 8601，附 +08:00（台灣時間）。

data_state（目前只有盤後資料，沒有盤中即時行情）：
- postclose_final：行情是最新交易日，主要來源都已更新，排行可用。
- postclose_partial：行情日期正確，但有來源還沒更新（例如上櫃行情、法人較晚公布）或歷史資料不足。
- stale：證交所已公布更新的交易日，網站資料還沒跟上。
- unavailable：沒有成功取得真實資料。
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone


TW = timezone(timedelta(hours=8))
SCHEDULE = [(14, 10), (14, 40), (16, 30), (21, 0)]  # 與 .github/workflows/update.yml 的排程一致（台灣時間、週一到週五）
STATE_TXT = {"postclose_final": "正式盤後", "postclose_partial": "盤後（部分資料未齊）", "stale": "資料落後", "unavailable": "沒有資料"}


def _iso(t: str | None) -> str | None:
    """snapshot 裡的時間是台灣時間、沒有時區；補上 +08:00。"""
    if not t:
        return None
    return t if ("+" in t[10:] or t.endswith("Z")) else t + "+08:00"


def next_run(now: datetime | None = None) -> str:
    """下一次排程更新的台灣時間（GitHub 排程常會晚幾分鐘到幾十分鐘開始）。"""
    now = (now or datetime.now(TW)).astimezone(TW)
    for d in range(8):
        day = (now + timedelta(days=d)).date()
        if day.weekday() >= 5:
            continue
        for h, m in SCHEDULE:
            t = datetime(day.year, day.month, day.day, h, m, tzinfo=TW)
            if t > now:
                return t.isoformat(timespec="minutes")
    return ""


def state(snap: dict | None) -> tuple[str, list[str]]:
    if not snap or not snap.get("stocks"):
        return "unavailable", ["還沒有成功取得真實行情"]
    price = (snap.get("data_dates") or {}).get("prices")
    exp = snap.get("expected_date")
    if exp and price and price < exp and snap.get("expected_basis") == "official":
        return "stale", [f"證交所已公布 {exp} 的行情，網站資料只到 {price}"]
    why = []
    if exp and price and price < exp:
        why.append(f"推估最新交易日應為 {exp}，網站資料到 {price}（無法連到證交所確認；休市則正常）")
    for h in snap.get("health") or []:
        if not h.get("last_ok") or (price and h["last_ok"] < price):
            why.append(f"{h.get('label') or h.get('source')}最後更新：{h.get('last_ok') or '從未成功'}")
    hist = snap.get("history") or {}
    if hist and not hist.get("ready", True):
        why.append(f"歷史股價 {hist.get('days')}／{hist.get('need')} 天，技術指標與排行還不完整")
    if snap.get("rankings_blocked"):
        why.append(str(snap["rankings_blocked"]))
    return ("postclose_partial" if why else "postclose_final"), why


# 每次執行都會變、但不代表資料有變的欄位：算指紋時忽略，重跑同一份資料時版本才不會變
VOLATILE = {"generated_at", "ran_at", "checked_at", "last_update", "updated_prices", "updated_revenue", "updated_financials", "backup", "last_attempt"}


def _strip(v):
    if isinstance(v, dict):
        return {k: _strip(x) for k, x in v.items() if k not in VOLATILE}
    if isinstance(v, list):
        return [_strip(x) for x in v]
    return v


def _fingerprint(b: bytes) -> str:
    try:
        data = json.dumps(_strip(json.loads(b)), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    except ValueError:
        data = b
    return hashlib.sha1(data).hexdigest()[:12]


def build(snap: dict | None, files: dict[str, bytes], now: datetime | None = None) -> dict:
    """files：已寫出的網站資料（路徑 → 內容），每份算一個內容指紋；內容沒變，版本就不變。"""
    now = (now or datetime.now(TW)).astimezone(TW)
    ds = {p: _fingerprint(b) for p, b in sorted(files.items())}
    st, why = state(snap)
    meta = (snap or {}).get("meta") or {}
    return {
        "schema_version": 1,
        "data_version": hashlib.sha1("|".join(f"{p}:{h}" for p, h in ds.items()).encode()).hexdigest()[:12],
        "data_state": st, "data_state_label": STATE_TXT[st], "market_session": "postclose",
        "source_name": "證交所、櫃買中心盤後資料；FinMind（營收、財報、新聞）",
        "trading_date": (snap or {}).get("data_dates", {}).get("prices"),
        "data_dates": (snap or {}).get("data_dates"),
        "fetched_at": _iso(meta.get("last_update")),
        "calculated_at": _iso((snap or {}).get("generated_at")),
        # fast＝盤後快速更新（只有最近幾天的行情與籌碼）；full＝完整更新（含歷史回補、財報、股利）
        "update_mode": meta.get("update_mode") or "full",
        "model": {"version": ((snap or {}).get("model") or {}).get("version"), "hash": ((snap or {}).get("track") or {}).get("model_hash"),
                  "calculated_at": _iso((snap or {}).get("generated_at")), "prices_date": (snap or {}).get("data_dates", {}).get("prices")},
        "published_at": now.isoformat(timespec="seconds"),
        "next_scheduled": next_run(now),
        "coverage": {"stocks": len((snap or {}).get("stocks") or [])},
        "warnings": why,
        "datasets": ds,
        "demo": (snap or {}).get("source") == "demo",
        "note": "只有盤後資料；網站不提供盤中即時行情。發布後 GitHub Pages 約需幾分鐘才會生效。",
    }
