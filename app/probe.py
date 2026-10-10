"""連線診斷：python -m app.probe → data/probe.json（GitHub Actions 會一起發布到網站，方便遠端查看）。

逐一呼叫各資料端點，記錄 HTTP 狀態、內容類型、是否為 JSON、欄位名稱與第一列資料。
"""
import json
from datetime import timedelta

import requests

from . import config
from .sources.twse import HEADERS

TPEX = "https://www.tpex.org.tw"


def _probe(url: str, params: dict) -> dict:
    out = {"url": url, "params": params}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        out.update(status=r.status_code, type=r.headers.get("content-type", ""), head=r.text[:300])
        try:
            js = r.json()
            out["json_keys"] = list(js)[:15] if isinstance(js, dict) else f"list[{len(js)}]"
            t = (js.get("tables") or [{}])[0] if isinstance(js, dict) else {}
            out["fields"] = t.get("fields") or (list(js[0]) if isinstance(js, list) and js else None)
            rows = t.get("data") or (js.get("aaData") if isinstance(js, dict) else None) or (js if isinstance(js, list) else [])
            out["rows"] = len(rows)
            out["first_row"] = rows[0] if rows else None
        except ValueError:
            out["json"] = False
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def main():
    d = config.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    slash, roc = d.strftime("%Y/%m/%d"), f"{d.year - 1911}/{d:%m/%d}"
    probes = [
        _probe(f"{TPEX}/www/zh-tw/afterTrading/dailyQuotes", {"date": slash, "response": "json"}),
        _probe(f"{TPEX}/www/zh-tw/afterTrading/otc", {"date": slash, "response": "json"}),
        _probe(f"{TPEX}/www/zh-tw/afterTrading/dailyQ", {"date": slash, "response": "json"}),
        _probe(f"{TPEX}/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php", {"l": "zh-tw", "o": "json", "d": roc}),
        _probe(f"{TPEX}/openapi/v1/tpex_mainboard_daily_close_quotes", {}),
        _probe(f"{TPEX}/www/zh-tw/bulletin/exDailyQ", {"startDate": "2026/07/01", "endDate": "2026/07/31", "response": "json"}),
        _probe(f"{TPEX}/web/stock/exright/dailyquo/exDailyQ_result.php", {"l": "zh-tw", "o": "json", "d": "115/07/01", "ed": "115/07/31"}),
        _probe("https://www.twse.com.tw/rwd/zh/exRight/TWT49U", {"startDate": "20260701", "endDate": "20260731", "response": "json"}),
    ]
    for p in probes:  # OpenAPI 很大，只留摘要
        p["head"] = p.get("head", "")[:300]
    (config.DATA_DIR / "probe.json").write_text(json.dumps({"date": str(d), "probes": probes}, ensure_ascii=False, indent=1), "utf-8")
    for p in probes:
        print(p["url"], p.get("status"), p.get("rows"), p.get("error", ""))


if __name__ == "__main__":
    main()
