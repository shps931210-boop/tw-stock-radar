"""櫃買中心（上櫃）每日全市場批次資料。

注意：櫃買中心網站擋掉了開發環境的連線，這支程式的欄位對應「尚未用真實回應核對」。
解析時用欄位名稱關鍵字比對，對不上會丟出清楚的錯誤訊息，更新程式會記錄下來並略過，不影響上市資料。
"""
from __future__ import annotations

import time
from datetime import date

import requests

from . import http
from .twse import HEADERS, num

BASE = "https://www.tpex.org.tw/www/zh-tw"


def _col(fields: list[str], *keywords: str, exclude: tuple[str, ...] = ()) -> int:
    for i, f in enumerate(fields):
        name = str(f).replace(" ", "")
        if all(k in name for k in keywords) and not any(e in name for e in exclude):
            return i
    raise KeyError(f"櫃買中心欄位找不到 {keywords}，實際欄位：{fields}")


class TpexSource:
    market = "tpex"

    def __init__(self, interval: float = 3.5):
        self.interval = interval
        self._last = 0.0
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def _table(self, path: str, d: date, extra: dict | None = None) -> tuple[list[str], list[list]]:
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"date": d.strftime("%Y/%m/%d"), "response": "json", **(extra or {})}
        try:
            js = http.get(self.s, f"{BASE}{path}", params=params).json()
        finally:
            self._last = time.time()
        tables = js.get("tables") or []
        if not tables or not tables[0].get("data"):
            return [], []
        return tables[0]["fields"], tables[0]["data"]

    def daily_quotes(self, d: date) -> tuple[list[dict], float | None]:
        f, data = self._table("/afterTrading/dailyQ", d)
        if not data:
            return [], None
        c = {k: _col(f, k) for k in ("代號", "名稱", "收盤", "開盤", "最高", "最低")}
        c["vol"] = _col(f, "成交股數")
        c["val"] = _col(f, "成交金額")
        rows = []
        for r in data:
            close = num(r[c["收盤"]])
            sid = str(r[c["代號"]]).strip()
            if close is None or not sid:
                continue
            rows.append({"stock_id": sid, "name": str(r[c["名稱"]]).strip(), "open": num(r[c["開盤"]]),
                         "high": num(r[c["最高"]]), "low": num(r[c["最低"]]), "close": close,
                         "volume": num(r[c["vol"]]), "value": num(r[c["val"]])})
        return rows, None

    def institutional(self, d: date) -> list[dict]:
        f, data = self._table("/insti/dailyTrade", d, {"type": "Daily", "sect": "EW"})
        if not data:
            return []
        sid = _col(f, "代號")
        fi = _col(f, "外資", "買賣超", exclude=("自營商",))
        ti = _col(f, "投信", "買賣超")
        di = _col(f, "自營商", "買賣超")
        return [{"stock_id": str(r[sid]).strip(), "foreign_net": num(r[fi]) or 0, "trust_net": num(r[ti]) or 0,
                 "dealer_net": num(r[di]) or 0} for r in data]

    def margin(self, d: date) -> list[dict]:
        f, data = self._table("/margin/balance", d)
        if not data:
            return []
        sid = _col(f, "代號")
        bal = [i for i, x in enumerate(f) if "餘額" in str(x) and "前日" not in str(x)]
        if len(bal) < 2:
            raise KeyError(f"櫃買融資融券欄位無法辨識：{f}")
        return [{"stock_id": str(r[sid]).strip(), "margin_bal": num(r[bal[0]]), "short_bal": num(r[bal[1]])}
                for r in data]

    def valuation(self, d: date) -> list[dict]:
        f, data = self._table("/afterTrading/peQryDate", d)
        if not data:
            return []
        sid, pe, pb, dy = _col(f, "代號"), _col(f, "本益比"), _col(f, "淨值比"), _col(f, "殖利率")
        return [{"stock_id": str(r[sid]).strip(), "per": num(r[pe]), "pbr": num(r[pb]), "dy": num(r[dy])}
                for r in data]


# ---------- 櫃買中心 OpenAPI（備援，欄位尚未核對） ----------
OPENAPI = "https://www.tpex.org.tw/openapi/v1"


def _pick(x: dict, *names):
    for n in names:
        if n in x:
            return x[n]
    raise KeyError(f"櫃買 OpenAPI 欄位找不到 {names}，實際欄位：{list(x)}")


def openapi_quotes(session: requests.Session | None = None) -> tuple[str | None, list[dict]]:
    """上櫃個股最近一日收盤行情。欄位名稱依官方文件撰寫，對不上會丟出 KeyError 並列出實際欄位。"""
    from .twse import roc_date
    s = session or requests.Session()
    r = http.get(s, f"{OPENAPI}/tpex_mainboard_daily_close_quotes", headers=HEADERS, timeout=60)
    rows, day = [], None
    for x in r.json():
        close = num(_pick(x, "Close", "ClosingPrice"))
        if close is None:
            continue
        day = day or roc_date(x.get("Date"))
        rows.append({"stock_id": str(_pick(x, "SecuritiesCompanyCode", "Code")).strip(),
                     "name": str(_pick(x, "CompanyName", "Name")).strip(),
                     "open": num(_pick(x, "Open", "OpeningPrice")), "high": num(_pick(x, "High", "HighestPrice")),
                     "low": num(_pick(x, "Low", "LowestPrice")), "close": close,
                     "volume": num(_pick(x, "TradingShares", "TradeVolume")),
                     "value": num(_pick(x, "TransactionAmount", "TradeValue"))})
    return day, rows
