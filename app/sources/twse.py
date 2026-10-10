"""臺灣證券交易所（上市）每日全市場批次資料。

一天只需要 4 次請求就能拿到全部上市股票的行情、三大法人、融資融券、本益比。
欄位名稱已於 2026-10-07 的實際回應核對過。證交所有流量限制，請求間隔請保持 3 秒以上。
"""
from __future__ import annotations

import re
import time
from datetime import date

import requests

from . import http
from .. import config

BASE = "https://www.twse.com.tw"
HEADERS = {"User-Agent": "Mozilla/5.0 (tw-stock-radar)"}


def num(x):
    """'1,234.5' → 1234.5；'--'、'-'、'' → None"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"<[^>]+>", "", str(x)).replace(",", "").strip()
    if s in ("", "-", "--", "---", "X", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


class TwseSource:
    market = "twse"

    def __init__(self, interval: float = 3.5):
        self.interval = interval
        self._last = 0.0
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def _get(self, path: str, params: dict) -> dict:
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            return http.get(self.s, BASE + path, params=params).json()
        finally:
            self._last = time.time()

    @staticmethod
    def _ok(js: dict) -> bool:
        return js.get("stat") == "OK"

    def daily_quotes(self, d: date) -> tuple[list[dict], float | None]:
        """回傳 (個股行情, 加權指數收盤)。休市日回傳 ([], None)。"""
        js = self._get("/rwd/zh/afterTrading/MI_INDEX",
                       {"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"})
        if not self._ok(js):
            return [], None
        rows, taiex = [], None
        for t in js.get("tables", []):
            f = t.get("fields") or []
            if "證券代號" in f and "收盤價" in f:
                i = {k: f.index(k) for k in ("證券代號", "證券名稱", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價")}
                for r in t.get("data", []):
                    close = num(r[i["收盤價"]])
                    if close is None:
                        continue  # 當日無成交
                    rows.append({"stock_id": r[i["證券代號"]].strip(), "name": r[i["證券名稱"]].strip(),
                                 "open": num(r[i["開盤價"]]), "high": num(r[i["最高價"]]), "low": num(r[i["最低價"]]),
                                 "close": close, "volume": num(r[i["成交股數"]]), "value": num(r[i["成交金額"]])})
            elif "價格指數(臺灣證券交易所)" in (t.get("title") or ""):
                for r in t.get("data", []):
                    if r and str(r[0]).strip() == "發行量加權股價指數":
                        taiex = num(r[1])
        return rows, taiex

    def institutional(self, d: date) -> list[dict]:
        js = self._get("/rwd/zh/fund/T86", {"date": d.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"})
        if not self._ok(js):
            return []
        f = js["fields"]
        ix = lambda name: f.index(name)
        fi, fdi = ix("外陸資買賣超股數(不含外資自營商)"), ix("外資自營商買賣超股數")
        ti, di = ix("投信買賣超股數"), ix("自營商買賣超股數")
        return [{"stock_id": r[0].strip(), "foreign_net": (num(r[fi]) or 0) + (num(r[fdi]) or 0),
                 "trust_net": num(r[ti]) or 0, "dealer_net": num(r[di]) or 0} for r in js.get("data", [])]

    def margin(self, d: date) -> list[dict]:
        """融資、融券今日餘額（單位：張）。"""
        js = self._get("/rwd/zh/marginTrading/MI_MARGN", {"date": d.strftime("%Y%m%d"), "selectType": "STOCK", "response": "json"})
        if not self._ok(js):
            return []
        out = []
        for t in js.get("tables", []):
            f = t.get("fields") or []
            if not f or f[0] != "代號":
                continue
            bal = [i for i, x in enumerate(f) if x == "今日餘額"]
            if len(bal) < 2:
                continue
            for r in t.get("data", []):
                sid = str(r[0]).strip()
                if sid and sid != "合計":
                    out.append({"stock_id": sid, "margin_bal": num(r[bal[0]]), "short_bal": num(r[bal[1]])})
        return out

    def valuation(self, d: date) -> list[dict]:
        js = self._get("/exchangeReport/BWIBBU_d", {"date": d.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"})
        if not self._ok(js):
            return []
        f = js["fields"]
        i = {k: f.index(k) for k in ("證券代號", "殖利率(%)", "本益比", "股價淨值比")}
        return [{"stock_id": r[i["證券代號"]].strip(), "per": num(r[i["本益比"]]), "pbr": num(r[i["股價淨值比"]]),
                 "dy": num(r[i["殖利率(%)"]])} for r in js.get("data", [])]

    def ex_rights(self, start: date, end: date) -> list[dict]:
        """除權息計算結果表 TWT49U：除權息日、前一日收盤、除權息參考價（2026-10-10 核對過欄位）。"""
        js = self._get("/rwd/zh/exRight/TWT49U", {"startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d"),
                                                  "response": "json"})
        if not self._ok(js):
            return []
        return parse_ex_rights(js.get("fields") or [], js.get("data") or [])

    def disposition(self) -> list[dict]:
        """目前處置中的股票（OpenAPI）。"""
        r = http.get(self.s, "https://openapi.twse.com.tw/v1/announcement/punish")
        today = config.today()
        out = []
        for x in r.json():
            m = re.findall(r"(\d+)/(\d+)/(\d+)", x.get("DispositionPeriod", ""))
            if len(m) == 2:
                y, mo, dd = map(int, m[1])
                if date(y + 1911, mo, dd) < today:
                    continue
            out.append({"stock_id": x["Code"].strip(), "period": x.get("DispositionPeriod", ""),
                        "reason": x.get("ReasonsOfDisposition", "")})
        return out


def any_date(s) -> str | None:
    """'115年07月01日'、'115/07/01'、'1150701'、'2026-07-01' → '2026-07-01'"""
    d = re.findall(r"\d+", str(s or ""))
    if len(d) == 1 and len(d[0]) in (7, 8):
        d = [d[0][:-4], d[0][-4:-2], d[0][-2:]]
    if len(d) < 3:
        return None
    y = int(d[0]) + (1911 if int(d[0]) < 1000 else 0)
    return f"{y}-{int(d[1]):02d}-{int(d[2]):02d}"


def parse_ex_rights(fields: list, data: list) -> list[dict]:
    """依欄位名稱找出：日期、代號、除權息前收盤價、除權息參考價（上市、上櫃共用）。"""
    f = [str(x).replace(" ", "") for x in fields]
    def col(*keys, exclude=()):
        for i, x in enumerate(f):
            if all(k in x for k in keys) and not any(e in x for e in exclude):
                return i
        raise KeyError(f"除權息欄位找不到 {keys}，實際欄位：{fields}")
    di, si = col("日期"), col("代號")
    pi, ri = col("前收盤"), col("參考價", exclude=("減除", "開盤"))
    out = []
    for r in data:
        day, prev, ref = any_date(r[di]), num(r[pi]), num(r[ri])
        if day and prev and ref and 0 < ref <= prev:
            out.append({"stock_id": str(r[si]).strip(), "date": day, "prev_close": prev, "ref_price": ref})
    return out


# ---------- 證交所 OpenAPI（備援） ----------
# openapi.twse.com.tw 只提供「最近一個交易日」，不能回補歷史，所以當主要來源失敗時才用。
# 欄位已於 2026-10-08 實際回應核對：Date 是民國年 1151006 這種格式。
OPENAPI = "https://openapi.twse.com.tw/v1"


def roc_date(s) -> str | None:
    s = str(s or "").strip()
    if len(s) == 7 and s.isdigit():
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:]}"
    return None


def openapi_quotes(session: requests.Session | None = None) -> tuple[str | None, list[dict]]:
    """STOCK_DAY_ALL：上市個股最近一日成交資訊。回傳 (資料日期, 行情)。"""
    s = session or requests.Session()
    r = http.get(s, f"{OPENAPI}/exchangeReport/STOCK_DAY_ALL", headers=HEADERS, timeout=60)
    rows, day = [], None
    for x in r.json():
        close = num(x.get("ClosingPrice"))
        if close is None:
            continue
        day = day or roc_date(x.get("Date"))
        rows.append({"stock_id": str(x["Code"]).strip(), "name": str(x.get("Name", "")).strip(),
                     "open": num(x.get("OpeningPrice")), "high": num(x.get("HighestPrice")), "low": num(x.get("LowestPrice")),
                     "close": close, "volume": num(x.get("TradeVolume")), "value": num(x.get("TradeValue"))})
    return day, rows


def openapi_valuation(session: requests.Session | None = None) -> tuple[str | None, list[dict]]:
    """BWIBBU_ALL：上市個股最近一日本益比、殖利率、股價淨值比。"""
    s = session or requests.Session()
    r = http.get(s, f"{OPENAPI}/exchangeReport/BWIBBU_ALL", headers=HEADERS, timeout=60)
    data = r.json()
    day = next((roc_date(x.get("Date")) for x in data if x.get("Date")), None)
    return day, [{"stock_id": str(x["Code"]).strip(), "per": num(x.get("PEratio")), "pbr": num(x.get("PBratio")),
                  "dy": num(x.get("DividendYield"))} for x in data]
