"""FinMind（https://finmind.github.io/）逐檔資料：月營收、財報三表、股利、新聞、股票清單。

這些資料更新頻率低（月、季、年），所以只需要偶爾逐檔補抓。
免費註冊取得 token 後每小時約 600 次請求。
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import http
from .. import config

API_URL = "https://api.finmindtrade.com/api/v4/data"

# 財報科目名稱（依序嘗試，第一個找到的為準）
INCOME = {"revenue": ["Revenue", "OperatingRevenue"], "gross_profit": ["GrossProfit"],
          "operating_income": ["OperatingIncome"],
          "net_income": ["EquityAttributableToOwnersOfParent", "IncomeAfterTaxes", "IncomeAfterTax"],
          "eps": ["EPS"]}
BALANCE = {"total_assets": ["TotalAssets"], "total_liabilities": ["Liabilities", "TotalLiabilities"],
           "equity": ["EquityAttributableToOwnersOfParent", "Equity", "TotalEquity"],
           "current_assets": ["CurrentAssets"], "current_liabilities": ["CurrentLiabilities"],
           "share_capital": ["OrdinaryShare", "CapitalStock", "ShareCapital"]}
CASHFLOW = {"ocf": ["CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities"],
            "capex": ["PropertyAndPlantAndEquipment", "AcquisitionOfPropertyPlantAndEquipment"]}


class FinMindError(RuntimeError):
    pass


def _period(d: pd.Timestamp) -> str:
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


class FinMindSource:
    def __init__(self, token: str = "", interval: float = 0.7):
        self.token = token
        self.interval = interval
        self._last = 0.0
        self.s = requests.Session()

    def _get(self, dataset: str, data_id: str | None = None, start: str | None = None, end: str | None = None):
        wait = self.interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = start
        if end:
            params["end_date"] = end
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = http.get(self.s, API_URL, params=params, headers=headers, timeout=60)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 402:
                raise FinMindError("FinMind 本小時請求次數已用完") from e
            raise
        finally:
            self._last = time.time()
        js = r.json()
        if js.get("status") not in (200, None) or "data" not in js:
            raise FinMindError(f"{dataset} {data_id}: {js.get('msg')}")
        return js["data"]

    @staticmethod
    def _since(days: int) -> str:
        return (config.today() - timedelta(days=days)).isoformat()

    def stock_info(self) -> list[dict]:
        out = {}
        for r in self._get("TaiwanStockInfo"):
            out[r["stock_id"]] = {"stock_id": r["stock_id"], "name": r["stock_name"],
                                  "market": r.get("type", ""), "industry": r.get("industry_category", "")}
        return list(out.values())

    def prices(self, sid: str, days: int) -> list[dict]:
        rows = self._get("TaiwanStockPrice", sid, self._since(days))
        return [{"stock_id": sid, "date": r["date"], "open": r["open"], "high": r["max"], "low": r["min"],
                 "close": r["close"], "volume": r["Trading_Volume"], "value": r["Trading_money"]}
                for r in rows if r.get("Trading_Volume")]

    def month_revenue(self, sid: str, days: int = 1200) -> list[dict]:
        rows = self._get("TaiwanStockMonthRevenue", sid, self._since(days))
        return [{"stock_id": sid, "ym": f"{int(r['revenue_year'])}-{int(r['revenue_month']):02d}", "revenue": r["revenue"]}
                for r in rows]

    def _statement(self, dataset: str, sid: str, mapping: dict, days: int) -> pd.DataFrame:
        df = pd.DataFrame(self._get(dataset, sid, self._since(days)))
        if df.empty:
            return pd.DataFrame()
        df = df[~df["type"].str.endswith("_per")]
        wide = df.pivot_table(index="date", columns="type", values="value", aggfunc="last")
        out = pd.DataFrame(index=wide.index)
        for col, names in mapping.items():
            hit = next((n for n in names if n in wide.columns), None)
            out[col] = wide[hit] if hit else None
        out.index = pd.to_datetime(out.index)
        return out

    def financials(self, sid: str, days: int = 1500) -> list[dict]:
        inc = self._statement("TaiwanStockFinancialStatements", sid, INCOME, days)
        bal = self._statement("TaiwanStockBalanceSheet", sid, BALANCE, days)
        cf = self._statement("TaiwanStockCashFlowsStatement", sid, CASHFLOW, days)
        if not cf.empty:
            # 現金流量表是「年初至今累計」，換算成單季
            cf = cf.sort_index()
            cf["capex"] = cf["capex"].abs() if cf["capex"].notna().any() else cf["capex"]
            q = cf.copy()
            for y, g in cf.groupby(cf.index.year):
                prev = None
                for idx in g.index:
                    if prev is not None and idx.month - prev.month == 3:
                        q.loc[idx] = cf.loc[idx] - cf.loc[prev]
                    elif idx.month != 3:
                        q.loc[idx] = None  # 缺前一季，無法換算
                    prev = idx
            cf = q
        df = inc.join(bal, how="outer").join(cf, how="outer")
        rows = []
        for d, r in df.iterrows():
            row = {"stock_id": sid, "period": _period(d), "period_end": d.strftime("%Y-%m-%d")}
            row.update({k: (None if pd.isna(v) else float(v)) for k, v in r.items()})
            rows.append(row)
        return rows

    def dividends(self, sid: str) -> list[dict]:
        rows = self._get("TaiwanStockDividend", sid, self._since(365 * 12))
        agg: dict[int, dict] = {}
        for r in rows:
            m = re.search(r"\d+", str(r.get("year", "")))
            if not m:
                continue
            y = int(m.group())
            y = y + 1911 if y < 1911 else y
            a = agg.setdefault(y, {"stock_id": sid, "year": y, "cash": 0.0, "stock": 0.0})
            a["cash"] += float(r.get("CashEarningsDistribution") or 0) + float(r.get("CashStatutorySurplus") or 0)
            a["stock"] += float(r.get("StockEarningsDistribution") or 0) + float(r.get("StockStatutorySurplus") or 0)
        return list(agg.values())

    def news(self, sid: str, days: int = 14) -> list[dict]:
        rows = self._get("TaiwanStockNews", sid, self._since(days), config.today().isoformat())
        return [{"stock_id": sid, "date": str(r.get("date", ""))[:10], "title": (r.get("title") or "").strip(),
                 "source": r.get("source", ""), "link": r.get("link", "")} for r in rows if r.get("title")]
