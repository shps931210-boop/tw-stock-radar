"""券商介面（第一階段：只有模擬，真實下單永久關閉）。

設計：所有下單都經過 Broker 介面。目前只有兩種實作——
- PaperBroker：模擬撮合，結果寫進模擬帳本（實際的模擬規則在 app/paper.py）。
- YuantaSparkBroker：元大 SPARK API 的「預留位置」。沒有任何連線或送單程式碼，任何呼叫都會丟出 LiveTradingDisabled。

安全規則（測試會檢查）
- LIVE_TRADING 固定是 False，程式碼裡沒有任何開關可以從環境變數、設定檔或網站打開它。
- 這個專案不存放、不讀取任何券商帳號、密碼、憑證或 API 金鑰；網站也不會要求使用者提供。
  依證交所提醒，帳號、密碼和憑證不應交給未經核准的第三方。
- 之後如果要做第二階段（人工確認後才送單），必須另外開發、另外審查，並由使用者本人在券商官方環境設定憑證；
  不能只改這個檔案的一個布林值就打開。
"""
from __future__ import annotations

from dataclasses import dataclass

LIVE_TRADING = False


class LiveTradingDisabled(RuntimeError):
    """真實下單在第一階段永久關閉。"""


@dataclass(frozen=True)
class Order:
    stock_id: str
    side: str          # "buy" / "sell"
    shares: int
    limit: float | None = None
    note: str = ""


class Broker:
    name = "base"
    live = False

    def place(self, order: Order) -> dict:
        raise NotImplementedError


class PaperBroker(Broker):
    """模擬券商：只把委託記下來，實際撮合由 app/paper.py 依固定規則用之後的真實行情計算。"""
    name = "paper"
    live = False

    def __init__(self):
        self.orders: list[Order] = []

    def place(self, order: Order) -> dict:
        if order.side not in ("buy", "sell") or order.shares <= 0:
            raise ValueError("委託內容不正確")
        self.orders.append(order)
        return {"broker": self.name, "live": False, "status": "simulated", "order": order.__dict__}


class YuantaSparkBroker(Broker):
    """元大 SPARK API 預留位置。第一階段不連線、不登入、不送單，也不接受任何憑證參數。"""
    name = "yuanta_spark"
    live = True

    def __init__(self, *_, **__):
        raise LiveTradingDisabled("第一階段只做模擬交易：真實下單永久關閉，不接受帳號、密碼或憑證")

    def place(self, order: Order) -> dict:  # pragma: no cover  建構時就會擋下
        raise LiveTradingDisabled("第一階段只做模擬交易：真實下單永久關閉")


def get_broker(kind: str = "paper") -> Broker:
    """唯一取得券商物件的入口。除了 paper 以外一律拒絕。"""
    if kind != "paper" or LIVE_TRADING:
        raise LiveTradingDisabled("第一階段只做模擬交易：真實下單永久關閉")
    return PaperBroker()
