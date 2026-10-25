"""行情供應商抽象層（V12，尚未接上真實盤中行情）。

三種來源用同一個介面，接收程式（receiver.py）不需要知道是哪一家：
- OfficialCloseProvider：既有盤後資料（資料庫最後收盤價），標示「盤後」。
- SimulatedProvider：模擬行情，只用來測試架構，每筆都標示 simulated，不能當真實價格顯示。
- YuantaSparkProvider：元大 SPARK API 的接口位置。目前沒有權限，也還沒確認網站展示與再散布授權，
  所以只做設定檢查：帳號與憑證只能從伺服端環境變數讀取，SDK 只在真正連線時才載入，不會進前端。
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .. import config


@dataclass(frozen=True)
class Quote:
    sid: str
    price: float
    prev_close: float | None
    volume: int | None
    ts: str                 # 行情時間（交易所時間，不是收到的時間）
    source: str             # official_close / simulated / yuanta
    kind: str               # official（盤後正式）/ intraday（盤中暫時）/ simulated（模擬）

    @property
    def change_pct(self) -> float | None:
        return None if not self.prev_close else round((self.price / self.prev_close - 1) * 100, 2)

    def to_dict(self) -> dict:
        return {**asdict(self), "change_pct": self.change_pct}


class ProviderError(Exception):
    """連線中斷、登入失敗等可以重試的錯誤。"""


class NotConfigured(Exception):
    """沒有權限或設定，不能使用這個來源（不重試）。"""


@dataclass
class Provider:
    name: str = "base"
    kind: str = "intraday"
    max_subscriptions: int = 200
    subscribed: set = field(default_factory=set)
    connected: bool = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def subscribe(self, ids: list[str]) -> list[str]:
        """回傳這次真的新增的代號；超過上限的部分不訂閱。"""
        new = [s for s in dict.fromkeys(ids) if s not in self.subscribed]
        room = max(0, self.max_subscriptions - len(self.subscribed))
        self.subscribed.update(new[:room])
        return new[:room]

    def unsubscribe(self, ids: list[str]) -> None:
        self.subscribed.difference_update(ids)

    def poll(self) -> list[Quote]:
        raise NotImplementedError


class OfficialCloseProvider(Provider):
    """盤後正式資料：資料庫裡每檔股票最後一天的收盤價。"""

    def __init__(self, dbname: str = "stocks.db", **kw):
        super().__init__(name="official_close", kind="official", max_subscriptions=10_000, **kw)
        self.dbname = dbname

    def poll(self) -> list[Quote]:
        from .. import db
        if not self.subscribed:
            return []
        q = ",".join("?" * len(self.subscribed))
        with db.connect(self.dbname) as con:
            rows = con.execute(f"""SELECT p.stock_id, p.date, p.close, p.volume,
                       (SELECT close FROM prices q WHERE q.stock_id = p.stock_id AND q.date < p.date ORDER BY date DESC LIMIT 1)
                FROM prices p WHERE p.stock_id IN ({q}) AND p.date = (SELECT MAX(date) FROM prices r WHERE r.stock_id = p.stock_id)""",
                               sorted(self.subscribed)).fetchall()
        return [Quote(s, float(c), float(pc) if pc else None, int(v) if v is not None else None, f"{d}T13:30:00", self.name, self.kind)
                for s, d, c, v, pc in rows if c]


class SimulatedProvider(Provider):
    """模擬行情：從給定的參考價隨機漫步。可以設定第幾次 poll 斷線，用來測試重連。"""

    def __init__(self, base: dict[str, float] | None = None, seed: int = 7, fail_at: set[int] | None = None,
                 fail_connect: set[int] | None = None, clock=None, **kw):
        super().__init__(name="simulated", kind="simulated", **kw)
        self.base = dict(base or {})
        self.last = dict(self.base)
        self.rng = random.Random(seed)
        self.fail_at = set(fail_at or ())          # 第幾次 poll 斷線
        self.fail_connect = set(fail_connect or ())  # 第幾次連線失敗
        self.polls = self.connects = 0
        self.clock = clock or config.now

    def connect(self) -> None:
        self.connects += 1
        if self.connects in self.fail_connect:
            raise ProviderError("模擬：連線失敗")
        super().connect()

    def poll(self) -> list[Quote]:
        self.polls += 1
        if self.polls in self.fail_at:
            self.connected = False
            raise ProviderError("模擬：連線中斷")
        if not self.connected:
            raise ProviderError("尚未連線")
        ts = self.clock().isoformat(timespec="seconds")
        out = []
        for s in sorted(self.subscribed):
            p0 = self.base.setdefault(s, 100.0)
            p = round(max(0.01, self.last.get(s, p0) * (1 + self.rng.gauss(0, 0.002))), 2)
            self.last[s] = p
            out.append(Quote(s, p, p0, self.rng.randint(1, 500), ts, self.name, self.kind))
        return out


class YuantaSparkProvider(Provider):
    """元大 SPARK API（尚未啟用）。

    需要的伺服端環境變數（不能放進 GitHub、不能放進網站）：YUANTA_ID、YUANTA_PASSWORD、YUANTA_CERT_PATH、YUANTA_CERT_PASSWORD，
    以及確認過授權後才設定的 YUANTA_DISPLAY_LICENSED=1。SDK 的實際呼叫要等取得權限、拿到官方文件後再實作。
    """
    ENV = ("YUANTA_ID", "YUANTA_PASSWORD", "YUANTA_CERT_PATH", "YUANTA_CERT_PASSWORD")

    def __init__(self, **kw):
        super().__init__(name="yuanta", kind="intraday", max_subscriptions=int(os.environ.get("YUANTA_MAX_SUBS", "200")), **kw)
        self._secrets = {k: os.environ.get(k) for k in self.ENV}

    def __repr__(self) -> str:  # 不印出任何帳號或憑證
        return f"YuantaSparkProvider(configured={all(self._secrets.values())}, subscribed={len(self.subscribed)})"

    def check(self) -> list[str]:
        miss = [k for k, v in self._secrets.items() if not v]
        out = [f"沒有設定 {', '.join(miss)}"] if miss else []
        if os.environ.get("YUANTA_DISPLAY_LICENSED") != "1":
            out.append("還沒確認元大行情可以在網站公開展示（確認後設定 YUANTA_DISPLAY_LICENSED=1）")
        return out

    def connect(self) -> None:
        bad = self.check()
        if bad:
            raise NotConfigured("；".join(bad))
        raise NotConfigured("元大 SPARK SDK 串接尚未實作：取得權限與官方文件後再完成（見 docs/realtime.md）")

    def poll(self) -> list[Quote]:
        raise NotConfigured("元大 SPARK API 尚未啟用")


def make(name: str, **kw) -> Provider:
    return {"official_close": OfficialCloseProvider, "simulated": SimulatedProvider, "yuanta": YuantaSparkProvider}[name](**kw)


def parse_ts(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=config.now().tzinfo)
