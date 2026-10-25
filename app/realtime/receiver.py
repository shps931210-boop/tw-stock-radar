"""盤中行情接收程式（V12）：python -m app.realtime.receiver --provider simulated --ids 2330,0050 --seconds 30

常駐在一台盤中開著的電腦或伺服器上（GitHub Pages、Actions 不能常駐）。負責：
訂閱管理（去重、數量上限、優先順序）、心跳檢查、斷線自動重連（指數退避）、重複行情過濾、
錯誤紀錄、Ctrl+C／系統關閉時安全結束，最後把最新行情寫到 data/intraday/quotes.json（不寫資料庫）。
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from .. import config
from . import providers as pv
from . import store

PRIORITY = ("page", "watchlist", "ai_picks", "popular", "other")   # 訂閱額度不夠時，先保留前面的


class Receiver:
    def __init__(self, provider: pv.Provider, interval: float = 5.0, heartbeat: float = 30.0, max_backoff: float = 60.0,
                 out=store.FILE, sleep=time.sleep, clock=time.monotonic, log=print):
        self.p, self.interval, self.heartbeat, self.max_backoff = provider, interval, heartbeat, max_backoff
        self.out, self.sleep, self.clock, self.log = out, sleep, clock, log
        self.quotes: dict[str, pv.Quote] = {}
        self.seen: set[tuple] = set()
        self.wanted: dict[str, str] = {}
        self.errors: list[dict] = []
        self.reconnects = 0
        self.duplicates = 0
        self.last_msg: float | None = None
        self.running = False
        self.deadline: float | None = None

    # ---- 訂閱 ----
    def want(self, ids: list[str], why: str = "other") -> None:
        for s in ids:
            if s not in self.wanted or PRIORITY.index(why) < PRIORITY.index(self.wanted[s]):
                self.wanted[s] = why

    def plan(self) -> list[str]:
        """依優先順序排好、在數量上限內要訂閱的代號。"""
        order = sorted(self.wanted, key=lambda s: (PRIORITY.index(self.wanted[s]), s))
        return order[: self.p.max_subscriptions]

    def sync(self) -> None:
        keep = set(self.plan())
        drop = [s for s in self.p.subscribed if s not in keep]
        if drop:
            self.p.unsubscribe(drop)
        self.p.subscribe([s for s in self.plan() if s not in self.p.subscribed])

    # ---- 連線 ----
    def _error(self, e: Exception) -> None:
        self.errors.append({"at": config.now().isoformat(timespec="seconds"), "error": f"{type(e).__name__}: {e}"})
        self.errors = self.errors[-50:]
        self.log(f"行情錯誤：{e}")

    def connect(self) -> bool:
        """連線並重新訂閱。可重試的錯誤用指數退避重試（只在常駐執行時），沒有權限就直接放棄。"""
        wait = 1.0
        while True:
            try:
                self.p.connect()
                self.p.subscribed.clear()
                self.sync()
                self.last_msg = self.clock()
                return True
            except pv.NotConfigured as e:   # 沒有權限或設定：重試也沒用，停止接收
                self._error(e)
                self.running = False
                return False
            except pv.ProviderError as e:
                self._error(e)
                self.reconnects += 1
                if not self.running or (self.deadline is not None and self.clock() >= self.deadline):
                    return False
                self.sleep(wait)
                wait = min(wait * 2, self.max_backoff)

    def step(self) -> int:
        """收一次行情。回傳新增（非重複）的筆數。"""
        if not self.p.connected and not self.connect():
            return 0
        if self.last_msg is not None and self.clock() - self.last_msg > self.heartbeat:
            self._error(pv.ProviderError(f"超過 {self.heartbeat:.0f} 秒沒有行情，重新連線"))
            self.p.close()
            self.reconnects += 1
            return 0
        try:
            qs = self.p.poll()
        except pv.ProviderError as e:
            self._error(e)
            self.p.close()
            self.reconnects += 1
            return 0
        n = 0
        for q in qs:
            key = (q.sid, q.ts, q.price, q.volume)
            if key in self.seen:
                self.duplicates += 1
                continue
            self.seen.add(key)
            self.quotes[q.sid] = q
            n += 1
        if qs:
            self.last_msg = self.clock()
        if len(self.seen) > 100_000:
            self.seen.clear()
        self.save()
        return n

    def status(self) -> dict:
        return {"provider": self.p.name, "kind": self.p.kind, "connected": self.p.connected, "subscribed": len(self.p.subscribed),
                "limit": self.p.max_subscriptions, "reconnects": self.reconnects, "duplicates": self.duplicates,
                "errors": self.errors[-5:], "simulated": self.p.kind == "simulated"}

    def save(self) -> None:
        store.write(self.quotes, self.status(), self.out)

    def stop(self, *_):
        self.running = False

    def run(self, seconds: float | None = None) -> None:
        self.running = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, self.stop)
            except ValueError:   # 不在主執行緒（測試）
                pass
        end = self.deadline = None if seconds is None else self.clock() + seconds
        while self.running and (end is None or self.clock() < end):
            self.step()
            self.sleep(self.interval)
        self.p.close()
        self.save()   # 關閉後寫入 connected=False，網站會退回盤後價格
        self.log("行情接收程式已安全結束")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="simulated", choices=["simulated", "official_close", "yuanta"])
    ap.add_argument("--ids", default="2330,0050")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--interval", type=float, default=5)
    a = ap.parse_args(argv)
    ids = [s for s in a.ids.split(",") if s]
    kw = {}
    if a.provider == "simulated":
        base = {q.sid: q.price for q in _official(ids)}
        kw = {"base": base}
    r = Receiver(pv.make(a.provider, **kw), interval=a.interval)
    r.want(ids, "page")
    r.run(a.seconds)
    return 0


def _official(ids: list[str]) -> list[pv.Quote]:
    try:
        p = pv.OfficialCloseProvider()
        p.subscribe(ids)
        return p.poll()
    except Exception:  # noqa: BLE001  沒有資料庫時模擬行情從 100 元開始
        return []


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
