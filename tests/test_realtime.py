"""V12 盤中行情架構：模擬行情連續更新、斷線重連、過期退回盤後、多人同時讀、來源切換、盤中與盤後隔離、憑證不外洩。"""
import json
import os
import re
import sqlite3
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path

from app import config
from app.realtime import providers as pv
from app.realtime import receiver as rc
from app.realtime import store

ROOT = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(provider, tmp, **kw):
    c = Clock()
    return rc.Receiver(provider, out=tmp / "q.json", sleep=c.sleep, clock=c, log=lambda *a: None, **kw), c


class Simulated(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_continuous_updates_marked_simulated(self):
        r, _ = make(pv.SimulatedProvider({"2330": 1000.0, "0050": 180.0}), self.tmp)
        r.want(["2330", "0050"], "page")
        r.run(seconds=30)
        doc = store.read(self.tmp / "q.json")
        self.assertEqual(set(doc["quotes"]), {"2330", "0050"})
        self.assertTrue(doc["status"]["simulated"])
        self.assertFalse(doc["status"]["connected"])            # 結束後標示未連線
        self.assertTrue(all(q["kind"] == "simulated" for q in doc["quotes"].values()))
        self.assertGreater(r.p.polls, 3)

    def test_disconnect_then_reconnect(self):
        r, _ = make(pv.SimulatedProvider({"2330": 1000.0}, fail_at={3}, fail_connect={2, 3}), self.tmp)
        r.want(["2330"], "page")
        r.run(seconds=60)
        self.assertGreaterEqual(r.reconnects, 2)
        self.assertTrue(any("中斷" in e["error"] or "失敗" in e["error"] for e in r.errors))
        self.assertIn("2330", r.quotes)

    def test_heartbeat_timeout_forces_reconnect(self):
        p = pv.SimulatedProvider({"2330": 1.0})
        p.poll = lambda: []                                       # 連著但一直沒有行情
        r, c = make(p, self.tmp, heartbeat=10)
        r.want(["2330"])
        r.connect()
        c.t += 11
        r.step()
        self.assertFalse(p.connected)
        self.assertEqual(r.reconnects, 1)

    def test_duplicates_filtered(self):
        q = pv.Quote("2330", 1000.0, 990.0, 5, "2026-10-27T10:00:00", "simulated", "simulated")
        p = pv.SimulatedProvider()
        p.poll = lambda: [q, q]
        r, _ = make(p, self.tmp)
        r.want(["2330"])
        self.assertEqual(r.step(), 1)
        self.assertEqual(r.step(), 0)
        self.assertEqual(r.duplicates, 3)

    def test_subscription_limit_keeps_priority(self):
        p = pv.SimulatedProvider(max_subscriptions=3)
        r, _ = make(p, self.tmp)
        r.want([f"9{i:03d}" for i in range(10)], "other")
        r.want(["2330"], "watchlist")
        r.want(["0050", "2330"], "page")
        r.connect()
        self.assertEqual(len(p.subscribed), 3)
        self.assertTrue({"0050", "2330"} <= p.subscribed)


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.now = config.now().replace(microsecond=0)
        self.off = pv.Quote("2330", 990.0, 980.0, 1, "2026-10-24T13:30:00", "official_close", "official")

    def _write(self, age, connected=True):
        q = pv.Quote("2330", 1000.0, 990.0, 1, (self.now - timedelta(seconds=age)).isoformat(), "simulated", "simulated")
        store.write({"2330": q}, {"connected": connected}, self.tmp / "q.json")

    def test_fresh_intraday_shown_with_label(self):
        self._write(10)
        v = store.view("2330", self.off, self.tmp / "q.json", now=self.now)
        self.assertEqual(v["price"], 1000.0)
        self.assertIn("模擬", v["label"])

    def test_stale_or_disconnected_falls_back_to_official(self):
        self._write(600)
        v = store.view("2330", self.off, self.tmp / "q.json", now=self.now)
        self.assertEqual((v["price"], v["label"]), (990.0, "盤後"))
        self.assertIn("收盤價", v["note"])
        self._write(5, connected=False)
        self.assertEqual(store.view("2330", self.off, self.tmp / "q.json", now=self.now)["price"], 990.0)

    def test_many_readers_never_see_partial_file(self):
        bad = []

        def reader():
            for _ in range(200):
                d = store.read(self.tmp / "q.json")
                if d is not None and "quotes" not in d:
                    bad.append(d)
        self._write(1)
        ts = [threading.Thread(target=reader) for _ in range(8)]
        for t in ts:
            t.start()
        for i in range(200):
            self._write(i % 5)
        for t in ts:
            t.join()
        self.assertEqual(bad, [])

    def test_intraday_never_touches_official_prices(self):
        con = sqlite3.connect(config.DATA_DIR / "stocks.db") if (config.DATA_DIR / "stocks.db").exists() else None
        before = con.execute("SELECT COUNT(*), MAX(date) FROM prices").fetchone() if con else None
        r, _ = make(pv.SimulatedProvider({"2330": 1000.0}), self.tmp)
        r.want(["2330"])
        r.run(seconds=20)
        if con:
            self.assertEqual(con.execute("SELECT COUNT(*), MAX(date) FROM prices").fetchone(), before)
            con.close()
        for f in (ROOT / "app" / "realtime").glob("*.py"):
            self.assertNotRegex(f.read_text("utf-8"), r"INSERT|UPDATE\s+prices|REPLACE INTO", f.name)


class Switch(unittest.TestCase):
    def test_make_switches_sources(self):
        self.assertIsInstance(pv.make("simulated"), pv.SimulatedProvider)
        self.assertIsInstance(pv.make("official_close"), pv.OfficialCloseProvider)
        self.assertIsInstance(pv.make("yuanta"), pv.YuantaSparkProvider)

    def test_yuanta_refuses_without_config_and_license(self):
        env = {k: os.environ.pop(k, None) for k in pv.YuantaSparkProvider.ENV + ("YUANTA_DISPLAY_LICENSED",)}
        try:
            r, _ = make(pv.make("yuanta"), Path(tempfile.mkdtemp()))
            r.want(["2330"])
            self.assertFalse(r.connect())
            self.assertIn("YUANTA_ID", r.errors[0]["error"])
            self.assertIn("公開展示", r.errors[0]["error"])
        finally:
            for k, v in env.items():
                if v is not None:
                    os.environ[k] = v

    def test_secrets_never_printed(self):
        os.environ.update(YUANTA_ID="A123456789", YUANTA_PASSWORD="pw-secret-xyz", YUANTA_CERT_PATH="/x.pfx", YUANTA_CERT_PASSWORD="c-secret")
        try:
            p = pv.make("yuanta")
            self.assertNotIn("pw-secret-xyz", repr(p))
            self.assertNotIn("A123456789", repr(p))
        finally:
            for k in pv.YuantaSparkProvider.ENV:
                os.environ.pop(k, None)


class NoSecretsInRepo(unittest.TestCase):
    def test_frontend_has_no_broker_credentials_or_sdk(self):
        html = (ROOT / "static" / "index.html").read_text("utf-8")
        for pat in (r"YUANTA_", r"SPARK", r"\.pfx", r"sb_secret_"):
            self.assertIsNone(re.search(pat, html), pat)


if __name__ == "__main__":
    unittest.main()
