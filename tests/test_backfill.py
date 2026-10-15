"""夜間財報補齊：會員申請的股票最先補、財報與營收先於股利、額度用完會停並記錄還沒補到的股票。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import db, update
from app.sources.finmind import FinMindError


class FakeFM:
    calls: list = []
    limit = 99

    def __init__(self, *a):
        pass

    def _hit(self, src, sid):
        if len(FakeFM.calls) >= FakeFM.limit:
            raise FinMindError("FinMind 本小時請求次數已用完")
        FakeFM.calls.append((src, sid))
        return []

    def stock_info(self):
        return []

    def month_revenue(self, sid):
        return self._hit("rev", sid)

    def financials(self, sid):
        return self._hit("fin", sid)

    def dividends(self, sid):
        return self._hit("div", sid)


class Backfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR
        db.DATA_DIR = Path(self.tmp.name)
        with db.connect() as con:  # 成交金額：2330 > 1101 > 1234
            for sid, v in (("2330", 9e9), ("1101", 5e8), ("1234", 1e6)):
                db.upsert(con, "prices", [{"stock_id": sid, "date": "2026-10-14", "close": 10, "value": v}])
        FakeFM.calls, FakeFM.limit = [], 99
        self.cfg = {"finmind_token": "", "finmind_interval_sec": 0, "fundamental_top_n": 3000, "watchlist": []}

    def tearDown(self):
        db.DATA_DIR = self.old
        self.tmp.cleanup()

    def run_fill(self, req):
        with mock.patch.object(update, "FinMindSource", FakeFM), mock.patch.object(update, "_fin_requests", return_value=(req, None)):
            update.update_fundamentals(self.cfg)
        with db.connect() as con:
            return json.loads(db.get_meta(con, "fundamental_queue"))

    def test_requested_first_dividends_last(self):
        Q = self.run_fill(["1234"])
        self.assertEqual(FakeFM.calls[0][1], "1234")
        self.assertEqual([c for c, _ in FakeFM.calls[-3:]], ["div"] * 3)
        self.assertEqual(Q["requested"], 1)
        self.assertEqual(Q["requested_done"], ["1234"])
        self.assertEqual(Q["pending"], [])
        self.assertEqual(sorted(Q["no_data"]), ["1101", "1234", "2330"])  # 查過但來源沒有季報

    def test_quota_stops_and_keeps_order(self):
        FakeFM.limit = 2
        Q = self.run_fill([])
        self.assertTrue(Q["quota_hit"])
        self.assertEqual(Q["done"], 2)
        self.assertEqual(Q["pending"], ["1101", "1234"])  # 2330 已補，其餘依成交金額排隊

    def test_unknown_request_ignored(self):
        self.run_fill(["9999"])
        self.assertNotIn("9999", {s for _, s in FakeFM.calls})


if __name__ == "__main__":
    unittest.main()


class Inventory(unittest.TestCase):
    def test_answers_from_db(self):
        import sqlite3
        from app import data_inventory, news
        con = sqlite3.connect(":memory:")
        con.executescript(db.SCHEMA)
        con.executescript(news.SCHEMA)
        con.executemany("INSERT INTO prices (stock_id, date, close) VALUES (?,?,?)",
                        [("2330", "2020-01-02", 1), ("9999", "2021-05-03", 1)] + [("2330", f"2026-09-{d:02d}", 2) for d in range(1, 26)])
        I = data_inventory.build(con)
        Q = {x["id"]: x for x in I["questions"]}
        self.assertEqual(set(Q), {"prices", "announce", "survivorship", "news", "pit"})
        self.assertEqual(Q["prices"]["level"], "ok")             # 6.7 年
        self.assertEqual(Q["survivorship"]["facts"]["最近 20 個交易日沒有成交的股票"], 1)
        self.assertEqual(Q["news"]["level"], "gap")               # 沒有新聞
        self.assertEqual(Q["announce"]["level"], "partial")       # 沒有實際公告時間，不能說足夠
        self.assertIn("資料工程", I["verdict"])
