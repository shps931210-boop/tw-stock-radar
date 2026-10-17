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

    def month_revenue(self, sid, start=None):
        return self._hit("rev_long" if start else "rev", sid)

    def financials(self, sid, start=None):
        return self._hit("fin_long" if start else "fin", sid)

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


class DeepHistory(Backfill):
    """長期歷史：第一次抓就從 fin_history_start 抓起；以前只抓近幾年的股票排在最後補一次。"""

    def test_new_stocks_fetch_long_once(self):
        self.cfg["fin_history_start"] = "2017-01-01"
        self.run_fill([])
        kinds = {c for c, _ in FakeFM.calls}
        self.assertEqual(kinds, {"rev_long", "fin_long", "div"})
        FakeFM.calls = []
        Q = self.run_fill([])
        self.assertEqual(FakeFM.calls, [])  # 都補過了，不再重抓
        self.assertEqual(Q["hist"], 0)
        self.assertEqual(Q["hist_left"], 0)

    def test_old_stocks_get_history_pass_last(self):
        self.run_fill([])  # 以前沒有長期起點：只抓近幾年
        self.cfg["fin_history_start"] = "2017-01-01"
        FakeFM.calls = []
        Q = self.run_fill([])
        self.assertEqual(Q["hist"], 6)
        self.assertEqual(sorted(c for c, _ in FakeFM.calls), ["fin_long"] * 3 + ["rev_long"] * 3)
        self.assertEqual(Q["hist_left"], 0)

    def test_history_pass_stops_on_quota(self):
        self.run_fill([])
        self.cfg["fin_history_start"] = "2017-01-01"
        FakeFM.calls, FakeFM.limit = [], 2
        Q = self.run_fill([])
        self.assertTrue(Q["quota_hit"])
        self.assertGreater(Q["hist_left"], 0)


class FakeQuotes:
    """假的證交所／櫃買：empty 裡的日期回傳空資料（休市或端點不支援）。"""
    seen: list = []

    def __init__(self, market, empty=()):
        self.market, self.empty = market, set(empty)

    def daily_quotes(self, d):
        FakeQuotes.seen.append((self.market, d.isoformat()))
        if d.isoformat() in self.empty:
            return [], None
        return [{"stock_id": "2330" if self.market == "twse" else "6488", "name": "x", "open": 1, "high": 1, "low": 1,
                 "close": 1, "volume": 1, "value": 1}], None


class HistoryPrices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR
        db.DATA_DIR = Path(self.tmp.name)
        FakeQuotes.seen = []
        self.cfg = {"markets": ["twse", "tpex"], "twse_interval_sec": 0, "price_backfill_days": 30, "recent_backfill_days": 30,
                    "history_start": None, "history_dates_backfill": 5}

    def tearDown(self):
        db.DATA_DIR = self.old
        self.tmp.cleanup()

    def fill(self, twse_empty=(), tpex_empty=()):
        with mock.patch.object(update, "TwseSource", lambda *a: FakeQuotes("twse", twse_empty)), \
             mock.patch.object(update, "TpexSource", lambda *a: FakeQuotes("tpex", tpex_empty)):
            errs = []
            update.update_history(self.cfg, errors=errs)
        return errs

    def test_span(self):
        from datetime import date
        self.assertEqual(update._span({"price_backfill_days": 30}, date(2026, 10, 1)), 30)
        self.assertEqual(update._span({"price_backfill_days": 30, "history_start": "2026-01-01"}, date(2026, 10, 1)), 274)
        self.assertEqual(update._span({"price_backfill_days": 10, "history_start": "2018-01-01"}, date(2026, 10, 1)), 10)  # 快速更新不回補

    def test_walks_back_in_chunks_without_recent_days(self):
        from app import config
        start = (config.today() - __import__("datetime").timedelta(days=60)).isoformat()
        self.cfg["history_start"] = start
        self.fill()
        first = [d for m, d in FakeQuotes.seen if m == "twse"]
        self.assertEqual(len(first), 5)  # 每個市場每次最多 history_dates_backfill 天
        cut = (config.today() - __import__("datetime").timedelta(days=30)).isoformat()
        self.assertTrue(all(d <= cut for d in first))  # 最近的日子交給每日更新
        self.assertEqual(first, sorted(first, reverse=True))  # 從新往舊補
        FakeQuotes.seen = []
        self.fill()
        second = [d for m, d in FakeQuotes.seen if m == "twse"]
        self.assertTrue(set(second).isdisjoint(first))  # 補過的不再抓
        with db.connect() as con:
            h = json.loads(db.get_meta(con, "history_backfill"))
        self.assertEqual(h["target"], start)
        self.assertGreater(h["left_days"]["twse"], 0)

    def test_tpex_empty_on_trading_day_is_retried(self):
        from app import config
        self.cfg["history_start"] = (config.today() - __import__("datetime").timedelta(days=60)).isoformat()
        self.fill()
        days = [d for m, d in FakeQuotes.seen if m == "tpex"]
        with db.connect() as con:
            for d in days:
                con.execute("DELETE FROM fetch_log WHERE source='tpex_quotes' AND key=?", (d,))
            con.commit()
        FakeQuotes.seen = []
        errs = self.fill(tpex_empty=days)  # 上市有行情、上櫃空的：不能當成休市
        self.assertTrue(any("上櫃沒有資料" in e for e in errs))
        with db.connect() as con:
            n = con.execute("SELECT COUNT(*) FROM fetch_log WHERE source='tpex_quotes' AND key IN (%s)" % ",".join("?" * len(days)), days).fetchone()[0]
        self.assertEqual(n, 0)

    def test_long_empty_run_is_not_holiday(self):
        from app import config
        import datetime as dt
        self.cfg["history_start"] = (config.today() - dt.timedelta(days=90)).isoformat()
        self.cfg["history_dates_backfill"] = 15
        cut = config.today() - dt.timedelta(days=30)
        days = [d.isoformat() for d in (cut - dt.timedelta(days=i) for i in range(40)) if d.weekday() < 5][:15]
        errs = self.fill(twse_empty=days, tpex_empty=days)
        self.assertTrue(any("連續" in e for e in errs))
        with db.connect() as con:
            n = con.execute("SELECT COUNT(*) FROM fetch_log WHERE source='twse_quotes'").fetchone()[0]
        self.assertEqual(n, 0)  # 全部取消標記，下次再試


class AutoStart(unittest.TestCase):
    """回測起點跟著資料長度走：股價滿約一年、季報覆蓋六成以上的下一個月。"""

    def D(self, n_dates, fin):
        import pandas as pd
        dates = [f"{2018 + i // 240}-{(i // 20) % 12 + 1:02d}-{i % 20 + 1:02d}" for i in range(n_dates)]
        return {"dates": dates, "fin": {sid: pd.DataFrame({"period": ps}) for sid, ps in fin.items()}}

    def test_price_warmup(self):
        from app import pit_backtest as pb
        D = self.D(600, {"a": ["2017-Q4"], "b": ["2017-Q4"]})
        self.assertEqual(pb.auto_start(D), "2019-03-01")  # 第 260 個交易日 2019-02-01 → 下個月

    def test_financial_coverage_gate(self):
        from app import pit_backtest as pb
        # 2018-Q4 起才有大多數股票的季報（舊季度只有一檔先補到）→ 等 2018 年報公布（2019-04-01）之後
        fin = {"a": ["2017-Q1", "2018-Q4"], "b": ["2018-Q4"], "c": ["2018-Q4"]}
        self.assertEqual(pb.auto_start(self.D(600, fin)), "2019-05-01")


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
