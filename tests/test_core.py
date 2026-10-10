"""核心資料正確性測試：python -m unittest discover tests

不連網，全部用人工資料，檢查「資料不完整或日期不一致時不能正常計分」這類規則。
"""
from __future__ import annotations

import sqlite3
import unittest
from datetime import date
from types import SimpleNamespace

import pandas as pd

from app import db, metrics, snapshot, update


def prices(n=30, start="2026-08-03"):
    d = [x.strftime("%Y-%m-%d") for x in pd.bdate_range(start, periods=n)]
    return pd.DataFrame({"date": d, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1_000_000.0, "value": 1e8})


def flows(dates, val=1000.0):
    return pd.DataFrame({"date": dates, "foreign_net": val, "trust_net": val, "dealer_net": val})


class ChipAlignment(unittest.TestCase):
    def test_aligned_dates_use_same_volume(self):
        px = prices()
        m = metrics.chip_metrics(flows(px["date"].tolist()), pd.DataFrame(), px)
        self.assertEqual(m["inst_lag"], 0)
        self.assertAlmostEqual(m["inst_ratio_5d"], 3 * 1000 * 5 / (1_000_000 * 5) * 100)

    def test_lagging_institutional_data_is_not_scored(self):
        px = prices()
        m = metrics.chip_metrics(flows(px["date"].tolist()[:-2]), pd.DataFrame(), px)  # 落後 2 個交易日
        self.assertEqual(m["inst_lag"], 2)
        self.assertNotIn("inst_ratio_5d", m)
        self.assertNotIn("foreign_5d", m)

    def test_one_day_lag_is_allowed_and_uses_matching_volume(self):
        px = prices()
        px.loc[len(px) - 1, "volume"] = 9e9  # 最新一天量很大，但法人還沒公布，不能拿來當分母
        m = metrics.chip_metrics(flows(px["date"].tolist()[:-1]), pd.DataFrame(), px)
        self.assertEqual(m["inst_lag"], 1)
        self.assertAlmostEqual(m["inst_ratio_5d"], 0.3)

    def test_gap_in_institutional_data_is_not_scored(self):
        px = prices()
        d = px["date"].tolist()
        m = metrics.chip_metrics(flows(d[:-3] + d[-2:]), pd.DataFrame(), px)  # 中間缺一天
        self.assertNotIn("inst_ratio_5d", m)

    def test_margin_uses_same_20_days(self):
        px = prices()
        px["close"] = [100.0 + i for i in range(len(px))]
        mg = pd.DataFrame({"date": px["date"], "margin_bal": [1000.0 - i for i in range(len(px))]})
        m = metrics.chip_metrics(pd.DataFrame(), mg, px)
        self.assertAlmostEqual(m["margin_chg_20d"], ((1000 - 29) / (1000 - 10) - 1) * 100)
        self.assertTrue(m["price_up_margin_down"])


class ExRights(unittest.TestCase):
    def test_adjusts_prices_before_ex_date(self):
        px = prices(6)
        px.loc[3:, ["open", "high", "low", "close"]] = 95.0  # 第 4 天除息 5 元
        ex = pd.DataFrame([{"date": px["date"][3], "prev_close": 100.0, "ref_price": 95.0}])
        adj, n = metrics.adjust_prices(px, ex)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(adj["close"][0], 95.0)
        self.assertAlmostEqual(adj["close"].iloc[-1], 95.0)

    def test_mismatched_prev_close_is_ignored(self):
        px = prices(6)
        ex = pd.DataFrame([{"date": px["date"][3], "prev_close": 120.0, "ref_price": 114.0}])
        _, n = metrics.adjust_prices(px, ex)
        self.assertEqual(n, 0)

    def test_future_ex_date_is_ignored(self):
        px = prices(6)
        ex = pd.DataFrame([{"date": "2099-01-01", "prev_close": 100.0, "ref_price": 95.0}])
        adj, n = metrics.adjust_prices(px, ex)
        self.assertEqual(n, 0)
        self.assertAlmostEqual(adj["close"].iloc[-1], 100.0)

    def test_parse_twse_fields(self):
        from app.sources.twse import parse_ex_rights
        f = ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價", "權值+息值", "權/息", "漲停價格", "跌停價格",
             "開盤競價基準", "減除股利參考價"]
        r = parse_ex_rights(f, [["115年07月01日", "1101", "台泥", "24.05", "23.25", "0.80", "息", "", "", "", "23.25"]])
        self.assertEqual(r, [{"stock_id": "1101", "date": "2026-07-01", "prev_close": 24.05, "ref_price": 23.25}])


class Fundamentals(unittest.TestCase):
    def test_negative_base_growth(self):
        self.assertEqual(metrics._turn(1.0, -0.5), (None, "虧轉盈"))
        self.assertEqual(metrics._turn(-1.0, -0.5), (None, "持續虧損"))
        self.assertEqual(metrics._turn(-1.0, 2.0)[1], "盈轉虧")
        self.assertAlmostEqual(metrics._turn(3.0, 2.0)[0], 50.0)

    def _fin(self, drop=None):
        rows = []
        for i, end in enumerate(pd.date_range("2024-03-31", periods=9, freq="QE")):
            r = {"period": f"{end.year}Q{end.quarter}", "period_end": end.strftime("%Y-%m-%d"), "revenue": 100 + i,
                 "gross_profit": 30 + i, "operating_income": 10 + i, "net_income": 8 + i, "eps": 1 + i / 10,
                 "total_assets": 1000.0, "total_liabilities": 400 - i, "equity": 600.0 + i, "current_assets": 500.0 + i,
                 "current_liabilities": 300.0, "share_capital": 100.0, "ocf": 12.0 + i, "capex": 3.0}
            if drop:
                r[drop] = None
            rows.append(r)
        return pd.DataFrame(rows)

    def test_full_fscore(self):
        m = metrics.fundamental_metrics(self._fin(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertEqual(m["fscore_n"], 9)
        self.assertIsNotNone(m["fscore"])

    def test_partial_fscore_is_not_reported(self):
        m = metrics.fundamental_metrics(self._fin("current_assets"), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertEqual(m["fscore_n"], 8)
        self.assertIsNone(m["fscore"])


class Freshness(unittest.TestCase):
    def test_expected_financial_period(self):
        self.assertEqual(snapshot.expected_fin_end(date(2026, 10, 10)), "2026-06-30")
        self.assertEqual(snapshot.expected_fin_end(date(2026, 11, 30)), "2026-09-30")
        self.assertEqual(snapshot.expected_fin_end(date(2026, 4, 20)), "2025-12-31")

    def test_expected_revenue_month(self):
        self.assertEqual(snapshot.expected_rev_ym(date(2026, 10, 10)), "2026-08")
        self.assertEqual(snapshot.expected_rev_ym(date(2026, 10, 20)), "2026-09")

    def test_weekdays_between(self):
        self.assertEqual(snapshot._weekdays_between("2026-10-09", "2026-10-12"), 1)  # 週五 → 週一
        self.assertEqual(snapshot._weekdays_between("2026-10-12", "2026-10-09"), 0)


class OpenApiSync(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(db.SCHEMA)

    def _seed(self, day, n, mkt="twse"):
        for i in range(n):
            sid = str(1101 + i)
            self.con.execute("INSERT OR IGNORE INTO stocks VALUES (?,?,?,'')", (sid, sid, mkt))
            self.con.execute("INSERT INTO prices VALUES (?,?,1,1,1,1,1,1)", (sid, day))

    def rows(self, n):
        return [{"stock_id": str(1101 + i), "name": "x", "open": 1, "high": 1, "low": 1, "close": 2, "volume": 1, "value": 1}
                for i in range(n)]

    def test_partial_day_is_filled(self):
        self._seed("2026-10-08", 100)
        self._seed("2026-10-09", 40)
        update._sync_quotes(self.con, "twse", "2026-10-09", self.rows(100), [])
        n = self.con.execute("SELECT COUNT(*) FROM prices WHERE date='2026-10-09'").fetchone()[0]
        self.assertEqual(n, 100)
        self.assertTrue(self.con.execute("SELECT ok FROM fetch_log WHERE source='twse_quotes'").fetchone()[0])

    def test_incomplete_response_is_not_marked_ok(self):
        self._seed("2026-10-08", 100)
        errs = []
        update._sync_quotes(self.con, "twse", "2026-10-09", self.rows(50), errs)
        self.assertIsNone(self.con.execute("SELECT ok FROM fetch_log WHERE source='twse_quotes'").fetchone())
        self.assertTrue(errs)

    def test_valuation_synced_even_when_prices_exist(self):
        self._seed("2026-10-09", 3)
        con = self.con
        import app.sources.twse as T
        old = T.openapi_quotes, T.openapi_valuation
        T.openapi_quotes = lambda: ("2026-10-09", self.rows(3))
        T.openapi_valuation = lambda: ("2026-10-09", [{"stock_id": "1101", "per": 10, "pbr": 1, "dy": 3}])
        try:
            update._openapi_fallback(con, {"markets": ["twse"]}, [])
        finally:
            T.openapi_quotes, T.openapi_valuation = old
        self.assertEqual(con.execute("SELECT per FROM valuation WHERE stock_id='1101'").fetchone()[0], 10)


class Snapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.sources import demo
        demo.build()
        with db.connect("demo.db") as con:  # 第一檔：拿掉總資產 → ROA 算不出來 → 財務品質不完整
            cls.sid = con.execute("SELECT stock_id FROM stocks WHERE length(stock_id)=4 AND stock_id NOT LIKE '0%' "
                                  "ORDER BY stock_id LIMIT 1").fetchone()[0]
            con.execute("UPDATE financials SET total_assets=NULL WHERE stock_id=?", (cls.sid,))
        cls.snap = snapshot.build(demo=True)
        cls.by = {s["stock_id"]: s for s in cls.snap["stocks"]}

    @classmethod
    def tearDownClass(cls):  # 還原成乾淨的示範資料
        from app.sources import demo
        demo.build()
        snapshot.build(demo=True)

    def test_missing_required_field_blocks_ranking(self):
        s = self.by[self.sid]
        self.assertFalse(s["complete"])
        self.assertIsNone(s["composite"])
        self.assertIsNone(s["ai"]["modes"]["balanced"]["rank"])
        self.assertNotEqual(s["timing"]["status"], "buy")
        self.assertTrue(any("財務品質" in x for x in s["missing_detail"]))

    def test_complete_stocks_still_ranked(self):
        self.assertTrue(any(s["ai"]["modes"]["balanced"]["rank"] for s in self.snap["stocks"]))

    def test_signal_rule_is_marked_untested(self):
        self.assertFalse(self.snap["buy_point"]["backtested"])


class Tracking(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from app import track
        from app.sources import demo
        demo.build()
        self.snap = snapshot.build(demo=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.old, track.HISTORY = track.HISTORY, Path(self.tmp.name)

    def tearDown(self):
        from app import track
        track.HISTORY = self.old
        self.tmp.cleanup()

    def test_record_is_write_once_and_skips_demo(self):
        from app import track
        self.assertIsNone(track.record(self.snap))  # 示範資料不記錄
        live = {**self.snap, "source": "live"}
        self.assertIsNotNone(track.record(live))
        self.assertIsNone(track.record(live))  # 同一天不覆寫
        self.assertIsNone(track.record({**live, "rankings_blocked": "x", "data_dates": {"prices": "2099-01-01"}}))

    def test_performance_against_index(self):
        import json
        from app import track
        with db.connect("demo.db") as con:
            d0 = con.execute("SELECT date FROM index_prices ORDER BY date DESC LIMIT 1 OFFSET 10").fetchone()[0]
            c0 = con.execute("SELECT close FROM prices WHERE stock_id='2330' AND date=?", (d0,)).fetchone()[0]
            c1 = con.execute("SELECT close FROM prices WHERE stock_id='2330' ORDER BY date DESC LIMIT 1").fetchone()[0]
        log = {"date": d0, "model_version": "test", "picks": {"balanced": [{"stock_id": "2330"}]}, "signals": []}
        (track.HISTORY / f"{d0}.json").write_text(json.dumps(log), "utf-8")
        perf = track.performance("demo.db")
        g = perf["logs"][0]["groups"]["balanced"]
        self.assertEqual(perf["logs"][0]["held_days"], 10)
        self.assertAlmostEqual(g["avg"], round((c1 / c0 - 1) * 100, 2), places=2)
        self.assertAlmostEqual(g["excess"], round(g["avg"] - perf["logs"][0]["index_ret"], 2), places=1)


class Backtest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # 不依賴其他測試先建好示範資料庫
        from app.sources import demo
        demo.build()

    def test_signals_match_live_definition(self):
        from app import backtest
        snap = {s["stock_id"]: s for s in snapshot.build(demo=True)["stocks"]}
        with db.connect("demo.db") as con:
            px = db.read(con, "SELECT * FROM prices ORDER BY stock_id, date")
        for sid, p in px.groupby("stock_id"):
            last = backtest.signals(p.reset_index(drop=True)).iloc[-1]
            for k, v in last.items():
                self.assertEqual(bool(v), bool(snap[sid].get(k)), f"{sid} {k}")

    def test_run_reports_samples(self):
        from app import backtest
        r = backtest.run("demo.db")
        self.assertTrue(r["ready"])
        self.assertGreater(r["baseline"][20]["n"], 0)


class AdminAuth(unittest.TestCase):
    def test_remote_request_needs_token(self):
        import os
        from fastapi import HTTPException
        from app import server
        req = lambda host: SimpleNamespace(client=SimpleNamespace(host=host))
        os.environ.pop("ADMIN_TOKEN", None)
        server._require_admin(req("127.0.0.1"), None)
        with self.assertRaises(HTTPException):
            server._require_admin(req("8.8.8.8"), None)
        os.environ["ADMIN_TOKEN"] = "s3cret"
        try:
            with self.assertRaises(HTTPException):
                server._require_admin(req("127.0.0.1"), "wrong")
            server._require_admin(req("8.8.8.8"), "s3cret")
        finally:
            os.environ.pop("ADMIN_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
