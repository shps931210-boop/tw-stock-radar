"""V19 資料完整性稽核與「資料不足」原因拆分：虧損股不能被說成缺財報，不符合推薦資格不等於沒有資料。"""
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from app import coverage, db, snapshot


class GapText(unittest.TestCase):
    def test_loss_base_is_not_missing_data(self):
        df = pd.DataFrame({"eps_q_yoy": [None], "eps_q_turn": ["持續虧損"], "rev_yoy_3m": [None], "earnings_yield": [None], "eps_ttm": [-1.0]})
        gs = [snapshot._gap_label(df, k, 0) for k in ("rev_yoy_3m", "eps_q_yoy", "earnings_yield")]
        txt = snapshot._gap_text(gs)
        self.assertIn("缺近三月營收", txt)
        self.assertIn("因去年同期虧損，EPS 年增不適用", txt)
        self.assertIn("近四季虧損，本益比不適用", txt)

    def test_stale_value(self):
        df = pd.DataFrame({"rev_yoy": [5.0]})
        self.assertEqual(snapshot._gap_label(df, "rev_yoy", 0), ("stale", "月營收年增"))


class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR
        db.DATA_DIR = Path(self.tmp.name)

    def tearDown(self):
        db.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_counts_and_reasons(self):
        days = pd.bdate_range("2025-01-01", "2026-10-22").strftime("%Y-%m-%d")
        with db.connect() as con:
            db.upsert(con, "prices", [{"stock_id": "2330", "date": d, "close": 1, "value": 1} for d in days] +
                      [{"stock_id": "9999", "date": d, "close": 1, "value": 1} for d in days[-50:]])
            db.upsert(con, "revenue", [{"stock_id": "2330", "ym": f"{y}-{m:02d}", "revenue": 1} for y in (2025, 2026) for m in range(1, 13)])
            db.upsert(con, "financials", [{"stock_id": "2330", "period": p, "period_end": "x", "revenue": 1} for p in
                                          ("2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2")])
            ok = {"status": "ready"}
            rows = [{"stock_id": "2330", "name": "台積電", "market": "twse", "is_common": True, "base_ok": True, "complete": True,
                     "ai": {"rec": {"balanced": {"recommendation_eligibility": "normal"}}}},
                    {"stock_id": "9999", "name": "新股", "market": "tpex", "is_common": True, "base_ok": False, "complete": False,
                     "exclude_reasons": ["成交金額過低"], "ai": {"rec": {}}}]
            dec = {"2330": {"modules": {k: ok for k in ("quote", "fundamental", "valuation", "ai")}},
                   "9999": {"modules": {"quote": ok, "fundamental": {"status": "missing_source"}, "valuation": {"status": "missing_source"},
                                        "ai": {"status": "partial"}}}}
            a = coverage.audit(con, rows, dec, today=date(2026, 10, 26))
        S = {x["key"]: x["stocks"] for x in a["summary"]}
        self.assertEqual(a["universe"]["common"], 2)
        self.assertEqual((S["px120"], S["rev12"], S["fin4"], S["eligible"]), (1, 1, 1, 1))
        why = {g["why"] for g in a["gaps"]}
        self.assertIn("股價歷史未滿 120 個交易日（新上市或資料起點較晚）", why)
        self.assertIn("不符合正式推薦資格：成交金額過低", why)
        st = {x["id"]: x for x in a["stocks"]}
        self.assertEqual(st["2330"]["px"][0], days[0])
        self.assertEqual(st["9999"]["st"], "insufficient_history")
        self.assertNotIn("不符合正式推薦資格：成交金額過低", st["9999"]["why"])  # 推薦資格另外標示，不混進資料缺口
