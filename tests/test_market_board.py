import os
import unittest

from app import db, market_board as mb

DB = "test_market_board.db"


def stock(sid, market, score=None):
    return {"stock_id": sid, "name": sid, "market": market, "industry": "半導體業", "is_common": True,
            "ai": {"modes": {"balanced": {"score": score, "rank": 1 if score else None}}}}


class LimitPrice(unittest.TestCase):
    def test_tick_rules(self):
        self.assertEqual(mb.limits(100), (110.0, 90.0))
        self.assertEqual(mb.limits(47.3), (52.0, 42.6))     # 52.03 取到 0.1；42.57 往上取到 0.05
        self.assertEqual(mb.limits(9.1), (10.0, 8.19))      # 10.01 落在 10 元以上，用 0.05 單位
        self.assertEqual(mb.limits(455), (500.0, 409.5))
        self.assertEqual(mb.limits(1000), (1100.0, 900.0))
        self.assertEqual(mb.limits(1234), (1355.0, 1115.0))  # 1357.4 用 5 元單位往下取


class Board(unittest.TestCase):
    def setUp(self):
        p = db.path(DB)
        if p.exists():
            os.remove(p)
        px = []
        for d in ("2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"):
            for sid in ("A", "B", "C", "D", "E"):
                px.append({"stock_id": sid, "date": d, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000, "value": 1e5})
        today = "2026-10-12"
        px += [
            {"stock_id": "A", "date": today, "open": 100, "high": 110, "low": 100, "close": 110, "volume": 5000, "value": 5e5},   # 上市漲停
            {"stock_id": "B", "date": today, "open": 100, "high": 109.5, "low": 100, "close": 109.5, "volume": 1, "value": 1},  # 漲 9.5% 但不是漲停
            {"stock_id": "C", "date": today, "open": 100, "high": 100, "low": 90, "close": 90, "volume": 1, "value": 1},      # 上櫃跌停
            {"stock_id": "D", "date": today, "open": 100, "high": 110, "low": 100, "close": 104, "volume": 1, "value": 1},    # 摸漲停後打開
            {"stock_id": "N", "date": "2026-10-09", "open": 50, "high": 50, "low": 50, "close": 50, "volume": 1, "value": 1},
            {"stock_id": "N", "date": today, "open": 60, "high": 60, "low": 60, "close": 60, "volume": 1, "value": 1},        # 新上市，不判斷
        ]
        with db.connect(DB) as con:
            db.upsert(con, "prices", px)
            db.upsert(con, "exrights", [{"stock_id": "E", "date": today, "prev_close": 100, "ref_price": 95}])
            db.upsert(con, "prices", [{"stock_id": "E", "date": today, "open": 95, "high": 104.5, "low": 95, "close": 104.5, "volume": 1, "value": 1}])
        self.snap = {"data_dates": {"prices": today},
                     "stocks": [stock("A", "twse", 80), stock("B", "twse"), stock("C", "tpex"), stock("D", "twse"), stock("E", "tpex"), stock("N", "tpex")]}

    def tearDown(self):
        os.remove(db.path(DB))

    def test_board(self):
        b = mb.build(self.snap, DB)
        self.assertEqual([i["stock_id"] for i in b["limit_up"]], ["A", "E"])   # E：除權息參考價 95 × 1.1 = 104.5
        self.assertEqual([i["stock_id"] for i in b["limit_down"]], ["C"])
        self.assertEqual(b["touched_up"], 1)
        self.assertEqual(b["markets"]["twse"]["limit_up"], 1)
        self.assertEqual(b["markets"]["tpex"]["limit_up"], 1)
        self.assertEqual(b["markets"]["tpex"]["new_listing"], 1)
        self.assertEqual(b["markets"]["all"]["up"], 5)
        self.assertEqual(b["limit_up"][0]["ai_score"], 80)
        self.assertEqual(b["gainers"][0]["stock_id"], "N")
        self.assertTrue(b["warnings"])


if __name__ == "__main__":
    unittest.main()
