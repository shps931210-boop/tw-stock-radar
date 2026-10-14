import unittest

import pandas as pd

from app import pit_backtest as pb


class Availability(unittest.TestCase):
    def test_deadlines(self):
        self.assertEqual(pb.fin_available("2025-Q1"), "2025-05-16")
        self.assertEqual(pb.fin_available("2025-Q2"), "2025-08-15")
        self.assertEqual(pb.fin_available("2025-Q2", is_financial=True), "2025-09-01")
        self.assertEqual(pb.fin_available("2025-Q3"), "2025-11-15")
        self.assertEqual(pb.fin_available("2025-Q4"), "2026-04-01")
        self.assertEqual(pb.rev_available("2025-12"), "2026-01-11")
        self.assertEqual(pb.rev_available("2025-07"), "2025-08-11")

    def test_rebalance_dates(self):
        d = ["2025-01-27", "2025-02-03", "2025-02-04", "2025-02-27", "2025-03-03", "2025-03-04"]
        self.assertEqual(pb.rebalance_dates(d, "2025-02-01"), [("2025-01-27", "2025-02-03"), ("2025-02-27", "2025-03-03")])


def _px(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "value"])


class Hold(unittest.TestCase):
    def setUp(self):
        a = _px([("2025-02-27", 100, 100, 100, 100, 1e6, 1e8), ("2025-03-03", 101, 105, 100, 104, 1e6, 1e8), ("2025-04-01", 110, 110, 110, 110, 1e6, 1e8)])
        lock = _px([("2025-02-27", 50, 50, 50, 50, 1e6, 1e8), ("2025-03-03", 55, 55, 55, 55, 1e6, 1e8), ("2025-04-01", 60, 60, 60, 60, 1e6, 1e8)])
        thin = _px([("2025-03-03", 10, 10, 10, 10, 100, 1000), ("2025-04-01", 11, 11, 11, 11, 100, 1000)])
        gone = _px([("2025-03-03", 20, 20, 20, 20, 1e6, 1e8), ("2025-03-10", 18, 18, 18, 18, 1e6, 1e8)])
        px = {"A": a, "L": lock, "T": thin, "G": gone}
        self.D = {"px": px, "rawi": {k: v.set_index("date") for k, v in px.items()}, "adj": {k: v.set_index("date") for k, v in px.items()}}

    def test_rules(self):
        r, st = pb.hold_return(self.D, "A", "2025-03-03", "2025-04-01")
        self.assertAlmostEqual(r, 110 / 101 - 1)
        self.assertEqual(st, "ok")
        self.assertEqual(pb.hold_return(self.D, "L", "2025-03-03", "2025-04-01"), (None, "locked"))
        self.assertEqual(pb.hold_return(self.D, "T", "2025-03-03", "2025-04-01"), (None, "thin"))
        r, st = pb.hold_return(self.D, "G", "2025-03-03", "2025-04-01")   # 中途下市：用最後收盤價
        self.assertAlmostEqual(r, 18 / 20 - 1)
        self.assertEqual(st, "stale")
        self.assertEqual(pb.hold_return(self.D, "X", "2025-03-03", "2025-04-01"), (None, "not_traded"))

    def test_split_gap(self):
        px = _px([("2025-06-17", 190, 190, 190, 190, 1, 1), ("2025-06-18", 47.5, 48, 47, 47.6, 1, 1), ("2025-06-19", 48, 48, 48, 48, 1, 1)])
        a = pb.fix_gaps(px)
        self.assertAlmostEqual(a["close"].iloc[0], 190 * 47.5 / 190)   # 一拆四：前一天的價格換算成拆分後
        self.assertAlmostEqual(a["close"].iloc[2], 48)

    def test_stats(self):
        s = pb._stats([0.1, -0.1], [0.05, None])
        self.assertEqual(s["periods"], 2)
        self.assertEqual(s["total"], -1.0)
        self.assertEqual(s["mdd"], -10.0)
        self.assertEqual(s["win_vs_bench"], 100.0)


if __name__ == "__main__":
    unittest.main()
