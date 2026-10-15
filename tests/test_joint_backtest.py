"""AI 選股＋進出場聯合回測：資金、成本、停損、買不到留現金。"""
import unittest

import numpy as np
import pandas as pd

from app import joint_backtest as jb, strategy


def mk(dates, closes, opens=None, value=1e9, highs=None, lows=None):
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else c, float)
    h = np.array(highs if highs is not None else np.maximum(o, c), float)
    lo = np.array(lows if lows is not None else np.minimum(o, c), float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1e6, "value": value}, index=pd.Index(dates, name="date"))


class Joint(unittest.TestCase):
    def setUp(self):
        self.dates = list(pd.bdate_range("2025-01-01", periods=120).strftime("%Y-%m-%d"))
        d = self.dates
        self.lists = [{"signal": d[59], "entry": d[60], "exit": d[90], "next_signal": d[89], "ids": ["A"]},
                      {"signal": d[89], "entry": d[90], "exit": d[119], "next_signal": d[119], "ids": ["A"]}]

    def D(self, a):
        return {"dates": self.dates, "adj": {"A": a}, "rawi": {"A": a}}

    def run_(self, a, variant, n=1):
        D = self.D(a)
        F = jb._frames(D, {"A"})
        return jb.simulate(D, F, self.lists, n, variant)

    def test_hold_tracks_price_and_charges_cost_once(self):
        c = [100.0] * 60 + list(np.linspace(100, 110, 60))
        sim = self.run_(mk(self.dates, c), "hold")
        end = sim["curve"][-1][1]
        self.assertAlmostEqual(end, (1 - jb.COST) * 110 / 100, places=4)  # 續抱不重複扣成本
        self.assertEqual(len(sim["trades"]), 1)

    def test_stop_moves_to_cash(self):
        c = [100.0] * 60 + [100.0] * 5 + [80.0] * 55
        lows = [x * 0.99 for x in c]
        sim = self.run_(mk(self.dates, c, lows=lows), "hold_stop")
        t = sim["trades"][0]
        self.assertEqual(t["why"], "停損（跳空）")  # 開盤就跳空跌破停損價：用開盤價出場，不假裝停在停損價
        self.assertEqual(t["exit_px"], 80)
        inv = [x for d, _, x in sim["curve"] if d > t["exit"] and d < self.dates[90]]
        self.assertTrue(all(x == 0 for x in inv))  # 停損後留現金到下個月

    def test_locked_limit_up_is_skipped(self):
        c = [100.0] * 60 + [110.0] * 60
        o = [100.0] * 60 + [110.0] * 60
        a = mk(self.dates, c, opens=o, highs=c, lows=c)
        sim = self.run_(a, "hold")
        self.assertEqual(sim["skips"]["locked"], 1)
        self.assertEqual(sim["curve"][5][2], 0)  # 沒買到：資金全部是現金

    def test_thin_is_skipped(self):
        sim = self.run_(mk(self.dates, [100.0] * 120, value=1e5), "hold")
        self.assertGreaterEqual(sim["skips"]["thin"], 1)
        self.assertEqual(sim["trades"], [])

    def test_no_signal_no_trade(self):
        sim = self.run_(mk(self.dates, [100.0] * 120), "swing")
        self.assertEqual(sim["trades"], [])
        self.assertAlmostEqual(sim["curve"][-1][1], 1.0)

    def test_short_exit_rules_match_strategy(self):
        rng = np.random.default_rng(3)
        c = list(100 * np.cumprod(1 + rng.normal(0.003, 0.025, 120)))
        a = mk(self.dates, c, value=1e9)
        sim = self.run_(a, "short")
        for t in sim["trades"]:
            if t["why"] in ("停利", "停損", "持有期滿", "停損（跳空）"):
                self.assertLessEqual((pd.Index(self.dates).get_loc(t["exit"]) - pd.Index(self.dates).get_loc(t["entry"])) + 1,
                                     strategy.RULES["short"]["max_days"])


if __name__ == "__main__":
    unittest.main()
