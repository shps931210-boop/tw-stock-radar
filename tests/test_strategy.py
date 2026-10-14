"""進出場策略引擎：停損停利的成交假設、跳空、移動停利、狀態和回測用同一套規則；估值只用當時已公布的 EPS。"""
import unittest

import numpy as np
import pandas as pd

from app import strategy, valuation


def bars(closes, opens=None, highs=None, lows=None):
    n = len(closes)
    c = np.array(closes, float)
    o = np.array(opens if opens is not None else c, float)
    h = np.array(highs if highs is not None else np.maximum(o, c) * 1.005, float)
    lo = np.array(lows if lows is not None else np.minimum(o, c) * 0.995, float)
    d = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": d, "open": o, "high": h, "low": lo, "close": c, "volume": 1e6, "value": 1e9})


class Simulate(unittest.TestCase):
    def setUp(self):
        self.base = [100.0] * 80

    def frame(self, extra_c, extra_o=None, extra_h=None, extra_l=None):
        c = self.base + extra_c
        o = self.base + (extra_o or extra_c)
        h = [x * 1.005 for x in self.base] + (extra_h or [max(a, b) * 1.005 for a, b in zip(o[80:], extra_c)])
        lo = [x * 0.995 for x in self.base] + (extra_l or [min(a, b) * 0.995 for a, b in zip(o[80:], extra_c)])
        return strategy.frame(bars(c, o, h, lo))

    def test_stop_and_gap(self):
        f = self.frame([100, 100, 80], extra_o=[100, 100, 80], extra_l=[99.5, 99.5, 79])
        t = strategy.simulate(f, 79, "short")  # 第 80 天開盤進場
        self.assertEqual(t["why"], "停損（跳空）")
        self.assertAlmostEqual(t["exit"], 80)

    def test_target_first_day_and_cost(self):
        f = self.frame([100, 100], extra_h=[100.5, 120], extra_l=[99.5, 99.9])
        t = strategy.simulate(f, 79, "short")
        lv = strategy.levels("short", 100, float(f["atr"].iloc[79]))
        self.assertEqual(t["why"], "停利")
        self.assertAlmostEqual(t["exit"], lv["target"])
        self.assertAlmostEqual(t["ret"], lv["target"] / 100 - 1 - strategy.COST)

    def test_both_hit_same_day_assumes_stop(self):
        f = self.frame([100, 100], extra_h=[100.5, 130], extra_l=[99.5, 70])
        self.assertEqual(strategy.simulate(f, 79, "short")["why"], "停損")

    def test_time_exit(self):
        f = self.frame([100] * 10)
        t = strategy.simulate(f, 79, "short")
        self.assertEqual((t["why"], t["days"]), ("持有期滿", 5))

    def test_swing_trailing_only_moves_up(self):
        up = list(np.linspace(101, 140, 30)) + [139, 130, 120]
        f = self.frame(up)
        t = strategy.simulate(f, 79, "swing")
        self.assertIn(t["why"], ("停損", "停損（跳空）", "跌破季線"))
        self.assertGreater(t["exit"], 101)  # 移動停利鎖住部分獲利


class StatusMatchesBacktest(unittest.TestCase):
    def test_status_uses_same_entries(self):
        rng = np.random.default_rng(1)
        c = 100 * np.cumprod(1 + rng.normal(0.002, 0.02, 300))
        px = bars(c)
        f = strategy.frame(px)
        e = strategy.entries(f)
        st = strategy.status(px)
        for m in ("short", "swing"):
            self.assertEqual(st[m]["state"] == "triggered", bool(e[m].iloc[-1]))
            lv = strategy.levels(m, float(f["close"].iloc[-1]), float(f["atr"].iloc[-1]))
            self.assertAlmostEqual(st[m]["stop"], round(lv["stop"], 2))

    def test_not_ready_short_history(self):
        st = strategy.status(bars([100] * 10))
        self.assertFalse(st["ready"])
        self.assertEqual(st["status"], "insufficient_history")

    def test_partial_history_runs_short_rule_only(self):
        # 21～79 天：短線規則照算，波段明確標示歷史不足，不硬算
        st = strategy.status(bars(list(np.linspace(100, 120, 50))))
        self.assertTrue(st["ready"])
        self.assertEqual(st["status"], "partial")
        self.assertIn(st["short"]["state"], ("triggered", "setup", "none"))
        self.assertIsNotNone(st["short"]["stop"])
        self.assertEqual(st["swing"]["state"], "na")
        self.assertEqual(st["swing"]["status"], "insufficient_history")


class Valuation(unittest.TestCase):
    def test_ttm_only_after_deadline(self):
        fin = pd.DataFrame({"period": ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"], "eps": [1, 1, 1, 1, 2]})
        s = valuation.eps_ttm_series(fin, False)
        self.assertEqual(s[0], ("2025-04-01", 4.0))   # 2024 年報 4/1 才算公布
        self.assertEqual(s[1], ("2025-05-16", 5.0))   # 2025-Q1 5/16 才算公布

    def test_band_and_losses(self):
        fin = pd.DataFrame({"period": [f"{y}-Q{q}" for y in (2023, 2024) for q in (1, 2, 3, 4)], "eps": [1] * 8})
        d = pd.bdate_range("2024-04-01", periods=300).strftime("%Y-%m-%d")
        raw = pd.DataFrame({"date": d, "close": np.linspace(40, 80, 300)})
        b = valuation.band(raw, fin, False)
        self.assertEqual(b["eps"], 4.0)
        self.assertAlmostEqual(b["pe"], 20.0)
        self.assertTrue(b["low"] < b["base"] < b["high"])
        loss = pd.DataFrame({"period": fin["period"], "eps": [-1] * 8})
        self.assertIn("虧損", valuation.band(raw, loss, False)["why"])


if __name__ == "__main__":
    unittest.main()
