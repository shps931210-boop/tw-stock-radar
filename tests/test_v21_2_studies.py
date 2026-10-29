"""V21.2 研究：進場方法的觸發條件、MSURGE 組合的選股規則、失準分類。"""
import unittest

import numpy as np
import pandas as pd

from app import entry_study as es
from app import miss_review as mr
from app import msurge_study as ms


def _a(closes, vol=None, high=None):
    dates = [f"d{i:02d}" for i in range(len(closes))]
    c = np.array(closes, float)
    return dates, pd.DataFrame({"open": c, "close": c, "high": high if high is not None else c, "low": c,
                                "volume": vol if vol is not None else np.full(len(c), 1e6)}, index=dates)


class Entry(unittest.TestCase):
    def test_a_is_signal_day(self):
        dates, a = _a([100] * 40)
        self.assertEqual(es.trigger(a, dates, dates[25], "A"), 25)

    def test_b_needs_close_above_ma_and_volume(self):
        c = [100] * 26 + [99, 99, 103, 104] + [104] * 10
        v = np.full(40, 1e6); v[28] = 3e6
        dates, a = _a(c, v)
        self.assertEqual(es.trigger(a, dates, dates[25], "B"), 28)
        v2 = np.full(40, 1e6)
        dates, a = _a(c, v2)
        self.assertIsNone(es.trigger(a, dates, dates[25], "B"))   # 沒放量就不買

    def test_c_needs_pullback_then_breakout_and_expires(self):
        c = [100] * 26 + [98, 96, 96.5, 99] + [99] * 10
        h = np.array(c, float) + 0.5
        dates, a = _a(c, high=h)
        self.assertEqual(es.trigger(a, dates, dates[25], "C"), 29)   # 96 回檔 4% 之後，99 > 前一日高 97
        dates, a = _a([100] * 40)
        self.assertIsNone(es.trigger(a, dates, dates[25], "C"))       # 沒回檔，10 日內不買 → 錯失


class Msurge(unittest.TestCase):
    def setUp(self):
        n = 60
        rng = np.random.default_rng(1)
        self.X = pd.DataFrame({"rk_balanced": np.arange(1, n + 1, dtype=float), "sc_balanced": np.linspace(90, 30, n),
                               "msurge": rng.normal(size=n), "rev_yoy": rng.normal(size=n), "ocf_ni": rng.normal(size=n),
                               "low_accruals": rng.normal(size=n), "ind_mom": rng.normal(size=n), "trend": rng.normal(size=n),
                               "st_reversal": rng.normal(size=n)}, index=[f"s{i}" for i in range(n)])
        self.X.loc["s59", "rk_balanced"] = np.nan   # 不在可排名母體

    def test_rules(self):
        self.assertEqual(ms.picks(self.X, "A"), [f"s{i}" for i in range(10)])
        for k in "BCDEF":
            p = ms.picks(self.X, k)
            self.assertEqual(len(p), 10)
            self.assertNotIn("s59", p)
        b30 = (0.75 * self.X["sc_balanced"].rank(pct=True) + 0.25 * self.X["msurge"].rank(pct=True)).drop("s59")
        self.assertTrue(set(ms.picks(self.X, "F")) <= set(b30.sort_values(ascending=False).index[:30]))


class Classify(unittest.TestCase):
    def test_kinds(self):
        base = {"ret": -3.0, "pre20": 5.0, "vs_industry": -0.5, "industry_ret": -2.5, "excess_net": -4.0}
        self.assertEqual(mr.classify(base, 0.01), "產業拖累")
        self.assertEqual(mr.classify({**base, "vs_industry": -3.0}, 0.01), "個股選錯")
        self.assertEqual(mr.classify({**base, "pre20": 20.0}, 0.01), "追高後回落")
        self.assertEqual(mr.classify({**base, "ret": None}, 0.01), "買不到")
        self.assertEqual(mr.classify({**base, "ret": 0.0, "excess_net": 0.5, "vs_industry": 0.0, "industry_ret": 2.0}, 0.01), "表現正常")
