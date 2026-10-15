import unittest

import numpy as np
import pandas as pd

from app import factor_research as fr


class FactorResearchTest(unittest.TestCase):
    def test_ic_perfect_and_reverse(self):
        s = pd.Series(np.arange(50, dtype=float))
        self.assertAlmostEqual(fr._ic(s, s * 2), 1.0)
        self.assertAlmostEqual(fr._ic(s, -s), -1.0)

    def test_ic_needs_enough_stocks(self):
        s = pd.Series(np.arange(10, dtype=float))
        self.assertIsNone(fr._ic(s, s))

    def test_ic_ignores_missing(self):
        s = pd.Series(np.arange(60, dtype=float))
        r = s.copy()
        r[:20] = np.nan
        self.assertAlmostEqual(fr._ic(s, r), 1.0)

    def test_industry_neutral_removes_industry_effect(self):
        # 分數只反映產業：扣掉產業平均後應該沒有預測力
        ind = pd.Series(["A"] * 30 + ["B"] * 30)
        score = pd.Series([1.0] * 30 + [2.0] * 30) + np.linspace(0, 0.01, 60)
        rng = np.random.default_rng(0)
        ret = pd.Series([0.0] * 30 + [0.1] * 30) + rng.normal(0, 0.01, 60)
        self.assertGreater(fr._ic(score, ret), 0.5)
        self.assertLess(abs(fr._ic(fr._demean(score, ind), fr._demean(ret, ind))), 0.4)

    def test_small_industries_grouped(self):
        g = fr._ind_groups(pd.Series(["A", "A", "A", "B", "", None]))
        self.assertEqual(list(g), ["A", "A", "A", "其他", "其他", "其他"])

    def test_cap_pick(self):
        ranked = [(f"s{i}", "半導體" if i < 8 else f"x{i}") for i in range(20)]
        self.assertEqual(len(fr._cap_pick(ranked, 10, None)), 10)
        pick = fr._cap_pick(ranked, 10, 0.2)
        self.assertEqual(sum(1 for s in pick if int(s[1:]) < 8), 2)
        self.assertEqual(len(pick), 10)
        self.assertEqual(sum(1 for s in fr._cap_pick(ranked, 10, 0.3) if int(s[1:]) < 8), 3)

    def test_cap_pick_blank_industry_counts_alone(self):
        ranked = [(f"s{i}", "") for i in range(5)]
        self.assertEqual(len(fr._cap_pick(ranked, 5, 0.2)), 5)

    def test_summary_t_value(self):
        x = fr._summ([0.05, 0.04, 0.06, 0.05, None])
        self.assertEqual(x["n"], 4)
        self.assertAlmostEqual(x["ic"], 0.05)
        self.assertEqual(x["pos"], 100.0)
        self.assertGreater(x["t"], 2)
        self.assertIsNone(fr._summ([0.1, None])["ic"])

    def test_quintiles(self):
        s = pd.Series(np.arange(100, dtype=float))
        q = fr._quint(s, s / 100)
        self.assertEqual(len(q), 5)
        self.assertTrue(all(a < b for a, b in zip(q, q[1:])))


    def test_t_pvalue_matches_table(self):
        self.assertAlmostEqual(fr.t_pvalue(2.101, 18), 0.05, places=3)   # t 分布表：df 18、雙尾 5%
        self.assertAlmostEqual(fr.t_pvalue(1.96, 100000), 0.05, places=3)
        self.assertAlmostEqual(fr.t_pvalue(0, 10), 1.0, places=3)

    def test_holm(self):
        adj = fr.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(adj, {"a": 0.03, "c": 0.06, "b": 0.06})
        self.assertEqual(fr.holm({"a": None}), {})

    def test_overlap_lowers_t(self):
        rng = np.random.default_rng(1)
        x = np.convolve(rng.normal(0.03, 0.05, 40), np.ones(3) / 3, "valid")  # 前後相關的序列
        self.assertLess(fr._summ(list(x), 3)["t"], fr._summ(list(x), 1)["t"])

    def test_summary_has_ci_and_p(self):
        x = fr._summ([0.05, 0.04, 0.06, 0.05, 0.03, 0.07])
        lo, hi = x["ci"]
        self.assertTrue(lo < x["ic"] < hi)
        self.assertLess(x["p"], 0.05)


if __name__ == "__main__":
    unittest.main()
