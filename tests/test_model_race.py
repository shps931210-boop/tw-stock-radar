"""V28 選股準確度：模型比賽的驗收規則、產業／市場特徵不用未來資料、稽核工具只改未來資料。"""
import unittest

import numpy as np
import pandas as pd

from app import leak_audit as la
from app import model_race as mr


class Verdict(unittest.TestCase):
    def test_needs_all_conditions(self):
        good = {"net_excess": 2.0, "mdd": -20.0}
        champ = {"net_excess": 0.0, "mdd": -18.0}
        v = mr.verdict(good, champ, {"net_excess": 1.0}, {"net_excess": 0.5}, {"ic": 0.08}, {"ic": 0.06}, 3.0)
        self.assertTrue(v["passed"])
        self.assertFalse(mr.verdict(good, champ, {"net_excess": 1.0}, {"net_excess": 0.5}, {"ic": 0.08}, {"ic": 0.06}, 2.0)["passed"])  # t 不夠
        self.assertFalse(mr.verdict(good, champ, {"net_excess": 0.1}, {"net_excess": 0.5}, {"ic": 0.08}, {"ic": 0.06}, 3.0)["passed"])  # 保留期較差
        self.assertFalse(mr.verdict({**good, "mdd": -30.0}, champ, {"net_excess": 1.0}, {"net_excess": 0.5}, {"ic": 0.08}, {"ic": 0.06}, 3.0)["passed"])


class Features(unittest.TestCase):
    def test_industry_and_market_features_use_same_day_only(self):
        P = pd.DataFrame({"date": ["d1"] * 6 + ["d2"] * 6, "sid": list("abcdef") * 2, "industry": ["X"] * 5 + ["Y"] + ["X"] * 5 + ["Y"],
                          "ret_1m": [1.0] * 6 + [9.0] * 6, "ret_3m": range(12), "mom_12_1": [0.0] * 12})
        A = mr.add_features(P, None)
        self.assertEqual(A.loc[0, "ind_ret_1m"], 1.0)    # d1 的產業平均不含 d2
        self.assertTrue(np.isnan(A.loc[5, "ind_ret_1m"]))  # 產業只有 1 檔：不算
        self.assertEqual(A.loc[6, "breadth"], 1.0)

    def test_portfolio_charges_cost_only_on_new_names(self):
        preds = pd.DataFrame({"date": ["d1"] * 12 + ["d2"] * 12, "sid": [f"s{i}" for i in range(12)] * 2,
                              "y20": [0.01] * 24, "v21": list(range(12)) * 2})
        r = mr.portfolio(preds, "v21", ["d1", "d2"], {})
        self.assertEqual(r["trades"], 10)   # 第二期名單相同，不再收成本
        self.assertAlmostEqual(r["series"][1], 1.0)


class Audit(unittest.TestCase):
    def test_future_copy_changes_only_future_rows(self):
        px = pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "close": [10.0, 10.0]})
        rev = pd.DataFrame({"ym": ["2025-11", "2025-12"], "revenue": [1.0, 1.0]})   # 12 月營收 1/11 才能用
        D = {"px": {"a": px}, "ex": {}, "rev": {"a": rev}, "fin": {}, "div": {}, "info": {}, "idx_s": None}
        E = la.future_copy(D, "2026-01-02")
        self.assertEqual(E["px"]["a"]["close"].iloc[0], 10.0)
        self.assertNotEqual(E["px"]["a"]["close"].iloc[1], 10.0)
        self.assertEqual(E["rev"]["a"]["revenue"].iloc[0], 1.0)
        self.assertNotEqual(E["rev"]["a"]["revenue"].iloc[1], 1.0)

    def test_purge_check(self):
        dates = [f"d{i:03d}" for i in range(200)]
        sigs = dates[::10][:15]
        self.assertTrue(la.purge_check(sigs, dates, 20, 3)["ok"])


if __name__ == "__main__":
    unittest.main()
