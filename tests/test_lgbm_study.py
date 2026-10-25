"""V13 LightGBM 專項研究：比較清單事前固定、目標只用當時已知資料、多重比較門檻對得上挑戰者數量。"""
import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

from app import lgbm_study as ls


class Study(unittest.TestCase):
    def test_threshold_matches_number_of_challengers(self):
        k = len(ls.VARIANTS) - 1 + len(ls.FUSED)               # 基準以外的挑戰者
        self.assertEqual(k, 7)
        self.assertAlmostEqual(NormalDist().inv_cdf(1 - 0.05 / k / 2), ls.T_NEED, places=2)

    def test_targets(self):
        rng = np.random.default_rng(0)
        P = pd.DataFrame({"date": ["d1"] * 200 + ["d2"] * 200, "y20": rng.normal(0, 0.1, 400), "vol_1y": rng.uniform(10, 60, 400)})
        P.loc[3, "vol_1y"] = np.nan
        P.loc[5, "y20"] = 5.0                                     # 單一極端值
        T = ls._targets(P, "y20")
        self.assertTrue(T["tg_rank"].between(0, 1).all())
        self.assertLess(T.loc[5, "tg_excess"], 1.0)               # 截掉頭尾 1%
        self.assertTrue(np.isfinite(T["tg_riskadj"]).all())       # 缺波動率用當日中位數，不會變成無限大
        # 風險調整的分母只用訊號日已知的波動率（不低於當日第 5 百分位）
        r = T.iloc[10]
        floor = P[P["date"] == "d1"]["vol_1y"].quantile(0.05)
        self.assertAlmostEqual(r["tg_riskadj"], r["y20"] / max(r["vol_1y"], floor))

if __name__ == "__main__":
    unittest.main()
