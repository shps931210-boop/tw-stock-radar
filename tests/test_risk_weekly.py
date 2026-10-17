"""風險規則逐項檢驗的選股規則、每週成績單的口徑。"""
import unittest

import numpy as np
import pandas as pd

from app import risk_research as rr


def frame():
    rows = []
    for i in range(20):
        rows.append({"id": f"{1000 + i}", "ind": "半導體業" if i < 8 else f"產業{i}", "base": True,
                     "vol": 30 + i, "ret3m": i * 5, "ma20": 20 if i == 1 else 3, "liq": 1e8 if i != 2 else 1e7,
                     "value": 10 if i == 3 else 50, **{f"rk_{m}": i + 1 for m in rr.ai_model.MODES}})
    df = pd.DataFrame(rows).set_index("id")
    base = df[df["base"]]
    df["vol_hi"] = df["vol"] > base["vol"].quantile(rr.VOL_PCT)
    df["hot"] = (df["ret3m"] > base["ret3m"].quantile(rr.HOT_PCT)) | (df["ma20"] > rr.HOT_MA20)
    return df


class Rules(unittest.TestCase):
    def test_original_is_rank_order(self):
        self.assertEqual(rr.picks(frame(), "aggressive", "A", 5), ["1000", "1001", "1002", "1003", "1004"])

    def test_each_rule_removes_only_its_target(self):
        df = frame()
        self.assertNotIn("1001", rr.picks(df, "aggressive", "F", 5))  # 高於 20 日均線 20%
        self.assertNotIn("1002", rr.picks(df, "aggressive", "G", 5))  # 成交金額不足
        self.assertNotIn("1003", rr.picks(df, "aggressive", "C", 5))  # 估值分數太低
        self.assertIn("1001", rr.picks(df, "aggressive", "C", 5))
        top_vol = rr.picks(df, "aggressive", "B", 20)
        self.assertNotIn("1019", top_vol)  # 波動率最高的 20% 剔除

    def test_industry_cap(self):
        p = rr.picks(frame(), "aggressive", "D", 10)
        self.assertEqual(sum(1 for s in p if int(s) < 1008), 3)  # 同產業最多 30%

    def test_verdict(self):
        v = rr._verdict(np.array([0.01, 0.02, 0.015, 0.01, 0.012, 0.02]))
        self.assertEqual(v["verdict"], "pass")
        v = rr._verdict(np.array([0.05, -0.04, 0.03, -0.02]))
        self.assertNotEqual(v["verdict"], "pass")

    def test_relative_drawdown(self):
        self.assertEqual(rr._rel_mdd([0.0, 0.0], [0.1, 0.0]), round((1 / 1.1 - 1) * 100, 2))


if __name__ == "__main__":
    unittest.main()
