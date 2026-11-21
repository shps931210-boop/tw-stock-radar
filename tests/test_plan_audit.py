"""2026-11-21 網站買賣結論回測（app/plan_audit.py）：當時的本益比位置、分組規則與 buyplan 一致、重疊期間的 t 值、網站摘要。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import buyplan
from app import plan_audit as pa


class PEAt(unittest.TestCase):
    def test_uses_only_published_eps_and_past_prices(self):
        days = [str(d.date()) for d in pd.bdate_range("2024-01-01", periods=400)]
        raw = pd.DataFrame({"stock_id": "1101", "date": days, "close": np.linspace(50, 100, 400)})
        fin = pd.DataFrame({"stock_id": "1101", "period": ["2022-Q4", "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"],
                            "eps": [1.0, 1.0, 1.0, 1.0, 2.0]})
        pe = pa.PE(raw, fin)
        cur, pct, p50 = pe.at("1101", "2024-03-29")         # 2023 年報 4/1 才公布：還在用 2023-Q3 的近四季 EPS 4
        self.assertAlmostEqual(cur, raw.set_index("date").loc["2024-03-29", "close"] / 4)
        self.assertIsNone(pct)                             # 不到 120 天歷史，不算位置
        cur2, pct2, _ = pe.at("1101", days[-1])
        self.assertAlmostEqual(cur2, 100 / 5)               # 2023-Q4 公布後：近四季 1+1+1+2
        self.assertIsNotNone(pct2)
        self.assertEqual(pe.at("9999", days[-1]), (None, None, None))


class Groups(unittest.TestCase):
    def test_groups_follow_buyplan_thresholds(self):
        T = pd.DataFrame({"rule": "swing", "score": [70, 70, 70, 70, 40, 50],
                          "pe_pct": [50, np.nan, 75, 90, 10, 20], "date": "2020-01-02"})
        G = pa.swing_groups(T)
        self.assertEqual(list(G["S-BUY"].index), [0, 1])     # 沒有本益比歷史不擋（和網站相同）
        self.assertEqual(list(G["S-WAIT"].index), [2])
        self.assertEqual(sorted(G["S-AVOID"].index), [3, 4])
        self.assertEqual(buyplan.GOOD_CO, 60)
        self.assertEqual((buyplan.RICH, buyplan.HOT), (70, 85))


class NW(unittest.TestCase):
    def test_overlap_lowers_t(self):
        rng = np.random.default_rng(0)
        x = np.convolve(rng.normal(0.01, 0.05, 80), np.ones(6) / 6, "valid")   # 重疊期間：相鄰高度相關
        naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        self.assertLess(abs(pa._nw_t(x)), abs(naive))
        self.assertIsNone(pa._nw_t(np.array([0.1, 0.2])))


class Summary(unittest.TestCase):
    def test_site_summary(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "a.json"
            sw = {"dev": {"month_excess": -0.25, "n": 10, "t": -0.5}, "holdout": {"month_excess": 0.5, "n": 3}, "passed": False}
            lg = {"dev_excess": -5.3, "dev_beat": 33, "holdout_excess": 1, "holdout_beat": 30, "periods_dev": 60, "passed": False}
            f.write_text(json.dumps({"study_version": "x", "data_end": "2026-11-18", "holdout_from": "2025-01-01", "long_hold": 120,
                                     "swing": {"S-BUY": sw}, "long": {"L-BUY": lg}}))
            s = pa.site_summary(f)
            self.assertEqual(s["swing"]["S-BUY"]["dev"], -0.25)
            self.assertFalse(s["long"]["L-BUY"]["passed"])
            self.assertIsNone(pa.site_summary(Path(t) / "missing.json"))


if __name__ == "__main__":
    unittest.main()
