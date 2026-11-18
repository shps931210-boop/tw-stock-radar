"""2026-11-18 買賣計畫（app/buyplan.py）：三件事怎麼組合成結論、價位從哪裡來。"""
import unittest

from app import buyplan as bp

V = {"p50": 15.0, "pct": 50.0, "pe": 15.0, "low": 90.0, "base": 100.0, "high": 120.0, "peer": 14.0}
ST = {"ready": True, "trend": {"label": "偏多"},
      "swing": {"state": "triggered", "stop": 95.0, "target": 110.0, "risk_pct": 5.0, "miss": []}}


def row(score=70, close=100.0, dist60=5.0, hot=None, complete=True):
    return {"close": close, "complete": complete, "dist_ma60": dist60, "timing": {"hot": hot or []},
            "ai": {"modes": {"balanced": {"score": score}}}}


class BuyPlan(unittest.TestCase):
    def test_all_three_ok_is_buy(self):
        p = bp.build(row(), ST, V)
        self.assertEqual(p["key"], "buy")
        self.assertEqual(p["stop"], 95.0)
        self.assertIn("103.00", p["entry"])          # 開盤高於收盤 3% 不追
        self.assertEqual(p["fair"]["base"], 100.0)
        self.assertAlmostEqual(p["ma60"], 100 / 1.05, places=2)

    def test_expensive_good_company_waits(self):
        p = bp.build(row(), ST, {**V, "pct": 75.0})
        self.assertEqual(p["key"], "wait_price")
        self.assertIn("100.00", p["entry"])

    def test_overheated_or_weak_is_avoid(self):
        self.assertEqual(bp.build(row(), ST, {**V, "pct": 90.0})["key"], "avoid")
        self.assertEqual(bp.build(row(score=30), ST, V)["key"], "avoid")
        down = {**ST, "trend": {"label": "偏弱"}}
        self.assertEqual(bp.build(row(dist60=-3), down, V)["key"], "avoid")

    def test_no_signal_waits_for_signal(self):
        st = {**ST, "swing": {**ST["swing"], "state": "setup", "miss": ["等回檔到月線"]}}
        p = bp.build(row(), st, V)
        self.assertEqual(p["key"], "wait_signal")
        self.assertIn("等回檔到月線", p["entry"])

    def test_hot_risk_blocks_buy(self):
        self.assertNotEqual(bp.build(row(hot=["近 5 日漲 25%"]), ST, V)["key"], "buy")

    def test_missing_everything_is_na(self):
        p = bp.build(row(complete=False), {"ready": False}, {})
        self.assertEqual(p["key"], "na")
        self.assertIsNone(p["fair"])

    def test_peer_fallback_when_no_history(self):
        p = bp.build(row(), ST, {"pe": 12.0, "peer": 14.0})
        self.assertEqual(p["key"], "buy")
        self.assertTrue(any("同業" in w for w in p["why"]))


if __name__ == "__main__":
    unittest.main()
