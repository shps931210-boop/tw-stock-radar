"""2026-11-17「不輸 0050」研究（app/bench_study）的權重與報酬計算。"""
import unittest

from app import bench_study as bs


def _period(bench=0.02, ret=None):
    big = {f"S{i}": float(100 - i) for i in range(30)}
    rk = {f"S{i}": i + 1 for i in range(30)}   # S0 排第 1
    ret = ret or {s: 0.01 for s in big}
    return {"signal": "2020-01-02", "bench": bench, "big": big, "ret": ret, "regime": "bull",
            "rk": {m: rk for m in ("stable", "balanced", "aggressive")},
            "top": {m: [f"S{i}" for i in range(10)] for m in ("stable", "balanced", "aggressive")}}


class Weights(unittest.TestCase):
    def test_core_share(self):
        w = bs.weights(_period(), "balanced", "A70")
        self.assertAlmostEqual(sum(w.values()), 0.3)
        self.assertEqual(len(w), 10)

    def test_b0_is_cap_weight(self):
        P = _period()
        w = bs.weights(P, "balanced", "B0")
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["S0"] / w["S29"], 100 / 71)

    def test_tilts(self):
        P = _period()
        b0, b1, b2 = (bs.weights(P, "balanced", r) for r in ("B0", "B1", "B2"))
        self.assertGreater(b1["S0"], b0["S0"])
        self.assertLess(b1["S29"], b0["S29"])
        self.assertNotIn("S29", b2)          # 最後 10 名 ×0 → 不持有
        self.assertAlmostEqual(sum(b2.values()), 1.0)


class Series(unittest.TestCase):
    def test_core_excess_scales(self):
        """做法 A 的超額 = (1 − 核心比例) ×（前 10 名 − 0050），只扣成本差異。"""
        ps = [_period(bench=0.05), _period(bench=0.05)]
        top = bs.series(ps, "balanced", "TOP10")
        a70 = bs.series(ps, "balanced", "A70")
        # 第二期沒換股：沒有成本
        self.assertAlmostEqual(top[1] - 0.05, -0.04)
        self.assertAlmostEqual(a70[1] - 0.05, 0.3 * -0.04)

    def test_unfilled_is_cash(self):
        P = _period(ret={f"S{i}": (None if i == 0 else 0.0) for i in range(30)})
        r = bs.series([P], "balanced", "TOP10")[0]
        self.assertAlmostEqual(r, -bs.COST * 1.0)

    def test_gate(self):
        s = {"mean_excess": 0.1, "lag3_share": 5.0, "mdd": -10.0, "mdd_bench": -9.0}
        self.assertTrue(bs.gate(s, s, s, s)["passed"])
        self.assertFalse(bs.gate(s, s, s, {**s, "lag3_share": 20.0})["passed"])
        self.assertFalse(bs.gate(s, {**s, "mean_excess": -0.1}, s, s)["passed"])


if __name__ == "__main__":
    unittest.main()
