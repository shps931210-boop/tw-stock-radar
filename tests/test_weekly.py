"""每週固定名單：正式精選不含流動性不合格股票、診斷、影子模型只用封存預測。"""
import unittest

from app import track, weekly


class OfficialList(unittest.TestCase):
    def test_rec_picks_exclude_illiquid(self):
        def st(sid, rank, rec_rank, elig, g):
            return {"stock_id": sid, "name": sid, "is_common": True, "liquidity_grade": g, "close": 10, "industry": "X",
                    "ai": {"modes": {"stable": {"rank": rank, "score": 90 - rank}},
                           "rec": {"stable": {"rank": rec_rank, "recommendation_eligibility": elig}}}}
        snap = {"ai_modes": {"stable": {}}, "stocks": [st("1615", 1, None, "excluded", "F"), st("2330", 2, 1, "normal", "A"), st("6651", 3, 11, "capped", "D")]}
        rp = track._rec_picks(snap)["stable"]
        self.assertEqual([p["stock_id"] for p in rp], ["2330", "6651"])
        self.assertNotIn("F", [p["liq"] for p in rp])


class Diagnose(unittest.TestCase):
    def test_worst_industry_and_liquidity_effect(self):
        rows = [{"stock_id": str(i), "name": "", "industry": "A" if i < 5 else "B", "ret": float(i - 5),
                 "factors": {"momentum": 90 if i > 5 else 10}} for i in range(10)]
        g = weekly.diagnose(rows, {"net": -1.0}, {"net": 0.5})
        self.assertEqual(g["worst"][0]["stock_id"], "0")
        self.assertEqual(g["industry"][0]["industry"], "A")
        self.assertEqual(g["liquidity_effect"], 1.5)
        self.assertGreater(g["factor_gap"]["momentum"], 0)

    def test_ml_picks_only_from_sealed_file(self):
        self.assertEqual(weekly._ml_picks("1999-01-01"), {})   # 沒有當天封存的預測：不事後補算


if __name__ == "__main__":
    unittest.main()
