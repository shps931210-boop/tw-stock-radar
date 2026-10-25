"""V13 正式推薦資格：excluded、E／F、處置股不進正式名單；合格股不足 10 檔時照實際數量，不用受限股票補滿；績效中心另算正式名單。"""
import unittest

from app import liquidity, track
from tests.test_liquidity import stock


def few_normal():
    # 只有 6 檔流動性正常，另外 5 檔是 D 級（受限）、2 檔 F 級、1 檔處置股
    rows = [stock(f"2{i:03d}", 95 - i, 2e9) for i in range(6)] + [stock(f"3{i:03d}", 99 - i, 5e7) for i in range(5)]
    rows += [stock("4001", 100, 5e6), stock("4002", 98, 5e6)]
    dis = stock("5001", 97, 3e9); dis["disposition"] = True; rows.append(dis)
    for i, r in enumerate(sorted(rows, key=lambda r: -r["ai"]["modes"]["balanced"]["score"]), 1):
        r["ai"]["modes"]["balanced"]["rank"] = i
        r["is_common"], r["name"], r["complete"] = True, r["stock_id"], True
    liquidity.annotate(rows, ["balanced"])
    for r in rows:
        r["ai"].setdefault("rec", {})
    return rows


class Eligibility(unittest.TestCase):
    def setUp(self):
        self.rows = few_normal()
        self.by = {r["stock_id"]: r for r in self.rows}

    def rec(self, k="balanced"):
        return self.by_rank([r for r in self.rows if r["ai"]["rec"][k]["rank"]], k)

    @staticmethod
    def by_rank(xs, k):
        return sorted(xs, key=lambda r: r["ai"]["rec"][k]["rank"])

    def test_top10_has_only_eligible_and_is_not_padded(self):
        top10 = [r for r in self.rec() if r["ai"]["rec"]["balanced"]["rank"] <= liquidity.TOP]
        self.assertEqual(len(top10), 6)                                  # 只有 6 檔合格，不補滿 10 檔
        self.assertTrue(all(r["ai"]["rec"]["balanced"]["recommendation_eligibility"] == "normal" for r in top10))
        caps = [r["ai"]["rec"]["balanced"]["rank"] for r in self.rows if r["ai"]["rec"]["balanced"]["recommendation_eligibility"] == "capped"]
        self.assertEqual(min(caps), liquidity.TOP + 1)                     # 受限股票從第 11 名開始

    def test_excluded_and_disposition_never_ranked(self):
        for sid in ("4001", "4002", "5001"):
            r = self.by[sid]["ai"]["rec"]["balanced"]
            self.assertEqual(r["recommendation_eligibility"], "excluded", sid)
            self.assertIsNone(r["rank"])
        self.assertIn("處置", self.by["5001"]["ai"]["rec"]["balanced"]["recommendation_cap_reason"])
        self.assertEqual(self.by["5001"]["ai"]["modes"]["balanced"]["rank"], 6)   # 原始名次照舊，研究排行榜還看得到

    def test_logged_official_list_guard(self):
        snap = {"stocks": self.rows, "ai_modes": {"balanced": {}}}
        ids = [p["stock_id"] for p in track._rec_picks(snap)["balanced"]]
        for sid in ("4001", "4002", "5001"):
            self.assertNotIn(sid, ids)
        self.rows[0]["liquidity_grade"] = "F"                                # 就算上游標錯，紀錄時也會再擋一次
        self.assertFalse(track.eligible(self.rows[0], "balanced"))


if __name__ == "__main__":
    unittest.main()
