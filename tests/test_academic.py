"""V21.1 學術假說研究：特徵只用當時看得到的資料、方向與公式、DOI 核對。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import academic_study as ac
from app import literature


def _rev(n=30, start="2024-01", bump=None):
    ym = [str(p) for p in pd.period_range(start, periods=n, freq="M")]
    rev = [100.0 + i + (i % 5) for i in range(n)]   # 每個月年增額略有起伏
    if bump:
        rev[ym.index(bump)] += 50
    return pd.DataFrame({"stock_id": "x", "ym": ym, "revenue": rev})


class Revenue(unittest.TestCase):
    def test_not_used_before_11th(self):
        R = _rev(30, bump="2026-05")
        a = ac._rev_feats(R, "2026-06-10")   # 5 月營收 6/11 才能用
        b = ac._rev_feats(R, "2026-06-11")
        self.assertLess(a["msurge"], 3)   # 6/10 只看得到 4 月營收，5 月的驚喜還不能用
        self.assertGreater(b["msurge"], 3)

    def test_yoy_and_steady_surprise(self):
        f = ac._rev_feats(_rev(30), "2026-06-11")
        self.assertAlmostEqual(f["rev_yoy"], (100 + 28 + 3) / (100 + 16 + 1) - 1)
        self.assertLess(abs(f["msurge"]), 3)   # 沒有異常的月份 → 驚喜不大

    def test_stale_revenue_ignored(self):
        self.assertEqual(ac._rev_feats(_rev(20), "2026-06-11"), {})


class Fin(unittest.TestCase):
    def _F(self):
        per = [f"{y}-Q{q}" for y in (2025, 2026) for q in (1, 2, 3, 4)][:6]
        return pd.DataFrame({"period": per, "period_end": [f"2025-{3 * i + 3:02d}-28" if i < 4 else f"2026-{3 * (i - 4) + 3:02d}-28" for i in range(6)],
                             "gross_profit": [10.0] * 6, "net_income": [5.0] * 6, "ocf": [6.0] * 6,
                             "total_assets": [100, 100, 100, 100, 110, 120]})

    def test_only_published_quarters(self):
        f = ac._fin_feats(self._F(), "2026-05-20", False)   # 2026-Q1 已公布（5/16 起）
        self.assertAlmostEqual(f["gp_assets"], 40 / 110)
        self.assertAlmostEqual(f["low_asset_growth"], -(110 / 100 - 1))
        self.assertGreater(f["low_accruals"], 0)          # 現金流 > 淨利 → 應計為負 → 分數為正
        self.assertAlmostEqual(f["ocf_ni"], 1.2)
        g = ac._fin_feats(self._F(), "2026-05-10", False)  # 還沒公布 Q1
        self.assertEqual(g, {})                             # 只有 4 季可用，不足 5 季不算

    def test_financial_sector_skipped(self):
        self.assertEqual(ac._fin_feats(self._F(), "2026-09-01", True), {})


class Price(unittest.TestCase):
    def test_directions(self):
        idx = [f"d{i:04d}" for i in range(300)]
        c = np.linspace(50, 100, 300)
        c[-1] = 90   # 最後一天大跌
        a = pd.DataFrame({"close": c, "value": 1e8}, index=idx)
        f = ac._px_feats(a, idx[-1])
        self.assertAlmostEqual(f["hi52"], 90 / c[-2])
        self.assertGreater(f["st_reversal"], 0 - 1)        # 近 20 日跌 → 反轉分數較高
        self.assertLessEqual(f["low_max"], 0)
        self.assertGreater(f["illiq"], 0)
        self.assertEqual(ac._px_feats(a.iloc[:200], idx[199]), {})   # 不足一年不算


class Doi(unittest.TestCase):
    def test_compare_and_verify(self):
        it = {"title": "Do Industries Explain Momentum?", "year": 1999}
        c = literature.compare(it, {"title": ["Do Industries Explain Momentum?"], "issued": {"date-parts": [[1999, 8]]}, "container-title": ["JF"]})
        self.assertTrue(c["ok"]) and self.assertTrue(c["year_match"])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "lib.json"
            p.write_text(json.dumps({"items": [{"id": "P1", "doi": "10.1/x", "title": "A B", "year": 2000}, {"id": "P2", "doi": None, "title": "C", "year": 2001}]}), "utf-8")

            class R:
                status_code = 200
                def json(self):
                    return {"message": {"title": ["A B"], "issued": {"date-parts": [[2000]]}}}
            out = literature.verify(p, sleep=0, get=lambda *a, **k: R())
            self.assertEqual(out["ok"], 1)
            L = json.loads(p.read_text("utf-8"))
            self.assertIsNone(L["items"][1]["doi_check"]["ok"])

    def test_library_well_formed(self):
        L = json.loads(Path("history/research/literature.json").read_text("utf-8"))
        self.assertEqual(len(L["items"]), 31)
        used = {f for x in L["items"] for f in x["used_by"]}
        for f, (_, _, ref, _) in ac.FACTORS.items():
            self.assertIn(f, used, f)
            self.assertTrue(any(x["id"] == ref for x in L["items"]))


if __name__ == "__main__":
    unittest.main()


class Shadow(unittest.TestCase):
    def test_build_blend_and_base(self):
        from app import shadow_academic as sa
        rows = [{"stock_id": f"s{i}", "name": f"n{i}", "ai": {"modes": {"balanced": {"score": 90 - i, "rank": i + 1}}}} for i in range(30)]
        rows.append({"stock_id": "x", "ai": {"modes": {"balanced": {"score": None, "rank": None}}}})   # 不可排名的不算
        rev = {"s29": _rev(30, bump="2026-05")}   # 排最後但營收大驚喜
        rec = sa.build(rows, rev, "2026-06-11")
        self.assertEqual([r["stock_id"] for r in rec["lists"]["balanced"]], [f"s{i}" for i in range(10)])
        self.assertEqual(rec["universe_n"], 30)
        self.assertEqual(len(rec["lists"][sa.CANDIDATE]), 10)
        self.assertEqual(rec["status"], "shadow")


class Layers(unittest.TestCase):
    def test_site_and_seal_layers(self):
        from app import shadow_academic as sa
        rows = [{"stock_id": f"s{i}", "name": f"n{i}", "score_quality": i, "score_growth": i, "score_value": i, "score_trend": 30 - i,
                 "ret_20d": i, "ai": {"modes": {"balanced": {"score": 90 - i, "rank": i + 1}}}} for i in range(30)]
        with tempfile.TemporaryDirectory() as t:
            out = sa.site(rows, {}, "2026-06-11", Path(t) / "tl.json")
            v29, e29 = out["stocks"]["s29"]
            v0, e0 = out["stocks"]["s0"]
            self.assertGreater(v29, v0)      # 品質、成長、價值都最高
            self.assertLess(e29, e0)         # 趨勢最低、近 20 日漲最多（沒有回檔）
            self.assertIn("研究中", out["status"])
        rec = sa.build(rows, {}, "2026-06-11")
        self.assertEqual(len(rec["layers"]), 30)
