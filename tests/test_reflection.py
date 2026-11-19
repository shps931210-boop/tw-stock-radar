"""2026-11-18 每日反省（app/reflection.py）：數字怎麼翻成結論。"""
import unittest

from app import reflection as rf


def _fc(rows, h=5):
    """rows: [(0050 報酬, {mode: 淨報酬})]"""
    B = []
    for k, (b, m) in enumerate(rows):
        B.append({"signal_date": f"2026-10-{k + 1:02d}", "horizon": h, "eligible": True,
                  "modes": {x: {"open": {"net": v, "e0050": b}} for x, v in m.items()}})
    return B


class Reflection(unittest.TestCase):
    def test_low_beta_illusion_detected(self):
        # 只有 0050 的一半漲跌：大跌時贏、大漲時輸
        xs = [-4, 3, -2, 5, -6, 2, -3, 4, -1, 6, -5, 3]
        fc = {"batches": _fc([(x, {"stable": 0.5 * x}) for x in xs])}
        r = rf.build(fc, {}, [])
        codes = {f["code"] for f in r["modes"]["stable"]["findings"]}
        self.assertIn("low_beta", codes)
        self.assertAlmostEqual(r["modes"]["stable"]["h5"]["beta"], 0.5, places=2)

    def test_lose_0050_and_bad_alpha(self):
        xs = [-4, 3, -2, 5, -6, 2, -3, 4, -1, 6, -5, 3]
        fc = {"batches": _fc([(x, {"balanced": x - 2}) for x in xs])}
        r = rf.build(fc, {}, [])
        codes = {f["code"] for f in r["modes"]["balanced"]["findings"]}
        self.assertTrue({"lose_0050", "bad_alpha", "streak"} <= codes)
        self.assertIn("均衡型", r["headline"])

    def test_few_batches_no_conclusion(self):
        fc = {"batches": _fc([(1, {"stable": -5})] * 3)}
        f = rf.build(fc, {}, [])["modes"]["stable"]["findings"]
        self.assertEqual([x["code"] for x in f], ["few"])

    def test_ineligible_batches_ignored(self):
        B = _fc([(1, {"stable": -5})] * 10)
        for b in B:
            b["eligible"] = False
        r = rf.build({"batches": B}, {}, [])
        self.assertEqual(r["batches"]["h5"], 0)

    def test_overtest_flag(self):
        exp = {"items": [{"date": f"2026-10-{k + 1:02d}", "decision": "rejected"} for k in range(12)], "auto": []}
        p = rf.process_check(exp, {"batches": _fc([(1, {"stable": 1})], h=20)})
        codes = {f["code"] for f in p["findings"]}
        self.assertEqual(codes, {"overtest", "no_adopt"})

    def test_archive_written_once(self):
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        rf.run(d / "r.json", d / "arch")
        f = next((d / "arch").glob("*.json"))
        f.write_text("{}", "utf-8")
        rf.run(d / "r.json", d / "arch")
        self.assertEqual(f.read_text("utf-8"), "{}")   # 當天第一次的反省不被覆寫


if __name__ == "__main__":
    unittest.main()
