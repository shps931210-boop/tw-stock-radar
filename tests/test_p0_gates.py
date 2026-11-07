"""2026-11-07 交接企劃書 P0：發布閘門（GZ-P0-03）與正式績效資格（GZ-P0-04）的故障注入／負面測試。"""
import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from app import fin_watch, publish_gate, track


def _setup(tmp: Path, files=("2026-10-20_h5.json", "2026-10-21_h5.json")):
    mat = tmp / "maturity"
    mat.mkdir()
    entries = []
    for name in files:
        p = mat / name
        p.write_text(json.dumps({"signal_date": name[:10]}), "utf-8")
        entries.append({"file": name, "ok": True, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (mat / "index.json").write_text(json.dumps({"generated_at": "2026-11-07T15:00:00"}), "utf-8")
    rc = {"generated_at": "2026-11-07T15:05:00", "all_ok": True, "violations": 0, "files": entries}
    return mat, rc


class PublishGate(unittest.TestCase):
    """8 種情境：前 7 種必須擋下，只有完整合法的對帳結果才發布。"""

    def run_case(self, mutate, files=("2026-10-20_h5.json", "2026-10-21_h5.json")):
        tmp = Path(tempfile.mkdtemp())
        mat, rc = _setup(tmp, files)
        p = tmp / "reconcile.json"
        out = mutate(mat, rc, p)
        if out is not False:
            p.write_text(json.dumps(rc), "utf-8")
        return publish_gate.check(p, mat)

    def test_valid_passes(self):
        g = self.run_case(lambda m, rc, p: None)
        self.assertTrue(g["ok"], g["why"])

    def test_missing_file_blocks(self):
        g = self.run_case(lambda m, rc, p: False)
        self.assertFalse(g["ok"])
        self.assertIn("找不到", g["why"][0])

    def test_malformed_blocks(self):
        g = self.run_case(lambda m, rc, p: (p.write_text("{not json", "utf-8"), False)[1])
        self.assertFalse(g["ok"])
        self.assertIn("格式錯誤", g["why"][0])

    def test_stale_blocks(self):
        g = self.run_case(lambda m, rc, p: rc.update(generated_at="2026-11-06T15:00:00"))
        self.assertFalse(g["ok"])
        self.assertTrue(any("舊" in w for w in g["why"]))

    def test_all_ok_false_or_truthy_non_bool_blocks(self):
        for bad in (False, "true", 1, None):
            g = self.run_case(lambda m, rc, p, b=bad: rc.update(all_ok=b))
            self.assertFalse(g["ok"], bad)
        g = self.run_case(lambda m, rc, p: rc.update(violations=1))
        self.assertFalse(g["ok"])
        g = self.run_case(lambda m, rc, p: rc.update(violations=True))
        self.assertFalse(g["ok"])

    def test_missing_batch_blocks(self):
        def add_unreconciled(m, rc, p):
            (m / "2026-10-22_h5.json").write_text("{}", "utf-8")
        g = self.run_case(add_unreconciled)
        self.assertFalse(g["ok"])
        self.assertIn("2026-10-22_h5.json", g["files"])

    def test_extra_batch_blocks(self):
        g = self.run_case(lambda m, rc, p: rc["files"].append({"file": "2026-10-23_h5.json", "ok": True, "sha256": "x"}))
        self.assertFalse(g["ok"])
        self.assertIn("2026-10-23_h5.json", g["files"])

    def test_checksum_mismatch_blocks(self):
        def tamper(m, rc, p):
            (m / "2026-10-20_h5.json").write_text(json.dumps({"signal_date": "2026-10-20", "x": 1}), "utf-8")
        g = self.run_case(tamper)
        self.assertFalse(g["ok"])
        self.assertIn("2026-10-20_h5.json", g["files"])
        g = self.run_case(lambda m, rc, p: rc["files"][0].pop("sha256"))
        self.assertFalse(g["ok"])

    def test_batch_not_ok_blocks(self):
        g = self.run_case(lambda m, rc, p: rc["files"][1].update(ok="yes"))
        self.assertFalse(g["ok"])


GOOD = {"signal_date": "2026-11-05", "generated_at": "2026-11-05T15:28:17", "data_version": {"prices": "2026-11-05"},
        "model_version": "2026-11-02-84867621e3", "trained_through": {"5": "2026-10-14", "20": "2026-09-29"},
        "time_check": {"trained_before_signal": True}}
ENTRY = "2026-11-06"


class PerfEligibleFailClosed(unittest.TestCase):
    def test_good_record_passes(self):
        self.assertEqual(track.perf_eligible(GOOD, ENTRY), (True, None))
        picks = {"date": "2026-11-05", "generated_at": "2026-11-05T15:25:07", "data_dates": {"prices": "2026-11-05"},
                 "model_version": "v1.1", "model_hash": "ce1e61207b30"}
        self.assertTrue(track.perf_eligible(picks, ENTRY)[0])
        acad = {"signal_date": "2026-11-05", "generated_at": "2026-11-05T15:28:50", "data_date": "2026-11-05", "candidate": "balanced_msurge"}
        self.assertTrue(track.perf_eligible(acad, ENTRY)[0])

    def test_ten_negative_cases(self):
        cases = {
            "缺訊號日": ({k: v for k, v in GOOD.items() if k != "signal_date"}, ENTRY),
            "缺產生時間": ({k: v for k, v in GOOD.items() if k != "generated_at"}, ENTRY),
            "缺進場日": (GOOD, None),
            "開盤後才產生": ({**GOOD, "generated_at": "2026-11-06T09:30:00"}, ENTRY),
            "產生時間早於訊號日": ({**GOOD, "generated_at": "2026-11-04T20:00:00"}, ENTRY),
            "進場日不在訊號日之後": (GOOD, "2026-11-05"),
            "缺資料日": ({k: v for k, v in GOOD.items() if k != "data_version"}, ENTRY),
            "資料日不同": ({**GOOD, "data_version": {"prices": "2026-11-04"}}, ENTRY),
            "缺模型版本": ({k: v for k, v in GOOD.items() if k != "model_version"}, ENTRY),
            "訓練截止不早於訊號日": ({**GOOD, "trained_through": {"5": "2026-11-05", "20": "2026-09-29"}}, ENTRY),
            "訓練截止是空的": ({**GOOD, "trained_through": {}}, ENTRY),
            "訓練時間檢查不通過": ({**GOOD, "time_check": {"trained_before_signal": False}}, ENTRY),
        }
        for name, (log, entry) in cases.items():
            ok, why = track.perf_eligible(log, entry)
            self.assertFalse(ok, name)
            self.assertTrue(why, name)


class IntegrationSeason(unittest.TestCase):
    """整合測試 11/2 起連續失敗的原因：11/2 官方 OpenAPI 換成 Q3，只有已公告的公司（283 家）。"""

    def test_season_quarter_in_november_is_q3(self):
        self.assertEqual(fin_watch.season_quarter(date(2026, 11, 6)), "2026-Q3")
        self.assertIsNone(fin_watch.season_quarter(date(2026, 12, 10)))


if __name__ == "__main__":
    unittest.main()
