"""V20.1 預測封存日曆與每日學習「有沒有新資料」：休市原因、缺漏不補、到期日、對帳統計落後警示。"""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from app import daily_learning, seal_calendar


class Calendar(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        r = Path(self.t.name)
        self.pred, self.picks = r / "pred", r / "picks"
        self.pred.mkdir(); self.picks.mkdir()
        for d, g in (("2026-10-16", "2026-10-17T12:00:00"), ("2026-10-19", "2026-10-20T10:00:00"), ("2026-10-22", "2026-10-22T15:30:00")):
            (self.pred / f"{d}.json").write_text(json.dumps({"signal_date": d, "generated_at": g, "n": 700}))
            (self.picks / f"{d}.json").write_text("{}")
        (self.pred / "scorecard.json").write_text(json.dumps({"last": "2026-10-22", "horizons": {"5": {"matured": 0}}}))
        (r / "hol.json").write_text(json.dumps({"dates": ["2026-10-26"]}))
        (r / "notes.json").write_text(json.dumps({"2026-10-23": "颱風休市"}))
        self.kw = dict(pred=self.pred, picks=self.picks, hol=r / "hol.json", notes=r / "notes.json")

    def tearDown(self):
        self.t.cleanup()

    def test_gaps_closures_and_maturity(self):
        days = ["2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
        c = seal_calendar.build(today=date(2026, 10, 26), data_days=days, **self.kw)
        S, D = c["summary"], {d["date"]: d for d in c["days"]}
        self.assertEqual(S["expected"], 5)
        self.assertEqual(S["missing"], ["2026-10-20", "2026-10-21"])  # 沒封存就是缺漏，不補做
        self.assertIn("缺漏", D["2026-10-20"]["reason"])
        self.assertFalse(D["2026-10-19"]["on_time"])  # 10/20 10:00 才封存，晚於進場日開盤
        self.assertEqual(D["2026-10-23"]["reason"], "颱風休市")
        self.assertFalse(D["2026-10-26"]["trading"])
        self.assertNotIn("2026-10-24", D)  # 週末不列
        self.assertEqual(D["2026-10-16"]["maturity"]["5"]["due"], "2026-10-28")  # 進場 10/19，跳過 10/23、10/26
        self.assertEqual(D["2026-10-22"]["entry"], "2026-10-27")

    def test_scorecard_lag_flagged(self):
        days = [f"2026-10-{d:02d}" for d in (16, 19, 20, 21, 22, 27, 28, 29)]
        c = seal_calendar.build(today=date(2026, 10, 29), data_days=days, **self.kw)
        self.assertEqual(c["summary"]["matured"]["5"], 2)  # 10/16（到期 10/28）、10/19（到期 10/29）
        self.assertEqual(c["summary"]["scorecard_lag"], {"5": 2})  # 對帳統計還是 0：要警示
        self.assertIn("2026-10-27", c["summary"]["missing"])  # 有行情但沒封存


class LearningNewData(unittest.TestCase):
    def test_success_without_new_trading_day_is_labelled(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "runs.json"
            for i, d in enumerate(("2026-10-22", "2026-10-22", "2026-10-22")):
                with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": str(i)}):
                    daily_learning.record(True, None, d, 10, path=p, learned={"matured": 0, "labels": 0})
            runs = json.loads(p.read_text())["runs"]
            st = daily_learning.run_status(p)
        self.assertEqual([r["new_day"] for r in runs], [True, False, False])
        self.assertEqual(st["no_new_data_streak"], 2)
        self.assertEqual(st["last_new_data"], "2026-10-22")
        self.assertIn("沒有新交易日", st["data_label"])
        self.assertEqual(st["fail_streak"], 0)  # 流程本身是成功的
