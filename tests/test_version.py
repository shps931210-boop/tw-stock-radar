import json
import unittest
from datetime import datetime

from app import version


def snap(**kw):
    s = {"source": "twse", "stocks": [{"stock_id": "2330"}], "data_dates": {"prices": "2026-10-13"}, "generated_at": "2026-10-13T14:50:00",
         "meta": {"last_update": "2026-10-13T14:45:00"}, "health": [{"source": "twse_quotes", "label": "上市股價", "last_ok": "2026-10-13"}],
         "history": {"days": 300, "need": 260, "ready": True}, "expected_date": "2026-10-13", "expected_basis": "official"}
    s.update(kw)
    return s


class Version(unittest.TestCase):
    def test_states(self):
        self.assertEqual(version.state(snap())[0], "postclose_final")
        self.assertEqual(version.state(snap(expected_date="2026-10-14"))[0], "stale")
        self.assertEqual(version.state(snap(expected_date="2026-10-14", expected_basis="estimated"))[0], "postclose_partial")
        self.assertEqual(version.state(snap(health=[{"source": "tpex_quotes", "label": "上櫃股價", "last_ok": "2026-10-12"}]))[0], "postclose_partial")
        self.assertEqual(version.state(snap(history={"days": 30, "need": 260, "ready": False}))[0], "postclose_partial")
        self.assertEqual(version.state(None)[0], "unavailable")

    def test_version_ignores_run_timestamps_but_not_data(self):
        a = snap()
        b = snap(generated_at="2026-10-13T16:35:00", meta={"last_update": "2026-10-13T16:31:00"})
        c = snap(data_dates={"prices": "2026-10-14"})
        f = lambda s: {"snapshot.json": json.dumps(s).encode()}
        self.assertEqual(version.build(a, f(a))["data_version"], version.build(b, f(b))["data_version"])
        self.assertNotEqual(version.build(a, f(a))["data_version"], version.build(c, f(c))["data_version"])

    def test_times_are_iso_with_timezone(self):
        v = version.build(snap(), {}, datetime(2026, 10, 13, 15, 0, tzinfo=version.TW))
        self.assertEqual(v["calculated_at"], "2026-10-13T14:50:00+08:00")
        self.assertEqual(v["fetched_at"], "2026-10-13T14:45:00+08:00")
        self.assertTrue(v["published_at"].endswith("+08:00"))
        self.assertEqual(v["trading_date"], "2026-10-13")

    def test_next_scheduled_skips_weekend(self):
        self.assertEqual(version.next_run(datetime(2026, 10, 13, 15, 0, tzinfo=version.TW)), "2026-10-13T16:30+08:00")
        self.assertEqual(version.next_run(datetime(2026, 10, 16, 21, 30, tzinfo=version.TW)), "2026-10-19T14:40+08:00")

    def test_schedule_matches_workflow(self):
        wf = open(".github/workflows/update.yml", encoding="utf-8").read()
        for h, m in version.SCHEDULE:
            utc = (h - 8) % 24
            self.assertIn(f'cron: "{m} {utc} * * 1-5"', wf)


if __name__ == "__main__":
    unittest.main()
