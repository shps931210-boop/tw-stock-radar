"""V26 盤前情報：週末事件、週一整合、不用未來資訊、休市日、不可覆寫、來源失效、資料過期、跳空不算報酬。"""
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

from app import db
from app import premarket as pm

H = {"dates": ["2026-10-26"], "ok": True}   # 假設 10/26（週一）休市，用來測休市日
SNAP = {"ai_modes": {"balanced": {"name": "均衡型"}},
        "stocks": [{"stock_id": "2330", "name": "台積電", "industry": "半導體業", "ai": {"modes": {"balanced": {"rank": 1}}}},
                   {"stock_id": "2603", "name": "長榮", "industry": "航運業", "ai": {"modes": {"balanced": {"rank": 2}}}}]}
FRESH_US = {"markets": [{"symbol": "^SOX", "name": "費城半導體指數", "date": "2026-10-16", "chg_pct": -3.5},
                        {"symbol": "^IXIC", "name": "那斯達克", "date": "2026-10-16", "chg_pct": -0.5}],
            "fx_usd": {"date": "2026-10-16", "chg": 0.02}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.dbp, self.pre, self.pmd = str(t / "s.db"), t / "pred" / "pre_open", t / "pm"
        with db.connect(self.dbp) as con:
            con.executescript(pm.N.SCHEMA)
            con.executemany("INSERT INTO stocks (stock_id, name, market, industry) VALUES (?,?,?,?)",
                            [("2330", "台積電", "twse", "半導體業"), ("2603", "長榮", "twse", "航運業")])
            con.execute("INSERT INTO prices (stock_id, date, close) VALUES ('2330', '2026-10-16', 1000)")

    def tearDown(self):
        self.tmp.cleanup()

    def news(self, i, d, t, title, kind="announcement", sid="2330"):
        with db.connect(self.dbp) as con:
            con.execute("INSERT INTO news_items (id, kind, stock_id, date, time, title, source, fetched_at) VALUES (?,?,?,?,?,?,?,?)",
                        (i, kind, sid, d, t, title, "公開資訊觀測站", f"{d}T{t}:30"))

    def run_at(self, ts, G=FRESH_US):
        return pm.run(datetime.fromisoformat(ts), self.dbp, SNAP, G, H, self.pre, self.pmd)


class Weekend(Base):
    def test_weekend_events_are_saved_for_next_trading_day(self):
        self.news("a", "2026-10-17", "10:00", "本公司董事會決議停工")
        out = self.run_at("2026-10-17T12:00:00")
        self.assertEqual(out["target_date"], "2026-10-19")
        saved = json.loads((self.pmd / "events" / "2026-10-19.json").read_text("utf-8"))
        self.assertEqual([e["id"] for e in saved["events"]], ["a"])
        self.assertFalse((self.pre / "2026-10-19.json").exists())  # 週末只收集，不封存

    def test_monday_pre_open_integrates_friday_to_monday_and_keeps_first_seen(self):
        self.news("fri", "2026-10-16", "15:10", "公告本公司重大損失")
        self.news("sat", "2026-10-17", "09:00", "本公司接獲大單")
        self.run_at("2026-10-17T10:00:00")
        first = json.loads((self.pmd / "events" / "2026-10-19.json").read_text("utf-8"))
        seen = {e["id"]: e["first_seen_at"] for e in first["events"]}
        self.news("sun", "2026-10-18", "20:00", "本公司遭裁罰")
        self.news("before_close", "2026-10-16", "11:00", "盤中公告（收盤前就已反映）")
        out = self.run_at("2026-10-19T08:15:00")
        self.assertEqual(out["status"]["mode"], "sealed")
        rec = json.loads((self.pre / "2026-10-19.json").read_text("utf-8"))
        ids = {e["id"] for e in rec["events"]}
        self.assertEqual(ids, {"fri", "sat", "sun"})
        self.assertEqual({e["id"]: e["first_seen_at"] for e in rec["events"] if e["id"] in seen}, seen)
        self.assertEqual(rec["prediction_type"], "pre_open")
        self.assertFalse(rec["scores_changed"])
        self.assertEqual(rec["flags"][0]["id"], "2330")
        self.assertEqual(rec["flags"][0]["flag"], "風險上升")


class NoFuture(Base):
    def test_events_after_cutoff_are_not_used(self):
        self.news("late", "2026-10-19", "08:30", "本公司重大損失")
        self.run_at("2026-10-19T08:15:00")
        rec = json.loads((self.pre / "2026-10-19.json").read_text("utf-8"))
        self.assertNotIn("late", {e["id"] for e in rec["events"]})

    def test_seal_refuses_future_events(self):
        ev = [{"id": "x", "published_at": "2026-10-19T08:30:00"}]
        with self.assertRaises(ValueError):
            pm.seal(self.pre / "2026-10-19.json", datetime(2026, 10, 19, 8, 15), pd.Timestamp("2026-10-19").date(), "2026-10-16",
                    ev, {}, [], {}, {})

    def test_after_open_run_does_not_seal(self):
        out = self.run_at("2026-10-19T09:05:00")
        self.assertEqual(out["target_date"], "2026-10-20")
        self.assertFalse((self.pre / "2026-10-19.json").exists())


class Calendar(Base):
    def test_holiday_produces_no_prediction(self):
        with db.connect(self.dbp) as con:
            con.execute("INSERT INTO prices (stock_id, date, close) VALUES ('2330', '2026-10-23', 1000)")
        out = self.run_at("2026-10-26T08:15:00")
        self.assertFalse(out["today_trading"])
        self.assertEqual(out["target_date"], "2026-10-27")
        self.assertFalse(list(self.pre.glob("*.json")) if self.pre.exists() else [])

    def test_holiday_list_skips_trading_start_end_rows(self):
        rows = [{"Name": "中華民國開國紀念日", "Date": "1160101"}, {"Name": "春節前最後交易日", "Date": "1160210"}]
        self.assertEqual(pm.parse_holidays(rows), ["2027-01-01"])

    def test_calendar_failure_degrades_to_weekends(self):
        with mock.patch.object(pm, "HOLIDAYS", Path(self.tmp.name) / "none.json"), mock.patch("requests.get", side_effect=OSError("down")):
            Hc = pm.holidays()
        self.assertFalse(Hc["ok"])
        self.assertTrue(pm.is_trading_day(pd.Timestamp("2026-10-19").date(), Hc))
        self.assertFalse(pm.is_trading_day(pd.Timestamp("2026-10-18").date(), Hc))


class WriteOnce(Base):
    def test_second_run_does_not_overwrite_and_late_events_become_alerts(self):
        self.news("a", "2026-10-17", "10:00", "本公司重大損失")
        self.run_at("2026-10-19T08:15:00")
        p = self.pre / "2026-10-19.json"
        before = p.read_text("utf-8")
        self.news("b", "2026-10-19", "08:20", "本公司董事長辭職")
        out = self.run_at("2026-10-19T08:30:00")
        self.assertEqual(out["status"]["mode"], "already_sealed")
        self.assertEqual(p.read_text("utf-8"), before)
        alerts = json.loads((self.pre / "2026-10-19.alerts.json").read_text("utf-8"))["alerts"]
        self.assertEqual([a["id"] for a in alerts], ["b"])
        self.news("c", "2026-10-19", "10:40", "本公司遭停止買賣")
        self.run_at("2026-10-19T11:30:00")   # 盤中檢查：目標日已是下一個交易日，但警示仍掛在今天的盤前版
        alerts = json.loads((self.pre / "2026-10-19.alerts.json").read_text("utf-8"))["alerts"]
        self.assertEqual([a["id"] for a in alerts], ["b", "c"])
        with self.assertRaises(FileExistsError):
            pm.seal(p, datetime(2026, 10, 19, 8, 40), pd.Timestamp("2026-10-19").date(), "2026-10-16", [], {}, [], {}, {})

    def test_after_close_file_is_separate(self):
        ac = self.pre.parent / "2026-10-16.json"
        ac.parent.mkdir(parents=True)
        ac.write_text('{"model_version": "v1"}', "utf-8")
        self.run_at("2026-10-19T08:15:00")
        self.assertEqual(ac.read_text("utf-8"), '{"model_version": "v1"}')
        rec = json.loads((self.pre / "2026-10-19.json").read_text("utf-8"))
        self.assertEqual(rec["base"]["model_version"], "v1")


class Degrade(Base):
    def test_missing_global_data_still_runs(self):
        out = self.run_at("2026-10-19T08:15:00", G={})
        self.assertEqual(out["risk"]["level"], "中性")
        self.assertEqual(out["status"]["mode"], "sealed")

    def test_stale_us_data_is_flagged_and_not_used(self):
        G = {"markets": [{"symbol": "^SOX", "name": "費半", "date": "2026-10-14", "chg_pct": -5.0}]}
        out = self.run_at("2026-10-19T08:15:00", G=G)
        self.assertEqual(out["risk"]["level"], "中性")
        self.assertTrue(out["risk"]["stale"])

    def test_sox_crash_raises_risk_and_flags_semis(self):
        out = self.run_at("2026-10-19T08:15:00")
        self.assertEqual(out["risk"]["level"], "偏高")
        self.assertEqual([f["id"] for f in out["flags"]], ["2330"])
        self.assertEqual(out["flags"][0]["flag"], "產業風險")


class GapNotCounted(unittest.TestCase):
    def test_gap_is_reported_but_not_counted_as_return(self):
        dates = [f"2026-10-{d:02d}" for d in (16, 19, 20, 21, 22, 23, 26, 27)]
        def px(gap_to):
            o = [100.0, gap_to] + [gap_to] * 6
            c = [100.0] + [gap_to] * 7
            return pd.DataFrame({"open": o, "close": c, "high": [x * 1.01 for x in c], "low": [x * 0.99 for x in o],
                                 "volume": [1e6] * 8, "value": [1e9] * 8}, index=dates)
        D = {"dates": dates, "adj": {"2330": px(108.0), "0050": px(100.0)}, "rawi": {"2330": px(108.0), "0050": px(100.0)}}
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "pre_open"
            d.mkdir()
            (d / "2026-10-19.json").write_text(json.dumps({"target_date": "2026-10-19", "signal_date": "2026-10-16", "status": "on_time",
                                                           "risk": {"level": "中性"}, "flags": [], "watchlist": ["2330"]}), "utf-8")
            res = pm.evaluate(D, d)
        A = res["groups"]["A"]
        self.assertAlmostEqual(A["gap"], 8.0)
        self.assertAlmostEqual(A["excess_5d_from_open"], 0.0)   # 跳空的 8% 不算策略報酬


if __name__ == "__main__":
    unittest.main()
