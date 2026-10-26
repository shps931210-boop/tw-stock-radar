"""V19 新財報公告偵測與補抓：申報期判斷、官方欄位解析、補抓佇列優先順序、來源未同步會重試、更正紀錄、改分紀錄。"""
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

from app import config, db, fin_watch, update
from tests.test_backfill import FakeFM

NOW = datetime(2026, 10, 30, 21, 0)


def fin_row(sid, period, eps=1.0, rev=100.0, ni=10.0):
    y, q = int(period[:4]), int(period[-1])
    return {"stock_id": sid, "period": period, "period_end": f"{y}-{q * 3:02d}-{30 if q in (2, 3) else 31}", "revenue": rev, "eps": eps,
            "net_income": ni}


class Calendar(unittest.TestCase):
    def test_due_and_season(self):
        self.assertEqual(fin_watch.due_quarter(date(2026, 10, 26)), "2026-Q2")
        self.assertEqual(fin_watch.season_quarter(date(2026, 10, 26)), "2026-Q3")
        self.assertEqual(fin_watch.due_quarter(date(2026, 11, 15)), "2026-Q3")
        self.assertEqual(fin_watch.season_quarter(date(2026, 3, 20)), "2025-Q4")
        self.assertEqual(fin_watch.due_quarter(date(2026, 3, 20)), "2025-Q3")
        self.assertIsNone(fin_watch.season_quarter(date(2026, 12, 20)))
        self.assertEqual(fin_watch.due_quarter(date(2027, 1, 5)), "2026-Q3")


class Parse(unittest.TestCase):
    def test_twse_chinese_fields_roc_year(self):
        rows = [{"出表日期": "1151030", "年度": "115", "季別": "3", "公司代號": "2330", "公司名稱": "台積電",
                 "營業收入": "989,918,318", "基本每股盈餘（元）": "17.44"},
                {"出表日期": "1151030", "年度": "115", "季別": "2", "公司代號": "1101", "公司名稱": "台泥", "營業收入": "1", "基本每股盈餘（元）": "-0.1"}]
        got, info = fin_watch.parse(rows)
        self.assertEqual(got["2330"]["period"], "2026-Q3")
        self.assertAlmostEqual(got["2330"]["eps"], 17.44)
        self.assertEqual(got["2330"]["revenue"], 989918318)
        self.assertEqual(got["1101"]["period"], "2026-Q2")
        self.assertEqual(info["eps"], "基本每股盈餘（元）")

    def test_unknown_fields_reported(self):
        got, info = fin_watch.parse([{"foo": 1}])
        self.assertEqual(got, {})
        self.assertIn("找不到", info["error"])
        self.assertEqual(info["fields"], ["foo"])


class _DB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, config.DATA_DIR, config.ROOT
        db.DATA_DIR = config.DATA_DIR = Path(self.tmp.name)
        config.ROOT = Path(self.tmp.name)
        with db.connect() as con:
            for sid, v in (("2330", 9e9), ("1101", 5e8), ("1234", 1e6)):
                db.upsert(con, "prices", [{"stock_id": sid, "date": "2026-10-29", "close": 10, "value": v}])
                db.upsert(con, "financials", [fin_row(sid, "2026-Q2")])
            db.set_meta(con, "scores_today", json.dumps({"date": "2026-10-29", "s": {"1234": {"r": {"balanced": 3}, "sc": {"balanced": 61.0}}}}))

    def tearDown(self):
        db.DATA_DIR, config.DATA_DIR, config.ROOT = self.old
        self.tmp.cleanup()

    def detect(self, found):
        with db.connect() as con:
            return fin_watch.record(con, found, [{"ok": True}], now=NOW)


class Queue(_DB):
    def test_new_quarter_goes_first_and_ranked_stock_leads(self):
        self.detect({"2330": {"period": "2026-Q3"}, "1234": {"period": "2026-Q3"}, "1101": {"period": "2026-Q2"}})
        with db.connect() as con:
            hot = fin_watch.hot_list(con, ["2330", "1101", "1234"], now=NOW)
            st = dict(con.execute("SELECT stock_id, status FROM fin_announce").fetchall())
        self.assertEqual(hot, [("1234", "2026-Q3"), ("2330", "2026-Q3")])  # 今天在 AI 前 30 名的先補
        self.assertEqual(st["1101"], "already")

    def test_source_not_synced_retries_later(self):
        self.detect({"2330": {"period": "2026-Q3"}})
        with db.connect() as con:
            self.assertFalse(fin_watch.after_fetch(con, "2330", "2026-Q3", now=NOW))
            self.assertEqual(fin_watch.hot_list(con, [], now=NOW), [])  # 6 小時內不重試
            self.assertEqual(fin_watch.hot_list(con, [], now=NOW.replace(hour=23, minute=30))[:1], [])
            self.assertEqual(fin_watch.hot_list(con, [], now=datetime(2026, 10, 31, 4, 0)), [("2330", "2026-Q3")])
            st = con.execute("SELECT status, attempts FROM fin_announce").fetchone()
        self.assertEqual(st, ("waiting_source", 1))

    def test_update_fundamentals_fetches_announced_quarter_first(self):
        self.detect({"1101": {"period": "2026-Q3"}})
        calls = []

        class FM(FakeFM):
            def financials(self, sid, start=None):
                calls.append(sid)
                return [fin_row(sid, "2026-Q2", eps=1.0), fin_row(sid, "2026-Q3", eps=2.0)] if sid == "1101" else []

        cfg = {"finmind_token": "", "finmind_interval_sec": 0, "fundamental_top_n": 3000, "watchlist": []}
        with db.connect() as con:
            for sid in ("2330", "1101", "1234"):
                for src in ("fm_financials", "fm_revenue", "fm_dividends"):
                    db.log_fetch(con, src, sid)
        with mock.patch.object(update, "FinMindSource", FM), mock.patch.object(update, "_fin_requests", return_value=([], None)), \
                mock.patch.object(update.fin_watch, "detect", return_value=({}, [])):
            update.update_fundamentals(cfg)
        self.assertEqual(calls, ["1101"])  # 20 天內抓過，但官方已公告新季度，所以立刻補抓
        with db.connect() as con:
            st = con.execute("SELECT status, score_before FROM fin_announce WHERE stock_id='1101'").fetchone()
            Q = json.loads(db.get_meta(con, "fundamental_queue"))
            seen = con.execute("SELECT COUNT(*) FROM fin_seen WHERE stock_id='1101' AND period='2026-Q3'").fetchone()[0]
        self.assertEqual(st[0], "synced")
        self.assertEqual(Q["hot"], 1)
        self.assertEqual(Q["hot_synced"], 1)
        self.assertEqual(seen, 1)

    def test_revision_recorded(self):
        with db.connect() as con:
            revs = fin_watch.compare(con, "2330", [fin_row("2330", "2026-Q2", eps=1.2)], now=NOW)
            n = con.execute("SELECT COUNT(*) FROM fin_revision").fetchone()[0]
        self.assertEqual(revs[0]["fields"]["eps"], [1.0, 1.2])
        self.assertEqual(n, 1)

    def test_season_shortens_refetch_interval(self):
        with db.connect() as con:
            self.assertEqual(fin_watch.fin_age(con, "2330", date(2026, 10, 26)), 5)
            self.assertEqual(fin_watch.fin_age(con, "2330", date(2026, 12, 20)), 20)
            db.upsert(con, "financials", [fin_row("2330", "2026-Q3")])
            self.assertEqual(fin_watch.fin_age(con, "2330", date(2026, 10, 26)), 20)

    def test_summary_and_rescore(self):
        self.detect({"1234": {"period": "2026-Q3", "eps": 0.5}, "2330": {"period": "2026-Q3"}})
        with db.connect() as con:
            db.upsert(con, "financials", [fin_row("1234", "2026-Q3")])
            fin_watch.after_fetch(con, "1234", "2026-Q3", now=NOW)
            db.set_meta(con, "scores_today", json.dumps({"date": "2026-10-30", "s": {"1234": {"sc": {"balanced": 66.0}}}}))
            with mock.patch.object(config, "today", return_value=date(2026, 10, 30)), mock.patch.object(config, "now", return_value=NOW):
                s = fin_watch.after_snapshot(con)
        self.assertEqual(s["quarter"], "2026-Q3")
        self.assertEqual((s["announced"], s["synced"], s["waiting"]), (2, 1, 1))
        self.assertEqual(s["rescored"][0]["before"], 61.0)
        self.assertEqual(s["rescored"][0]["after"], 66.0)
        self.assertTrue((Path(self.tmp.name) / "history" / "fin" / "updates.json").exists())
        self.assertTrue((Path(self.tmp.name) / "fin_watch.json").exists())


class StepLog(unittest.TestCase):
    def test_two_failures_in_a_row_alert(self):
        from app import step_log
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "steps.json"
            with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "1"}):
                step_log.record(["新聞與公告=failure", "盤前情報=skipped"], p)
            self.assertEqual(step_log.alerts(p), [])
            with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "2"}):
                d = step_log.record(["新聞與公告=failure", "盤前情報="], p)
            self.assertEqual(d["status"]["新聞與公告"]["streak"], 2)
            self.assertNotIn("盤前情報", d["status"])
            self.assertIn("新聞與公告", step_log.alerts(p)[0])
            with mock.patch.dict("os.environ", {"GITHUB_RUN_ID": "3"}):
                step_log.record(["新聞與公告=success"], p)
            self.assertEqual(step_log.alerts(p), [])


class SlowAndCorrection(_DB):
    def test_after_max_attempts_switches_to_slow_tracking(self):
        self.detect({"2330": {"period": "2026-Q3"}})
        with db.connect() as con:
            con.execute("UPDATE fin_announce SET attempts=?, last_try=?, status='waiting_source'", (fin_watch.MAX_ATTEMPTS, NOW.isoformat()))
            self.assertEqual(fin_watch.hot_list(con, [], now=datetime(2026, 11, 1, 22, 0)), [])  # 低頻：3 天內不重試
            self.assertEqual(fin_watch.hot_list(con, [], now=datetime(2026, 11, 2, 22, 0)), [("2330", "2026-Q3")])  # 但不會永久放棄

    def test_official_change_retriggers_even_after_sync(self):
        self.detect({"2330": {"period": "2026-Q2", "eps": 1.0}})  # 已同步
        with db.connect() as con:
            con.execute("UPDATE fin_announce SET attempts=?", (fin_watch.MAX_ATTEMPTS,))
        self.detect({"2330": {"period": "2026-Q2", "eps": 1.3}})  # 官方更正
        with db.connect() as con:
            row = con.execute("SELECT status, attempts, corrected_at FROM fin_announce").fetchone()
            hot = fin_watch.hot_list(con, [], now=NOW)
        self.assertEqual(row[:2], ("pending", 0))
        self.assertTrue(row[2])
        self.assertEqual(hot, [("2330", "2026-Q2")])


class OfficialCheck(_DB):
    """V20.1：不只看 EPS；官方是年初至今累計（千元），和資料庫各季加總比對；舊數字不覆蓋新數字。"""
    OFF = {"period": "2026-Q2", "source": "twse:t187ap06_L_ci", "eps": 2.0, "revenue": 200000, "net_income": 20000, "parent_ni": 20000}

    def setUp(self):
        super().setUp()
        with db.connect() as con:  # 每季營收 1 億元、淨利 1 千萬元；Q1+Q2＝官方累計 20 萬、2 萬千元
            db.upsert(con, "financials", [self.row("2026-Q1"), self.row("2026-Q2")])

    @staticmethod
    def row(period, **kw):
        return {**fin_row("2330", period, rev=1e8, ni=1e7), **kw}

    def test_parse_extra_fields_and_empty_endpoint(self):
        got, info = fin_watch.parse([{"出表日期": "1151030", "年度": "115", "季別": "2", "公司代號": "2330", "營業收入": "10", "營業利益（損失）": "3",
                                      "本期淨利（淨損）": "2", "淨利（淨損）歸屬於母公司業主": "1.5", "基本每股盈餘（元）": "1"}])
        self.assertEqual({k: got["2330"][k] for k in ("operating_income", "net_income", "parent_ni")}, {"operating_income": 3, "net_income": 2, "parent_ni": 1.5})
        got, info = fin_watch.parse([{"Date": "1151026", "Year": "", "Season": "", "SecuritiesCompanyCode": "", "CompanyName": ""}])
        self.assertEqual(got, {})
        self.assertTrue(info.get("empty"))  # 櫃買中心沒有這類公司：端點正常，不算失敗

    def test_revenue_only_correction_retriggers(self):
        self.detect({"2330": self.OFF})
        with db.connect() as con:
            self.assertEqual(con.execute("SELECT status FROM fin_announce").fetchone()[0], "already")
        self.detect({"2330": {**self.OFF, "revenue": 250000}})  # EPS 沒變，只有營收更正
        with db.connect() as con:
            row = con.execute("SELECT status, corrected_at, note FROM fin_announce").fetchone()
            rev = json.loads(con.execute("SELECT detail FROM fin_revision").fetchone()[0])
            hot = fin_watch.hot_list(con, [], now=NOW)
        self.assertEqual(row[0], "pending")
        self.assertTrue(row[1])
        self.assertIn("營業收入", row[2])
        self.assertEqual(rev["revenue"], [200000, 250000])
        self.assertEqual(hot, [("2330", "2026-Q2")])

    def test_db_disagrees_with_official_goes_to_queue(self):
        self.detect({"2330": {**self.OFF, "net_income": 50000, "parent_ni": 50000}})  # 資料庫淨利加總 2 千萬，官方 5 千萬
        with db.connect() as con:
            st, note = con.execute("SELECT status, note FROM fin_announce").fetchone()
            hot = fin_watch.hot_list(con, [], now=NOW)
            self.assertFalse(fin_watch.after_fetch(con, "2330", "2026-Q2", now=NOW))  # 重抓還是舊數字：不算同步
            st2, note2 = con.execute("SELECT status, note FROM fin_announce").fetchone()
        self.assertEqual(st, "pending")
        self.assertIn("本期淨利", note)
        self.assertEqual(hot, [("2330", "2026-Q2")])
        self.assertEqual(st2, "waiting_source")
        self.assertIn("不一致", note2)

    def test_stale_source_does_not_overwrite_and_blank_keeps_old(self):
        self.detect({"2330": self.OFF})
        stale = self.row("2026-Q2", revenue=7e7)  # 和官方對不上的舊版數字
        blank = self.row("2026-Q1", net_income=None, eps=1.1)
        with db.connect() as con:
            rows = fin_watch.guard(con, "2330", [blank, stale], now=NOW)
            rec = json.loads(con.execute("SELECT detail FROM fin_revision").fetchone()[0])
        self.assertEqual([r["period"] for r in rows], ["2026-Q1"])  # Q2 舊數字被擋下
        self.assertEqual(rows[0]["net_income"], 1e7)  # 空白欄位保留原本的數字
        self.assertEqual(rows[0]["eps"], 1.1)
        self.assertEqual(rec["source"], "拒絕寫入")

    def test_full_cycle_times_and_rescore(self):
        """公告 → 補抓 → 入庫 → 重新評分，四個時間都有紀錄，分數變化可以追。"""
        self.detect({"2330": {**self.OFF, "period": "2026-Q3", "revenue": 300000, "net_income": 30000, "parent_ni": 30000}})
        with db.connect() as con:
            self.assertEqual(fin_watch.hot_list(con, [], now=NOW), [("2330", "2026-Q3")])
            db.upsert(con, "financials", fin_watch.guard(con, "2330", [self.row("2026-Q3")]))
            db.set_meta(con, "scores_today", json.dumps({"s": {"2330": {"sc": {"balanced": 50.0}}}}))
            self.assertTrue(fin_watch.after_fetch(con, "2330", "2026-Q3", now=NOW))
            db.set_meta(con, "scores_today", json.dumps({"s": {"2330": {"sc": {"balanced": 57.5}}}}))
            with mock.patch.object(config, "today", return_value=date(2026, 10, 30)), mock.patch.object(config, "now", return_value=NOW):
                s = fin_watch.after_snapshot(con)
        r = s["rescored"][0]
        self.assertEqual((r["before"], r["after"]), (50.0, 57.5))
        for k in ("first_seen", "fetched_at", "synced_at", "rescored_at"):
            self.assertTrue(r[k], k)
