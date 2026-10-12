"""事件日曆：解析證交所除權除息預告表、去重、狀態更新、法定期限只能標為推估。"""
from __future__ import annotations

import sqlite3
import unittest
from datetime import date

from app import db, events

FIELDS = ["除權除息日期", "股票代號", "名稱", "除權息", "無償配股率", "現金增資配股率", "現金增資認購價", "現金股利", "詳細資料", "參考價試算", "最近一次申報資料 季別/日期", "最近一次申報每股 (單位)淨值", "最近一次申報每股 (單位)盈餘"]


def row(d, sid, name, kind, cash, stock=""):
    return [d, sid, name, kind, stock, "", "", cash, "", "", "", "", ""]


class ExDivForecast(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(db.SCHEMA)
        self.addCleanup(self.con.close)

    def test_parse_roc_dates_and_pending_cash(self):
        rows = events.parse_twt48u(FIELDS, [row("115年10月20日", "2330", "台積電", "息", "5.0"), row("115年10月21日", "1101", "台泥", "權息", "", "0.05")])
        self.assertEqual(rows[0]["date"], "2026-10-20")
        self.assertEqual(rows[0]["cash"], 5.0)
        self.assertIsNone(rows[1]["cash"])
        self.assertEqual(rows[1]["cash_note"], "待公告實際金額")
        self.assertIn("除權息", events._title(rows[1]))

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            events.parse_twt48u(["日期", "代號"], [])

    def status(self, eid):
        return self.con.execute("SELECT status, first_seen FROM events WHERE event_id=?", (eid,)).fetchone()

    def test_dedupe_keeps_first_seen(self):
        r = events.parse_twt48u(FIELDS, [row("115年10月20日", "2330", "台積電", "息", "5.0")])
        events.store_exdiv(self.con, r, date(2026, 10, 12))
        first = self.status("exdiv:2330:2026-10-20")[1]
        self.con.execute("UPDATE events SET first_seen='2026-10-01T00:00:00'")
        events.store_exdiv(self.con, r, date(2026, 10, 13))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        self.assertEqual(self.status("exdiv:2330:2026-10-20"), ("confirmed", "2026-10-01T00:00:00"))
        self.assertTrue(first)

    def test_disappeared_future_event_is_unverified_and_past_is_completed(self):
        a = events.parse_twt48u(FIELDS, [row("115年10月20日", "2330", "台積電", "息", "5.0"), row("115年10月14日", "2317", "鴻海", "息", "5.4")])
        events.store_exdiv(self.con, a, date(2026, 10, 12))
        events.store_exdiv(self.con, [], date(2026, 10, 15))
        self.assertEqual(self.status("exdiv:2330:2026-10-20")[0], "unverified")
        self.assertEqual(self.status("exdiv:2317:2026-10-14")[0], "completed")


class Deadlines(unittest.TestCase):
    def test_deadlines_are_estimated_never_confirmed(self):
        dl = events.deadlines(date(2026, 10, 12))
        self.assertTrue(dl)
        self.assertTrue(all(d["status"] == "estimated" and d["stock_id"] is None for d in dl))
        ids = {d["event_id"] for d in dl}
        self.assertIn("deadline:revenue:2026-09", ids)   # 10/10 前公布 9 月營收
        self.assertIn("deadline:fin:2026Q3", ids)        # 11/14 第三季財報
        self.assertEqual(next(d for d in dl if d["event_id"] == "deadline:fin:2026Q3")["event_date"], "2026-11-14")
        self.assertEqual(len(ids), len(dl))              # 不重複

    def test_annual_report_belongs_to_previous_year(self):
        dl = events.deadlines(date(2027, 2, 20))
        self.assertIn("deadline:fin:2026Y", {d["event_id"] for d in dl})


if __name__ == "__main__":
    unittest.main()


class TpexForecast(unittest.TestCase):
    F = ["除權息日期", "代號", "名稱", "除權息", "每股無償配股率", "現金增資配股率", "現金增資認購價", "現金股利", "公開承銷股數", "員工認購股數", "原股東認購股數", "按持股比例仟股認購", "參考價試算"]

    def test_parse_and_sources_are_separate(self):
        rows = events.parse_tpex(self.F, [["115/10/19", "00694B", "富邦美債1-3", "除息", "0.00000000", "0", "0.00", "尚未公告", "0", "0", "0", "0", ""],
                                          ["115/10/20", "6488", "環球晶", "除權息", "0.05", "0", "0", "3.5", "0", "0", "0", "0", ""]])
        self.assertEqual(rows[0]["date"], "2026-10-19")
        self.assertIsNone(rows[0]["cash"])
        self.assertEqual(rows[1]["kind"], "權息")
        self.assertIn("除權息", events._title(rows[1]))
        con = sqlite3.connect(":memory:")
        con.executescript(db.SCHEMA)
        self.addCleanup(con.close)
        events.store_exdiv(con, rows, date(2026, 10, 12), events.SRC_TPEX, events.TPEX_PAGE)
        events.store_exdiv(con, [], date(2026, 10, 12))   # 上市這次沒有資料：不影響上櫃事件
        st = dict(con.execute("SELECT stock_id, status FROM events").fetchall())
        self.assertEqual(st, {"00694B": "confirmed", "6488": "confirmed"})
