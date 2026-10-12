"""對帳：獨立公式的除權息調整與報酬計算。"""
from __future__ import annotations

import sqlite3
import unittest

from app import audit, db


class Audit(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.executescript(db.SCHEMA)
        self.addCleanup(self.con.close)
        rows = [("2026-10-01", 100, 100), ("2026-10-02", 101, 102), ("2026-10-05", 97, 98), ("2026-10-06", 99, 99)]
        for d, o, c in rows:
            self.con.execute("INSERT INTO prices (stock_id, date, open, high, low, close, volume, value) VALUES ('1101', ?, ?, ?, ?, ?, 1, 1)", (d, o, c, o, c))
        # 10/05 除息 5 元：前收盤 102 → 參考價 97
        self.con.execute("INSERT INTO exrights VALUES ('1101', '2026-10-05', 102, 97)")

    def test_return_includes_dividend(self):
        r, ev = audit._ret(self.con, "1101", "2026-10-02", "2026-10-06", 0.0)
        self.assertAlmostEqual(r, (99 / 101 * 102 / 97 - 1) * 100, places=6)
        self.assertEqual(ev["ex_events"][0]["used"], True)

    def test_ex_on_entry_day_not_counted(self):
        r, ev = audit._ret(self.con, "1101", "2026-10-05", "2026-10-06", 0.0)
        self.assertAlmostEqual(r, (99 / 97 - 1) * 100, places=6)
        self.assertEqual(ev["ex_events"], [])

    def test_bad_prev_close_rejected(self):
        self.con.execute("UPDATE exrights SET prev_close=120")
        r, ev = audit._ret(self.con, "1101", "2026-10-02", "2026-10-06", 0.0)
        self.assertAlmostEqual(r, (99 / 101 - 1) * 100, places=6)
        self.assertFalse(ev["ex_events"][0]["used"])

    def test_missing_entry(self):
        self.assertEqual(audit._ret(self.con, "1101", "2026-10-03", "2026-10-06", 0.0), (None, "進場日沒有成交"))


if __name__ == "__main__":
    unittest.main()
