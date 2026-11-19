"""2026-11-19：股價有資料、加權指數收盤卻缺的日子（OpenAPI 備援那天），用證交所 MI_5MINS_HIST 的收盤補上，不覆寫既有的。"""
import sqlite3
import unittest

from app import taiex_ohlc


class Fill(unittest.TestCase):
    def test_fills_only_missing_trading_days(self):
        c = sqlite3.connect(":memory:")
        c.executescript("""
        CREATE TABLE prices (stock_id TEXT, date TEXT, close REAL, PRIMARY KEY (stock_id, date));
        CREATE TABLE index_prices (date TEXT PRIMARY KEY, close REAL);
        CREATE TABLE index_ohlc (date TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL);
        INSERT INTO prices VALUES ('0050','2026-11-13',1),('0050','2026-11-16',1),('0050','2026-11-17',1);
        INSERT INTO index_prices VALUES ('2026-11-13',43675.12),('2026-11-17',40092.11);
        INSERT INTO index_ohlc VALUES ('2026-11-13',1,1,1,99999),('2026-11-16',43512.42,43512.42,41040.08,41040.08),
                                      ('2026-11-17',1,1,1,40092.11),('2026-11-20',1,1,1,1);
        """)
        self.assertEqual(taiex_ohlc.fill_index_prices(c), ["2026-11-16"])  # 11/20 沒有股價，不補
        got = dict(c.execute("SELECT date, close FROM index_prices"))
        self.assertEqual(got["2026-11-16"], 41040.08)
        self.assertEqual(got["2026-11-13"], 43675.12)  # 既有的不覆寫
        self.assertNotIn("2026-11-20", got)
        self.assertEqual(taiex_ohlc.fill_index_prices(c), [])


if __name__ == "__main__":
    unittest.main()
