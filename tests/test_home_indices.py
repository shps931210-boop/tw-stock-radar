"""首頁主要市場指標：類股指數、櫃買指數。沒有資料時不能編數字。"""
import sqlite3
import unittest
from unittest import mock

from app import snapshot
from app.sources import tpex


class SectorIndices(unittest.TestCase):
    def test_change_from_last_two_rows_and_missing_is_none(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE sector_index (date TEXT, name TEXT, close REAL, PRIMARY KEY (date, name))")
        con.executemany("INSERT INTO sector_index VALUES (?,?,?)",
                        [("2026-10-13", "電子類指數", 1000.0), ("2026-10-14", "電子類指數", 1010.0)])
        out = {x["name"]: x for x in snapshot._sector_indices(con)}
        self.assertEqual(out["電子類股指數"]["close"], 1010.0)
        self.assertAlmostEqual(out["電子類股指數"]["chg_pct"], 1.0)
        self.assertIsNone(out["上櫃指數"]["close"])   # 沒有來源：網站顯示「資料源確認中」
        self.assertIsNone(out["金融類股指數"]["chg"])


class OtcIndexProbe(unittest.TestCase):
    def _run(self, rows):
        r = mock.MagicMock(); r.json.return_value = rows
        with mock.patch.object(tpex.http, "get", return_value=r):
            return tpex.openapi_otc_index(mock.MagicMock())

    def test_finds_index_field_and_skips_change(self):
        day, v, keys = self._run([{"Date": "1151016", "TPExIndexChange": "3.2", "TPExIndex": "290.15"}])
        self.assertEqual((day, v), ("2026-10-16", 290.15))

    def test_unknown_fields_return_none(self):
        day, v, keys = self._run([{"Date": "1151016", "TradeValue": "123"}])
        self.assertIsNone(v)
        self.assertEqual(keys, ["Date", "TradeValue"])


if __name__ == "__main__":
    unittest.main()
