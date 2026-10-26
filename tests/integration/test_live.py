"""整合測試（V20.1）：需要網路與正式資料庫，在 GitHub Actions 完整更新後執行（INTEGRATION=1）。

和單元測試分開：這裡讀真實的證交所／櫃買中心 OpenAPI、正式資料庫，時間較長，也可能因為官方網站暫時故障而失敗；
失敗會記在 history/audit/steps.json（連續 2 次寄信），不影響網站更新。
"""
import os
import sqlite3
import unittest

from app import config, fin_watch, seal_calendar

LIVE = os.environ.get("INTEGRATION") == "1"
DB = config.DATA_DIR / "stocks.db"


def _db_ok() -> bool:
    try:
        return DB.exists() and sqlite3.connect(DB).execute("SELECT COUNT(*) FROM prices").fetchone()[0] > 100000
    except sqlite3.Error:
        return False


@unittest.skipUnless(LIVE, "整合測試：設定 INTEGRATION=1 才執行（需要網路）")
class OfficialOpenAPI(unittest.TestCase):
    def test_latest_quarter_endpoints_still_parse(self):
        found, diag = fin_watch.detect()
        ok = [d for d in diag if d.get("ok")]
        self.assertGreaterEqual(len(ok), 8, [(d["url"].rsplit("/", 1)[-1], d.get("error")) for d in diag if not d.get("ok")])
        self.assertGreater(len(found), 1500)  # 上市櫃一般公司
        ci = next(d for d in diag if d["url"].endswith("t187ap06_L_ci"))
        for k in ("eps", "revenue", "operating_income", "net_income", "parent_ni"):
            self.assertTrue(ci.get(k), f"證交所一般業找不到 {k} 欄位，官方可能改了欄位名稱")


@unittest.skipUnless(LIVE and _db_ok(), "整合測試：需要正式資料庫 data/stocks.db")
class ProductionDB(unittest.TestCase):
    def test_db_matches_official_for_almost_all(self):
        from app import db
        with db.connect() as con:
            fin_watch.ensure(con)
            n = con.execute("SELECT COUNT(*) FROM fin_announce WHERE status IN ('already','synced')").fetchone()[0]
            bad = con.execute("SELECT COUNT(*) FROM fin_announce WHERE note LIKE '資料庫和官方%'").fetchone()[0]
        self.assertGreater(n, 1000)
        self.assertLess(bad, n * 0.05, "超過 5% 的股票資料庫和官方累計數字對不上")

    def test_seal_calendar_has_no_unexplained_gap(self):
        cal = seal_calendar.build(data_days=seal_calendar.db_days())
        self.assertEqual(cal["summary"]["missing"], [], "有交易日沒有封存預測")
        self.assertFalse(cal["summary"]["scorecard_lag"], "對帳統計沒有跟上已到期的預測")
