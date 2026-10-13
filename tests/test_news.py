"""新聞中心：公告與新聞的解析、語氣標籤、輸出格式。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import config, db, news


class Parse(unittest.TestCase):
    def test_twse_announcement_keys(self):
        # 證交所 t187ap04_L 的真實欄位：「主旨 」後面有空白、民國日期、時間沒有冒號
        row = {"出表日期": "1151013", "發言日期": "1151012", "發言時間": "70004", "公司代號": "2330", "公司名稱": "台積電",
               "主旨 ": "公告本公司董事會決議", "符合條款": "第11款", "事實發生日": "1151012", "說明": "1.事實發生日..."}
        r = news.parse_announcements([row], "測試")[0]
        self.assertEqual((r["stock_id"], r["date"], r["time"], r["title"], r["rule"]), ("2330", "2026-10-12", "07:00", "公告本公司董事會決議", "第11款"))
        self.assertEqual(r["kind"], "announcement")
        self.assertTrue(r["link"])

    def test_other_date_formats_and_bad_rows(self):
        rows = [{"SecuritiesCompanyCode": "6488", "Subject": "公告", "AnnouncementDate": "115/10/13", "AnnouncementTime": "173000"},
                {"公司代號": "", "主旨": "沒有代號"}, {"公司代號": "1101", "主旨": "沒有日期"}]
        out = news.parse_announcements(rows, "測試")
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["date"], out[0]["time"]), ("2026-10-13", "17:30"))

    def test_finmind(self):
        rows = [{"date": "2026-10-13 09:31:00", "stock_id": "2330", "title": " 台積電營收創新高 ", "source": "某報", "link": "https://x", "description": "d"},
                {"date": "2026-10-13", "stock_id": "2330", "title": ""}]
        out = news.parse_finmind(rows)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["time"], out[0]["title"], out[0]["kind"]), ("09:31", "台積電營收創新高", "news"))
        # 同一連結只存一次
        self.assertEqual(out[0]["id"], news.parse_finmind(rows)[0]["id"])

    def test_tone_has_unclear(self):
        self.assertEqual(news.tone("營收創新高"), "positive")
        self.assertEqual(news.tone("獲利衰退"), "negative")
        self.assertEqual(news.tone("公告董事會日期"), "unclear")
        self.assertEqual(news.tone("營收成長但毛利下滑"), "unclear")


class Latest(unittest.TestCase):
    def test_latest_shape(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(config, "DATA_DIR", Path(d)):
            today = config.today().isoformat()
            with db.connect("t.db") as con:
                con.executescript(news.SCHEMA)
                db.upsert(con, "news_items", news.parse_finmind([{"date": today, "stock_id": "2330", "title": "接單看旺", "source": "s", "link": "https://x"}]))
                db.upsert(con, "news_items", news.parse_announcements([{"公司代號": "9999", "主旨": "公告", "發言日期": today.replace("-", "")}], "測試"))
                db.log_fetch(con, "news_center", today)
            snap = {"stocks": [{"stock_id": "2330", "name": "台積電", "industry": "半導體業", "close": 1000, "chg_pct": 1.2, "date": today,
                                "ai": {"modes": {"balanced": {"score": 80, "rank": 3}}}}]}
            out = news.latest(snap, "t.db")
        self.assertEqual(out["coverage"]["items"], 2)
        it = {x["stock_id"]: x for x in out["items"]}
        self.assertEqual(it["2330"]["tone"], "positive")
        self.assertEqual(it["2330"]["ctx"]["rank"], 3)
        self.assertIsNone(it["9999"]["tone"])  # 公告不標語氣
        self.assertIsNone(it["9999"]["ctx"])
        self.assertIn("不是 AI", out["tone_note"])
        self.assertTrue(out["last_fetched_at"])


if __name__ == "__main__":
    unittest.main()
