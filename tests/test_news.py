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
        with tempfile.TemporaryDirectory() as d, mock.patch.object(config, "DATA_DIR", Path(d)), mock.patch.object(db, "DATA_DIR", Path(d)):
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


class Copy(unittest.TestCase):
    def test_feature_descriptions(self):
        html = (Path(__file__).parent.parent / "static" / "index.html").read_text("utf-8")
        for t in ["根據固定量化模型，查看全市場股票評分與排名", "使用預設投資策略，尋找符合條件的股票", "自行設定財務、估值、技術與籌碼條件",
                  "查看符合指定技術條件的股票及觸發日期", "根據研究偏好重新排序，不改變客觀模型分數", "查看歷史選股表現，與大盤及產業基準比較", "無法判定"]:
            self.assertIn(t, html)


class Group(unittest.TestCase):
    def test_clean_title(self):
        self.assertEqual(news.clean_title("緯創砸逾70億元擴產 - CMoney投資網誌", "CMoney投資網誌"), "緯創砸逾70億元擴產")
        self.assertEqual(news.clean_title("A - B"), "A - B")  # 太短，不確定是不是來源

    def test_similar_reports_grouped(self):
        it = [{"id": "a", "kind": "news", "stock_id": "2330", "date": "2026-10-13", "title": "台積電9月營收創同期新高 年增三成"},
              {"id": "b", "kind": "news", "stock_id": "2330", "date": "2026-10-12", "title": "台積電9月營收創同期新高，年增三成"},
              {"id": "c", "kind": "news", "stock_id": "2330", "date": "2026-10-13", "title": "台積電董事會通過資本預算"},
              {"id": "d", "kind": "news", "stock_id": "2317", "date": "2026-10-13", "title": "台積電9月營收創同期新高 年增三成"}]
        news.group_similar(it)
        self.assertEqual([x["group"] for x in it], ["a", "a", "c", "d"])


class Importance(unittest.TestCase):
    def test_classify_real_titles(self):
        cases = {
            "公告本公司接獲證交所通知本公司上市有價證券併案列為變更交易方法(4月7日已停止買賣)": (3, "交易限制"),
            "公告本公司董事長退休": (3, "經營層異動"),
            "代重要子公司聯友先進材料股份有限公司公告總經理異動": (3, "經營層異動"),
            "公告董事會決議發行國內第六次無擔保可轉換公司債": (2, "籌資"),
            "公告本公司115年9月自結損益": (2, "營運數字"),
            "公告本公司董事長訂定115年第2季股利配息基準日": (2, "股利"),
            "公告本公司薪資報酬委員會委員辭職。": (2, "主管與董事異動"),
            "本公司召開電話法人說明會說明115年度第3季營運情形": (2, "法說會"),
            "公告本公司會計主管異動": (2, "主管與董事異動"),
            "公告本公司處分有價證券": (1, "資產交易"),
            "依公開發行公司資金貸與及背書保證處理準則第二十五條第一項第二款至第四款規定辦理公告": (1, "背書保證與資金貸與"),
            "公告本公司名稱更名": (1, "其他"),
        }
        for t, want in cases.items():
            self.assertEqual(news.classify(t), want, t)
