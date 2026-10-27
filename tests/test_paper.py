"""AI 模擬交易（app/paper.py）與券商介面（app/broker.py）：只模擬、不偷看未來、成本、漲跌停、部分成交、可重算。"""
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from app import broker, config, paper

DAYS = [(date(2026, 10, 27) + timedelta(days=i)).isoformat() for i in range(40)]


def bars(prices, vol=1_000_000):
    """prices：每天的收盤價；開高低都用收盤價附近（開盤 = 前一天收盤）。"""
    out, prev = {}, prices[0]
    for d, c in zip(DAYS, prices):
        out[d] = (prev, max(prev, c), min(prev, c), c, vol)
        prev = c
    return out


def sig(*sids, gen="2026-10-27T15:00:00"):
    return {"file": "x.json", "generated_at": gen, "model_version": "v1.1", "model_hash": "h", "data_dates": {"prices": DAYS[0]},
            "lists": {m: [{"stock_id": s, "name": s, "rank": i + 1, "score": 80, "close": 100} for i, s in enumerate(sids)] for m in paper.MODES}}


class Engine(unittest.TestCase):
    def test_entry_next_open_with_cost(self):
        px = {"1101": bars([100] * 10)}
        px["1101"][DAYS[1]] = (101.0, 102.0, 99.0, 100.0, 1_000_000)
        B = paper.simulate("balanced", DAYS[:10], px, {DAYS[0]: sig("1101")})
        p = B["positions"][0]
        self.assertEqual(p["entry_date"], DAYS[1])                     # 訊號日隔天，不是訊號日
        self.assertAlmostEqual(p["entry_price"], 101 * 1.001, places=3)  # 開盤＋滑價
        self.assertEqual(p["shares"], int(200_000 / (101.101 * (1 + paper.FEE))))
        self.assertEqual(p["buy_fee"], paper.fee(p["shares"] * p["entry_price"], p["shares"]))
        self.assertEqual(B["curve"][0]["date"], DAYS[1])                # 起點以前不模擬

    def test_locked_limit_up_unfilled(self):
        px = {"1101": bars([100] * 5)}
        px["1101"][DAYS[1]] = (110.0, 110.0, 110.0, 110.0, 5000)
        B = paper.simulate("stable", DAYS[:5], px, {DAYS[0]: sig("1101")})
        self.assertEqual(B["positions"], [])
        self.assertEqual(B["orders"][0]["status"], "unfilled")
        self.assertIn("漲停", B["orders"][0]["why"])
        self.assertEqual(B["cash"], paper.CASH0)

    def test_volume_cap_partial(self):
        px = {"1101": bars([100] * 5, vol=10_000)}
        B = paper.simulate("stable", DAYS[:5], px, {DAYS[0]: sig("1101")})
        self.assertEqual(B["positions"][0]["shares"], 500)   # 成交量 5%
        self.assertEqual(B["orders"][0]["status"], "partial")

    def test_no_trade_unfilled(self):
        px = {"1101": bars([100] * 5)}
        del px["1101"][DAYS[1]]
        B = paper.simulate("stable", DAYS[:5], px, {DAYS[0]: sig("1101")})
        self.assertEqual(B["orders"][0]["status"], "unfilled")

    def test_stop_then_limit_down_retry(self):
        c = [100, 100, 89, 80.1, 75, 76, 77]
        px = {"1101": bars(c)}
        px["1101"][DAYS[3]] = (80.1, 80.1, 80.1, 80.1, 1000)   # 跌停鎖死：賣不掉
        B = paper.simulate("stable", DAYS[:7], px, {DAYS[0]: sig("1101")})
        t = B["trades"][0]
        self.assertEqual(t["exit_kind"], "stop")
        self.assertEqual(t["exit_signal_date"], DAYS[2])
        self.assertTrue(any(o["side"] == "sell" and "跌停" in (o["why"] or "") for o in B["orders"]))
        self.assertEqual(t["exit_date"], DAYS[4])            # 隔天再賣
        self.assertLess(t["net_ret"], t["gross_ret"])
        self.assertEqual(t["sell_tax"], paper.tax(t["shares"] * t["exit_price"], "1101"))

    def test_target_and_expiry(self):
        px = {"1101": bars([100, 100, 121, 121, 121]), "2330": bars([100] * 30)}
        B = paper.simulate("stable", DAYS[:30], px, {DAYS[0]: sig("1101", "2330")})
        k = {t["sid"]: t for t in B["trades"]}
        self.assertEqual(k["1101"]["exit_kind"], "target")
        self.assertEqual(k["2330"]["exit_kind"], "expiry")
        self.assertEqual(k["2330"]["exit_signal_date"], DAYS[paper.HOLD])   # 進場日算第 1 天，第 20 天收盤判斷
        self.assertEqual(k["2330"]["exit_date"], DAYS[paper.HOLD + 1])

    def test_costs(self):
        self.assertEqual(paper.fee(1000, 10), 1)          # 零股最低 1 元
        self.assertEqual(paper.fee(10000, 1000), 20)      # 整股最低 20 元
        self.assertEqual(paper.tax(100000, "0050"), 100)  # ETF 0.1%
        self.assertEqual(paper.tax(100000, "2330"), 300)

    def test_dividend_cash(self):
        px = {"1101": bars([100] * 6)}
        B = paper.simulate("stable", DAYS[:6], px, {DAYS[0]: sig("1101")}, exr={("1101", DAYS[3]): (100.0, 97.0)})
        self.assertAlmostEqual(B["positions"][0]["dividends"], 3 * B["positions"][0]["shares"])

    def test_no_future_signal(self):
        """訊號日之後才封存的名單不能用。"""
        with tempfile.TemporaryDirectory() as t:
            T = Path(t)
            for name, g in (("2026-10-27.json", "2026-10-27T15:00:00"), ("2026-10-27.v2.json", "2026-10-28T09:30:00")):
                (T / name).write_text(json.dumps({"generated_at": g, "rec_picks": {"balanced": [{"stock_id": name[:13]}]}}))
            s = paper.signal_for("2026-10-27", "2026-10-28", T)
            self.assertEqual(s["file"], "2026-10-27.json")
            self.assertIsNone(paper.signal_for("2026-10-27", "2026-10-27", T))


class Stats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = paper.wilson(5, 10)
        self.assertLess(lo, 50)
        self.assertGreater(hi, 50)
        self.assertIsNone(paper.wilson(0, 0))

    def test_odds_only_matured(self):
        tr = [{"entry_date": DAYS[1], "exit_kind": "stop", "net_ret": -11.0}, {"entry_date": DAYS[25], "exit_kind": "stop", "net_ret": -11.0}]
        O = paper.odds(tr, DAYS[:30])
        self.assertEqual(O["n"], 1)
        self.assertEqual(O["early"]["stop"], 1)
        self.assertFalse(O["enough"])

    def test_mdd(self):
        self.assertEqual(paper.mdd([100, 120, 90, 130]), -25.0)

    def test_research_items_threshold(self):
        L = {"books": {"balanced": {"review": [{"tags": ["entry"]}] * 3 + [{"tags": ["cost"]}]}}}
        it = paper.research_items(L)
        self.assertEqual([x["key"] for x in it], ["balanced:entry"])
        self.assertIn("不會自動改策略", it[0]["text"])


class Ledger(unittest.TestCase):
    def test_recompute_and_fingerprint(self):
        px = {"1101": bars([100, 100, 102, 103, 101, 104]), "0050": bars([100] * 6)}
        sg = {DAYS[0]: sig("1101")}
        L1 = paper.build(DAYS[:6], px, sg)
        L2 = paper.build(DAYS[:6], px, sg)
        self.assertEqual(json.dumps(L1["books"], sort_keys=True), json.dumps(L2["books"], sort_keys=True))  # 完全可重算
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "days.json"
            paper.update_days(L1, f)
            px["1101"][DAYS[3]] = (102.0, 110.0, 101.0, 109.0, 1_000_000)   # 官方修正價格
            D = paper.update_days(paper.build(DAYS[:6], px, sg), f)
            self.assertTrue(D["mismatches"])
            self.assertEqual(D["mismatches"][0]["date"], DAYS[3])
            self.assertNotEqual(D["days"][DAYS[3]]["balanced"]["equity"], D["mismatches"][0]["new_equity"])  # 舊紀錄不被改

    def test_bench(self):
        px = {"1101": bars([100] * 6), "0050": bars([100, 100, 110, 110, 110, 110])}
        L = paper.build(DAYS[:6], px, {DAYS[0]: sig("1101")}, idx={d: 100 for d in DAYS})
        self.assertGreater(L["bench"]["e0050"]["ret"], 9)
        self.assertEqual(L["bench"]["taiex"]["ret"], 0)


class Replay(unittest.TestCase):
    def test_first_version_and_outcome(self):
        with tempfile.TemporaryDirectory() as t:
            T = Path(t)
            rec = {"generated_at": "2026-10-27T15:00:00", "model_version": "v1.1",
                   "rec_picks": {"balanced": [{"stock_id": "1101", "rank": 1, "score": 80}]},
                   "picks": {"balanced": [{"stock_id": "1101", "name": "台泥", "close": 100, "factors": {"growth": 90}}]}}
            (T / "2026-10-27.json").write_text(json.dumps(rec))
            (T / "2026-10-27.v2.json").write_text(json.dumps({**rec, "rec_picks": {"balanced": [{"stock_id": "9999"}]}}))
            px = {"1101": bars([100, 100, 89, 90, 91]), "0050": bars([100] * 5)}
            R = paper.replay(DAYS[:5], px, {d: 100 for d in DAYS}, picks=T)
            d = R["days"][0]
            self.assertTrue(d["on_time"])
            r = d["modes"]["balanced"]["picks"][0]
            self.assertEqual(r["sid"], "1101")            # 用首次發布版，不用之後的補正版
            self.assertEqual(r["name"], "台泥")
            self.assertEqual(r["kind"], "stop")
            self.assertEqual(r["exit"], DAYS[3])


class Safety(unittest.TestCase):
    def test_live_disabled(self):
        self.assertFalse(broker.LIVE_TRADING)
        self.assertFalse(paper.LIVE_TRADING)
        with self.assertRaises(broker.LiveTradingDisabled):
            broker.get_broker("yuanta_spark")
        with self.assertRaises(broker.LiveTradingDisabled):
            broker.YuantaSparkBroker(account="x")
        b = broker.get_broker("paper")
        self.assertFalse(b.live)
        self.assertEqual(b.place(broker.Order("2330", "buy", 1))["status"], "simulated")

    def test_no_credentials_or_network(self):
        for f in ("app/broker.py", "app/paper.py"):
            s = (config.ROOT / f).read_text("utf-8")
            for bad in ("os.environ", "getenv", "requests", "urllib", "socket", "password=", "cert_path", "LIVE_TRADING = True"):
                self.assertNotIn(bad, s, f"{f} 不應包含 {bad}")


if __name__ == "__main__":
    unittest.main()
