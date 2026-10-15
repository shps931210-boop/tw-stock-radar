// 我的投資組合計算模組測試：node tests/portfolio.test.cjs（由 tests/test_portfolio.py 呼叫）
const assert = require("assert");
const PF = require("../static/portfolio.js");
const st = PF.settings({});
const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const T = (o) => ({ book: "real", fee: 0, tax: 0, at: o.date + "T00", ...o });

// 手續費：0.1425% 無條件捨去，整股最低 20、零股最低 1；折扣
assert.strictEqual(PF.estFee(1000000, 1000, st), 1425);
assert.strictEqual(PF.estFee(10000, 1000, st), 20);
assert.strictEqual(PF.estFee(1000, 10, st), 1);
assert.strictEqual(PF.estFee(1000000, 1000, PF.settings({ settings: { discount: 0.6 } })), 855);
// 交易稅：股票 0.3%、ETF 0.1%
assert.strictEqual(PF.estTax(1000000, "2330", st), 3000);
assert.strictEqual(PF.estTax(1000000, "0050", st), 1000);

// 移動平均成本、已實現損益
const tr = [T({ id: "a", sid: "2330", side: "buy", shares: 1000, price: 100, fee: 142, date: "2026-01-02" }),
  T({ id: "b", sid: "2330", side: "buy", shares: 1000, price: 120, fee: 171, date: "2026-01-05" }),
  T({ id: "c", sid: "2330", side: "sell", shares: 500, price: 130, fee: 92, tax: 195, date: "2026-01-06" }),
  T({ id: "d", sid: "2330", side: "div", amount: 3000, date: "2026-01-07" })];
const L = PF.ledger(tr, "real"), p = L.pos["2330"];
assert.strictEqual(L.errors.length, 0);
assert.strictEqual(p.shares, 1500);
near(p.cost, 220313 * 0.75);
near(p.realized, 65000 - 92 - 195 - 220313 / 4);
assert.strictEqual(p.divs, 3000);
// 模擬投資和實際持股分開
assert.strictEqual(Object.keys(PF.ledger(tr, "sim").pos).length, 0);
// 刪除的交易不算
assert.strictEqual(PF.ledger(tr.map((t) => (t.id === "b" ? { ...t, del: true } : t)), "real").pos["2330"].shares, 500);

// 持股損益
const H = PF.holdings(L, { "2330": 140 }, st)[0];
assert.strictEqual(H.mv, 210000);
near(H.unreal, 210000 - 220313 * 0.75);
near(H.avg, 220313 / 2000);
assert.ok(H.unreal_net < H.unreal);  // 扣掉估計的賣出手續費與交易稅

// 檢查：賣超、補登過去的賣出造成之後賣超、未來日期、缺成交價
const sell = (shares, date) => T({ id: "x", sid: "2330", side: "sell", shares, price: 100, date });
assert.ok(PF.validate(tr, sell(3000, "2026-01-08"), "2026-02-01").some((m) => m.includes("只持有")));
assert.deepStrictEqual(PF.validate(tr, sell(1500, "2026-01-08"), "2026-02-01"), []);
assert.ok(PF.validate(tr, sell(1200, "2026-01-03"), "2026-02-01").length);  // 1/3 只有 1000 股
const back = [T({ id: "a", sid: "1101", side: "buy", shares: 1000, price: 30, date: "2026-03-02" }), T({ id: "b", sid: "1101", side: "sell", shares: 1000, price: 31, date: "2026-03-10" })];
assert.ok(PF.validate(back, T({ id: "n", sid: "1101", side: "sell", shares: 500, price: 30, date: "2026-03-05" }), "2026-04-01").length);  // 之後那筆會變成賣超
assert.ok(PF.validate([], T({ id: "f", sid: "2330", side: "buy", shares: 1000, price: 100, date: "2026-12-01" }), "2026-10-15").some((m) => m.includes("未來")));
assert.ok(PF.validate([], T({ id: "f", sid: "2330", side: "buy", shares: 1000, date: "2026-10-01" }), "2026-10-15").some((m) => m.includes("成交價")));
assert.ok(PF.validate([], T({ id: "f", sid: "2330", side: "buy", shares: 1.5, price: 10, date: "2026-10-01" }), "2026-10-15").length);

// XIRR：一年 +10%
near(PF.xirr([["2025-01-01", -100], ["2026-01-01", 110]]), 0.1, 1e-4);
assert.strictEqual(PF.xirr([["2026-01-01", -100], ["2026-01-10", 110]]), null);  // 不到 30 天不年化

// 時間加權報酬不受加碼金額影響
const S = { "2330": [["2026-01-02", 100], ["2026-01-05", 110], ["2026-01-06", 110], ["2026-01-07", 99]] };
const tw = [T({ id: "1", sid: "2330", side: "buy", shares: 10, price: 100, date: "2026-01-02" }), T({ id: "2", sid: "2330", side: "buy", shares: 1000, price: 110, date: "2026-01-06" })];
const R = PF.daily(tw, "real", S, "2026-01-07");
near(R.twr, -1, 1e-9);
assert.strictEqual(R.curve.at(-1)[1], 1010 * 99);
// 週末補登的交易算到下一個交易日；全部賣出當天的報酬也正確
const wk = [T({ id: "1", sid: "2330", side: "buy", shares: 10, price: 100, date: "2026-01-03" }), T({ id: "2", sid: "2330", side: "sell", shares: 10, price: 110, date: "2026-01-06" })];
const R2 = PF.daily(wk, "real", S, "2026-01-07");
assert.strictEqual(R2.curve[0][0], "2026-01-05");
near(R2.twr, 10, 1e-9);

// 0050 對照：同樣的金額買 0050
const B = [["2026-01-02", 50], ["2026-01-05", 55], ["2026-01-07", 60]];
const sh = PF.shadow([T({ id: "1", sid: "2330", side: "buy", shares: 10, price: 100, date: "2026-01-02" })], "real", B, "2026-01-07");
near(sh.value, 1200); near(sh.bench_ret, 20); assert.strictEqual(sh.capped, false);
console.log("portfolio.js: all tests passed");
