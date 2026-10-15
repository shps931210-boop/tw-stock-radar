// 正式網站自動驗收：用手機尺寸的瀏覽器打開網站，隨機挑 20 檔不同產業、市場、規模的股票，逐一檢查個股頁。
// 用法：node tests/e2e/live.mjs <網站網址> <輸出 JSON> [playwright 路徑]
// 每天的抽樣用日期當種子：同一天重跑抽到同一批，方便重現問題；不同天會換一批。
import fs from "node:fs";
import path from "node:path";

const [base0, out, pwPath] = process.argv.slice(2);
const base = base0.replace(/\/?$/, "/");
const pw = await import(pwPath || "playwright");
const chromium = pw.chromium || pw.default.chromium;
const bust = `?e2e=${Date.now()}`;
const snap = await (await fetch(base + "data/snapshot.json" + bust)).json();
const asOf = snap.data_dates?.prices;

let seed = [...(asOf || "x")].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const shuffle = (xs) => xs.map((x) => [rnd(), x]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
const common = snap.stocks.filter((s) => s.is_common);
const byCap = [...common].sort((a, b) => (b.market_cap || 0) - (a.market_cap || 0));
const groups = [
  ["上市大型股", byCap.filter((s) => s.market !== "tpex").slice(0, 50), 3],
  ["上市中小型股", byCap.filter((s) => s.market !== "tpex").slice(300), 3],
  ["上櫃股票", common.filter((s) => s.market === "tpex"), 3],
  ["金融股", common.filter((s) => s.is_financial), 2],
  ["虧損公司", common.filter((s) => (s.eps_ttm ?? 0) < 0), 2],
  ["新上市（歷史不足）", common.filter((s) => (s.history_days || 0) < 120), 2],
  ["未進排行榜（成交少）", common.filter((s) => !s.base_ok), 3],
  ["隨機", common, 2],
];
const picked = [], seen = new Set(), inds = new Set();
for (const [g, xs, n] of groups) {
  let k = 0;
  for (const s of shuffle(xs)) {  // 盡量不同產業
    if (k >= n) break;
    if (seen.has(s.stock_id) || (inds.has(s.industry) && xs.length > 30)) continue;
    picked.push([g, s]); seen.add(s.stock_id); inds.add(s.industry); k++;
  }
}

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
let errs = [];
p.on("pageerror", (e) => errs.push(e.message));
await p.goto(base + bust + "#/", { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => document.querySelector("#app") && !/載入中/.test(document.querySelector("#app").textContent), null, { timeout: 60000 });
const results = [];
let pageErrors = 0;
for (const [group, s] of picked) {
  errs = [];
  const problems = [];
  await p.evaluate((id) => { location.hash = "#/stock/" + id; }, s.stock_id);
  try {
    await p.waitForFunction((id) => { const h = document.querySelector(".s-head h1"); return h && h.textContent.includes(id); }, s.stock_id, { timeout: 20000 });
    await p.waitForTimeout(1200);
  } catch (e) { problems.push("個股頁 20 秒內沒有載入"); }
  const info = await p.evaluate(() => ({
    head: document.querySelector(".s-head")?.textContent || "",
    dcard: !!document.querySelector(".dcard"),
    mods: [...document.querySelectorAll(".dcard .mods .pill")].map((x) => x.textContent.trim()),
    width: document.documentElement.scrollWidth,
    empty: /沒有詳細資料|找不到/.test(document.querySelector("#app")?.textContent || ""),
  }));
  if (info.empty) problems.push("個股頁顯示沒有資料");
  if (!info.dcard) problems.push("研究決策卡沒有出現");
  if (info.dcard && !info.mods.length) problems.push("研究卡沒有各部分的資料狀態");
  const quoteReady = info.mods.some((m) => m.startsWith("行情・完成"));
  if (quoteReady && asOf && !info.head.includes(asOf)) problems.push(`行情完成，但頁面資料日不是 ${asOf}`);
  if (info.width > 392) problems.push(`手機版畫面超出寬度（${info.width}px）`);
  if (errs.length) { problems.push("頁面錯誤：" + errs.slice(0, 2).join("；").slice(0, 160)); pageErrors += errs.length; }
  results.push({ sid: s.stock_id, name: s.name, industry: s.industry, group, ok: !problems.length, problems, modules: info.mods });
}
await b.close();
const report = { schema_version: 1, ran_at: new Date(Date.now() + 8 * 36e5).toISOString().slice(0, 19), base, data_date: asOf,
  site_generated_at: snap.generated_at, page_errors: pageErrors, passed: results.filter((r) => r.ok).length, results };
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(report, null, 1));
console.log(`驗收 ${report.passed}/${results.length} 通過（資料日 ${asOf}）`);
for (const r of results) console.log(r.ok ? "✓" : "✗", r.sid, r.name, r.group, r.problems.join("；"), "|", r.modules.join(" "));
