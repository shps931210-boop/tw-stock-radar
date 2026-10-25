// 首頁「不可退步清單」端到端驗收：用真的瀏覽器（電腦＋手機尺寸）打開網站，實際操作搜尋、切換策略、加入自選股、進入個股與績效頁。
// 用法：node tests/e2e/home.mjs <網站網址> <輸出 JSON> [playwright 路徑]
// 任何一項沒通過，程式結束碼為 1（網站驗收工作流程會標成失敗並寄信）。
import fs from "node:fs";
import path from "node:path";

const [base0, out, pwPath] = process.argv.slice(2);
const base = base0.replace(/\/?$/, "/");
const pw = await import(pwPath || "playwright");
const chromium = pw.chromium || pw.default.chromium;
const bust = `?e2e=${Date.now()}`;
const snap = await (await fetch(base + "data/snapshot.json" + bust)).json();
const asOf = snap.data_dates?.prices || "";

const browser = await chromium.launch();
const results = [];
for (const [w, h, dev] of [[1280, 900, "電腦"], [390, 844, "手機"]]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, isMobile: dev === "手機", hasTouch: dev === "手機" });
  const page = await ctx.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(e.message));
  const check = async (id, name, fn) => {
    let ok = false, note = "";
    try { [ok, note] = await fn(); } catch (e) { note = "錯誤：" + e.message.split("\n")[0]; }
    results.push({ device: dev, id, name, ok: !!ok, note });
  };
  const home = async () => { await page.goto(base + bust + "#/"); await page.waitForSelector("#hmkt", { timeout: 30000 }); };
  await home();

  await check(1, "大盤資訊清楚可見（第一屏）", async () => {
    const r = await page.$eval("#hmkt", (el) => ({ top: el.getBoundingClientRect().top, txt: el.innerText }));
    const ok = r.top < h * 0.6 && /加權指數\s*[\d,]+\.\d\d/.test(r.txt) && r.txt.includes("0050") && /上漲/.test(r.txt) && /下跌/.test(r.txt);
    return [ok, `卡片頂端 ${Math.round(r.top)}px`];
  });
  await check(2, "AI 數字評分清楚可見", async () => {
    const xs = await page.$$eval("#hpick .pk-sc b", (bs) => bs.map((b) => ({ t: b.innerText.trim(), fs: parseFloat(getComputedStyle(b).fontSize) })));
    const nums = xs.map((x) => Number(x.t)).filter((n) => Number.isFinite(n) && n >= 0 && n <= 100);
    return [xs.length > 0 && nums.length === xs.length && Math.min(...xs.map((x) => x.fs)) >= 18, `${xs.length} 檔，分數 ${xs.map((x) => x.t).join("、")}`];
  });
  await check(3, "股票搜尋正常", async () => {
    const sid = (snap.stocks.find((s) => s.stock_id === "2330") || snap.stocks[0]).stock_id;
    await page.fill("#hq", sid); await page.press("#hq", "Enter");
    await page.waitForFunction((id) => location.hash === `#/stock/${id}`, sid, { timeout: 10000 });
    await page.waitForSelector(".qj-top", { timeout: 30000 });
    return [true, `搜尋 ${sid} 進入個股頁`];
  });
  await home();
  await check(4, "AI 三種策略可以切換", async () => {
    const seen = [];
    for (const b of await page.$$("#hmode button")) {
      const k = await b.getAttribute("data-k");
      await page.click(`#hmode button[data-k="${k}"]`);
      const on = await page.$eval("#hmode button.on", (x) => x.dataset.k);
      const n = await page.$$eval("#hpick .pk", (xs) => xs.length);
      seen.push(`${k}:${on === k ? "切換" : "沒切換"}/${n} 檔`);
      if (on !== k || !n) return [false, seen.join("，")];
    }
    return [seen.length === 3, seen.join("，")];
  });
  await check(5, "自選股可以加入並顯示", async () => {
    await page.evaluate(() => localStorage.removeItem("watch"));
    await home();
    const star = await page.$("#hpick .pk [data-star]");
    const sid = await star.getAttribute("data-star");
    await star.click();
    const stayed = await page.evaluate(() => location.hash);
    const live = await page.$eval("#hwatch", (el, id) => el.innerHTML.includes(`#/stock/${id}`), sid);
    await page.reload(); await page.waitForSelector("#hwatch");
    const shown = await page.$eval("#hwatch", (el, id) => el.innerHTML.includes(`#/stock/${id}`), sid);
    const scored = await page.$$eval("#hwatch .pk-sc b", (bs) => bs.length);
    return [live && shown && scored > 0 && (stayed === "#/" || stayed === ""), `加入 ${sid}；立即出現=${live}、重新整理後仍在=${shown}、有 AI 評分=${scored > 0}、按星號不跳頁=${stayed === "#/" || stayed === ""}`];
  });
  await check(6, "漲跌停資訊正常", async () => {
    const t = await page.$eval("#hmkt .mk-sub", (el) => el.innerText);
    return [/漲停\s*\d+/.test(t) && /跌停\s*\d+/.test(t), t.replace(/\s+/g, " ")];
  });
  await check(7, "個股分析正常", async () => {
    const sid = await page.$eval("#hpick .pk", (a) => a.getAttribute("href").split("/").pop());
    await page.goto(base + bust + `#/stock/${sid}`);
    await page.waitForSelector(".qj-top", { timeout: 30000 });
    const t = await page.$eval("#app", (el) => el.innerText);
    return [t.includes(sid) && !t.includes("找不到") && !/上漲機率/.test(t), `個股 ${sid}`];
  });
  await check(8, "AI 績效資料正常", async () => {
    await page.goto(base + bust + "#/performance");
    await page.waitForFunction(() => document.querySelector("#app")?.innerText.includes("0050"), null, { timeout: 30000 });
    return [true, "績效中心有 0050 比較"];
  });
  await home();
  await check(9, "版面沒有橫向超出", async () => {
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    return [sw <= w, `頁寬 ${sw}px（視窗 ${w}px）`];
  });
  await check(10, "資料時間與盤後狀態正確", async () => {
    const t = await page.$eval("#hmkt", (el) => el.innerText);
    return [asOf && t.includes(asOf) && /盤後資料|收盤|資料待更新/.test(t) && !/即時行情(?!）)/.test(t.replace("非即時行情", "")), `資料日 ${asOf}`];
  });
  results.push({ device: dev, id: 0, name: "沒有 JavaScript 錯誤", ok: errs.length === 0, note: errs.slice(0, 3).join("；") });
  await ctx.close();
}
await browser.close();
const passed = results.filter((r) => r.ok).length;
const rep = { at: new Date().toISOString(), site: base, as_of: asOf, passed, total: results.length, results };
if (out) { fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, JSON.stringify(rep, null, 1)); }
for (const r of results) console.log(`${r.ok ? "✅" : "❌"} [${r.device}] ${r.id} ${r.name}：${r.note}`);
console.log(`通過 ${passed}／${results.length}`);
process.exit(passed === results.length ? 0 : 1);
