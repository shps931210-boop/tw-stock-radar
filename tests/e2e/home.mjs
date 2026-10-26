// 「不可退步清單」端到端驗收（22 項）：用真的瀏覽器（電腦＋手機尺寸）打開網站，實際操作搜尋、切換策略、加入自選股、進入個股與各功能頁。
// 用法：node tests/e2e/home.mjs <網站網址> <輸出 JSON> [playwright 路徑]
// 任何一項沒通過，程式結束碼為 1（網站驗收工作流程會標成失敗並寄信）。
// 本機預覽沒有外部圖表元件時，可設 LWC_FILE=<lightweight-charts 檔案路徑> 讓瀏覽器改讀本機檔案。
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
  if (process.env.LWC_FILE) await ctx.route(/lightweight-charts/, (r) => r.fulfill({ path: process.env.LWC_FILE, contentType: "application/javascript" }));
  const page = await ctx.newPage();
  const errs = [];
  page.on("pageerror", (e) => errs.push(e.message));
  const check = async (id, name, fn) => {
    let ok = false, note = "";
    try { [ok, note] = await fn(); } catch (e) { note = "錯誤：" + e.message.split("\n")[0]; }
    results.push({ device: dev, id, name, ok: !!ok, note });
  };
  const go = async (hash, sel) => { await page.goto(base + bust + hash); await page.waitForSelector(sel, { timeout: 30000 }); };
  const home = () => go("#/", "#hmkt");
  const chartOk = (sel) => page.$eval(sel, (el) => { const c = el.querySelector("canvas, svg"); const r = c?.getBoundingClientRect(); return { ok: !!c && r.width > 100 && r.height > 50, kind: c?.tagName, wh: r ? `${Math.round(r.width)}×${Math.round(r.height)}` : "" }; });
  await home();

  await check(1, "加權指數", async () => {
    const r = await page.$eval("#hmkt", (el) => ({ top: el.getBoundingClientRect().top, txt: el.innerText }));
    return [r.top < h * 0.6 && /加權指數\s*[\d,]+\.\d\d/.test(r.txt), `大盤卡頂端 ${Math.round(r.top)}px`];
  });
  await check(2, "0050", async () => {
    const t = await page.$eval("#hmkt .mk-top", (el) => el.innerText);
    return [/0050\s*[\d,]+\.\d\d/.test(t), (t.match(/0050\s*[\d,.]+/)?.[0] || "").replace(/\s+/g, " ")];
  });
  await check(3, "大盤走勢線圖", async () => {
    await page.waitForTimeout(800);
    const c = await chartOk("#hix");
    const top = await page.$eval("#hix", (el) => el.getBoundingClientRect().top);
    const sw = [];
    for (const r of ["5D", "1M", "1Y", "3M"]) { await page.click(`#ixr button[data-r="${r}"]`); await page.waitForTimeout(300); sw.push(r + ":" + ((await chartOk("#hix")).ok ? "有圖" : "沒圖")); }
    return [c.ok && sw.every((x) => x.endsWith("有圖")) && (dev === "手機" || top < h), `${c.kind} ${c.wh}，頂端 ${Math.round(top)}px；${sw.join("，")}`];
  });
  await check(5, "AI 數字評分", async () => {
    const xs = await page.$$eval("#hpick .pk-sc b", (bs) => bs.map((b) => ({ t: b.innerText.trim(), fs: parseFloat(getComputedStyle(b).fontSize) })));
    const nums = xs.map((x) => Number(x.t)).filter((n) => Number.isFinite(n) && n >= 0 && n <= 100);
    return [xs.length > 0 && nums.length === xs.length && Math.min(...xs.map((x) => x.fs)) >= 18, `${xs.length} 檔，分數 ${xs.map((x) => x.t).join("、")}`];
  });
  await check(6, "AI 精選股票", async () => {
    const n = await page.$$eval("#hpick .pk", (xs) => xs.length);
    const t = await page.$eval("#hpick", (el) => el.innerText);
    return [n > 0 && /風險/.test(t) && !/上漲機率/.test(t), `${n} 檔，有風險等級`];
  });
  await check(7, "穩健、均衡、積極策略", async () => {
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
  await check(11, "市場漲跌與漲跌停家數", async () => {
    const t = await page.$eval("#hmkt", (el) => el.innerText);
    return [/上漲\s*\d+/.test(t) && /下跌\s*\d+/.test(t) && /漲停\s*\d+/.test(t) && /跌停\s*\d+/.test(t), (t.match(/漲停\s*\d+.跌停\s*\d+/) || [""])[0]];
  });
  await check(17, "資料時間標示", async () => {
    const t = await page.$eval("#hmkt", (el) => el.innerText);
    return [asOf && t.includes(asOf) && /盤後資料|收盤|資料待更新/.test(t) && !/即時行情/.test(t.replace("非即時行情", "")), `資料日 ${asOf}`];
  });
  await check(18, "手機與電腦主要版面", async () => {
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    return [sw <= w, `頁寬 ${sw}px（視窗 ${w}px）`];
  });
  const shot = async (name) => { if (process.env.SHOT_DIR) await page.screenshot({ path: path.join(process.env.SHOT_DIR, `${name}-${dev === "手機" ? "m" : "d"}.png`), fullPage: true }); };
  await shot("home");
  await check(8, "股票搜尋", async () => {
    const sid = (snap.stocks.find((s) => s.stock_id === "2330") || snap.stocks[0]).stock_id;
    await page.fill("#hq", sid); await page.press("#hq", "Enter");
    await page.waitForFunction((id) => location.hash === `#/stock/${id}`, sid, { timeout: 10000 });
    await page.waitForSelector(".qj-top", { timeout: 30000 });
    return [true, `搜尋 ${sid} 進入個股頁`];
  });
  await check(9, "個股分析", async () => {
    const t = await page.$eval("#app", (el) => el.innerText);
    const sc = await page.$eval(".qj-sc b", (b) => b.innerText.trim());
    return [!t.includes("找不到") && !/上漲機率/.test(t) && /^(\d+|—)$/.test(sc), `AI 評分 ${sc}`];
  });
  await check(4, "個股走勢圖", async () => {
    await page.waitForTimeout(800);
    const c = await chartOk("#pc");
    const top = await page.$eval("#j-px", (el) => el.getBoundingClientRect().top + scrollY);
    const rs = await page.$eval("#j-rs", (el) => ({ open: el.open, top: el.getBoundingClientRect().top + scrollY }));
    return [c.ok && top < rs.top && !rs.open, `${c.kind} ${c.wh}；走勢圖在研究摘要之前、研究摘要預設收合`];
  });
  await shot("overview");
  await check(21, "走勢頁：期間切換與交易訊號", async () => {
    await page.click('.stabs [data-t="trend"]');
    await page.waitForTimeout(800);
    const c = await chartOk("#tc");
    const n = await page.$$eval("#hzSeg .hzc", (xs) => xs.length);
    const pills = await page.$$eval("#hzSeg .hzc .pill", (xs) => xs.map((x) => x.innerText.trim()));
    await page.click('#hzSeg [data-h="short"]'); await page.waitForTimeout(500);
    const t = await page.$eval("#t-ck h2", (e) => e.innerText);
    const plan = await page.$eval("#t-plan", (e) => e.innerText);
    const c2 = await chartOk("#tc");
    const order = await page.$eval("#tr-top", (e) => e.getBoundingClientRect().top < document.querySelector("#t-adv").getBoundingClientRect().top);
    await page.click('#hzSeg [data-h="swing"]'); await page.waitForTimeout(400);
    return [c.ok && c2.ok && n === 3 && /短線/.test(t) && /進場/.test(plan) && order && pills.length === 3,
      `${c.kind} ${c.wh}；三期間狀態：${pills.join("／")}；切到短線後條件檢查=${t.trim()}；研究細節在後`];
  });
  await check(22, "走勢頁：三種期間各自的圖表、條件與依據", async () => {
    const out = [];
    let ok = true;
    for (const [k, name] of [["short", "短線"], ["swing", "波段"], ["long", "長期"]]) {
      await page.click(`#hzSeg [data-h="${k}"]`); await page.waitForTimeout(600);
      const c = await chartOk("#tc");
      const r = await page.evaluate(() => ({ h: document.querySelector("#t-ck h2")?.innerText || "", rows: document.querySelectorAll("#t-ck .ck2").length,
        plan: document.querySelector("#t-plan")?.innerText || "", sig: Number(document.querySelector("#tc")?.dataset.sig ?? -1),
        basis: document.querySelector("#t-lbasis")?.innerText || "", cap: document.querySelector("#t-px p")?.innerText || "" }));
      const good = c.ok && r.h.includes(name) && r.rows > 0 && /進場/.test(r.plan) && (k === "long" ? r.sig === 0 && /季報到/.test(r.basis) && /不標進場點/.test(r.cap) : r.sig >= 0);
      ok = ok && good;
      out.push(`${name}：${good ? "通過" : "失敗"}（圖 ${c.wh}、條件 ${r.rows} 項、圖上訊號 ${r.sig} 個${k === "long" ? "、有判斷依據=" + /季報到/.test(r.basis) : ""}）`);
      if (process.env.SHOT_DIR) await page.locator("#tr-top").screenshot({ path: path.join(process.env.SHOT_DIR, `trend-${k}-${dev === "手機" ? "m" : "d"}.png`) });
    }
    await page.click('#hzSeg [data-h="swing"]'); await page.waitForTimeout(300);
    return [ok, out.join("；")];
  });
  await home();
  await check(10, "自選股", async () => {
    const star = await page.$("#hpick .pk [data-star]");
    const sid = await star.getAttribute("data-star");
    await star.click();
    const stayed = await page.evaluate(() => location.hash);
    const live = await page.$eval("#hwatch", (el, id) => el.innerHTML.includes(`#/stock/${id}`), sid);
    await page.reload(); await page.waitForSelector("#hwatch");
    const shown = await page.$eval("#hwatch", (el, id) => el.innerHTML.includes(`#/stock/${id}`), sid);
    const scored = await page.$$eval("#hwatch .pk-sc b", (bs) => bs.length);
    const notMoved = stayed === "#/" || stayed === "";
    return [live && shown && scored > 0 && notMoved, `加入 ${sid}；立即出現=${live}、重新整理後仍在=${shown}、有 AI 評分=${scored > 0}、按星號不跳頁=${notMoved}`];
  });
  await check(12, "產業熱力圖", async () => {
    await go("#/sectors", ".tmap");
    await page.waitForTimeout(500);
    const n = await page.$$eval(".tmap [data-k], .tmap > *", (xs) => xs.length);
    return [n > 3, `${n} 個色塊`];
  });
  await check(13, "正式選股成績單", async () => {
    await go("#/performance", "#app");
    await page.waitForFunction(() => document.querySelector("#app")?.innerText.includes("0050"), null, { timeout: 30000 });
    return [true, "績效中心有對照 0050 的成績"];
  });
  await check(14, "AI 每日自主學習", async () => {
    await go("#/research", "#app");
    await page.waitForFunction(() => { const e = document.querySelector("#evo"); return e && !e.hidden && /執行|學習/.test(e.innerText); }, null, { timeout: 30000 });
    const t = await page.$eval("#evo", (el) => el.innerText);
    return [/成功率|最近/.test(t), (t.match(/成功率\s*[\d.]+%/) || ["有學習紀錄"])[0]];
  });
  await check(15, "模型競賽", async () => {
    await page.waitForFunction(() => { const e = document.querySelector("#race"); return e && !e.hidden && e.innerText.length > 50; }, null, { timeout: 30000 });
    return [true, "研究頁有模型競賽結果"];
  });
  await check(16, "預測封存", async () => {
    const r = await (await fetch(base + "data/predictions/latest.json" + bust)).json();
    return [!!r.signal_date && !!r.model_hash && (r.n || r.stocks?.length) > 0, `封存日 ${r.signal_date}・${r.n ?? r.stocks?.length} 檔・模型 ${String(r.model_hash).slice(0, 8)}`];
  });
  await check(19, "會員登入（已實作部分）", async () => {
    await go("#/account", "#app");
    await page.waitForTimeout(800);
    const em = await page.$("#em");
    const t = await page.$eval("#app", (el) => el.innerText);
    return [!!em || /還沒有開啟會員功能/.test(t), em ? "有 Email 登入表單（不實際登入）" : "網站標示會員功能未啟用"];
  });
  await check(20, "AI 問答（已實作部分）", async () => {
    await go("#/ask", "#askIn");
    const before = await page.$eval("#app", (el) => el.innerText.length);
    if (await page.$eval("#askIn", (x) => x.disabled)) return [true, "問答輸入框標示暫停（有說明原因）"];
    await page.fill("#askIn", "台積電的主要風險是什麼？"); await page.press("#askIn", "Enter");
    await page.waitForFunction((n) => document.querySelector("#app").innerText.length > n + 20, before, { timeout: 30000 });
    return [true, "輸入問題後有回答"];
  });
  results.push({ device: dev, id: 0, name: "沒有 JavaScript 錯誤", ok: errs.length === 0, note: errs.slice(0, 3).join("；") });
  await ctx.close();
}
await browser.close();
results.sort((a, b) => (a.device === b.device ? 0 : a.device === "電腦" ? -1 : 1) || (a.id || 99) - (b.id || 99));
const passed = results.filter((r) => r.ok).length;
const rep = { at: new Date().toISOString(), site: base, as_of: asOf, passed, total: results.length, results };
if (out) { fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, JSON.stringify(rep, null, 1)); }
for (const r of results) console.log(`${r.ok ? "✅" : "❌"} [${r.device}] ${r.id} ${r.name}：${r.note}`);
console.log(`通過 ${passed}／${results.length}`);
process.exit(passed === results.length ? 0 : 1);
