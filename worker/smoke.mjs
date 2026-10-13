// 部署後的實際測試：真的呼叫 Claude（用 ANTHROPIC_API_KEY），讀正式網站的資料，問三題並檢查：
// 有查資料、有回答、有資料來源、追問時記得前面比較的兩家公司。結果寫成 Markdown（GitHub Actions 的執行摘要）。
// 花費大約幾美分到十幾美分。用法：ANTHROPIC_API_KEY=... node smoke.mjs
import Anthropic from "@anthropic-ai/sdk";
import { runAgent } from "./agent.js";

const SITE = process.env.SITE || "https://shps931210-boop.github.io/tw-stock-radar";
const env = { MODEL: process.env.MODEL || "claude-opus-5-5" };
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const load = async (p) => { const r = await fetch(`${SITE}/data/${p}`); if (!r.ok) throw new Error(`${p} ${r.status}`); return r.json(); };

const CASES = [
  { q: "台積電和聯發科的毛利率哪一家比較高？", need: ["台積電", "聯發科"], tool: true },
  { q: "那哪一家估值比較高？", need: ["台積電", "聯發科"], tool: true, followUp: true },
  { q: "最近有哪些重大的公司公告？", need: [], tool: true },
];

let fail = 0, total = 0, hist = [];
const out = [`## AI 研究助理實際測試（模型 ${env.MODEL}）`, ""];
for (const c of CASES) {
  const msgs = c.followUp ? [...hist, { role: "user", content: c.q }] : [{ role: "user", content: c.q }];
  let text = "", steps = [], sources = [], err = null;
  const meter = { usd: 0 };
  try {
    await runAgent({ client, env, messages: msgs, load, meter, send: async (t, v) => { if (t === "text") text += v; else if (t === "status") steps.push(v); else if (t === "sources") sources = v; } });
  } catch (e) { err = `${e?.status || ""} ${e?.message || e}`; }
  total += meter.usd;
  const problems = [];
  if (err) problems.push(`錯誤：${err}`);
  if (!text.trim()) problems.push("沒有回答");
  if (c.tool && !steps.length) problems.push("沒有查資料");
  if (c.tool && !sources.length) problems.push("沒有資料來源");
  for (const w of c.need) if (!text.includes(w)) problems.push(`回答沒有提到「${w}」`);
  if (!text.includes("不是投資建議")) problems.push("少了免責聲明");
  fail += problems.length ? 1 : 0;
  out.push(`### ${problems.length ? "❌" : "✅"} ${c.q}`, "", `查資料：${steps.join("、") || "無"}`, "", `來源：${sources.map((s) => s.name).join("、") || "無"}・花費約 $${meter.usd.toFixed(3)}`, "",
    ...(problems.length ? [`**問題**：${problems.join("；")}`, ""] : []), "<details><summary>回答</summary>", "", text || "（空白）", "", "</details>", "");
  if (!c.followUp) hist = [{ role: "user", content: c.q }, { role: "assistant", content: text || "（無）" }];
}
out.push(`總花費約 $${total.toFixed(3)}。`);
console.log(out.join("\n"));
process.exit(fail ? 1 : 0);
