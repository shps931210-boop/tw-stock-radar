// 部署後的實際測試：直接問「剛部署好的 Worker」三題（有 Claude 金鑰就是 Claude，沒有就是免費模型），檢查：
// 有查資料、有回答、有資料來源、追問時記得前面比較的兩家公司。結果寫成 Markdown（GitHub Actions 的執行摘要）。
// 用法：WORKER_URL=https://... SMOKE_TOKEN=... node smoke.mjs（SMOKE_TOKEN 每次部署由流程隨機產生）
const URL_ = process.env.WORKER_URL, TOKEN = process.env.SMOKE_TOKEN;

async function ask(messages) {
  const r = await fetch(`${URL_}/smoke`, { method: "POST", headers: { "Content-Type": "application/json", "X-Smoke": TOKEN }, body: JSON.stringify({ messages }) });
  const provider = r.headers.get("X-AI-Provider") || "?";
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  let text = "", steps = [], sources = [], err = null;
  for (const line of (await r.text()).split("\n")) {
    if (!line.trim()) continue;
    const { t, v } = JSON.parse(line);
    if (t === "text") text += v; else if (t === "status") steps.push(v); else if (t === "sources") sources = v; else if (t === "error") err = v;
  }
  return { text, steps, sources, err, provider };
}

const CASES = [
  { q: "台積電和聯發科的毛利率哪一家比較高？", need: ["台積電", "聯發科"], tool: true },
  { q: "那哪一家估值比較高？", need: ["台積電", "聯發科"], tool: true, followUp: true },
  { q: "最近有哪些重大的公司公告？", need: [], tool: true },
];

let fail = 0, hist = [], provider = "?";
const out = [];
for (const c of CASES) {
  const msgs = c.followUp ? [...hist, { role: "user", content: c.q }] : [{ role: "user", content: c.q }];
  let text = "", steps = [], sources = [], err = null;
  try {
    ({ text, steps, sources, err, provider } = await ask(msgs));
  } catch (e) { err = String(e?.message || e); }
  const problems = [];
  if (err) problems.push(`錯誤：${err}`);
  if (!text.trim()) problems.push("沒有回答");
  if (c.tool && !steps.length) problems.push("沒有查資料");
  if (c.tool && !sources.length) problems.push("沒有資料來源");
  for (const w of c.need) if (!text.includes(w)) problems.push(`回答沒有提到「${w}」`);
  if (!text.includes("不是投資建議")) problems.push("少了免責聲明");
  fail += problems.length ? 1 : 0;
  out.push(`### ${problems.length ? "❌" : "✅"} ${c.q}`, "", `查資料：${steps.join("、") || "無"}`, "", `來源：${sources.map((s) => s.name).join("、") || "無"}`, "",
    ...(problems.length ? [`**問題**：${problems.join("；")}`, ""] : []), "<details><summary>回答</summary>", "", text || "（空白）", "", "</details>", "");
  if (!c.followUp) hist = [{ role: "user", content: c.q }, { role: "assistant", content: text || "（無）" }];
}
const name = provider === "claude" ? "Claude（付費）" : provider === "free" ? "Cloudflare Workers AI 免費模型" : provider;
console.log([`## AI 研究助理實際測試（${name}）`, "", `${CASES.length - fail}／${CASES.length} 題通過。`, "", ...out].join("\n"));
process.exit(fail ? 1 : 0);
