// 免費模型版的研究助理對話迴圈：用 Cloudflare Workers AI（每天有免費額度），工具和 Claude 版（agent.js）共用 tools.js。
// 沒有設定 ANTHROPIC_API_KEY 時，index.js 會改用這裡。介面和 runAgent 一樣：send(t, v) 傳回 status / text / sources。
// 不同模型回傳的格式不太一樣（OpenAI 格式的 choices[0].message，或 {response, tool_calls}），這裡都接得住。
// 工具結果用一般的對話文字送回模型，不依賴各家模型不一致的 tool 訊息格式。
import { SYSTEM } from "./prompt.js";
import { TOOLS, makeRunner, toolLabel } from "./tools.js";

export const FREE_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8";  // 繁體中文較好、支援工具呼叫、很便宜（免費額度用得久）
export const MAX_STEPS = 5;
const DISCLAIMER = "以上根據股栗子網站的公開資料整理，不是投資建議。";
const MAX_RESULT = 6000;  // 每次工具結果最多給模型幾個字（免費模型的上下文較小）

export const FREE_TOOLS = TOOLS.map((t) => ({ name: t.name, description: t.description, parameters: t.input_schema }));

const EXTRA = `

補充規則（免費模型版）：
- 你可以呼叫工具查資料。每次最多呼叫 2 個工具，查到足夠的資料就直接回答，不要重複查同樣的東西。
- 只能用工具查到的數字，不能自己編數字。工具沒有回傳的資料，就說網站目前查不到。
- 一律使用繁體中文（台灣用語），不要用簡體字。`;

/** 把模型回應整理成 { text, calls:[{name, args}] } */
export function parseReply(res) {
  const m = res?.choices?.[0]?.message;
  let text = m ? (m.content ?? "") : (res?.response ?? "");
  if (typeof text !== "string") text = JSON.stringify(text);
  const raw = m?.tool_calls || res?.tool_calls || [];
  const calls = raw.map((c) => {
    const name = c.function?.name || c.name;
    let args = c.function?.arguments ?? c.arguments ?? {};
    if (typeof args === "string") { try { args = JSON.parse(args || "{}"); } catch { args = null; } }
    return { name, args };
  }).filter((c) => c.name);
  text = text.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/^[\s\S]*?<\/think>/, "").trim();
  return { text, calls };
}

const clip = (v) => {
  const s = JSON.stringify(v);
  return s.length > MAX_RESULT ? s.slice(0, MAX_RESULT) + "…（資料太長，後面省略）" : s;
};

export async function runFreeAgent({ ai, env, messages, page, load, send, meter = { usd: 0 } }) {
  const runTool = makeRunner(load);
  const model = env.FREE_MODEL || FREE_MODEL;
  const system = SYSTEM + EXTRA + (page ? `\n\n使用者目前正在看 ${page} 的個股頁；問題沒有指明股票時，通常是在問這一檔。` : "");
  const convo = [{ role: "system", content: system }, ...messages.map((m) => ({ role: m.role, content: String(m.content) }))];
  const sources = new Map();
  const known = new Set(FREE_TOOLS.map((t) => t.name));
  for (let step = 0; step < MAX_STEPS; step++) {
    const last = step === MAX_STEPS - 1;
    const res = await ai.run(model, { messages: convo, ...(last ? {} : { tools: FREE_TOOLS }), max_tokens: 2000, temperature: 0.3 });
    const { text, calls } = parseReply(res);
    const use = calls.filter((c) => known.has(c.name)).slice(0, 2);
    if (!use.length) {
      const body = text || "（AI 沒有產生回答，請換個方式再問一次。）";
      await send("text", body.includes("不是投資建議") ? body : body + "\n\n" + DISCLAIMER);
      break;
    }
    const parts = [];
    for (const c of use) {
      await send("status", toolLabel(c.name, c.args || {}));
      const r = c.args ? await runTool(c.name, c.args) : { error: "參數格式錯誤" };
      (r.sources || []).forEach((x) => sources.set(x.path, x));
      parts.push(`【${c.name} ${JSON.stringify(c.args || {})} 的結果】\n${r.error ? `錯誤：${r.error}` : clip(r.result)}`);
    }
    convo.push({ role: "assistant", content: (text ? text + "\n" : "") + `（查詢：${use.map((c) => c.name).join("、")}）` });
    convo.push({ role: "user", content: `${parts.join("\n\n")}\n\n請根據以上資料繼續；資料足夠就直接用繁體中文回答原本的問題。` });
  }
  if (sources.size) await send("sources", [...sources.values()]);
  return meter.usd;
}
