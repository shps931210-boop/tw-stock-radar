// 研究助理的對話迴圈：Claude 依問題呼叫資料工具（tools.js），直到寫出回答。index.js（正式服務）和 smoke.mjs（部署後的實際測試）共用。
import Anthropic from "@anthropic-ai/sdk";
import { costOf } from "./auth.js";
import { SYSTEM } from "./prompt.js";
import { TOOLS, makeRunner, toolLabel } from "./tools.js";

export const MAX_STEPS = 6;  // 一題最多查幾輪資料

/** send(t, v)：t = "status" | "text" | "sources"。花費（美元）一邊算一邊加進 meter.usd（中途出錯也記得到）。錯誤會往外丟，由呼叫端轉成訊息。 */
export async function runAgent({ client, env, messages, page, load, send, meter = { usd: 0 } }) {
  const runTool = makeRunner(load);
  const system = [{ type: "text", text: SYSTEM + (page ? `\n\n使用者目前正在看 ${page} 的個股頁；問題沒有指明股票時，通常是在問這一檔。` : ""), cache_control: { type: "ephemeral" } }];
  const sources = new Map();
  const convo = [...messages];
  let retries = 0, useBeta = env.NO_FALLBACK_BETA !== "1";
  for (let step = 0; step < MAX_STEPS; step++) {
    const base = {
      model: env.MODEL || "claude-opus-5-5",
      max_tokens: 3000,  // 每一輪的輸出上限；花費上限見 supabase/schema.sql
      thinking: { type: "adaptive" },
      output_config: { effort: "medium" },
      system, tools: TOOLS, messages: convo,
    };
    // 預設加上「安全機制拒答時自動改用備援模型」；如果帳號或模型不支援這個參數（400），改用基本參數重試，不讓使用者看到錯誤
    const stream = useBeta
      ? client.beta.messages.stream({ ...base, betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" })
      : client.messages.stream(base);
    let wrote = false;
    stream.on("text", (delta) => { wrote = true; send("text", delta); });
    let msg;
    try {
      msg = await stream.finalMessage();
      retries = 0;
    } catch (e) {
      if (useBeta && !wrote && e instanceof Anthropic.BadRequestError && !/credit|billing/i.test(e.message)) {
        console.warn("beta params rejected, retrying without", e.message);
        useBeta = false; step--; continue;
      }
      if (e instanceof Anthropic.APIError || retries++ >= 2) throw e;
      continue;  // 工具參數不是完整的 JSON：同一輪重來
    }
    meter.usd += costOf(msg.model, msg.usage);
    if (msg.stop_reason === "refusal") { await send("text", "\n（這個問題 AI 無法回答，請換個方式問。）"); break; }
    if (msg.stop_reason === "max_tokens") { await send("text", "\n（回答太長被截斷了，可以請它精簡一點。）"); break; }
    const uses = msg.content.filter((b) => b.type === "tool_use");
    if (msg.stop_reason !== "tool_use" || !uses.length) break;
    convo.push({ role: "assistant", content: msg.content });
    const results = [];
    for (const u of uses) {
      await send("status", toolLabel(u.name, u.input));
      const r = await runTool(u.name, u.input);
      (r.sources || []).forEach((x) => sources.set(x.path, x));
      results.push({ type: "tool_result", tool_use_id: u.id, content: JSON.stringify(r.error ? { 錯誤: r.error } : r.result), ...(r.error ? { is_error: true } : {}) });
    }
    convo.push({ role: "user", content: results });
    if (step === MAX_STEPS - 1) await send("text", "\n（查詢步驟太多，先回答到這裡；可以把問題拆小一點再問。）");
  }
  if (sources.size) await send("sources", [...sources.values()]);
  return meter.usd;
}
