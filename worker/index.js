// 股栗子 AI 個股問答：網站送來「股票代號 + 對話」，這裡讀網站上這檔股票的公開資料，交給 Claude 回答，邊寫邊傳回去。
// 金鑰只存在 Cloudflare（ANTHROPIC_API_KEY secret），網站和 GitHub 上都看不到。
import Anthropic from "@anthropic-ai/sdk";
import { SYSTEM, buildContext, cleanMessages, validId } from "./prompt.js";

const cors = (env) => ({
  "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN,
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Vary": "Origin",
});
const fail = (env, status, msg) =>
  new Response(JSON.stringify({ error: msg }), { status, headers: { ...cors(env), "Content-Type": "application/json; charset=utf-8" } });

export default {
  async fetch(req, env, ctx) {
    if (req.method === "OPTIONS") return new Response(null, { headers: cors(env) });
    if (req.method !== "POST") return fail(env, 405, "只接受 POST");
    if (req.headers.get("Origin") !== env.ALLOWED_ORIGIN) return fail(env, 403, "只開放股栗子網站使用");
    if (!env.ANTHROPIC_API_KEY) return fail(env, 503, "AI 問答還沒設定金鑰");
    const ip = req.headers.get("CF-Connecting-IP") || "unknown";
    if (env.LIMITER) {
      const { success } = await env.LIMITER.limit({ key: ip });
      if (!success) return fail(env, 429, "問得太快了，請一分鐘後再試");
    }

    let body;
    try {
      body = await req.json();
    } catch {
      return fail(env, 400, "格式錯誤");
    }
    if (!validId(body.stock_id)) return fail(env, 400, "股票代號格式錯誤");
    let messages;
    try {
      messages = cleanMessages(body.messages);
    } catch (e) {
      return fail(env, 400, e.message);
    }

    // 讀網站上這檔股票的最新公開資料（GitHub Pages，快取 10 分鐘）
    const r = await fetch(`${env.SITE}/data/stock/${body.stock_id}.json`, { cf: { cacheTtl: 600 } });
    if (!r.ok) return fail(env, 404, `網站上沒有 ${body.stock_id} 的資料`);
    const data = await r.json();

    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
    const stream = client.beta.messages.stream({
      model: env.MODEL || "claude-opus-5-5",
      max_tokens: 4000,
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default", // 安全機制拒答時，由 Anthropic 自動改用建議的備援模型重試
      output_config: { effort: "medium" },
      // 規則 + 這檔股票的資料放在最前面並快取，同一檔接著問比較便宜
      system: [
        { type: "text", text: SYSTEM },
        { type: "text", text: `這檔股票的資料（JSON，來自股栗子網站）：\n${buildContext(data)}`, cache_control: { type: "ephemeral" } },
      ],
      messages,
    });

    const { readable, writable } = new TransformStream();
    const w = writable.getWriter();
    const enc = new TextEncoder();
    ctx.waitUntil((async () => {
      try {
        for await (const ev of stream) {
          if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") await w.write(enc.encode(ev.delta.text));
        }
        const final = await stream.finalMessage();
        if (final.stop_reason === "refusal") await w.write(enc.encode("\n（這個問題 AI 無法回答，請換個方式問。）"));
        else if (final.stop_reason === "max_tokens") await w.write(enc.encode("\n（回答太長被截斷了，可以請它精簡一點。）"));
      } catch (e) {
        const msg = e instanceof Anthropic.RateLimitError ? "AI 服務忙碌中，請稍後再試。"
          : e instanceof Anthropic.AuthenticationError ? "AI 金鑰無效，請檢查設定。"
          : e instanceof Anthropic.APIError && e.status === 400 && /credit|billing/i.test(e.message) ? "AI 額度用完了，請到 Anthropic Console 儲值。"
          : "AI 暫時無法回答，請稍後再試。";
        console.error("chat error", e?.status, e?.message);
        await w.write(enc.encode(`\n（${msg}）`));
      } finally {
        await w.close();
      }
    })());
    return new Response(readable, { headers: { ...cors(env), "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" } });
  },
};
