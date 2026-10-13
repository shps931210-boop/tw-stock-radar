// 股栗子 AI 研究助理：網站送來「對話」（可附上目前所在的個股頁），Claude 依問題呼叫資料工具（worker/tools.js）
// 查個股、多檔比較、條件篩選、大盤、產業、公告新聞、選股績效，再整理回答，邊寫邊傳回去。
// 回傳格式是一行一個 JSON（NDJSON）：{"t":"status"} 查資料進度、{"t":"text"} 回答文字、{"t":"sources"} 用到的資料、{"t":"error"}。
// 金鑰只存在 Cloudflare（ANTHROPIC_API_KEY secret），網站和 GitHub 上都看不到。
// 只回答登入的會員：每次都用會員的登入憑證向 Supabase 確認身分並扣一題額度（每人、全站每天都有上限，見 supabase/schema.sql）。
// Origin 檢查和每分鐘次數限制只是額外的防護，不能當成身分驗證。
import Anthropic from "@anthropic-ai/sdk";
import { runAgent } from "./agent.js";
import { checkQuota, recordCost } from "./auth.js";
import { cleanMessages, validId } from "./prompt.js";

const cors = (env) => ({
  "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN,
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Expose-Headers": "X-Quota-Used, X-Quota-Limit",
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
    const page = validId(body.stock_id) ? body.stock_id : null;  // 選填：使用者目前在哪一檔的個股頁
    let messages;
    try {
      messages = cleanMessages(body.messages);
    } catch (e) {
      return fail(env, 400, e.message);
    }

    // 會員身分與每日額度（沒通過就不會呼叫付費 AI）
    const quota = await checkQuota(env, req.headers.get("Authorization"));
    if (!quota.ok) {
      console.warn("quota denied", quota.status, quota.reason || "", ip);
      return fail(env, quota.status, quota.error);
    }

    // 資料工具：讀網站上的公開 JSON（GitHub Pages，快取 10 分鐘；同一題內重複讀只抓一次）
    const memo = new Map();
    const load = (path) => {
      if (!memo.has(path)) memo.set(path, fetch(`${env.SITE}/data/${path}`, { cf: { cacheTtl: 600 } }).then((r) => {
        if (!r.ok) throw new Error(`讀不到 ${path}（${r.status}）`);
        return r.json();
      }));
      return memo.get(path);
    };
    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });

    const { readable, writable } = new TransformStream();
    const w = writable.getWriter();
    const enc = new TextEncoder();
    const send = (t, v) => w.write(enc.encode(JSON.stringify({ t, v }) + "\n"));
    ctx.waitUntil((async () => {
      const meter = { usd: 0 };
      try {
        await runAgent({ client, env, messages, page, load, send, meter });
      } catch (e) {
        const msg = e instanceof Anthropic.RateLimitError ? "AI 服務忙碌中，請稍後再試。"
          : e instanceof Anthropic.AuthenticationError ? "AI 金鑰無效，請檢查設定。"
          : e instanceof Anthropic.APIError && e.status === 400 && /credit|billing/i.test(e.message) ? "AI 額度用完了，請到 Anthropic Console 儲值。"
          : "AI 暫時無法回答，請稍後再試。";
        console.error("chat error", e?.status, e?.message);
        await send("error", msg);
      } finally {
        console.log("cost", meter.usd);
        await recordCost(env, req.headers.get("Authorization"), meter.usd).catch(() => {});
        await w.close();
      }
    })());
    return new Response(readable, { headers: { ...cors(env), "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store",
      "X-Quota-Used": String(quota.used ?? ""), "X-Quota-Limit": String(quota.limit ?? "") } });
  },
};
