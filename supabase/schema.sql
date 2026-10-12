-- 股栗子會員資料表：每位使用者一列，存觀察清單、自訂策略、個人偏好、研究筆記。
-- 在 Supabase 後台 SQL Editor 貼上整段、按 Run 執行一次即可（重複執行也不會出錯）。
-- Row Level Security：每個人只能讀寫自己的那一列，網站上的公開金鑰無法讀到別人的資料。

create table if not exists public.user_state (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  watch      jsonb not null default '{}'::jsonb,   -- 觀察清單 {代號: {added, price}}
  strategies jsonb not null default '[]'::jsonb,   -- 自訂條件選股策略
  pref       jsonb not null default '{}'::jsonb,   -- 我的 AI 排行偏好（期間、風格、風險、隱藏的股票）
  notes      jsonb not null default '{}'::jsonb,   -- 個股研究筆記 {代號: 文字}
  updated_at timestamptz not null default now()
);
-- 版本號：每次存檔 +1。網站存檔時要帶上次讀到的版本號，不一樣就代表另一台裝置剛改過，不覆寫（避免資料遺失）。
alter table public.user_state add column if not exists version integer not null default 1;

alter table public.user_state enable row level security;

drop policy if exists "own row: read" on public.user_state;
drop policy if exists "own row: insert" on public.user_state;
drop policy if exists "own row: update" on public.user_state;
drop policy if exists "own row: delete" on public.user_state;
create policy "own row: read"   on public.user_state for select to authenticated using ((select auth.uid()) = user_id);
create policy "own row: insert" on public.user_state for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "own row: update" on public.user_state for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "own row: delete" on public.user_state for delete to authenticated using ((select auth.uid()) = user_id);

grant usage on schema public to authenticated;
grant select, insert, update, delete on public.user_state to authenticated;
revoke all on public.user_state from anon;

-- ===== AI 問答用量（每位會員每天上限、全站每天上限） =====
-- 只有 AI 問答後端會呼叫 use_ai_quota()，並帶上會員的登入憑證；沒登入的人呼叫會得到 false。
-- 上限寫死在函式裡，使用者不能自己改；要調整請改下面兩個數字後重新執行整段 SQL。
create table if not exists public.ai_usage (
  day     date not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  n       integer not null default 0,
  primary key (day, user_id)
);
alter table public.ai_usage enable row level security;
drop policy if exists "own usage: read" on public.ai_usage;
create policy "own usage: read" on public.ai_usage for select to authenticated using ((select auth.uid()) = user_id);
grant select on public.ai_usage to authenticated;
revoke all on public.ai_usage from anon;

create or replace function public.use_ai_quota()
returns json
language plpgsql
security definer
set search_path = ''
as $$
declare
  uid uuid := auth.uid();
  today date := (now() at time zone 'Asia/Taipei')::date;
  per_user constant integer := 20;    -- 每位會員每天最多 20 題
  per_site constant integer := 100;   -- 全站每天最多 100 題（約 5 美元）
  mine integer;
  total integer;
begin
  if uid is null then
    return json_build_object('ok', false, 'reason', 'login');
  end if;
  perform pg_advisory_xact_lock(hashtext('ai_usage'));  -- 同時有很多人問時，計數不會算錯
  select coalesce(sum(n), 0) into total from public.ai_usage where day = today;
  select coalesce(n, 0) into mine from public.ai_usage where day = today and user_id = uid;
  mine := coalesce(mine, 0);
  if total >= per_site then
    return json_build_object('ok', false, 'reason', 'site', 'used', mine, 'limit', per_user);
  end if;
  if mine >= per_user then
    return json_build_object('ok', false, 'reason', 'user', 'used', mine, 'limit', per_user);
  end if;
  insert into public.ai_usage (day, user_id, n) values (today, uid, 1)
    on conflict (day, user_id) do update set n = public.ai_usage.n + 1;
  return json_build_object('ok', true, 'used', mine + 1, 'limit', per_user);
end;
$$;
revoke all on function public.use_ai_quota() from public, anon;
grant execute on function public.use_ai_quota() to authenticated;
