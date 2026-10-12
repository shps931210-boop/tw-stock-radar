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
