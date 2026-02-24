<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1Bxt23-OfQovUeGfFHeNdu0MokVuHV1Pm

## Run Locally

**Prerequisites:** Node.js (v18+), pnpm 또는 npm

1. **의존성 설치**
   ```bash
   pnpm install
   ```
2. **(선택)** `.env` 또는 `.env.local`에 `GEMINI_API_KEY` 설정. Supabase 사용 시 `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` 또는 `NEXT_PUBLIC_*` 설정.
3. **개발 서버 실행**
   ```bash
   pnpm run dev
   ```
4. 브라우저에서 **http://localhost:3001** 로 접속합니다.

**실행이 안 될 때**
- 터미널을 **프로젝트 폴더**(`DFA-Auto-Flow`)에서 연 뒤 `pnpm run dev` 실행.
- 한 번도 설치하지 않았다면 먼저 `pnpm install` 실행.
- 포트 3001이 사용 중이면 다른 앱을 종료하거나, `vite.config.ts`의 `server.port`를 변경.
