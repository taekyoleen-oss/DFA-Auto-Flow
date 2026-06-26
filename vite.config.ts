import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    // 보안: API 키를 빌드 산출물(번들)에 주입하지 않는다.
    // AI 키는 런타임에 사용자 브라우저의 localStorage에서 읽는다(utils/aiClient.ts).
    // dev 폴백이 필요하면 VITE_GEMINI_API_KEY 등 VITE_ 접두 변수를 import.meta.env로 사용한다.
    // 환경 변수 로드(.env, .env.local, .env.[mode], .env.[mode].local) — 고급기능 비밀번호 해시 주입용.
    const env = loadEnv(mode, process.cwd(), '');
    return {
      envPrefix: ['VITE_', 'NEXT_PUBLIC_'],
      define: {
        // 고급기능 비밀번호 검증값(비밀번호의 SHA-256 hex 권장, 평문도 허용되나 번들 노출).
        // Vercel/로컬 환경변수에서 빌드 시 주입. 변수명: ADVANCED_PASSWORD_HASH
        // 또는 VITE_ADVANCED_PASSWORD_HASH / NEXT_PUBLIC_ADVANCED_PASSWORD_HASH 모두 허용.
        // 미설정 시 빈 문자열 → 항상 잠금(안전한 기본값).
        '__ADVANCED_PASSWORD_HASH__': JSON.stringify(
          process.env.ADVANCED_PASSWORD_HASH
          || process.env.VITE_ADVANCED_PASSWORD_HASH
          || process.env.NEXT_PUBLIC_ADVANCED_PASSWORD_HASH
          || env.ADVANCED_PASSWORD_HASH
          || env.VITE_ADVANCED_PASSWORD_HASH
          || env.NEXT_PUBLIC_ADVANCED_PASSWORD_HASH
          || ''
        )
      },
      server: {
        port: 3001,
        host: '0.0.0.0',
        proxy: {
          '/api/split-data': {
            target: 'http://localhost:3001',
            changeOrigin: true,
            secure: false,
          },
          '/api/proxy-csv': {
            target: 'http://localhost:3001',
            changeOrigin: true,
            secure: false,
          },
          '/api/samples': {
            target: 'http://localhost:3002',
            changeOrigin: true,
            secure: false,
          }
        }
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
