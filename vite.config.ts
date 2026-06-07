import path from 'path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
    // 보안: API 키를 빌드 산출물(번들)에 주입하지 않는다.
    // AI 키는 런타임에 사용자 브라우저의 localStorage에서 읽는다(utils/aiClient.ts).
    // dev 폴백이 필요하면 VITE_GEMINI_API_KEY 등 VITE_ 접두 변수를 import.meta.env로 사용한다.
    return {
      envPrefix: ['VITE_', 'NEXT_PUBLIC_'],
      server: {
        port: 3001,
        host: '0.0.0.0',
        proxy: {
          '/api/split-data': {
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
