---
name: ai-provider-integration
description: DFA-Auto-Flow의 멀티 LLM 프로바이더(Gemini/OpenAI/Claude) 통합과 사용자별 로컬 API 키 관리 스킬. API 키 하드코딩 제거, localStorage 런타임 로드, 중앙 aiClient 단일화, 키 미설정 시 우아한 비활성화, AI 기능(코드 설명·에러 해결·결과 해석·자동완성) 추가 시 사용.
---

# AI Provider Integration

## 왜 이렇게 하나
빌드 시 키를 번들에 주입하면 정적 호스팅(Vercel)에서 키가 클라이언트 코드에 그대로 노출된다. 각 사용자가 자기 키를 로컬에 넣으면 키 유출·쿼터 공유 문제가 사라지고 배포가 안전해진다.

## 중앙 클라이언트 단일화 (`utils/aiClient.ts`)
- 모든 `new GoogleGenAI({ apiKey: process.env.API_KEY })` 직접 호출을 `getAiClient()` / `generateAiText()` 단일 진입점으로 대체한다.
- 키 출처 우선순위: **localStorage(사용자 입력) → import.meta.env(dev 폴백)**. 빌드 주입(`process.env.API_KEY`)은 제거.
- 키가 없으면 `MissingApiKeyError`를 던지고, 호출부는 이를 잡아 "설정에서 API 키를 입력하세요" 안내 + 설정 모달 오픈을 유도한다. 앱을 크래시시키지 않는다.

## 멀티 프로바이더 추상화
- `provider`: `'gemini' | 'openai' | 'anthropic'`. 기본 gemini.
- 프로바이더별 키·모델을 localStorage에 저장(`dfa_ai_settings`). 응답은 `{ text }`로 정규화.
- 새 프로바이더 추가 시 호출부를 건드리지 않도록, 프로바이더 분기는 aiClient 내부에만 둔다.

## 보안 원칙
- 키를 콘솔/로그/Supabase/외부로 절대 전송하지 않는다.
- localStorage 평문 저장의 한계를 설정 UI에 1줄 고지한다("이 브라우저에만 저장됩니다").

## AI 기능 확장 패턴
각 기능은 `generateAiText({ provider, prompt, system })`을 호출하고, 키 미설정 시 비활성/안내한다.
- **코드 설명·에러 해결:** 생성 코드 또는 실행 에러 + 코드를 프롬프트로, 한국어 설명 + 수정안.
- **결과 해석:** 파이프라인 실행 출력(통계/모델/시뮬)을 계리 관점에서 요약.
- **파이프라인 자동완성:** 목표/데이터 → 모듈+연결+파라미터 JSON(기존 스키마 준수).

## 검증
키 미설정/무효/유효 3케이스로 호출부가 각각 안내·동작하는지 수동 확인. 빌드 산출물(`dist/`)에 평문 키가 없는지 grep으로 확인.
