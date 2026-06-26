/**
 * 중앙 AI 클라이언트 — 멀티 프로바이더(Anthropic Claude / Gemini / OpenAI) + 사용자 로컬 API 키.
 *
 * 설계 원칙(.claude/skills/ai-provider-integration):
 *  - 기본 프로바이더는 Anthropic Claude. 모든 AI 호출은 프로바이더 비의존 경로(generateAiText)를 사용한다.
 *  - API 키는 번들에 넣지 않는다. 런타임에 localStorage에서 읽는다.
 *  - 키가 없으면 MissingApiKeyError를 던지고, UI가 설정 모달을 열도록 유도한다(앱 크래시 금지).
 *  - 모든 LLM 호출은 이 파일을 단일 진입점으로 사용한다.
 *  - 키는 콘솔/로그/외부로 전송하지 않는다(프로바이더 API 호출 제외).
 */
import Anthropic from "@anthropic-ai/sdk";

export type AiProvider = "anthropic" | "gemini" | "openai";

export interface AiSettings {
  provider: AiProvider;
  keys: Record<AiProvider, string>;
  models: Record<AiProvider, string>;
}

const STORAGE_KEY = "dfa_ai_settings";

/**
 * 작업 티어 → 권장 모델.
 *  - fast: 결과 해설/짧은 요약류(빠르고 저렴). Claude Haiku.
 *  - smart: 모듈 추천/JSON 구조화/코드·인사이트 해석(추론 필요). Claude Sonnet.
 * 설정 모달의 anthropic 기본값은 smart(=Sonnet)이며, fast 티어 호출은 자동으로 Haiku를 쓴다.
 */
export type AiTier = "fast" | "smart";

export const ANTHROPIC_TIER_MODELS: Record<AiTier, string> = {
  fast: "claude-haiku-4-5",
  smart: "claude-sonnet-4-6",
};

export const DEFAULT_MODELS: Record<AiProvider, string> = {
  anthropic: ANTHROPIC_TIER_MODELS.smart,
  gemini: "gemini-2.5-flash",
  openai: "gpt-4o-mini",
};

export const PROVIDER_LABELS: Record<AiProvider, string> = {
  anthropic: "Anthropic Claude",
  gemini: "Google Gemini",
  openai: "OpenAI",
};

export class MissingApiKeyError extends Error {
  provider: AiProvider;
  constructor(provider: AiProvider) {
    super(`${PROVIDER_LABELS[provider]} API 키가 설정되지 않았습니다. 설정에서 키를 입력하세요.`);
    this.name = "MissingApiKeyError";
    this.provider = provider;
  }
}

/** dev 환경 폴백: 빌드 주입이 아니라 Vite의 import.meta.env(VITE_*)에서만 읽는다. */
function envFallbackKey(provider: AiProvider): string {
  try {
    const env = (import.meta as any)?.env || {};
    if (provider === "anthropic") return env.VITE_ANTHROPIC_API_KEY || "";
    if (provider === "gemini") return env.VITE_GEMINI_API_KEY || "";
    if (provider === "openai") return env.VITE_OPENAI_API_KEY || "";
  } catch {
    /* import.meta 미지원 환경 */
  }
  return "";
}

export function getAiSettings(): AiSettings {
  const base: AiSettings = {
    provider: "anthropic",
    keys: { anthropic: "", gemini: "", openai: "" },
    models: { ...DEFAULT_MODELS },
  };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      base.provider = parsed.provider || base.provider;
      base.keys = { ...base.keys, ...(parsed.keys || {}) };
      base.models = { ...base.models, ...(parsed.models || {}) };
    }
  } catch {
    /* 손상된 설정은 무시하고 기본값 사용 */
  }
  return base;
}

export function saveAiSettings(settings: AiSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  try {
    window.dispatchEvent(new CustomEvent("dfa:ai-settings-changed"));
  } catch {
    /* SSR/비브라우저 */
  }
}

/** 해당 프로바이더의 유효 키(로컬 입력 우선, 없으면 dev 폴백). */
export function getApiKey(provider?: AiProvider): string {
  const settings = getAiSettings();
  const p = provider || settings.provider;
  return settings.keys[p] || envFallbackKey(p);
}

export function hasApiKey(provider?: AiProvider): boolean {
  return !!getApiKey(provider);
}

/** 설정 모달을 열도록 앱에 신호를 보낸다. */
export function requestOpenAiSettings(): void {
  try {
    window.dispatchEvent(new CustomEvent("dfa:open-ai-settings"));
  } catch {
    /* noop */
  }
}

/** 코드펜스(```json … ```)를 제거하고 순수 JSON 문자열만 반환한다. */
function stripJsonFence(text: string): string {
  let t = (text || "").trim();
  if (t.startsWith("```")) {
    // ```json\n … \n``` 또는 ```\n … \n```
    t = t.replace(/^```[a-zA-Z]*\s*\n?/, "").replace(/\n?```\s*$/, "").trim();
  }
  return t;
}

export interface GenerateTextOptions {
  prompt: string;
  system?: string;
  /** JSON 출력 강제(프로바이더별로 JSON 모드/프롬프트 가드를 적용). */
  json?: boolean;
  /**
   * Gemini용 responseSchema(@google/genai Type 스키마)였던 인자.
   * Claude/OpenAI에선 사용하지 않으며 json 모드만 적용한다.
   * (구조 강제는 json:true + 프롬프트에서 출력 형태를 명시하는 방식으로 처리한다.)
   */
  schema?: any;
  provider?: AiProvider;
  /** 명시 모델. 없으면 tier(anthropic) 또는 설정/기본 모델을 사용한다. */
  model?: string;
  /** 작업 티어. anthropic에서 model 미지정 시 fast=Haiku / smart=Sonnet를 선택한다. */
  tier?: AiTier;
  temperature?: number;
  /**
   * 최대 출력 토큰. 미지정 시 4096(기본). 모델 분석보고서처럼 긴 HTML 출력에는
   * 더 큰 값(예: 16000)을 넘긴다. 프로바이더별 max_tokens/max_output_tokens에 반영.
   */
  maxTokens?: number;
}

/** anthropic 분기에서 사용할 모델 결정: model > tier 매핑 > 설정/기본. */
function resolveAnthropicModel(opts: GenerateTextOptions, settings: AiSettings): string {
  if (opts.model) return opts.model;
  if (opts.tier) return ANTHROPIC_TIER_MODELS[opts.tier];
  return settings.models.anthropic || DEFAULT_MODELS.anthropic;
}

/**
 * 프로바이더 비의존 텍스트 생성. 새 AI 기능은 이 함수를 사용한다.
 * 반환값은 모델이 생성한 텍스트(json 모드면 코드펜스를 제거한 JSON 문자열).
 */
export async function generateAiText(opts: GenerateTextOptions): Promise<string> {
  const settings = getAiSettings();
  const provider = opts.provider || settings.provider;
  const key = getApiKey(provider);
  if (!key) {
    requestOpenAiSettings();
    throw new MissingApiKeyError(provider);
  }

  // ── Anthropic Claude (기본) ──────────────────────────────────────────────
  if (provider === "anthropic") {
    const model = resolveAnthropicModel(opts, settings);
    const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
    const userContent = opts.json
      ? `${opts.prompt}\n\n반드시 유효한 JSON만 출력하세요. 코드펜스(\`\`\`)나 다른 설명 텍스트 없이 JSON 객체만 응답하세요.`
      : opts.prompt;
    const response = await client.messages.create({
      model,
      max_tokens: opts.maxTokens ?? 4096,
      ...(opts.system ? { system: opts.system } : {}),
      ...(opts.temperature != null ? { temperature: opts.temperature } : {}),
      messages: [{ role: "user", content: userContent }],
    });
    const text = (response.content || [])
      .filter((b: any) => b.type === "text")
      .map((b: any) => b.text)
      .join("")
      .trim();
    return opts.json ? stripJsonFence(text) : text;
  }

  const model = opts.model || settings.models[provider] || DEFAULT_MODELS[provider];

  // ── Google Gemini ────────────────────────────────────────────────────────
  if (provider === "gemini") {
    // @google/genai는 anthropic 전환 후 선택적 의존성. 동적 import로 번들에서 분리한다.
    const { GoogleGenAI } = await import("@google/genai");
    const ai = new GoogleGenAI({ apiKey: key });
    const config: any = {};
    if (opts.json) config.responseMimeType = "application/json";
    if (opts.schema) config.responseSchema = opts.schema;
    if (opts.temperature != null) config.temperature = opts.temperature;
    if (opts.system) config.systemInstruction = opts.system;
    if (opts.maxTokens != null) config.maxOutputTokens = opts.maxTokens;
    const response = await ai.models.generateContent({
      model,
      contents: opts.prompt,
      ...(Object.keys(config).length ? { config } : {}),
    });
    const text = (response.text || "").trim();
    return opts.json ? stripJsonFence(text) : text;
  }

  // ── OpenAI ────────────────────────────────────────────────────────────────
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        ...(opts.system ? [{ role: "system", content: opts.system }] : []),
        { role: "user", content: opts.prompt },
      ],
      ...(opts.temperature != null ? { temperature: opts.temperature } : {}),
      ...(opts.maxTokens != null ? { max_tokens: opts.maxTokens } : {}),
      ...(opts.json ? { response_format: { type: "json_object" } } : {}),
    }),
  });
  if (!res.ok) throw new Error(`OpenAI 오류 ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const text = (data.choices?.[0]?.message?.content || "").trim();
  return opts.json ? stripJsonFence(text) : text;
}
