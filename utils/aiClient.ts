/**
 * 중앙 AI 클라이언트 — 멀티 프로바이더(Gemini / OpenAI / Anthropic) + 사용자 로컬 API 키.
 *
 * 설계 원칙(.claude/skills/ai-provider-integration):
 *  - API 키는 번들에 넣지 않는다. 런타임에 localStorage에서 읽는다.
 *  - 키가 없으면 MissingApiKeyError를 던지고, UI가 설정 모달을 열도록 유도한다(앱 크래시 금지).
 *  - 모든 LLM 호출은 이 파일을 단일 진입점으로 사용한다.
 *  - 키는 콘솔/로그/외부로 전송하지 않는다(프로바이더 API 호출 제외).
 */
import { GoogleGenAI } from "@google/genai";

export type AiProvider = "gemini" | "openai" | "anthropic";

export interface AiSettings {
  provider: AiProvider;
  keys: Record<AiProvider, string>;
  models: Record<AiProvider, string>;
}

const STORAGE_KEY = "dfa_ai_settings";

export const DEFAULT_MODELS: Record<AiProvider, string> = {
  gemini: "gemini-2.5-flash",
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
};

export const PROVIDER_LABELS: Record<AiProvider, string> = {
  gemini: "Google Gemini",
  openai: "OpenAI",
  anthropic: "Anthropic Claude",
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
    if (provider === "gemini") return env.VITE_GEMINI_API_KEY || "";
    if (provider === "openai") return env.VITE_OPENAI_API_KEY || "";
    if (provider === "anthropic") return env.VITE_ANTHROPIC_API_KEY || "";
  } catch {
    /* import.meta 미지원 환경 */
  }
  return "";
}

export function getAiSettings(): AiSettings {
  const base: AiSettings = {
    provider: "gemini",
    keys: { gemini: "", openai: "", anthropic: "" },
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

/**
 * 기존 Gemini 전용 호출부(구조화 스키마 등)를 위한 클라이언트.
 * 로컬 Gemini 키를 사용하며, 없으면 MissingApiKeyError를 던진다.
 */
export function getGeminiClient(): GoogleGenAI {
  const key = getApiKey("gemini");
  if (!key) {
    requestOpenAiSettings();
    throw new MissingApiKeyError("gemini");
  }
  return new GoogleGenAI({ apiKey: key });
}

export interface GenerateTextOptions {
  prompt: string;
  system?: string;
  /** JSON 출력 강제 */
  json?: boolean;
  /** Gemini용 responseSchema(@google/genai Type 스키마). 다른 프로바이더에선 무시되고 json 모드만 적용. */
  schema?: any;
  provider?: AiProvider;
  model?: string;
  temperature?: number;
}

/**
 * 프로바이더 비의존 텍스트 생성. 새 AI 기능은 이 함수를 사용한다.
 * 반환값은 모델이 생성한 텍스트(json 모드면 JSON 문자열).
 */
export async function generateAiText(opts: GenerateTextOptions): Promise<string> {
  const settings = getAiSettings();
  const provider = opts.provider || settings.provider;
  const model = opts.model || settings.models[provider] || DEFAULT_MODELS[provider];
  const key = getApiKey(provider);
  if (!key) {
    requestOpenAiSettings();
    throw new MissingApiKeyError(provider);
  }

  if (provider === "gemini") {
    const ai = new GoogleGenAI({ apiKey: key });
    const config: any = {};
    if (opts.json) config.responseMimeType = "application/json";
    if (opts.schema) config.responseSchema = opts.schema;
    if (opts.temperature != null) config.temperature = opts.temperature;
    if (opts.system) config.systemInstruction = opts.system;
    const response = await ai.models.generateContent({
      model,
      contents: opts.prompt,
      ...(Object.keys(config).length ? { config } : {}),
    });
    return (response.text || "").trim();
  }

  if (provider === "openai") {
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
        ...(opts.json ? { response_format: { type: "json_object" } } : {}),
      }),
    });
    if (!res.ok) throw new Error(`OpenAI 오류 ${res.status}: ${await res.text()}`);
    const data = await res.json();
    return (data.choices?.[0]?.message?.content || "").trim();
  }

  // anthropic
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      ...(opts.system ? { system: opts.system } : {}),
      ...(opts.temperature != null ? { temperature: opts.temperature } : {}),
      messages: [
        {
          role: "user",
          content: opts.json
            ? `${opts.prompt}\n\n반드시 유효한 JSON만 출력하세요. 다른 텍스트 금지.`
            : opts.prompt,
        },
      ],
    }),
  });
  if (!res.ok) throw new Error(`Anthropic 오류 ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return (data.content?.[0]?.text || "").trim();
}
