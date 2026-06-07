import React, { useEffect, useMemo, useState } from "react";
import {
  AiProvider,
  AiSettings,
  DEFAULT_MODELS,
  PROVIDER_LABELS,
  getAiSettings,
  saveAiSettings,
} from "../utils/aiClient";

interface AiSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const PROVIDER_ORDER: AiProvider[] = ["gemini", "openai", "anthropic"];

const KEY_HELP: Record<AiProvider, { url: string; hint: string }> = {
  gemini: { url: "https://aistudio.google.com/apikey", hint: "Google AI Studio에서 무료 발급" },
  openai: { url: "https://platform.openai.com/api-keys", hint: "OpenAI Platform → API keys" },
  anthropic: { url: "https://console.anthropic.com/settings/keys", hint: "Anthropic Console → API Keys" },
};

export const AiSettingsModal: React.FC<AiSettingsModalProps> = ({ isOpen, onClose }) => {
  const [settings, setSettings] = useState<AiSettings>(() => getAiSettings());
  const [reveal, setReveal] = useState<Record<AiProvider, boolean>>({
    gemini: false,
    openai: false,
    anthropic: false,
  });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setSettings(getAiSettings());
      setSaved(false);
    }
  }, [isOpen]);

  const activeHasKey = useMemo(
    () => !!settings.keys[settings.provider]?.trim(),
    [settings]
  );

  if (!isOpen) return null;

  const update = (patch: Partial<AiSettings>) =>
    setSettings((s) => ({ ...s, ...patch }));

  const handleSave = () => {
    const trimmed: AiSettings = {
      provider: settings.provider,
      keys: {
        gemini: settings.keys.gemini.trim(),
        openai: settings.keys.openai.trim(),
        anthropic: settings.keys.anthropic.trim(),
      },
      models: {
        gemini: settings.models.gemini.trim() || DEFAULT_MODELS.gemini,
        openai: settings.models.openai.trim() || DEFAULT_MODELS.openai,
        anthropic: settings.models.anthropic.trim() || DEFAULT_MODELS.anthropic,
      },
    };
    saveAiSettings(trimmed);
    setSettings(trimmed);
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  };

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl bg-white dark:bg-gray-800 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 px-5 py-4">
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">AI API 키 설정</h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              키는 이 브라우저에만 저장되며 서버로 전송되지 않습니다.
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none"
            aria-label="닫기"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-5">
          {/* 활성 프로바이더 */}
          <div>
            <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              사용할 AI 제공자
            </label>
            <div className="flex gap-2">
              {PROVIDER_ORDER.map((p) => (
                <button
                  key={p}
                  onClick={() => update({ provider: p })}
                  className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                    settings.provider === p
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300"
                      : "border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  }`}
                >
                  {PROVIDER_LABELS[p]}
                </button>
              ))}
            </div>
          </div>

          {/* 키 입력 (프로바이더별) */}
          {PROVIDER_ORDER.map((p) => (
            <div key={p} className={settings.provider === p ? "" : "opacity-70"}>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                  {PROVIDER_LABELS[p]} API 키
                  {settings.provider === p && (
                    <span className="ml-1 text-blue-600 dark:text-blue-400">(활성)</span>
                  )}
                </label>
                <a
                  href={KEY_HELP[p].url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[11px] text-blue-600 dark:text-blue-400 hover:underline"
                >
                  {KEY_HELP[p].hint} ↗
                </a>
              </div>
              <div className="flex gap-2">
                <input
                  type={reveal[p] ? "text" : "password"}
                  value={settings.keys[p]}
                  onChange={(e) =>
                    update({ keys: { ...settings.keys, [p]: e.target.value } })
                  }
                  placeholder={`${PROVIDER_LABELS[p]} 키 입력`}
                  className="flex-1 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-xs text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:outline-none"
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  onClick={() => setReveal((r) => ({ ...r, [p]: !r[p] }))}
                  className="rounded-md border border-gray-300 dark:border-gray-600 px-2 text-xs text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                  type="button"
                >
                  {reveal[p] ? "숨김" : "표시"}
                </button>
              </div>
              <input
                type="text"
                value={settings.models[p]}
                onChange={(e) =>
                  update({ models: { ...settings.models, [p]: e.target.value } })
                }
                placeholder={`모델 (기본: ${DEFAULT_MODELS[p]})`}
                className="mt-1.5 w-full rounded-md border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-3 py-1.5 text-[11px] text-gray-700 dark:text-gray-300 focus:border-blue-500 focus:outline-none"
                spellCheck={false}
              />
            </div>
          ))}

          {!activeHasKey && (
            <p className="rounded-md bg-yellow-50 dark:bg-yellow-900/30 px-3 py-2 text-[11px] text-yellow-700 dark:text-yellow-300">
              활성 제공자({PROVIDER_LABELS[settings.provider]})의 키가 비어 있어 AI 기능이 비활성화됩니다.
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-gray-200 dark:border-gray-700 px-5 py-3">
          <button
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            닫기
          </button>
          <button
            onClick={handleSave}
            className="rounded-md bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-700"
          >
            {saved ? "저장됨 ✓" : "저장"}
          </button>
        </div>
      </div>
    </div>
  );
};
