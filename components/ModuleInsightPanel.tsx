import React, { useMemo, useState } from 'react';
import { CanvasModule, Connection } from '../types';
import { getModuleInsight } from '../utils/moduleInsights';
import { getModuleCode } from '../codeSnippets';
import { generateAiText, MissingApiKeyError } from '../utils/aiClient';

interface ModuleInsightPanelProps {
  module: CanvasModule;
  allModules?: CanvasModule[];
  allConnections?: Connection[];
}

/**
 * View Details 공통 인사이트 패널.
 * (1) 핵심 요약 + 계리 해석 + 다음 단계  (2) 생성 파이썬 코드 보기
 * (3) AI 결과 해석  — 모든 모듈의 View Details 상단에 일관되게 삽입한다.
 */
export const ModuleInsightPanel: React.FC<ModuleInsightPanelProps> = ({
  module,
  allModules,
  allConnections,
}) => {
  const insight = useMemo(() => getModuleInsight(module), [module]);
  const code = useMemo(() => {
    try {
      return getModuleCode(module, allModules, allConnections);
    } catch {
      return '# 코드를 생성할 수 없습니다.';
    }
  }, [module, allModules, allConnections]);

  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiText, setAiText] = useState('');
  const [aiError, setAiError] = useState<string | null>(null);

  if (!insight) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* noop */
    }
  };

  const handleAi = async () => {
    setAiBusy(true);
    setAiError(null);
    setAiText('');
    const metricsText = insight.metrics.map((m) => `${m.label}: ${m.value}`).join(', ');
    const prompt =
      `보험·계리 파이프라인의 '${module.name}'(${module.type}) 모듈 실행 결과를 계리 관점에서 한국어로 해석해 주세요.\n` +
      `핵심 지표: ${metricsText || '(표 참조)'}\n` +
      `1) 이 수치가 의미하는 바, 2) 리스크/프라이싱 시사점, 3) 점검 포인트 1~2개를 간결히.`;
    try {
      const text = await generateAiText({
        prompt,
        system: '당신은 보험·계리(XoL 재보험, DFA) 전문가입니다. 간결하고 정확한 한국어로 답합니다.',
        temperature: 0.3,
      });
      setAiText(text || '(응답 없음)');
    } catch (e: any) {
      setAiError(
        e instanceof MissingApiKeyError
          ? 'AI API 키가 없습니다. 우측 상단 🔑 설정에서 키를 입력하세요.'
          : e?.message || String(e)
      );
    } finally {
      setAiBusy(false);
    }
  };

  return (
    <div className="mb-4 rounded-lg border border-blue-200 dark:border-blue-900 bg-blue-50/60 dark:bg-blue-950/30 p-3">
      {/* 제목 + 액션 */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <h4 className="text-sm font-bold text-blue-900 dark:text-blue-200">📌 {insight.title}</h4>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowCode((v) => !v)}
            className="px-2 py-1 text-[11px] font-medium rounded-md border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-white dark:hover:bg-gray-800"
            title="이 결과를 만든 파이썬 코드"
          >
            🐍 코드
          </button>
          <button
            onClick={handleAi}
            disabled={aiBusy}
            className="px-2 py-1 text-[11px] font-medium rounded-md bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white"
            title="AI로 이 결과 해석"
          >
            {aiBusy ? '...' : '🤖 AI 해석'}
          </button>
        </div>
      </div>

      {/* 지표(좌) + 해석·다음단계(우) 가로 배치로 세로 공간 축소 */}
      <div className="flex gap-3 items-start">
        {insight.metrics.length > 0 && (
          <div className="flex flex-col gap-1 flex-shrink-0">
            {insight.metrics.map((m, i) => (
              <div
                key={i}
                className="flex items-center justify-between gap-2 rounded-md bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 px-2 py-0.5 min-w-[120px]"
              >
                <span className="text-[10px] text-gray-500 dark:text-gray-400">{m.label}</span>
                <span className="text-xs font-semibold text-gray-900 dark:text-gray-100">{m.value}</span>
              </div>
            ))}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs text-gray-700 dark:text-gray-300 leading-relaxed">{insight.interpretation}</p>
          <p className="text-xs text-blue-700 dark:text-blue-300 mt-1">
            <span className="font-semibold">다음 단계 →</span> {insight.nextSteps}
          </p>
        </div>
      </div>

      {/* 생성 파이썬 코드 (접이식) */}
      {showCode && (
        <div className="mt-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-medium text-gray-600 dark:text-gray-400">생성 파이썬 코드 (모듈↔코드 1:1)</span>
            <button
              onClick={handleCopy}
              className="text-[11px] text-blue-600 dark:text-blue-400 hover:underline"
            >
              {copied ? '복사됨 ✓' : '복사'}
            </button>
          </div>
          <pre className="max-h-56 overflow-auto rounded-md bg-gray-900 p-2 text-[11px] font-mono text-gray-100 whitespace-pre-wrap">
            <code>{code}</code>
          </pre>
        </div>
      )}

      {/* AI 해석 결과 */}
      {(aiText || aiError) && (
        <div className="mt-2 rounded-md bg-white dark:bg-gray-800 border border-purple-200 dark:border-purple-900 p-2">
          <span className="text-[11px] font-semibold text-purple-700 dark:text-purple-300">🤖 AI 해석</span>
          {aiText && (
            <pre className="mt-1 whitespace-pre-wrap text-xs text-gray-800 dark:text-gray-200 font-sans">{aiText}</pre>
          )}
          {aiError && <p className="mt-1 text-xs text-red-500">{aiError}</p>}
        </div>
      )}
    </div>
  );
};
