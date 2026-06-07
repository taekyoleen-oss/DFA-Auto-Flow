import React, { useState, useMemo } from 'react';
import { CodeBracketIcon, ClipboardIcon, CheckIcon } from './icons';
import { CanvasModule, Connection } from '../types';
import { generateFullPipelineCode } from '../utils/generatePipelineCode';
import { generateAiText, MissingApiKeyError } from '../utils/aiClient';
import { useTheme } from '../contexts/ThemeContext';

interface PipelineCodePanelProps {
    modules: CanvasModule[];
    connections: Connection[];
    isVisible: boolean;
    onToggle: () => void;
}

const TIPS = [
    {
        icon: '▶',
        title: '실행 전 준비',
        desc: '캔버스에서 각 모듈을 먼저 실행해주세요. LoadData 모듈이 실행되면 데이터가 코드에 자동 주입되어 Pyodide 환경에서도 실행됩니다.',
    },
    {
        icon: '💻',
        title: 'Jupyter / 스크립트 실행',
        desc: '복사 버튼으로 코드를 복사하여 Jupyter Notebook이나 .py 파일에 붙여넣으면 동일한 결과를 얻을 수 있습니다.',
    },
    {
        icon: '⚡',
        title: '첫 실행 시 초기화',
        desc: 'Python 환경(Pyodide + pandas, scikit-learn)을 처음 로드할 때 30~60초 소요됩니다. 이후 실행은 즉시 처리됩니다.',
    },
    {
        icon: '🔄',
        title: '단계별 실행',
        desc: '전체 파이프라인 대신 각 모듈의 ▶ 버튼을 클릭하면 해당 모듈만 단계적으로 실행하고 결과를 바로 확인할 수 있습니다.',
    },
    {
        icon: '📋',
        title: '코드 구조',
        desc: '각 [모듈 N/전체] 섹션이 하나의 처리 단계입니다. 섹션 사이의 data_xxx 변수가 모듈 간 데이터를 연결합니다.',
    },
];

export const PipelineCodePanel: React.FC<PipelineCodePanelProps> = ({
    modules,
    connections,
    isVisible,
    onToggle
}) => {
    const { theme } = useTheme();
    const [copied, setCopied] = useState(false);
    const [isRunning, setIsRunning] = useState(false);
    const [output, setOutput] = useState('');
    const [outputError, setOutputError] = useState<string | null>(null);
    const [showOutput, setShowOutput] = useState(false);
    const [showTips, setShowTips] = useState(false);
    // AI 코드 설명·에러 해결
    const [aiBusy, setAiBusy] = useState(false);
    const [aiResult, setAiResult] = useState('');
    const [aiError, setAiError] = useState<string | null>(null);

    // 표시용 코드 (pd.read_csv 포함 - 외부 실행 가능한 형태)
    const fullPipelineCode = useMemo(() => {
        if (modules.length === 0) {
            return '# 파이프라인이 비어있습니다. 모듈을 추가해주세요.';
        }
        try {
            return generateFullPipelineCode(modules, connections, false);
        } catch (e: any) {
            return `# 코드 생성 중 오류가 발생했습니다.\n# ${e?.message || String(e)}`;
        }
    }, [modules, connections]);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(fullPipelineCode);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy code:', err);
        }
    };

    const handleRun = async () => {
        setIsRunning(true);
        setOutput('');
        setOutputError(null);
        setShowOutput(true);
        try {
            // 실행용 코드: LoadData에 outputData 주입 (Pyodide 파일 접근 불가 대응)
            const executionCode = generateFullPipelineCode(modules, connections, true);
            const { runPythonWithOutput } = await import('../utils/pyodideRunner');
            const { stdout, error } = await runPythonWithOutput(executionCode);
            setOutput(stdout || '(출력 없음)');
            setOutputError(error);
        } catch (e: any) {
            setOutputError(e.message);
        } finally {
            setIsRunning(false);
        }
    };

    // 현재 컨텍스트에 따른 AI 모드: 에러 해결 / 결과 해석 / 코드 설명
    const aiMode: 'error' | 'interpret' | 'explain' = outputError
        ? 'error'
        : (output && output !== '(출력 없음)') ? 'interpret' : 'explain';
    const aiLabel = aiMode === 'error' ? '🤖 에러해결' : aiMode === 'interpret' ? '🤖 결과해석' : '🤖 설명';

    // AI: 컨텍스트 인지형 — 에러 해결 / 결과 해석(계리 관점) / 코드 설명 (활성 프로바이더)
    const handleAiExplain = async () => {
        setAiBusy(true);
        setAiError(null);
        setAiResult('');
        const system =
            '당신은 보험·계리(XoL 재보험 프라이싱, DFA) 도메인의 파이썬 데이터 분석 전문가입니다. ' +
            '한국어로 간결하고 정확하게 답합니다. 코드는 pandas/numpy/scikit-learn/statsmodels 기반입니다.';
        let prompt: string;
        if (aiMode === 'error') {
            prompt =
                `다음 파이썬 파이프라인 코드를 실행하니 에러가 발생했습니다.\n` +
                `에러를 한국어로 설명하고, 원인과 수정 방법을 제시한 뒤, 수정된 코드 조각을 보여주세요.\n\n` +
                `### 에러\n${outputError}\n\n### 코드\n\`\`\`python\n${fullPipelineCode}\n\`\`\``;
        } else if (aiMode === 'interpret') {
            prompt =
                `다음은 보험·계리 파이프라인의 실제 실행 결과입니다. 계리 관점에서 한국어로 해석해 주세요:\n` +
                `1) 핵심 수치(분포 적합 결과, VaR/TVaR, 시뮬레이션 통계 등)가 의미하는 바\n` +
                `2) 리스크/프라이싱 관점의 시사점\n` +
                `3) 결과의 타당성 점검 포인트나 주의사항 1~3개\n\n` +
                `### 실행 결과\n\`\`\`\n${output.slice(0, 6000)}\n\`\`\``;
        } else {
            prompt =
                `다음 파이썬 파이프라인 코드가 무엇을 하는지 단계별로 한국어로 설명하고, ` +
                `계리 관점에서 주의할 점이나 개선 아이디어를 1~3개 제시하세요.\n\n` +
                `\`\`\`python\n${fullPipelineCode}\n\`\`\``;
        }
        try {
            const text = await generateAiText({ prompt, system, temperature: 0.3 });
            setAiResult(text || '(응답 없음)');
        } catch (e: any) {
            if (e instanceof MissingApiKeyError) {
                setAiError('AI API 키가 설정되지 않았습니다. 우측 상단 🔑 설정에서 키를 입력하세요.');
            } else {
                setAiError(e?.message || String(e));
            }
        } finally {
            setAiBusy(false);
        }
    };

    const hasUnrunLoadData = modules.some(
        (m) => m.type === 'LoadData' && !m.outputData
    );

    return (
        <div
            className={`absolute top-0 right-0 h-full bg-white dark:bg-gray-800 border-l border-gray-300 dark:border-gray-700 z-10 transition-transform duration-300 ease-in-out flex flex-col ${
                isVisible ? 'translate-x-0' : 'translate-x-full'
            }`}
            style={{ width: '420px' }}
        >
            {/* 헤더 */}
            <div className="flex items-center justify-between p-3 border-b border-gray-300 dark:border-gray-700 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <CodeBracketIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <h3 className="text-sm font-bold text-gray-900 dark:text-white">전체 파이프라인 코드</h3>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setShowTips((v) => !v)}
                        className={`p-1.5 rounded-md transition-colors text-xs font-medium ${
                            showTips
                                ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                                : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400'
                        }`}
                        title="사용 안내"
                    >
                        ?
                    </button>
                    <button
                        onClick={handleCopy}
                        className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
                        title="코드 복사 (외부 Python 실행용)"
                    >
                        {copied ? (
                            <CheckIcon className="w-4 h-4 text-green-600 dark:text-green-400" />
                        ) : (
                            <ClipboardIcon className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                        )}
                    </button>
                    <button
                        onClick={handleRun}
                        disabled={isRunning || modules.length === 0}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white text-xs font-medium rounded-md transition-colors"
                        title="Pyodide로 실행 (브라우저 내 Python)"
                    >
                        {isRunning ? (
                            <>
                                <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                실행 중...
                            </>
                        ) : (
                            <>▶ 실행</>
                        )}
                    </button>
                    <button
                        onClick={handleAiExplain}
                        disabled={aiBusy || modules.length === 0}
                        className="flex items-center gap-1.5 px-2.5 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white text-xs font-medium rounded-md transition-colors"
                        title={aiMode === 'error' ? 'AI로 에러 해결안 받기' : aiMode === 'interpret' ? 'AI로 실행 결과 해석받기' : 'AI로 코드 설명 받기'}
                    >
                        {aiBusy ? '...' : aiLabel}
                    </button>
                </div>
            </div>

            {/* 사용 안내 패널 */}
            {showTips && (
                <div className="flex-shrink-0 border-b border-gray-300 dark:border-gray-700 bg-blue-50 dark:bg-blue-950/40 p-3 overflow-y-auto max-h-56">
                    <p className="text-xs font-bold text-blue-700 dark:text-blue-300 mb-2">사용 안내</p>
                    <div className="space-y-2">
                        {TIPS.map((tip, i) => (
                            <div key={i} className="flex gap-2">
                                <span className="text-xs flex-shrink-0 w-4">{tip.icon}</span>
                                <div>
                                    <span className="text-xs font-semibold text-gray-800 dark:text-gray-200">{tip.title}: </span>
                                    <span className="text-xs text-gray-600 dark:text-gray-400">{tip.desc}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* LoadData 미실행 경고 */}
            {hasUnrunLoadData && (
                <div className="flex-shrink-0 px-3 py-2 bg-yellow-50 dark:bg-yellow-900/30 border-b border-yellow-200 dark:border-yellow-700">
                    <p className="text-xs text-yellow-700 dark:text-yellow-300">
                        ⚠️ LoadData 모듈을 먼저 실행해야 Pyodide에서 데이터를 읽을 수 있습니다.
                    </p>
                </div>
            )}

            {/* 코드 영역 */}
            <div className="flex-1 overflow-auto p-3 min-h-0">
                <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded-md overflow-x-auto text-xs font-mono text-gray-900 dark:text-gray-200 whitespace-pre-wrap">
                    <code>{fullPipelineCode}</code>
                </pre>
            </div>

            {/* 출력 영역 */}
            {showOutput && (
                <div className="flex-shrink-0 border-t border-gray-300 dark:border-gray-700">
                    <div className="flex items-center justify-between px-3 py-1.5 bg-gray-100 dark:bg-gray-700">
                        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">실행 결과</span>
                        <button
                            onClick={() => setShowOutput(false)}
                            className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                        >
                            지우기
                        </button>
                    </div>
                    <div className="h-48 overflow-auto bg-gray-950 p-3 font-mono text-xs">
                        {isRunning && (
                            <p className="text-yellow-400">실행 중... (처음 실행 시 Python 환경 초기화로 30~60초 소요)</p>
                        )}
                        {output && (
                            <pre className="text-green-400 whitespace-pre-wrap">{output}</pre>
                        )}
                        {outputError && (
                            <pre className="text-red-400 whitespace-pre-wrap mt-2">{outputError}</pre>
                        )}
                    </div>
                </div>
            )}

            {/* AI 설명 / 에러 해결 결과 */}
            {(aiBusy || aiResult || aiError) && (
                <div className="flex-shrink-0 border-t border-purple-300 dark:border-purple-700">
                    <div className="flex items-center justify-between px-3 py-1.5 bg-purple-100 dark:bg-purple-900/40">
                        <span className="text-xs font-medium text-purple-700 dark:text-purple-300">🤖 AI 설명 / 결과 해석 / 에러 해결</span>
                        <button
                            onClick={() => { setAiResult(''); setAiError(null); }}
                            className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                        >
                            지우기
                        </button>
                    </div>
                    <div className="max-h-64 overflow-auto bg-gray-50 dark:bg-gray-900 p-3 text-xs">
                        {aiBusy && <p className="text-purple-500">AI가 분석 중입니다...</p>}
                        {aiResult && (
                            <pre className="whitespace-pre-wrap text-gray-800 dark:text-gray-200 font-sans">{aiResult}</pre>
                        )}
                        {aiError && <p className="text-red-500">{aiError}</p>}
                    </div>
                </div>
            )}
        </div>
    );
};
