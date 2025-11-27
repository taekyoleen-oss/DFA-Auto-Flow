import React, { useState } from 'react';
import { CanvasModule, TrainedModelOutput, ModuleType } from '../types';
import { XCircleIcon, SparklesIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';

interface TrainedModelPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

const ModalModelMetrics: React.FC<{ metrics: TrainedModelOutput['metrics'] }> = ({ metrics }) => (
    <div>
        <h4 className="text-md font-semibold text-gray-700 mb-2">Evaluation Metrics</h4>
        <div className="bg-gray-50 rounded-lg p-3 space-y-2 border border-gray-200">
            {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-sm">
                    <span className="text-gray-600">{key}:</span>
                    <span className="font-mono text-gray-800 font-medium">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                </div>
            ))}
        </div>
    </div>
);

// fix: Corrected ModuleType enums to be more specific (e.g., DecisionTreeClassifier) to match the type definitions.
const complexModels = [
    ModuleType.DecisionTree,
    ModuleType.RandomForest,
    ModuleType.SVM,
    ModuleType.KNN,
    ModuleType.NaiveBayes,
    ModuleType.LinearDiscriminantAnalysis
];

export const TrainedModelPreviewModal: React.FC<TrainedModelPreviewModalProps> = ({ module, projectName, onClose }) => {
    const [isInterpreting, setIsInterpreting] = useState(false);
    const [aiInterpretation, setAiInterpretation] = useState<string | null>(null);

    const output = module.outputData as TrainedModelOutput;
    if (!output || output.type !== 'TrainedModelOutput') return null;

    const { modelType, coefficients, intercept, metrics, featureColumns, labelColumn } = output;

    const handleInterpret = async () => {
        setIsInterpreting(true);
        setAiInterpretation(null);
        try {
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

            const metricsText = Object.entries(metrics).map(([key, value]) => `- ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`).join('\n');
            const topFeatures = Object.entries(coefficients)
                .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                .slice(0, 3)
                .map(([key, value]) => `- ${key}: ${value.toFixed(4)}`)
                .join('\n');

            const prompt = `
You are a data scientist writing a brief report for a non-technical audience. Please use Korean and simple Markdown.

### 머신러닝 모델 분석 보고서

**프로젝트:** ${projectName}
**모델:** ${modelType}
**분석 대상:** ${labelColumn}

**성능 (학습 데이터 기준):**
${metricsText}

**주요 영향 변수 (상위 3개):**
${topFeatures}

---

**1. 모델 성능 요약:**
- 이 모델의 성능을 주요 지표(예: Accuracy, R-squared)를 사용하여 한 문장으로 요약해 주십시오.

**2. 핵심 발견:**
- 분석 대상에 가장 큰 영향을 미치는 변수는 무엇이며, 이 변수가 결과에 긍정적인 영향을 미칩니까, 부정적인 영향을 미치습니까?

**3. 모델 활용 방안:**
- 이 모델을 비즈니스에 어떻게 활용할 수 있을지 간단한 아이디어 한 가지를 제안해 주십시오.

**지시:** 각 항목을 한두 문장으로 매우 간결하게 작성하십시오.
`;
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
            });

            setAiInterpretation(response.text);
        } catch (error) {
            console.error("AI interpretation failed:", error);
            setAiInterpretation("결과를 해석하는 동안 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.");
        } finally {
            setIsInterpreting(false);
        }
    };
    
    let formulaParts: string[] = [];
    if (!complexModels.includes(modelType)) {
        if (modelType === ModuleType.LogisticRegression) {
            formulaParts = [`ln(p / (1 - p)) = ${intercept.toFixed(4)}`];
        } else {
            formulaParts = [`${labelColumn} ≈ ${intercept.toFixed(4)}`];
        }

        featureColumns.forEach(feature => {
            const value = coefficients[feature];
            const coeff = typeof value === 'number' ? value : 0;
            if (coeff >= 0) {
                formulaParts.push(` + ${coeff.toFixed(4)} * [${feature}]`);
            } else {
                formulaParts.push(` - ${Math.abs(coeff).toFixed(4)} * [${feature}]`);
            }
        });
    }


    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Trained Model Details: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                <main className="flex-grow p-6 overflow-auto space-y-6">
                    <div className="flex justify-end font-sans">
                        <button
                            onClick={handleInterpret}
                            disabled={isInterpreting}
                            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:bg-purple-400 disabled:cursor-wait transition-colors"
                        >
                            <SparklesIcon className="w-5 h-5" />
                            {isInterpreting ? '분석 중...' : 'AI로 결과 해석하기'}
                        </button>
                    </div>

                    {isInterpreting && (
                        <div className="text-center p-8 text-gray-600 font-sans">
                            <p>AI가 모델 결과를 분석하고 있습니다...</p>
                        </div>
                    )}
                    {aiInterpretation && (
                         <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                            <h3 className="text-lg font-bold text-purple-800 mb-2 font-sans flex items-center gap-2">
                                <SparklesIcon className="w-5 h-5"/>
                                AI 분석 요약
                            </h3>
                            <MarkdownRenderer text={aiInterpretation} />
                        </div>
                    )}

                    <div>
                        <h4 className="text-md font-semibold text-gray-700 mb-2">Model Equation</h4>
                        <div className="bg-gray-50 rounded-lg p-3 font-mono text-xs text-green-700 whitespace-normal break-words border border-gray-200">
                            {formulaParts.length > 0 ? (
                                <>
                                    <span>{formulaParts[0]}</span>
                                    {formulaParts.slice(1).map((part, i) => <span key={i}>{part}</span>)}
                                </>
                            ) : (
                                <span className="text-gray-500 font-sans text-sm">Model equation is not available for this model type.</span>
                            )}
                        </div>
                    </div>
                    
                    <ModalModelMetrics metrics={metrics} />

                    <div>
                        <h4 className="text-md font-semibold text-gray-700 mb-2">Coefficients / Feature Importances</h4>
                        <div className="bg-gray-50 rounded-lg border border-gray-200 max-h-60 overflow-y-auto">
                            <table className="w-full text-sm">
                                <thead className="sticky top-0 bg-gray-100">
                                    <tr className="text-left">
                                        <th className="p-2 font-semibold">Feature</th>
                                        <th className="p-2 font-semibold text-right">Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr className="border-t">
                                        <td className="p-2 font-mono text-gray-600">(Intercept)</td>
                                        <td className="p-2 font-mono text-right">{intercept.toFixed(4)}</td>
                                    </tr>
                                    {Object.entries(coefficients).map(([feature, value]) => (
                                        <tr key={feature} className="border-t">
                                            <td className="p-2 font-mono">{feature}</td>
                                            <td className="p-2 font-mono text-right">{value.toFixed(4)}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
};