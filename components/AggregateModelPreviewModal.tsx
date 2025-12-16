import React, { useState, useMemo, useRef } from 'react';
import { CanvasModule, AggregateModelOutput, AggregateModelFitResult } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { useCopyOnCtrlC } from '../hooks/useCopyOnCtrlC';
import { SpreadViewModal } from './SpreadViewModal';

interface AggregateModelPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
  onSelectDistribution?: (moduleId: string, distributionType: string) => void;
}

export const AggregateModelPreviewModal: React.FC<AggregateModelPreviewModalProps> = ({ 
  module, 
  onClose,
  onSelectDistribution 
}) => {
  const output = module.outputData as AggregateModelOutput;
  if (!output || output.type !== 'AggregateModelOutput') {
    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <div 
          className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-md p-6"
          onClick={e => e.stopPropagation()}
        >
          <h2 className="text-xl font-bold text-gray-800 mb-4">Error</h2>
          <p className="text-gray-600">No valid output data available.</p>
          <button 
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  // 기존 구조와 새 구조 모두 지원
  let results: AggregateModelFitResult[] = [];
  let selectedDistribution: string | undefined;
  let yearlyAggregates: Array<{ year: number; totalAmount: number }> = [];

  if (output.results && Array.isArray(output.results) && output.results.length > 0) {
    // 새 구조: results 배열
    results = output.results;
    selectedDistribution = output.selectedDistribution;
    yearlyAggregates = output.yearlyAggregates || [];
  } else if ((output as any).distributionType) {
    // 기존 구조: 단일 분포 (하위 호환성)
    results = [{
      distributionType: (output as any).distributionType,
      parameters: (output as any).parameters || {},
      fitStatistics: (output as any).fitStatistics || {},
    }];
    selectedDistribution = (output as any).distributionType;
    yearlyAggregates = (output as any).yearlyAggregates || [];
  }

  if (results.length === 0) {
    return (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <div 
          className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-md p-6"
          onClick={e => e.stopPropagation()}
        >
          <h2 className="text-xl font-bold text-gray-800 mb-4">No Results</h2>
          <p className="text-gray-600">No distribution fit results available.</p>
          <button 
            onClick={onClose}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  const [localSelected, setLocalSelected] = useState<string>(() => {
    if (selectedDistribution) return selectedDistribution;
    if (results.length > 0 && results[0]?.distributionType) {
      return results[0].distributionType;
    }
    return '';
  });

  const handleSelectDistribution = (distType: string) => {
    setLocalSelected(distType);
    if (onSelectDistribution) {
      onSelectDistribution(module.id, distType);
    }
  };

  const viewDetailsRef = useRef<HTMLDivElement>(null);
  useCopyOnCtrlC(viewDetailsRef);

  // AIC 기준으로 정렬 (낮을수록 좋음)
  const sortedResults = [...results].sort((a, b) => {
    const aicA = a.fitStatistics?.aic ?? Infinity;
    const aicB = b.fitStatistics?.aic ?? Infinity;
    return aicA - aicB;
  });

  const recommended = sortedResults.length > 0 ? sortedResults[0] : null;
  const [showSpreadView, setShowSpreadView] = useState(false);

  // 통계 계산 (yearlyAggregates의 totalAmount 값들 사용)
  const statistics = useMemo(() => {
    if (!yearlyAggregates || yearlyAggregates.length === 0) {
      return null;
    }

    const amounts = yearlyAggregates.map(item => item.totalAmount).filter(a => typeof a === 'number' && !isNaN(a));
    if (amounts.length === 0) return null;

    amounts.sort((a, b) => a - b);
    const n = amounts.length;
    const sum = amounts.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const variance = amounts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);
    
    // 왜도 (skewness)
    const skewness = stdDev > 0 
      ? amounts.reduce((s, val) => s + Math.pow(val - mean, 3), 0) / (n * Math.pow(stdDev, 3))
      : 0;
    
    // 첨도 (kurtosis) - excess kurtosis
    const kurtosis = stdDev > 0
      ? amounts.reduce((s, val) => s + Math.pow(val - mean, 4), 0) / (n * Math.pow(stdDev, 4)) - 3
      : 0;

    // 백분위수 계산
    const getQuantile = (q: number) => {
      const pos = (n - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      return amounts[base + 1] !== undefined 
        ? amounts[base] + rest * (amounts[base + 1] - amounts[base])
        : amounts[base];
    };

    return {
      mean,
      stdDev,
      skewness,
      kurtosis,
      min: amounts[0],
      max: amounts[n - 1],
      percentile25: getQuantile(0.25),
      percentile50: getQuantile(0.5), // median
      percentile75: getQuantile(0.75),
      percentile90: getQuantile(0.90),
      percentile95: getQuantile(0.95),
      percentile99: getQuantile(0.99),
      count: n,
    };
  }, [yearlyAggregates]);

  // yearlyAggregates를 Spread View용 데이터로 변환
  const spreadViewData = useMemo(() => {
    if (!yearlyAggregates || yearlyAggregates.length === 0) return [];
    return yearlyAggregates.map(agg => ({
      year: agg.year,
      total_amount: agg.totalAmount,
    }));
  }, [yearlyAggregates]);

  const spreadViewColumns = [
    { name: 'year', type: 'number' },
    { name: 'total_amount', type: 'number' },
  ];

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">Model Details: {module.name}</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSpreadView(true)}
              className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              Spread View
            </button>
            <button
              onClick={() => {
                if (!yearlyAggregates || yearlyAggregates.length === 0) return;
                const csvContent = [
                  'year,total_amount',
                  ...yearlyAggregates.map(agg => `${agg.year},${agg.totalAmount}`)
                ].join('\n');
                // BOM 추가하여 한글 인코딩 문제 해결
                const bom = '\uFEFF';
                const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `${module.name}_yearly_aggregates.csv`;
                link.click();
              }}
              className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-1"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download CSV
            </button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
              <XCircleIcon className="w-6 h-6" />
            </button>
          </div>
        </header>
        <main className="flex-grow p-6 overflow-auto">
          <div className="space-y-6">
            {/* Data Statistics */}
            {statistics && (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Data Statistics (데이터 통계)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Count</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.count}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Mean</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.mean.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Std Dev</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.stdDev.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Skewness</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.skewness.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Kurtosis</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.kurtosis.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Min</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.min.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">25th Percentile</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile25.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">50th Percentile (Median)</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile50.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">75th Percentile</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile75.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">90th Percentile</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile90.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">95th Percentile</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile95.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">99th Percentile</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile99.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs text-gray-500 mb-1">Max</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.max.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* All Distributions Comparison Table */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Distribution Comparison</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left font-semibold text-gray-700">Distribution</th>
                      <th className="px-4 py-2 text-right font-semibold text-gray-700">AIC</th>
                      <th className="px-4 py-2 text-right font-semibold text-gray-700">BIC</th>
                      <th className="px-4 py-2 text-right font-semibold text-gray-700">Log Likelihood</th>
                      <th className="px-4 py-2 text-center font-semibold text-gray-700">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedResults.map((result, index) => {
                      const isSelected = localSelected === result.distributionType;
                      const isRecommended = result.distributionType === recommended?.distributionType && !(result as any).error;
                      const hasError = !!(result as any).error;
                      return (
                        <tr 
                          key={result.distributionType} 
                          className={`border-b border-gray-100 ${
                            hasError ? 'bg-red-50' : isSelected ? 'bg-blue-50' : isRecommended ? 'bg-indigo-50' : ''
                          }`}
                        >
                          <td className="px-4 py-2">
                            <div className="flex items-center gap-2">
                              <span className={`font-semibold ${hasError ? 'text-red-600' : 'text-gray-800'}`}>
                                {result.distributionType}
                              </span>
                              {hasError && (
                                <span className="px-2 py-0.5 text-xs bg-red-600 text-white rounded">Failed</span>
                              )}
                              {!hasError && isRecommended && (
                                <span className="px-2 py-0.5 text-xs bg-indigo-600 text-white rounded">Recommended</span>
                              )}
                              {!hasError && isSelected && (
                                <span className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded">Selected</span>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-gray-800">
                            {hasError ? (
                              <span className="text-red-500 text-xs">Error</span>
                            ) : result.fitStatistics?.aic !== undefined && result.fitStatistics.aic !== null
                              ? result.fitStatistics.aic.toFixed(4)
                              : 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-gray-800">
                            {hasError ? (
                              <span className="text-red-500 text-xs">Error</span>
                            ) : result.fitStatistics?.bic !== undefined && result.fitStatistics.bic !== null
                              ? result.fitStatistics.bic.toFixed(4)
                              : 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-gray-800">
                            {hasError ? (
                              <span className="text-red-500 text-xs">Error</span>
                            ) : result.fitStatistics?.logLikelihood !== undefined && result.fitStatistics.logLikelihood !== null
                              ? result.fitStatistics.logLikelihood.toFixed(4)
                              : 'N/A'}
                          </td>
                          <td className="px-4 py-2 text-center">
                            {hasError ? (
                              <span className="text-xs text-red-500">Cannot Select</span>
                            ) : (
                              <button
                                onClick={() => handleSelectDistribution(result.distributionType)}
                                className={`px-3 py-1 rounded text-xs font-semibold ${
                                  isSelected
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                }`}
                              >
                                {isSelected ? 'Selected' : 'Select'}
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* All Distributions Parameters */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Distribution Parameters</h3>
              <div className="space-y-4">
                {sortedResults.map((result) => {
                  const isSelected = localSelected === result.distributionType;
                  const hasError = !!(result as any).error;
                  return (
                    <div 
                      key={result.distributionType}
                      className={`border rounded-lg p-4 ${hasError ? 'border-red-500 bg-red-50' : isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <h4 className={`font-semibold ${hasError ? 'text-red-600' : 'text-gray-800'}`}>
                          {result.distributionType}
                        </h4>
                        <div className="flex items-center gap-2">
                          {hasError && (
                            <span className="px-2 py-1 text-xs bg-red-600 text-white rounded">Failed</span>
                          )}
                          {!hasError && isSelected && (
                            <span className="px-2 py-1 text-xs bg-blue-600 text-white rounded">Selected</span>
                          )}
                        </div>
                      </div>
                      {hasError ? (
                        <div className="text-sm text-red-600">
                          <p className="font-semibold mb-1">Error:</p>
                          <p className="font-mono text-xs">{(result as any).error}</p>
                        </div>
                      ) : (
                        <>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                        {result.parameters && Object.entries(result.parameters).map(([key, value]) => (
                          <div key={key} className="flex flex-col">
                            <span className="text-xs text-gray-500 mb-1">{key}</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {typeof value === 'number' ? value.toFixed(6) : String(value || 'N/A')}
                            </span>
                          </div>
                        ))}
                      </div>
                      {/* Fit Statistics */}
                      {result.fitStatistics && (
                        <div className="border-t pt-3 mt-3">
                          <h5 className="text-sm font-semibold text-gray-700 mb-2">Fit Statistics</h5>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {result.fitStatistics.aic !== undefined && result.fitStatistics.aic !== null && (
                              <div className="flex flex-col">
                                <span className="text-xs text-gray-500 mb-1">AIC</span>
                                <span className="font-mono text-sm text-gray-800 font-semibold">
                                  {result.fitStatistics.aic.toFixed(4)}
                                </span>
                              </div>
                            )}
                            {result.fitStatistics.bic !== undefined && result.fitStatistics.bic !== null && (
                              <div className="flex flex-col">
                                <span className="text-xs text-gray-500 mb-1">BIC</span>
                                <span className="font-mono text-sm text-gray-800 font-semibold">
                                  {result.fitStatistics.bic.toFixed(4)}
                                </span>
                              </div>
                            )}
                            {result.fitStatistics.logLikelihood !== undefined && result.fitStatistics.logLikelihood !== null && (
                              <div className="flex flex-col">
                                <span className="text-xs text-gray-500 mb-1">Log Likelihood</span>
                                <span className="font-mono text-sm text-gray-800 font-semibold">
                                  {result.fitStatistics.logLikelihood.toFixed(4)}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Yearly Aggregates */}
            {yearlyAggregates && yearlyAggregates.length > 0 && (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Yearly Aggregates</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left font-semibold text-gray-700">Year</th>
                        <th className="px-4 py-2 text-right font-semibold text-gray-700">Total Amount</th>
                      </tr>
                    </thead>
                    <tbody>
                      {yearlyAggregates.map((agg, index) => (
                        <tr key={index} className="border-b border-gray-100">
                          <td className="px-4 py-2 text-gray-600">{agg?.year ?? 'N/A'}</td>
                          <td className="px-4 py-2 text-right font-mono text-gray-800">
                            {agg?.totalAmount !== undefined && agg.totalAmount !== null 
                              ? agg.totalAmount.toLocaleString('ko-KR', { maximumFractionDigits: 2 })
                              : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* View Details Section */}
            {sortedResults.length > 0 && sortedResults[0] && !(sortedResults[0] as any).error && (
              <div className="bg-white border border-gray-200 rounded-lg p-4" ref={viewDetailsRef}>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">View Details (Ctrl+C to copy)</h3>
                <div className="space-y-4">
                  {/* Cumulative Distribution Chart (누적 막대그래프 + 각 분포 선 그래프) */}
                  {(() => {
                    // 실제 데이터의 누적 분포 (첫 번째 결과에서 가져옴)
                    const firstResult = sortedResults.find(r => r.cumulativeDistribution);
                    const cumulativeData = firstResult?.cumulativeDistribution;

                    // 각 분포의 이론적 누적 확률별 금액
                    const distributionLines = sortedResults
                      .filter(r => r.theoreticalCumulative && !r.error)
                      .map(r => ({
                        distributionType: r.distributionType,
                        data: r.theoreticalCumulative!,
                        color: r.distributionType === "Lognormal" ? "#3b82f6" :
                               r.distributionType === "Exponential" ? "#10b981" :
                               r.distributionType === "Gamma" ? "#f59e0b" :
                               r.distributionType === "Pareto" ? "#ef4444" : "#6b7280"
                      }));

                    if (!cumulativeData) {
                      return null;
                    }

                    const CumulativeChart: React.FC = () => {
                      const margin = { top: 20, right: 150, bottom: 60, left: 80 };
                      const width = 800;
                      const height = 400;
                      const chartWidth = width - margin.left - margin.right;
                      const chartHeight = height - margin.top - margin.bottom;

                      // 실제 데이터에서 5개 포인트 선택 (균등하게 샘플링)
                      const select5Points = <T,>(arr: T[]): T[] => {
                        if (arr.length <= 5) return arr;
                        const indices = [
                          Math.floor(arr.length * 0.0),
                          Math.floor(arr.length * 0.25),
                          Math.floor(arr.length * 0.5),
                          Math.floor(arr.length * 0.75),
                          arr.length - 1
                        ];
                        return indices.map(i => arr[i]);
                      };

                      // 실제 데이터의 5개 금액 값 선택
                      const sortedAmounts = [...cumulativeData.amounts].filter(a => isFinite(a) && a >= 0).sort((a, b) => a - b);
                      const selectedAmounts = select5Points(sortedAmounts);
                      const actualDataPoints = selectedAmounts.map((amount, i) => ({
                        amount,
                        probability: (i + 1) / 5 // 0.2, 0.4, 0.6, 0.8, 1.0
                      }));

                      // 각 분포에 대해 같은 금액 값들에 대한 CDF 계산
                      const distributionPoints = distributionLines.map(line => {
                        const points = actualDataPoints.map(actualPoint => {
                          // theoreticalCumulative에서 선형 보간을 사용하여 정확한 확률 계산
                          let probability = 0;
                          
                          // 정확히 일치하는 값 찾기
                          const exactIndex = line.data.amounts.findIndex(a => Math.abs(a - actualPoint.amount) < 1e-10);
                          if (exactIndex >= 0) {
                            probability = line.data.probabilities[exactIndex];
                          } else {
                            // 선형 보간
                            let lowerIndex = -1;
                            let upperIndex = -1;
                            
                            for (let i = 0; i < line.data.amounts.length - 1; i++) {
                              if (line.data.amounts[i] <= actualPoint.amount && actualPoint.amount <= line.data.amounts[i + 1]) {
                                lowerIndex = i;
                                upperIndex = i + 1;
                                break;
                              }
                            }
                            
                            if (lowerIndex >= 0 && upperIndex >= 0) {
                              // 선형 보간
                              const lowerAmount = line.data.amounts[lowerIndex];
                              const upperAmount = line.data.amounts[upperIndex];
                              const lowerProb = line.data.probabilities[lowerIndex];
                              const upperProb = line.data.probabilities[upperIndex];
                              
                              const ratio = (actualPoint.amount - lowerAmount) / (upperAmount - lowerAmount);
                              probability = lowerProb + ratio * (upperProb - lowerProb);
                            } else {
                              // 범위 밖이면 가장 가까운 값 사용
                              let closestProb = 0;
                              let minDiff = Infinity;
                              
                              for (let i = 0; i < line.data.amounts.length; i++) {
                                const diff = Math.abs(line.data.amounts[i] - actualPoint.amount);
                                if (diff < minDiff) {
                                  minDiff = diff;
                                  closestProb = line.data.probabilities[i];
                                }
                              }
                              probability = closestProb;
                            }
                          }
                          
                          return {
                            amount: actualPoint.amount,
                            probability: Math.max(0, Math.min(1, probability)) // 0과 1 사이로 제한
                          };
                        });
                        
                        return {
                          distributionType: line.distributionType,
                          color: line.color,
                          points
                        };
                      });

                      // 데이터 범위 계산
                      const allAmounts = [
                        ...actualDataPoints.map(p => p.amount),
                        ...distributionPoints.flatMap(d => d.points.map(p => p.amount))
                      ].filter(a => isFinite(a) && a >= 0);
                      
                      const maxAmount = Math.max(...allAmounts);
                      const minAmount = Math.min(...allAmounts);

                      // 가로축: 금액, 세로축: 확률
                      const xScale = (amount: number) => margin.left + ((amount - minAmount) / (maxAmount - minAmount || 1)) * chartWidth;
                      const yScale = (probability: number) => height - margin.bottom - (probability * chartHeight);

                      const getTicks = (min: number, max: number, count: number) => {
                        if (min === max) return [min];
                        const ticks = [];
                        const step = (max - min) / (count - 1);
                        for (let i = 0; i < count; i++) {
                          ticks.push(min + i * step);
                        }
                        return ticks;
                      };

                      const xTicks = getTicks(minAmount, maxAmount, 5);
                      const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

                      return (
                        <div className="border border-gray-300 rounded-lg p-4 bg-white">
                          <h4 className="text-md font-semibold text-gray-700 mb-3">Cumulative Distribution: Data vs Theoretical Distributions</h4>
                          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
                            {/* 그리드 */}
                            <defs>
                              <pattern id="cumulative-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
                              </pattern>
                            </defs>
                            <rect width={width} height={height} fill="url(#cumulative-grid)" />

                            {/* 축 */}
                            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />
                            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />

                            {/* 실제 데이터 포인트 (막대 그래프) - 5개만 표시 */}
                            {actualDataPoints.map((point, i) => {
                              const barWidth = (maxAmount - minAmount) / 20; // 적절한 막대 너비
                              const barHeight = yScale(0) - yScale(point.probability);
                              const barX = xScale(point.amount) - barWidth / 2;
                              const barY = yScale(point.probability);

                              return (
                                <rect
                                  key={i}
                                  x={barX}
                                  y={barY}
                                  width={barWidth}
                                  height={Math.max(barHeight, 1)}
                                  fill="#6b7280"
                                  opacity="0.5"
                                />
                              );
                            })}

                            {/* 각 분포의 선 그래프 - 5개 포인트만 표시 */}
                            {distributionPoints.map((dist, lineIdx) => {
                              if (dist.points.length === 0) return null;

                              const pathData = dist.points
                                .map((p, i) => {
                                  const x = xScale(p.amount);
                                  const y = yScale(p.probability);
                                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                                })
                                .join(' ');

                              return (
                                <g key={lineIdx}>
                                  <path
                                    d={pathData}
                                    fill="none"
                                    stroke={dist.color}
                                    strokeWidth="2"
                                    opacity="0.8"
                                  />
                                  {/* 데이터 포인트 표시 */}
                                  {dist.points.map((p, i) => (
                                    <circle
                                      key={i}
                                      cx={xScale(p.amount)}
                                      cy={yScale(p.probability)}
                                      r="3"
                                      fill={dist.color}
                                      opacity="0.9"
                                    />
                                  ))}
                                </g>
                              );
                            })}

                            {/* 범례 */}
                            <g>
                              {/* 실제 데이터 범례 */}
                              <rect
                                x={width - margin.right + 10}
                                y={margin.top + 10}
                                width="12"
                                height="12"
                                fill="#6b7280"
                                opacity="0.5"
                              />
                              <text
                                x={width - margin.right + 28}
                                y={margin.top + 20}
                                fontSize="11"
                                fill="#374151"
                                fontWeight="bold"
                              >
                                Actual Data
                              </text>
                              
                              {/* 각 분포 범례 */}
                              {distributionPoints.map((dist, lineIdx) => (
                                <g key={lineIdx}>
                                  <line
                                    x1={width - margin.right + 10}
                                    y1={margin.top + 35 + (lineIdx * 20)}
                                    x2={width - margin.right + 22}
                                    y2={margin.top + 35 + (lineIdx * 20)}
                                    stroke={dist.color}
                                    strokeWidth="2"
                                    opacity="0.8"
                                  />
                                  <circle
                                    cx={width - margin.right + 16}
                                    cy={margin.top + 35 + (lineIdx * 20)}
                                    r="3"
                                    fill={dist.color}
                                    opacity="0.9"
                                  />
                                  <text
                                    x={width - margin.right + 28}
                                    y={margin.top + 38 + (lineIdx * 20)}
                                    fontSize="11"
                                    fill={dist.color}
                                    fontWeight="bold"
                                  >
                                    {dist.distributionType}
                                  </text>
                                </g>
                              ))}
                            </g>

                            {/* X축 눈금 및 라벨 (금액) */}
                            {xTicks.map((tick) => (
                              <g key={tick}>
                                <line
                                  x1={xScale(tick)}
                                  y1={height - margin.bottom}
                                  x2={xScale(tick)}
                                  y2={height - margin.bottom + 5}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={xScale(tick)}
                                  y={height - margin.bottom + 20}
                                  textAnchor="middle"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {tick.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                </text>
                              </g>
                            ))}

                            {/* Y축 눈금 및 라벨 (확률) */}
                            {yTicks.map((tick) => (
                              <g key={tick}>
                                <line
                                  x1={margin.left}
                                  y1={yScale(tick)}
                                  x2={margin.left - 5}
                                  y2={yScale(tick)}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={margin.left - 10}
                                  y={yScale(tick) + 4}
                                  textAnchor="end"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {(tick * 100).toFixed(0)}%
                                </text>
                              </g>
                            ))}

                            {/* X축 라벨 (금액) */}
                            <text
                              x={width / 2}
                              y={height - 10}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                            >
                              Amount
                            </text>

                            {/* Y축 라벨 (확률) */}
                            <text
                              x={20}
                              y={height / 2}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                              transform={`rotate(-90 20 ${height / 2})`}
                            >
                              Cumulative Probability
                            </text>
                          </svg>
                        </div>
                      );
                    };

                    return <CumulativeChart />;
                  })()}

                  {/* Q-Q Plot and P-P Plot Charts */}
                  {(() => {
                    const selectedResult = sortedResults.find(r => r.distributionType === localSelected);
                    const qqPlot = selectedResult?.qqPlot;
                    const ppPlot = selectedResult?.ppPlot;

                    // Q-Q Plot 컴포넌트
                    const QQPlotChart: React.FC<{ data: { theoreticalQuantiles: number[]; sampleQuantiles: number[] } }> = ({ data }) => {
                      const { theoreticalQuantiles, sampleQuantiles } = data;
                      if (!theoreticalQuantiles || !sampleQuantiles || theoreticalQuantiles.length === 0) {
                        return <div className="text-gray-500 text-sm">Q-Q Plot 데이터가 없습니다.</div>;
                      }

                      const margin = { top: 20, right: 20, bottom: 50, left: 60 };
                      const width = 500;
                      const height = 400;
                      const chartWidth = width - margin.left - margin.right;
                      const chartHeight = height - margin.top - margin.bottom;

                      // 유효한 데이터만 필터링
                      const validPairs = theoreticalQuantiles
                        .map((x, i) => ({ x, y: sampleQuantiles[i] }))
                        .filter(p => isFinite(p.x) && isFinite(p.y) && p.x !== null && p.y !== null);

                      if (validPairs.length === 0) {
                        return <div className="text-gray-500 text-sm">Q-Q Plot 데이터가 없습니다.</div>;
                      }

                      const xMin = Math.min(...validPairs.map(p => p.x));
                      const xMax = Math.max(...validPairs.map(p => p.x));
                      const yMin = Math.min(...validPairs.map(p => p.y));
                      const yMax = Math.max(...validPairs.map(p => p.y));

                      // 범위에 여유 공간 추가 (5%)
                      const xRange = xMax - xMin || 1;
                      const yRange = yMax - yMin || 1;
                      const xMinAdjusted = xMin - xRange * 0.05;
                      const xMaxAdjusted = xMax + xRange * 0.05;
                      const yMinAdjusted = yMin - yRange * 0.05;
                      const yMaxAdjusted = yMax + yRange * 0.05;

                      const xScale = (x: number) => margin.left + ((x - xMinAdjusted) / (xMaxAdjusted - xMinAdjusted || 1)) * chartWidth;
                      const yScale = (y: number) => height - margin.bottom - ((y - yMinAdjusted) / (yMaxAdjusted - yMinAdjusted || 1)) * chartHeight;

                      // Perfect fit line (y = x)
                      const lineMin = Math.min(xMinAdjusted, yMinAdjusted);
                      const lineMax = Math.max(xMaxAdjusted, yMaxAdjusted);

                      const getTicks = (min: number, max: number, count: number) => {
                        if (min === max) return [min];
                        const ticks = [];
                        const step = (max - min) / (count - 1);
                        for (let i = 0; i < count; i++) {
                          ticks.push(min + i * step);
                        }
                        return ticks;
                      };

                      const xTicks = getTicks(xMinAdjusted, xMaxAdjusted, 5);
                      const yTicks = getTicks(yMinAdjusted, yMaxAdjusted, 5);

                      return (
                        <div className="border border-gray-300 rounded-lg p-4 bg-white">
                          <h4 className="text-md font-semibold text-gray-700 mb-3">Q-Q Plot: {localSelected} Distribution</h4>
                          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
                            {/* 그리드 */}
                            <defs>
                              <pattern id="qq-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
                              </pattern>
                            </defs>
                            <rect width={width} height={height} fill="url(#qq-grid)" />

                            {/* 축 */}
                            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />
                            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />

                            {/* Perfect fit line */}
                            <line
                              x1={xScale(lineMin)}
                              y1={yScale(lineMin)}
                              x2={xScale(lineMax)}
                              y2={yScale(lineMax)}
                              stroke="#ef4444"
                              strokeWidth="2"
                              strokeDasharray="5,5"
                              opacity="0.7"
                            />

                            {/* X축 눈금 및 라벨 */}
                            {xTicks.map((tick, i) => (
                              <g key={`x-${i}`}>
                                <line
                                  x1={xScale(tick)}
                                  y1={height - margin.bottom}
                                  x2={xScale(tick)}
                                  y2={height - margin.bottom + 5}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={xScale(tick)}
                                  y={height - margin.bottom + 20}
                                  textAnchor="middle"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {tick.toFixed(2)}
                                </text>
                              </g>
                            ))}

                            {/* Y축 눈금 및 라벨 */}
                            {yTicks.map((tick, i) => (
                              <g key={`y-${i}`}>
                                <line
                                  x1={margin.left}
                                  y1={yScale(tick)}
                                  x2={margin.left - 5}
                                  y2={yScale(tick)}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={margin.left - 10}
                                  y={yScale(tick) + 4}
                                  textAnchor="end"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {tick.toFixed(2)}
                                </text>
                              </g>
                            ))}

                            {/* 데이터 포인트 */}
                            {validPairs.map((pair, i) => (
                              <circle
                                key={i}
                                cx={xScale(pair.x)}
                                cy={yScale(pair.y)}
                                r="3"
                                fill="#3b82f6"
                                opacity="0.7"
                              />
                            ))}

                            {/* X축 라벨 */}
                            <text
                              x={width / 2}
                              y={height - 10}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                            >
                              Theoretical Quantiles
                            </text>

                            {/* Y축 라벨 */}
                            <text
                              x={15}
                              y={height / 2}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                              transform={`rotate(-90 15 ${height / 2})`}
                            >
                              Sample Quantiles
                            </text>
                          </svg>
                        </div>
                      );
                    };

                    // P-P Plot 컴포넌트
                    const PPPlotChart: React.FC<{ data: { theoreticalCDF: number[]; empiricalCDF: number[] } }> = ({ data }) => {
                      const { theoreticalCDF, empiricalCDF } = data;
                      if (!theoreticalCDF || !empiricalCDF || theoreticalCDF.length === 0) {
                        return <div className="text-gray-500 text-sm">P-P Plot 데이터가 없습니다.</div>;
                      }

                      const margin = { top: 20, right: 20, bottom: 50, left: 60 };
                      const width = 500;
                      const height = 400;
                      const chartWidth = width - margin.left - margin.right;
                      const chartHeight = height - margin.top - margin.bottom;

                      // 유효한 데이터만 필터링
                      const validPairs = theoreticalCDF
                        .map((x, i) => ({ x, y: empiricalCDF[i] }))
                        .filter(p => isFinite(p.x) && isFinite(p.y) && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);

                      if (validPairs.length === 0) {
                        return <div className="text-gray-500 text-sm">P-P Plot 데이터가 없습니다.</div>;
                      }

                      const xScale = (x: number) => margin.left + x * chartWidth;
                      const yScale = (y: number) => height - margin.bottom - y * chartHeight;

                      return (
                        <div className="border border-gray-300 rounded-lg p-4 bg-white">
                          <h4 className="text-md font-semibold text-gray-700 mb-3">P-P Plot: {localSelected} Distribution</h4>
                          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
                            {/* 그리드 */}
                            <defs>
                              <pattern id="pp-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
                              </pattern>
                            </defs>
                            <rect width={width} height={height} fill="url(#pp-grid)" />

                            {/* 축 */}
                            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />
                            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />

                            {/* Perfect fit line (y = x) */}
                            <line
                              x1={xScale(0)}
                              y1={yScale(0)}
                              x2={xScale(1)}
                              y2={yScale(1)}
                              stroke="#ef4444"
                              strokeWidth="2"
                              strokeDasharray="5,5"
                              opacity="0.7"
                            />

                            {/* 눈금 및 라벨 */}
                            {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
                              <g key={tick}>
                                <line
                                  x1={xScale(tick)}
                                  y1={height - margin.bottom}
                                  x2={xScale(tick)}
                                  y2={height - margin.bottom + 5}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={xScale(tick)}
                                  y={height - margin.bottom + 20}
                                  textAnchor="middle"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {tick.toFixed(2)}
                                </text>
                                <line
                                  x1={margin.left}
                                  y1={yScale(tick)}
                                  x2={margin.left - 5}
                                  y2={yScale(tick)}
                                  stroke="#6b7280"
                                  strokeWidth="1"
                                />
                                <text
                                  x={margin.left - 10}
                                  y={yScale(tick) + 4}
                                  textAnchor="end"
                                  fontSize="10"
                                  fill="#6b7280"
                                >
                                  {tick.toFixed(2)}
                                </text>
                              </g>
                            ))}

                            {/* 데이터 포인트 */}
                            {validPairs.map((pair, i) => (
                              <circle
                                key={i}
                                cx={xScale(pair.x)}
                                cy={yScale(pair.y)}
                                r="3"
                                fill="#3b82f6"
                                opacity="0.7"
                              />
                            ))}

                            {/* X축 라벨 */}
                            <text
                              x={width / 2}
                              y={height - 10}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                            >
                              Theoretical CDF
                            </text>

                            {/* Y축 라벨 */}
                            <text
                              x={15}
                              y={height / 2}
                              textAnchor="middle"
                              fontSize="12"
                              fill="#374151"
                              fontWeight="bold"
                              transform={`rotate(-90 15 ${height / 2})`}
                            >
                              Empirical CDF
                            </text>
                          </svg>
                        </div>
                      );
                    };

                    return (
                      <div className="space-y-4">
                        {/* Q-Q Plot Chart */}
                        {qqPlot ? (
                          <QQPlotChart data={qqPlot} />
                        ) : (
                          <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Q-Q Plot</h4>
                            <p className="text-sm text-gray-500">Q-Q Plot 데이터를 사용할 수 없습니다.</p>
                          </div>
                        )}

                        {/* P-P Plot Chart */}
                        {ppPlot ? (
                          <PPPlotChart data={ppPlot} />
                        ) : (
                          <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
                            <h4 className="text-md font-semibold text-gray-700 mb-2">P-P Plot</h4>
                            <p className="text-sm text-gray-500">P-P Plot 데이터를 사용할 수 없습니다.</p>
                          </div>
                        )}
                      </div>
                    );
                  })()}

                </div>
              </div>
            )}
          </div>
        </main>
      </div>
      {showSpreadView && spreadViewData.length > 0 && (
        <SpreadViewModal
          onClose={() => setShowSpreadView(false)}
          data={spreadViewData}
          columns={spreadViewColumns}
          title={`Spread View: ${module.name} - Yearly Aggregates`}
        />
      )}
    </div>
  );
};

