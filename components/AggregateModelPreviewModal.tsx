import React, { useState } from 'react';
import { CanvasModule, AggregateModelOutput, AggregateModelFitResult } from '../types';
import { XCircleIcon } from './icons';

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

  // AIC 기준으로 정렬 (낮을수록 좋음)
  const sortedResults = [...results].sort((a, b) => {
    const aicA = a.fitStatistics?.aic ?? Infinity;
    const aicB = b.fitStatistics?.aic ?? Infinity;
    return aicA - aicB;
  });

  const recommended = sortedResults.length > 0 ? sortedResults[0] : null;

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
              onClick={() => {
                if (!yearlyAggregates || yearlyAggregates.length === 0) return;
                const csvContent = [
                  'year,total_amount',
                  ...yearlyAggregates.map(agg => `${agg.year},${agg.totalAmount}`)
                ].join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
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
          </div>
        </main>
      </div>
    </div>
  );
};

