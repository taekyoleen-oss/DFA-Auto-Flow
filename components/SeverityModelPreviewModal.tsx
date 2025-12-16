import React, { useState, useMemo, useRef } from 'react';
import { CanvasModule, SeverityModelOutput, SeverityModelFitResult } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { useCopyOnCtrlC } from '../hooks/useCopyOnCtrlC';
import { SpreadViewModal } from './SpreadViewModal';

interface SeverityModelPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
  onSelectDistribution?: (moduleId: string, distributionType: string) => void;
}

export const SeverityModelPreviewModal: React.FC<SeverityModelPreviewModalProps> = ({ 
  module, 
  onClose,
  onSelectDistribution 
}) => {
  const output = module.outputData as SeverityModelOutput;
  if (!output || output.type !== 'SeverityModelOutput') {
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

  let results: SeverityModelFitResult[] = output.results || [];
  const selectedDistribution = output.selectedDistribution;

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
  const sortedResults = [...results].filter(r => !r.error).sort((a, b) => {
    const aicA = a.fitStatistics?.aic ?? Infinity;
    const aicB = b.fitStatistics?.aic ?? Infinity;
    return aicA - aicB;
  });

  const recommended = sortedResults.length > 0 ? sortedResults[0] : null;
  const [showSpreadView, setShowSpreadView] = useState(false);

  // 통계 계산 (originalData 사용)
  const statistics = useMemo(() => {
    if (!output.originalData || output.originalData.length === 0) {
      return null;
    }

    const amounts = [...output.originalData].filter(a => typeof a === 'number' && !isNaN(a) && a > 0);
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
  }, [output.originalData]);

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    return sortedResults.map(result => ({
      Distribution: result.distributionType,
      AIC: result.fitStatistics?.aic?.toFixed(4) || 'N/A',
      BIC: result.fitStatistics?.bic?.toFixed(4) || 'N/A',
      'Log Likelihood': result.fitStatistics?.logLikelihood?.toFixed(4) || 'N/A',
      'KS Statistic': result.fitStatistics?.ksStatistic?.toFixed(4) || 'N/A',
      'KS P-Value': result.fitStatistics?.ksPValue?.toFixed(4) || 'N/A',
      Mean: result.fitStatistics?.mean?.toFixed(4) || 'N/A',
      Variance: result.fitStatistics?.variance?.toFixed(4) || 'N/A',
    }));
  }, [sortedResults]);

  const spreadViewColumns = [
    { name: 'Distribution', type: 'string' },
    { name: 'AIC', type: 'number' },
    { name: 'BIC', type: 'number' },
    { name: 'Log Likelihood', type: 'number' },
    { name: 'KS Statistic', type: 'number' },
    { name: 'KS P-Value', type: 'number' },
    { name: 'Mean', type: 'number' },
    { name: 'Variance', type: 'number' },
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
          <h2 className="text-xl font-bold text-gray-800">Severity Model: {module.name}</h2>
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

            {/* Distribution Comparison Table */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Distribution Comparison</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Distribution</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">AIC</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">BIC</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Log Likelihood</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">KS Statistic</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">KS P-Value</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {sortedResults.map((result, idx) => (
                      <tr 
                        key={idx}
                        className={result.distributionType === localSelected ? 'bg-blue-50' : ''}
                        onClick={() => handleSelectDistribution(result.distributionType)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">
                          {result.distributionType}
                          {result.distributionType === recommended?.distributionType && (
                            <span className="ml-2 text-xs text-green-600">(Recommended)</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {result.fitStatistics?.aic?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {result.fitStatistics?.bic?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {result.fitStatistics?.logLikelihood?.toFixed(2) || 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {result.fitStatistics?.ksStatistic?.toFixed(4) || 'N/A'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900">
                          {result.fitStatistics?.ksPValue?.toFixed(4) || 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Selected Distribution Details */}
            {localSelected && sortedResults.find(r => r.distributionType === localSelected) && (
              <div className="bg-white border border-gray-200 rounded-lg p-4" ref={viewDetailsRef}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">View Details (Ctrl+C to copy)</h3>
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium text-gray-700">Select Distribution:</label>
                    <select
                      value={localSelected}
                      onChange={(e) => handleSelectDistribution(e.target.value)}
                      className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {sortedResults.map((result) => (
                        <option key={result.distributionType} value={result.distributionType}>
                          {result.distributionType}
                          {result.distributionType === recommended?.distributionType && ' (Recommended)'}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="space-y-4">
                  {(() => {
                    const selectedResult = sortedResults.find(r => r.distributionType === localSelected);
                    if (!selectedResult) return null;

                    return (
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-md font-semibold text-gray-700 mb-2">Parameters</h4>
                          <div className="bg-gray-50 p-3 rounded">
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                              {Object.entries(selectedResult.parameters).map(([key, value]) => (
                                <div key={key} className="flex flex-col">
                                  <span className="text-xs text-gray-500 mb-1">{key}</span>
                                  <span className="font-mono text-sm text-gray-800 font-semibold">
                                    {typeof value === 'number' ? value.toFixed(6) : String(value || 'N/A')}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                        {selectedResult.fitStatistics && (
                          <div>
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Fit Statistics</h4>
                            <div className="bg-gray-50 p-3 rounded">
                              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                {selectedResult.fitStatistics.aic !== undefined && selectedResult.fitStatistics.aic !== null && (
                                  <div className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">AIC</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {selectedResult.fitStatistics.aic.toFixed(4)}
                                    </span>
                                  </div>
                                )}
                                {selectedResult.fitStatistics.bic !== undefined && selectedResult.fitStatistics.bic !== null && (
                                  <div className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">BIC</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {selectedResult.fitStatistics.bic.toFixed(4)}
                                    </span>
                                  </div>
                                )}
                                {selectedResult.fitStatistics.logLikelihood !== undefined && selectedResult.fitStatistics.logLikelihood !== null && (
                                  <div className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">Log Likelihood</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {selectedResult.fitStatistics.logLikelihood.toFixed(4)}
                                    </span>
                                  </div>
                                )}
                                {selectedResult.fitStatistics.ksStatistic !== undefined && selectedResult.fitStatistics.ksStatistic !== null && (
                                  <div className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">KS Statistic</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {selectedResult.fitStatistics.ksStatistic.toFixed(4)}
                                    </span>
                                  </div>
                                )}
                                {selectedResult.fitStatistics.ksPValue !== undefined && selectedResult.fitStatistics.ksPValue !== null && (
                                  <div className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">KS P-Value</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {selectedResult.fitStatistics.ksPValue.toFixed(4)}
                                    </span>
                                  </div>
                                )}
                              </div>
                            </div>
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
    </div>
  );
};

