import React, { useState, useMemo } from "react";
import { CanvasModule } from "../types";
import { CombineLossModelOutput } from "../types";
import { XMarkIcon, ArrowDownTrayIcon } from "./icons";
import { useCopyOnCtrlC } from "../hooks/useCopyOnCtrlC";
import { SpreadViewModal } from "./SpreadViewModal";

interface CombineLossModelPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
}

export const CombineLossModelPreviewModal: React.FC<
  CombineLossModelPreviewModalProps
> = ({ module, onClose }) => {
  const outputData = module.outputData as CombineLossModelOutput;
  const contentRef = useCopyOnCtrlC<HTMLDivElement>();

  if (!outputData || outputData.type !== "CombineLossModelOutput") {
    return null;
  }

  const { combinedStatistics, var: varData, tvar, percentiles, aggregateLossDistribution, aggDistPercentiles, freqServPercentiles } = outputData;
  const [showSpreadView, setShowSpreadView] = useState(false);

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    const data: Array<Record<string, any>> = [];
    
    // Combined Percentiles
    if (percentiles) {
      Object.entries(percentiles).forEach(([level, value]) => {
        data.push({
          type: 'Combined',
          percentile: `${level}%`,
          value: value,
        });
      });
    }
    
    // Aggregate Distribution Percentiles
    if (aggDistPercentiles) {
      Object.entries(aggDistPercentiles).forEach(([level, value]) => {
        data.push({
          type: 'Agg Dist',
          percentile: `${level}%`,
          value: value,
        });
      });
    }
    
    // Frequency-Severity Percentiles
    if (freqServPercentiles) {
      Object.entries(freqServPercentiles).forEach(([level, value]) => {
        data.push({
          type: 'Freq-Serv',
          percentile: `${level}%`,
          value: value,
        });
      });
    }
    
    return data;
  }, [percentiles, aggDistPercentiles, freqServPercentiles]);

  const spreadViewColumns = [
    { name: 'type', type: 'string' },
    { name: 'percentile', type: 'string' },
    { name: 'value', type: 'number' },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg shadow-xl w-[90vw] max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">
            {module.name} - Combined Loss Model Results
          </h2>
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
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
        </div>

        <div
          ref={contentRef}
          className="flex-1 overflow-y-auto p-6 space-y-6"
        >
          {/* Combined Statistics */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">
              Combined Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Mean</p>
                <p className="text-xl font-bold text-white">
                  {combinedStatistics.mean.toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Std Dev</p>
                <p className="text-xl font-bold text-white">
                  {combinedStatistics.stdDev.toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Min</p>
                <p className="text-xl font-bold text-white">
                  {combinedStatistics.min.toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Max</p>
                <p className="text-xl font-bold text-white">
                  {combinedStatistics.max.toLocaleString(undefined, {
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
              {combinedStatistics.skewness !== undefined && (
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400">Skewness</p>
                  <p className="text-xl font-bold text-white">
                    {combinedStatistics.skewness.toFixed(4)}
                  </p>
                </div>
              )}
              {combinedStatistics.kurtosis !== undefined && (
                <div className="bg-gray-800 rounded-lg p-4">
                  <p className="text-sm text-gray-400">Kurtosis</p>
                  <p className="text-xl font-bold text-white">
                    {combinedStatistics.kurtosis.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* VaR and TVaR */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Value at Risk (VaR)
              </h3>
              <div className="bg-gray-800 rounded-lg p-4 space-y-3">
                {Object.entries(varData).map(([level, value]) => (
                  <div
                    key={level}
                    className="flex justify-between items-center border-b border-gray-700 pb-2 last:border-0 last:pb-0"
                  >
                    <span className="text-gray-300 font-semibold">
                      {level}%
                    </span>
                    <span className="text-white font-bold">
                      {value.toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Tail Value at Risk (TVaR)
              </h3>
              <div className="bg-gray-800 rounded-lg p-4 space-y-3">
                {Object.entries(tvar).map(([level, value]) => (
                  <div
                    key={level}
                    className="flex justify-between items-center border-b border-gray-700 pb-2 last:border-0 last:pb-0"
                  >
                    <span className="text-gray-300 font-semibold">
                      {level}%
                    </span>
                    <span className="text-white font-bold">
                      {value.toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Percentiles */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">
              Percentiles
            </h3>
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(percentiles).map(([level, value]) => (
                  <div key={level} className="text-center">
                    <p className="text-sm text-gray-400">{level}%</p>
                    <p className="text-lg font-bold text-white">
                      {value.toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Aggregate Loss Distribution Chart */}
          {aggregateLossDistribution && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">
                Aggregate Loss Distribution
              </h3>
              <div className="bg-gray-800 rounded-lg p-4">
                <svg
                  width="100%"
                  height="400"
                  viewBox="0 0 800 400"
                  className="border border-gray-700 rounded"
                >
                  <defs>
                    <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#1e40af" stopOpacity="0.3" />
                    </linearGradient>
                  </defs>
                  
                  {/* Axes */}
                  <line
                    x1="50"
                    y1="350"
                    x2="750"
                    y2="350"
                    stroke="#4b5563"
                    strokeWidth="2"
                  />
                  <line
                    x1="50"
                    y1="350"
                    x2="50"
                    y2="50"
                    stroke="#4b5563"
                    strokeWidth="2"
                  />

                  {/* Axis labels */}
                  <text
                    x="400"
                    y="390"
                    textAnchor="middle"
                    fill="#9ca3af"
                    fontSize="14"
                    fontWeight="bold"
                  >
                    Percentile
                  </text>
                  <text
                    x="15"
                    y="200"
                    textAnchor="middle"
                    fill="#9ca3af"
                    fontSize="14"
                    fontWeight="bold"
                    transform="rotate(-90, 15, 200)"
                  >
                    Amount
                  </text>

                  {/* Chart area */}
                  <g clipPath="url(#chartClip)">
                    {aggregateLossDistribution.percentiles.map((p, i) => {
                      if (i === 0) return null;
                      const x1 =
                        50 +
                        ((aggregateLossDistribution.percentiles[i - 1] / 100) *
                          (750 - 50));
                      const x2 = 50 + ((p / 100) * (750 - 50));
                      const y1 =
                        350 -
                        ((aggregateLossDistribution.amounts[i - 1] /
                          Math.max(...aggregateLossDistribution.amounts)) *
                          (350 - 50));
                      const y2 =
                        350 -
                        ((aggregateLossDistribution.amounts[i] /
                          Math.max(...aggregateLossDistribution.amounts)) *
                          (350 - 50));

                      return (
                        <line
                          key={i}
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="#3b82f6"
                          strokeWidth="2"
                        />
                      );
                    })}
                  </g>

                  {/* Clip path */}
                  <defs>
                    <clipPath id="chartClip">
                      <rect x="50" y="50" width="700" height="300" />
                    </clipPath>
                  </defs>
                </svg>
              </div>
            </div>
          )}
        </div>
      </div>
      {showSpreadView && spreadViewData.length > 0 && (
        <SpreadViewModal
          onClose={() => setShowSpreadView(false)}
          data={spreadViewData}
          columns={spreadViewColumns}
          title={`Spread View: ${module.name} - Combined Loss Model`}
        />
      )}
    </div>
  );
};
