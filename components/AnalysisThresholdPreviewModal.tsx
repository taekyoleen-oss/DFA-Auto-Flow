import React, { useMemo, useState } from 'react';
import { CanvasModule, AnalysisThresholdOutput } from '../types';
import { XCircleIcon } from './icons';
import { useModalCopy } from '../hooks/useModalCopy';

interface AnalysisThresholdPreviewModalProps {
  module: CanvasModule;
  projectName: string;
  onClose: () => void;
}

// 첫 번째 탭: 데이터 분포 플롯
const DistributionPlot: React.FC<{ output: AnalysisThresholdOutput }> = ({ output }) => {
  const chartWidth = 800;
  const chartHeight = 400;
  const padding = { top: 20, right: 20, bottom: 60, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const distribution = output.distribution;
  if (!distribution) {
    return <div className="text-gray-500 p-4">데이터 분포 정보가 없습니다.</div>;
  }

  const { claimSizes, counts, bins, frequencies } = distribution;
  
  // bins와 frequencies가 있으면 히스토그램 사용
  if (bins && frequencies) {
    const min = Math.min(...bins);
    const max = Math.max(...bins);
    const maxFrequency = Math.max(...frequencies, 1);
    const numBins = frequencies.length;
    const binWidth = innerWidth / numBins;

    return (
      <div className="w-full overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
          {/* Y-axis */}
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          
          {/* X-axis */}
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />

          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, idx) => {
            const y = padding.top + innerHeight - (ratio * innerHeight);
            const value = Math.round(ratio * maxFrequency);
            return (
              <g key={`y-label-${idx}`}>
                <line
                  x1={padding.left - 5}
                  y1={y}
                  x2={padding.left}
                  y2={y}
                  stroke="#9ca3af"
                  strokeWidth="1"
                />
                <text
                  x={padding.left - 10}
                  y={y + 4}
                  fontSize="12"
                  fill="#6b7280"
                  textAnchor="end"
                >
                  {value}
                </text>
              </g>
            );
          })}

          {/* Histogram bars */}
          {frequencies.map((freq, idx) => {
            if (idx >= bins.length - 1) return null;
            const barHeight = (freq / maxFrequency) * innerHeight;
            const binStart = bins[idx];
            const binEnd = bins[idx + 1];
            const binCenter = (binStart + binEnd) / 2;
            const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth - (binWidth * 0.9 / 2);
            const y = padding.top + innerHeight - barHeight;
            
            return (
              <rect
                key={`bar-${idx}`}
                x={x}
                y={y}
                width={binWidth * 0.9}
                height={barHeight}
                fill="#3b82f6"
                opacity={0.7}
              />
            );
          })}

          {/* X-axis label */}
          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 10}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
          >
            클레임 크기 (Claim Size)
          </text>

          {/* Y-axis label */}
          <text
            x={15}
            y={padding.top + innerHeight / 2}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
          >
            건수 (Count)
          </text>
        </svg>
      </div>
    );
  }

  // claimSizes와 counts가 있으면 산점도 사용
  if (claimSizes && counts) {
    const xMin = Math.min(...claimSizes);
    const xMax = Math.max(...claimSizes);
    const yMax = Math.max(...counts, 1);
    
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - ((y / yMax) * innerHeight);

    return (
      <div className="w-full overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          
          {claimSizes.map((size, idx) => (
            <circle
              key={idx}
              cx={xScale(size)}
              cy={yScale(counts[idx])}
              r="3"
              fill="#3b82f6"
              opacity={0.7}
            />
          ))}
          
          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 10}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
          >
            클레임 크기 (Claim Size)
          </text>
          <text
            x={15}
            y={padding.top + innerHeight / 2}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
          >
            건수 (Count)
          </text>
        </svg>
      </div>
    );
  }

  return <div className="text-gray-500 p-4">데이터를 표시할 수 없습니다.</div>;
};

// 두 번째 탭: 경험적 분포 (Histogram, ECDF, QQ-Plot)
const EmpiricalDistributionTab: React.FC<{ output: AnalysisThresholdOutput }> = ({ output }) => {
  const chartWidth = 800;
  const chartHeight = 400;
  const padding = { top: 20, right: 20, bottom: 60, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const empirical = output.empiricalDistribution;
  if (!empirical) {
    return <div className="text-gray-500 p-4">경험적 분포 정보가 없습니다.</div>;
  }

  const [activePlot, setActivePlot] = useState<'histogram' | 'ecdf' | 'qqplot'>('histogram');

  // Histogram
  const HistogramPlot = () => {
    if (!empirical.histogram) return null;
    const { bins, frequencies, tailChangePoint } = empirical.histogram;
    const min = Math.min(...bins);
    const max = Math.max(...bins);
    const maxFrequency = Math.max(...frequencies, 1);
    const numBins = frequencies.length;
    const binWidth = innerWidth / numBins;

    return (
      <div className="w-full overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />

          {frequencies.map((freq, idx) => {
            if (idx >= bins.length - 1) return null;
            const barHeight = (freq / maxFrequency) * innerHeight;
            const binStart = bins[idx];
            const binEnd = bins[idx + 1];
            const binCenter = (binStart + binEnd) / 2;
            const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth - (binWidth * 0.9 / 2);
            const y = padding.top + innerHeight - barHeight;
            
            return (
              <rect
                key={`bar-${idx}`}
                x={x}
                y={y}
                width={binWidth * 0.9}
                height={barHeight}
                fill="#3b82f6"
                opacity={0.7}
              />
            );
          })}

          {tailChangePoint && (
            <line
              x1={padding.left + ((tailChangePoint - min) / (max - min)) * innerWidth}
              y1={padding.top}
              x2={padding.left + ((tailChangePoint - min) / (max - min)) * innerWidth}
              y2={padding.top + innerHeight}
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          )}

          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 10}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
          >
            값 (Value)
          </text>
          <text
            x={15}
            y={padding.top + innerHeight / 2}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
          >
            빈도 (Frequency)
          </text>
          {tailChangePoint && (
            <text
              x={padding.left + ((tailChangePoint - min) / (max - min)) * innerWidth}
              y={padding.top + innerHeight + 20}
              fontSize="11"
              fill="#ef4444"
              textAnchor="middle"
              fontWeight="bold"
            >
              꼬리 변화점: {tailChangePoint.toLocaleString()}
            </text>
          )}
        </svg>
      </div>
    );
  };

  // ECDF
  const ECDFPlot = () => {
    if (!empirical.ecdf) return null;
    const { sortedValues, cumulativeProbabilities, tailChangePoint } = empirical.ecdf;
    const xMin = Math.min(...sortedValues);
    const xMax = Math.max(...sortedValues);
    
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - (y * innerHeight);

    return (
      <div className="w-full overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          
          <polyline
            points={sortedValues.map((x, i) => `${xScale(x)},${yScale(cumulativeProbabilities[i])}`).join(' ')}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
          />

          {tailChangePoint && (
            <line
              x1={xScale(tailChangePoint)}
              y1={padding.top}
              x2={xScale(tailChangePoint)}
              y2={padding.top + innerHeight}
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          )}

          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 10}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
          >
            값 (Value)
          </text>
          <text
            x={15}
            y={padding.top + innerHeight / 2}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
          >
            누적 확률 (Cumulative Probability)
          </text>
          {tailChangePoint && (
            <text
              x={xScale(tailChangePoint)}
              y={padding.top + innerHeight + 20}
              fontSize="11"
              fill="#ef4444"
              textAnchor="middle"
              fontWeight="bold"
            >
              꼬리 변화점: {tailChangePoint.toLocaleString()}
            </text>
          )}
        </svg>
      </div>
    );
  };

  // QQ-Plot
  const QQPlot = () => {
    if (!empirical.qqPlot) return null;
    const { theoreticalQuantiles, sampleQuantiles, tailChangePoint } = empirical.qqPlot;
    const xMin = Math.min(...theoreticalQuantiles);
    const xMax = Math.max(...theoreticalQuantiles);
    const yMin = Math.min(...sampleQuantiles);
    const yMax = Math.max(...sampleQuantiles);
    const diagonalMax = Math.max(xMax, yMax);
    const diagonalMin = Math.min(xMin, yMin);
    
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - ((y - yMin) / (yMax - yMin || 1)) * innerHeight;

    return (
      <div className="w-full overflow-x-auto">
        <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          <line
            x1={padding.left}
            y1={padding.top + innerHeight}
            x2={padding.left + innerWidth}
            y2={padding.top + innerHeight}
            stroke="#374151"
            strokeWidth="2"
          />
          <line
            x1={xScale(diagonalMin)}
            y1={yScale(diagonalMin)}
            x2={xScale(diagonalMax)}
            y2={yScale(diagonalMax)}
            stroke="#ef4444"
            strokeWidth="1"
            strokeDasharray="5,5"
            opacity={0.7}
          />
          {theoreticalQuantiles.map((x, i) => (
            <circle
              key={i}
              cx={xScale(x)}
              cy={yScale(sampleQuantiles[i])}
              r="3"
              fill="#3b82f6"
              opacity={0.7}
            />
          ))}
          {tailChangePoint && (
            <line
              x1={xScale(tailChangePoint)}
              y1={padding.top}
              x2={xScale(tailChangePoint)}
              y2={padding.top + innerHeight}
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          )}
          <text
            x={padding.left + innerWidth / 2}
            y={chartHeight - 10}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
          >
            이론적 분위수 (Theoretical Quantiles)
          </text>
          <text
            x={15}
            y={padding.top + innerHeight / 2}
            fontSize="14"
            fill="#374151"
            textAnchor="middle"
            fontWeight="semibold"
            transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
          >
            표본 분위수 (Sample Quantiles)
          </text>
          {tailChangePoint && (
            <text
              x={xScale(tailChangePoint)}
              y={padding.top + innerHeight + 20}
              fontSize="11"
              fill="#ef4444"
              textAnchor="middle"
              fontWeight="bold"
            >
              꼬리 변화점: {tailChangePoint.toLocaleString()}
            </text>
          )}
        </svg>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-gray-300">
        <button
          onClick={() => setActivePlot('histogram')}
          className={`px-4 py-2 text-sm font-semibold ${
            activePlot === 'histogram'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          Histogram
        </button>
        <button
          onClick={() => setActivePlot('ecdf')}
          className={`px-4 py-2 text-sm font-semibold ${
            activePlot === 'ecdf'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          ECDF
        </button>
        <button
          onClick={() => setActivePlot('qqplot')}
          className={`px-4 py-2 text-sm font-semibold ${
            activePlot === 'qqplot'
              ? 'border-b-2 border-blue-600 text-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          QQ-Plot
        </button>
      </div>
      
      {activePlot === 'histogram' && <HistogramPlot />}
      {activePlot === 'ecdf' && <ECDFPlot />}
      {activePlot === 'qqplot' && <QQPlot />}
    </div>
  );
};

// 세 번째 탭: Mean Excess Plot
const MeanExcessPlotTab: React.FC<{ output: AnalysisThresholdOutput }> = ({ output }) => {
  const chartWidth = 800;
  const chartHeight = 400;
  const padding = { top: 20, right: 20, bottom: 60, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const meanExcess = output.meanExcessPlot;
  if (!meanExcess) {
    return <div className="text-gray-500 p-4">Mean Excess Plot 정보가 없습니다.</div>;
  }

  const { thresholds, meanExcesses, linearRange } = meanExcess;
  const xMin = Math.min(...thresholds);
  const xMax = Math.max(...thresholds);
  const yMin = Math.min(...meanExcesses);
  const yMax = Math.max(...meanExcesses);
  
  const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
  const yScale = (y: number) => padding.top + innerHeight - ((y - yMin) / (yMax - yMin || 1)) * innerHeight;

  return (
    <div className="w-full overflow-x-auto">
      <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={padding.top + innerHeight}
          stroke="#374151"
          strokeWidth="2"
        />
        <line
          x1={padding.left}
          y1={padding.top + innerHeight}
          x2={padding.left + innerWidth}
          y2={padding.top + innerHeight}
          stroke="#374151"
          strokeWidth="2"
        />
        
        <polyline
          points={thresholds.map((x, i) => `${xScale(x)},${yScale(meanExcesses[i])}`).join(' ')}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
        />
        
        {thresholds.map((x, i) => (
          <circle
            key={i}
            cx={xScale(x)}
            cy={yScale(meanExcesses[i])}
            r="3"
            fill="#3b82f6"
          />
        ))}

        {linearRange && (
          <>
            <line
              x1={xScale(linearRange.start)}
              y1={padding.top}
              x2={xScale(linearRange.start)}
              y2={padding.top + innerHeight}
              stroke="#10b981"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
            <line
              x1={xScale(linearRange.end)}
              y1={padding.top}
              x2={xScale(linearRange.end)}
              y2={padding.top + innerHeight}
              stroke="#10b981"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
            <text
              x={xScale(linearRange.start)}
              y={padding.top + innerHeight + 20}
              fontSize="11"
              fill="#10b981"
              textAnchor="middle"
              fontWeight="bold"
            >
              선형 구간 시작: {linearRange.start.toLocaleString()}
            </text>
            <text
              x={xScale(linearRange.end)}
              y={padding.top + innerHeight + 35}
              fontSize="11"
              fill="#10b981"
              textAnchor="middle"
              fontWeight="bold"
            >
              선형 구간 종료: {linearRange.end.toLocaleString()}
            </text>
          </>
        )}

        <text
          x={padding.left + innerWidth / 2}
          y={chartHeight - 10}
          fontSize="14"
          fill="#374151"
          textAnchor="middle"
          fontWeight="semibold"
        >
          Threshold (u)
        </text>
        <text
          x={15}
          y={padding.top + innerHeight / 2}
          fontSize="14"
          fill="#374151"
          textAnchor="middle"
          fontWeight="semibold"
          transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
        >
          Mean Excess e(u)
        </text>
      </svg>
    </div>
  );
};

export const AnalysisThresholdPreviewModal: React.FC<AnalysisThresholdPreviewModalProps> = ({
  module,
  projectName,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'distribution' | 'empirical' | 'meanexcess'>('distribution');
  const { contentRef, handleContextMenu, ContextMenuComponent } = useModalCopy();

  const output = module.outputData as AnalysisThresholdOutput;
  if (!output || output.type !== 'AnalysisThresholdOutput') {
    return null;
  }

  const { claimColumn, statistics } = output;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">
            Analysis Threshold: {module.name}
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>

        <div className="flex-shrink-0 border-b border-gray-200">
          <div className="flex">
            <button
              onClick={() => setActiveTab('distribution')}
              className={`flex-1 px-4 py-3 text-sm font-semibold ${
                activeTab === 'distribution'
                  ? 'bg-gray-100 text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              데이터 분포
            </button>
            <button
              onClick={() => setActiveTab('empirical')}
              className={`flex-1 px-4 py-3 text-sm font-semibold ${
                activeTab === 'empirical'
                  ? 'bg-gray-100 text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              경험적 분포
            </button>
            <button
              onClick={() => setActiveTab('meanexcess')}
              className={`flex-1 px-4 py-3 text-sm font-semibold ${
                activeTab === 'meanexcess'
                  ? 'bg-gray-100 text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              Mean Excess Plot
            </button>
          </div>
        </div>

        <main
          ref={contentRef}
          className="flex-grow p-6 overflow-auto"
          onContextMenu={handleContextMenu}
        >
          <div className="mb-4">
            <p className="text-sm text-gray-600">
              <span className="font-semibold">클레임 열:</span> {claimColumn}
            </p>
            <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-semibold">평균:</span> {statistics.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div>
                <span className="font-semibold">중앙값:</span> {statistics.median.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
              <div>
                <span className="font-semibold">표준편차:</span> {statistics.std.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
            </div>
          </div>

          {activeTab === 'distribution' && <DistributionPlot output={output} />}
          {activeTab === 'empirical' && <EmpiricalDistributionTab output={output} />}
          {activeTab === 'meanexcess' && <MeanExcessPlotTab output={output} />}
        </main>

        {ContextMenuComponent}
      </div>
    </div>
  );
};
