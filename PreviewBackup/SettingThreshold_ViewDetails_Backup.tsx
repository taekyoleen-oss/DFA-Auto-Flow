/**
 * Setting Threshold View Details 백업 파일
 *
 * 이 파일은 SettingThresholdPreviewModal.tsx의 전체 내용을 백업한 것입니다.
 * Setting Threshold의 View Details가 의도치 않게 수정될 경우 이 파일을 참조하여 복구할 수 있습니다.
 *
 * 백업 일자: 2025-01-17
 *
 * 복구 방법:
 * 1. SettingThresholdPreviewModal.tsx 파일을 확인합니다.
 * 2. 이 파일의 내용으로 교체합니다.
 *
 * 포함된 내용:
 * - HistogramChart 컴포넌트 (Threshold 라인 포함)
 * - SettingThresholdPreviewModal 컴포넌트
 * - 분석 탭: 전체 요약, 연도별 건수 테이블
 * - 분포 탭: 데이터 분포 히스토그램
 * - Spread View 기능
 * - CSV 다운로드 기능
 */

import React, { useMemo, useState, useRef } from 'react';
import { CanvasModule, SettingThresholdOutput } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { SpreadViewModal } from './SpreadViewModal';
import { useModalCopy } from '../hooks/useModalCopy';

interface HistogramChartProps {
  bins: number[];
  frequencies: number[];
  thresholds: number[];
  min: number;
  max: number;
}

const HistogramChart: React.FC<HistogramChartProps> = ({ bins, frequencies, thresholds, min, max }) => {
  const chartWidth = 800;
  const chartHeight = 400;
  const padding = { top: 20, right: 20, bottom: 60, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  const maxFrequency = Math.max(...frequencies, 1);
  const numBins = frequencies.length;
  const binWidth = innerWidth / numBins;

  // Threshold 위치 계산
  const thresholdPositions = thresholds.map(threshold => {
    if (threshold < min || threshold > max) return null;
    const position = ((threshold - min) / (max - min)) * innerWidth + padding.left;
    return { threshold, position };
  }).filter(t => t !== null) as Array<{ threshold: number; position: number }>;

  // 색상 배열 (Threshold별로 다른 색상)
  const thresholdColors = [
    '#ef4444', // red
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // orange
    '#8b5cf6', // purple
    '#ec4899', // pink
  ];

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

        {/* Threshold lines */}
        {thresholdPositions.map(({ threshold, position }, idx) => (
          <g key={`threshold-${idx}`}>
            <line
              x1={position}
              y1={padding.top}
              x2={position}
              y2={padding.top + innerHeight}
              stroke={thresholdColors[idx % thresholdColors.length]}
              strokeWidth="2"
              strokeDasharray="5,5"
            />
            <text
              x={position}
              y={padding.top + innerHeight + 20}
              fontSize="11"
              fill={thresholdColors[idx % thresholdColors.length]}
              textAnchor="middle"
              fontWeight="bold"
            >
              T{idx + 1}: {threshold.toLocaleString()}
            </text>
            <text
              x={position}
              y={padding.top + innerHeight + 35}
              fontSize="10"
              fill={thresholdColors[idx % thresholdColors.length]}
              textAnchor="middle"
            >
              ({threshold.toLocaleString()})
            </text>
          </g>
        ))}

        {/* X-axis label */}
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
          빈도 (Frequency)
        </text>
      </svg>
    </div>
  );
};

interface SettingThresholdPreviewModalProps {
  module: CanvasModule;
  projectName: string;
  onClose: () => void;
  onThresholdChange?: (moduleId: string, threshold: number) => void;
}

export const SettingThresholdPreviewModal: React.FC<SettingThresholdPreviewModalProps> = ({ 
  module, 
  projectName,
  onClose,
  onThresholdChange
}) => {
  const [activeTab, setActiveTab] = useState<'analysis' | 'distribution'>('analysis');
  const [showSpreadView, setShowSpreadView] = useState(false);
  const { contentRef, handleContextMenu, ContextMenuComponent } = useModalCopy();

  const output = module.outputData as SettingThresholdOutput;
  if (!output || output.type !== 'SettingThresholdOutput') return null;

  const { thresholds, yearlyCounts, statistics, thresholdResults, dataDistribution, selectedThreshold } = output;
  const [localSelectedThreshold, setLocalSelectedThreshold] = useState<number | undefined>(
    selectedThreshold || (thresholds.length > 0 ? thresholds[0] : undefined)
  );

  const handleThresholdSelect = (threshold: number) => {
    setLocalSelectedThreshold(threshold);
    if (onThresholdChange) {
      onThresholdChange(module.id, threshold);
    }
  };

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    if (!yearlyCounts || yearlyCounts.length === 0) return [];
    
    const data: Array<Record<string, any>> = [];
    
    // 총계, 평균, 표준편차 행 추가
    const totalsRow: Record<string, any> = { Year: '총계' };
    const meanRow: Record<string, any> = { Year: '평균' };
    const stdRow: Record<string, any> = { Year: '표준편차' };
    
    thresholds.forEach((threshold, idx) => {
      const thresholdLabel = `Threshold ${idx + 1} (${threshold.toLocaleString()})`;
      const allCounts = yearlyCounts.map(yc => yc.counts[idx] || 0);
      
      totalsRow[thresholdLabel] = allCounts.reduce((sum, val) => sum + val, 0);
      meanRow[thresholdLabel] = allCounts.length > 0 
        ? (allCounts.reduce((sum, val) => sum + val, 0) / allCounts.length).toFixed(2)
        : 0;
      
      if (allCounts.length > 1) {
        const mean = allCounts.reduce((sum, val) => sum + val, 0) / allCounts.length;
        const variance = allCounts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / allCounts.length;
        stdRow[thresholdLabel] = Math.sqrt(variance).toFixed(2);
      } else {
        stdRow[thresholdLabel] = '0.00';
      }
    });
    
    data.push(totalsRow);
    data.push(meanRow);
    data.push(stdRow);
    
    // 연도별 데이터 추가
    yearlyCounts.forEach(yc => {
      const row: Record<string, any> = { Year: yc.year };
      thresholds.forEach((threshold, idx) => {
        const thresholdLabel = `Threshold ${idx + 1} (${threshold.toLocaleString()})`;
        row[thresholdLabel] = yc.counts[idx] || 0;
      });
      data.push(row);
    });
    
    return data;
  }, [yearlyCounts, thresholds]);

  const spreadViewColumns = useMemo(() => {
    const cols = [{ name: 'Year', type: 'string' }];
    thresholds.forEach((threshold, idx) => {
      cols.push({ 
        name: `Threshold ${idx + 1} (${threshold.toLocaleString()})`, 
        type: 'number' 
      });
    });
    return cols;
  }, [thresholds]);

  const handleDownloadCSV = () => {
    if (!spreadViewData || spreadViewData.length === 0) return;
    
    const headers = spreadViewColumns.map(col => col.name);
    const rows = spreadViewData.map(row => 
      headers.map(header => {
        const value = row[header];
        return value === null || value === undefined ? '' : String(value);
      })
    );
    
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${module.name}_threshold_analysis.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <div 
          className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
          onClick={e => e.stopPropagation()}
          onContextMenu={handleContextMenu}
        >
          <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
            <h2 className="text-xl font-bold text-gray-800">Setting Threshold Preview: {module.name}</h2>
            <div className="flex items-center gap-2">
              {spreadViewData && spreadViewData.length > 0 && (
                <>
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
                    onClick={handleDownloadCSV}
                    className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                    title="Download CSV"
                  >
                    <ArrowDownTrayIcon className="w-5 h-5" />
                  </button>
                </>
              )}
              <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                <XCircleIcon className="w-6 h-6" />
              </button>
            </div>
          </header>

          <main className="flex-1 overflow-y-auto p-6">
            {/* Tabs */}
            <div className="flex gap-2 mb-4 border-b border-gray-200">
              <button
                onClick={() => setActiveTab('analysis')}
                className={`px-4 py-2 font-medium text-sm transition-colors ${
                  activeTab === 'analysis'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                분석
              </button>
              <button
                onClick={() => setActiveTab('distribution')}
                className={`px-4 py-2 font-medium text-sm transition-colors ${
                  activeTab === 'distribution'
                    ? 'text-blue-600 border-b-2 border-blue-600'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                분포
              </button>
            </div>

            {/* Analysis Tab */}
            {activeTab === 'analysis' && (
              <div ref={contentRef} className="space-y-4">
                {yearlyCounts && yearlyCounts.length > 0 ? (
                  <div className="space-y-4">
                    {/* Summary Statistics */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="text-lg font-semibold text-gray-800 mb-3">전체 요약(건수)</h3>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">총계: </span>
                          <span className="font-semibold">
                            {yearlyCounts.reduce((sum, yc) => 
                              sum + (yc.totals?.total || 0), 0
                            ).toLocaleString()}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">평균: </span>
                          <span className="font-semibold">
                            {yearlyCounts.length > 0
                              ? (yearlyCounts.reduce((sum, yc) => 
                                  sum + (yc.totals?.mean || 0), 0
                                ) / yearlyCounts.length).toFixed(2)
                              : '0.00'}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-600">표준편차: </span>
                          <span className="font-semibold">
                            {yearlyCounts.length > 0
                              ? (() => {
                                  const means = yearlyCounts.map(yc => yc.totals?.mean || 0);
                                  const overallMean = means.reduce((sum, m) => sum + m, 0) / means.length;
                                  const variance = means.reduce((sum, m) => sum + Math.pow(m - overallMean, 2), 0) / means.length;
                                  return Math.sqrt(variance).toFixed(2);
                                })()
                              : '0.00'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Yearly Counts */}
                    <div className="bg-white border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-800">연도별 건수</h3>
                        {thresholds.length > 0 && (
                          <div className="flex items-center gap-2">
                            <label className="text-sm font-medium text-gray-700">선택된 Threshold:</label>
                            <select
                              value={localSelectedThreshold || thresholds[0]}
                              onChange={(e) => handleThresholdSelect(Number(e.target.value))}
                              className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            >
                              {thresholds.map((threshold, idx) => (
                                <option key={idx} value={threshold}>
                                  Threshold {idx + 1}: {threshold.toLocaleString()}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                      </div>
                      <div className="space-y-2 text-sm">
                        {/* Header */}
                        <div className="grid gap-2 font-semibold text-gray-700 pb-2 border-b border-gray-300"
                          style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}>
                          <div>연도</div>
                          {thresholds.map((threshold, idx) => (
                            <div key={idx} className="text-center">
                              Threshold {idx + 1}<br />
                              <span className="text-xs font-normal">({threshold.toLocaleString()})</span>
                            </div>
                          ))}
                        </div>
                        
                        {/* Totals, Ratio, Mean, Std rows */}
                        {(() => {
                          const allCountsByThreshold = thresholds.map((_, idx) => 
                            yearlyCounts.map(yc => yc.counts[idx] || 0)
                          );
                          
                          const totals = allCountsByThreshold.map(counts => 
                            counts.reduce((sum, val) => sum + val, 0)
                          );
                          const grandTotal = totals.reduce((sum, val) => sum + val, 0);
                          const ratios = totals.map(total => 
                            grandTotal > 0 ? (total / grandTotal * 100).toFixed(2) : '0.00'
                          );
                          const means = allCountsByThreshold.map(counts => 
                            counts.length > 0 ? counts.reduce((sum, val) => sum + val, 0) / counts.length : 0
                          );
                          const stds = allCountsByThreshold.map((counts, idx) => {
                            const mean = means[idx];
                            const variance = counts.length > 0
                              ? counts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / counts.length
                              : 0;
                            return Math.sqrt(variance);
                          });
                          
                          return (
                            <>
                              <div className="grid gap-2 py-1 font-medium text-gray-800 bg-gray-50"
                                style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}>
                                <div>총계</div>
                                {totals.map((total, idx) => (
                                  <div key={idx} className="text-center">{total.toLocaleString()}</div>
                                ))}
                              </div>
                              <div className="grid gap-2 py-1 text-gray-700"
                                style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}>
                                <div>비율 (%)</div>
                                {ratios.map((ratio, idx) => (
                                  <div key={idx} className="text-center">{ratio}%</div>
                                ))}
                              </div>
                              <div className="grid gap-2 py-1 text-gray-700"
                                style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}>
                                <div>평균</div>
                                {means.map((mean, idx) => (
                                  <div key={idx} className="text-center">{mean.toFixed(2)}</div>
                                ))}
                              </div>
                              <div className="grid gap-2 py-1 text-gray-700 pb-2 border-b border-gray-300"
                                style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}>
                                <div>표준편차</div>
                                {stds.map((std, idx) => (
                                  <div key={idx} className="text-center">{std.toFixed(2)}</div>
                                ))}
                              </div>
                            </>
                          );
                        })()}
                        
                        {/* Year rows */}
                        {yearlyCounts.map((yc, rowIdx) => (
                          <div 
                            key={rowIdx}
                            className="grid gap-2 py-1 hover:bg-gray-50 border-b border-gray-100"
                            style={{ gridTemplateColumns: `100px repeat(${thresholds.length}, 1fr)` }}
                          >
                            <div className="font-medium">{yc.year}</div>
                            {yc.counts.map((count, colIdx) => (
                              <div key={colIdx} className="text-center">{count.toLocaleString()}</div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <p className="text-yellow-800">
                      연도별 건수 데이터가 없습니다. 속성창에서 연도 컬럼을 선택해주세요.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Distribution Tab */}
            {activeTab === 'distribution' && (
              <div ref={contentRef} className="space-y-4">
                {dataDistribution && dataDistribution.bins && dataDistribution.frequencies ? (
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">데이터 분포 히스토그램</h3>
                    <HistogramChart 
                      bins={dataDistribution.bins}
                      frequencies={dataDistribution.frequencies}
                      thresholds={thresholds}
                      min={statistics.min}
                      max={statistics.max}
                    />
                  </div>
                ) : (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <p className="text-yellow-800">
                      히스토그램 데이터가 없습니다.
                    </p>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>

      {showSpreadView && spreadViewData && spreadViewData.length > 0 && (
        <SpreadViewModal
          data={spreadViewData}
          columns={spreadViewColumns}
          title={`${module.name} - Threshold Analysis`}
          onClose={() => setShowSpreadView(false)}
        />
      )}

      {ContextMenuComponent}
    </>
  );
};


