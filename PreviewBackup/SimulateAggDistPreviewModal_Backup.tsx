import React, { useMemo, useState } from 'react';
import { CanvasModule, SimulateAggDistOutput } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { SpreadViewModal } from './SpreadViewModal';

interface SimulateAggDistPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
}

export const SimulateAggDistPreviewModal: React.FC<SimulateAggDistPreviewModalProps> = ({ 
  module, 
  onClose 
}) => {
  const output = module.outputData as SimulateAggDistOutput;
  if (!output || output.type !== 'SimulateAggDistOutput') {
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

  const { results, statistics, simulationCount } = output;
  const [showSpreadView, setShowSpreadView] = useState(false);

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    if (!results || results.length === 0) return [];
    return results.map(result => ({
      amount: result.amount,
      count: result.count,
    }));
  }, [results]);

  const spreadViewColumns = [
    { name: 'amount', type: 'number' },
    { name: 'count', type: 'number' },
  ];

  // 히스토그램 차트 데이터 계산
  const histogramChartData = useMemo(() => {
    if (!results || results.length === 0) return null;

    const width = 800;
    const height = 400;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };

    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // 최대 count 찾기
    const maxCount = Math.max(...results.map(r => r.count));
    const minAmount = Math.min(...results.map(r => r.amount));
    const maxAmount = Math.max(...results.map(r => r.amount));

    // 막대 너비 계산
    const barWidth = chartWidth / results.length;
    const maxBarHeight = chartHeight * 0.9; // 최대 막대 높이 (90% 사용)

    return {
      width,
      height,
      padding,
      chartWidth,
      chartHeight,
      maxCount,
      minAmount,
      maxAmount,
      barWidth,
      maxBarHeight,
      bars: results.map((result, index) => {
        const barHeight = (result.count / maxCount) * maxBarHeight;
        const x = padding.left + index * barWidth;
        const y = padding.top + chartHeight - barHeight;
        
        return {
          x,
          y,
          width: barWidth * 0.9, // 막대 간 간격을 위해 90%만 사용
          height: barHeight,
          count: result.count,
          amount: result.amount,
        };
      }),
    };
  }, [results]);

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">Simulation Results: {module.name}</h2>
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
                if (!results || results.length === 0) return;
                const csvContent = [
                  'count,amount',
                  ...results.map(r => `${r.count},${r.amount}`)
                ].join('\n');
                // BOM 추가하여 한글 인코딩 문제 해결
                const bom = '\uFEFF';
                const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `${module.name}_simulation_results.csv`;
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
        <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
          {/* Simulation Info */}
          <div className="bg-gray-50 rounded-lg p-3 flex-shrink-0">
            <h3 className="text-sm font-semibold text-gray-800 mb-1">Simulation Information</h3>
            <p className="text-xs text-gray-600">
              Total Simulations: <span className="font-semibold">{simulationCount.toLocaleString('ko-KR')}</span>
            </p>
          </div>

          {/* 왼쪽: 테이블, 오른쪽: 통계량 */}
          <div className="flex gap-2 flex-shrink-0" style={{ maxHeight: '60vh' }}>
            {/* 왼쪽: Raw Simulation Results Table */}
            <div className="flex-1 flex flex-col gap-1 min-w-0">
              {output.rawSimulations && output.rawSimulations.length > 0 ? (
                <>
                  <div className="flex justify-between items-center flex-shrink-0 px-1">
                    <h3 className="text-sm font-semibold text-gray-800">
                      Simulation Results (Raw Values)
                        <span className="text-xs font-normal text-gray-500 ml-2">
                         (Total: {simulationCount.toLocaleString('ko-KR')})
                        </span>
                    </h3>
                  </div>
                   <div className="flex-1 rounded-lg overflow-auto min-h-0" style={{ maxHeight: 'calc(10 * 2.5rem + 2.5rem)' }}>
                    <table className="min-w-full text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="px-1.5 py-1 text-right font-semibold text-gray-600">#</th>
                          <th className="px-1.5 py-1 text-right font-semibold text-gray-600">Simulated Value</th>
                        </tr>
                      </thead>
                      <tbody>
                         {output.rawSimulations.map((value, index) => (
                          <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="px-1.5 py-1 text-right font-mono text-gray-600">
                              {index + 1}
                            </td>
                            <td className="px-1.5 py-1 text-right font-mono text-gray-800">
                              {value.toLocaleString('ko-KR', { maximumFractionDigits: 4 })}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="flex-1 rounded-lg p-4 flex items-center justify-center">
                  <p className="text-sm text-gray-500">No raw simulation data available.</p>
                </div>
              )}
            </div>

            {/* 오른쪽: Statistics */}
            <div className="w-96 flex-shrink-0">
              <div className="h-full rounded-lg p-3 overflow-auto">
                <h3 className="text-sm font-semibold text-gray-800 mb-3">통계량</h3>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Mean:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.mean.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Std Dev:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.std.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Min:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.min.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">Max:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.max.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">5th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile5.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">25th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile25.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">50th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile50.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">75th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile75.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">95th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile95.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">99th Percentile:</span>
                    <span className="font-mono text-sm text-gray-800 font-semibold">
                      {statistics.percentile99.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 아래: Histogram Results Chart */}
          <div className="flex-shrink-0 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-800 mb-3">히스토그램</h3>
            {histogramChartData ? (
              <div className="overflow-x-auto">
                <svg
                  width={histogramChartData.width}
                  height={histogramChartData.height}
                  className="rounded"
                >
                  {/* 배경 그리드 */}
                  <defs>
                    <pattern
                      id="grid"
                      width="40"
                      height="40"
                      patternUnits="userSpaceOnUse"
                    >
                      <path
                        d="M 40 0 L 0 0 0 40"
                        fill="none"
                        stroke="#e5e7eb"
                        strokeWidth="1"
                      />
                    </pattern>
                  </defs>
                  <rect
                    width={histogramChartData.width}
                    height={histogramChartData.height}
                    fill="url(#grid)"
                  />

                  {/* Y축 라인 */}
                  <line
                    x1={histogramChartData.padding.left}
                    y1={histogramChartData.padding.top}
                    x2={histogramChartData.padding.left}
                    y2={histogramChartData.padding.top + histogramChartData.chartHeight}
                    stroke="#374151"
                    strokeWidth="2"
                  />

                  {/* X축 라인 */}
                  <line
                    x1={histogramChartData.padding.left}
                    y1={histogramChartData.padding.top + histogramChartData.chartHeight}
                    x2={histogramChartData.padding.left + histogramChartData.chartWidth}
                    y2={histogramChartData.padding.top + histogramChartData.chartHeight}
                    stroke="#374151"
                    strokeWidth="2"
                  />

                  {/* Y축 눈금 및 라벨 */}
                  {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
                    const y = histogramChartData.padding.top + histogramChartData.chartHeight - (ratio * histogramChartData.chartHeight);
                    const value = Math.round(ratio * histogramChartData.maxCount);
                    return (
                      <g key={ratio}>
                        <line
                          x1={histogramChartData.padding.left - 5}
                          y1={y}
                          x2={histogramChartData.padding.left}
                          y2={y}
                          stroke="#6b7280"
                          strokeWidth="1"
                        />
                        <text
                          x={histogramChartData.padding.left - 10}
                          y={y + 4}
                          textAnchor="end"
                          fontSize="11"
                          fill="#6b7280"
                        >
                          {value.toLocaleString('ko-KR')}
                        </text>
                      </g>
                    );
                  })}

                  {/* X축 눈금 및 라벨 (간격 조정) */}
                  {(() => {
                    const tickCount = Math.min(10, histogramChartData.bars.length);
                    const tickInterval = Math.max(1, Math.floor(histogramChartData.bars.length / tickCount));
                    return histogramChartData.bars
                      .filter((_, index) => index % tickInterval === 0 || index === histogramChartData.bars.length - 1)
                      .map((bar, idx) => {
                        const x = bar.x + bar.width / 2;
                        const y = histogramChartData.padding.top + histogramChartData.chartHeight + 20;
                        return (
                          <g key={idx}>
                            <line
                              x1={x}
                              y1={histogramChartData.padding.top + histogramChartData.chartHeight}
                              x2={x}
                              y2={histogramChartData.padding.top + histogramChartData.chartHeight + 5}
                              stroke="#6b7280"
                              strokeWidth="1"
                            />
                            <text
                              x={x}
                              y={y}
                              textAnchor="middle"
                              fontSize="10"
                              fill="#6b7280"
                              transform={`rotate(-45 ${x} ${y})`}
                            >
                              {bar.amount.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </text>
                          </g>
                        );
                      });
                  })()}

                  {/* 히스토그램 막대 */}
                  {histogramChartData.bars.map((bar, index) => (
                    <g key={index}>
                      <rect
                        x={bar.x}
                        y={bar.y}
                        width={bar.width}
                        height={bar.height}
                        fill="#3b82f6"
                        stroke="#1e40af"
                        strokeWidth="1"
                        className="hover:opacity-80 cursor-pointer"
                      >
                        <title>
                          Amount: {bar.amount.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                          {'\n'}Count: {bar.count.toLocaleString('ko-KR')}
                        </title>
                      </rect>
                    </g>
                  ))}

                  {/* Y축 라벨 */}
                  <text
                    x={15}
                    y={histogramChartData.padding.top + histogramChartData.chartHeight / 2}
                    textAnchor="middle"
                    fontSize="12"
                    fill="#374151"
                    transform={`rotate(-90 15 ${histogramChartData.padding.top + histogramChartData.chartHeight / 2})`}
                  >
                    Count
                  </text>

                  {/* X축 라벨 */}
                  <text
                    x={histogramChartData.padding.left + histogramChartData.chartWidth / 2}
                    y={histogramChartData.height - 10}
                    textAnchor="middle"
                    fontSize="12"
                    fill="#374151"
                  >
                    Amount
                  </text>
                </svg>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No histogram data available.</p>
            )}
          </div>
        </main>
      </div>
      {showSpreadView && spreadViewData.length > 0 && (
        <SpreadViewModal
          onClose={() => setShowSpreadView(false)}
          data={spreadViewData}
          columns={spreadViewColumns}
          title={`Spread View: ${module.name} - Simulation Results`}
        />
      )}
    </div>
  );
};

