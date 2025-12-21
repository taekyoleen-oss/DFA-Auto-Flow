import React, { useRef, useState, useMemo } from 'react';
import { CanvasModule, SplitFreqServOutput } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { useCopyOnCtrlC } from '../hooks/useCopyOnCtrlC';
import { SpreadViewModal } from './SpreadViewModal';

interface SplitFreqServPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
}

export const SplitFreqServPreviewModal: React.FC<SplitFreqServPreviewModalProps> = ({ 
  module, 
  onClose 
}) => {
  const output = module.outputData as SplitFreqServOutput;
  const viewDetailsRef = useRef<HTMLDivElement>(null);
  useCopyOnCtrlC(viewDetailsRef);
  const [showSpreadView, setShowSpreadView] = useState(false);
  const [spreadViewTab, setSpreadViewTab] = useState<'frequency' | 'severity' | 'amount'>('frequency');
  const [activeTab, setActiveTab] = useState<'frequency' | 'severity'>('frequency');

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    if (spreadViewTab === 'frequency') {
      return output.yearlyFrequency.map(item => ({
        Year: item.year,
        Count: item.count,
      }));
    } else if (spreadViewTab === 'severity') {
      return output.yearlySeverity.map(item => ({
        Year: item.year,
        'Total Amount': item.totalAmount,
        Count: item.count,
        'Mean Amount': item.meanAmount,
      }));
    } else {
      // amount: 보험금 데이터
      return output.severityData.rows.map((row, idx) => {
        const amountColumn = output.severityData.columns[0]?.name || 'amount';
        return {
          Index: idx + 1,
          [amountColumn]: row[amountColumn],
        };
      });
    }
  }, [output, spreadViewTab]);

  const spreadViewColumns = useMemo(() => {
    if (spreadViewTab === 'frequency') {
      return [
        { name: 'Year', type: 'number' },
        { name: 'Count', type: 'number' },
      ];
    } else if (spreadViewTab === 'severity') {
      return [
        { name: 'Year', type: 'number' },
        { name: 'Total Amount', type: 'number' },
        { name: 'Count', type: 'number' },
        { name: 'Mean Amount', type: 'number' },
      ];
    } else {
      // amount: 보험금 데이터
      const amountColumn = output.severityData.columns[0] || { name: 'amount', type: 'number' };
      return [
        { name: 'Index', type: 'number' },
        { name: amountColumn.name, type: amountColumn.type },
      ];
    }
  }, [output, spreadViewTab]);

  if (!output || output.type !== 'SplitFreqServOutput') {
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
          <h2 className="text-xl font-bold text-gray-800">Split By Freq-Sev: {module.name}</h2>
          <div className="flex items-center gap-2">
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
              <XCircleIcon className="w-6 h-6" />
            </button>
          </div>
        </header>
        <main className="flex-grow p-6 overflow-auto flex flex-col" ref={viewDetailsRef}>
          {/* Tabs */}
          <div className="flex gap-2 mb-4 border-b border-gray-200 flex-shrink-0">
            <button
              onClick={() => setActiveTab('frequency')}
              className={`px-4 py-2 font-medium text-sm transition-colors ${
                activeTab === 'frequency'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Frequency
            </button>
            <button
              onClick={() => setActiveTab('severity')}
              className={`px-4 py-2 font-medium text-sm transition-colors ${
                activeTab === 'severity'
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Severity
            </button>
          </div>
          <div className="flex-grow overflow-auto">
          {activeTab === 'frequency' ? (
          <div className="space-y-6">
            {/* Yearly Frequency */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Yearly Frequency (연도별 빈도)</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setSpreadViewTab('frequency');
                      setShowSpreadView(true);
                    }}
                    className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                    </svg>
                    Spread View
                  </button>
                  <button
                    onClick={() => {
                      const csvContent = [
                        'Year,Count',
                        ...output.yearlyFrequency.map(item => `${item.year},${item.count}`)
                      ].join('\n');
                      // BOM 추가하여 한글 인코딩 문제 해결
                      const bom = '\uFEFF';
                      const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                      const link = document.createElement('a');
                      link.href = URL.createObjectURL(blob);
                      link.download = `${module.name}_frequency.csv`;
                      link.click();
                    }}
                    className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                    title="Download CSV"
                  >
                    <ArrowDownTrayIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
              {(() => {
                // 통계량 계산
                const counts = output.yearlyFrequency.map(item => item.count).sort((a, b) => a - b);
                const n = counts.length;
                const mean = counts.reduce((sum, val) => sum + val, 0) / n;
                const variance = counts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
                const stdDev = Math.sqrt(variance);
                const min = counts[0];
                const max = counts[n - 1];
                
                // 중앙값 계산
                const median = n % 2 === 0
                  ? (counts[n / 2 - 1] + counts[n / 2]) / 2
                  : counts[Math.floor(n / 2)];
                
                // 왜도 계산 (3차 모멘트 기반)
                const skewness = n > 2 && stdDev > 0
                  ? counts.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 3), 0) / n
                  : 0;
                
                // 첨도 계산 (4차 모멘트 기반, excess kurtosis)
                const kurtosis = n > 3 && stdDev > 0
                  ? (counts.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 4), 0) / n) - 3
                  : 0;

                return (
                  <div className="flex gap-2" style={{ maxHeight: '60vh' }}>
                    {/* 왼쪽: 테이블 */}
                    <div className="flex-1 flex flex-col gap-1 min-w-0">
                      <div className="flex-1 overflow-auto rounded-lg">
                <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Year</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Count</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {output.yearlyFrequency.map((item, idx) => (
                      <tr key={idx}>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.year}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.count.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
                    </div>

                    {/* 오른쪽: 통계량 */}
                    <div className="w-96 flex-shrink-0">
                      <div className="h-full rounded-lg p-3 overflow-auto">
                        <h3 className="text-sm font-semibold text-gray-800 mb-3">통계량</h3>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">평균:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {mean.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">분산:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {variance.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">표준편차:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {stdDev.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">중앙값:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {median.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">최소값:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {min.toLocaleString('ko-KR')}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">최대값:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {max.toLocaleString('ko-KR')}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">왜도:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {skewness.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-gray-500">첨도:</span>
                            <span className="font-mono text-sm text-gray-800 font-semibold">
                              {kurtosis.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
            </div>
          ) : (
            <div className="space-y-6">
            {/* Yearly Severity */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Yearly Severity (연도별 심도)</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setSpreadViewTab('severity');
                      setShowSpreadView(true);
                    }}
                    className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                    </svg>
                    Spread View
                  </button>
                  <button
                    onClick={() => {
                      const csvContent = [
                        'Year,Total Amount,Count,Mean Amount',
                        ...output.yearlySeverity.map(item => 
                          `${item.year},${item.totalAmount},${item.count},${item.meanAmount}`
                        )
                      ].join('\n');
                      // BOM 추가하여 한글 인코딩 문제 해결
                      const bom = '\uFEFF';
                      const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                      const link = document.createElement('a');
                      link.href = URL.createObjectURL(blob);
                      link.download = `${module.name}_severity.csv`;
                      link.click();
                    }}
                    className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                    title="Download CSV"
                  >
                    <ArrowDownTrayIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Year</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Amount</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Count</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Mean Amount</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {output.yearlySeverity.map((item, idx) => (
                      <tr key={idx}>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.year}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.totalAmount.toLocaleString()}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.count.toLocaleString()}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.meanAmount.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Amount (보험금) - 연도 없이 개별 보험금 데이터 */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Amount (보험금)</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      setSpreadViewTab('amount');
                      setShowSpreadView(true);
                    }}
                    className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                    </svg>
                    Spread View
                  </button>
                  <button
                    onClick={() => {
                      const amountColumn = output.severityData.columns[0]?.name || 'amount';
                      const csvContent = [
                        amountColumn,
                        ...output.severityData.rows.map(row => {
                          const val = row[amountColumn];
                          return val === null || val === undefined ? '' : String(val);
                        })
                      ].join('\n');
                      // BOM 추가하여 한글 인코딩 문제 해결
                      const bom = '\uFEFF';
                      const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                      const link = document.createElement('a');
                      link.href = URL.createObjectURL(blob);
                      link.download = `${module.name}_amount.csv`;
                      link.click();
                    }}
                    className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                    title="Download CSV"
                  >
                    <ArrowDownTrayIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {output.severityData.columns.map((col, idx) => (
                        <th key={idx} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          {col.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {output.severityData.rows.slice(0, 100).map((row, idx) => (
                      <tr key={idx}>
                        {output.severityData.columns.map((col, colIdx) => (
                          <td key={colIdx} className="px-4 py-3 text-sm text-gray-900">
                            {typeof row[col.name] === 'number' 
                              ? row[col.name].toLocaleString() 
                              : row[col.name]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {output.severityData.rows.length > 100 && (
                  <div className="px-4 py-3 text-sm text-gray-500 text-center">
                    Showing first 100 rows of {output.severityData.rows.length} total rows
                  </div>
                )}
              </div>
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
          title={`Spread View: ${module.name} - ${
            spreadViewTab === 'frequency' ? 'Frequency' : 
            spreadViewTab === 'severity' ? 'Severity' : 
            'Amount'
          }`}
        />
      )}
    </div>
  );
};

