import React from 'react';
import { CanvasModule, SimulateAggDistOutput } from '../types';
import { XCircleIcon } from './icons';

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
              onClick={() => {
                if (!results || results.length === 0) return;
                const csvContent = [
                  'count,amount',
                  ...results.map(r => `${r.count},${r.amount}`)
                ].join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
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
        <main className="flex-grow p-6 overflow-auto">
          <div className="space-y-6">
            {/* Simulation Info */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Simulation Information</h3>
              <p className="text-gray-600">
                Total Simulations: <span className="font-semibold">{simulationCount.toLocaleString('ko-KR')}</span>
              </p>
            </div>

            {/* Statistics */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Statistics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">Mean</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.mean.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">Std Dev</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.std.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">Min</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.min.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">Max</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.max.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">5th Percentile</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile5.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">25th Percentile</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile25.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">50th Percentile (Median)</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile50.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">75th Percentile</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile75.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">95th Percentile</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile95.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-xs text-gray-500 mb-1">99th Percentile</span>
                  <span className="font-mono text-sm text-gray-800 font-semibold">
                    {statistics.percentile99.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
            </div>

            {/* Results Table */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Simulation Results (Count & Amount)</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-right font-semibold text-gray-700">Count</th>
                      <th className="px-4 py-2 text-right font-semibold text-gray-700">Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result, index) => (
                      <tr key={index} className="border-b border-gray-100">
                        <td className="px-4 py-2 text-right font-mono text-gray-800">
                          {result.count.toLocaleString('ko-KR')}
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-gray-800">
                          {result.amount.toLocaleString('ko-KR', { maximumFractionDigits: 2 })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

