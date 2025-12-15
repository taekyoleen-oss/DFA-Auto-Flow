import React, { useRef } from 'react';
import { CanvasModule, SplitFreqServOutput } from '../types';
import { XCircleIcon } from './icons';
import { useCopyOnCtrlC } from '../hooks/useCopyOnCtrlC';

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
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>
        <main className="flex-grow p-6 overflow-auto" ref={viewDetailsRef}>
          <div className="space-y-6">
            {/* Yearly Frequency */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Yearly Frequency (연도별 빈도)</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Year</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Count</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {output.yearlyFrequency.map((item, idx) => (
                      <tr key={idx}>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.year}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Yearly Severity */}
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Yearly Severity (연도별 심도)</h3>
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
                        <td className="px-4 py-3 text-sm text-gray-900">{item.count}</td>
                        <td className="px-4 py-3 text-sm text-gray-900">{item.meanAmount.toLocaleString()}</td>
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

