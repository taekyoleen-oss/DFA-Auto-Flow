import React from 'react';
import { CanvasModule, EvaluationOutput } from '../types';
import { XCircleIcon } from './icons';

interface EvaluationPreviewModalProps {
    module: CanvasModule;
    onClose: () => void;
}

export const EvaluationPreviewModal: React.FC<EvaluationPreviewModalProps> = ({ module, onClose }) => {
    const output = module.outputData as EvaluationOutput;
    if (!output || output.type !== 'EvaluationOutput') return null;

    const { modelType, metrics } = output;

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-lg max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <div className="flex flex-col">
                        <h2 className="text-xl font-bold text-gray-800">Evaluation Results: {module.name}</h2>
                        <p className="text-sm text-gray-500">Model Type: <span className="capitalize">{modelType}</span></p>
                    </div>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                <main className="flex-grow p-6 overflow-auto">
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-gray-700 mb-4 text-center">
                            Performance Metrics
                        </h3>
                        <div className="space-y-3 text-base">
                            {Object.entries(metrics).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center py-2 border-b last:border-b-0">
                                    <span className="text-gray-600">{key}:</span>
                                    <span className="font-mono text-gray-800 font-medium">
                                        {/* fix: The value of a metric can be a string. Added a type check to prevent calling .toFixed on a non-number. */}
                                        {typeof value === 'number' ? value.toFixed(4) : value}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
};