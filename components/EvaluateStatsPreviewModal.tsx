import React from 'react';
import { CanvasModule, EvaluateStatsOutput } from '../types';
import { XCircleIcon } from './icons';

interface EvaluateStatsPreviewModalProps {
    module: CanvasModule;
    onClose: () => void;
}

export const EvaluateStatsPreviewModal: React.FC<EvaluateStatsPreviewModalProps> = ({ 
    module, 
    onClose
}) => {
    const output = module.outputData as EvaluateStatsOutput;
    if (!output || output.type !== 'EvaluateStatsOutput') return null;

    const { modelType, metrics, actualColumn, predictedColumn, residuals, deviance, pearsonChi2, dispersion, aic, bic, logLikelihood } = output;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl w-[90vw] max-w-6xl max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center z-10">
                    <h2 className="text-xl font-semibold text-gray-800">
                        Evaluate Stats - {module.name}
                    </h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {/* Model Info */}
                    <div className="mb-6">
                        <h3 className="text-lg font-semibold text-gray-700 mb-3">Model Information</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <span className="text-sm text-gray-600">Model Type:</span>
                                <span className="ml-2 font-medium">{modelType}</span>
                            </div>
                            <div>
                                <span className="text-sm text-gray-600">Actual Column:</span>
                                <span className="ml-2 font-medium">{actualColumn}</span>
                            </div>
                            <div>
                                <span className="text-sm text-gray-600">Predicted Column:</span>
                                <span className="ml-2 font-medium">{predictedColumn}</span>
                            </div>
                        </div>
                    </div>

                    {/* Basic Metrics */}
                    <div className="mb-6">
                        <h3 className="text-lg font-semibold text-gray-700 mb-3">Basic Metrics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {Object.entries(metrics).map(([key, value]) => (
                                <div key={key} className="bg-gray-50 p-3 rounded">
                                    <div className="text-xs text-gray-600 mb-1">{key}</div>
                                    <div className="text-lg font-semibold text-gray-800">
                                        {typeof value === 'number' ? value.toFixed(6) : String(value)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* GLM Statistics */}
                    {(deviance !== undefined || pearsonChi2 !== undefined || dispersion !== undefined || aic !== undefined || bic !== undefined || logLikelihood !== undefined) && (
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">GLM Statistics</h3>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                {deviance !== undefined && (
                                    <div className="bg-blue-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">Deviance</div>
                                        <div className="text-lg font-semibold text-blue-800">
                                            {deviance.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                                {pearsonChi2 !== undefined && (
                                    <div className="bg-blue-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">Pearson chi²</div>
                                        <div className="text-lg font-semibold text-blue-800">
                                            {pearsonChi2.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                                {dispersion !== undefined && (
                                    <div className="bg-blue-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">Dispersion (φ)</div>
                                        <div className="text-lg font-semibold text-blue-800">
                                            {dispersion.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                                {aic !== undefined && (
                                    <div className="bg-green-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">AIC</div>
                                        <div className="text-lg font-semibold text-green-800">
                                            {aic.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                                {bic !== undefined && (
                                    <div className="bg-green-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">BIC</div>
                                        <div className="text-lg font-semibold text-green-800">
                                            {bic.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                                {logLikelihood !== undefined && (
                                    <div className="bg-green-50 p-3 rounded">
                                        <div className="text-xs text-gray-600 mb-1">Log-Likelihood</div>
                                        <div className="text-lg font-semibold text-green-800">
                                            {logLikelihood.toFixed(6)}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Residuals Statistics */}
                    {residuals && residuals.length > 0 && (
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-700 mb-3">Residuals Statistics</h3>
                            <div className="bg-gray-50 p-4 rounded">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div>
                                        <div className="text-xs text-gray-600 mb-1">Count</div>
                                        <div className="text-lg font-semibold text-gray-800">
                                            {residuals.length}
                                        </div>
                                    </div>
                                    {metrics['Mean Residual'] !== undefined && (
                                        <div>
                                            <div className="text-xs text-gray-600 mb-1">Mean</div>
                                            <div className="text-lg font-semibold text-gray-800">
                                                {typeof metrics['Mean Residual'] === 'number' 
                                                    ? metrics['Mean Residual'].toFixed(6) 
                                                    : String(metrics['Mean Residual'])}
                                            </div>
                                        </div>
                                    )}
                                    {metrics['Std Residual'] !== undefined && (
                                        <div>
                                            <div className="text-xs text-gray-600 mb-1">Std Dev</div>
                                            <div className="text-lg font-semibold text-gray-800">
                                                {typeof metrics['Std Residual'] === 'number' 
                                                    ? metrics['Std Residual'].toFixed(6) 
                                                    : String(metrics['Std Residual'])}
                                            </div>
                                        </div>
                                    )}
                                    {metrics['Min Residual'] !== undefined && (
                                        <div>
                                            <div className="text-xs text-gray-600 mb-1">Min</div>
                                            <div className="text-lg font-semibold text-gray-800">
                                                {typeof metrics['Min Residual'] === 'number' 
                                                    ? metrics['Min Residual'].toFixed(6) 
                                                    : String(metrics['Min Residual'])}
                                            </div>
                                        </div>
                                    )}
                                    {metrics['Max Residual'] !== undefined && (
                                        <div>
                                            <div className="text-xs text-gray-600 mb-1">Max</div>
                                            <div className="text-lg font-semibold text-gray-800">
                                                {typeof metrics['Max Residual'] === 'number' 
                                                    ? metrics['Max Residual'].toFixed(6) 
                                                    : String(metrics['Max Residual'])}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Model Type Specific Information */}
                    {modelType === 'Poisson' && (
                        <div className="mb-6 bg-yellow-50 p-4 rounded">
                            <h4 className="font-semibold text-yellow-800 mb-2">Poisson Model Notes</h4>
                            <ul className="text-sm text-yellow-700 space-y-1">
                                <li>• Dispersion (φ) close to 1 indicates good fit</li>
                                <li>• φ &gt; 1 suggests overdispersion (consider Negative Binomial)</li>
                                <li>• Deviance should be compared to degrees of freedom</li>
                            </ul>
                        </div>
                    )}

                    {modelType === 'NegativeBinomial' && (
                        <div className="mb-6 bg-yellow-50 p-4 rounded">
                            <h4 className="font-semibold text-yellow-800 mb-2">Negative Binomial Model Notes</h4>
                            <ul className="text-sm text-yellow-700 space-y-1">
                                <li>• Suitable for overdispersed count data</li>
                                <li>• Lower AIC/BIC compared to Poisson indicates better fit</li>
                                <li>• Dispersion parameter accounts for extra-Poisson variation</li>
                            </ul>
                        </div>
                    )}

                    {modelType === 'QuasiPoisson' && (
                        <div className="mb-6 bg-yellow-50 p-4 rounded">
                            <h4 className="font-semibold text-yellow-800 mb-2">Quasi-Poisson Model Notes</h4>
                            <ul className="text-sm text-yellow-700 space-y-1">
                                <li>• Handles overdispersion by scaling variance</li>
                                <li>• Dispersion (φ) adjusts standard errors</li>
                                <li>• Useful when Negative Binomial is computationally expensive</li>
                            </ul>
                        </div>
                    )}

                    {modelType === 'Logistic' && (
                        <div className="mb-6 bg-yellow-50 p-4 rounded">
                            <h4 className="font-semibold text-yellow-800 mb-2">Logistic Model Notes</h4>
                            <ul className="text-sm text-yellow-700 space-y-1">
                                <li>• Deviance measures model fit (lower is better)</li>
                                <li>• AIC/BIC for model comparison</li>
                                <li>• Pearson chi² tests goodness of fit</li>
                            </ul>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-6 py-4 flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

