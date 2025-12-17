import React from "react";
import { XCircleIcon } from "./icons";
import { CanvasModule } from "../types";
import { XolPricingOutput } from "../types";

interface XolPricingPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
}

const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat("ko-KR", {
    style: "currency",
    currency: "KRW",
    maximumFractionDigits: 0,
  }).format(value);
};

const formatNumber = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const XolPricingPreviewModal: React.FC<XolPricingPreviewModalProps> = ({ module, onClose }) => {
  const output = module.outputData as XolPricingOutput;
  if (!output || output.type !== 'XolPricingOutput') return null;

  const { 
    xolClaimMean, 
    xolClaimStdDev, 
    xolPremiumRateMean, 
    reluctanceFactor, 
    expenseRate, 
    netPremium, 
    grossPremium 
  } = output;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">XoL Pricing Details: {module.name}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>
        <main className="flex-grow p-6 overflow-auto">
          <div className="space-y-6">
            {/* 입력값 표시 */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">입력값</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">XoL Claim 평균:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatCurrency(xolClaimMean)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">XoL Claim 표준편차:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatCurrency(xolClaimStdDev)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">XoL Premium Rate 평균:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatNumber(xolPremiumRateMean)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">Reluctance Factor:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatNumber(reluctanceFactor)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">Expense Rate:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatNumber(expenseRate)}</span>
                </div>
              </div>
            </div>

            {/* 계산 결과 */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">계산 결과</h3>
              <div className="space-y-4">
                {/* Net Premium */}
                <div className="space-y-2">
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-semibold">Net Premium 수식:</span>
                    <div className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-xs">
                      ({formatCurrency(xolClaimMean)} / {formatNumber(xolPremiumRateMean)}) + {formatCurrency(xolClaimStdDev)} × {formatNumber(reluctanceFactor)}
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-gray-600">Net Premium:</span>
                    <span className="font-mono text-gray-800 font-medium">{formatCurrency(netPremium)}</span>
                  </div>
                </div>
                
                {/* Gross Premium */}
                <div className="space-y-2">
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-semibold">Gross Premium 수식:</span>
                    <div className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-xs">
                      {formatCurrency(netPremium)} / (1 - {formatNumber(expenseRate)})
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-3 mt-2 bg-blue-50 rounded-md px-3">
                    <span className="font-bold text-blue-800">Gross Premium:</span>
                    <span className="font-mono text-blue-800 font-bold text-lg">{formatCurrency(grossPremium)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

