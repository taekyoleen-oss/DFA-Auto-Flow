import React, { useRef, useEffect, useCallback, useState } from "react";
import { XCircleIcon } from "./icons";
import { CanvasModule } from "../types";
import { XolPricingOutput } from "../types";
import { useCopyOnCtrlC } from "../hooks/useCopyOnCtrlC";

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

const formatNumberNoUnit = (value: number, decimals: number = 0): string => {
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

const formatPercent = (value: number, decimals: number = 1): string => {
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value * 100) + "%";
};

const formatYears = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat("ko-KR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value) + "년";
};

export const XolPricingPreviewModal: React.FC<XolPricingPreviewModalProps> = ({ module, onClose }) => {
  const output = module.outputData as XolPricingOutput;
  if (!output || output.type !== 'XolPricingOutput') return null;

  const contentRef = useRef<HTMLDivElement>(null);
  const [contextMenu, setContextMenu] = React.useState<{ x: number; y: number } | null>(null);
  const [activeTab, setActiveTab] = useState<string>('Summary');
  
  useCopyOnCtrlC(contentRef);
  
  // Calculator 결과 가져오기 (다중 입력 지원)
  const calculatorResults = output.calculatorResults || [];
  
  // Summary 탭을 위한 데이터 준비
  const summaryData = calculatorResults.map(result => ({
    name: result.calculatorName,
    netPremium: result.netPremium,
    grossPremium: result.grossPremium,
    rol: result.limit > 0 ? result.grossPremium / result.limit : 0,
    paybackPeriod: result.limit > 0 && result.grossPremium > 0 ? result.limit / result.grossPremium : 0,
  }));

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const handleCopy = useCallback(async () => {
    if (contentRef.current) {
      const text = contentRef.current.innerText || contentRef.current.textContent || '';
      if (text.trim()) {
        try {
          await navigator.clipboard.writeText(text);
          setContextMenu(null);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      }
    }
  }, []);

  // 컨텍스트 메뉴 외부 클릭 시 닫기
  useEffect(() => {
    if (!contextMenu) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Element;
      if (contextMenu && !target.closest('.context-menu')) {
        setContextMenu(null);
      }
    };
    
    document.addEventListener('click', handleClickOutside, true);
    document.addEventListener('mousedown', handleClickOutside, true);
    
    return () => {
      document.removeEventListener('click', handleClickOutside, true);
      document.removeEventListener('mousedown', handleClickOutside, true);
    };
  }, [contextMenu]);

  // 현재 선택된 Calculator 결과 (기본값은 첫 번째)
  const currentResult = calculatorResults.find(r => r.calculatorName === activeTab) || calculatorResults[0] || {
    calculatorName: '',
    calculatorId: '',
    xolClaimMean: 0,
    xolClaimStdDev: 0,
    xolPremiumRateMean: 0,
    reluctanceFactor: 0,
    expenseRate: 0,
    netPremium: 0,
    grossPremium: 0,
    limit: 0,
    deductible: 0,
    reinstatements: 0,
    aggDeductible: 0,
    reinstatementPremiums: [],
  };

  const { 
    xolClaimMean, 
    xolClaimStdDev, 
    xolPremiumRateMean, 
    reluctanceFactor, 
    expenseRate, 
    netPremium, 
    grossPremium,
    limit,
    deductible,
    reinstatements,
    aggDeductible,
    reinstatementPremiums
  } = currentResult;

  // ROL 계산
  const rol = limit > 0 ? grossPremium / limit : 0;
  const rolFormatted = limit > 0 ? formatPercent(rol) : "N/A";
  
  // Payback Period 계산
  const paybackPeriod = limit > 0 && grossPremium > 0 ? limit / grossPremium : 0;
  const paybackPeriodFormatted = limit > 0 && grossPremium > 0 ? formatYears(paybackPeriod) : "N/A";

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
        onContextMenu={handleContextMenu}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">XoL Pricing Details: {module.name}</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>
        
        {/* Tabs */}
        <div className="border-b border-gray-200 px-4">
          <div className="flex gap-4 overflow-x-auto">
            <button
              onClick={() => setActiveTab('Summary')}
              className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${
                activeTab === 'Summary'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Summary
            </button>
            {calculatorResults.map((result) => (
              <button
                key={result.calculatorId}
                onClick={() => setActiveTab(result.calculatorName)}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${
                  activeTab === result.calculatorName
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {result.calculatorName}
              </button>
            ))}
          </div>
        </div>
        
        <main ref={contentRef} className="flex-grow p-6 overflow-auto">
          {activeTab === 'Summary' ? (
            <div className="space-y-6">
              {/* Summary Table */}
              <div className="border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Summary</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Calculator</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">순보험료 (Net Premium)</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">영업보험료 (Gross Premium)</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">ROL</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Payback Period</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {summaryData.map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-sm font-medium text-gray-900">{row.name}</td>
                          <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">{formatNumberNoUnit(row.netPremium)}</td>
                          <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">{formatNumberNoUnit(row.grossPremium)}</td>
                          <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">{formatPercent(row.rol)}</td>
                          <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">{formatYears(row.paybackPeriod)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
            {/* XoL 조건 */}
            <div className="bg-gray-100 rounded-lg p-3 border border-gray-300">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                Limit Xs Deductible Condition: {formatNumberNoUnit(limit)} Xs {formatNumberNoUnit(deductible)}
              </h3>
              <div className="text-xs text-gray-600 space-y-1">
                <div>복원횟수: {reinstatements}회</div>
                {reinstatementPremiums && reinstatementPremiums.length > 0 && (
                  <div>
                    복원비율: {reinstatementPremiums.map((rate, idx) => 
                      `${idx + 1}차 ${rate}%`
                    ).join(", ")}
                  </div>
                )}
                {aggDeductible > 0 && (
                  <div>Agg Ded: {formatNumberNoUnit(aggDeductible)}</div>
                )}
              </div>
            </div>

            {/* 입력값 표시 */}
            <div className="border border-gray-200 rounded-lg p-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">입력값</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">XoL Claim 평균:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatNumberNoUnit(xolClaimMean)}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-gray-600">XoL Claim 표준편차:</span>
                  <span className="font-mono text-gray-800 font-medium">{formatNumberNoUnit(xolClaimStdDev)}</span>
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
                      ({formatNumberNoUnit(xolClaimMean)} / {formatNumber(xolPremiumRateMean)}) + {formatNumberNoUnit(xolClaimStdDev)} × {formatNumber(reluctanceFactor)}
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-gray-600">Net Premium:</span>
                    <span className="font-mono text-gray-800 font-medium">{formatNumberNoUnit(netPremium)}</span>
                  </div>
                </div>
                
                {/* Gross Premium */}
                <div className="space-y-2">
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-semibold">Gross Premium 수식:</span>
                    <div className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-xs">
                      {formatNumberNoUnit(netPremium)} / (1 - {formatNumber(expenseRate)})
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-3 mt-2 bg-blue-50 rounded-md px-3">
                    <span className="font-bold text-blue-800">Gross Premium:</span>
                    <span className="font-mono text-blue-800 font-bold text-lg">{formatNumberNoUnit(grossPremium)}</span>
                  </div>
                </div>

                {/* ROL */}
                <div className="space-y-2 mt-4">
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-semibold">ROL 수식:</span>
                    <div className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-xs">
                      {formatNumberNoUnit(grossPremium)} / {formatNumberNoUnit(limit)}
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-gray-600">ROL:</span>
                    <span className="font-mono text-gray-800 font-medium">{rolFormatted}</span>
                  </div>
                </div>

                {/* Payback Period */}
                <div className="space-y-2 mt-4">
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-semibold">Payback Period 수식:</span>
                    <div className="mt-1 p-2 bg-gray-50 rounded border border-gray-200 font-mono text-xs">
                      1 / {rolFormatted}
                    </div>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b">
                    <span className="text-gray-600">Payback Period:</span>
                    <span className="font-mono text-gray-800 font-medium">{paybackPeriodFormatted}</span>
                  </div>
                </div>
              </div>
            </div>
            </div>
          )}
        </main>
      </div>
      {contextMenu && (
        <div
          className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleCopy();
            }}
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer focus:outline-none focus:bg-gray-100"
          >
            복사 (Ctrl+C)
          </button>
        </div>
      )}
    </div>
  );
};

