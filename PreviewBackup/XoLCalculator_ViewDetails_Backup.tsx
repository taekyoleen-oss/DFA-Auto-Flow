/**
 * XoL Calculator View Details 백업 파일
 *
 * 이 파일은 DataPreviewModal.tsx의 XoL Calculator View Details 관련 코드를 백업한 것입니다.
 * XoL Calculator의 View Details가 의도치 않게 수정될 경우 이 파일을 참조하여 복구할 수 있습니다.
 *
 * 백업 일자: 2025-01-17 (최신 업데이트)
 *
 * 복구 방법:
 * 1. DataPreviewModal.tsx에서 XoL Calculator 관련 코드를 찾습니다.
 * 2. 이 파일의 내용으로 교체합니다.
 *
 * 포함된 내용:
 * - YearlyAmountBarPlot 컴포넌트
 * - getXolData 함수
 * - XoL Calculator 탭 및 데이터 표시 UI
 *   - "건별 XoL 적용" 탭: 테이블(10행, 가로 스크롤), 통계량, 히스토그램(보험금, XoL Claim(Incl. Limit))
 *   - "연도별 XoL 적용" 탭: 테이블(text-sm), 통계 정보, 통계량 + 그래프
 */

import React, { useState, useMemo } from "react";
import { CanvasModule, ColumnInfo, DataPreview, ModuleType } from "../types";

// 연도별 금액을 표시하는 Bar Plot 컴포넌트
const YearlyAmountBarPlot: React.FC<{
  rows: Record<string, any>[];
  yearColumn: string;
  amountColumn: string;
}> = ({ rows, yearColumn, amountColumn }) => {
  const data = useMemo(() => {
    return rows
      .map((row) => ({
        year: row[yearColumn],
        amount: parseFloat(row[amountColumn]) || 0,
      }))
      .filter((d) => !isNaN(d.amount));
  }, [rows, yearColumn, amountColumn]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        No data available to plot.
      </div>
    );
  }

  const maxAmount = Math.max(...data.map((d) => d.amount), 0);
  const sortedData = [...data].sort((a, b) => a.year - b.year);

  return (
    <div className="w-full h-full p-4 flex flex-col border border-gray-200 rounded-lg">
      <div className="flex-grow flex items-center gap-2 overflow-hidden">
        {/* Y-axis Label */}
        <div className="flex items-center justify-center h-full">
          <p className="text-sm text-gray-600 font-semibold transform -rotate-90 whitespace-nowrap">
            Amount
          </p>
        </div>

        {/* Plot area */}
        <div className="flex-grow h-full flex flex-col">
          <div className="flex-grow flex items-end justify-around gap-2 pt-4">
            {sortedData.map((d, index) => {
              const heightPercentage =
                maxAmount > 0 ? (d.amount / maxAmount) * 100 : 0;
              return (
                <div
                  key={index}
                  className="flex-1 h-full flex flex-col justify-end items-center group relative"
                  title={`Year: ${
                    d.year
                  }, Amount: ${d.amount.toLocaleString()}`}
                >
                  <span className="absolute -top-5 text-xs bg-gray-800 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                    {d.amount.toLocaleString()}
                  </span>
                  <div
                    className="w-full bg-green-500 hover:bg-green-600 transition-colors"
                    style={{ height: `${heightPercentage}%` }}
                  ></div>
                </div>
              );
            })}
          </div>
          {/* X-axis Labels */}
          <div className="w-full flex justify-around gap-2 text-center text-xs text-gray-600 font-semibold mt-2 border-t pt-1">
            {sortedData.map((d, index) => (
              <div key={index} className="flex-1">
                {d.year}
              </div>
            ))}
          </div>
          {/* X-axis Label */}
          <div className="w-full text-center text-sm text-gray-600 font-semibold mt-1">
            {yearColumn}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * XoL Calculator 모듈의 경우 데이터 필터링 함수
 *
 * @param data - 원본 데이터
 * @param module - XoL Calculator 모듈
 * @param activeXolTab - 활성화된 탭 ('limit' | 'aggreinst')
 * @param allModules - 모든 모듈 배열
 * @param allConnections - 모든 연결 배열
 * @returns 필터링된 데이터
 */
export const getXolData = (
  data: DataPreview | null,
  module: CanvasModule,
  activeXolTab: "limit" | "aggreinst",
  allModules: CanvasModule[],
  allConnections: Array<{
    from: { moduleId: string; portName: string };
    to: { moduleId: string; portName: string };
  }>
): DataPreview | null => {
  if (!data || module.type !== ModuleType.XolCalculator) return data;

  const columnName =
    activeXolTab === "limit"
      ? "XoL Claim(Incl. Limit)"
      : "XoL Claim(Incl. Agg/Reinst)";

  // 두 번째 탭 (aggreinst)인 경우 연도별 집계
  if (activeXolTab === "aggreinst") {
    // XoL Contract 모듈 정보 가져오기
    const contractConnection = allConnections.find(
      (c) => c.to.moduleId === module.id && c.to.portName === "contract_in"
    );
    const contractModule = contractConnection
      ? allModules.find((m) => m.id === contractConnection.from.moduleId)
      : null;
    const contract =
      contractModule?.outputData?.type === "XolContractOutput"
        ? (contractModule.outputData as any)
        : null;

    // 연도 컬럼 찾기
    const yearColumn = data.columns.find(
      (col) =>
        col.name === "연도" ||
        col.name === "year" ||
        col.name.toLowerCase() === "year" ||
        col.name.toLowerCase().includes("year")
    );

    // 클레임 금액_infl 컬럼 찾기
    const claimAmountColumn = data.columns.find(
      (col) =>
        col.name === "클레임 금액_infl" ||
        col.name.endsWith("_infl") ||
        (col.name.includes("금액") && col.name.includes("infl"))
    );

    // XoL Claim(Incl. Limit) 컬럼 찾기
    const xolLimitColumn = data.columns.find(
      (col) => col.name === "XoL Claim(Incl. Limit)"
    );

    if (!yearColumn || !data.rows || data.rows.length === 0) {
      // 연도 컬럼이 없으면 빈 데이터 반환
      return {
        ...data,
        columns: [],
        rows: [],
        totalRowCount: 0,
      };
    }

    // 연도별 집계: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit)
    const yearMap = new Map<
      number,
      { claimAmount: number; xolLimit: number }
    >();
    const yearColumnName = yearColumn.name;
    const claimAmountColumnName = claimAmountColumn?.name || null;
    const xolLimitColumnName = xolLimitColumn?.name || null;

    data.rows.forEach((row) => {
      const year = row[yearColumnName];

      if (year !== null && year !== undefined) {
        const yearNum =
          typeof year === "number" ? year : parseInt(String(year));

        if (!isNaN(yearNum)) {
          const current = yearMap.get(yearNum) || {
            claimAmount: 0,
            xolLimit: 0,
          };

          // 클레임 금액_infl 합계
          if (claimAmountColumnName) {
            const claimValue = row[claimAmountColumnName];
            if (claimValue !== null && claimValue !== undefined) {
              const claimNum =
                typeof claimValue === "number"
                  ? claimValue
                  : parseFloat(String(claimValue));
              if (!isNaN(claimNum)) {
                current.claimAmount += claimNum;
              }
            }
          }

          // XoL Claim(Incl. Limit) 합계
          if (xolLimitColumnName) {
            const limitValue = row[xolLimitColumnName];
            if (limitValue !== null && limitValue !== undefined) {
              const limitNum =
                typeof limitValue === "number"
                  ? limitValue
                  : parseFloat(String(limitValue));
              if (!isNaN(limitNum)) {
                current.xolLimit += limitNum;
              }
            }
          }

          yearMap.set(yearNum, current);
        }
      }
    });

    // 집계된 데이터를 행 형태로 변환하고 XoL Claim(Incl. Agg/Reinst) 계산
    const aggregatedRows = Array.from(yearMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([year, totals]) => {
        const row: Record<string, any> = {
          [yearColumnName]: year,
        };
        if (claimAmountColumnName) {
          row[claimAmountColumnName] = totals.claimAmount;
        }
        if (xolLimitColumnName) {
          row[xolLimitColumnName] = totals.xolLimit;
        }

        // XoL Claim(Incl. Agg/Reinst) 계산
        let xolClaimAggReinst = 0;
        if (contract && xolLimitColumnName) {
          const maxValue =
            contract.limit * (contract.reinstatements + 1) +
            contract.aggDeductible;
          xolClaimAggReinst =
            totals.xolLimit >= maxValue ? maxValue : totals.xolLimit;
          row["XoL Claim(Incl. Agg/Reinst)"] = xolClaimAggReinst;
        }

        // XoL Premium Rate 계산
        if (contract && xolClaimAggReinst > 0) {
          // 1. Agg Deductible 제외
          const afterAggDed = xolClaimAggReinst - contract.aggDeductible;

          if (afterAggDed > 0 && contract.limit > 0) {
            // 2. Limit로 나누기(몇 개의 limit로 나뉘는지)
            const limitCount = afterAggDed / contract.limit;

            // 3. 각 limit 배수에 해당하는 복원 프리미엄 계산
            let totalReinstatementRate = 0;
            const fullLimitCount = Math.floor(limitCount);
            const partialLimit = limitCount - fullLimitCount; // 나머지 부분 limit

            // 전체 limit 배수에 대한 복원 프리미엄
            // 각 limit 배수는 1배, 2배, 3배... 복원에 해당
            for (let i = 0; i < fullLimitCount; i++) {
              if (
                i < contract.reinstatements &&
                contract.reinstatementPremiums &&
                contract.reinstatementPremiums[i] !== undefined
              ) {
                // 각 limit 배수를 limit로 나눈 값(1)에 복원 프리미엄을 곱함
                // 복원 프리미엄은 백분율로 들어오므로 100으로 나눔
                totalReinstatementRate +=
                  1 * (contract.reinstatementPremiums[i] / 100);
              }
            }

            // 나머지 limit에 대한 복원 프리미엄 (나머지 limit)
            if (partialLimit > 0) {
              const reinstatementIndex = fullLimitCount; // 나머지 limit의 복원 인덱스
              if (
                reinstatementIndex < contract.reinstatements &&
                contract.reinstatementPremiums &&
                contract.reinstatementPremiums[reinstatementIndex] !== undefined
              ) {
                // 나머지 limit을 limit로 나눈 값(partialLimit)에 복원 프리미엄을 곱함
                totalReinstatementRate +=
                  partialLimit *
                  (contract.reinstatementPremiums[reinstatementIndex] / 100);
              }
            }

            // 4. 초기 재보험 프리미엄 1에 모든 복원 프리미엄을 더함
            const xolPremiumRate = 1 + totalReinstatementRate;
            row["XoL Premium Rate"] = xolPremiumRate;
          } else {
            // Agg Deductible보다 작으면 0 반환하는 경우
            row["XoL Premium Rate"] = 1;
          }
        }

        return row;
      });

    // 컬럼 정의: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit), XoL Claim(Incl. Agg/Reinst), XoL Premium Rate
    const aggregatedColumns = [{ name: yearColumnName, type: yearColumn.type }];
    if (claimAmountColumn) {
      aggregatedColumns.push({ name: claimAmountColumnName!, type: "number" });
    }
    if (xolLimitColumn) {
      aggregatedColumns.push({ name: xolLimitColumnName!, type: "number" });
    }
    if (contract) {
      aggregatedColumns.push({
        name: "XoL Claim(Incl. Agg/Reinst)",
        type: "number",
      });
      aggregatedColumns.push({ name: "XoL Premium Rate", type: "number" });
    }

    return {
      ...data,
      columns: aggregatedColumns,
      rows: aggregatedRows,
      totalRowCount: aggregatedRows.length,
    };
  }

  // 첫 번째 탭 (limit)인 경우 기본 동작
  const filteredColumns = data.columns.filter(
    (col) => col.name === columnName || !col.name.startsWith("XoL Claim")
  );

  return {
    ...data,
    columns: filteredColumns,
  };
};

/**
 * XoL Calculator View Details UI 컴포넌트
 *
 * 이 컴포넌트는 DataPreviewModal.tsx의 main 섹션 내부에 포함되어야 합니다.
 *
 * 사용 예시:
 * ```tsx
 * {module.type === ModuleType.XolCalculator ? (
 *   <XoLCalculatorViewDetails
 *     module={module}
 *     xolData={xolData}
 *     activeXolTab={activeXolTab}
 *     setActiveXolTab={setActiveXolTab}
 *     allModules={allModules}
 *     allConnections={allConnections}
 *     selectedColumn={selectedColumn}
 *     setSelectedColumn={setSelectedColumn}
 *     selectedColumnData={selectedColumnData}
 *     isSelectedColNumeric={isSelectedColNumeric}
 *     displayColumns={displayColumns}
 *     displayRows={displayRows}
 *     sortedRows={sortedRows}
 *     requestSort={requestSort}
 *     sortConfig={sortConfig}
 *     xolData={xolData}
 *   />
 * ) : (
 *   // 일반 모듈 UI
 * )}
 * ```
 */
export const XoLCalculatorViewDetails: React.FC<{
  module: CanvasModule;
  xolData: DataPreview | null;
  activeXolTab: "limit" | "aggreinst";
  setActiveXolTab: (tab: "limit" | "aggreinst") => void;
  allModules: CanvasModule[];
  allConnections: Array<{
    from: { moduleId: string; portName: string };
    to: { moduleId: string; portName: string };
  }>;
  selectedColumn: string | null;
  setSelectedColumn: (col: string | null) => void;
  selectedColumnData: (string | number | null)[] | null;
  isSelectedColNumeric: boolean;
  displayColumns: ColumnInfo[];
  displayRows: Record<string, any>[];
  sortedRows: Record<string, any>[];
  requestSort: (key: string) => void;
  sortConfig: { key: string; direction: "ascending" | "descending" } | null;
}> = ({
  module,
  xolData,
  activeXolTab,
  setActiveXolTab,
  allModules,
  allConnections,
  selectedColumn,
  setSelectedColumn,
  selectedColumnData,
  isSelectedColNumeric,
  displayColumns,
  displayRows,
  sortedRows,
  requestSort,
  sortConfig,
}) => {
  // ColumnStatistics 컴포넌트는 DataPreviewModal.tsx에 정의되어 있어야 합니다.
  // ChevronUpIcon, ChevronDownIcon도 import되어 있어야 합니다.

  return (
    <>
      {/* 탭 구성 */}
      <div className="flex-shrink-0 border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            onClick={() => setActiveXolTab("limit")}
            className={`${
              activeXolTab === "limit"
                ? "border-indigo-500 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            건별 XoL 적용
          </button>
          <button
            onClick={() => setActiveXolTab("aggreinst")}
            className={`${
              activeXolTab === "aggreinst"
                ? "border-indigo-500 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            연도별 XoL 적용
          </button>
        </nav>
      </div>

      {/* 데이터 표시 */}
      <div className="flex-grow flex flex-col gap-4 overflow-y-auto">
        {!xolData && (
          <div className="text-center text-gray-500 p-4">
            No data available to display.
          </div>
        )}
        {xolData && (
          <>
            {/* 연도별 XoL 적용 탭의 경우 연도별 집계 표시 */}
            {activeXolTab === "aggreinst" ? (
              <>
                {!xolData.rows ||
                xolData.rows.length === 0 ||
                xolData.columns.length === 0 ? (
                  <div className="text-center text-gray-500 p-8">
                    <p className="text-lg font-semibold mb-2">
                      연도별 데이터가 없습니다.
                    </p>
                    <p className="text-sm">
                      연도 컬럼이 없거나 데이터가 없습니다. 입력 데이터에 연도
                      컬럼이 있는지 확인해주세요.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="flex justify-between items-center flex-shrink-0">
                      <div className="text-xs text-gray-600">
                        Showing {Math.min((xolData.rows || []).length, 1000)} of{" "}
                        {xolData.totalRowCount.toLocaleString()} rows and{" "}
                        {xolData.columns.length} columns. Click a column to see
                        details.
                      </div>
                    </div>

                    {/* 1. 테이블 표시 영역 */}
                    <div className="overflow-x-auto overflow-y-auto border border-gray-200 rounded-lg flex-shrink-0">
                      <table
                        className="min-w-full text-sm text-left"
                        style={{ minWidth: "max-content" }}
                      >
                        <thead className="bg-gray-50 sticky top-0">
                          <tr>
                            {xolData.columns.map((col) => (
                              <th
                                key={col.name}
                                className="py-1 px-2 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                onClick={() => setSelectedColumn(col.name)}
                              >
                                <span className="truncate" title={col.name}>
                                  {col.name}
                                </span>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(xolData.rows || [])
                            .slice(0, 1000)
                            .map((row, rowIndex) => (
                              <tr
                                key={rowIndex}
                                className="border-b border-gray-200 last:border-b-0"
                              >
                                {xolData.columns.map((col) => {
                                  const isYearColumn =
                                    col.name === "연도" ||
                                    col.name === "year" ||
                                    col.name.toLowerCase() === "year";
                                  const isNumeric =
                                    col.type === "number" && !isYearColumn;
                                  const value = row[col.name];

                                  let displayValue: string;
                                  if (
                                    value === null ||
                                    value === undefined ||
                                    value === ""
                                  ) {
                                    displayValue = "";
                                  } else if (isNumeric) {
                                    const numValue =
                                      typeof value === "number"
                                        ? value
                                        : parseFloat(String(value));
                                    if (!isNaN(numValue)) {
                                      displayValue = numValue.toLocaleString(
                                        "ko-KR",
                                        {
                                          maximumFractionDigits: 2,
                                          minimumFractionDigits: 0,
                                        }
                                      );
                                    } else {
                                      displayValue = String(value);
                                    }
                                  } else {
                                    displayValue = String(value);
                                  }

                                  return (
                                    <td
                                      key={col.name}
                                      className={`py-1 px-2 font-mono truncate ${
                                        selectedColumn === col.name
                                          ? "bg-blue-100"
                                          : "hover:bg-gray-50 cursor-pointer"
                                      } ${
                                        isNumeric ? "text-right" : "text-left"
                                      }`}
                                      onClick={() =>
                                        setSelectedColumn(col.name)
                                      }
                                      title={String(row[col.name])}
                                    >
                                      {displayValue === "" ? (
                                        <i className="text-gray-400">null</i>
                                      ) : (
                                        displayValue
                                      )}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>

                    {/* 통계 정보: XoL Claim 평균, 표준편차, XoL Premium Rate 평균, Reluctance Factor */}
                    {(() => {
                      // XoL Contract 모듈 정보 가져오기
                      const contractConnection = allConnections.find(
                        (c) =>
                          c.to.moduleId === module.id &&
                          c.to.portName === "contract_in"
                      );
                      const contractModule = contractConnection
                        ? allModules.find(
                            (m) => m.id === contractConnection.from.moduleId
                          )
                        : null;
                      const contract =
                        contractModule?.outputData?.type === "XolContractOutput"
                          ? (contractModule.outputData as any)
                          : null;

                      // XoL Claim(Incl. Agg/Reinst) 컬럼 찾기
                      const xolClaimAggReinstColumn = xolData.columns.find(
                        (col) => col.name === "XoL Claim(Incl. Agg/Reinst)"
                      );

                      // XoL Premium Rate 컬럼 찾기
                      const xolPremiumRateColumn = xolData.columns.find(
                        (col) => col.name === "XoL Premium Rate"
                      );

                      // 통계값 계산
                      let xolClaimMean = 0;
                      let xolClaimStdDev = 0;
                      let xolPremiumMean = 0;

                      if (
                        xolClaimAggReinstColumn &&
                        xolData.rows &&
                        xolData.rows.length > 0
                      ) {
                        const values = xolData.rows
                          .map((row) => {
                            const val = row[xolClaimAggReinstColumn.name];
                            return val !== null && val !== undefined
                              ? typeof val === "number"
                                ? val
                                : parseFloat(String(val))
                              : null;
                          })
                          .filter((v) => v !== null && !isNaN(v)) as number[];

                        if (values.length > 0) {
                          // 평균 계산
                          xolClaimMean =
                            values.reduce((sum, v) => sum + v, 0) /
                            values.length;

                          // 표준편차 계산
                          const variance =
                            values.reduce(
                              (sum, v) => sum + Math.pow(v - xolClaimMean, 2),
                              0
                            ) / values.length;
                          xolClaimStdDev = Math.sqrt(variance);
                        }
                      }

                      if (
                        xolPremiumRateColumn &&
                        xolData.rows &&
                        xolData.rows.length > 0
                      ) {
                        const values = xolData.rows
                          .map((row) => {
                            const val = row[xolPremiumRateColumn.name];
                            return val !== null && val !== undefined
                              ? typeof val === "number"
                                ? val
                                : parseFloat(String(val))
                              : null;
                          })
                          .filter((v) => v !== null && !isNaN(v)) as number[];

                        if (values.length > 0) {
                          // 평균 계산
                          xolPremiumMean =
                            values.reduce((sum, v) => sum + v, 0) /
                            values.length;
                        }
                      }

                      // Reluctance Factor 값 가져오기
                      const reluctanceFactor = contract?.expenseRatio || 0;

                      return (
                        <div className="flex-shrink-0 border border-gray-200 rounded-lg p-4">
                          <div className="grid grid-cols-4 gap-4">
                            {/* XoL Claim 평균 */}
                            <div>
                              <div className="text-sm font-semibold text-gray-800 mb-2">
                                XoL Claim 평균
                              </div>
                              <div className="text-lg font-mono text-gray-700">
                                {xolClaimMean.toLocaleString("ko-KR", {
                                  maximumFractionDigits: 2,
                                  minimumFractionDigits: 2,
                                })}
                              </div>
                            </div>

                            {/* XoL Claim 표준편차 */}
                            <div>
                              <div className="text-sm font-semibold text-gray-800 mb-2">
                                XoL Claim 표준편차
                              </div>
                              <div className="text-lg font-mono text-gray-700">
                                {xolClaimStdDev.toLocaleString("ko-KR", {
                                  maximumFractionDigits: 2,
                                  minimumFractionDigits: 2,
                                })}
                              </div>
                            </div>

                            {/* XoL Premium Rate 평균 */}
                            <div>
                              <div className="text-sm font-semibold text-gray-800 mb-2">
                                XoL Premium Rate 평균
                              </div>
                              <div className="text-lg font-mono text-gray-700">
                                {xolPremiumMean.toLocaleString("ko-KR", {
                                  maximumFractionDigits: 2,
                                  minimumFractionDigits: 2,
                                })}
                              </div>
                            </div>

                            {/* Reluctance Factor */}
                            <div>
                              <div className="text-sm font-semibold text-gray-800 mb-2">
                                Reluctance Factor
                              </div>
                              <div className="text-lg font-mono text-gray-700">
                                {reluctanceFactor.toLocaleString("ko-KR", {
                                  maximumFractionDigits: 2,
                                  minimumFractionDigits: 2,
                                })}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* 2. 통계량 + XoL Claim(Incl. Agg/Reinst) 그래프 표시 */}
                    {(() => {
                      // 연도 컬럼 찾기
                      const yearColumn = xolData.columns.find(
                        (col) =>
                          col.name === "연도" ||
                          col.name === "year" ||
                          col.name.toLowerCase() === "year"
                      );

                      // XoL Claim(Incl. Agg/Reinst) 컬럼 찾기
                      const xolClaimAggReinstColumn = xolData.columns.find(
                        (col) => col.name === "XoL Claim(Incl. Agg/Reinst)"
                      );

                      return (
                        <div className="flex gap-4 flex-shrink-0">
                          {/* 왼쪽: 통계량 */}
                          {selectedColumn && (
                            <div className="w-1/2 flex-shrink-0">
                              <div className="h-full border border-gray-200 rounded-lg p-4 overflow-auto">
                                {/* ColumnStatistics 컴포넌트 사용 */}
                                {/* <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} /> */}
                              </div>
                            </div>
                          )}

                          {/* 오른쪽: XoL Claim(Incl. Agg/Reinst) 그래프 */}
                          {yearColumn && xolClaimAggReinstColumn && (
                            <div className="w-1/2 flex-shrink-0">
                              <div className="h-full border border-gray-200 rounded-lg p-4">
                                <h3 className="text-sm font-semibold text-gray-800 mb-4">
                                  {xolClaimAggReinstColumn.name}
                                </h3>
                                <div className="h-64">
                                  <YearlyAmountBarPlot
                                    rows={xolData.rows || []}
                                    yearColumn={yearColumn.name}
                                    amountColumn={xolClaimAggReinstColumn.name}
                                  />
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </>
                )}
              </>
            ) : (
              <>
                {/* 건별 XoL 적용 탭 */}
                {/* 테이블 영역 - 10행만 보이도록 고정 높이, 가로 전체 */}
                <div className="flex-shrink-0 mb-4">
                  <div className="overflow-x-auto overflow-y-hidden border border-gray-200 rounded-lg">
                    <table
                      className="min-w-full text-sm text-left"
                      style={{ minWidth: "max-content" }}
                    >
                      <thead className="bg-gray-50">
                        <tr>
                          {displayColumns.map((col) => {
                            const isYearColumn =
                              col.name === "연도" ||
                              col.name === "year" ||
                              col.name === "Year" ||
                              col.name.toLowerCase() === "year";
                            const isNumeric =
                              col.type === "number" && !isYearColumn;
                            return (
                              <th
                                key={col.name}
                                className={`py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 ${
                                  isNumeric ? "text-right" : "text-left"
                                }`}
                                onClick={() => requestSort(col.name)}
                              >
                                <div
                                  className={`flex items-center gap-1 ${
                                    isNumeric ? "justify-end" : ""
                                  }`}
                                >
                                  <span className="truncate" title={col.name}>
                                    {col.name}
                                  </span>
                                  {sortConfig?.key === col.name &&
                                    (sortConfig.direction === "ascending" ? (
                                      <ChevronUpIcon className="w-3 h-3" />
                                    ) : (
                                      <ChevronDownIcon className="w-3 h-3" />
                                    ))}
                                </div>
                              </th>
                            );
                          })}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedRows.slice(0, 10).map((row, rowIndex) => (
                          <tr
                            key={rowIndex}
                            className="border-b border-gray-200 last:border-b-0"
                          >
                            {displayColumns.map((col) => {
                              const isYearColumn =
                                col.name === "연도" ||
                                col.name === "year" ||
                                col.name === "Year" ||
                                col.name.toLowerCase() === "year";
                              const isNumeric =
                                col.type === "number" && !isYearColumn;
                              const value = row[col.name];
                              let displayValue: string;

                              if (value === null || value === undefined) {
                                displayValue = "";
                              } else if (isYearColumn) {
                                // 연도는 천단위 구분 없이 표시
                                displayValue = String(value);
                              } else if (isNumeric) {
                                const numValue =
                                  typeof value === "number"
                                    ? value
                                    : parseFloat(String(value));
                                if (!isNaN(numValue)) {
                                  displayValue = numValue.toLocaleString(
                                    "ko-KR",
                                    {
                                      maximumFractionDigits: 2,
                                      minimumFractionDigits: 0,
                                    }
                                  );
                                } else {
                                  displayValue = String(value);
                                }
                              } else {
                                displayValue = String(value);
                              }

                              return (
                                <td
                                  key={col.name}
                                  className={`py-1.5 px-3 font-mono truncate ${
                                    selectedColumn === col.name
                                      ? "bg-blue-100"
                                      : "hover:bg-gray-50 cursor-pointer"
                                  } ${isNumeric ? "text-right" : "text-left"}`}
                                  onClick={() => setSelectedColumn(col.name)}
                                  title={String(row[col.name])}
                                >
                                  {displayValue === "" ? (
                                    <i className="text-gray-400">null</i>
                                  ) : (
                                    displayValue
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-xs text-gray-500 mt-2 text-right">
                    Showing 10 of {xolData.totalRowCount.toLocaleString()} rows
                  </div>
                </div>

                {/* 통계량 */}
                {selectedColumnData && (
                  <div className="flex-shrink-0 mb-4">
                    {/* ColumnStatistics 컴포넌트 사용 */}
                    {/* <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} noBorder={true} backgroundColor="bg-gray-50" /> */}
                  </div>
                )}

                {/* 하단: 히스토그램 영역 - 보험금과 XoL Claim(Incl. Limit)를 상하로 표시 */}
                {(() => {
                  const claimColumn = module.parameters?.claim_column;
                  const xolLimitColumn = displayColumns.find(
                    (col) => col.name === "XoL Claim(Incl. Limit)"
                  );

                  if (!claimColumn || !xolLimitColumn) {
                    return null;
                  }

                  // 보험금 히스토그램 데이터 계산 (useMemo 대신 일반 계산)
                  const claimColumnData = displayRows
                    .map((r) => {
                      const val = r[claimColumn];
                      return val !== null && val !== undefined
                        ? typeof val === "number"
                          ? val
                          : parseFloat(String(val))
                        : null;
                    })
                    .filter((v) => v !== null && !isNaN(v)) as number[];

                  let claimHistogramData: {
                    bins: number[];
                    frequencies: number[];
                    min: number;
                    max: number;
                  } | null = null;
                  if (claimColumnData.length > 0) {
                    const min = Math.min(...claimColumnData);
                    const max = Math.max(...claimColumnData);
                    const numBins = 30;
                    const binSize = (max - min) / numBins;

                    const bins: number[] = [];
                    const frequencies: number[] = Array(numBins).fill(0);

                    for (let i = 0; i <= numBins; i++) {
                      bins.push(min + i * binSize);
                    }

                    for (const value of claimColumnData) {
                      let binIndex =
                        binSize > 0 ? Math.floor((value - min) / binSize) : 0;
                      if (binIndex === numBins) binIndex--;
                      if (binIndex >= 0 && binIndex < numBins) {
                        frequencies[binIndex]++;
                      }
                    }

                    claimHistogramData = { bins, frequencies, min, max };
                  }

                  // XoL Claim(Incl. Limit) 히스토그램 데이터 계산 (useMemo 대신 일반 계산)
                  const xolLimitColumnData = displayRows
                    .map((r) => {
                      const val = r["XoL Claim(Incl. Limit)"];
                      return val !== null && val !== undefined
                        ? typeof val === "number"
                          ? val
                          : parseFloat(String(val))
                        : null;
                    })
                    .filter((v) => v !== null && !isNaN(v)) as number[];

                  let xolLimitHistogramData: {
                    bins: number[];
                    frequencies: number[];
                    min: number;
                    max: number;
                  } | null = null;
                  if (xolLimitColumnData.length > 0) {
                    const min = Math.min(...xolLimitColumnData);
                    const max = Math.max(...xolLimitColumnData);
                    const numBins = 30;
                    const binSize = (max - min) / numBins;

                    const bins: number[] = [];
                    const frequencies: number[] = Array(numBins).fill(0);

                    for (let i = 0; i <= numBins; i++) {
                      bins.push(min + i * binSize);
                    }

                    for (const value of xolLimitColumnData) {
                      let binIndex =
                        binSize > 0 ? Math.floor((value - min) / binSize) : 0;
                      if (binIndex === numBins) binIndex--;
                      if (binIndex >= 0 && binIndex < numBins) {
                        frequencies[binIndex]++;
                      }
                    }

                    xolLimitHistogramData = { bins, frequencies, min, max };
                  }

                  // 히스토그램 렌더링 함수
                  const renderHistogram = (
                    histogramData: {
                      bins: number[];
                      frequencies: number[];
                      min: number;
                      max: number;
                    } | null,
                    title: string
                  ) => {
                    if (
                      !histogramData ||
                      !histogramData.bins ||
                      !histogramData.frequencies
                    ) {
                      return (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                          <p className="text-yellow-800">
                            히스토그램 데이터가 없습니다.
                          </p>
                        </div>
                      );
                    }

                    const { bins, frequencies, min, max } = histogramData;
                    const chartWidth = 800;
                    const chartHeight = 400;
                    const padding = {
                      top: 20,
                      right: 20,
                      bottom: 60,
                      left: 60,
                    };
                    const innerWidth =
                      chartWidth - padding.left - padding.right;
                    const innerHeight =
                      chartHeight - padding.top - padding.bottom;

                    const maxFrequency = Math.max(...frequencies, 1);
                    const numBins = frequencies.length;
                    const binWidth = innerWidth / numBins;

                    // X축 눈금 (6개)
                    const numXTicks = 6;
                    const xTickInterval = Math.max(
                      1,
                      Math.floor(numBins / numXTicks)
                    );

                    return (
                      <div className="bg-white border border-gray-200 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-gray-800 mb-4">
                          {title}
                        </h3>
                        <div className="w-full overflow-x-auto">
                          <svg
                            width={chartWidth}
                            height={chartHeight}
                            className="border border-gray-300 rounded"
                          >
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
                              const y =
                                padding.top + innerHeight - ratio * innerHeight;
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
                              const barHeight =
                                (freq / maxFrequency) * innerHeight;
                              const binStart = bins[idx];
                              const binEnd = bins[idx + 1];
                              const binCenter = (binStart + binEnd) / 2;
                              const x =
                                padding.left +
                                ((binCenter - min) / (max - min)) * innerWidth -
                                (binWidth * 0.9) / 2;
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

                            {/* X-axis ticks and labels */}
                            {frequencies.map((_, idx) => {
                              if (
                                idx % xTickInterval !== 0 &&
                                idx !== frequencies.length - 1
                              )
                                return null;
                              if (idx >= bins.length - 1) return null;
                              const binStart = bins[idx];
                              const binEnd = bins[idx + 1];
                              const binCenter = (binStart + binEnd) / 2;
                              const x =
                                padding.left +
                                ((binCenter - min) / (max - min)) * innerWidth;

                              return (
                                <g key={`x-tick-${idx}`}>
                                  <line
                                    x1={x}
                                    y1={padding.top + innerHeight}
                                    x2={x}
                                    y2={padding.top + innerHeight + 5}
                                    stroke="#6b7280"
                                    strokeWidth="1"
                                  />
                                  <text
                                    x={x}
                                    y={padding.top + innerHeight + 20}
                                    fontSize="10"
                                    fill="#6b7280"
                                    textAnchor="middle"
                                  >
                                    {binCenter.toLocaleString("ko-KR", {
                                      maximumFractionDigits: 0,
                                    })}
                                  </text>
                                </g>
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
                              transform={`rotate(-90, 15, ${
                                padding.top + innerHeight / 2
                              })`}
                            >
                              빈도 (Frequency)
                            </text>
                          </svg>
                        </div>
                      </div>
                    );
                  };

                  return (
                    <div className="flex-shrink-0 mt-4 space-y-4">
                      {/* 보험금 히스토그램 */}
                      {renderHistogram(
                        claimHistogramData,
                        `보험금 (${claimColumn})`
                      )}

                      {/* XoL Claim(Incl. Limit) 히스토그램 */}
                      {renderHistogram(
                        xolLimitHistogramData,
                        "XoL Claim(Incl. Limit)"
                      )}
                    </div>
                  );
                })()}
              </>
            )}
          </>
        )}
      </div>
    </>
  );
};
