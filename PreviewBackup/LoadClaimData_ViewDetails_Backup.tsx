/**
 * Load Claim Data View Details 백업 파일
 *
 * 이 파일은 DataPreviewModal.tsx의 Load Claim Data View Details 관련 코드를 백업한 것입니다.
 * Load Claim Data의 View Details가 의도치 않게 수정될 경우 이 파일을 참조하여 복구할 수 있습니다.
 *
 * 백업 일자: 2025-01-17
 *
 * 복구 방법:
 * 1. DataPreviewModal.tsx에서 Load Claim Data 관련 코드를 찾습니다.
 * 2. 이 파일의 내용으로 교체합니다.
 *
 * 포함된 내용:
 * - Detail 탭: 테이블 (숫자 천단위 구분, 우측 정렬) + 통계량 (테두리 없음, 옅은 회색 배경)
 * - Graphs 탭: 히스토그램 (Setting Threshold 스타일)
 * - activeLoadClaimDataTab 상태 관리
 * - graphColumn 상태 관리
 * - loadClaimDataHistogramData useMemo
 */

import React, { useState, useMemo, useEffect } from "react";
import { CanvasModule, ColumnInfo, DataPreview, ModuleType } from "../types";

/**
 * Load Claim Data View Details UI 컴포넌트
 *
 * 이 컴포넌트는 DataPreviewModal.tsx의 main 섹션 내부에 포함되어야 합니다.
 *
 * 사용 예시:
 * ```tsx
 * {module.type === ModuleType.LoadClaimData ? (
 *   <LoadClaimDataViewDetails
 *     module={module}
 *     data={data}
 *     displayColumns={displayColumns}
 *     displayRows={displayRows}
 *     sortedRows={sortedRows}
 *     selectedColumn={selectedColumn}
 *     setSelectedColumn={setSelectedColumn}
 *     selectedColumnData={selectedColumnData}
 *     isSelectedColNumeric={isSelectedColNumeric}
 *     sortConfig={sortConfig}
 *     requestSort={requestSort}
 *     activeLoadClaimDataTab={activeLoadClaimDataTab}
 *     setActiveLoadClaimDataTab={setActiveLoadClaimDataTab}
 *     graphColumn={graphColumn}
 *     setGraphColumn={setGraphColumn}
 *     numericCols={numericCols}
 *     loadClaimDataHistogramData={loadClaimDataHistogramData}
 *   />
 * ) : (
 *   // 일반 모듈 UI
 * )}
 * ```
 */

export const LoadClaimDataViewDetails: React.FC<{
  module: CanvasModule;
  data: DataPreview;
  displayColumns: ColumnInfo[];
  displayRows: Record<string, any>[];
  sortedRows: Record<string, any>[];
  selectedColumn: string | null;
  setSelectedColumn: (col: string | null) => void;
  selectedColumnData: (string | number | null)[] | null;
  isSelectedColNumeric: boolean;
  sortConfig: { key: string; direction: "ascending" | "descending" } | null;
  requestSort: (key: string) => void;
  activeLoadClaimDataTab: "detail" | "graphs";
  setActiveLoadClaimDataTab: (tab: "detail" | "graphs") => void;
  graphColumn: string | null;
  setGraphColumn: (col: string | null) => void;
  numericCols: string[];
  loadClaimDataHistogramData: {
    bins: number[];
    frequencies: number[];
    min: number;
    max: number;
  } | null;
}> = ({
  module,
  data,
  displayColumns,
  displayRows,
  sortedRows,
  selectedColumn,
  setSelectedColumn,
  selectedColumnData,
  isSelectedColNumeric,
  sortConfig,
  requestSort,
  activeLoadClaimDataTab,
  setActiveLoadClaimDataTab,
  graphColumn,
  setGraphColumn,
  numericCols,
  loadClaimDataHistogramData,
}) => {
  // ColumnStatistics와 ChevronUpIcon, ChevronDownIcon은 DataPreviewModal.tsx에 정의되어 있어야 합니다.
  // 여기서는 import하거나 동일한 컴포넌트를 사용해야 합니다.
  const ColumnStatistics = ({
    data,
    columnName,
    isNumeric,
    noBorder,
  }: {
    data: (string | number | null)[];
    columnName: string | null;
    isNumeric: boolean;
    noBorder?: boolean;
  }) => {
    // ColumnStatistics 구현은 DataPreviewModal.tsx를 참조하세요
    return null;
  };

  const ChevronUpIcon = ({ className }: { className?: string }) => (
    <span className={className}>↑</span>
  );
  const ChevronDownIcon = ({ className }: { className?: string }) => (
    <span className={className}>↓</span>
  );

  return (
    <>
      {/* Load Claim Data 모듈의 경우 탭 구성 */}
      <div className="flex-shrink-0 border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            onClick={() => setActiveLoadClaimDataTab("detail")}
            className={`${
              activeLoadClaimDataTab === "detail"
                ? "border-indigo-500 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            Detail
          </button>
          <button
            onClick={() => setActiveLoadClaimDataTab("graphs")}
            className={`${
              activeLoadClaimDataTab === "graphs"
                ? "border-indigo-500 text-indigo-600"
                : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
          >
            Graphs
          </button>
        </nav>
      </div>

      {/* Load Claim Data 모듈의 경우 데이터 표시 */}
      {activeLoadClaimDataTab === "detail" ? (
        <>
          {/* Detail 탭: 왼쪽 테이블, 오른쪽 통계량 */}
          <div className="flex justify-between items-center flex-shrink-0">
            <div className="text-sm text-gray-600">
              Showing {Math.min(displayRows.length, 1000)} of{" "}
              {data.totalRowCount.toLocaleString()} rows and{" "}
              {displayColumns.length} columns. Click a column to see details.
            </div>
          </div>
          <div className="flex-grow flex gap-4 overflow-hidden">
            {/* 왼쪽: 테이블 */}
            <div
              className={`overflow-auto ${
                selectedColumnData ? "w-1/2" : "w-full"
              }`}
            >
              <table className="min-w-full text-sm text-left">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    {displayColumns.map((col) => (
                      <th
                        key={col.name}
                        className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                        onClick={() => requestSort(col.name)}
                      >
                        <div className="flex items-center gap-1">
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
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-b border-gray-200 last:border-b-0"
                    >
                      {displayColumns.map((col) => {
                        const isNumeric = col.type === "number";
                        const value = row[col.name];
                        let displayValue: string;

                        if (value === null || value === undefined) {
                          displayValue = "";
                        } else if (isNumeric) {
                          const numValue =
                            typeof value === "number"
                              ? value
                              : parseFloat(String(value));
                          if (!isNaN(numValue)) {
                            displayValue = numValue.toLocaleString("ko-KR", {
                              maximumFractionDigits: 2,
                              minimumFractionDigits: 0,
                            });
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
            {/* 오른쪽: 통계량 */}
            {selectedColumnData && (
              <div className="w-1/2 flex flex-col gap-4 bg-gray-50 p-4">
                <ColumnStatistics
                  data={selectedColumnData}
                  columnName={selectedColumn}
                  isNumeric={isSelectedColNumeric}
                  noBorder={true}
                />
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          {/* Graphs 탭: 콤보박스로 열 선택, 히스토그램 표시 */}
          <div className="space-y-4">
            <div className="flex-shrink-0 flex items-center gap-2 mb-4">
              <label
                htmlFor="graph-column-select"
                className="font-semibold text-gray-700 text-sm"
              >
                Select Column:
              </label>
              <select
                id="graph-column-select"
                value={graphColumn || ""}
                onChange={(e) => setGraphColumn(e.target.value || null)}
                className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="">Select a column</option>
                {numericCols.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
            {loadClaimDataHistogramData &&
            loadClaimDataHistogramData.bins &&
            loadClaimDataHistogramData.frequencies ? (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  데이터 분포 히스토그램
                </h3>
                {(() => {
                  const { bins, frequencies, min, max } =
                    loadClaimDataHistogramData;
                  const chartWidth = 800;
                  const chartHeight = 400;
                  const padding = { top: 20, right: 20, bottom: 60, left: 60 };
                  const innerWidth =
                    chartWidth - padding.left - padding.right;
                  const innerHeight =
                    chartHeight - padding.top - padding.bottom;

                  const maxFrequency = Math.max(...frequencies, 1);
                  const numBins = frequencies.length;
                  const binWidth = innerWidth / numBins;

                  return (
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

                        {/* X-axis labels (넓은 간격으로 표시) */}
                        {(() => {
                          // X축에 5-6개의 숫자만 표시 (넓은 간격)
                          const numTicks = 6;
                          const tickStep = (max - min) / (numTicks - 1);
                          const ticks = [];
                          for (let i = 0; i < numTicks; i++) {
                            ticks.push(min + i * tickStep);
                          }

                          return ticks.map((tick, idx) => {
                            const xPos =
                              padding.left +
                              ((tick - min) / (max - min)) * innerWidth;
                            return (
                              <g key={`x-tick-${idx}`}>
                                <line
                                  x1={xPos}
                                  y1={padding.top + innerHeight}
                                  x2={xPos}
                                  y2={padding.top + innerHeight + 5}
                                  stroke="#9ca3af"
                                  strokeWidth="1"
                                />
                                <text
                                  x={xPos}
                                  y={padding.top + innerHeight + 20}
                                  fontSize="11"
                                  fill="#6b7280"
                                  textAnchor="middle"
                                >
                                  {tick.toLocaleString(undefined, {
                                    maximumFractionDigits: 2,
                                  })}
                                </text>
                              </g>
                            );
                          });
                        })()}

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
                  );
                })()}
              </div>
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-yellow-800">
                  {graphColumn
                    ? "히스토그램 데이터가 없습니다."
                    : "열을 선택해주세요."}
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
};


