import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { CanvasModule, ColumnInfo, DataPreview, ModuleType, ThresholdAnalysisOutput } from '../types';
import { XCircleIcon, ChevronUpIcon, ChevronDownIcon, SparklesIcon, ArrowDownTrayIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';
import { SpreadViewModal } from './SpreadViewModal';
// import { calculatePCAForScoreVisualization } from '../utils/pyodideRunner'; // Python 버전 (이상치 처리)
import { calculatePCA } from '../utils/pcaCalculator'; // JavaScript 버전 (ml-pca 사용)

interface DataPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
    allModules?: CanvasModule[];
    allConnections?: Array<{ from: { moduleId: string; portName: string }; to: { moduleId: string; portName: string } }>;
}

type SortConfig = {
    key: string;
    direction: 'ascending' | 'descending';
} | null;

const HistogramPlot: React.FC<{ rows: Record<string, any>[]; column: string; }> = ({ rows, column }) => {
    const data = useMemo(() => rows.map(r => r[column]), [rows, column]);
    const numericData = useMemo(() => data.map(v => parseFloat(v as string)).filter(v => !isNaN(v)), [data]);

    if (numericData.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400 text-sm">No numeric data in this column to plot.</div>;
    }

    const { bins } = useMemo(() => {
        const min = Math.min(...numericData);
        const max = Math.max(...numericData);
        const numBins = 10;
        const binSize = (max - min) / numBins;
        const bins = Array(numBins).fill(0);

        for (const value of numericData) {
            let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
            if (binIndex === numBins) binIndex--;
            if (binIndex >= 0 && binIndex < numBins) {
                bins[binIndex]++;
            }
        }
        return { bins };
    }, [numericData]);
    
    const maxBinCount = Math.max(...bins, 0);

    return (
        <div className="w-full h-full p-4 flex flex-col border border-gray-200 rounded-lg">
             <div className="flex-grow flex items-center gap-2 overflow-hidden">
                {/* Y-axis Label */}
                <div className="flex items-center justify-center h-full">
                    <p className="text-sm text-gray-600 font-semibold transform -rotate-90 whitespace-nowrap">
                        Frequency
                    </p>
                </div>
                
                {/* Plot area */}
                <div className="flex-grow h-full flex flex-col">
                    <div className="flex-grow flex items-end justify-around gap-1 pt-4">
                        {bins.map((count, index) => {
                            const heightPercentage = maxBinCount > 0 ? (count / maxBinCount) * 100 : 0;
                            return (
                                <div key={index} className="flex-1 h-full flex flex-col justify-end items-center group relative" title={`Count: ${count}`}>
                                    <span className="absolute -top-5 text-xs bg-gray-800 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">{count}</span>
                                    <div 
                                        className="w-full bg-blue-400 hover:bg-blue-500 transition-colors"
                                        style={{ height: `${heightPercentage}%` }}
                                    >
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    {/* X-axis Label */}
                    <div className="w-full text-center text-sm text-gray-600 font-semibold mt-2 border-t pt-1">
                        {column}
                    </div>
                </div>
             </div>
        </div>
    );
};

// 연도별 금액을 표시하는 Bar Plot 컴포넌트
const YearlyAmountBarPlot: React.FC<{ rows: Record<string, any>[]; yearColumn: string; amountColumn: string }> = ({ rows, yearColumn, amountColumn }) => {
    const data = useMemo(() => {
        return rows.map(row => {
            const year = row[yearColumn];
            const amount = parseFloat(row[amountColumn]) || 0;
            
            // 연도 값 정규화 (숫자로 변환)
            let yearValue: number;
            if (typeof year === 'number') {
                yearValue = year;
            } else if (typeof year === 'string') {
                // 문자열에서 4자리 숫자 추출
                const yearMatch = year.match(/\d{4}/);
                if (yearMatch) {
                    yearValue = parseInt(yearMatch[0], 10);
                } else {
                    const parsed = parseFloat(year);
                    yearValue = !isNaN(parsed) ? parsed : 0;
                }
            } else {
                yearValue = 0;
            }
            
            return {
                year: yearValue,
                amount: amount
            };
        }).filter(d => !isNaN(d.amount) && d.amount > 0 && !isNaN(d.year) && d.year > 0);
    }, [rows, yearColumn, amountColumn]);

    if (data.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400 text-sm">No data available to plot.</div>;
    }

    const maxAmount = Math.max(...data.map(d => d.amount), 0);
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
                            const heightPercentage = maxAmount > 0 ? (d.amount / maxAmount) * 100 : 0;
                            return (
                                <div key={index} className="flex-1 h-full flex flex-col justify-end items-center group relative" title={`Year: ${d.year}, Amount: ${d.amount.toLocaleString()}`}>
                                    <span className="absolute -top-5 text-xs bg-gray-800 text-white px-1.5 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                                        {d.amount.toLocaleString()}
                                    </span>
                                    <div 
                                        className="w-full bg-green-500 hover:bg-green-600 transition-colors"
                                        style={{ height: `${heightPercentage}%` }}
                                    >
                                    </div>
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

const ScatterPlot: React.FC<{ rows: Record<string, any>[], xCol: string, yCol: string }> = ({ rows, xCol, yCol }) => {
    const dataPoints = useMemo(() => rows.map(r => ({ x: Number(r[xCol]), y: Number(r[yCol]) })).filter(p => !isNaN(p.x) && !isNaN(p.y)), [rows, xCol, yCol]);

    if (dataPoints.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400">No valid data points for scatter plot.</div>;
    }

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = 600;
    const height = 400;

    const xMin = Math.min(...dataPoints.map(d => d.x));
    const xMax = Math.max(...dataPoints.map(d => d.x));
    const yMin = Math.min(...dataPoints.map(d => d.y));
    const yMax = Math.max(...dataPoints.map(d => d.y));

    const xScale = (x: number) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
    const yScale = (y: number) => height - margin.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);
    
    const getTicks = (min: number, max: number, count: number) => {
        if (min === max) return [min];
        const ticks = [];
        const step = (max - min) / (count - 1);
        for (let i = 0; i < count; i++) {
            ticks.push(min + i * step);
        }
        return ticks;
    };
    
    const xTicks = getTicks(xMin, xMax, 5);
    const yTicks = getTicks(yMin, yMax, 5);

    return (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
            {/* Axes */}
            <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="currentColor" strokeWidth="1" />
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="currentColor" strokeWidth="1" />

            {/* X Ticks and Labels */}
            {xTicks.map((tick, i) => (
                <g key={`x-${i}`} transform={`translate(${xScale(tick)}, ${height - margin.bottom})`}>
                    <line y2="5" stroke="currentColor" strokeWidth="1" />
                    <text y="20" textAnchor="middle" fill="currentColor" fontSize="10">{tick.toFixed(1)}</text>
                </g>
            ))}
            <text x={width/2} y={height - 5} textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">{xCol}</text>
            
            {/* Y Ticks and Labels */}
            {yTicks.map((tick, i) => (
                <g key={`y-${i}`} transform={`translate(${margin.left}, ${yScale(tick)})`}>
                    <line x2="-5" stroke="currentColor" strokeWidth="1" />
                    <text x="-10" y="3" textAnchor="end" fill="currentColor" fontSize="10">{tick.toFixed(1)}</text>
                </g>
            ))}
            <text transform={`translate(${15}, ${height/2}) rotate(-90)`} textAnchor="middle" fill="currentColor" fontSize="12" fontWeight="bold">{yCol}</text>

            {/* Points */}
            <g>
                {dataPoints.map((d, i) => (
                    <circle key={i} cx={xScale(d.x)} cy={yScale(d.y)} r="2.5" fill="rgba(59, 130, 246, 0.7)" />
                ))}
            </g>
        </svg>
    );
};

// Threshold Analysis 차트 컴포넌트들
const ThresholdHistogramPlot: React.FC<{ histogram: { bins: number[]; frequencies: number[] } }> = ({ histogram }) => {
    const { bins, frequencies } = histogram;
    const chartWidth = 800;
    const chartHeight = 400;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;
    const maxFrequency = Math.max(...frequencies, 1);
    const numBins = frequencies.length;
    const binWidth = innerWidth / numBins;
    const min = bins[0];
    const max = bins[bins.length - 1];

    return (
        <div className="w-full overflow-x-auto">
            <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
                <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <line x1={padding.left} y1={padding.top + innerHeight} x2={padding.left + innerWidth} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                {[0, 0.25, 0.5, 0.75, 1.0].map((ratio, idx) => {
                    const y = padding.top + innerHeight - (ratio * innerHeight);
                    const value = Math.round(ratio * maxFrequency);
                    return (
                        <g key={`y-label-${idx}`}>
                            <line x1={padding.left - 5} y1={y} x2={padding.left} y2={y} stroke="#9ca3af" strokeWidth="1" />
                            <text x={padding.left - 10} y={y + 4} fontSize="12" fill="#6b7280" textAnchor="end">{value}</text>
                        </g>
                    );
                })}
                {frequencies.map((freq, idx) => {
                    if (idx >= bins.length - 1) return null;
                    const barHeight = (freq / maxFrequency) * innerHeight;
                    const binStart = bins[idx];
                    const binEnd = bins[idx + 1];
                    const binCenter = (binStart + binEnd) / 2;
                    const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth - (binWidth * 0.9 / 2);
                    const y = padding.top + innerHeight - barHeight;
                    return (
                        <rect key={`bar-${idx}`} x={x} y={y} width={binWidth * 0.9} height={barHeight} fill="#3b82f6" opacity={0.7} />
                    );
                })}
                <text x={padding.left + innerWidth / 2} y={chartHeight - 10} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold">값 (Value)</text>
                <text x={15} y={padding.top + innerHeight / 2} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold" transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}>빈도 (Frequency)</text>
            </svg>
        </div>
    );
};

const ECDFPlot: React.FC<{ ecdf: { sortedValues: number[]; cumulativeProbabilities: number[] } }> = ({ ecdf }) => {
    const { sortedValues, cumulativeProbabilities } = ecdf;
    const chartWidth = 800;
    const chartHeight = 400;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;
    const xMin = Math.min(...sortedValues);
    const xMax = Math.max(...sortedValues);
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - (y * innerHeight);

    return (
        <div className="w-full overflow-x-auto">
            <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
                <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <line x1={padding.left} y1={padding.top + innerHeight} x2={padding.left + innerWidth} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <polyline
                    points={sortedValues.map((x, i) => `${xScale(x)},${yScale(cumulativeProbabilities[i])}`).join(' ')}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="2"
                />
                <text x={padding.left + innerWidth / 2} y={chartHeight - 10} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold">값 (Value)</text>
                <text x={15} y={padding.top + innerHeight / 2} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold" transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}>누적 확률 (Cumulative Probability)</text>
            </svg>
        </div>
    );
};

const QQPlot: React.FC<{ qqPlot: { theoreticalQuantiles: number[]; sampleQuantiles: number[] } }> = ({ qqPlot }) => {
    const { theoreticalQuantiles, sampleQuantiles } = qqPlot;
    const chartWidth = 800;
    const chartHeight = 400;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;
    const xMin = Math.min(...theoreticalQuantiles);
    const xMax = Math.max(...theoreticalQuantiles);
    const yMin = Math.min(...sampleQuantiles);
    const yMax = Math.max(...sampleQuantiles);
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - ((y - yMin) / (yMax - yMin || 1)) * innerHeight;

    // 45도 대각선 (y=x)
    const diagonalMin = Math.min(xMin, yMin);
    const diagonalMax = Math.max(xMax, yMax);

    return (
        <div className="w-full overflow-x-auto">
            <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
                <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <line x1={padding.left} y1={padding.top + innerHeight} x2={padding.left + innerWidth} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <line x1={xScale(diagonalMin)} y1={yScale(diagonalMin)} x2={xScale(diagonalMax)} y2={yScale(diagonalMax)} stroke="#ef4444" strokeWidth="1" strokeDasharray="5,5" opacity={0.7} />
                {theoreticalQuantiles.map((x, i) => (
                    <circle key={i} cx={xScale(x)} cy={yScale(sampleQuantiles[i])} r="3" fill="#3b82f6" opacity={0.7} />
                ))}
                <text x={padding.left + innerWidth / 2} y={chartHeight - 10} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold">이론적 분위수 (Theoretical Quantiles)</text>
                <text x={15} y={padding.top + innerHeight / 2} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold" transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}>표본 분위수 (Sample Quantiles)</text>
            </svg>
        </div>
    );
};

const MeanExcessPlot: React.FC<{ meanExcessPlot: { thresholds: number[]; meanExcesses: number[] } }> = ({ meanExcessPlot }) => {
    const { thresholds, meanExcesses } = meanExcessPlot;
    const chartWidth = 800;
    const chartHeight = 400;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;
    const xMin = Math.min(...thresholds);
    const xMax = Math.max(...thresholds);
    const yMin = Math.min(...meanExcesses);
    const yMax = Math.max(...meanExcesses);
    const xScale = (x: number) => padding.left + ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
    const yScale = (y: number) => padding.top + innerHeight - ((y - yMin) / (yMax - yMin || 1)) * innerHeight;

    return (
        <div className="w-full overflow-x-auto">
            <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
                <line x1={padding.left} y1={padding.top} x2={padding.left} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <line x1={padding.left} y1={padding.top + innerHeight} x2={padding.left + innerWidth} y2={padding.top + innerHeight} stroke="#374151" strokeWidth="2" />
                <polyline
                    points={thresholds.map((x, i) => `${xScale(x)},${yScale(meanExcesses[i])}`).join(' ')}
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="2"
                />
                {thresholds.map((x, i) => (
                    <circle key={i} cx={xScale(x)} cy={yScale(meanExcesses[i])} r="3" fill="#3b82f6" />
                ))}
                <text x={padding.left + innerWidth / 2} y={chartHeight - 10} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold">Threshold</text>
                <text x={15} y={padding.top + innerHeight / 2} fontSize="14" fill="#374151" textAnchor="middle" fontWeight="semibold" transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}>Mean Excess</text>
            </svg>
        </div>
    );
};

// PCA Scatter Plot 컴포넌트
const PCAScatterPlot: React.FC<{
    coordinates: number[][];
    actualValues?: (string | number)[];
    predictedValues?: (string | number)[];
    modelType?: 'classification' | 'regression';
    explainedVariance: number[];
    hasLabel?: boolean; // Label Column이 있는지 여부
}> = ({ coordinates, actualValues, predictedValues, modelType, explainedVariance, hasLabel = false }) => {
    const dataPoints = useMemo(() => {
        return coordinates.map((coord, idx) => ({
            x: coord[0] || 0,
            y: coord[1] || 0,
            actual: actualValues?.[idx],
            predicted: predictedValues?.[idx],
            index: idx,
        }));
    }, [coordinates, actualValues, predictedValues]);
    
    // 이진 분류인지 확인 (0과 1만 있는지 확인)
    const isBinaryClassification = useMemo(() => {
        if (!hasLabel || modelType !== 'classification' || !actualValues || actualValues.length === 0) {
            return false;
        }
        
        const uniqueValues = new Set(actualValues.map(v => String(v)));
        const validBinaryValues = ['0', '1'];
        return uniqueValues.size <= 2 && Array.from(uniqueValues).every(v => validBinaryValues.includes(v));
    }, [hasLabel, modelType, actualValues]);

    if (dataPoints.length === 0) {
        return <div className="flex items-center justify-center h-full text-gray-400">No data points available.</div>;
    }

    const margin = { top: 40, right: 40, bottom: 70, left: 70 };
    const width = 1400;
    const height = 650;

    const xMin = Math.min(...dataPoints.map(d => d.x));
    const xMax = Math.max(...dataPoints.map(d => d.x));
    const yMin = Math.min(...dataPoints.map(d => d.y));
    const yMax = Math.max(...dataPoints.map(d => d.y));

    const xScale = (x: number) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
    const yScale = (y: number) => height - margin.bottom - ((y - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);

    // 분류 모델: 클래스별 색상
    const getColorForClassification = (actual: string | number, predicted: string | number) => {
        const isCorrect = String(actual) === String(predicted);
        const classKey = String(actual);
        
        // 이진 분류인 경우 명확한 색상 사용
        if (isBinaryClassification) {
            if (classKey === '0') {
                // 실제가 0: 파란 계열
                return isCorrect ? '#2563eb' : '#93c5fd'; // 맞으면 진한 파란, 틀리면 연한 파란
            } else if (classKey === '1') {
                // 실제가 1: 빨간 계열
                return isCorrect ? '#dc2626' : '#fca5a5'; // 맞으면 진한 빨강, 틀리면 연한 빨강
            }
        }
        
        // 일반 다중 분류: 클래스별 기본 색상
        const classColors: Record<string, string> = {
            '0': '#2563eb', // 진한 파란
            '1': '#dc2626', // 진한 빨강
            '2': '#16a34a', // 진한 초록
            '3': '#ea580c', // 진한 주황
            '4': '#9333ea', // 진한 보라
            '5': '#0891b2', // 청록
            '6': '#ca8a04', // 노랑
            '7': '#e11d48', // 분홍
        };
        
        const baseColor = classColors[classKey] || '#6b7280';
        return isCorrect ? baseColor : `${baseColor}60`; // 틀린 경우 투명도 추가
    };

    // 회귀 모델: 실제 값에 따른 색상 (gradient) - 연속적인 색상 사용
    const getColorForRegression = (actual: number) => {
        const allActuals = (actualValues || []).filter(v => typeof v === 'number') as number[];
        if (allActuals.length === 0) return '#6366f1';
        
        const minVal = Math.min(...allActuals);
        const maxVal = Math.max(...allActuals);
        const range = maxVal - minVal || 1;
        const normalized = (actual - minVal) / range;
        
        // 파란(낮음) -> 보라 -> 빨강(높음) gradient (저에서 높이로 전환)
        if (normalized < 0.5) {
            // 파란 -> 보라
            const t = normalized * 2;
            const r = Math.round(37 + t * 99);
            const g = Math.round(99 + t * 99);
            const b = Math.round(235 - t * 99);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // 보라 -> 빨강
            const t = (normalized - 0.5) * 2;
            const r = Math.round(136 + t * 120);
            const g = Math.round(198 - t * 172);
            const b = Math.round(136 - t * 136);
            return `rgb(${r}, ${g}, ${b})`;
        }
    };
    
    // Label이 없을 경우 사용할 색상 (기본 색상)
    const getDefaultColor = () => '#6366f1'; // 인디고
    // 회귀 모델: 오차 크기 계산
    const getErrorSize = (actual: number, predicted: number) => {
        const error = Math.abs(actual - predicted);
        const allErrors = dataPoints
            .filter(p => typeof p.actual === 'number' && typeof p.predicted === 'number')
            .map(p => Math.abs((p.actual as number) - (p.predicted as number)));
        const maxError = Math.max(...allErrors, 1);
        return 4 + (error / maxError) * 6; // 최소 4, 최대 10
    };
    
    // 클래스별 통계 (범례)
    const classStats = useMemo(() => {
        if (!hasLabel || modelType !== 'classification' || !actualValues) return null;
        const stats: Record<string, { count: number; correct: number }> = {};
        actualValues.forEach((actual, idx) => {
            const key = String(actual);
            if (!stats[key]) {
                stats[key] = { count: 0, correct: 0 };
            }
            stats[key].count++;
            if (predictedValues && String(actual) === String(predictedValues[idx])) {
                stats[key].correct++;
            }
        });
        return stats;
    }, [hasLabel, modelType, actualValues, predictedValues]);

    const totalVariance = explainedVariance.reduce((a, b) => a + b, 0) * 100;
    const getTicks = (min: number, max: number, count: number) => {
        if (min === max) return [min];
        const ticks = [];
        const step = (max - min) / (count - 1);
        for (let i = 0; i < count; i++) {
            ticks.push(min + i * step);
        }
        return ticks;
    };
    
    const xTicks = getTicks(xMin, xMax, 6);
    const yTicks = getTicks(yMin, yMax, 6);
    
    // 그리드선을 위한 tick 값
    const gridXTicks = getTicks(xMin, xMax, 6);
    const gridYTicks = getTicks(yMin, yMax, 6);

    return (
        <div className="w-full h-full flex flex-col items-center gap-3 p-4">
            {/* 설명된 분산 정보 표시 */}
            <div className="w-full max-w-6xl bg-indigo-50 border border-indigo-200 rounded-lg p-3">
                <div className="text-sm font-semibold text-indigo-900 mb-1">Explained Variance</div>
                <div className="flex items-center gap-4 text-xs text-indigo-700">
                    <span>Total: <strong>{totalVariance.toFixed(1)}%</strong></span>
                    <span>PC1: <strong>{(explainedVariance[0] * 100).toFixed(1)}%</strong></span>
                    <span>PC2: <strong>{(explainedVariance[1] * 100).toFixed(1)}%</strong></span>
                </div>
            </div>
            
            <div className="w-full max-w-6xl">
                {/* 스캐터 플롯 */}
                <div className="w-full">
                    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
                        {/* 그리드선 */}
                        <g stroke="#e5e7eb" strokeWidth="0.5" opacity="0.5">
                            {gridXTicks.map((tick, i) => (
                                <line 
                                    key={`grid-x-${i}`} 
                                    x1={xScale(tick)} 
                                    y1={margin.top} 
                                    x2={xScale(tick)} 
                                    y2={height - margin.bottom} 
                                />
                            ))}
                            {gridYTicks.map((tick, i) => (
                                <line 
                                    key={`grid-y-${i}`} 
                                    x1={margin.left} 
                                    y1={yScale(tick)} 
                                    x2={width - margin.right} 
                                    y2={yScale(tick)} 
                                />
                            ))}
                        </g>
                        
                        {/* Axes */}
                        <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />
                        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#374151" strokeWidth="2" />

                        {/* X Ticks and Labels */}
                        {xTicks.map((tick, i) => (
                            <g key={`x-${i}`} transform={`translate(${xScale(tick)}, ${height - margin.bottom})`}>
                                <line y2="6" stroke="#374151" strokeWidth="1.5" />
                                <text y="25" textAnchor="middle" fill="#374151" fontSize="11" fontWeight="500">{tick.toFixed(2)}</text>
                            </g>
                        ))}
                        <text x={width/2} y={height - 10} textAnchor="middle" fill="#1f2937" fontSize="13" fontWeight="bold">
                            PC1 ({(explainedVariance[0] * 100).toFixed(1)}% variance)
                        </text>
                        
                        {/* Y Ticks and Labels */}
                        {yTicks.map((tick, i) => (
                            <g key={`y-${i}`} transform={`translate(${margin.left}, ${yScale(tick)})`}>
                                <line x2="-6" stroke="#374151" strokeWidth="1.5" />
                                <text x="-12" y="4" textAnchor="end" fill="#374151" fontSize="11" fontWeight="500">{tick.toFixed(2)}</text>
                            </g>
                        ))}
                        <text transform={`translate(${20}, ${height/2}) rotate(-90)`} textAnchor="middle" fill="#1f2937" fontSize="13" fontWeight="bold">
                            PC2 ({(explainedVariance[1] * 100).toFixed(1)}% variance)
                        </text>

                        {/* Points */}
                        <g>
                            {dataPoints.map((d, i) => {
                                let color: string;
                                let size: number;
                                let stroke: string = 'none';
                                let strokeWidth: number = 0;
                                let opacity: number = 0.75;
                                
                                if (hasLabel && modelType) {
                                    if (modelType === 'classification' && d.actual !== undefined && d.predicted !== undefined) {
                                        color = getColorForClassification(d.actual, d.predicted);
                                        size = isBinaryClassification ? 6 : 5; // 이진 분류는 약간 크게
                                        
                                        // 이진 분류: 틀린 경우 테두리 추가
                                        if (isBinaryClassification && String(d.actual) !== String(d.predicted)) {
                                            stroke = '#000';
                                            strokeWidth = 1.5;
                                            opacity = 0.9; // 틀린 경우 더 선명하게
                                        } else if (!isBinaryClassification && String(d.actual) !== String(d.predicted)) {
                                            // 일반 다중: 틀린 경우 검은 테두리
                                            stroke = '#000';
                                            strokeWidth = 2;
                                        }
                                    } else if (modelType === 'regression' && typeof d.actual === 'number' && typeof d.predicted === 'number') {
                                        color = getColorForRegression(d.actual);
                                        size = getErrorSize(d.actual, d.predicted);
                                    } else {
                                        color = getDefaultColor();
                                        size = 5;
                                    }
                                } else {
                                    color = getDefaultColor();
                                    size = 5;
                                }
                                
                                return (
                                    <circle 
                                        key={i} 
                                        cx={xScale(d.x)} 
                                        cy={yScale(d.y)} 
                                        r={size} 
                                        fill={color}
                                        stroke={stroke}
                                        strokeWidth={strokeWidth}
                                        opacity={opacity}
                                        className="hover:opacity-100 hover:r-7 transition-all cursor-pointer"
                                    >
                                        <title>
                                            {hasLabel && d.actual !== undefined && d.predicted !== undefined
                                                ? (modelType === 'classification' 
                                                    ? `Actual: ${d.actual}, Predicted: ${d.predicted}${String(d.actual) === String(d.predicted) ? ' ✓' : ' ✗'}`
                                                    : `Actual: ${d.actual}, Predicted: ${d.predicted}, Error: ${typeof d.actual === 'number' && typeof d.predicted === 'number' ? Math.abs(d.actual - d.predicted).toFixed(2) : 'N/A'}`)
                                                : `Point ${i + 1}: (${d.x.toFixed(2)}, ${d.y.toFixed(2)})`
                                            }
                                        </title>
                                    </circle>
                                );
                            })}
                        </g>
                    </svg>
                </div>
            </div>
            
            {/* 범례 표시 */}
            {hasLabel && modelType && (
                <div className="text-xs text-gray-500 text-center max-w-6xl">
                    {modelType === 'classification' 
                        ? (isBinaryClassification 
                            ? 'Blue = Class 0, Red = Class 1. Black border indicates incorrect predictions.'
                            : 'Color represents actual class. Black border indicates incorrect predictions.')
                        : 'Color represents actual value (blue=low, red=high). Size represents prediction error magnitude.'}
                </div>
            )}
        </div>
    );
};

const ColumnStatistics: React.FC<{ data: (string | number | null)[]; columnName: string | null; isNumeric: boolean; noBorder?: boolean; backgroundColor?: string; }> = ({ data, columnName, isNumeric, noBorder = false, backgroundColor }) => {
    const stats = useMemo(() => {
        const isNull = (v: any) => v === null || v === undefined || v === '';
        const nonNullValues = data.filter(v => !isNull(v));
        const nulls = data.length - nonNullValues.length;
        const count = data.length;

        let mode: number | string = 'N/A';
        if (nonNullValues.length > 0) {
            const counts: Record<string, number> = {};
            for(const val of nonNullValues) {
                const key = String(val);
                counts[key] = (counts[key] || 0) + 1;
            }
            mode = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        }

        if (!isNumeric) {
            return {
                Count: count,
                Null: nulls,
                Mode: mode,
            };
        }
        
        const numericValues = nonNullValues.map(v => Number(v)).filter(v => !isNaN(v));

        if (numericValues.length === 0) {
             return {
                Count: count,
                Null: nulls,
                Mode: mode,
            };
        }
        
        numericValues.sort((a,b) => a - b);
        const sum = numericValues.reduce((a, b) => a + b, 0);
        const mean = sum / numericValues.length;
        const n = numericValues.length;
        const stdDev = Math.sqrt(numericValues.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
        const skewness = stdDev > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 3), 0) / (n * Math.pow(stdDev, 3)) : 0;
        const kurtosis = stdDev > 0 ? numericValues.reduce((s, val) => s + Math.pow(val - mean, 4), 0) / (n * Math.pow(stdDev, 4)) - 3 : 0;


        const getQuantile = (q: number) => {
            const pos = (numericValues.length - 1) * q;
            const base = Math.floor(pos);
            const rest = pos - base;
            if (numericValues[base + 1] !== undefined) {
                return numericValues[base] + rest * (numericValues[base + 1] - numericValues[base]);
            } else {
                return numericValues[base];
            }
        };

        const numericMode = Number(mode);

        return {
            Count: data.length,
            Mean: mean.toFixed(2),
            'Std Dev': stdDev.toFixed(2),
            Median: getQuantile(0.5).toFixed(2),
            Min: numericValues[0].toFixed(2),
            Max: numericValues[numericValues.length - 1].toFixed(2),
            '25%': getQuantile(0.25).toFixed(2),
            '75%': getQuantile(0.75).toFixed(2),
            Mode: isNaN(numericMode) ? mode : numericMode,
            Null: nulls,
            Skew: skewness.toFixed(2),
            Kurt: kurtosis.toFixed(2),
        };
    }, [data, isNumeric]);
    
    const statOrder = isNumeric 
        ? ['Count', 'Mean', 'Std Dev', 'Median', 'Min', 'Max', '25%', '75%', 'Mode', 'Null', 'Skew', 'Kurt']
        : ['Count', 'Null', 'Mode'];

    const containerClassName = `w-full p-4 ${noBorder ? '' : 'border border-gray-200 rounded-lg'} ${backgroundColor || ''}`;

    return (
        <div className={containerClassName}>
            <h4 className="font-semibold text-gray-700 mb-3">Statistics for {columnName}</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 text-sm">
                {statOrder.map(key => {
                    const value = (stats as Record<string, any>)[key];
                    if (value === undefined || value === null) return null;
                    return (
                        <React.Fragment key={key}>
                           <span className="text-gray-500">{key}:</span> 
                           <span className="font-mono text-gray-800 font-medium">{String(value)}</span>
                        </React.Fragment>
                    );
                })}
            </div>
        </div>
    );
};


export const DataPreviewModal: React.FC<DataPreviewModalProps> = ({ module, projectName, onClose, allModules = [], allConnections = [] }) => {
    // XoL Contract 모듈의 경우 입력 데이터 가져오기
    const getXolContractInputData = (): DataPreview | null => {
        if (module.type !== ModuleType.DefineXolContract) return null;
        
        // 입력 데이터 연결 찾기
        const dataConnection = allConnections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === 'data_in'
        );
        
        if (!dataConnection) return null;
        
        const sourceModule = allModules.find((m) => m.id === dataConnection.from.moduleId);
        if (!sourceModule || !sourceModule.outputData) return null;
        
        // 데이터 추출
        if (sourceModule.outputData.type === 'DataPreview') {
            return sourceModule.outputData;
        }
        if (sourceModule.outputData.type === 'ClaimDataOutput' || 
            sourceModule.outputData.type === 'InflatedDataOutput' || 
            sourceModule.outputData.type === 'FormatChangeOutput') {
            return (sourceModule.outputData as any).data || null;
        }
        
        return null;
    };
    
    // 데이터를 가져오는 함수
    const getPreviewData = (): DataPreview | null => {
        try {
            if (!module || !module.outputData) return null;
        if (module.outputData.type === 'DataPreview') return module.outputData;
        if (module.outputData.type === 'ClaimDataOutput' || module.outputData.type === 'InflatedDataOutput' || module.outputData.type === 'FormatChangeOutput') {
                return (module.outputData as any).data || null;
        }
        if (module.outputData.type === 'KMeansOutput' || module.outputData.type === 'HierarchicalClusteringOutput' || module.outputData.type === 'DBSCANOutput') {
                return module.outputData.clusterAssignments || null;
        }
        if (module.outputData.type === 'PCAOutput') {
                return module.outputData.transformedData || null;
        }
        if (module.outputData.type === 'ThresholdSplitOutput') {
                // ThresholdSplitOutput의 경우 기본적으로 aboveThreshold 반환 (탭에서 처리)
                return (module.outputData as any).aboveThreshold || null;
        }
        return null;
        } catch (error) {
            console.error('Error in getPreviewData:', error);
            return null;
        }
    };
    
    const data = getPreviewData();
    
    // Split By Threshold 모듈의 경우 탭에 따라 데이터 선택
    const [activeThresholdSplitTab, setActiveThresholdSplitTab] = useState<'above' | 'below'>('below'); // Split By Threshold 탭 (기본값을 below로 변경)
    const thresholdSplitData = useMemo(() => {
        if (module.type === ModuleType.SplitByThreshold && module.outputData?.type === 'ThresholdSplitOutput') {
            const output = module.outputData as any;
            // 첫 번째 탭은 첫 번째 출력 포트(belowThreshold), 두 번째 탭은 두 번째 출력 포트(aboveThreshold)
            return activeThresholdSplitTab === 'below' ? output.belowThreshold : output.aboveThreshold;
        }
        return null;
    }, [module, activeThresholdSplitTab]);
    
    // Threshold Analysis 모듈의 경우
    const thresholdAnalysisOutput = useMemo(() => {
        if (module.type === ModuleType.ThresholdAnalysis && module.outputData?.type === 'ThresholdAnalysisOutput') {
            return module.outputData as ThresholdAnalysisOutput;
        }
        return null;
    }, [module]);
    
    const [activeThresholdAnalysisTab, setActiveThresholdAnalysisTab] = useState<'histogram' | 'ecdf' | 'qqplot' | 'meanexcess'>('histogram');
    
    const displayData = thresholdSplitData || data;
    const columns = Array.isArray(displayData?.columns) ? displayData.columns : [];
    const rows = Array.isArray(displayData?.rows) ? displayData.rows : [];
    
    const [sortConfig, setSortConfig] = useState<SortConfig>(null);
    const [selectedColumn, setSelectedColumn] = useState<string | null>(columns[0]?.name || null);
    const [activeXolTab, setActiveXolTab] = useState<'limit' | 'aggreinst'>('limit'); // XoL Calculator 탭
    const [activeLoadClaimDataTab, setActiveLoadClaimDataTab] = useState<'detail' | 'graphs'>('detail'); // Load Claim Data 탭
    const [graphColumn, setGraphColumn] = useState<string | null>(null); // Graphs 탭에서 선택된 열
    const [showSpreadView, setShowSpreadView] = useState(false);
    
    // Spread View용 데이터 변환
    const spreadViewData = useMemo(() => {
        if (!displayData || !displayData.rows || displayData.rows.length === 0) return [];
        return displayData.rows;
    }, [displayData]);
    
    const spreadViewColumns = useMemo(() => {
        if (!displayData || !displayData.columns) return [];
        return displayData.columns.map(col => ({ name: col.name, type: col.type }));
    }, [displayData]);
    
    // CSV 다운로드 함수
    const handleDownloadCSV = useCallback(() => {
        if (!displayData || !displayData.rows || displayData.rows.length === 0) return;
        
        const csvContent = [
            displayData.columns.map(col => col.name).join(','),
            ...displayData.rows.map(row => 
                displayData.columns.map(col => {
                    const value = row[col.name];
                    if (value === null || value === undefined) return '';
                    const str = String(value);
                    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                        return `"${str.replace(/"/g, '""')}"`;
                    }
                    return str;
                }).join(',')
            )
        ].join('\n');
        
        const bom = '\uFEFF';
        const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `${module.name}_data.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }, [displayData, module.name]);
    
    // XoL Calculator 모듈의 경우 데이터 필터링 함수
    const getXolData = (): DataPreview | null => {
        if (!data || module.type !== ModuleType.XolCalculator) return data;
        
        const columnName = activeXolTab === 'limit' 
            ? 'XoL Claim(Incl. Limit)' 
            : 'XoL Claim(Incl. Agg/Reinst)';
        
        // 두 번째 탭 (aggreinst)인 경우 연도별 집계
        if (activeXolTab === 'aggreinst') {
            // XoL Contract 모듈 정보 가져오기
            const contractConnection = allConnections.find(
                (c) => c.to.moduleId === module.id && c.to.portName === 'contract_in'
            );
            const contractModule = contractConnection 
                ? allModules.find((m) => m.id === contractConnection.from.moduleId)
                : null;
            const contract = contractModule?.outputData?.type === 'XolContractOutput' 
                ? contractModule.outputData as any
                : null;
            
            // 연도 컬럼 찾기: 모듈의 year_column 파라미터 우선 사용
            let yearColumn = null;
            const moduleYearColumn = module.parameters?.year_column;
            
            if (moduleYearColumn) {
                // 모듈에서 지정된 연도 컬럼 사용
                yearColumn = data.columns.find(col => col.name === moduleYearColumn);
            } else {
                // 자동 감지: 시뮬레이션 데이터인 경우 시뮬레이션 번호를 연도로 사용
                const hasSimulationNumber = data.columns.some(col => col.name === "시뮬레이션 번호");
                if (hasSimulationNumber) {
                    yearColumn = data.columns.find(col => col.name === "시뮬레이션 번호");
                } else {
                    // 일반 데이터인 경우 연도 컬럼 찾기
                    yearColumn = data.columns.find(col => 
                        col.name === '연도' || 
                        col.name === 'year' || 
                        col.name.toLowerCase() === 'year' ||
                        col.name.toLowerCase().includes('year')
                    );
                }
            }
            
            // 클레임 금액_infl 컬럼 찾기
            const claimAmountColumn = data.columns.find(col => 
                col.name === '클레임 금액_infl' || 
                col.name.endsWith('_infl') ||
                col.name.includes('금액') && col.name.includes('infl')
            );
            
            // XoL Claim(Incl. Limit) 컬럼 찾기
            const xolLimitColumn = data.columns.find(col => 
                col.name === 'XoL Claim(Incl. Limit)'
            );
            
            if (!yearColumn || !data.rows || data.rows.length === 0) {
                // 연도 컬럼이 없으면 빈 데이터 반환
                return {
                    ...data,
                    columns: [],
                    rows: [],
                    totalRowCount: 0
                };
            }
            
            // 연도별 집계: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit)
            const yearMap = new Map<number, { claimAmount: number; xolLimit: number }>();
            const yearColumnName = yearColumn.name;
            const claimAmountColumnName = claimAmountColumn?.name || null;
            const xolLimitColumnName = xolLimitColumn?.name || null;
            
            data.rows.forEach(row => {
                const year = row[yearColumnName];
                
                if (year !== null && year !== undefined) {
                    const yearNum = typeof year === 'number' ? year : parseInt(String(year));
                    
                    if (!isNaN(yearNum)) {
                        const current = yearMap.get(yearNum) || { claimAmount: 0, xolLimit: 0 };
                        
                        // 클레임 금액_infl 합계
                        if (claimAmountColumnName) {
                            const claimValue = row[claimAmountColumnName];
                            if (claimValue !== null && claimValue !== undefined) {
                                const claimNum = typeof claimValue === 'number' ? claimValue : parseFloat(String(claimValue));
                                if (!isNaN(claimNum)) {
                                    current.claimAmount += claimNum;
                                }
                            }
                        }
                        
                        // XoL Claim(Incl. Limit) 합계
                        if (xolLimitColumnName) {
                            const limitValue = row[xolLimitColumnName];
                            if (limitValue !== null && limitValue !== undefined) {
                                const limitNum = typeof limitValue === 'number' ? limitValue : parseFloat(String(limitValue));
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
                        [yearColumnName]: year
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
                        const maxValue = contract.limit * (contract.reinstatements + 1) + contract.aggDeductible;
                        xolClaimAggReinst = totals.xolLimit >= maxValue 
                            ? maxValue 
                            : totals.xolLimit;
                        row['XoL Claim(Incl. Agg/Reinst)'] = xolClaimAggReinst;
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
                                if (i < contract.reinstatements && contract.reinstatementPremiums && contract.reinstatementPremiums[i] !== undefined) {
                                    // 각 limit 배수를 limit로 나눈 값(1)에 복원 프리미엄을 곱함
                                    // 복원 프리미엄은 백분율로 들어오므로 100으로 나눔
                                    totalReinstatementRate += 1 * (contract.reinstatementPremiums[i] / 100);
                                }
                            }
                            
                            // 나머지 limit에 대한 복원 프리미엄 (나머지 limit)
                            if (partialLimit > 0) {
                                const reinstatementIndex = fullLimitCount; // 나머지 limit의 복원 인덱스
                                if (reinstatementIndex < contract.reinstatements && contract.reinstatementPremiums && contract.reinstatementPremiums[reinstatementIndex] !== undefined) {
                                    // 나머지 limit을 limit로 나눈 값(partialLimit)에 복원 프리미엄을 곱함
                                    totalReinstatementRate += partialLimit * (contract.reinstatementPremiums[reinstatementIndex] / 100);
                                }
                            }
                            
                            // 4. 초기 재보험 프리미엄 1에 모든 복원 프리미엄을 더함
                            const xolPremiumRate = 1 + totalReinstatementRate;
                            row['XoL Premium Rate'] = xolPremiumRate;
                        } else {
                            // Agg Deductible보다 작으면 0 반환하는 경우
                            row['XoL Premium Rate'] = 1;
                        }
                    }
                    
                    return row;
                });
            
            // 컬럼 정의: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit), XoL Claim(Incl. Agg/Reinst), XoL Premium Rate
            const aggregatedColumns = [
                { name: yearColumnName, type: yearColumn.type }
            ];
            if (claimAmountColumn) {
                aggregatedColumns.push({ name: claimAmountColumnName!, type: 'number' });
            }
            if (xolLimitColumn) {
                aggregatedColumns.push({ name: xolLimitColumnName!, type: 'number' });
            }
            if (contract) {
                aggregatedColumns.push({ name: 'XoL Claim(Incl. Agg/Reinst)', type: 'number' });
                aggregatedColumns.push({ name: 'XoL Premium Rate', type: 'number' });
            }
            
            return {
                ...data,
                columns: aggregatedColumns,
                rows: aggregatedRows,
                totalRowCount: aggregatedRows.length
            };
        }
        
        // 첫 번째 탭 (limit)인 경우 기본 동작
        const filteredColumns = data.columns.filter(col => 
            col.name === columnName || 
            !col.name.startsWith('XoL Claim')
        );
        
        return {
            ...data,
            columns: filteredColumns
        };
    };
    
    const xolData = module.type === ModuleType.XolCalculator ? getXolData() : data;
    const displayColumns = Array.isArray((module.type === ModuleType.XolCalculator ? xolData : data)?.columns) ? (module.type === ModuleType.XolCalculator ? xolData : data)!.columns : [];
    const displayRows = Array.isArray((module.type === ModuleType.XolCalculator ? xolData : data)?.rows) ? (module.type === ModuleType.XolCalculator ? xolData : data)!.rows : [];

    // Score Model의 label column state
    const predictCol = useMemo(() => {
        try {
            if (module.type === ModuleType.ScoreModel && Array.isArray(columns) && columns.length > 0) {
                return columns.find(c => c && c.name && (c.name === 'Predict' || c.name.toLowerCase().includes('predict'))) || null;
            }
            return null;
        } catch (error) {
            console.error('Error finding predict column:', error);
            return null;
        }
    }, [module.type, columns]);
    
    const labelCols = useMemo(() => {
        try {
            if (module.type === ModuleType.ScoreModel && Array.isArray(columns) && columns.length > 0) {
                return columns.filter(c => c && c.type === 'number' && c.name && c.name !== predictCol?.name);
            }
            return [];
        } catch (error) {
            console.error('Error filtering label columns:', error);
            return [];
        }
    }, [module.type, columns, predictCol]);
    
    const defaultLabelCol = useMemo(() => {
        try {
            return labelCols && labelCols.length > 0 ? (labelCols[0]?.name || null) : null;
        } catch (error) {
            console.error('Error getting default label column:', error);
            return null;
        }
    }, [labelCols]);
    
    const moduleIdRef = useRef<string | null>(null); // 모듈 ID 추적
    const [scoreLabelCol, setScoreLabelCol] = useState<string | null>(null);
    
    // Score Model의 경우 모듈이 변경될 때 초기화
    useEffect(() => {
        try {
            if (module.type === ModuleType.ScoreModel) {
                // 모듈이 변경된 경우만 초기화
                if (moduleIdRef.current !== module.id) {
                    moduleIdRef.current = module.id;
                    setScoreLabelCol(defaultLabelCol);
                }
            } else {
                // Score Model이 아닌 경우 초기화
                if (moduleIdRef.current !== null) {
                    moduleIdRef.current = null;
                    setScoreLabelCol(null);
                }
            }
        } catch (error) {
            console.error('Error in Score Model label column initialization:', error);
        }
    }, [module.id, module.type, defaultLabelCol]); // defaultLabelCol을 의존성에 추가

// PCA Score Visualization 컴포넌트
const PCAScoreVisualization: React.FC<{
    module: CanvasModule;
    rows: Record<string, any>[];
    columns: ColumnInfo[];
    predictCol: ColumnInfo | null;
    labelCols: ColumnInfo[];
    scoreLabelCol: string | null;
    setScoreLabelCol: (col: string | null) => void;
}> = ({ module, rows, columns, predictCol, labelCols, scoreLabelCol, setScoreLabelCol }) => {
    const [pcaData, setPcaData] = useState<{
        coordinates: number[][];
        explainedVariance: number[];
        actualValues?: (string | number)[];
        predictedValues?: (string | number)[];
    } | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [modelType, setModelType] = useState<'classification' | 'regression'>('classification');

    // Feature columns 추출 (Predict 컬럼과 label 컬럼 제외한 숫자형 컬럼)
    const featureColumns = useMemo(() => {
        if (!predictCol) return [];
        return columns
            .filter(col => 
                col.type === 'number' && 
                col.name !== predictCol.name && 
                (!scoreLabelCol || col.name !== scoreLabelCol)
            )
            .map(col => col.name);
    }, [columns, predictCol, scoreLabelCol]);

    // y 컬럼 찾기 (labelCols에서 "y"라는 이름의 컬럼)
    const yCol = useMemo(() => {
        return labelCols.find(col => col.name.toLowerCase() === 'y') || null;
    }, [labelCols]);

    // 실제 값과 예측 값 추출
    const actualValues = useMemo(() => {
        if (!scoreLabelCol) return [];
        return rows.map(row => row[scoreLabelCol]);
    }, [rows, scoreLabelCol]);

    const predictedValues = useMemo(() => {
        if (!predictCol) return [];
        // scoreLabelCol이 predictCol.name이면, y 컬럼을 predicted로 사용
        // 그렇지 않으면 기본 predictCol.name을 사용
        if (scoreLabelCol === predictCol.name && yCol) {
            return rows.map(row => row[yCol.name]);
        }
        return rows.map(row => row[predictCol.name]);
    }, [rows, predictCol, scoreLabelCol, yCol]);

    // Label Column 기본값을 Predict로 설정
    useEffect(() => {
        if (predictCol && !scoreLabelCol) {
            setScoreLabelCol(predictCol.name);
        }
    }, [predictCol, scoreLabelCol]); // predictCol이 변경되거나 scoreLabelCol이 없을 때 실행

    // 모델 타입 판단
    useEffect(() => {
        if (actualValues.length > 0) {
            const firstActual = actualValues[0];
            const isNumeric = typeof firstActual === 'number' || !isNaN(Number(firstActual));
            setModelType(isNumeric ? 'regression' : 'classification');
        }
    }, [actualValues]);

    // PCA 계산 (Label Column은 선택 사항)
    useEffect(() => {
        if (featureColumns.length < 2 || rows.length === 0 || !predictCol) {
            setPcaData(null);
            return;
        }

        setIsLoading(true);
        setError(null);

        // JavaScript 버전 PCA 계산 (비동기 함수)
        try {
            const result = calculatePCA(rows, featureColumns, 2);
            
            // validIndices를 사용하여 유효한 데이터만 필터링
            let validCoords: number[][] = [];
            let validActuals: (string | number)[] = [];
            let validPredicteds: (string | number)[] = [];
            
            if (result.validIndices && result.validIndices.length > 0) {
                // validIndices에 해당하는 데이터만 사용
                validCoords = result.validIndices
                    .map((idx: number) => result.coordinates[idx])
                    .filter((c: number[]) => c && c.length >= 2 && !isNaN(c[0]) && !isNaN(c[1]));
                
                // Label Column이 있는 경우만 actual/predicted 값 추출
                if (scoreLabelCol) {
                    validActuals = result.validIndices
                        .map((idx: number) => actualValues[idx])
                        .filter((v: any) => v !== undefined);
                    
                    validPredicteds = result.validIndices
                        .map((idx: number) => predictedValues[idx])
                        .filter((v: any) => v !== undefined);
                }
            } else {
                // validIndices가 없으면 NaN 필터링
                for (let i = 0; i < result.coordinates.length; i++) {
                    const coord = result.coordinates[i];
                    if (coord && coord.length >= 2 && !isNaN(coord[0]) && !isNaN(coord[1])) {
                        validCoords.push(coord);
                        // Label Column이 있는 경우만 추가
                        if (scoreLabelCol && actualValues[i] !== undefined && predictedValues[i] !== undefined) {
                            validActuals.push(actualValues[i]);
                            validPredicteds.push(predictedValues[i]);
                        }
                    }
                }
            }
            
            if (validCoords.length === 0) {
                throw new Error('No valid data points after filtering');
            }

            setPcaData({
                coordinates: validCoords,
                explainedVariance: result.explainedVariance,
                actualValues: scoreLabelCol ? validActuals : undefined,
                predictedValues: scoreLabelCol ? validPredicteds : undefined,
            });
            setIsLoading(false);
        } catch (err: any) {
            console.error('PCA calculation error:', err);
            setError(err.message || 'Failed to calculate PCA');
            setIsLoading(false);
        }
    }, [rows, featureColumns, predictCol, scoreLabelCol, actualValues, predictedValues]);

    if (!predictCol) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-gray-500">Predict column not found in the data.</p>
            </div>
        );
    }

    if (featureColumns.length < 2) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-gray-500">Need at least 2 numeric feature columns for PCA visualization.</p>
            </div>
        );
    }

    return (
        <div className="w-full h-full flex flex-col gap-3">
            {/* Label Column 선택 (선택 사항) - Predict와 y만 표시 */}
            {(yCol || predictCol) && (
                <div className="flex-shrink-0 flex items-center gap-2 px-2">
                    <label htmlFor="label-select" className="font-semibold text-gray-700 text-sm">
                        Label Column (Optional):
                    </label>
                    <select
                        id="label-select"
                        value={scoreLabelCol || ''}
                        onChange={e => setScoreLabelCol(e.target.value || null)}
                        className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                        {predictCol && (
                            <option key={predictCol.name} value={predictCol.name}>
                                {predictCol.name} (Predict)
                            </option>
                        )}
                        {yCol && (
                            <option key={yCol.name} value={yCol.name}>
                                {yCol.name} (y)
                            </option>
                        )}
                    </select>
                    {scoreLabelCol && (
                        <span className="text-xs text-gray-500">
                            (Compare actual vs predicted values)
                        </span>
                    )}
                </div>
            )}
            
            {/* PCA 스캐터 플롯 */}
            <div className="w-full flex-grow min-h-0 flex items-center justify-center overflow-auto">
                {isLoading ? (
                    <div className="flex flex-col items-center gap-3">
                        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
                        <p className="text-gray-500 font-medium">Calculating PCA...</p>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center h-full gap-2">
                        <p className="text-red-600 font-semibold">Error</p>
                        <p className="text-red-500 text-sm">{error}</p>
                    </div>
                ) : pcaData ? (
                    <PCAScatterPlot
                        coordinates={pcaData.coordinates}
                        actualValues={pcaData.actualValues}
                        predictedValues={pcaData.predictedValues}
                        modelType={scoreLabelCol ? modelType : undefined}
                        explainedVariance={pcaData.explainedVariance}
                        hasLabel={!!scoreLabelCol}
                    />
                ) : (
                    <div className="flex items-center justify-center h-full">
                        <p className="text-gray-500">Calculating PCA visualization...</p>
                    </div>
                )}
            </div>
        </div>
    );
};

    const sortedRows = useMemo(() => {
        try {
            if (!Array.isArray(displayRows)) return [];
        let sortableItems = [...displayRows];
            if (sortConfig !== null && sortConfig.key) {
            sortableItems.sort((a, b) => {
                const valA = a[sortConfig.key];
                const valB = b[sortConfig.key];
                if (valA === null || valA === undefined) return 1;
                if (valB === null || valB === undefined) return -1;
                if (valA < valB) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (valA > valB) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableItems;
        } catch (error) {
            console.error('Error sorting rows:', error);
            return Array.isArray(displayRows) ? displayRows : [];
        }
    }, [displayRows, sortConfig]);

    const requestSort = (key: string) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const selectedColumnData = useMemo(() => {
        try {
            if (!selectedColumn || !Array.isArray(displayRows)) return null;
            return displayRows.map(row => row[selectedColumn]);
        } catch (error) {
            console.error('Error getting selected column data:', error);
            return null;
        }
    }, [selectedColumn, displayRows]);
    
    const selectedColInfo = useMemo(() => {
        try {
            if (!Array.isArray(displayColumns) || !selectedColumn) return null;
            return displayColumns.find(c => c && c.name === selectedColumn) || null;
        } catch (error) {
            console.error('Error finding selected column info:', error);
            return null;
        }
    }, [displayColumns, selectedColumn]);
    
    const isSelectedColNumeric = useMemo(() => selectedColInfo?.type === 'number', [selectedColInfo]);
    
    const numericCols = useMemo(() => {
        try {
            if (!Array.isArray(displayColumns)) return [];
            return displayColumns.filter(c => c && c.type === 'number').map(c => c.name).filter(Boolean);
        } catch (error) {
            console.error('Error getting numeric columns:', error);
            return [];
        }
    }, [displayColumns]);

    // Load Claim Data, Format Change, Apply Inflation, Select Data, Apply Threshold의 경우 graphColumn 초기화
    useEffect(() => {
        const targetTypes = [
            ModuleType.LoadClaimData,
            ModuleType.FormatChange,
            ModuleType.ApplyInflation,
            ModuleType.SelectData,
            ModuleType.ApplyThreshold
        ];
        if (targetTypes.includes(module.type) && numericCols.length > 0 && !graphColumn) {
            setGraphColumn(numericCols[0]);
        }
    }, [module.type, numericCols]);

    // Load Claim Data, Format Change, Apply Inflation, Select Data, Apply Threshold Graphs 탭용 히스토그램 데이터 계산
    const loadClaimDataHistogramData = useMemo(() => {
        const targetTypes = [
            ModuleType.LoadClaimData,
            ModuleType.FormatChange,
            ModuleType.ApplyInflation,
            ModuleType.SelectData,
            ModuleType.ApplyThreshold
        ];
        if (!targetTypes.includes(module.type) || !graphColumn) return null;
        
        const columnData = displayRows.map(r => {
            const val = r[graphColumn];
            return val !== null && val !== undefined ? (typeof val === 'number' ? val : parseFloat(String(val))) : null;
        }).filter(v => v !== null && !isNaN(v)) as number[];

        if (columnData.length === 0) return null;
        
        const min = Math.min(...columnData);
        const max = Math.max(...columnData);
        const numBins = 30; // Setting Threshold와 동일하게 30개 bins
        const binSize = (max - min) / numBins;
        
        const bins: number[] = [];
        const frequencies: number[] = Array(numBins).fill(0);
        
        // bins 배열 생성
        for (let i = 0; i <= numBins; i++) {
            bins.push(min + i * binSize);
        }
        
        // frequencies 계산
        for (const value of columnData) {
            let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
            if (binIndex === numBins) binIndex--;
            if (binIndex >= 0 && binIndex < numBins) {
                frequencies[binIndex]++;
            }
        }
        
        return { bins, frequencies, min, max };
    }, [module.type, graphColumn, displayRows]);

    // XoL Contract 모듈의 경우 특별한 View Details 표시
    if (module.type === ModuleType.DefineXolContract) {
        const contractOutput = module.outputData as any;
        const inputData = getXolContractInputData();
        const { deductible = 0, limit = 0, reinstatements = 5, aggDeductible = 0, expenseRatio = 0, reinstatementPremiums = [] } = contractOutput || {};
        
        // 입력 데이터 요약 계산
        const inputSummary = useMemo(() => {
            if (!inputData || !inputData.rows || inputData.rows.length === 0) {
                return null;
            }
            
            // 클레임 금액 컬럼 찾기
            const claimAmountColumn = inputData.columns.find(col => 
                col.name.includes('금액') || 
                col.name.includes('Amount') || 
                col.name.includes('amount') ||
                (col.type === 'number' && col.name !== '연도' && !col.name.toLowerCase().includes('year'))
            );
            
            // 연도 컬럼 찾기
            const yearColumn = inputData.columns.find(col => 
                col.name === '연도' || 
                col.name === 'year' || 
                col.name.toLowerCase() === 'year' ||
                col.name.toLowerCase().includes('year')
            );
            
            if (!claimAmountColumn) return null;
            
            const claimAmounts = inputData.rows
                .map(row => {
                    const val = row[claimAmountColumn.name];
                    return typeof val === 'number' && !isNaN(val) ? val : 0;
                })
                .filter(amt => amt > 0);
            
            if (claimAmounts.length === 0) return null;
            
            const totalClaims = claimAmounts.length;
            const totalAmount = claimAmounts.reduce((sum, amt) => sum + amt, 0);
            const meanAmount = totalAmount / totalClaims;
            const maxAmount = Math.max(...claimAmounts);
            const minAmount = Math.min(...claimAmounts);
            
            // XoL 적용 계산
            const xolApplied = claimAmounts.map(amt => {
                const excess = Math.max(0, amt - deductible);
                const xolClaim = Math.min(excess, limit);
                return xolClaim;
            });
            
            const totalXolClaim = xolApplied.reduce((sum, amt) => sum + amt, 0);
            const meanXolClaim = totalXolClaim / totalClaims;
            
            // 추가 통계 계산
            const sortedAmounts = [...claimAmounts].sort((a, b) => a - b);
            const medianAmount = sortedAmounts.length > 0 
                ? (sortedAmounts[Math.floor(sortedAmounts.length / 2)] + sortedAmounts[Math.ceil(sortedAmounts.length / 2)]) / 2
                : 0;
            
            const variance = claimAmounts.reduce((acc, val) => acc + Math.pow(val - meanAmount, 2), 0) / totalClaims;
            const stdDev = Math.sqrt(variance);
            
            // XoL 관련 추가 통계
            const excessAmounts = claimAmounts.map(amt => Math.max(0, amt - deductible));
            const totalExcess = excessAmounts.reduce((sum, amt) => sum + amt, 0);
            const meanExcess = totalExcess / totalClaims;
            const retentionAmounts = claimAmounts.map(amt => Math.min(amt, deductible));
            const totalRetention = retentionAmounts.reduce((sum, amt) => sum + amt, 0);
            const meanRetention = totalRetention / totalClaims;
            
            // XoL 적용 건수 통계
            const claimsExceedingDeductible = excessAmounts.filter(amt => amt > 0).length;
            const claimsHittingLimit = xolApplied.filter(amt => amt === limit && limit > 0).length;
            const claimsFullyCovered = xolApplied.filter(amt => amt > 0).length;
            
            // 연도별 통계
            let yearlyStats: Array<{ 
                year: number | string; 
                count: number; 
                totalAmount: number; 
                meanAmount: number; 
                totalExcess: number;
                totalRetention: number;
                totalXolClaim: number;
                xolRatio: number;
            }> = [];
            if (yearColumn) {
                const yearMap = new Map<number | string, { 
                    amounts: number[]; 
                    excesses: number[];
                    retentions: number[];
                    xolClaims: number[] 
                }>();
                inputData.rows.forEach(row => {
                    const year = row[yearColumn.name];
                    const amount = row[claimAmountColumn.name];
                    if (typeof amount === 'number' && !isNaN(amount) && amount > 0) {
                        if (!yearMap.has(year)) {
                            yearMap.set(year, { amounts: [], excesses: [], retentions: [], xolClaims: [] });
                        }
                        const data = yearMap.get(year)!;
                        data.amounts.push(amount);
                        const excess = Math.max(0, amount - deductible);
                        const retention = Math.min(amount, deductible);
                        const xolClaim = Math.min(excess, limit);
                        data.excesses.push(excess);
                        data.retentions.push(retention);
                        data.xolClaims.push(xolClaim);
                    }
                });
                
                yearlyStats = Array.from(yearMap.entries())
                    .map(([year, data]) => {
                        const totalAmount = data.amounts.reduce((sum, amt) => sum + amt, 0);
                        const totalExcess = data.excesses.reduce((sum, amt) => sum + amt, 0);
                        const totalRetention = data.retentions.reduce((sum, amt) => sum + amt, 0);
                        const totalXolClaim = data.xolClaims.reduce((sum, amt) => sum + amt, 0);
                        return {
                            year,
                            count: data.amounts.length,
                            totalAmount,
                            meanAmount: totalAmount / data.amounts.length,
                            totalExcess,
                            totalRetention,
                            totalXolClaim,
                            xolRatio: totalAmount > 0 ? (totalXolClaim / totalAmount) * 100 : 0,
                        };
                    })
                    .sort((a, b) => {
                        const yearA = typeof a.year === 'number' ? a.year : parseInt(String(a.year)) || 0;
                        const yearB = typeof b.year === 'number' ? b.year : parseInt(String(b.year)) || 0;
                        return yearA - yearB;
                    });
            }
            
            return {
                totalClaims,
                totalAmount,
                meanAmount,
                medianAmount,
                stdDev,
                maxAmount,
                minAmount,
                totalExcess,
                meanExcess,
                totalRetention,
                meanRetention,
                totalXolClaim,
                meanXolClaim,
                claimsExceedingDeductible,
                claimsHittingLimit,
                claimsFullyCovered,
                yearlyStats,
                claimAmountColumn: claimAmountColumn.name,
                yearColumn: yearColumn?.name || null,
            };
        }, [inputData, deductible, limit]);
        
        return (
            <>
            <div 
                className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
                onClick={onClose}
            >
                <div 
                    className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
                    onClick={e => e.stopPropagation()}
                >
                    <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                        <h2 className="text-xl font-bold text-gray-800">XoL Contract: {module.name}</h2>
                        <div className="flex items-center gap-2">
                            {inputData && inputData.rows && inputData.rows.length > 0 && (
                                <>
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
                                            if (!inputData || !inputData.rows || inputData.rows.length === 0) return;
                                            const csvContent = [
                                                inputData.columns.map(col => col.name).join(','),
                                                ...inputData.rows.map(row => 
                                                    inputData.columns.map(col => {
                                                        const value = row[col.name];
                                                        if (value === null || value === undefined) return '';
                                                        const str = String(value);
                                                        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                                                            return `"${str.replace(/"/g, '""')}"`;
                                                        }
                                                        return str;
                                                    }).join(',')
                                                )
                                            ].join('\n');
                                            const bom = '\uFEFF';
                                            const blob = new Blob([bom + csvContent], { type: 'text/csv;charset=utf-8;' });
                                            const link = document.createElement('a');
                                            const url = URL.createObjectURL(blob);
                                            link.setAttribute('href', url);
                                            link.setAttribute('download', `${module.name}_data.csv`);
                                            link.style.visibility = 'hidden';
                                            document.body.appendChild(link);
                                            link.click();
                                            document.body.removeChild(link);
                                            URL.revokeObjectURL(url);
                                        }}
                                        className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                                        title="Download CSV"
                                    >
                                        <ArrowDownTrayIcon className="w-5 h-5" />
                                    </button>
                                </>
                            )}
                            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                                <XCircleIcon className="w-6 h-6" />
                            </button>
                        </div>
                    </header>
                    <main className="flex-grow p-6 overflow-auto">
                        {/* XoL Contract 파라미터 */}
                        <div className="mb-6">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">XoL Contract Parameters</h3>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <div className="text-sm text-gray-500 mb-1">Deductible</div>
                                    <div className="text-lg font-semibold text-gray-800">
                                        {deductible.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                    </div>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <div className="text-sm text-gray-500 mb-1">Limit</div>
                                    <div className="text-lg font-semibold text-gray-800">
                                        {limit.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                    </div>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <div className="text-sm text-gray-500 mb-1">Reinstatements</div>
                                    <div className="text-lg font-semibold text-gray-800">{reinstatements}</div>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <div className="text-sm text-gray-500 mb-1">Aggregate Deductible</div>
                                    <div className="text-lg font-semibold text-gray-800">
                                        {aggDeductible.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                    </div>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <div className="text-sm text-gray-500 mb-1">Expense Ratio</div>
                                    <div className="text-lg font-semibold text-gray-800">
                                        {(expenseRatio * 100).toFixed(2)}%
                                    </div>
                                </div>
                                {reinstatementPremiums && reinstatementPremiums.length > 0 && (
                                    <div className="bg-gray-50 p-4 rounded-lg">
                                        <div className="text-sm text-gray-500 mb-1">Reinstatement Premiums</div>
                                        <div className="text-sm font-semibold text-gray-800">
                                            {reinstatementPremiums.map((rate: number, idx: number) => (
                                                <div key={idx}>
                                                    {idx + 1}차: {rate}%
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                        
                        {/* 입력 데이터 요약 */}
                        {inputSummary ? (
                            <>
                                <div className="mb-6">
                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Input Data Summary (입력 데이터 요약)</h3>
                                    
                                    {/* 기본 통계 */}
                                    <div className="mb-4">
                                        <h4 className="text-md font-medium text-gray-700 mb-3">Basic Statistics (기본 통계)</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Total Claims</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.totalClaims.toLocaleString('ko-KR')}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Total Amount</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.totalAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Mean Amount</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.meanAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Median Amount</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.medianAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Std Dev</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.stdDev.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Min Amount</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.minAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                                <div className="text-sm text-blue-600 mb-1">Max Amount</div>
                                                <div className="text-xl font-bold text-blue-800">
                                                    {inputSummary.maxAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    {/* XoL 적용 결과 */}
                                    <div className="mb-4">
                                        <h4 className="text-md font-medium text-gray-700 mb-3">XoL Application Results (XoL 적용 결과)</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                                                <div className="text-sm text-yellow-700 mb-1">Total Retention</div>
                                                <div className="text-lg font-bold text-yellow-800">
                                                    {inputSummary.totalRetention.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                                <div className="text-xs text-yellow-600 mt-1">
                                                    Mean: {inputSummary.meanRetention.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                                                <div className="text-sm text-orange-700 mb-1">Total Excess</div>
                                                <div className="text-lg font-bold text-orange-800">
                                                    {inputSummary.totalExcess.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                                <div className="text-xs text-orange-600 mt-1">
                                                    Mean: {inputSummary.meanExcess.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                                <div className="text-sm text-green-600 mb-1">Total XoL Claim</div>
                                                <div className="text-xl font-bold text-green-800">
                                                    {inputSummary.totalXolClaim.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                                <div className="text-xs text-green-600 mt-1">
                                                    Mean: {inputSummary.meanXolClaim.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                </div>
                                            </div>
                                            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                                <div className="text-sm text-purple-600 mb-1">XoL Ratio</div>
                                                <div className="text-xl font-bold text-purple-800">
                                                    {inputSummary.totalAmount > 0 
                                                        ? ((inputSummary.totalXolClaim / inputSummary.totalAmount) * 100).toFixed(2)
                                                        : '0.00'}%
                                                </div>
                                                <div className="text-xs text-purple-600 mt-1">
                                                    {inputSummary.totalAmount > 0 
                                                        ? ((inputSummary.totalXolClaim / inputSummary.totalAmount) * 100).toFixed(2)
                                                        : '0.00'}% of Total
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    {/* XoL 적용 건수 통계 */}
                                    <div className="mb-4">
                                        <h4 className="text-md font-medium text-gray-700 mb-3">XoL Application Counts (XoL 적용 건수)</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                <div className="text-sm text-gray-600 mb-1">Claims Exceeding Deductible</div>
                                                <div className="text-lg font-bold text-gray-800">
                                                    {inputSummary.claimsExceedingDeductible.toLocaleString('ko-KR')}
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    ({inputSummary.totalClaims > 0 
                                                        ? ((inputSummary.claimsExceedingDeductible / inputSummary.totalClaims) * 100).toFixed(1)
                                                        : '0.0'}% of total)
                                                </div>
                                            </div>
                                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                <div className="text-sm text-gray-600 mb-1">Claims Hitting Limit</div>
                                                <div className="text-lg font-bold text-gray-800">
                                                    {inputSummary.claimsHittingLimit.toLocaleString('ko-KR')}
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    ({inputSummary.totalClaims > 0 
                                                        ? ((inputSummary.claimsHittingLimit / inputSummary.totalClaims) * 100).toFixed(1)
                                                        : '0.0'}% of total)
                                                </div>
                                            </div>
                                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                <div className="text-sm text-gray-600 mb-1">Claims Fully Covered</div>
                                                <div className="text-lg font-bold text-gray-800">
                                                    {inputSummary.claimsFullyCovered.toLocaleString('ko-KR')}
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    ({inputSummary.totalClaims > 0 
                                                        ? ((inputSummary.claimsFullyCovered / inputSummary.totalClaims) * 100).toFixed(1)
                                                        : '0.0'}% of total)
                                                </div>
                                            </div>
                                            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                                                <div className="text-sm text-gray-600 mb-1">Claims Below Deductible</div>
                                                <div className="text-lg font-bold text-gray-800">
                                                    {(inputSummary.totalClaims - inputSummary.claimsExceedingDeductible).toLocaleString('ko-KR')}
                                                </div>
                                                <div className="text-xs text-gray-500 mt-1">
                                                    ({inputSummary.totalClaims > 0 
                                                        ? (((inputSummary.totalClaims - inputSummary.claimsExceedingDeductible) / inputSummary.totalClaims) * 100).toFixed(1)
                                                        : '0.0'}% of total)
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                {/* 연도별 통계 */}
                                {inputSummary.yearlyStats && inputSummary.yearlyStats.length > 0 && (
                                    <div className="mb-6">
                                        <h3 className="text-lg font-semibold text-gray-800 mb-4">Yearly Statistics (연도별 통계)</h3>
                                        <div className="overflow-x-auto">
                                            <table className="min-w-full divide-y divide-gray-200">
                                                <thead className="bg-gray-50">
                                                    <tr>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Year</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Count</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total Amount</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Mean Amount</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total Retention</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total Excess</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total XoL Claim</th>
                                                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">XoL Ratio</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="bg-white divide-y divide-gray-200">
                                                    {inputSummary.yearlyStats.map((stat, idx) => (
                                                        <tr key={idx} className="hover:bg-gray-50">
                                                            <td className="px-4 py-3 text-sm text-gray-900 font-medium">{stat.year}</td>
                                                            <td className="px-4 py-3 text-sm text-gray-900 text-right">
                                                                {stat.count.toLocaleString('ko-KR')}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-gray-900 text-right">
                                                                {stat.totalAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-gray-900 text-right">
                                                                {stat.meanAmount.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-yellow-700 text-right font-medium">
                                                                {stat.totalRetention.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-orange-700 text-right font-medium">
                                                                {stat.totalExcess.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-green-700 text-right font-medium">
                                                                {stat.totalXolClaim.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                                            </td>
                                                            <td className="px-4 py-3 text-sm text-purple-700 text-right font-medium">
                                                                {stat.xolRatio.toFixed(2)}%
                                                            </td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="text-center text-gray-500 p-8">
                                <p>입력 데이터가 연결되지 않았거나 데이터가 없습니다.</p>
                                <p className="text-sm mt-2">XoL Contract에 데이터를 연결하면 요약 정보를 확인할 수 있습니다.</p>
                            </div>
                        )}
                    </main>
                </div>
            </div>
            
            {/* Spread View Modal for XoL Contract */}
            {showSpreadView && inputData && inputData.rows && inputData.rows.length > 0 && (
                <SpreadViewModal
                    onClose={() => setShowSpreadView(false)}
                    data={inputData.rows}
                    columns={inputData.columns.map(col => ({ name: col.name, type: col.type }))}
                    title={`Spread View: ${module.name}`}
                />
            )}
        </>
        );
    }

    if (!data) {
        console.warn('DataPreviewModal: No data available for module', module.id, module.type, module.outputData);
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
                <div className="bg-white p-6 rounded-lg shadow-xl" onClick={e => e.stopPropagation()}>
                    <h3 className="text-lg font-bold">No Data Available</h3>
                    <p>The selected module has no previewable data.</p>
                    <p className="text-sm text-gray-500 mt-2">Module Type: {module.type}</p>
                    <p className="text-sm text-gray-500">Output Data Type: {module.outputData?.type || 'null'}</p>
                </div>
            </div>
        );
    }
    
    return (
        <>
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-7xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Data Preview: {module.name}</h2>
                    <div className="flex items-center gap-2">
                        {spreadViewData && spreadViewData.length > 0 && (
                            <>
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
                                    onClick={handleDownloadCSV}
                                    className="p-1.5 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
                                    title="Download CSV"
                                >
                                    <ArrowDownTrayIcon className="w-5 h-5" />
                                </button>
                            </>
                        )}
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    </div>
                </header>
                <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
                    {/* XoL Calculator 모듈의 경우 탭 구성 */}
                    {module.type === ModuleType.XolCalculator ? (
                    <div className="flex-shrink-0 border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button
                                    onClick={() => setActiveXolTab('limit')}
                                className={`${
                                        activeXolTab === 'limit'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                    건별 XoL 적용
                            </button>
                            <button
                                    onClick={() => setActiveXolTab('aggreinst')}
                                className={`${
                                        activeXolTab === 'aggreinst'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                    연도별 XoL 적용
                            </button>
                        </nav>
                    </div>
                    ) : null}

                    {/* Split By Threshold 모듈의 경우 탭 구성 */}
                    {module.type === ModuleType.SplitByThreshold && module.outputData?.type === 'ThresholdSplitOutput' ? (
                        <div className="flex-shrink-0 border-b border-gray-200">
                            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                                <button
                                    onClick={() => setActiveThresholdSplitTab('below')}
                                    className={`${
                                        activeThresholdSplitTab === 'below'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    &lt; {typeof module.parameters.threshold === 'number' ? module.parameters.threshold.toLocaleString('ko-KR') : (module.parameters.threshold || 1000000).toLocaleString('ko-KR')}
                                </button>
                                <button
                                    onClick={() => setActiveThresholdSplitTab('above')}
                                    className={`${
                                        activeThresholdSplitTab === 'above'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    &gt;= {typeof module.parameters.threshold === 'number' ? module.parameters.threshold.toLocaleString('ko-KR') : (module.parameters.threshold || 1000000).toLocaleString('ko-KR')}
                                </button>
                            </nav>
                        </div>
                    ) : module.type === ModuleType.ThresholdAnalysis && thresholdAnalysisOutput ? (
                        <div className="flex-shrink-0 border-b border-gray-200">
                            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                                <button
                                    onClick={() => setActiveThresholdAnalysisTab('histogram')}
                                    className={`${
                                        activeThresholdAnalysisTab === 'histogram'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Histogram
                                </button>
                                <button
                                    onClick={() => setActiveThresholdAnalysisTab('ecdf')}
                                    className={`${
                                        activeThresholdAnalysisTab === 'ecdf'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    ECDF
                                </button>
                                <button
                                    onClick={() => setActiveThresholdAnalysisTab('qqplot')}
                                    className={`${
                                        activeThresholdAnalysisTab === 'qqplot'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    QQ-Plot
                                </button>
                                <button
                                    onClick={() => setActiveThresholdAnalysisTab('meanexcess')}
                                    className={`${
                                        activeThresholdAnalysisTab === 'meanexcess'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Mean Excess Plot
                                </button>
                            </nav>
                        </div>
                    ) : null}

                    {/* Load Claim Data, Format Change, Apply Inflation, Select Data, Apply Threshold 모듈의 경우 탭 구성 */}
                    {[ModuleType.LoadClaimData, ModuleType.FormatChange, ModuleType.ApplyInflation, ModuleType.SelectData, ModuleType.ApplyThreshold].includes(module.type) ? (
                        <div className="flex-shrink-0 border-b border-gray-200">
                            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                                <button
                                    onClick={() => setActiveLoadClaimDataTab('detail')}
                                    className={`${
                                        activeLoadClaimDataTab === 'detail'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Detail
                                </button>
                                <button
                                    onClick={() => setActiveLoadClaimDataTab('graphs')}
                                    className={`${
                                        activeLoadClaimDataTab === 'graphs'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Graphs
                                </button>
                            </nav>
                        </div>
                    ) : null}

                    {/* XoL Calculator 모듈의 경우 데이터 표시 */}
                    {module.type === ModuleType.XolCalculator ? (
                        <div className="flex-grow flex flex-col gap-4 overflow-y-auto">
                            {!xolData && (
                                <div className="text-center text-gray-500 p-4">
                                    No data available to display.
                                </div>
                            )}
                            {xolData && (
                                <>
                                    {/* 연도별 XoL 적용 탭의 경우 연도별 집계 표시 */}
                                    {activeXolTab === 'aggreinst' ? (
                        <>
                            {(!xolData.rows || xolData.rows.length === 0 || xolData.columns.length === 0) ? (
                                <div className="text-center text-gray-500 p-8">
                                    <p className="text-lg font-semibold mb-2">연도별 데이터가 없습니다.</p>
                                    <p className="text-sm">연도 컬럼이 없거나 데이터가 없습니다. 입력 데이터에 연도 컬럼이 있는지 확인해주세요.</p>
                                </div>
                            ) : (
                                <>
                            <div className="flex justify-between items-center flex-shrink-0">
                                                <div className="text-xs text-gray-600">
                                                    Showing {Math.min((xolData.rows || []).length, 1000)} of {xolData.totalRowCount.toLocaleString()} rows and {xolData.columns.length} columns. Click a column to see details.
                                </div>
                            </div>
                                            
                                            {/* 1. 테이블 표시 영역 */}
                                            <div className="overflow-x-auto overflow-y-auto border border-gray-200 rounded-lg flex-shrink-0" style={{ maxHeight: '400px' }}>
                                                <table className="min-w-full text-sm text-left" style={{ minWidth: 'max-content' }}>
                                            <thead className="bg-gray-50 sticky top-0">
                                                <tr>
                                                            {xolData.columns.map((col, colIndex) => {
                                                                // 처음 10개 열은 항상 보이도록, 나머지는 스크롤
                                                                const isFirstTenColumns = colIndex < 10;
                                                                return (
                                                                    <th 
                                                                        key={col.name} 
                                                                        className={`py-1 px-2 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 ${isFirstTenColumns ? 'sticky left-0 bg-gray-50 z-10' : ''}`}
                                                                        style={isFirstTenColumns ? { 
                                                                            position: 'sticky', 
                                                                            left: `${colIndex * 120}px`,
                                                                            backgroundColor: '#f9fafb',
                                                                            zIndex: 10
                                                                        } : {}}
                                                                        onClick={() => setSelectedColumn(col.name)}
                                                                    >
                                                                        <span className="truncate" title={col.name}>{col.name}</span>
                                                                    </th>
                                                                );
                                                            })}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                        {(xolData.rows || []).slice(0, Math.min(1000, (xolData.rows || []).length)).map((row, rowIndex) => (
                                                    <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                                {xolData.columns.map((col, colIndex) => {
                                                                    const isYearColumn = col.name === '연도' || col.name === 'year' || col.name.toLowerCase() === 'year';
                                                                    const isNumeric = col.type === 'number' && !isYearColumn;
                                                                    const value = row[col.name];
                                                                    const isFirstTenColumns = colIndex < 10;
                                                                    
                                                                    let displayValue: string;
                                                                    if (value === null || value === undefined || value === '') {
                                                                        displayValue = '';
                                                                    } else if (isNumeric) {
                                                                        const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                                                                        if (!isNaN(numValue)) {
                                                                            displayValue = numValue.toLocaleString('ko-KR', { 
                                                                                maximumFractionDigits: 2,
                                                                                minimumFractionDigits: 0
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
                                                                className={`py-1 px-2 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'} ${isNumeric ? 'text-right' : 'text-left'} ${isFirstTenColumns ? 'sticky left-0 bg-white z-10' : ''}`}
                                                                style={isFirstTenColumns ? { 
                                                                    position: 'sticky', 
                                                                    left: `${colIndex * 120}px`,
                                                                    backgroundColor: selectedColumn === col.name ? '#dbeafe' : '#ffffff',
                                                                    zIndex: 10
                                                                } : {}}
                                                                onClick={() => setSelectedColumn(col.name)}
                                                                title={String(row[col.name])}
                                                            >
                                                                            {displayValue === '' ? <i className="text-gray-400">null</i> : displayValue}
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
                                                    (c) => c.to.moduleId === module.id && c.to.portName === 'contract_in'
                                                );
                                                const contractModule = contractConnection 
                                                    ? allModules.find((m) => m.id === contractConnection.from.moduleId)
                                                    : null;
                                                const contract = contractModule?.outputData?.type === 'XolContractOutput' 
                                                    ? contractModule.outputData as any
                                                    : null;
                                                
                                                // XoL Claim(Incl. Agg/Reinst) 컬럼 찾기
                                                const xolClaimAggReinstColumn = xolData.columns.find(col => 
                                                    col.name === 'XoL Claim(Incl. Agg/Reinst)'
                                                );
                                                
                                                // XoL Premium Rate 컬럼 찾기
                                                const xolPremiumRateColumn = xolData.columns.find(col => 
                                                    col.name === 'XoL Premium Rate'
                                                );
                                                
                                                // 통계값 계산
                                                let xolClaimMean = 0;
                                                let xolClaimStdDev = 0;
                                                let xolPremiumMean = 0;
                                                
                                                if (xolClaimAggReinstColumn && xolData.rows && xolData.rows.length > 0) {
                                                    const values = xolData.rows
                                                        .map(row => {
                                                            const val = row[xolClaimAggReinstColumn.name];
                                                            return val !== null && val !== undefined ? (typeof val === 'number' ? val : parseFloat(String(val))) : null;
                                                        })
                                                        .filter(v => v !== null && !isNaN(v)) as number[];
                                                    
                                                    if (values.length > 0) {
                                                        // 평균 계산
                                                        xolClaimMean = values.reduce((sum, v) => sum + v, 0) / values.length;
                                                        
                                                        // 표준편차 계산
                                                        const variance = values.reduce((sum, v) => sum + Math.pow(v - xolClaimMean, 2), 0) / values.length;
                                                        xolClaimStdDev = Math.sqrt(variance);
                                                    }
                                                }
                                                
                                                if (xolPremiumRateColumn && xolData.rows && xolData.rows.length > 0) {
                                                    const values = xolData.rows
                                                        .map(row => {
                                                            const val = row[xolPremiumRateColumn.name];
                                                            return val !== null && val !== undefined ? (typeof val === 'number' ? val : parseFloat(String(val))) : null;
                                                        })
                                                        .filter(v => v !== null && !isNaN(v)) as number[];
                                                    
                                                    if (values.length > 0) {
                                                        // 평균 계산
                                                        xolPremiumMean = values.reduce((sum, v) => sum + v, 0) / values.length;
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
                                                                    {xolClaimMean.toLocaleString('ko-KR', { 
                                                                        maximumFractionDigits: 2,
                                                                        minimumFractionDigits: 2
                                                                    })}
                                                                </div>
                                                            </div>
                                                            
                                                            {/* XoL Claim 표준편차 */}
                                                            <div>
                                                                <div className="text-sm font-semibold text-gray-800 mb-2">
                                                                    XoL Claim 표준편차
                                                                </div>
                                                                <div className="text-lg font-mono text-gray-700">
                                                                    {xolClaimStdDev.toLocaleString('ko-KR', { 
                                                                        maximumFractionDigits: 2,
                                                                        minimumFractionDigits: 2
                                                                    })}
                                                                </div>
                                                            </div>
                                                            
                                                            {/* XoL Premium Rate 평균 */}
                                                            <div>
                                                                <div className="text-sm font-semibold text-gray-800 mb-2">
                                                                    XoL Premium Rate 평균
                                                                </div>
                                                                <div className="text-lg font-mono text-gray-700">
                                                                    {xolPremiumMean.toLocaleString('ko-KR', { 
                                                                        maximumFractionDigits: 2,
                                                                        minimumFractionDigits: 2
                                                                    })}
                                                                </div>
                                                            </div>
                                                            
                                                            {/* Reluctance Factor */}
                                                            <div>
                                                                <div className="text-sm font-semibold text-gray-800 mb-2">
                                                                    Reluctance Factor
                                                                </div>
                                                                <div className="text-lg font-mono text-gray-700">
                                                                    {reluctanceFactor.toLocaleString('ko-KR', { 
                                                                        maximumFractionDigits: 2,
                                                                        minimumFractionDigits: 2
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
                                                const yearColumn = xolData.columns.find(col => 
                                                    col.name === '연도' || 
                                                    col.name === 'year' || 
                                                    col.name.toLowerCase() === 'year'
                                                );
                                                
                                                // XoL Claim(Incl. Agg/Reinst) 컬럼 찾기
                                                const xolClaimAggReinstColumn = xolData.columns.find(col => 
                                                    col.name === 'XoL Claim(Incl. Agg/Reinst)'
                                                );
                                                
                                                return (
                                                    <div className="flex gap-4 flex-shrink-0">
                                                        {/* 왼쪽: 통계량 */}
                                                        {selectedColumn && (
                                                            <div className="w-1/2 flex-shrink-0">
                                                                <div className="h-full border border-gray-200 rounded-lg p-4 overflow-auto">
                                                                    <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} />
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
                            {/* 테이블 영역 - 열 10개까지 바로 보이고 나머지는 스크롤, 행은 1000개까지만 표시 */}
                            <div className="flex-shrink-0 mb-4">
                                <div className="overflow-x-auto overflow-y-auto border border-gray-200 rounded-lg" style={{ maxHeight: '400px' }}>
                                    <table className="min-w-full text-sm text-left" style={{ minWidth: 'max-content' }}>
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                                {displayColumns.map((col, colIndex) => {
                                                                    const isYearColumn = col.name === '연도' || col.name === 'year' || col.name === 'Year' || col.name.toLowerCase() === 'year';
                                                                    const isNumeric = col.type === 'number' && !isYearColumn;
                                                                    const isFirstTenColumns = colIndex < 10;
                                                                    return (
                                                                        <th 
                                                                            key={col.name} 
                                                                            className={`py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 ${isNumeric ? 'text-right' : 'text-left'} ${isFirstTenColumns ? 'sticky left-0 bg-gray-50 z-10' : ''}`}
                                                                            style={isFirstTenColumns ? { 
                                                                                position: 'sticky', 
                                                                                left: `${colIndex * 120}px`,
                                                                                backgroundColor: '#f9fafb',
                                                                                zIndex: 10
                                                                            } : {}}
                                                                            onClick={() => requestSort(col.name)}
                                                                        >
                                                                            <div className={`flex items-center gap-1 ${isNumeric ? 'justify-end' : ''}`}>
                                                                                <span className="truncate" title={col.name}>{col.name}</span>
                                                                                {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                                            </div>
                                                                        </th>
                                                                    );
                                                                })}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sortedRows.slice(0, Math.min(1000, sortedRows.length)).map((row, rowIndex) => (
                                                <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                                    {displayColumns.map((col, colIndex) => {
                                                                        const isYearColumn = col.name === '연도' || col.name === 'year' || col.name === 'Year' || col.name.toLowerCase() === 'year';
                                                                        const isNumeric = col.type === 'number' && !isYearColumn;
                                                                        const value = row[col.name];
                                                                        let displayValue: string;
                                                                        
                                                                        if (value === null || value === undefined) {
                                                                            displayValue = '';
                                                                        } else if (isYearColumn) {
                                                                            // 연도는 천단위 구분 없이 표시
                                                                            displayValue = String(value);
                                                                        } else if (isNumeric) {
                                                                            const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                                                                            if (!isNaN(numValue)) {
                                                                                displayValue = numValue.toLocaleString('ko-KR', {
                                                                                    maximumFractionDigits: 2,
                                                                                    minimumFractionDigits: 0
                                                                                });
                                                                            } else {
                                                                                displayValue = String(value);
                                                                            }
                                                                        } else {
                                                                            displayValue = String(value);
                                                                        }
                                                                        
                                                                        const isFirstTenColumns = colIndex < 10;
                                                                        return (
                                                                            <td 
                                                                                key={col.name} 
                                                                                className={`py-1.5 px-3 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'} ${isNumeric ? 'text-right' : 'text-left'} ${isFirstTenColumns ? 'sticky left-0 bg-white z-10' : ''}`}
                                                                                style={isFirstTenColumns ? { 
                                                                                    position: 'sticky', 
                                                                                    left: `${colIndex * 120}px`,
                                                                                    backgroundColor: selectedColumn === col.name ? '#dbeafe' : '#ffffff',
                                                                                    zIndex: 10
                                                                                } : {}}
                                                                                onClick={() => setSelectedColumn(col.name)}
                                                                                title={String(row[col.name])}
                                                                            >
                                                                                {displayValue === '' ? <i className="text-gray-400">null</i> : displayValue}
                                                                            </td>
                                                                        );
                                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="text-xs text-gray-500 mt-2 text-right">
                                    Showing {Math.min(1000, sortedRows.length)} of {xolData.totalRowCount.toLocaleString()} rows
                                </div>
                            </div>
                            
                            {/* 통계량 */}
                            {selectedColumnData && (
                                <div className="flex-shrink-0 mb-4">
                                    <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} noBorder={true} backgroundColor="bg-gray-50" />
                                </div>
                            )}
                            
                            {/* 하단: 히스토그램 영역 - 보험금과 XoL Claim(Incl. Limit)를 상하로 표시 */}
                            {(() => {
                                const claimColumn = module.parameters?.claim_column;
                                const xolLimitColumn = displayColumns.find(col => col.name === 'XoL Claim(Incl. Limit)');
                                
                                if (!claimColumn || !xolLimitColumn) {
                                    return null;
                                }
                                
                                // 보험금 히스토그램 데이터 계산 (useMemo 대신 일반 계산)
                                const claimColumnData = displayRows.map(r => {
                                    const val = r[claimColumn];
                                    return val !== null && val !== undefined ? (typeof val === 'number' ? val : parseFloat(String(val))) : null;
                                }).filter(v => v !== null && !isNaN(v)) as number[];
                                
                                let claimHistogramData: { bins: number[]; frequencies: number[]; min: number; max: number } | null = null;
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
                                        let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
                                        if (binIndex === numBins) binIndex--;
                                        if (binIndex >= 0 && binIndex < numBins) {
                                            frequencies[binIndex]++;
                                        }
                                    }
                                    
                                    claimHistogramData = { bins, frequencies, min, max };
                                }
                                
                                // XoL Claim(Incl. Limit) 히스토그램 데이터 계산 (useMemo 대신 일반 계산)
                                const xolLimitColumnData = displayRows.map(r => {
                                    const val = r['XoL Claim(Incl. Limit)'];
                                    return val !== null && val !== undefined ? (typeof val === 'number' ? val : parseFloat(String(val))) : null;
                                }).filter(v => v !== null && !isNaN(v)) as number[];
                                
                                let xolLimitHistogramData: { bins: number[]; frequencies: number[]; min: number; max: number } | null = null;
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
                                        let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
                                        if (binIndex === numBins) binIndex--;
                                        if (binIndex >= 0 && binIndex < numBins) {
                                            frequencies[binIndex]++;
                                        }
                                    }
                                    
                                    xolLimitHistogramData = { bins, frequencies, min, max };
                                }
                                
                                // 히스토그램 렌더링 함수
                                const renderHistogram = (histogramData: { bins: number[]; frequencies: number[]; min: number; max: number } | null, title: string) => {
                                    if (!histogramData || !histogramData.bins || !histogramData.frequencies) {
                                        return (
                                            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                                                <p className="text-yellow-800">히스토그램 데이터가 없습니다.</p>
                                            </div>
                                        );
                                    }
                                    
                                    const { bins, frequencies, min, max } = histogramData;
                                    const chartWidth = 800;
                                    const chartHeight = 400;
                                    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
                                    const innerWidth = chartWidth - padding.left - padding.right;
                                    const innerHeight = chartHeight - padding.top - padding.bottom;
                                    
                                    const maxFrequency = Math.max(...frequencies, 1);
                                    const numBins = frequencies.length;
                                    const binWidth = innerWidth / numBins;
                                    
                                    // X축 눈금 (6개)
                                    const numXTicks = 6;
                                    const xTickInterval = Math.max(1, Math.floor(numBins / numXTicks));
                                    
                                    return (
                                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                                            <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
                                            <div className="w-full overflow-x-auto">
                                                <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
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
                                                        const y = padding.top + innerHeight - (ratio * innerHeight);
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
                                                        const barHeight = (freq / maxFrequency) * innerHeight;
                                                        const binStart = bins[idx];
                                                        const binEnd = bins[idx + 1];
                                                        const binCenter = (binStart + binEnd) / 2;
                                                        const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth - (binWidth * 0.9 / 2);
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
                                                        if (idx % xTickInterval !== 0 && idx !== frequencies.length - 1) return null;
                                                        if (idx >= bins.length - 1) return null;
                                                        const binStart = bins[idx];
                                                        const binEnd = bins[idx + 1];
                                                        const binCenter = (binStart + binEnd) / 2;
                                                        const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth;
                                                        
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
                                                                    {binCenter.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
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
                                                        transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
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
                                        {renderHistogram(claimHistogramData, `보험금 (${claimColumn})`)}
                                        
                                        {/* XoL Claim(Incl. Limit) 히스토그램 */}
                                        {renderHistogram(xolLimitHistogramData, 'XoL Claim(Incl. Limit)')}
                                    </div>
                                );
                            })()}
                                    </>
                                )}
                        </>
                    )}
                        </div>
                    ) : module.type === ModuleType.SplitByThreshold && module.outputData?.type === 'ThresholdSplitOutput' ? (
                        <>
                            {/* Split By Threshold 모듈의 경우: 좌측 테이블, 우측 통계량, 하단 그래프 */}
                            <div className="flex justify-between items-center flex-shrink-0">
                                <div className="text-sm text-gray-600">
                                    Showing {Math.min(rows.length, 1000)} of {displayData?.totalRowCount.toLocaleString() || 0} rows and {columns.length} columns. Click a column to see details.
                                </div>
                            </div>
                            
                            {/* 상단: 테이블과 통계량 */}
                            <div className="flex gap-4 flex-shrink-0" style={{ maxHeight: '50vh' }}>
                                {/* 좌측: 테이블 */}
                                <div className="w-1/2 overflow-x-auto overflow-y-auto border border-gray-200 rounded-lg">
                                    <table className="min-w-full text-sm text-left" style={{ minWidth: 'max-content' }}>
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                {columns.map(col => (
                                                    <th 
                                                        key={col.name} 
                                                        className="py-1 px-2 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                        onClick={() => setSelectedColumn(col.name)}
                                                    >
                                                        <span className="truncate" title={col.name}>{col.name}</span>
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {rows.slice(0, 1000).map((row, rowIndex) => (
                                                <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                    {columns.map(col => {
                                                        const isYearColumn = col.name === '연도' || col.name === 'year' || col.name.toLowerCase() === 'year';
                                                        const isNumeric = col.type === 'number' && !isYearColumn;
                                                        const value = row[col.name];
                                                        
                                                        let displayValue: string;
                                                        if (value === null || value === undefined || value === '') {
                                                            displayValue = '';
                                                        } else if (isYearColumn) {
                                                            displayValue = String(value);
                                                        } else if (isNumeric) {
                                                            const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                                                            if (!isNaN(numValue)) {
                                                                displayValue = numValue.toLocaleString('ko-KR', { 
                                                                    maximumFractionDigits: 2,
                                                                    minimumFractionDigits: 0
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
                                                                className={`py-1 px-2 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'} ${isNumeric ? 'text-right' : 'text-left'}`}
                                                                onClick={() => setSelectedColumn(col.name)}
                                                                title={String(row[col.name])}
                                                            >
                                                                {displayValue === '' ? <i className="text-gray-400">null</i> : displayValue}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                
                                {/* 우측: 통계량 (중요한 부분만) */}
                                {selectedColumn && selectedColumnData && (
                                    <div className="w-1/2 flex-shrink-0 bg-gray-50 p-4 rounded-lg overflow-y-auto">
                                        <h3 className="text-sm font-semibold text-gray-800 mb-3">통계량: {selectedColumn}</h3>
                                        {(() => {
                                            const stats = (() => {
                                                const isNull = (v: any) => v === null || v === undefined || v === '';
                                                const nonNullValues = selectedColumnData.filter(v => !isNull(v));
                                                if (nonNullValues.length === 0) return null;
                                                
                                                if (isSelectedColNumeric) {
                                                    const nums = nonNullValues.map(v => typeof v === 'number' ? v : parseFloat(String(v))).filter(v => !isNaN(v));
                                                    if (nums.length === 0) return null;
                                                    const sorted = [...nums].sort((a, b) => a - b);
                                                    const mean = nums.reduce((sum, v) => sum + v, 0) / nums.length;
                                                    const variance = nums.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / nums.length;
                                                    const std = Math.sqrt(variance);
                                                    return {
                                                        count: nums.length,
                                                        mean: mean,
                                                        std: std,
                                                        min: sorted[0],
                                                        max: sorted[sorted.length - 1],
                                                        median: sorted.length % 2 === 0 
                                                            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
                                                            : sorted[Math.floor(sorted.length / 2)]
                                                    };
                                                }
                                                return null;
                                            })();
                                            
                                            if (!stats) {
                                                return <p className="text-sm text-gray-500">통계량을 계산할 수 없습니다.</p>;
                                            }
                                            
                                            return (
                                                <div className="space-y-3">
                                                    <div className="grid grid-cols-2 gap-3">
                                                        <div>
                                                            <div className="text-xs text-gray-600">건수</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.count.toLocaleString()}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-xs text-gray-600">평균</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.mean.toFixed(2)}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-xs text-gray-600">표준편차</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.std.toFixed(2)}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-xs text-gray-600">중앙값</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.median.toFixed(2)}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-xs text-gray-600">최소값</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.min.toLocaleString()}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-xs text-gray-600">최대값</div>
                                                            <div className="text-base font-semibold text-gray-800">{stats.max.toLocaleString()}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })()}
                                    </div>
                                )}
                            </div>
                            
                            {/* 하단: 첫 번째 탭은 연도별 클레임 금액 Plot, 두 번째 탭은 히스토그램 */}
                            {activeThresholdSplitTab === 'below' ? (
                                // 첫 번째 탭: 연도별 클레임 금액 Plot (출력 포트 1의 테이블 데이터 사용)
                                (() => {
                                    if (rows.length === 0 || columns.length === 0) {
                                        return (
                                            <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
                                                <p className="text-sm text-gray-500">표시할 데이터가 없습니다.</p>
                                            </div>
                                        );
                                    }
                                    
                                    // 테이블 데이터에서 연도 컬럼 자동 감지
                                    const yearColumn = columns.find(col => {
                                        const colNameLower = col.name.toLowerCase();
                                        return (colNameLower.includes('연도') || 
                                                colNameLower.includes('year') ||
                                                colNameLower === '연도' ||
                                                colNameLower === 'year') && col.type === 'number';
                                    });
                                    
                                    // 테이블 데이터에서 금액 컬럼 자동 감지
                                    const amountColumn = columns.find(col => {
                                        if (col.type !== 'number') return false;
                                        const colNameLower = col.name.toLowerCase();
                                        // 연도 컬럼이 아니고, 금액/클레임 관련 키워드가 포함된 컬럼
                                        const isYearCol = colNameLower.includes('연도') || colNameLower.includes('year');
                                        const isAmountCol = colNameLower.includes('금액') || 
                                                           colNameLower.includes('amount') || 
                                                           colNameLower.includes('클레임') ||
                                                           colNameLower.includes('claim') ||
                                                           colNameLower.includes('보험금') ||
                                                           colNameLower.includes('loss');
                                        return !isYearCol && isAmountCol;
                                    }) || columns.find(col => {
                                        // 금액 관련 키워드가 없으면 숫자 타입 컬럼 중 연도가 아닌 첫 번째 컬럼
                                        if (col.type !== 'number') return false;
                                        const colNameLower = col.name.toLowerCase();
                                        return !(colNameLower.includes('연도') || colNameLower.includes('year'));
                                    });
                                    
                                    if (!yearColumn || !amountColumn) {
                                        return (
                                            <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
                                                <p className="text-sm text-gray-500">
                                                    {!yearColumn && !amountColumn 
                                                        ? '연도 컬럼과 금액 컬럼을 찾을 수 없습니다.'
                                                        : !yearColumn 
                                                        ? '연도 컬럼을 찾을 수 없습니다.'
                                                        : '금액 컬럼을 찾을 수 없습니다.'}
                                                </p>
                                            </div>
                                        );
                                    }
                                    
                                    // 연도별로 클레임 금액 집계
                                    const yearMap = new Map<number | string, number>();
                                    
                                    rows.forEach(row => {
                                        const year = row[yearColumn.name];
                                        const amount = row[amountColumn.name];
                                        
                                        if (year !== null && year !== undefined && amount !== null && amount !== undefined) {
                                            // 연도 값 처리 (문자열에서 숫자 추출 또는 직접 숫자 변환)
                                            let yearValue: number | string;
                                            if (typeof year === 'number') {
                                                yearValue = year;
                                            } else {
                                                const yearStr = String(year);
                                                // 4자리 숫자 추출 시도
                                                const yearMatch = yearStr.match(/\d{4}/);
                                                if (yearMatch) {
                                                    yearValue = parseInt(yearMatch[0], 10);
                                                } else {
                                                    const parsed = parseFloat(yearStr);
                                                    yearValue = !isNaN(parsed) ? parsed : yearStr;
                                                }
                                            }
                                            
                                            // 금액 값 처리
                                            const amountValue = typeof amount === 'number' ? amount : parseFloat(String(amount));
                                            
                                            if (!isNaN(amountValue) && amountValue > 0) {
                                                const currentTotal = yearMap.get(yearValue) || 0;
                                                yearMap.set(yearValue, currentTotal + amountValue);
                                            }
                                        }
                                    });
                                    
                                    const yearlyData = Array.from(yearMap.entries())
                                        .map(([year, totalAmount]) => ({
                                            year: year,
                                            amount: totalAmount
                                        }))
                                        .sort((a, b) => {
                                            const yearA = typeof a.year === 'number' ? a.year : (typeof a.year === 'string' ? parseFloat(a.year) : 0);
                                            const yearB = typeof b.year === 'number' ? b.year : (typeof b.year === 'string' ? parseFloat(b.year) : 0);
                                            if (isNaN(yearA) || isNaN(yearB)) return 0;
                                            return yearA - yearB;
                                        });
                                    
                                    if (yearlyData.length === 0) {
                                        return (
                                            <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
                                                <p className="text-sm text-gray-500">연도별 데이터를 집계할 수 없습니다. 데이터에 유효한 연도와 금액 값이 있는지 확인해주세요.</p>
                                            </div>
                                        );
                                    }
                                    
                                    // YearlyAmountBarPlot을 위한 데이터 변환
                                    const plotRows = yearlyData.map(d => ({
                                        [yearColumn.name]: d.year,
                                        [amountColumn.name]: d.amount
                                    }));
                                    
                                    return (
                                        <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-white">
                                            <h3 className="text-sm font-semibold text-gray-800 mb-4">
                                                연도별 클레임 금액 ({amountColumn.name})
                                            </h3>
                                            <div className="h-64">
                                                <YearlyAmountBarPlot 
                                                    rows={plotRows} 
                                                    yearColumn={yearColumn.name} 
                                                    amountColumn={amountColumn.name} 
                                                />
                                            </div>
                                        </div>
                                    );
                                })()
                            ) : (
                                // 두 번째 탭: 히스토그램 (출력 포트 2의 테이블 데이터 사용)
                                (() => {
                                    if (rows.length === 0 || columns.length === 0) {
                                        return (
                                            <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
                                                <p className="text-sm text-gray-500">표시할 데이터가 없습니다.</p>
                                            </div>
                                        );
                                    }
                                    
                                    // 테이블 데이터에서 숫자 컬럼 자동 감지 (연도 제외)
                                    const numericColumns = columns.filter(col => {
                                        if (col.type !== 'number') return false;
                                        const colNameLower = col.name.toLowerCase();
                                        // 연도 컬럼 제외
                                        return !(colNameLower.includes('연도') || colNameLower.includes('year'));
                                    });
                                    
                                    if (numericColumns.length === 0) {
                                        return (
                                            <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-gray-50">
                                                <p className="text-sm text-gray-500">히스토그램을 그릴 숫자 컬럼을 찾을 수 없습니다.</p>
                                            </div>
                                        );
                                    }
                                    
                                    // 금액/클레임 관련 컬럼 우선 선택
                                    const amountColumn = numericColumns.find(col => {
                                        const colNameLower = col.name.toLowerCase();
                                        return colNameLower.includes('금액') || 
                                               colNameLower.includes('amount') || 
                                               colNameLower.includes('클레임') ||
                                               colNameLower.includes('claim') ||
                                               colNameLower.includes('보험금') ||
                                               colNameLower.includes('loss');
                                    }) || numericColumns[0]; // 없으면 첫 번째 숫자 컬럼
                                    
                                    return (
                                        <div className="flex-shrink-0 mt-4 border border-gray-200 rounded-lg p-4 bg-white">
                                            <h3 className="text-sm font-semibold text-gray-800 mb-4">
                                                데이터 분포 히스토그램 ({amountColumn.name})
                                            </h3>
                                            <div className="h-64">
                                                <HistogramPlot 
                                                    rows={rows} 
                                                    column={amountColumn.name} 
                                                />
                                            </div>
                                        </div>
                                    );
                                })()
                            )}
                        </>
                    ) : [ModuleType.LoadClaimData, ModuleType.FormatChange, ModuleType.ApplyInflation, ModuleType.SelectData, ModuleType.ApplyThreshold].includes(module.type) ? (
                        <>
                            {/* Load Claim Data 모듈의 경우 */}
                            {activeLoadClaimDataTab === 'detail' ? (
                                <>
                                    {/* Detail 탭: 왼쪽 테이블, 오른쪽 통계량 */}
                                    <div className="flex justify-between items-center flex-shrink-0">
                                        <div className="text-sm text-gray-600">
                                            Showing {Math.min(displayRows.length, 1000)} of {data.totalRowCount.toLocaleString()} rows and {displayColumns.length} columns. Click a column to see details.
                                </div>
                                    </div>
                                    <div className="flex-grow flex gap-4 overflow-hidden">
                                        {/* 왼쪽: 테이블 */}
                                <div className={`overflow-auto ${selectedColumnData ? 'w-1/2' : 'w-full'}`}>
                                    <table className="min-w-full text-sm text-left">
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                        {displayColumns.map(col => (
                                                    <th 
                                                        key={col.name} 
                                                        className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                        onClick={() => requestSort(col.name)}
                                                    >
                                                        <div className="flex items-center gap-1">
                                                            <span className="truncate" title={col.name}>{col.name}</span>
                                                            {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                        </div>
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sortedRows.map((row, rowIndex) => (
                                                <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                            {displayColumns.map(col => {
                                                                const isYearColumn = col.name === '연도' || col.name === 'year' || col.name === 'Year' || col.name.toLowerCase() === 'year';
                                                                const isNumeric = col.type === 'number' && !isYearColumn;
                                                                const value = row[col.name];
                                                                let displayValue: string;
                                                                
                                                                if (value === null || value === undefined) {
                                                                    displayValue = '';
                                                                } else if (isYearColumn) {
                                                                    // 연도는 천단위 구분 없이 표시
                                                                    displayValue = String(value);
                                                                } else if (isNumeric) {
                                                                    const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                                                                    if (!isNaN(numValue)) {
                                                                        displayValue = numValue.toLocaleString('ko-KR', {
                                                                            maximumFractionDigits: 2,
                                                                            minimumFractionDigits: 0
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
                                                                        className={`py-1.5 px-3 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'} ${isNumeric ? 'text-right' : 'text-left'}`}
                                                                        onClick={() => setSelectedColumn(col.name)}
                                                                        title={String(row[col.name])}
                                                                    >
                                                                        {displayValue === '' ? <i className="text-gray-400">null</i> : displayValue}
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
                                        <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} noBorder={true} />
                                    </div>
                                        )}
                                    </div>
                                </>
                            ) : (
                                <>
                                    {/* Graphs 탭: 콤보박스로 열 선택, 히스토그램 표시 */}
                                    <div className="space-y-4">
                                        <div className="flex-shrink-0 flex items-center gap-2 mb-4">
                                            <label htmlFor="graph-column-select" className="font-semibold text-gray-700 text-sm">
                                                Select Column:
                                            </label>
                                            <select
                                                id="graph-column-select"
                                                value={graphColumn || ''}
                                                onChange={e => setGraphColumn(e.target.value || null)}
                                                className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                            >
                                                <option value="">Select a column</option>
                                                {numericCols.map(col => (
                                                    <option key={col} value={col}>{col}</option>
                                                ))}
                                            </select>
                                        </div>
                                        {loadClaimDataHistogramData && loadClaimDataHistogramData.bins && loadClaimDataHistogramData.frequencies ? (
                                            <div className="bg-white border border-gray-200 rounded-lg p-4">
                                                <h3 className="text-lg font-semibold text-gray-800 mb-4">데이터 분포 히스토그램</h3>
                                                {(() => {
                                                    const { bins, frequencies, min, max } = loadClaimDataHistogramData;
                                                    const chartWidth = 800;
                                                    const chartHeight = 400;
                                                    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
                                                    const innerWidth = chartWidth - padding.left - padding.right;
                                                    const innerHeight = chartHeight - padding.top - padding.bottom;

                                                    const maxFrequency = Math.max(...frequencies, 1);
                                                    const numBins = frequencies.length;
                                                    const binWidth = innerWidth / numBins;

                                                    return (
                                                        <div className="w-full overflow-x-auto">
                                                            <svg width={chartWidth} height={chartHeight} className="border border-gray-300 rounded">
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
                                                                    const y = padding.top + innerHeight - (ratio * innerHeight);
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
                                                                    const barHeight = (freq / maxFrequency) * innerHeight;
                                                                    const binStart = bins[idx];
                                                                    const binEnd = bins[idx + 1];
                                                                    const binCenter = (binStart + binEnd) / 2;
                                                                    const x = padding.left + ((binCenter - min) / (max - min)) * innerWidth - (binWidth * 0.9 / 2);
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
                                                                        const xPos = padding.left + ((tick - min) / (max - min)) * innerWidth;
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
                                                                                    {tick.toLocaleString(undefined, { maximumFractionDigits: 2 })}
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
                                                                    transform={`rotate(-90, 15, ${padding.top + innerHeight / 2})`}
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
                                                    {graphColumn ? '히스토그램 데이터가 없습니다.' : '열을 선택해주세요.'}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                        </>
                                    )}
                                </>
                            ) : module.type === ModuleType.ThresholdAnalysis && thresholdAnalysisOutput ? (
                                <>
                                    {/* Threshold Analysis 모듈의 경우: 통계량과 차트 표시 */}
                                    <div className="flex flex-col gap-6">
                                        {/* 통계량 표시 */}
                                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                                            <h3 className="text-lg font-semibold text-gray-800 mb-4">기본 통계량</h3>
                                            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                                                <div>
                                                    <p className="text-sm text-gray-600">최소값</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.min.toLocaleString()}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">최대값</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.max.toLocaleString()}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">평균</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">중앙값</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.median.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">표준편차</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.std.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">Q25</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.q25.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">Q75</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.q75.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">Q90</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.q90.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">Q95</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.q95.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                                <div>
                                                    <p className="text-sm text-gray-600">Q99</p>
                                                    <p className="text-lg font-semibold">{thresholdAnalysisOutput.statistics.q99.toLocaleString(undefined, { maximumFractionDigits: 2 })}</p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* 차트 표시 */}
                                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                                            {activeThresholdAnalysisTab === 'histogram' && thresholdAnalysisOutput.histogram && (
                                                <div>
                                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Histogram</h3>
                                                    <ThresholdHistogramPlot histogram={thresholdAnalysisOutput.histogram} />
                                                </div>
                                            )}
                                            {activeThresholdAnalysisTab === 'ecdf' && thresholdAnalysisOutput.ecdf && (
                                                <div>
                                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">ECDF (Empirical Cumulative Distribution Function)</h3>
                                                    <ECDFPlot ecdf={thresholdAnalysisOutput.ecdf} />
                                                </div>
                                            )}
                                            {activeThresholdAnalysisTab === 'qqplot' && thresholdAnalysisOutput.qqPlot && (
                                                <div>
                                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">QQ-Plot (Quantile-Quantile Plot)</h3>
                                                    <QQPlot qqPlot={thresholdAnalysisOutput.qqPlot} />
                                                </div>
                                            )}
                                            {activeThresholdAnalysisTab === 'meanexcess' && thresholdAnalysisOutput.meanExcessPlot && (
                                                <div>
                                                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Mean Excess Plot</h3>
                                                    <MeanExcessPlot meanExcessPlot={thresholdAnalysisOutput.meanExcessPlot} />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <>
                            {/* 일반 모듈의 경우 */}
                            <div className="flex justify-between items-center flex-shrink-0">
                                <div className="text-sm text-gray-600">
                                    Showing {Math.min(displayRows.length, 1000)} of {data.totalRowCount.toLocaleString()} rows and {displayColumns.length} columns. Click a column to see details.
                                </div>
                            </div>
                            <div className="flex-grow flex gap-4 overflow-hidden">
                                {/* Score Model의 경우 테이블만 표시 */}
                                {module.type === ModuleType.ScoreModel ? (
                                    <div className="w-full overflow-auto border border-gray-200 rounded-lg">
                                        <table className="min-w-full text-sm text-left">
                                            <thead className="bg-gray-50 sticky top-0">
                                                <tr>
                                                    {displayColumns.map(col => (
                                                        <th 
                                                            key={col.name} 
                                                            className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                            onClick={() => requestSort(col.name)}
                                                        >
                                                            <div className="flex items-center gap-1">
                                                                <span className="truncate" title={col.name}>{col.name}</span>
                                                                {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                            </div>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {sortedRows.map((row, rowIndex) => (
                                                    <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                        {displayColumns.map(col => (
                                                            <td 
                                                                key={col.name} 
                                                                className="py-1.5 px-3 font-mono truncate hover:bg-gray-50"
                                                                title={String(row[col.name])}
                                                            >
                                                                {row[col.name] === null ? <i className="text-gray-400">null</i> : String(row[col.name])}
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                </div>
                            ) : (
                                <>
                                <div className={`overflow-auto border border-gray-200 rounded-lg ${selectedColumnData ? 'w-1/2' : 'w-full'}`}>
                                    <table className="min-w-full text-sm text-left">
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                {displayColumns.map(col => (
                                                    <th 
                                                        key={col.name} 
                                                        className="py-2 px-3 font-semibold text-gray-600 cursor-pointer hover:bg-gray-100"
                                                        onClick={() => requestSort(col.name)}
                                                    >
                                                        <div className="flex items-center gap-1">
                                                            <span className="truncate" title={col.name}>{col.name}</span>
                                                            {sortConfig?.key === col.name && (sortConfig.direction === 'ascending' ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />)}
                                                        </div>
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {sortedRows.map((row, rowIndex) => (
                                                <tr key={rowIndex} className="border-b border-gray-200 last:border-b-0">
                                                    {displayColumns.map(col => (
                                                        <td 
                                                            key={col.name} 
                                                            className={`py-1.5 px-3 font-mono truncate ${selectedColumn === col.name ? 'bg-blue-100' : 'hover:bg-gray-50 cursor-pointer'}`}
                                                            onClick={() => setSelectedColumn(col.name)}
                                                            title={String(row[col.name])}
                                                        >
                                                            {row[col.name] === null ? <i className="text-gray-400">null</i> : String(row[col.name])}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                        </div>
                                {selectedColumnData && (
                                    <div className="w-1/2 flex flex-col gap-4">
                                        {isSelectedColNumeric ? (
                                            <HistogramPlot rows={displayRows} column={selectedColumn!} />
                                        ) : (
                                            <div className="w-full h-full p-4 flex flex-col border border-gray-200 rounded-lg items-center justify-center">
                                                <p className="text-gray-500">Plot is not available for non-numeric columns.</p>
                                            </div>
                                        )}
                                        <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} />
                        </div>
                    )}
                                </>
                            )}
                        </div>
                        </>
                    )}

                </main>
            </div>
        </div>
        
        {/* Spread View Modal */}
        {showSpreadView && spreadViewData && spreadViewData.length > 0 && (
            <SpreadViewModal
                onClose={() => setShowSpreadView(false)}
                data={spreadViewData}
                columns={spreadViewColumns}
                title={`Spread View: ${module.name}`}
            />
        )}
    </>
    );
};
