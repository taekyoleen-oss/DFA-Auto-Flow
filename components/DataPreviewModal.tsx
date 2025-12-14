import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { CanvasModule, ColumnInfo, DataPreview, ModuleType } from '../types';
import { XCircleIcon, ChevronUpIcon, ChevronDownIcon, SparklesIcon } from './icons';
import { GoogleGenAI } from "@google/genai";
import { MarkdownRenderer } from './MarkdownRenderer';
// import { calculatePCAForScoreVisualization } from '../utils/pyodideRunner'; // Python 기반 (주석 처리)
import { calculatePCA } from '../utils/pcaCalculator'; // JavaScript 기반 (ml-pca 사용)
import { DataTable } from './SplitDataPreviewModal';

interface DataPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
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
        return rows.map(row => ({
            year: row[yearColumn],
            amount: parseFloat(row[amountColumn]) || 0
        })).filter(d => !isNaN(d.amount));
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

const ClaimAmountHistogramTable: React.FC<{ rows: Record<string, any>[]; column?: string }> = ({ rows, column = '클레임 금액' }) => {
    const claimAmounts = useMemo(() => {
        return rows.map(r => parseFloat(r[column])).filter(v => !isNaN(v));
    }, [rows, column]);

    const { bins, binRanges } = useMemo(() => {
        if (claimAmounts.length === 0) return { bins: [], binRanges: [] };
        
        const min = Math.min(...claimAmounts);
        const max = Math.max(...claimAmounts);
        const numBins = 10;
        const binSize = (max - min) / numBins;
        const bins = Array(numBins).fill(0);
        const binRanges: Array<{ min: number; max: number }> = [];

        for (let i = 0; i < numBins; i++) {
            binRanges.push({
                min: min + i * binSize,
                max: min + (i + 1) * binSize
            });
        }

        for (const value of claimAmounts) {
            let binIndex = binSize > 0 ? Math.floor((value - min) / binSize) : 0;
            if (binIndex === numBins) binIndex--;
            if (binIndex >= 0 && binIndex < numBins) {
                bins[binIndex]++;
            }
        }

        return { bins, binRanges };
    }, [claimAmounts]);

    const total = claimAmounts.length;
    const mean = claimAmounts.reduce((a, b) => a + b, 0) / total;
    const sorted = [...claimAmounts].sort((a, b) => a - b);
    const median = sorted.length % 2 === 0 
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)];
    const stdDev = Math.sqrt(claimAmounts.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / total);

    return (
        <div className="bg-gray-50 rounded-lg p-4">
            <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                    <thead className="bg-gray-100">
                        <tr>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Bin Range</th>
                            <th className="px-4 py-2 text-center font-semibold text-gray-700">Frequency</th>
                            <th className="px-4 py-2 text-center font-semibold text-gray-700">Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {bins.map((count, index) => (
                            <tr key={index} className="border-b border-gray-200">
                                <td className="px-4 py-2 text-gray-600">
                                    {binRanges[index].min.toLocaleString('ko-KR', { maximumFractionDigits: 0 })} - {binRanges[index].max.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}
                                </td>
                                <td className="px-4 py-2 text-center text-gray-700">{count}</td>
                                <td className="px-4 py-2 text-center text-gray-700">
                                    {total > 0 ? ((count / total) * 100).toFixed(2) : 0}%
                                </td>
                            </tr>
                        ))}
                    </tbody>
                    <tfoot className="bg-gray-100 font-semibold">
                        <tr>
                            <td className="px-4 py-2 text-gray-700">Total</td>
                            <td className="px-4 py-2 text-center text-gray-700">{total}</td>
                            <td className="px-4 py-2 text-center text-gray-700">100.00%</td>
                        </tr>
                    </tfoot>
                </table>
            </div>
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white p-3 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Mean</div>
                    <div className="text-lg font-semibold text-gray-800">{mean.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white p-3 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Median</div>
                    <div className="text-lg font-semibold text-gray-800">{median.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white p-3 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Std Dev</div>
                    <div className="text-lg font-semibold text-gray-800">{stdDev.toLocaleString('ko-KR', { maximumFractionDigits: 0 })}</div>
                </div>
                <div className="bg-white p-3 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Total Claims</div>
                    <div className="text-lg font-semibold text-gray-800">{total}</div>
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
    
    // 이진 분류인지 감지 (0과 1만 있는지 확인)
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
        
        // 이진 분류인 경우 간소한 색상 사용
        if (isBinaryClassification) {
            if (classKey === '0') {
                // 실제값 0: 파랑 계열
                return isCorrect ? '#2563eb' : '#93c5fd'; // 맞으면 진한 파랑, 틀리면 연한 파랑
            } else if (classKey === '1') {
                // 실제값 1: 빨강 계열
                return isCorrect ? '#dc2626' : '#fca5a5'; // 맞으면 진한 빨강, 틀리면 연한 빨강
            }
        }
        
        // 다중 클래스 분류: 클래스별 기본 색상
        const classColors: Record<string, string> = {
            '0': '#2563eb', // 진한 파랑
            '1': '#dc2626', // 진한 빨강
            '2': '#16a34a', // 진한 초록
            '3': '#ea580c', // 진한 주황
            '4': '#9333ea', // 진한 보라
            '5': '#0891b2', // 청록
            '6': '#ca8a04', // 노랑
            '7': '#e11d48', // 분홍
        };
        
        const baseColor = classColors[classKey] || '#6b7280';
        return isCorrect ? baseColor : `${baseColor}60`; // 틀린 경우 더 투명하게
    };

    // 회귀 모델: 실제 값에 따른 색상 (gradient) - 더 나은 색상 스케일
    const getColorForRegression = (actual: number) => {
        const allActuals = (actualValues || []).filter(v => typeof v === 'number') as number[];
        if (allActuals.length === 0) return '#6366f1';
        
        const minVal = Math.min(...allActuals);
        const maxVal = Math.max(...allActuals);
        const range = maxVal - minVal || 1;
        const normalized = (actual - minVal) / range;
        
        // 파랑(낮음) -> 보라 -> 빨강(높음) gradient (더 부드러운 전환)
        if (normalized < 0.5) {
            // 파랑 -> 보라
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
    
    // Label이 없을 때 사용할 색상 (단일 색상)
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
    
    // 클래스별 통계 (범례용)
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
    
    // 그리드 라인을 위한 tick 값
    const gridXTicks = getTicks(xMin, xMax, 6);
    const gridYTicks = getTicks(yMin, yMax, 6);

    return (
        <div className="w-full h-full flex flex-col items-center gap-3 p-4">
            {/* 설명된 분산 비율 표시 */}
            <div className="w-full max-w-6xl bg-indigo-50 border border-indigo-200 rounded-lg p-3">
                <div className="text-sm font-semibold text-indigo-900 mb-1">Explained Variance</div>
                <div className="flex items-center gap-4 text-xs text-indigo-700">
                    <span>Total: <strong>{totalVariance.toFixed(1)}%</strong></span>
                    <span>PC1: <strong>{(explainedVariance[0] * 100).toFixed(1)}%</strong></span>
                    <span>PC2: <strong>{(explainedVariance[1] * 100).toFixed(1)}%</strong></span>
                </div>
            </div>
            
            <div className="w-full max-w-6xl">
                {/* 그래프 영역 */}
                <div className="w-full">
                    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
                        {/* 그리드 라인 */}
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
                                        size = isBinaryClassification ? 6 : 5; // 이진 분류는 조금 더 크게
                                        
                                        // 이진 분류: 예측이 틀린 경우 테두리 추가
                                        if (isBinaryClassification && String(d.actual) !== String(d.predicted)) {
                                            stroke = '#000';
                                            strokeWidth = 1.5;
                                            opacity = 0.9; // 틀린 경우 더 선명하게
                                        } else if (!isBinaryClassification && String(d.actual) !== String(d.predicted)) {
                                            // 다중 클래스: 틀린 경우 검은 테두리
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
            
            {/* 하단 설명 */}
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

const ColumnStatistics: React.FC<{ data: (string | number | null)[]; columnName: string | null; isNumeric: boolean; }> = ({ data, columnName, isNumeric }) => {
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

    return (
        <div className="w-full p-4 border border-gray-200 rounded-lg">
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


export const DataPreviewModal: React.FC<DataPreviewModalProps> = ({ module, projectName, onClose }) => {
    // ThresholdSplitOutput의 경우 별도 처리
    const thresholdOutput = module?.outputData?.type === 'ThresholdSplitOutput' 
        ? (module.outputData as any) 
        : null;
    
    const [activeThresholdTab, setActiveThresholdTab] = useState<'below' | 'above'>('below');
    const [activeTab, setActiveTab] = useState<'table' | 'visualization'>('table');
    
    // 안전한 데이터 가져오기
    const getPreviewData = (): DataPreview | null => {
        try {
            if (!module || !module.outputData) return null;
        if (module.outputData.type === 'DataPreview') return module.outputData;
        if (module.outputData.type === 'ClaimDataOutput') return module.outputData.data || null;
        if (module.outputData.type === 'InflatedDataOutput') return module.outputData.data || null;
        if (module.outputData.type === 'FormatChangeOutput') return module.outputData.data || null;
        if (module.outputData.type === 'KMeansOutput' || module.outputData.type === 'HierarchicalClusteringOutput' || module.outputData.type === 'DBSCANOutput') {
                return module.outputData.clusterAssignments || null;
        }
        if (module.outputData.type === 'PCAOutput') {
                return module.outputData.transformedData || null;
        }
        return null;
        } catch (error) {
            console.error('Error in getPreviewData:', error);
            return null;
        }
    };
    
    const data = getPreviewData();
    const columns = Array.isArray(data?.columns) ? data.columns : [];
    const rows = Array.isArray(data?.rows) ? data.rows : [];
    
    const [sortConfig, setSortConfig] = useState<SortConfig>(null);
    // LoadClaimData/ApplyInflation/FormatChange/SelectData 모듈의 경우 적절한 컬럼을 기본 선택
    const defaultSelectedColumn = useMemo(() => {
        if (module.type === ModuleType.LoadClaimData || module.type === ModuleType.FormatChange) {
            const claimAmountCol = columns.find(c => c.name === '클레임 금액');
            return claimAmountCol?.name || columns[0]?.name || null;
        } else if (module.type === ModuleType.ApplyInflation) {
            // ApplyInflation: amount_column 파라미터에서 컬럼명 가져와서 "_infl" 붙인 컬럼 찾기
            const amountColumn = (module.parameters?.amount_column as string) || '클레임 금액';
            const inflatedColumnName = `${amountColumn}_infl`;
            const inflatedCol = columns.find(c => c.name === inflatedColumnName);
            return inflatedCol?.name || columns.find(c => c.name.endsWith('_infl'))?.name || columns[0]?.name || null;
        } else if (module.type === ModuleType.SelectData) {
            // SelectData: 첫 번째 숫자 컬럼을 찾거나, "클레임 금액" 또는 "_infl"로 끝나는 컬럼을 찾음
            const numericCols = columns.filter(c => c.type === 'number').map(c => c.name);
            const claimAmountCol = columns.find(c => c.name === '클레임 금액' || c.name.endsWith('_infl'))?.name;
            return claimAmountCol || numericCols[0] || columns[0]?.name || null;
        }
        return columns[0]?.name || null;
    }, [module.type, module.parameters, columns]);
    const [selectedColumn, setSelectedColumn] = useState<string | null>(defaultSelectedColumn);
    const [yAxisCol, setYAxisCol] = useState<string | null>(null);

    // Score Model용 label column state
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
    
    // Score Model의 경우 모듈이 변경될 때만 초기화
    useEffect(() => {
        try {
            if (module.type === ModuleType.ScoreModel) {
                // 모듈이 변경된 경우에만 초기화
                if (moduleIdRef.current !== module.id) {
                    moduleIdRef.current = module.id;
                    setScoreLabelCol(defaultLabelCol);
                }
            } else {
                // Score Model이 아닌 경우 리셋
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
        // 그렇지 않으면 항상 predictCol.name을 사용
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

    // 모델 타입 감지
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

        // JavaScript 기반 PCA 계산 (동기 함수)
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
                
                // Label Column이 있는 경우에만 actual/predicted 값 추출
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
                        // Label Column이 있는 경우에만 추가
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
            
            {/* PCA 그래프 영역 */}
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
            if (!Array.isArray(rows)) return [];
        let sortableItems = [...rows];
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
            return Array.isArray(rows) ? rows : [];
        }
    }, [rows, sortConfig]);

    const requestSort = (key: string) => {
        let direction: 'ascending' | 'descending' = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const selectedColumnData = useMemo(() => {
        try {
            if (!selectedColumn || !Array.isArray(rows)) return null;
            return rows.map(row => row[selectedColumn]);
        } catch (error) {
            console.error('Error getting selected column data:', error);
            return null;
        }
    }, [selectedColumn, rows]);
    
    const selectedColInfo = useMemo(() => {
        try {
            if (!Array.isArray(columns) || !selectedColumn) return null;
            return columns.find(c => c && c.name === selectedColumn) || null;
        } catch (error) {
            console.error('Error finding selected column info:', error);
            return null;
        }
    }, [columns, selectedColumn]);
    
    const isSelectedColNumeric = useMemo(() => selectedColInfo?.type === 'number', [selectedColInfo]);
    
    const numericCols = useMemo(() => {
        try {
            if (!Array.isArray(columns)) return [];
            return columns.filter(c => c && c.type === 'number').map(c => c.name).filter(Boolean);
        } catch (error) {
            console.error('Error getting numeric columns:', error);
            return [];
        }
    }, [columns]);

    useEffect(() => {
        if (isSelectedColNumeric && selectedColumn) {
            const defaultY = numericCols.find(c => c !== selectedColumn);
            setYAxisCol(defaultY || null);
        } else {
            setYAxisCol(null);
        }
    }, [isSelectedColNumeric, selectedColumn, numericCols]);

    // ThresholdSplitOutput의 경우 별도 처리
    if (thresholdOutput) {
        const threshold = thresholdOutput.threshold || 0;
        const belowData = thresholdOutput.belowThreshold;
        const aboveData = thresholdOutput.aboveThreshold;
        
        // belowData에서 연도와 금액 컬럼 찾기
        const belowColumns = belowData?.columns || [];
        const yearColumn = belowColumns.find((c: ColumnInfo) => c.name === '연도' || c.name === 'year' || c.name.toLowerCase().includes('year'))?.name || belowColumns[0]?.name;
        const amountColumn = belowColumns.find((c: ColumnInfo) => c.name.includes('금액') || c.name.includes('amount') || c.name.includes('_infl'))?.name || belowColumns.find((c: ColumnInfo) => c.type === 'number')?.name;
        
        // aboveData에서 금액 컬럼 찾기
        const aboveColumns = aboveData?.columns || [];
        const aboveAmountColumn = aboveColumns.find((c: ColumnInfo) => c.name.includes('금액') || c.name.includes('amount') || c.name.includes('_infl'))?.name || aboveColumns.find((c: ColumnInfo) => c.type === 'number')?.name;
        
        const belowRows = belowData?.rows || [];
        const aboveRows = aboveData?.rows || [];
        
        // belowData의 금액 컬럼 데이터 추출 (통계량용)
        const belowAmountData = yearColumn && amountColumn ? belowRows.map((r: any) => r[amountColumn]).filter((v: any) => v !== null && v !== undefined) : [];
        const belowAmountColumnInfo = belowColumns.find((c: ColumnInfo) => c.name === amountColumn);
        const isBelowAmountNumeric = belowAmountColumnInfo?.type === 'number';
        
        // aboveData의 금액 컬럼 데이터 추출 (통계량용)
        const aboveAmountData = aboveAmountColumn ? aboveRows.map((r: any) => r[aboveAmountColumn]).filter((v: any) => v !== null && v !== undefined) : [];
        const aboveAmountColumnInfo = aboveColumns.find((c: ColumnInfo) => c.name === aboveAmountColumn);
        const isAboveAmountNumeric = aboveAmountColumnInfo?.type === 'number';
        
        return (
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
                            <button
                                onClick={() => {
                                    const currentData = activeThresholdTab === 'below' ? belowData : aboveData;
                                    if (!currentData || !currentData.rows || currentData.rows.length === 0) return;
                                    const csvContent = [
                                        currentData.columns.map(c => c.name).join(','),
                                        ...currentData.rows.map(row => 
                                            currentData.columns.map(col => {
                                                const val = row[col.name];
                                                if (val === null || val === undefined) return '';
                                                const str = String(val);
                                                return str.includes(',') || str.includes('"') || str.includes('\n') 
                                                    ? `"${str.replace(/"/g, '""')}"` 
                                                    : str;
                                            }).join(',')
                                        )
                                    ].join('\n');
                                    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                                    const link = document.createElement('a');
                                    link.href = URL.createObjectURL(blob);
                                    link.download = `${module.name}_${activeThresholdTab === 'below' ? 'below_threshold' : 'above_threshold'}.csv`;
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
                    <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
                        <div className="flex-shrink-0 border-b border-gray-200">
                            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                                <button
                                    onClick={() => setActiveThresholdTab('below')}
                                    className={`${
                                        activeThresholdTab === 'below'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Threshold &lt; {threshold.toLocaleString()}
                                </button>
                                <button
                                    onClick={() => setActiveThresholdTab('above')}
                                    className={`${
                                        activeThresholdTab === 'above'
                                            ? 'border-indigo-500 text-indigo-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                                >
                                    Threshold &gt;= {threshold.toLocaleString()}
                                </button>
                            </nav>
                        </div>
                        
                        {activeThresholdTab === 'below' && belowData && (
                            <div className="flex-grow flex gap-4 overflow-hidden">
                                {/* 왼쪽: 테이블 */}
                                <div className="w-1/2 overflow-auto border border-gray-200 rounded-lg">
                                    <DataTable title={`Threshold < ${threshold.toLocaleString()}`} data={belowData} />
                                </div>
                                
                                {/* 오른쪽: 그래프와 통계량 */}
                                <div className="w-1/2 flex flex-col gap-4">
                                    {/* 오른쪽 상단: 연도별 금액 그래프 */}
                                    {yearColumn && amountColumn && (
                                        <div className="flex-grow min-h-0 border border-gray-200 rounded-lg">
                                            <YearlyAmountBarPlot 
                                                rows={belowRows} 
                                                yearColumn={yearColumn} 
                                                amountColumn={amountColumn} 
                                            />
                                        </div>
                                    )}
                                    
                                    {/* 오른쪽 하단: 통계량 */}
                                    {amountColumn && (
                                        <div className="flex-shrink-0">
                                            <ColumnStatistics 
                                                data={belowAmountData} 
                                                columnName={amountColumn} 
                                                isNumeric={isBelowAmountNumeric} 
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        
                        {activeThresholdTab === 'above' && aboveData && (
                            <div className="flex-grow flex gap-4 overflow-hidden">
                                {/* 왼쪽: 테이블 */}
                                <div className="w-1/2 overflow-auto border border-gray-200 rounded-lg">
                                    <DataTable title={`Threshold >= ${threshold.toLocaleString()}`} data={aboveData} />
                                </div>
                                
                                {/* 오른쪽: 그래프와 통계량 */}
                                <div className="w-1/2 flex flex-col gap-4">
                                    {/* 오른쪽 상단: 금액에 대한 건수 그래프 */}
                                    {aboveAmountColumn && (
                                        <div className="flex-grow min-h-0 border border-gray-200 rounded-lg">
                                            <HistogramPlot 
                                                rows={aboveRows} 
                                                column={aboveAmountColumn} 
                                            />
                                        </div>
                                    )}
                                    
                                    {/* 오른쪽 하단: 통계량 */}
                                    {aboveAmountColumn && (
                                        <div className="flex-shrink-0">
                                            <ColumnStatistics 
                                                data={aboveAmountData} 
                                                columnName={aboveAmountColumn} 
                                                isNumeric={isAboveAmountNumeric} 
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </main>
                </div>
            </div>
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
                        <button
                            onClick={() => {
                                const csvContent = [
                                    columns.map(c => c.name).join(','),
                                    ...rows.map(row => 
                                        columns.map(col => {
                                            const val = row[col.name];
                                            if (val === null || val === undefined) return '';
                                            const str = String(val);
                                            return str.includes(',') || str.includes('"') || str.includes('\n') 
                                                ? `"${str.replace(/"/g, '""')}"` 
                                                : str;
                                        }).join(',')
                                    )
                                ].join('\n');
                                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                                const link = document.createElement('a');
                                link.href = URL.createObjectURL(blob);
                                link.download = `${module.name}_data.csv`;
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
                <main className="flex-grow p-4 overflow-auto flex flex-col gap-4">
                    <div className="flex-shrink-0 border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button
                                onClick={() => setActiveTab('table')}
                                className={`${
                                    activeTab === 'table'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                Data Table
                            </button>
                            <button
                                onClick={() => setActiveTab('visualization')}
                                className={`${
                                    activeTab === 'visualization'
                                        ? 'border-indigo-500 text-indigo-600'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
                            >
                                {(module.type === ModuleType.LoadClaimData || module.type === ModuleType.ApplyInflation || module.type === ModuleType.FormatChange || module.type === ModuleType.SelectData) ? 'Claim Amount Histogram' : 'PCA Visualization'}
                            </button>
                        </nav>
                    </div>

                    {activeTab === 'table' && (
                        <>
                            <div className="flex justify-between items-center flex-shrink-0">
                                <div className="text-sm text-gray-600">
                                    Showing {Math.min(rows.length, 1000)} of {data.totalRowCount.toLocaleString()} rows and {columns.length} columns. Click a column to see details.
                                </div>
                            </div>
                            <div className="flex-grow flex gap-4 overflow-hidden">
                                {/* Score Model인 경우 테이블만 표시 */}
                                {module.type === ModuleType.ScoreModel ? (
                                    <div className="w-full overflow-auto border border-gray-200 rounded-lg">
                                        <table className="min-w-full text-sm text-left">
                                            <thead className="bg-gray-50 sticky top-0">
                                                <tr>
                                                    {columns.map(col => (
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
                                                        {columns.map(col => {
                                                            const isYearColumn = col.name === '연도' || col.name === 'year' || col.name.toLowerCase() === 'year';
                                                            const isNumeric = col.type === 'number' && !isYearColumn;
                                                            const value = row[col.name];
                                                            
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
                                                                    className={`py-1.5 px-3 font-mono truncate hover:bg-gray-50 ${isNumeric ? 'text-right' : 'text-left'}`}
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
                                ) : (
                                    <>
                                <div className={`overflow-auto border border-gray-200 rounded-lg ${selectedColumn ? 'w-1/2' : 'w-full'}`}>
                                    <table className="min-w-full text-sm text-left">
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                {columns.map(col => (
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
                                                    {columns.map(col => {
                                                        const isYearColumn = col.name === '연도' || col.name === 'year' || col.name.toLowerCase() === 'year';
                                                        const isNumeric = col.type === 'number' && !isYearColumn;
                                                        const value = row[col.name];
                                                        
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
                                {selectedColumn && (
                                    <div className="w-1/2">
                                        {(module.type === ModuleType.LoadClaimData || 
                                          module.type === ModuleType.ApplyInflation || 
                                          module.type === ModuleType.FormatChange || 
                                          module.type === ModuleType.SelectData) ? (
                                            // 4개 모듈: 통계량만 표시 (그래프 제거)
                                            <div className="h-full border border-gray-200 rounded-lg p-4 overflow-auto">
                                                <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} />
                                            </div>
                                        ) : (
                                            // 다른 모듈: 기존 레이아웃 유지 (그래프 + 통계량)
                                            <div className="w-full flex flex-col gap-4">
                                                <div className="flex-grow min-h-0 border border-gray-200 rounded-lg p-4">
                                                    {isSelectedColNumeric ? (
                                                        <HistogramPlot rows={rows} column={selectedColumn} />
                                                    ) : (
                                                        <div className="w-full h-full flex flex-col items-center justify-center">
                                                            <p className="text-gray-500">Plot is not available for non-numeric columns.</p>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="flex-shrink-0 border border-gray-200 rounded-lg p-4">
                                                    <ColumnStatistics data={selectedColumnData} columnName={selectedColumn} isNumeric={isSelectedColNumeric} />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                                    </>
                                )}
                            </div>
                        </>
                    )}

                    {activeTab === 'visualization' && (
                        <div className="flex-grow flex flex-col items-center justify-start p-4 gap-4">
                            {/* Score Model인 경우 PCA Visualization */}
                            {module.type === ModuleType.ScoreModel ? (
                                <PCAScoreVisualization 
                                    module={module}
                                    rows={rows}
                                    columns={columns}
                                    predictCol={predictCol}
                                    labelCols={labelCols}
                                    scoreLabelCol={scoreLabelCol}
                                    setScoreLabelCol={setScoreLabelCol}
                                />
                            ) : (module.type === ModuleType.LoadClaimData || module.type === ModuleType.ApplyInflation || module.type === ModuleType.FormatChange || module.type === ModuleType.SelectData) ? (
                                <>
                                    {/* LoadClaimData/ApplyInflation/FormatChange/SelectData: 클레임 금액 히스토그램 */}
                                    <div className="w-full">
                                        <h3 className="text-lg font-semibold text-gray-800 mb-4">
                                            {(() => {
                                                if (module.type === ModuleType.ApplyInflation) {
                                                    return "Inflated Claim Amount Distribution";
                                                } else {
                                                    // LoadClaimData, FormatChange, SelectData는 동일하게 표시
                                                    return "Claim Amount Distribution";
                                                }
                                            })()}
                                        </h3>
                                        <div className="w-full h-96 mb-6">
                                            <HistogramPlot 
                                                rows={rows} 
                                                column={(() => {
                                                    if (module.type === ModuleType.ApplyInflation) {
                                                        const amountColumn = (module.parameters?.amount_column as string) || '클레임 금액';
                                                        const inflatedColumnName = `${amountColumn}_infl`;
                                                        return columns.find(c => c.name === inflatedColumnName)?.name || columns.find(c => c.name.endsWith('_infl'))?.name || '클레임 금액';
                                                    } else {
                                                        // LoadClaimData, FormatChange, SelectData는 "클레임 금액" 또는 "_infl"로 끝나는 컬럼 사용
                                                        const claimAmountCol = columns.find(c => c.name === '클레임 금액' || c.name.endsWith('_infl'))?.name;
                                                        return claimAmountCol || '클레임 금액';
                                                    }
                                                })()} 
                                            />
                                        </div>
                                        
                                        {/* 히스토그램 통계 표 */}
                                        <div className="mt-6">
                                            <h4 className="text-md font-semibold text-gray-700 mb-3">Histogram Statistics</h4>
                                            <ClaimAmountHistogramTable 
                                                rows={rows} 
                                                column={(() => {
                                                    if (module.type === ModuleType.ApplyInflation) {
                                                        const amountColumn = (module.parameters?.amount_column as string) || '클레임 금액';
                                                        const inflatedColumnName = `${amountColumn}_infl`;
                                                        return columns.find(c => c.name === inflatedColumnName)?.name || columns.find(c => c.name.endsWith('_infl'))?.name || '클레임 금액';
                                                    } else {
                                                        // LoadClaimData, FormatChange, SelectData는 "클레임 금액" 또는 "_infl"로 끝나는 컬럼 사용
                                                        const claimAmountCol = columns.find(c => c.name === '클레임 금액' || c.name.endsWith('_infl'))?.name;
                                                        return claimAmountCol || '클레임 금액';
                                                    }
                                                })()}
                                            />
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <>
                            {!selectedColumn ? (
                                <div className="flex items-center justify-center h-full">
                                    <p className="text-gray-500">Select a column from the Data Table to use as the X-axis.</p>
                                </div>
                            ) : (
                                <>
                                    {isSelectedColNumeric && numericCols.length > 1 && (
                                        <div className="flex-shrink-0 flex items-center gap-2 self-start">
                                            <label htmlFor="y-axis-select" className="font-semibold text-gray-700">Y-Axis:</label>
                                            <select
                                                id="y-axis-select"
                                                value={yAxisCol || ''}
                                                onChange={e => setYAxisCol(e.target.value)}
                                                className="p-2 border border-gray-300 rounded-md"
                                            >
                                                <option value="" disabled>Select a column</option>
                                                {numericCols.filter(c => c !== selectedColumn).map(col => (
                                                    <option key={col} value={col}>{col}</option>
                                                ))}
                                            </select>
                                        </div>
                                    )}

                                    <div className="w-full flex-grow min-h-0">
                                        {isSelectedColNumeric ? (
                                            yAxisCol ? (
                                                <ScatterPlot rows={rows} xCol={selectedColumn} yCol={yAxisCol} />
                                            ) : (
                                                <HistogramPlot rows={rows} column={selectedColumn} />
                                            )
                                        ) : (
                                            <HistogramPlot rows={rows} column={selectedColumn} />
                                        )}
                                    </div>
                                        </>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};