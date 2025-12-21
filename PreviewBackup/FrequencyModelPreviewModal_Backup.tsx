import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { CanvasModule, FrequencyModelOutput, FrequencyModelFitResult } from '../types';
import { XCircleIcon, ArrowDownTrayIcon } from './icons';
import { useCopyOnCtrlC } from '../hooks/useCopyOnCtrlC';
import { SpreadViewModal } from './SpreadViewModal';

// Magnifier hook for graphs
const useGraphMagnifier = () => {
  const [isMagnifying, setIsMagnifying] = useState(false);
  const [magnifierPos, setMagnifierPos] = useState<{ x: number; y: number } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const handleContextMenu = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMagnify = useCallback(() => {
    setIsMagnifying(true);
    setContextMenu(null);
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isMagnifying || !svgRef.current) return;
    
    const svg = svgRef.current;
    const rect = svg.getBoundingClientRect();
    
    // Get viewBox dimensions
    const viewBox = svg.getAttribute('viewBox');
    if (!viewBox) {
      // Fallback if no viewBox
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setMagnifierPos({ x, y });
      return;
    }
    
    const [vbX, vbY, vbWidth, vbHeight] = viewBox.split(' ').map(Number);
    
    // Calculate scale factors
    const scaleX = vbWidth / rect.width;
    const scaleY = vbHeight / rect.height;
    
    // Convert mouse position to viewBox coordinates
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const x = mouseX * scaleX;
    const y = mouseY * scaleY;
    
    setMagnifierPos({ x, y });
  }, [isMagnifying]);

  const handleMouseLeave = useCallback(() => {
    if (isMagnifying) {
      setMagnifierPos(null);
    }
  }, [isMagnifying]);

  useEffect(() => {
    if (!isMagnifying) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsMagnifying(false);
        setMagnifierPos(null);
      }
    };

    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenu && !(e.target as Element).closest('.context-menu')) {
        setContextMenu(null);
      }
    };

    document.addEventListener('keydown', handleEscape);
    document.addEventListener('click', handleClickOutside, true);

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [isMagnifying, contextMenu]);

  return {
    svgRef,
    isMagnifying,
    magnifierPos,
    contextMenu,
    handleContextMenu,
    handleMagnify,
    handleMouseMove,
    handleMouseLeave,
  };
};

interface FrequencyModelPreviewModalProps {
  module: CanvasModule;
  onClose: () => void;
  onSelectDistribution?: (moduleId: string, distributionType: string) => void;
}

export const FrequencyModelPreviewModal: React.FC<FrequencyModelPreviewModalProps> = ({ 
  module, 
  onClose,
  onSelectDistribution 
}) => {
  const output = module.outputData as FrequencyModelOutput;
  if (!output || output.type !== 'FrequencyModelOutput') {
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

  let results: FrequencyModelFitResult[] = output.results || [];
  const selectedDistribution = output.selectedDistribution;

  const [localSelected, setLocalSelected] = useState<string>(() => {
    if (selectedDistribution) return selectedDistribution;
    if (results.length > 0 && results[0]?.distributionType) {
      return results[0].distributionType;
    }
    return '';
  });

  const handleSelectDistribution = (distType: string) => {
    setLocalSelected(distType);
    if (onSelectDistribution) {
      onSelectDistribution(module.id, distType);
    }
  };

  const viewDetailsRef = useRef<HTMLDivElement>(null);
  useCopyOnCtrlC(viewDetailsRef);

  // AIC 기준으로 정렬 (낮을수록 좋음)
  // 성공한 결과와 실패한 결과를 분리
  const successfulResults = results.filter(r => !r.error);
  const failedResults = results.filter(r => r.error);
  
  // 성공한 결과는 AIC 기준으로 정렬
  const sortedSuccessfulResults = [...successfulResults].sort((a, b) => {
    const aicA = a.fitStatistics?.aic ?? Infinity;
    const aicB = b.fitStatistics?.aic ?? Infinity;
    return aicA - aicB;
  });
  
  // 모든 결과를 합치기 (성공한 것 먼저, 실패한 것 나중에)
  const sortedResults = [...sortedSuccessfulResults, ...failedResults];

  const recommended = sortedSuccessfulResults.length > 0 ? sortedSuccessfulResults[0] : null;
  const [showSpreadView, setShowSpreadView] = useState(false);
  const [activeTab, setActiveTab] = useState<'details' | 'graphs'>('details');

  // 통계 계산 (originalData 또는 yearlyCounts의 count 값들 사용)
  const statistics = useMemo(() => {
    let counts: number[] = [];
    
    if (output.originalData && output.originalData.length > 0) {
      counts = output.originalData.filter(c => typeof c === 'number' && !isNaN(c) && c >= 0);
    } else if (output.yearlyCounts && output.yearlyCounts.length > 0) {
      counts = output.yearlyCounts.map(item => item.count).filter(c => typeof c === 'number' && !isNaN(c));
    }
    
    if (counts.length === 0) return null;

    counts.sort((a, b) => a - b);
    const n = counts.length;
    const sum = counts.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const variance = counts.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);
    
    // 왜도 (skewness)
    const skewness = stdDev > 0 
      ? counts.reduce((s, val) => s + Math.pow(val - mean, 3), 0) / (n * Math.pow(stdDev, 3))
      : 0;
    
    // 첨도 (kurtosis) - excess kurtosis
    const kurtosis = stdDev > 0
      ? counts.reduce((s, val) => s + Math.pow(val - mean, 4), 0) / (n * Math.pow(stdDev, 4)) - 3
      : 0;

    // 백분위수 계산
    const getQuantile = (q: number) => {
      const pos = (n - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      return counts[base + 1] !== undefined 
        ? counts[base] + rest * (counts[base + 1] - counts[base])
        : counts[base];
    };

    return {
      mean,
      variance,
      stdDev,
      skewness,
      kurtosis,
      min: counts[0],
      max: counts[n - 1],
      percentile25: getQuantile(0.25),
      percentile50: getQuantile(0.5), // median
      percentile75: getQuantile(0.75),
      percentile90: getQuantile(0.90),
      percentile95: getQuantile(0.95),
      percentile99: getQuantile(0.99),
      count: n,
    };
  }, [output.originalData, output.yearlyCounts]);

  // Spread View용 데이터 변환
  const spreadViewData = useMemo(() => {
    return sortedResults.map(result => ({
      Distribution: result.distributionType,
      AIC: result.fitStatistics?.aic?.toFixed(4) || 'N/A',
      BIC: result.fitStatistics?.bic?.toFixed(4) || 'N/A',
      'Log Likelihood': result.fitStatistics?.logLikelihood?.toFixed(4) || 'N/A',
      Mean: result.fitStatistics?.mean?.toFixed(4) || 'N/A',
      Variance: result.fitStatistics?.variance?.toFixed(4) || 'N/A',
      Dispersion: result.fitStatistics?.dispersion?.toFixed(4) || 'N/A',
    }));
  }, [sortedResults]);

  const spreadViewColumns = [
    { name: 'Distribution', type: 'string' },
    { name: 'AIC', type: 'number' },
    { name: 'BIC', type: 'number' },
    { name: 'Log Likelihood', type: 'number' },
    { name: 'Mean', type: 'number' },
    { name: 'Variance', type: 'number' },
    { name: 'Dispersion', type: 'number' },
  ];

  // Graphs Tab Component
  interface GraphsTabProps {
    output: FrequencyModelOutput;
    sortedResults: FrequencyModelFitResult[];
    recommended: FrequencyModelFitResult | null;
    statistics: {
      mean: number;
      stdDev: number;
      min: number;
      max: number;
      count: number;
    } | null;
  }

  const GraphsTab: React.FC<GraphsTabProps> = ({ output, sortedResults, recommended, statistics }) => {
    const originalData = output.originalData || [];
    const counts = originalData.filter((c): c is number => typeof c === 'number' && !isNaN(c) && c >= 0).sort((a, b) => a - b);

    // Initialize selectedQQDistribution
    const qqResults = sortedResults.filter(r => !r.error && r.qqPlot);
    const initialQQDistribution = qqResults.length > 0 
      ? (recommended?.distributionType || qqResults[0].distributionType || '')
      : '';
    
    const [selectedQQDistribution, setSelectedQQDistribution] = useState<string>(initialQQDistribution);

    if (counts.length === 0 || !statistics) {
      return <div className="text-gray-500 text-center py-8">No data available for graphs.</div>;
    }

    const margin = { top: 40, right: 40, bottom: 80, left: 100 };
    const chartWidth = 1200;
    const chartHeight = 600;
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = chartHeight - margin.top - margin.bottom;

    // Color palette for distributions
    const colors = [
      '#3b82f6', // blue
      '#ef4444', // red
      '#10b981', // green
      '#f59e0b', // orange
      '#8b5cf6', // purple
      '#ec4899', // pink
      '#06b6d4', // cyan
      '#84cc16', // lime
    ];

    // Top 5 distributions by AIC
    const top5Results = sortedResults.filter(r => !r.error && r.fitStatistics?.aic !== undefined).slice(0, 5);

    // Helper function to calculate PMF for different distributions
    const calculatePMF = (k: number, result: FrequencyModelFitResult): number => {
      const params = result.parameters;
      const distType = result.distributionType;
      
      if (distType === 'Poisson') {
        const lambda = params.lambda || 0;
        if (lambda <= 0) return 0;
        return Math.exp(-lambda) * Math.pow(lambda, k) / (k < 20 ? Array.from({ length: k }, (_, i) => i + 1).reduce((a, b) => a * b, 1) : Infinity);
      } else if (distType === 'NegativeBinomial') {
        const n = params.n || 0;
        const p = params.p || 0;
        if (n <= 0 || p <= 0 || p >= 1) return 0;
        // Using approximation for large factorials
        if (k === 0) return Math.pow(p, n);
        const logPMF = Math.log(n) + (k - 1) * Math.log(1 - p) + n * Math.log(p) - Math.log(k);
        return Math.exp(logPMF);
      } else if (distType === 'ZeroInflatedPoisson') {
        const pi = params.pi || 0;
        const lambda = params.lambda || 0;
        if (lambda <= 0) return 0;
        if (k === 0) {
          return pi + (1 - pi) * Math.exp(-lambda);
        } else {
          const poissonPMF = Math.exp(-lambda) * Math.pow(lambda, k) / (k < 20 ? Array.from({ length: k }, (_, i) => i + 1).reduce((a, b) => a * b, 1) : Infinity);
          return (1 - pi) * poissonPMF;
        }
      } else if (distType === 'ZeroInflatedNegativeBinomial') {
        const pi = params.pi || 0;
        const n = params.n || 0;
        const p = params.p || 0;
        if (n <= 0 || p <= 0 || p >= 1) return 0;
        if (k === 0) {
          const nbZero = Math.pow(p, n);
          return pi + (1 - pi) * nbZero;
        } else {
          const logNB = Math.log(n) + (k - 1) * Math.log(1 - p) + n * Math.log(p) - Math.log(k);
          const nbPMF = Math.exp(logNB);
          return (1 - pi) * nbPMF;
        }
      }
      return 0;
    };

    // Helper function to calculate CDF
    const calculateCDF = (k: number, result: FrequencyModelFitResult): number => {
      let sum = 0;
      for (let i = 0; i <= k; i++) {
        sum += calculatePMF(i, result);
      }
      return sum;
    };

    // 1. Histogram + PMF
    const HistogramPMFChart = () => {
      const magnifier = useGraphMagnifier();
      const maxCount = Math.max(...counts);
      const minCount = Math.min(...counts);
      const bins = Math.min(40, maxCount - minCount + 1);
      const binWidth = (maxCount - minCount) / bins || 1;
      const histogram: number[] = Array(bins).fill(0);
      
      counts.forEach(count => {
        const binIndex = Math.min(Math.floor((count - minCount) / binWidth), bins - 1);
        histogram[binIndex]++;
      });

      const maxFreq = Math.max(...histogram);
      const normalizedHistogram = histogram.map(freq => freq / (maxFreq * binWidth * counts.length));

      const xScale = (x: number) => margin.left + ((x - minCount) / (maxCount - minCount || 1)) * innerWidth;
      const yScale = (y: number) => margin.top + innerHeight - (y / (Math.max(...normalizedHistogram) * 1.2 || 1)) * innerHeight;

      const magnifierSize = 200;
      const magnifierScale = 3;

      return (
        <div className="border border-gray-300 rounded-lg p-4 bg-white relative">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-md font-semibold text-gray-700">Histogram + Fitted PMFs</h4>
            <button
              onClick={() => magnifier.handleMagnify()}
              className={`p-2 rounded-md transition-colors ${
                magnifier.isMagnifying 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
          </div>
          <svg 
            ref={magnifier.svgRef}
            viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
            className="w-full h-auto"
            onContextMenu={magnifier.handleContextMenu}
            onMouseMove={magnifier.handleMouseMove}
            onMouseLeave={magnifier.handleMouseLeave}
          >
            <defs>
              <pattern id="hist-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={chartWidth} height={chartHeight} fill="url(#hist-grid)" />

            {/* Axes */}
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
            <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />

            {/* Histogram bars */}
            {histogram.map((freq, i) => {
              const binStart = minCount + i * binWidth;
              const barHeight = normalizedHistogram[i] * innerHeight;
              return (
                <rect
                  key={i}
                  x={xScale(binStart)}
                  y={yScale(normalizedHistogram[i])}
                  width={xScale(binStart + binWidth) - xScale(binStart)}
                  height={barHeight}
                  fill="#3b82f6"
                  opacity="0.5"
                />
              );
            })}

            {/* Fitted PMF curves (top 5) */}
            {top5Results.map((result, idx) => {
              const pmfPoints: { x: number; y: number }[] = [];
              for (let k = minCount; k <= maxCount; k++) {
                const pmf = calculatePMF(k, result);
                if (pmf > 0) {
                  pmfPoints.push({ x: k, y: pmf });
                }
              }
              if (pmfPoints.length === 0) return null;
              
              return (
                <g key={result.distributionType}>
                  <polyline
                    points={pmfPoints.map(p => `${xScale(p.x)},${yScale(p.y)}`).join(' ')}
                    fill="none"
                    stroke={colors[idx % colors.length]}
                    strokeWidth="3"
                  />
                </g>
              );
            })}

            {/* X-axis ticks */}
            {Array.from({ length: 5 }, (_, i) => minCount + (i / 4) * (maxCount - minCount)).map((tick, i) => (
              <g key={`x-${i}`}>
                <line x1={xScale(tick)} y1={margin.top + innerHeight} x2={xScale(tick)} y2={margin.top + innerHeight + 10} stroke="#6b7280" strokeWidth="2" />
                <text x={xScale(tick)} y={margin.top + innerHeight + 35} textAnchor="middle" fontSize="20" fill="#6b7280">
                  {Math.round(tick)}
                </text>
              </g>
            ))}

            {/* Y-axis ticks */}
            {Array.from({ length: 5 }, (_, i) => i / 4).map((tick, i) => {
              const maxY = Math.max(...normalizedHistogram) * 1.2;
              return (
                <g key={`y-${i}`}>
                  <line x1={margin.left} y1={margin.top + innerHeight - tick * innerHeight} x2={margin.left - 10} y2={margin.top + innerHeight - tick * innerHeight} stroke="#6b7280" strokeWidth="2" />
                  <text x={margin.left - 15} y={margin.top + innerHeight - tick * innerHeight + 6} textAnchor="end" fontSize="20" fill="#6b7280">
                    {(tick * maxY).toFixed(3)}
                  </text>
                </g>
              );
            })}

            {/* Axis labels */}
            <text x={chartWidth / 2} y={chartHeight - 20} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold">
              Count
            </text>
            <text x="30" y={chartHeight / 2} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold" transform={`rotate(-90 30 ${chartHeight / 2})`}>
              Probability Density
            </text>

            {/* Legend */}
            <g transform={`translate(${margin.left + innerWidth - 200}, ${margin.top + 20})`}>
              <rect width="180" height={20 + top5Results.length * 25} fill="white" opacity="0.9" stroke="#ccc" strokeWidth="1" />
              <text x="10" y="25" fontSize="18" fill="#374151" fontWeight="bold">Distributions</text>
              {top5Results.map((result, idx) => (
                <g key={result.distributionType} transform={`translate(10, ${35 + idx * 25})`}>
                  <line x1="0" y1="0" x2="20" y2="0" stroke={colors[idx % colors.length]} strokeWidth="3" />
                  <text x="25" y="5" fontSize="18" fill="#374151">
                    {result.distributionType} (AIC={result.fitStatistics?.aic?.toFixed(1) || 'N/A'})
                  </text>
                </g>
              ))}
            </g>

            {/* Magnifier overlay */}
            {magnifier.isMagnifying && magnifier.magnifierPos && (
              <>
                <defs>
                  <clipPath id="magnifier-clip-hist">
                    <circle cx={magnifier.magnifierPos.x} cy={magnifier.magnifierPos.y} r={magnifierSize / 2} />
                  </clipPath>
                </defs>
                <g clipPath="url(#magnifier-clip-hist)">
                  <rect
                    x={magnifier.magnifierPos.x - magnifierSize / 2}
                    y={magnifier.magnifierPos.y - magnifierSize / 2}
                    width={magnifierSize}
                    height={magnifierSize}
                    fill="white"
                    opacity="0.95"
                  />
                  <g transform={`translate(${magnifier.magnifierPos.x}, ${magnifier.magnifierPos.y}) scale(${magnifierScale}) translate(${-magnifier.magnifierPos.x}, ${-magnifier.magnifierPos.y})`}>
                    <rect width={chartWidth} height={chartHeight} fill="url(#hist-grid)" />
                    <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    {histogram.map((freq, i) => {
                      const binStart = minCount + i * binWidth;
                      return (
                        <rect
                          key={i}
                          x={xScale(binStart)}
                          y={yScale(normalizedHistogram[i])}
                          width={xScale(binStart + binWidth) - xScale(binStart)}
                          height={normalizedHistogram[i] * innerHeight}
                          fill="#3b82f6"
                          opacity="0.5"
                        />
                      );
                    })}
                    {top5Results.map((result, idx) => {
                      const pmfPoints: { x: number; y: number }[] = [];
                      for (let k = minCount; k <= maxCount; k++) {
                        const pmf = calculatePMF(k, result);
                        if (pmf > 0) {
                          pmfPoints.push({ x: k, y: pmf });
                        }
                      }
                      if (pmfPoints.length === 0) return null;
                      return (
                        <polyline
                          key={result.distributionType}
                          points={pmfPoints.map(p => `${xScale(p.x)},${yScale(p.y)}`).join(' ')}
                          fill="none"
                          stroke={colors[idx % colors.length]}
                          strokeWidth="3"
                        />
                      );
                    })}
                  </g>
                </g>
                <circle
                  cx={magnifier.magnifierPos.x}
                  cy={magnifier.magnifierPos.y}
                  r={magnifierSize / 2}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                />
              </>
            )}
          </svg>
          {magnifier.contextMenu && (
            <div
              className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
              style={{ left: magnifier.contextMenu.x, top: magnifier.contextMenu.y }}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  magnifier.handleMagnify();
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                돋보기
              </button>
            </div>
          )}
        </div>
      );
    };

    // 2. Empirical CDF vs Fitted CDF
    const CDFChart = () => {
      const magnifier = useGraphMagnifier();
      const ecdf = counts.map((_, i) => (i + 1) / counts.length);
      
      const xScale = (x: number) => margin.left + ((x - statistics.min) / (statistics.max - statistics.min || 1)) * innerWidth;
      const yScale = (y: number) => margin.top + innerHeight - y * innerHeight;

      const magnifierSize = 200;
      const magnifierScale = 3;

      return (
        <div className="border border-gray-300 rounded-lg p-4 bg-white relative">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-md font-semibold text-gray-700">Empirical CDF vs Fitted CDF</h4>
            <button
              onClick={() => magnifier.handleMagnify()}
              className={`p-2 rounded-md transition-colors ${
                magnifier.isMagnifying 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
          </div>
          <svg 
            ref={magnifier.svgRef}
            viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
            className="w-full h-auto"
            onContextMenu={magnifier.handleContextMenu}
            onMouseMove={magnifier.handleMouseMove}
            onMouseLeave={magnifier.handleMouseLeave}
          >
            <defs>
              <pattern id="cdf-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={chartWidth} height={chartHeight} fill="url(#cdf-grid)" />

            {/* Axes */}
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
            <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />

            {/* Empirical CDF */}
            <polyline
              points={counts.map((x, i) => `${xScale(x)},${yScale(ecdf[i])}`).join(' ')}
              fill="none"
              stroke="#000000"
              strokeWidth="3"
            />

            {/* Fitted CDF curves (top 5) */}
            {top5Results.map((result, idx) => {
              const cdfPoints: { x: number; y: number }[] = [];
              for (let k = statistics.min; k <= statistics.max; k++) {
                const cdf = calculateCDF(k, result);
                cdfPoints.push({ x: k, y: cdf });
              }
              
              return (
                <polyline
                  key={result.distributionType}
                  points={cdfPoints.map(p => `${xScale(p.x)},${yScale(p.y)}`).join(' ')}
                  fill="none"
                  stroke={colors[idx % colors.length]}
                  strokeWidth="3"
                />
              );
            })}

            {/* X-axis ticks */}
            {Array.from({ length: 5 }, (_, i) => statistics.min + (i / 4) * (statistics.max - statistics.min)).map((tick, i) => (
              <g key={`x-${i}`}>
                <line x1={xScale(tick)} y1={margin.top + innerHeight} x2={xScale(tick)} y2={margin.top + innerHeight + 10} stroke="#6b7280" strokeWidth="2" />
                <text x={xScale(tick)} y={margin.top + innerHeight + 35} textAnchor="middle" fontSize="20" fill="#6b7280">
                  {Math.round(tick)}
                </text>
              </g>
            ))}

            {/* Y-axis ticks */}
            {Array.from({ length: 5 }, (_, i) => i / 4).map((tick, i) => (
              <g key={`y-${i}`}>
                <line x1={margin.left} y1={margin.top + innerHeight - tick * innerHeight} x2={margin.left - 10} y2={margin.top + innerHeight - tick * innerHeight} stroke="#6b7280" strokeWidth="2" />
                <text x={margin.left - 15} y={margin.top + innerHeight - tick * innerHeight + 6} textAnchor="end" fontSize="20" fill="#6b7280">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Axis labels */}
            <text x={chartWidth / 2} y={chartHeight - 20} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold">
              Count
            </text>
            <text x="30" y={chartHeight / 2} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold" transform={`rotate(-90 30 ${chartHeight / 2})`}>
              Cumulative Probability
            </text>

            {/* Legend */}
            <g transform={`translate(${margin.left + innerWidth - 200}, ${margin.top + 20})`}>
              <rect width="180" height={30 + top5Results.length * 25} fill="white" opacity="0.9" stroke="#ccc" strokeWidth="1" />
              <text x="10" y="25" fontSize="18" fill="#374151" fontWeight="bold">Distributions</text>
              <g transform="translate(10, 35)">
                <line x1="0" y1="0" x2="20" y2="0" stroke="#000000" strokeWidth="3" />
                <text x="25" y="5" fontSize="18" fill="#374151">Empirical</text>
              </g>
              {top5Results.map((result, idx) => (
                <g key={result.distributionType} transform={`translate(10, ${60 + idx * 25})`}>
                  <line x1="0" y1="0" x2="20" y2="0" stroke={colors[idx % colors.length]} strokeWidth="3" />
                  <text x="25" y="5" fontSize="18" fill="#374151">
                    {result.distributionType}
                  </text>
                </g>
              ))}
            </g>

            {/* Magnifier overlay - similar to HistogramPMFChart */}
            {magnifier.isMagnifying && magnifier.magnifierPos && (
              <>
                <defs>
                  <clipPath id="magnifier-clip-cdf">
                    <circle cx={magnifier.magnifierPos.x} cy={magnifier.magnifierPos.y} r={magnifierSize / 2} />
                  </clipPath>
                </defs>
                <g clipPath="url(#magnifier-clip-cdf)">
                  <rect
                    x={magnifier.magnifierPos.x - magnifierSize / 2}
                    y={magnifier.magnifierPos.y - magnifierSize / 2}
                    width={magnifierSize}
                    height={magnifierSize}
                    fill="white"
                    opacity="0.95"
                  />
                  <g transform={`translate(${magnifier.magnifierPos.x}, ${magnifier.magnifierPos.y}) scale(${magnifierScale}) translate(${-magnifier.magnifierPos.x}, ${-magnifier.magnifierPos.y})`}>
                    <rect width={chartWidth} height={chartHeight} fill="url(#cdf-grid)" />
                    <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <polyline
                      points={counts.map((x, i) => `${xScale(x)},${yScale(ecdf[i])}`).join(' ')}
                      fill="none"
                      stroke="#000000"
                      strokeWidth="3"
                    />
                    {top5Results.map((result, idx) => {
                      const cdfPoints: { x: number; y: number }[] = [];
                      for (let k = statistics.min; k <= statistics.max; k++) {
                        const cdf = calculateCDF(k, result);
                        cdfPoints.push({ x: k, y: cdf });
                      }
                      return (
                        <polyline
                          key={result.distributionType}
                          points={cdfPoints.map(p => `${xScale(p.x)},${yScale(p.y)}`).join(' ')}
                          fill="none"
                          stroke={colors[idx % colors.length]}
                          strokeWidth="3"
                        />
                      );
                    })}
                  </g>
                </g>
                <circle
                  cx={magnifier.magnifierPos.x}
                  cy={magnifier.magnifierPos.y}
                  r={magnifierSize / 2}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                />
              </>
            )}
          </svg>
          {magnifier.contextMenu && (
            <div
              className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
              style={{ left: magnifier.contextMenu.x, top: magnifier.contextMenu.y }}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  magnifier.handleMagnify();
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                돋보기
              </button>
            </div>
          )}
        </div>
      );
    };

    // 3. Q-Q Plot
    const QQPlotChart = () => {
      const magnifier = useGraphMagnifier();
      const selectedResult = sortedResults.find(r => r.distributionType === selectedQQDistribution) || recommended;
      const magnifierSize = 200;
      const magnifierScale = 3;
      
      if (!selectedResult || !selectedResult.qqPlot) {
        return (
          <div className="border border-gray-300 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-700">Q-Q Plot</h4>
              <div className="flex items-center gap-2">
                <select
                  value={selectedQQDistribution}
                  onChange={(e) => setSelectedQQDistribution(e.target.value)}
                  className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {sortedResults.filter(r => !r.error && r.qqPlot).map((result) => (
                    <option key={result.distributionType} value={result.distributionType}>
                      {result.distributionType}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => magnifier.handleMagnify()}
                  className={`p-2 rounded-md transition-colors ${
                    magnifier.isMagnifying 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                  title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="text-gray-500 text-sm">Q-Q Plot 데이터가 없습니다.</div>
          </div>
        );
      }

      const { theoreticalQuantiles, sampleQuantiles } = selectedResult.qqPlot;
      if (!theoreticalQuantiles || !sampleQuantiles || theoreticalQuantiles.length === 0) {
        return (
          <div className="border border-gray-300 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-700">Q-Q Plot</h4>
              <div className="flex items-center gap-2">
                <select
                  value={selectedQQDistribution}
                  onChange={(e) => setSelectedQQDistribution(e.target.value)}
                  className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {sortedResults.filter(r => !r.error && r.qqPlot).map((result) => (
                    <option key={result.distributionType} value={result.distributionType}>
                      {result.distributionType}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => magnifier.handleMagnify()}
                  className={`p-2 rounded-md transition-colors ${
                    magnifier.isMagnifying 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                  title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="text-gray-500 text-sm">Q-Q Plot 데이터가 없습니다.</div>
          </div>
        );
      }

      const validPairs = theoreticalQuantiles
        .map((x, i) => ({ x, y: sampleQuantiles[i] }))
        .filter(p => isFinite(p.x) && isFinite(p.y) && p.x !== null && p.y !== null);
      
      if (validPairs.length === 0) {
        return (
          <div className="border border-gray-300 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-700">Q-Q Plot</h4>
              <div className="flex items-center gap-2">
                <select
                  value={selectedQQDistribution}
                  onChange={(e) => setSelectedQQDistribution(e.target.value)}
                  className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {sortedResults.filter(r => !r.error && r.qqPlot).map((result) => (
                    <option key={result.distributionType} value={result.distributionType}>
                      {result.distributionType}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => magnifier.handleMagnify()}
                  className={`p-2 rounded-md transition-colors ${
                    magnifier.isMagnifying 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                  title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="text-gray-500 text-sm">유효한 Q-Q Plot 데이터가 없습니다.</div>
          </div>
        );
      }

      const xMin = Math.min(...validPairs.map(p => p.x));
      const xMax = Math.max(...validPairs.map(p => p.x));
      const yMin = Math.min(...validPairs.map(p => p.y));
      const yMax = Math.max(...validPairs.map(p => p.y));
      const xRange = xMax - xMin || 1;
      const yRange = yMax - yMin || 1;
      const xMinAdjusted = xMin - xRange * 0.05;
      const xMaxAdjusted = xMax + xRange * 0.05;
      const yMinAdjusted = yMin - yRange * 0.05;
      const yMaxAdjusted = yMax + yRange * 0.05;

      const xScale = (x: number) => margin.left + ((x - xMinAdjusted) / (xMaxAdjusted - xMinAdjusted || 1)) * innerWidth;
      const yScale = (y: number) => margin.top + innerHeight - ((y - yMinAdjusted) / (yMaxAdjusted - yMinAdjusted || 1)) * innerHeight;

      const lineMin = Math.min(xMinAdjusted, yMinAdjusted);
      const lineMax = Math.max(xMaxAdjusted, yMaxAdjusted);

      const xTicks = Array.from({ length: 5 }, (_, i) => xMinAdjusted + (i / 4) * (xMaxAdjusted - xMinAdjusted));
      const yTicks = Array.from({ length: 5 }, (_, i) => yMinAdjusted + (i / 4) * (yMaxAdjusted - yMinAdjusted));

      return (
        <div className="border border-gray-300 rounded-lg p-4 bg-white relative">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-semibold text-gray-700">Q-Q Plot ({selectedResult.distributionType})</h4>
            <div className="flex items-center gap-2">
              <select
                value={selectedQQDistribution}
                onChange={(e) => setSelectedQQDistribution(e.target.value)}
                className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {sortedResults.filter(r => !r.error && r.qqPlot).map((result) => (
                  <option key={result.distributionType} value={result.distributionType}>
                    {result.distributionType}
                  </option>
                ))}
              </select>
              <button
                onClick={() => magnifier.handleMagnify()}
                className={`p-2 rounded-md transition-colors ${
                  magnifier.isMagnifying 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
            </div>
          </div>
          <svg 
            ref={magnifier.svgRef}
            viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
            className="w-full h-auto"
            onContextMenu={magnifier.handleContextMenu}
            onMouseMove={magnifier.handleMouseMove}
            onMouseLeave={magnifier.handleMouseLeave}
          >
            <defs>
              <pattern id="qq-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={chartWidth} height={chartHeight} fill="url(#qq-grid)" />

            {/* Axes */}
            <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />

            {/* Reference line (y=x) */}
            <line
              x1={xScale(lineMin)}
              y1={yScale(lineMin)}
              x2={xScale(lineMax)}
              y2={yScale(lineMax)}
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.7"
            />

            {/* X-axis ticks */}
            {xTicks.map((tick, i) => (
              <g key={`x-${i}`}>
                <line x1={xScale(tick)} y1={margin.top + innerHeight} x2={xScale(tick)} y2={margin.top + innerHeight + 10} stroke="#6b7280" strokeWidth="2" />
                <text x={xScale(tick)} y={margin.top + innerHeight + 35} textAnchor="middle" fontSize="20" fill="#6b7280">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Y-axis ticks */}
            {yTicks.map((tick, i) => (
              <g key={`y-${i}`}>
                <line x1={margin.left} y1={yScale(tick)} x2={margin.left - 10} y2={yScale(tick)} stroke="#6b7280" strokeWidth="2" />
                <text x={margin.left - 15} y={yScale(tick) + 6} textAnchor="end" fontSize="20" fill="#6b7280">
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Data points */}
            {validPairs.map((pair, i) => (
              <circle key={i} cx={xScale(pair.x)} cy={yScale(pair.y)} r="5" fill="#3b82f6" opacity="0.7" />
            ))}

            {/* Axis labels */}
            <text x={chartWidth / 2} y={chartHeight - 20} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold">
              Theoretical Quantiles
            </text>
            <text x="30" y={chartHeight / 2} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold" transform={`rotate(-90 30 ${chartHeight / 2})`}>
              Sample Quantiles
            </text>

            {/* Magnifier overlay */}
            {magnifier.isMagnifying && magnifier.magnifierPos && selectedResult && selectedResult.qqPlot && (
              <>
                <defs>
                  <clipPath id="magnifier-clip-qq">
                    <circle cx={magnifier.magnifierPos.x} cy={magnifier.magnifierPos.y} r={magnifierSize / 2} />
                  </clipPath>
                </defs>
                <g clipPath="url(#magnifier-clip-qq)">
                  <rect
                    x={magnifier.magnifierPos.x - magnifierSize / 2}
                    y={magnifier.magnifierPos.y - magnifierSize / 2}
                    width={magnifierSize}
                    height={magnifierSize}
                    fill="white"
                    opacity="0.95"
                  />
                  <g transform={`translate(${magnifier.magnifierPos.x}, ${magnifier.magnifierPos.y}) scale(${magnifierScale}) translate(${-magnifier.magnifierPos.x}, ${-magnifier.magnifierPos.y})`}>
                    <rect width={chartWidth} height={chartHeight} fill="url(#qq-grid)" />
                    <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <line x1={xScale(lineMin)} y1={yScale(lineMin)} x2={xScale(lineMax)} y2={yScale(lineMax)} stroke="#ef4444" strokeWidth="2" strokeDasharray="5,5" opacity="0.7" />
                    {validPairs.map((pair, i) => (
                      <circle key={i} cx={xScale(pair.x)} cy={yScale(pair.y)} r="5" fill="#3b82f6" opacity="0.7" />
                    ))}
                  </g>
                </g>
                <circle
                  cx={magnifier.magnifierPos.x}
                  cy={magnifier.magnifierPos.y}
                  r={magnifierSize / 2}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                />
              </>
            )}
          </svg>
          {magnifier.contextMenu && (
            <div
              className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
              style={{ left: magnifier.contextMenu.x, top: magnifier.contextMenu.y }}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  magnifier.handleMagnify();
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                돋보기
              </button>
            </div>
          )}
        </div>
      );
    };

    // 4. AIC Comparison
    const AICChart = () => {
      const magnifier = useGraphMagnifier();
      const validResults = sortedResults.filter(r => !r.error && r.fitStatistics?.aic !== undefined && r.fitStatistics?.aic !== null);
      
      if (validResults.length === 0) {
        return (
          <div className="border border-gray-300 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-md font-semibold text-gray-700">AIC Comparison</h4>
              <button
                onClick={() => magnifier.handleMagnify()}
                className={`p-2 rounded-md transition-colors ${
                  magnifier.isMagnifying 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
            </div>
            <div className="text-gray-500 text-sm">AIC 데이터가 없습니다.</div>
          </div>
        );
      }

      const maxAIC = Math.max(...validResults.map(r => r.fitStatistics!.aic!));
      const minAIC = Math.min(...validResults.map(r => r.fitStatistics!.aic!));
      const aicRange = maxAIC - minAIC || 1;

      const barWidth = innerWidth / validResults.length * 0.8;
      const barSpacing = innerWidth / validResults.length * 0.2;

      const magnifierSize = 200;
      const magnifierScale = 3;

      return (
        <div className="border border-gray-300 rounded-lg p-4 bg-white relative">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-md font-semibold text-gray-700">AIC Comparison</h4>
            <button
              onClick={() => magnifier.handleMagnify()}
              className={`p-2 rounded-md transition-colors ${
                magnifier.isMagnifying 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              title={magnifier.isMagnifying ? '돋보기 끄기 (ESC)' : '돋보기 켜기'}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
          </div>
          <svg 
            ref={magnifier.svgRef}
            viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
            className="w-full h-auto"
            onContextMenu={magnifier.handleContextMenu}
            onMouseMove={magnifier.handleMouseMove}
            onMouseLeave={magnifier.handleMouseLeave}
          >
            <defs>
              <pattern id="aic-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={chartWidth} height={chartHeight} fill="url(#aic-grid)" />

            {/* Axes */}
            <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
            <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />

            {/* Bars */}
            {validResults.map((result, idx) => {
              const aic = result.fitStatistics!.aic!;
              const barHeight = ((aic - minAIC) / aicRange) * innerHeight * 0.9;
              const x = margin.left + idx * (barWidth + barSpacing) + barSpacing / 2;
              const y = margin.top + innerHeight - barHeight;
              const isRecommended = result.distributionType === recommended?.distributionType;

              return (
                <g key={result.distributionType}>
                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    fill={isRecommended ? '#10b981' : colors[idx % colors.length]}
                    opacity="0.8"
                  />
                  <text
                    x={x + barWidth / 2}
                    y={y - 10}
                    textAnchor="middle"
                    fontSize="20"
                    fill="#374151"
                    fontWeight="bold"
                  >
                    {aic.toFixed(1)}
                  </text>
                  <text
                    x={x + barWidth / 2}
                    y={margin.top + innerHeight + 40}
                    textAnchor="middle"
                    fontSize="20"
                    fill="#374151"
                    transform={`rotate(-45 ${x + barWidth / 2} ${margin.top + innerHeight + 40})`}
                  >
                    {result.distributionType}
                  </text>
                </g>
              );
            })}

            {/* Y-axis label */}
            <text x="30" y={chartHeight / 2} textAnchor="middle" fontSize="24" fill="#374151" fontWeight="bold" transform={`rotate(-90 30 ${chartHeight / 2})`}>
              AIC
            </text>

            {/* Magnifier overlay */}
            {magnifier.isMagnifying && magnifier.magnifierPos && (
              <>
                <defs>
                  <clipPath id="magnifier-clip-aic">
                    <circle cx={magnifier.magnifierPos.x} cy={magnifier.magnifierPos.y} r={magnifierSize / 2} />
                  </clipPath>
                </defs>
                <g clipPath="url(#magnifier-clip-aic)">
                  <rect
                    x={magnifier.magnifierPos.x - magnifierSize / 2}
                    y={magnifier.magnifierPos.y - magnifierSize / 2}
                    width={magnifierSize}
                    height={magnifierSize}
                    fill="white"
                    opacity="0.95"
                  />
                  <g transform={`translate(${magnifier.magnifierPos.x}, ${magnifier.magnifierPos.y}) scale(${magnifierScale}) translate(${-magnifier.magnifierPos.x}, ${-magnifier.magnifierPos.y})`}>
                    <rect width={chartWidth} height={chartHeight} fill="url(#aic-grid)" />
                    <line x1={margin.left} y1={margin.top} x2={margin.left} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    <line x1={margin.left} y1={margin.top + innerHeight} x2={margin.left + innerWidth} y2={margin.top + innerHeight} stroke="#374151" strokeWidth="2" />
                    {validResults.map((result, idx) => {
                      const aic = result.fitStatistics!.aic!;
                      const barHeight = ((aic - minAIC) / aicRange) * innerHeight * 0.9;
                      const x = margin.left + idx * (barWidth + barSpacing) + barSpacing / 2;
                      const y = margin.top + innerHeight - barHeight;
                      const isRecommended = result.distributionType === recommended?.distributionType;
                      return (
                        <g key={result.distributionType}>
                          <rect
                            x={x}
                            y={y}
                            width={barWidth}
                            height={barHeight}
                            fill={isRecommended ? '#10b981' : colors[idx % colors.length]}
                            opacity="0.8"
                          />
                          <text
                            x={x + barWidth / 2}
                            y={y - 10}
                            textAnchor="middle"
                            fontSize="20"
                            fill="#374151"
                            fontWeight="bold"
                          >
                            {aic.toFixed(1)}
                          </text>
                          <text
                            x={x + barWidth / 2}
                            y={margin.top + innerHeight + 40}
                            textAnchor="middle"
                            fontSize="20"
                            fill="#374151"
                            transform={`rotate(-45 ${x + barWidth / 2} ${margin.top + innerHeight + 40})`}
                          >
                            {result.distributionType}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                </g>
                <circle
                  cx={magnifier.magnifierPos.x}
                  cy={magnifier.magnifierPos.y}
                  r={magnifierSize / 2}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="3"
                />
              </>
            )}
          </svg>
          {magnifier.contextMenu && (
            <div
              className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
              style={{ left: magnifier.contextMenu.x, top: magnifier.contextMenu.y }}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  magnifier.handleMagnify();
                }}
                className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                돋보기
              </button>
            </div>
          )}
        </div>
      );
    };

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 gap-6">
          <QQPlotChart />
          <AICChart />
          <HistogramPMFChart />
          <CDFChart />
        </div>
      </div>
    );
  };

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
          <h2 className="text-xl font-bold text-gray-800">Frequency Model: {module.name}</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSpreadView(true)}
              className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-1"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
              </svg>
              Spread View
            </button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
              <XCircleIcon className="w-6 h-6" />
            </button>
          </div>
        </header>
        <main className="flex-grow p-6 overflow-auto">
          {/* Tabs */}
          <div className="border-b border-gray-200 mb-6">
            <div className="flex gap-4">
              <button
                onClick={() => setActiveTab('details')}
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'details'
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Details
              </button>
              <button
                onClick={() => setActiveTab('graphs')}
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === 'graphs'
                    ? 'border-b-2 border-blue-600 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Graphs
              </button>
            </div>
          </div>

          {activeTab === 'details' && (
            <div className="space-y-6" ref={viewDetailsRef}>
              {/* Data Statistics */}
              {statistics && (
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Data Statistics (데이터 통계)</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Count</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.count}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Mean</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.mean.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Variance</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.variance.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Std Dev</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.stdDev.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Skewness</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.skewness.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Kurtosis</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.kurtosis.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Min</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.min.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">25th Percentile</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile25.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">50th Percentile (Median)</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile50.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">75th Percentile</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile75.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">90th Percentile</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile90.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">95th Percentile</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile95.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">99th Percentile</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.percentile99.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Max</span>
                      <span className="font-mono text-sm text-gray-800 font-semibold">
                        {statistics.max.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Distribution Comparison Table */}
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Distribution Comparison</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Distribution</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">AIC</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">BIC</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Log Likelihood</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Mean</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Variance</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Dispersion</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {sortedResults.map((result, idx) => (
                        <tr 
                          key={idx}
                          className={result.distributionType === localSelected ? 'bg-blue-50' : result.error ? 'bg-red-50' : ''}
                          onClick={() => !result.error && handleSelectDistribution(result.distributionType)}
                          style={{ cursor: result.error ? 'not-allowed' : 'pointer' }}
                        >
                          <td className="px-4 py-3 text-sm font-medium text-gray-900">
                            {result.distributionType}
                            {result.distributionType === recommended?.distributionType && !result.error && (
                              <span className="ml-2 text-xs text-green-600">(Recommended)</span>
                            )}
                            {result.error && (
                              <span className="ml-2 text-xs text-red-600">(Error)</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? (
                              <span className="text-red-600 text-xs">{result.error}</span>
                            ) : (
                              result.fitStatistics?.aic?.toFixed(2) || 'N/A'
                            )}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? '-' : (result.fitStatistics?.bic?.toFixed(2) || 'N/A')}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? '-' : (result.fitStatistics?.logLikelihood?.toFixed(2) || 'N/A')}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? '-' : (result.fitStatistics?.mean?.toFixed(2) || 'N/A')}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? '-' : (result.fitStatistics?.variance?.toFixed(2) || 'N/A')}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-900">
                            {result.error ? '-' : (result.fitStatistics?.dispersion?.toFixed(2) || 'N/A')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Selected Distribution Details */}
              {localSelected && sortedResults.find(r => r.distributionType === localSelected) && (
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-800">Selected Distribution Details (Ctrl+C to copy)</h3>
                    <div className="flex items-center gap-2">
                      <label className="text-sm font-medium text-gray-700">Select Distribution:</label>
                      <select
                        value={localSelected}
                        onChange={(e) => handleSelectDistribution(e.target.value)}
                        className="px-3 py-1.5 border border-gray-300 rounded-md bg-white text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      >
                        {sortedResults.map((result) => (
                          <option key={result.distributionType} value={result.distributionType}>
                            {result.distributionType}
                            {result.distributionType === recommended?.distributionType && ' (Recommended)'}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    {(() => {
                      const selectedResult = sortedResults.find(r => r.distributionType === localSelected);
                      if (!selectedResult) return null;

                      return (
                        <div className="space-y-4">
                          <div>
                            <h4 className="text-md font-semibold text-gray-700 mb-2">Parameters</h4>
                            <div className="bg-gray-50 p-3 rounded">
                              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                {Object.entries(selectedResult.parameters).map(([key, value]) => (
                                  <div key={key} className="flex flex-col">
                                    <span className="text-xs text-gray-500 mb-1">{key}</span>
                                    <span className="font-mono text-sm text-gray-800 font-semibold">
                                      {typeof value === 'number' ? value.toFixed(6) : String(value || 'N/A')}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                          {selectedResult.fitStatistics && (
                            <div>
                              <h4 className="text-md font-semibold text-gray-700 mb-2">Fit Statistics</h4>
                              <div className="bg-gray-50 p-3 rounded">
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                  {selectedResult.fitStatistics.aic !== undefined && selectedResult.fitStatistics.aic !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">AIC</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.aic.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                  {selectedResult.fitStatistics.bic !== undefined && selectedResult.fitStatistics.bic !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">BIC</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.bic.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                  {selectedResult.fitStatistics.logLikelihood !== undefined && selectedResult.fitStatistics.logLikelihood !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">Log Likelihood</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.logLikelihood.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                  {selectedResult.fitStatistics.mean !== undefined && selectedResult.fitStatistics.mean !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">Mean</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.mean.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                  {selectedResult.fitStatistics.variance !== undefined && selectedResult.fitStatistics.variance !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">Variance</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.variance.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                  {selectedResult.fitStatistics.dispersion !== undefined && selectedResult.fitStatistics.dispersion !== null && (
                                    <div className="flex flex-col">
                                      <span className="text-xs text-gray-500 mb-1">Dispersion</span>
                                      <span className="font-mono text-sm text-gray-800 font-semibold">
                                        {selectedResult.fitStatistics.dispersion.toFixed(4)}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'graphs' && statistics && (
            <GraphsTab 
              output={output}
              sortedResults={sortedSuccessfulResults}
              recommended={recommended}
              statistics={statistics}
            />
          )}
        </main>
      </div>
      {showSpreadView && spreadViewData.length > 0 && (
        <SpreadViewModal
          onClose={() => setShowSpreadView(false)}
          data={spreadViewData}
          columns={spreadViewColumns}
          title={`Spread View: ${module.name} - Frequency Model`}
        />
      )}
    </div>
  );
};

