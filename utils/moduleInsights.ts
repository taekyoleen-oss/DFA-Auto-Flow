/**
 * 모듈별 View Details 인사이트 생성기.
 * 각 모듈의 outputData에서 핵심 지표를 추출하고, 계리 관점의 해석과 다음 단계를 제공한다.
 * 모든 추출은 방어적(옵셔널 체이닝/try)으로, outputData 형태가 달라도 안전하게 동작한다.
 */
import { CanvasModule } from '../types';

export interface InsightMetric {
  label: string;
  value: string;
}

export interface ModuleInsight {
  title: string;
  metrics: InsightMetric[];
  interpretation: string; // 계리 관점 해석 (이 결과가 의미하는 바)
  nextSteps: string;      // 다음 단계 안내
}

/** 숫자를 천단위 콤마로. 큰 금액은 그대로(원 단위). */
function num(n: any, digits = 0): string {
  const v = Number(n);
  if (!isFinite(v)) return '-';
  return v.toLocaleString('ko-KR', { maximumFractionDigits: digits });
}

/** 키 기반 dict(VaR/TVaR 등)에서 특정 신뢰수준 값을 안전하게 꺼낸다. */
function pick(dict: any, ...keys: string[]): number | undefined {
  if (!dict || typeof dict !== 'object') return undefined;
  for (const k of keys) {
    if (dict[k] != null) return Number(dict[k]);
  }
  return undefined;
}

/** outputData 또는 outputData.data에서 DataPreview를 꺼낸다. */
function preview(od: any): any | null {
  if (!od) return null;
  if (od.type === 'DataPreview' && od.rows) return od;
  if (od.data && od.data.rows) return od.data;
  return null;
}

function dataMetrics(od: any): InsightMetric[] {
  const p = preview(od);
  if (!p) return [];
  const rows = p.totalRowCount ?? p.rows?.length ?? 0;
  const cols = p.columns?.length ?? (p.rows?.[0] ? Object.keys(p.rows[0]).length : 0);
  return [
    { label: '행', value: num(rows) },
    { label: '열', value: num(cols) },
  ];
}

const GENERIC: ModuleInsight = {
  title: '결과 요약',
  metrics: [],
  interpretation:
    '이 모듈의 출력입니다. 아래 표/차트에서 상세 내용을 확인하고, 생성된 파이썬 코드로 동일 결과를 재현할 수 있습니다.',
  nextSteps: '결과가 의도와 맞는지 확인한 뒤, 다음 모듈을 연결해 파이프라인을 이어가세요.',
};

/**
 * 모듈 타입/출력에 맞는 인사이트를 생성한다. outputData가 없으면 null.
 */
export function getModuleInsight(module: CanvasModule | null | undefined): ModuleInsight | null {
  if (!module || !module.outputData) return null;
  const od = module.outputData as any;
  const t = od.type as string;

  try {
    switch (t) {
      case 'ClaimDataOutput':
      case 'InflatedDataOutput':
      case 'FormatChangeOutput':
      case 'DataPreview': {
        const isInfl = t === 'InflatedDataOutput';
        return {
          title: isInfl ? '인플레이션 적용 데이터' : t === 'ClaimDataOutput' ? '클레임 원천 데이터' : '데이터',
          metrics: dataMetrics(od),
          interpretation: isInfl
            ? '과거 클레임을 목표 연도 가치로 환산(트렌딩)한 결과입니다. 인플레이션 보정은 빈도·심도 적합의 전제이며, 보정이 빠지면 손실 규모가 과소평가됩니다.'
            : '파이프라인의 입력이 되는 클레임 데이터입니다. 컬럼(금액·날짜·연도)이 후속 모듈의 파라미터와 일치하는지 확인하세요.',
          nextSteps: isInfl
            ? 'SelectData로 필요한 컬럼만 남기고, SplitByThreshold/SplitByFreqServ로 분리해 적합 단계로 진행하세요.'
            : 'FormatChange로 날짜→연도 파생, ApplyInflation으로 가치 환산을 적용하세요.',
        };
      }
      case 'ThresholdSplitOutput': {
        const below = od.belowThreshold?.totalRowCount ?? od.belowThreshold?.rows?.length;
        const above = od.aboveThreshold?.totalRowCount ?? od.aboveThreshold?.rows?.length;
        return {
          title: 'Threshold 분리 결과',
          metrics: [
            { label: '기준(Threshold)', value: `${num(od.threshold)} 원` },
            { label: '미만(집계)', value: `${num(below)} 건` },
            { label: '이상(원본)', value: `${num(above)} 건` },
          ],
          interpretation:
            'Threshold 미만 소액 클레임은 연도별 합계(집계분포용)로, 이상 대형 클레임은 원본(빈도-심도용)으로 분리됩니다. 임계값 선택이 대형손실의 꼬리(tail) 모델링 정확도를 좌우합니다.',
          nextSteps:
            '미만 데이터는 FitAggregateModel로, 이상 데이터는 SplitByFreqServ→FitFrequency/Severity로 연결하세요. 임계값은 SettingThreshold로 조정할 수 있습니다.',
        };
      }
      case 'AggregateModelOutput': {
        const results = od.results || [];
        const best = od.selectedDistribution
          ? results.find((r: any) => r.distributionType === od.selectedDistribution)
          : [...results].filter((r: any) => r.fitStatistics?.aic != null)
              .sort((a: any, b: any) => a.fitStatistics.aic - b.fitStatistics.aic)[0];
        return {
          title: '집계분포(Aggregate) 적합',
          metrics: [
            { label: '적합 분포 수', value: `${results.length}개` },
            { label: '선택/최적 분포', value: best?.distributionType || '-' },
            { label: 'AIC', value: best?.fitStatistics?.aic != null ? num(best.fitStatistics.aic, 1) : '-' },
          ],
          interpretation:
            'Threshold 미만 손실의 연도별 집계액에 확률분포를 적합한 결과입니다. AIC가 낮을수록 적합도가 좋으며, 선택된 분포가 시뮬레이션의 기반이 됩니다.',
          nextSteps:
            'QQ/PP 플롯으로 꼬리 적합을 점검하고, 분포를 선택한 뒤 SimulateAggDist로 손실 시뮬레이션을 수행하세요.',
        };
      }
      case 'SeverityModelOutput':
      case 'FrequencyModelOutput': {
        const isSev = t === 'SeverityModelOutput';
        const results = od.results || [];
        const best = [...results].filter((r: any) => r.fitStatistics?.aic != null)
          .sort((a: any, b: any) => a.fitStatistics.aic - b.fitStatistics.aic)[0];
        return {
          title: isSev ? '심도(Severity) 모델 적합' : '빈도(Frequency) 모델 적합',
          metrics: [
            { label: '후보 분포', value: `${results.length}개` },
            { label: '최적(AIC)', value: best?.distributionType || '-' },
            { label: 'AIC', value: best?.fitStatistics?.aic != null ? num(best.fitStatistics.aic, 1) : '-' },
          ],
          interpretation: isSev
            ? '개별 대형 클레임의 금액(심도)에 분포를 적합한 결과입니다. 꼬리가 두꺼운 분포(Pareto/Lognormal)가 재보험 프라이싱에서 특히 중요합니다.'
            : '연도별 클레임 건수(빈도)에 분포를 적합한 결과입니다. 분산이 평균보다 크면(과대산포) 음이항분포가 포아송보다 적합합니다.',
          nextSteps:
            '최적 분포를 확인한 뒤 SimulateFreqServ로 빈도×심도 몬테카를로 시뮬레이션을 수행하세요.',
        };
      }
      case 'SimulateAggDistOutput': {
        const s = od.statistics || {};
        return {
          title: '집계분포 시뮬레이션',
          metrics: [
            { label: '시뮬레이션 수', value: num(od.simulationCount) },
            { label: '평균 손실', value: `${num(s.mean)} 원` },
            { label: '99% 손실', value: `${num(s.percentile99)} 원` },
          ],
          interpretation:
            '적합된 집계분포에서 손실을 반복 추출한 결과입니다. 평균은 기대손실, 상위 퍼센타일(95/99%)은 극단 손실 규모를 나타내며 자본·보험료 산정의 근거가 됩니다.',
          nextSteps:
            'CombineLossModel로 빈도-심도 시뮬레이션과 합산해 통합 손실분포(VaR/TVaR)를 산출하세요.',
        };
      }
      case 'SimulateFreqServOutput': {
        const s = od.dfaOutput?.statistics || {};
        return {
          title: '빈도-심도 시뮬레이션',
          metrics: [
            { label: '형식', value: String(od.outputFormat || 'dfa') },
            { label: '평균 손실', value: s.mean != null ? `${num(s.mean)} 원` : '-' },
            { label: '99% 손실', value: s.percentile99 != null ? `${num(s.percentile99)} 원` : '-' },
          ],
          interpretation:
            '연도별 빈도와 개별 심도를 결합한 몬테카를로 집계손실 분포입니다. 대형손실의 누적 영향을 반영하므로 재보험 레이어 가격 산정의 핵심 입력입니다.',
          nextSteps: 'CombineLossModel로 집계분포 결과와 합산하거나, XoL 프라이싱 모듈로 연결하세요.',
        };
      }
      case 'CombineLossModelOutput': {
        const cs = od.combinedStatistics || {};
        const var995 = pick(od.var, '99.5', '99');
        const tvar995 = pick(od.tvar, '99.5', '99');
        return {
          title: '통합 손실 분포 (Aggregate + Freq-Sev)',
          metrics: [
            { label: '평균 손실', value: `${num(cs.mean)} 원` },
            { label: 'VaR 99.5%', value: var995 != null ? `${num(var995)} 원` : '-' },
            { label: 'TVaR 99.5%', value: tvar995 != null ? `${num(tvar995)} 원` : '-' },
          ],
          interpretation:
            '두 손실 경로를 합산한 최종 통합 손실분포입니다. VaR은 특정 신뢰수준의 손실 한도, TVaR(=CTE)은 그 이상 구간의 평균손실로, 지급여력·재보험 한도 설정의 핵심 지표입니다. TVaR ≥ VaR가 정상입니다.',
          nextSteps:
            '신뢰수준별 VaR/TVaR을 검토해 자본 요건·재보험 구조를 결정하세요. 꼬리가 과도하면 상류 분포 적합을 재점검하세요.',
        };
      }
      case 'XolContractOutput': {
        return {
          title: 'XoL 재보험 계약 정의',
          metrics: [
            { label: '자기부담(Deductible)', value: `${num(od.deductible)} 원` },
            { label: '한도(Limit)', value: `${num(od.limit)} 원` },
            { label: '복원(Reinstatements)', value: `${num(od.reinstatements)}회` },
          ],
          interpretation:
            'Excess of Loss 계약의 핵심 조건입니다. 자기부담 초과·한도 이내 손실을 재보험자가 부담하며, 복원조항은 한도 소진 후 보장을 되살리는 대가(복원보험료)를 규정합니다.',
          nextSteps:
            'CalculateCededLoss로 출재손실을 계산하고, PriceXolContract로 보험료를 산정하세요.',
        };
      }
      case 'SettingThresholdOutput': {
        const cands = od.candidateThresholds || od.thresholds || [];
        return {
          title: '임계값(Threshold) 설정',
          metrics: [
            { label: '후보 수', value: `${cands.length}개` },
            { label: '선택값', value: od.selectedThreshold != null ? `${num(od.selectedThreshold)} 원` : '미선택' },
          ],
          interpretation:
            '대형손실을 가르는 임계값 후보입니다. 임계값은 소액(집계)과 대형(빈도-심도)의 경계를 정하며, 꼬리 리스크 측정의 민감도를 좌우합니다.',
          nextSteps: '적절한 임계값을 선택하면 연결된 SplitByThreshold에 자동 반영됩니다.',
        };
      }
      case 'SplitFreqServOutput': {
        const fy = od.yearlyFrequency?.length;
        const sv = od.severityData?.totalRowCount ?? od.severityData?.rows?.length;
        return {
          title: '빈도-심도 분리',
          metrics: [
            { label: '연도 수(빈도)', value: num(fy) },
            { label: '심도 건수', value: num(sv) },
          ],
          interpretation:
            '대형 클레임을 연도별 건수(빈도)와 개별 금액(심도)으로 분리합니다. 빈도×심도 분해는 집합손실 모델링의 표준 접근입니다.',
          nextSteps: '빈도 데이터는 FitFrequencyModel, 심도 데이터는 FitSeverityModel로 연결하세요.',
        };
      }
      case 'StatisticsOutput':
        return {
          title: '기술통계',
          metrics: dataMetrics(od),
          interpretation:
            '변수별 분포·상관을 요약합니다. 이상치·결측·강한 상관을 먼저 파악하면 후속 모델링 품질이 올라갑니다.',
          nextSteps: '필요 시 결측 처리·스케일링·컬럼 선택으로 데이터를 정제하세요.',
        };
      case 'TrainedModelOutput':
        return {
          title: '학습된 모델',
          metrics: [],
          interpretation:
            '학습 데이터로 적합된 모델입니다. 과적합 여부는 별도 검증셋의 평가로 확인해야 합니다.',
          nextSteps: 'ScoreModel/PredictModel로 예측하고 EvaluateModel로 성능을 검증하세요.',
        };
      case 'EvaluationOutput':
        return {
          title: '모델 평가',
          metrics: [],
          interpretation:
            '검증 데이터 기준 성능 지표입니다. 분류는 정확도/정밀도/재현율·혼동행렬, 회귀는 오차 지표를 함께 보세요.',
          nextSteps: '성능이 부족하면 피처·하이퍼파라미터·임계값을 조정해 재학습하세요.',
        };
      case 'SplitDataOutput': {
        const tr = od.trainData?.totalRowCount ?? od.trainData?.rows?.length;
        const te = od.testData?.totalRowCount ?? od.testData?.rows?.length;
        return {
          title: '학습/검증 데이터 분리',
          metrics: [
            { label: 'Train', value: tr != null ? `${num(tr)} 행` : '-' },
            { label: 'Test', value: te != null ? `${num(te)} 행` : '-' },
          ],
          interpretation:
            '모델 학습용(train)과 검증용(test)으로 데이터를 나눕니다. 검증셋으로 평가해야 과적합 여부를 알 수 있으며, 분리 비율·층화(stratify)·시드가 재현성에 영향을 줍니다.',
          nextSteps: 'Train 데이터로 TrainModel 학습 후, Test 데이터로 EvaluateModel 검증을 진행하세요.',
        };
      }
      case 'StatsModelsResultOutput':
        return {
          title: '통계 모델 적합 결과 (statsmodels)',
          metrics: [],
          interpretation:
            '회귀계수·표준오차·p값·적합도(R²/유사도)를 담은 결과입니다. p값이 작은 변수는 통계적으로 유의하며, 계수 부호·크기로 영향 방향과 강도를 해석합니다.',
          nextSteps: '유의하지 않은 변수 제거·변환을 검토하고, PredictModel로 예측에 활용하세요.',
        };
      case 'EvaluateStatOutput':
        return {
          title: '통계 평가',
          metrics: [],
          interpretation:
            '적합 모델의 통계적 성능·진단 지표입니다. 잔차·적합도·정보기준(AIC/BIC)을 함께 보면 모델 타당성을 판단할 수 있습니다.',
          nextSteps: '지표가 미흡하면 변수·분포 가정을 재검토하세요.',
        };
      case 'DiversionCheckerOutput':
        return {
          title: '데이터 정합성 점검 (Diversion)',
          metrics: [],
          interpretation:
            '데이터 흐름의 이상·전환(diversion)을 점검한 결과입니다. 예상과 다른 분기·누락·왜곡을 조기에 발견해 후속 분석의 신뢰성을 확보합니다.',
          nextSteps: '경고 항목을 해소한 뒤 파이프라인을 이어가세요.',
        };
      case 'XoLPriceOutput':
      case 'FinalXolPriceOutput':
      case 'XolPricingOutput': {
        const premium = od.premium ?? od.finalPrice ?? od.price ?? od.expectedCededLoss;
        return {
          title: t === 'FinalXolPriceOutput' ? '최종 XoL 보험료' : 'XoL 레이어 프라이싱',
          metrics: premium != null ? [{ label: '보험료/손실', value: `${num(premium)} 원` }] : [],
          interpretation:
            '시뮬레이션 손실분포를 바탕으로 산정한 Excess of Loss 레이어 가격입니다. 기대 출재손실에 비용·이익마진·복원조항을 반영해 최종 보험료가 결정됩니다.',
          nextSteps:
            '레이어 구조(자기부담·한도·복원)나 마진 가정을 조정해 가격 민감도를 확인하세요.',
        };
      }
      case 'AnalysisThresholdOutput':
      case 'ThresholdAnalysisOutput':
        return {
          title: '임계값 분석',
          metrics: dataMetrics(od),
          interpretation:
            '여러 임계값 후보에 대한 분리·손실 영향을 분석합니다. 임계값에 따라 대형손실 정의와 꼬리 리스크 측정이 달라집니다.',
          nextSteps: '적절한 임계값을 골라 SplitByThreshold/SettingThreshold에 반영하세요.',
        };
      default:
        return { ...GENERIC, metrics: dataMetrics(od) };
    }
  } catch {
    return GENERIC;
  }
}
