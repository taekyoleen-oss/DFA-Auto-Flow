# Preview Backup

이 폴더는 모든 모듈의 View Details 내용을 백업하기 위한 폴더입니다.

## 백업된 파일 목록

### DataPreviewModal_Backup.tsx
- **사용 모듈**: LoadClaimData, FormatChange, ApplyInflation, SelectData, ApplyThreshold, XoL Calculator, Split By Threshold, DefineXolContract, ScoreModel 등
- **설명**: 여러 모듈에서 공통으로 사용되는 DataPreviewModal의 전체 코드 백업
- **주요 기능**:
  - 일반 데이터 테이블 표시
  - XoL Calculator 모듈 전용 View Details (건별/연도별 탭)
  - Split By Threshold 모듈 전용 View Details (Threshold 이하/이상 탭, 그래프 포함)
  - DefineXolContract 모듈 전용 View Details
  - LoadClaimData 모듈 전용 View Details (Detail/Graphs 탭)
  - ScoreModel 모듈 전용 View Details
  - Spread View 및 CSV 다운로드 기능
  - HistogramPlot, YearlyAmountBarPlot 컴포넌트 포함

### SettingThresholdPreviewModal_Backup.tsx
- **사용 모듈**: Setting Threshold
- **설명**: Setting Threshold 모듈의 View Details 백업
- **주요 기능**:
  - 분석 탭: 연도별 건수 분석, 통계량 표시
  - 분포 탭: 데이터 분포 히스토그램
  - Spread View 및 CSV 다운로드

### SplitFreqServPreviewModal_Backup.tsx
- **사용 모듈**: Split By Freq-Sev
- **설명**: Split By Freq-Sev 모듈의 View Details 백업
- **주요 기능**:
  - Frequency 탭: 연도별 빈도 데이터 및 통계량
  - Severity 탭: 연도별 심도 데이터 및 보험금 데이터
  - Spread View 및 CSV 다운로드

### AggregateModelPreviewModal_Backup.tsx
- **사용 모듈**: Fit Aggregate Model
- **설명**: Fit Aggregate Model 모듈의 View Details 백업
- **주요 기능**:
  - 데이터 통계량 표시
  - 분포 비교 테이블 (AIC, BIC, Log Likelihood)
  - 분포 파라미터 표시
  - 연도별 집계 데이터
  - Cumulative Distribution Chart, Q-Q Plot, P-P Plot
  - Spread View 및 CSV 다운로드

### SimulateAggDistPreviewModal_Backup.tsx
- **사용 모듈**: Simulate Agg Dist
- **설명**: Simulate Agg Dist 모듈의 View Details 백업
- **주요 기능**:
  - 시뮬레이션 정보 표시
  - Raw Simulation Results 테이블
  - 통계량 표시
  - 히스토그램 차트
  - Spread View 및 CSV 다운로드

### FrequencyModelPreviewModal_Backup.tsx
- **사용 모듈**: Fit Frequency Model
- **설명**: Fit Frequency Model 모듈의 View Details 백업
- **주요 기능**:
  - Details 탭: 데이터 통계량, 분포 비교 테이블, 선택된 분포 상세 정보
  - Graphs 탭: Histogram + PMF, Empirical CDF vs Fitted CDF, Q-Q Plot, AIC Comparison
  - 돋보기 기능 (Magnifier)
  - Spread View 및 CSV 다운로드

### SeverityModelPreviewModal_Backup.tsx
- **사용 모듈**: Fit Severity Model
- **설명**: Fit Severity Model 모듈의 View Details 백업
- **주요 기능**:
  - Details 탭: 데이터 통계량, 분포 비교 테이블, 선택된 분포 상세 정보
  - Graphs 탭: Histogram + PDF, Empirical CDF vs Fitted CDF, Q-Q Plot, AIC Comparison
  - 돋보기 기능 (Magnifier)
  - Spread View 및 CSV 다운로드

### SimulateFreqServPreviewModal_Backup.tsx
- **사용 모듈**: Simulate Freq-Sev
- **설명**: Simulate Freq-Sev 모듈의 View Details 백업
- **주요 기능**:
  - 시뮬레이션 정보 표시
  - XoL 형식: 사고별 집계 테이블 (시뮬레이션 번호, 보험금)
  - DFA 형식: Raw Simulation Results 테이블
  - 통계량 표시
  - 히스토그램 차트
  - Spread View 및 CSV 다운로드

### XolPricingPreviewModal_Backup.tsx
- **사용 모듈**: XoL Pricing
- **설명**: XoL Pricing 모듈의 View Details 백업

### CombineLossModelPreviewModal_Backup.tsx
- **사용 모듈**: Combine Loss Model
- **설명**: Combine Loss Model 모듈의 View Details 백업

### SplitDataPreviewModal_Backup.tsx
- **사용 모듈**: Split Data
- **설명**: Split Data 모듈의 View Details 백업

### StatisticsPreviewModal_Backup.tsx
- **사용 모듈**: Statistics
- **설명**: Statistics 모듈의 View Details 백업

### EvaluationPreviewModal_Backup.tsx
- **사용 모듈**: Evaluate Model
- **설명**: Evaluation 모듈의 View Details 백업

### TrainedModelPreviewModal_Backup.tsx
- **사용 모듈**: Train Model
- **설명**: Trained Model 모듈의 View Details 백업

### DiversionCheckerPreviewModal_Backup.tsx
- **사용 모듈**: Diversion Checker
- **설명**: Diversion Checker 모듈의 View Details 백업

### EvaluateStatPreviewModal_Backup.tsx
- **사용 모듈**: Evaluate Stat
- **설명**: Evaluate Stat 모듈의 View Details 백업

### XoLPricePreviewModal_Backup.tsx
- **사용 모듈**: XoL Price
- **설명**: XoL Price 모듈의 View Details 백업

### StatsModelsResultPreviewModal_Backup.tsx
- **사용 모듈**: Stats Models Result
- **설명**: Stats Models Result 모듈의 View Details 백업

### FinalXolPricePreviewModal_Backup.tsx
- **사용 모듈**: Final XoL Price
- **설명**: Final XoL Price 모듈의 View Details 백업

## 복구 방법

1. 문제가 발생한 모듈의 View Details를 확인합니다.
2. 해당 모듈에 맞는 백업 파일을 찾습니다.
3. 백업 파일의 내용을 참조하여 원본 파일을 복구합니다.

### DataPreviewModal 복구 시 주의사항

`DataPreviewModal.tsx`는 여러 모듈에서 사용되므로, 특정 모듈만 복구할 때는 다음을 확인하세요:

- **XoL Calculator**: `module.type === ModuleType.XolCalculator` 조건문 부분
- **Split By Threshold**: `module.type === ModuleType.SplitByThreshold` 조건문 부분
- **DefineXolContract**: `module.type === ModuleType.DefineXolContract` 조건문 부분
- **LoadClaimData**: `module.type === ModuleType.LoadClaimData` 조건문 부분
- **ScoreModel**: `module.type === ModuleType.ScoreModel` 조건문 부분

## 주의사항

- 백업 파일은 참조용입니다. 직접 import하여 사용할 수 없습니다.
- 백업 파일은 원본 파일의 전체 내용을 포함합니다.
- 복구 시 의존성 컴포넌트들도 함께 확인해야 합니다:
  - `SpreadViewModal`
  - `ColumnStatistics`
  - `HistogramPlot`
  - `YearlyAmountBarPlot`
  - Icons (`XCircleIcon`, `ArrowDownTrayIcon`, `ChevronUpIcon`, `ChevronDownIcon` 등)

## 백업 일자

- **전체 백업**: 2025-01-21
- **이전 백업**:
  - XoL Calculator: 2025-01-17
  - Load Claim Data: 2025-01-17
  - Setting Threshold: 2025-01-17

## 백업 파일 구조

```
PreviewBackup/
├── README.md (이 파일)
├── DataPreviewModal_Backup.tsx
├── SettingThresholdPreviewModal_Backup.tsx
├── SplitFreqServPreviewModal_Backup.tsx
├── AggregateModelPreviewModal_Backup.tsx
├── SimulateAggDistPreviewModal_Backup.tsx
├── FrequencyModelPreviewModal_Backup.tsx
├── SeverityModelPreviewModal_Backup.tsx
├── SimulateFreqServPreviewModal_Backup.tsx
├── XolPricingPreviewModal_Backup.tsx
├── CombineLossModelPreviewModal_Backup.tsx
├── SplitDataPreviewModal_Backup.tsx
├── StatisticsPreviewModal_Backup.tsx
├── EvaluationPreviewModal_Backup.tsx
├── TrainedModelPreviewModal_Backup.tsx
├── DiversionCheckerPreviewModal_Backup.tsx
├── EvaluateStatPreviewModal_Backup.tsx
├── XoLPricePreviewModal_Backup.tsx
├── StatsModelsResultPreviewModal_Backup.tsx
└── FinalXolPricePreviewModal_Backup.tsx
```
