# Change History

## 2025-12-14 18:40:00

### feat(dfa): Add CSV download functionality and simplify Python code snippets

**Description:**
- 모든 View Details 모달에 CSV 다운로드 버튼 추가
  - DataPreviewModal: 데이터 테이블을 CSV로 다운로드
  - AggregateModelPreviewModal: 연도별 집계 데이터를 CSV로 다운로드
  - SimulateAggDistPreviewModal: 시뮬레이션 결과를 CSV로 다운로드
- Simulate Agg Dist 모듈의 입력 데이터 가져오기 로직 수정
  - "evaluation" 포트 타입 지원 추가
  - AggregateModelOutput 반환 타입 추가
  - PropertiesPanel에서 선택된 분포 표시
- 파라미터 유효성 검사 강화
  - SimulateAggDist에서 유효하지 않은 파라미터 자동 보정
  - None, NaN, Inf 값 처리
  - 각 분포별 파라미터 검증 추가
- Python 코드 스니펫 간소화
  - FitAggModel: 실행에 필요한 핵심 코드만 유지
  - SimulateAggDist: 실행에 필요한 핵심 코드만 유지
  - 불필요한 import 및 출력 메시지 제거

**Files Affected:**
- `App.tsx` - SimulateAggDist 입력 데이터 가져오기 로직 수정, "evaluation" 포트 타입 지원
- `components/DataPreviewModal.tsx` - CSV 다운로드 버튼 추가
- `components/AggregateModelPreviewModal.tsx` - CSV 다운로드 버튼 추가
- `components/SimulateAggDistPreviewModal.tsx` - CSV 다운로드 버튼 추가
- `components/PropertiesPanel.tsx` - SimulateAggDist 속성에 선택된 분포 표시, AggregateModelOutput 타입 지원
- `utils/pyodideRunner.ts` - 파라미터 유효성 검사 강화, safe_get 함수 추가
- `codeSnippets.ts` - FitAggModel과 SimulateAggDist 코드 간소화

**Reason:**
- 사용자가 결과 데이터를 쉽게 다운로드할 수 있도록 하기 위해
- 코드 스니펫을 더 간결하고 이해하기 쉽게 만들기 위해
- 파라미터 오류를 방지하고 자동으로 보정하기 위해

**Commit Hash:** b81aaba8f2fbc0b1da83fa79dd09998aa65bfb15

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-12-14 13:57:31

### refactor(dfa): Change LoadClaimData to CSV file loading instead of auto-generation

**Description:**
- LoadClaimData 모듈을 자동 생성 방식에서 CSV 파일 로드 방식으로 변경
- 기존 Load Data 모듈과 동일한 방식으로 CSV 파일을 불러오도록 수정
- 샘플 클레임 데이터 CSV 파일 생성 (public/claim_data.csv)
  - 5년간 데이터 (2020-2024)
  - 연도별 20개씩 총 100건
  - 컬럼: 종목구분, 날짜, 클레임 금액, 기타
- LoadClaimData 모듈이 파일을 로드하면 ClaimDataOutput으로 래핑하여 출력
- DFA 파이프라인 자동 생성 시 LoadClaimData 파라미터를 파일 로드 방식으로 변경

**Files Affected:**
- `App.tsx` - LoadClaimData 실행 로직을 Load Data와 동일하게 변경 (CSV 파일 로드)
- `constants.ts` - LoadClaimData 기본 파라미터를 파일 로드 방식으로 변경
- `public/claim_data.csv` - 샘플 클레임 데이터 CSV 파일 생성
- `scripts/generate_claim_data.py` - 클레임 데이터 생성 스크립트 추가

**Reason:**
- 사용자가 실제 CSV 파일을 불러올 수 있도록 하기 위해
- 기존 Load Data 모듈과 일관된 사용자 경험 제공

**Commit Hash:** (커밋 후 업데이트 예정)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-12-14 13:49:51

### feat(dfa): Add DFA (Dynamic Financial Analysis) modules for insurance claim analysis

**Description:**
- 손해보험 DFA 시스템과 비비례재보험 모델 구축을 위한 5개의 DFA 모듈 추가
- LoadClaimData: 5년간의 클레임 데이터 자동 생성 (종목구분, 날짜, 클레임 금액, 기타)
- ApplyInflation: 연간 상승률을 적용하여 목표 연도까지 보험금 증가 계산
- SplitByThreshold: Threshold 기준으로 데이터 분리 (연도별 합계 및 원본 레이아웃 유지)
- FitAggregateModel: 연도별 집합 금액에 통계 분포 적합 (정규분포, 로그정규분포, 파레토, 감마, 와이블)
- FitFrequencySeverityModel: 빈도-심도 모델 적합 (Poisson/NegativeBinomial + Normal/Lognormal/Pareto/Gamma/Exponential/Weibull)
- 모든 모듈은 Python 코드로 구현되어 향후 검증 가능
- Toolbox에 "DFA Analysis" 카테고리 추가

**Files Affected:**
- `types.ts` - DFA 모듈 타입 및 출력 타입 추가
- `constants.ts` - DFA 모듈 정의 및 기본 설정 추가
- `components/Toolbox.tsx` - DFA Analysis 카테고리 추가
- `utils/pyodideRunner.ts` - DFA 모듈 Python 실행 함수 추가 (loadClaimDataPython, applyInflationPython, splitByThresholdPython, fitAggregateModelPython, fitFrequencySeverityModelPython)
- `App.tsx` - DFA 모듈 실행 로직 추가 및 getSingleInputData 함수 확장
- `codeSnippets.ts` - DFA 모듈 Python 코드 템플릿 추가

**Reason:**
- 손해보험 DFA 시스템과 비비례재보험 모델 구축을 위한 필수 기능 구현
- 클레임 데이터 생성부터 인플레이션 적용, Threshold 분리, 통계 모델 적합까지 전체 파이프라인 제공

**Commit Hash:** (커밋 후 업데이트 예정)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-12-14 13:23:43

### refactor(toolbox): Remove Data Analysis and Traditional Analysis module contents for DFA system

**Description:**
- Data Analysis 카테고리의 모든 모듈 내용 제거 (카테고리 이름만 유지)
- Traditional Analysis 카테고리의 모든 모듈 내용 제거 (카테고리 이름만 유지)
- 사용되지 않는 타입 정의 제거 (analysisOpTypes, supervisedLearningTypes, unsupervisedModelTypes, traditionAnalysisTypes)
- expandedCategories에서 사용되지 않는 서브카테고리 제거 (Operations, Supervised Learning, Unsupervised Learning)
- 손해보험 DFA 시스템과 비비례재보험 모델 구축을 위한 앱 정리

**Files Affected:**
- `components/Toolbox.tsx` - Data Analysis와 Traditional Analysis 카테고리의 모듈 내용 제거, 사용되지 않는 타입 정의 제거

**Reason:**
- 손해보험 DFA 시스템과 비비례재보험 모델 구축을 위해 Data Analysis와 Traditional Analysis 카테고리의 내용을 제거하고 카테고리 이름만 유지
- 앱이 정상적으로 실행되도록 불필요한 코드 정리

**Commit Hash:** (커밋 후 업데이트 예정)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-12-14 09:35:26

### feat(pca): Improve PCA visualization with JavaScript-based implementation

**Description:**
- PCA 시각화를 JavaScript 기반(ml-pca)으로 전환하여 성능 개선
- Python Pyodide 의존성 제거로 안정성 향상
- Label Column을 선택 사항으로 변경하고 Predict를 기본값으로 설정
- 이진 분류(0/1)를 위한 간소한 색상 체계 적용 (파란색: 클래스 0, 빨간색: 클래스 1)
- Color Scale 범례 제거 및 그래프 너비 1400px로 확장
- 콤보박스에서 "None (Basic PCA)" 옵션 제거
- 그래프 가시성 개선 (그리드 라인, 축 레이블, 레이아웃 개선)

**Files Affected:**
- `utils/pcaCalculator.ts` - ml-pca 라이브러리를 사용한 JavaScript 기반 PCA 계산 함수 추가
- `components/DataPreviewModal.tsx` - PCA Visualization 개선 (Label Column 선택 사항화, 그래프 크기 및 스타일 개선)
- `package.json` - ml-pca 라이브러리 의존성 추가
- `pnpm-lock.yaml` - 의존성 업데이트

**Reason:**
- Python Pyodide 기반 PCA 구현에서 발생한 패키지 로딩 및 데이터 마샬링 문제 해결
- 브라우저 환경에서 더 안정적이고 빠른 PCA 계산을 위해 JavaScript 기반 구현으로 전환
- 사용자 경험 개선을 위한 시각화 개선

**Commit Hash:** de7bb9092853b58ba903cf6788e0904a2c4d05d7

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7

# Or direct recovery
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7
```

## 2025-12-12 17:50:00

### feat(samples): Add Samples folder support and Linear Regression-1 sample

**Description:**
- Samples 폴더 기능 추가 및 Linear Regression-1 샘플 추가
- Samples 폴더의 파일을 자동으로 읽어서 Samples 메뉴에 표시하는 기능 구현
- Save 버튼으로 저장한 .mla 파일을 samples 폴더에 넣으면 자동으로 표시되도록 개선
- File System Access API 오류 처리 개선
- 파일 이름의 공백 및 특수문자 처리 (URL 인코딩/디코딩)

**Files Affected:**
- `App.tsx` - Samples 폴더 파일 로드 기능 추가, File System Access API 오류 처리 개선
- `server/samples-server.js` - Samples 폴더 파일 목록 및 읽기 API 구현
- `savedSamples.ts` - Linear Regression-1 샘플 추가
- `samples/README.md` - Samples 폴더 사용 방법 문서 추가
- `samples/example.json` - 예제 파일 추가
- `package.json` - samples-server 스크립트 추가
- `vite.config.ts` - /api/samples 프록시 설정 추가
- `types.ts` - StatsModelFamily에 Logit, QuasiPoisson 추가, DiversionCheckerOutput, EvaluateStatOutput 타입 추가

**Reason:**
- 사용자가 Save 버튼으로 저장한 모델을 samples 폴더에 넣으면 자동으로 Samples 메뉴에 표시되도록 하기 위해
- Linear Regression-1 샘플을 공유 가능한 샘플로 추가

**Commit Hash:** b7dfe9fc6c744f5d41e2d417afa575205c80fbec

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec

# Or direct recovery
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec
```
