# Change History

## 2025-01-16 09:00:00

### feat(xol): Add XoL Pricing module and improve XOL Calculator UI

**Description:**
- XoL Pricing 모듈 추가
  - XoL Pricing 카테고리에 새로운 모듈 추가
  - XOL Calculator의 출력 데이터를 입력으로 받아 Net Premium 및 Gross Premium 계산
  - Expense Rate 파라미터 지원 (기본값: 0.2)
  - View Details에 입력값, 계산 수식, 계산 결과 표시
- XOL Calculator View Details 개선
  - '연도별 XoL 적용' 탭 레이아웃 재설계: 테이블 상단, 통계 및 그래프 하단
  - 통계 정보 섹션 추가: XoL Claim 평균, 표준편차, XoL Premium Rate 평균, Reluctance Factor
  - XoL Premium Rate 컬럼 계산 로직 추가
  - 탭 이름 변경: '사고별 XoL 적용', '연도별 XoL 적용'
- XoL Contract 모듈에서 Expense Ratio를 Reluctance Factor로 이름 변경
- 사용하지 않는 모듈 제거: XoL Loading, Calculate Ceded Loss, Price XoL Contract

**Files Affected:**
- `types.ts` - ModuleType.XolPricing 및 XolPricingOutput 타입 추가
- `constants.ts` - XoL Pricing 모듈 정의 추가
- `components/Toolbox.tsx` - XoL Pricing 모듈 추가
- `components/PropertiesPanel.tsx` - XoL Pricing 속성 패널 추가, Expense Ratio → Reluctance Factor 이름 변경
- `App.tsx` - XoL Pricing 모듈 실행 로직 추가, 모달 닫기 기능 수정
- `components/DataPreviewModal.tsx` - XOL Calculator View Details 레이아웃 개선, XoL Premium Rate 계산 추가
- `components/XolPricingPreviewModal.tsx` - 새로 생성, View Details 모달 구현

**Reason:**
- 사용자 요청에 따라 XoL Pricing 모듈 추가 및 XOL Calculator UI 개선
- XoL Premium Rate 계산 기능 추가로 더 정확한 프리미엄 계산 지원

**Commit Hash:** c959f76ca21ec5f06c9f1a9fb8c387e0587d0b2d

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard c959f76ca21ec5f06c9f1a9fb8c387e0587d0b2d

# Or direct recovery
git reset --hard c959f76ca21ec5f06c9f1a9fb8c387e0587d0b2d
```

## 2025-01-16 08:10:00

### fix(config): Ensure XoL Contract default values apply to initial screen models

**Description:**
- XoL Contract 모듈의 기본값이 초기화면 모델과 샘플 모델에도 적용되도록 수정
  - 초기 모델 로드 시 (localStorage의 initialModel)
  - 샘플 모델 로드 시 (handleLoadSample)
  - XoL Contract 모듈의 경우 저장된 parameters를 무시하고 DEFAULT_MODULES의 기본값을 강제 적용
  - 이제 새로운 모듈 추가, 초기 모델 로드, 샘플 모델 로드 모든 경우에 동일한 기본값 적용

**Files Affected:**
- `App.tsx` - 초기 모델 로드 및 샘플 모델 로드 로직 수정
  - XoL Contract 모듈의 경우 기본값을 강제 적용하도록 조건 추가
  - 초기 모델 로드 시 (1608-1643 라인)
  - 샘플 모델 로드 시 (1150-1177 라인)

**Reason:**
- 사용자 요청에 따라 초기화면의 모델에도 변경된 기본값이 적용되도록 보장
- localStorage에 저장된 이전 값이 있어도 최신 기본값이 적용되도록 함

**Commit Hash:** c9082c251c79f3966136e044e126341d703b6391

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard c9082c251c79f3966136e044e126341d703b6391

# Or direct recovery
git reset --hard c9082c251c79f3966136e044e126341d703b6391
```

## 2025-01-16 08:00:00

### fix(config): Update XoL Contract module default values

**Description:**
- XoL Contract 모듈의 기본값 변경
  - Deductible: 1000000 → 3000000
  - Limit: 500000 → 2000000
  - Reinstatements: 5 → 2
  - 새로운 모듈 생성 시 변경된 기본값이 적용됨

**Files Affected:**
- `constants.ts` - DEFAULT_MODULES 배열의 DefineXolContract 모듈 기본값 수정

**Reason:**
- 사용자 요청에 따라 XoL Contract 모듈의 기본값을 변경하여 더 적절한 초기값 제공

**Commit Hash:** c9082c251c79f3966136e044e126341d703b6391

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard c9082c251c79f3966136e044e126341d703b6391

# Or direct recovery
git reset --hard c9082c251c79f3966136e044e126341d703b6391
```

## 2025-01-16 07:30:00

### feat(ui): Add horizontal scroll to XOL Calculator table and change histogram to bar chart

**Description:**
- XOL Calculator 모듈의 View Details 테이블에 좌우 스크롤 추가
  - 테이블 컨테이너에 `overflow-x-auto` 추가하여 좌우 스크롤 가능
  - 테이블에 `minWidth: 'max-content'` 스타일 추가하여 컬럼이 많을 때 스크롤 활성화
- XoL Claim(Incl. Agg/Reinst) 탭의 히스토그램을 연도별 막대그래프로 변경
  - 기존: 선택된 열의 히스토그램 표시
  - 변경: 연도를 가로축, 숫자형 열(클레임 금액_infl, XoL Claim(Incl. Limit), XoL Claim(Incl. Agg/Reinst))을 세로축으로 하는 막대그래프 표시
  - 각 숫자형 열마다 별도의 막대그래프 표시
  - YearlyAmountBarPlot 컴포넌트를 사용하여 연도별 값 표시
  - XoL Claim(Incl. Limit) 탭은 기존 히스토그램 유지

**Files Affected:**
- `components/DataPreviewModal.tsx` - XOL Calculator 테이블 스크롤 및 그래프 변경
  - 테이블 컨테이너에 overflow-x-auto 추가
  - 테이블에 minWidth 스타일 추가
  - XoL Claim(Incl. Agg/Reinst) 탭의 히스토그램을 연도별 막대그래프로 변경

**Reason:**
- 사용자 요청에 따라 테이블의 좌우 스크롤 기능 추가로 많은 컬럼을 가진 테이블도 편리하게 탐색 가능
- 연도별 집계 데이터를 막대그래프로 시각화하여 트렌드를 더 쉽게 파악 가능

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 07:00:00

### feat(ui): Add XoL Claim(Incl. Agg/Reinst) column to XOL Calculator second tab

**Description:**
- XOL Calculator 모듈의 두 번째 탭(XoL Claim(Incl. Agg/Reinst))에 새로운 열 추가
  - 테이블 열 구성: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit), XoL Claim(Incl. Agg/Reinst)
  - XoL Claim(Incl. Agg/Reinst) 계산 수식:
    - XoL Claim(Incl. Limit) >= Limit*(Reinstatements+1) + Aggregate Deductible 이라면
      → Limit*(Reinstatements+1) + Aggregate Deductible
    - 그렇지 않다면 → XoL Claim(Incl. Limit)
  - XoL Contract 모듈의 정보(limit, reinstatements, aggDeductible)를 사용하여 계산
  - allModules와 allConnections props를 DataPreviewModal에 추가하여 upstream 모듈 정보 접근 가능

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Claim(Incl. Agg/Reinst) 컬럼 추가 및 계산 로직 구현
  - DataPreviewModalProps에 allModules, allConnections 추가
  - getXolData 함수에서 XoL Contract 정보 가져오기
  - 연도별 집계 시 XoL Claim(Incl. Agg/Reinst) 계산 추가

**Reason:**
- 사용자 요청에 따라 XoL Claim(Incl. Agg/Reinst) 탭에 계산된 컬럼 추가
- XoL Contract 조건에 따른 최대값 제한을 반영한 계산 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 06:30:00

### feat(ui): Add year aggregation for XOL Calculator second tab with specific columns

**Description:**
- XOL Calculator 모듈의 두 번째 탭(XoL Claim(Incl. Agg/Reinst))에 연도별 집계 기능 추가
  - 두 번째 탭 선택 시 기존 테이블을 연도별로 집계하여 표시
  - 테이블 열 구성: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit)
  - 연도 컬럼(연도, year 등)을 자동으로 찾아서 연도별로 집계
  - 클레임 금액_infl과 XoL Claim(Incl. Limit) 컬럼을 연도별로 합계 계산
  - 집계된 데이터는 연도, 클레임 금액_infl(합계), XoL Claim(Incl. Limit)(합계)만 포함
  - 첫 번째 탭과 동일한 레이아웃 유지 (왼쪽 테이블, 오른쪽 통계량, 아래 히스토그램)
  - 연도 컬럼이 없는 경우 빈 데이터 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XOL Calculator 모듈의 getXolData 함수 수정
  - 두 번째 탭(aggreinst) 선택 시 연도별 집계 로직 추가
  - 연도별 합계 계산 및 새로운 데이터 구조 생성

**Reason:**
- 사용자 요청에 따라 XOL Calculator 모듈의 두 번째 탭에서 연도별 집계 데이터를 확인할 수 있도록 개선
- 연도별 트렌드를 쉽게 파악할 수 있도록 집계 기능 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 06:00:00

### fix(config): Update XoL Contract module default values

**Description:**
- XoL Contract 모듈의 기본값 변경
  - Deductible: 250000 → 1000000
  - Limit: 1000000 → 500000
  - 새로운 모듈 생성 시 변경된 기본값이 적용됨

**Files Affected:**
- `constants.ts` - DEFAULT_MODULES 배열의 DefineXolContract 모듈 기본값 수정

**Reason:**
- 사용자 요청에 따라 XoL Contract 모듈의 기본값을 변경하여 더 적절한 초기값 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 05:30:00

### fix(ui): Remove top statistics table from XOL Calculator View Details

**Description:**
- XOL Calculator 모듈의 View Details에서 상단 통계량 테이블 제거
  - 기존: 상단에 StatsTable 표시
  - 변경: 상단 통계량 테이블 제거, 테이블과 통계량/히스토그램만 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XOL Calculator 모듈의 상단 StatsTable 제거

**Reason:**
- 사용자 요청에 따라 XOL Calculator 모듈의 View Details에서 상단 통계량 테이블 제거
- 선택된 열의 통계량과 히스토그램만 표시하여 더 간결한 레이아웃 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 05:00:00

### feat(ui): Update XOL Calculator View Details with table, statistics, and histogram layout

**Description:**
- XOL Calculator 모듈의 View Details 레이아웃 수정
  - 왼쪽: 데이터 테이블 (컬럼 클릭 가능)
  - 오른쪽: 선택된 열의 통계량 표시
  - 아래: 선택된 열의 히스토그램 표시 (숫자형 컬럼인 경우)
  - 상하 스크롤 가능하도록 flex-grow와 overflow-auto 적용
  - 컬럼 클릭 시 선택된 컬럼에 대한 통계량과 히스토그램 자동 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XOL Calculator 모듈 View Details 레이아웃 수정
  - 왼쪽/오른쪽 레이아웃으로 변경 (테이블 + 통계량)
  - 선택된 열의 히스토그램을 아래에 추가
  - 상하 스크롤 가능하도록 컨테이너 구조 변경

**Reason:**
- 사용자 요청에 따라 XOL Calculator 모듈의 View Details를 더 직관적인 레이아웃으로 개선
- 테이블, 통계량, 히스토그램을 한 화면에서 확인할 수 있도록 개선
- 상하 스크롤을 통해 모든 정보를 쉽게 탐색 가능

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 04:30:00

### feat(ui): Update Select Data module View Details to match Apply Threshold layout

**Description:**
- Select Data 모듈의 View Details를 Apply Threshold 모듈과 동일한 형태로 수정
  - 탭 구성: Data Table + Histogram (visualization 탭 제거)
  - Data Table 탭: 왼쪽에 테이블, 오른쪽에 선택된 열의 통계량만 표시 (그래프 제거)
  - Histogram 탭: 히스토그램 및 통계량 표시
  - Select Data 모듈도 Apply Threshold와 동일하게 통계량만 표시하도록 변경

**Files Affected:**
- `components/DataPreviewModal.tsx` - Select Data 모듈 View Details 수정
  - Select Data 모듈에 Histogram 탭 추가
  - visualization 탭에서 Select Data 제외
  - Data Table 탭에서 Select Data는 통계량만 표시 (그래프 제거)
  - Histogram 탭에 Select Data 추가 (Apply Threshold와 동일한 구조)

**Reason:**
- 사용자 요청에 따라 Select Data 모듈을 Apply Threshold 모듈과 동일한 형태로 통일
- Data Table 탭에서는 테이블과 통계량에 집중, 히스토그램은 별도 탭에서 확인하도록 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 04:00:00

### fix(ui): Remove right-side graph from Apply Threshold Data Table tab and show statistics instead

**Description:**
- Apply Threshold 모듈의 View Details에서 Data Table 탭의 오른쪽 상단 그래프 제거 및 통계량 표시 추가
  - 기존: 컬럼 선택 시 오른쪽에 그래프와 통계량 표시
  - 변경: Apply Threshold 모듈의 경우 오른쪽에 선택한 열의 통계량만 표시 (그래프 제거)
  - ColumnStatistics 컴포넌트를 사용하여 선택한 열에 대한 통계량 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - Apply Threshold 모듈의 Data Table 탭에서 오른쪽 그래프 패널 제거
  - selectedColumn 조건에 `module.type !== ModuleType.ApplyThreshold` 추가

**Reason:**
- 사용자 요청에 따라 Apply Threshold 모듈의 Data Table 탭에서 오른쪽 상단 그래프 제거
- 히스토그램은 별도 Histogram 탭에서 확인 가능하므로 Data Table 탭에서는 테이블만 표시

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 03:30:00

### fix(ui): Fix Apply Threshold module View Details and add histogram tab

**Description:**
- Apply Threshold 모듈의 View Details 수정
  - Apply Threshold 모듈은 Split By Threshold 모듈과 달리 Threshold 미만의 데이터를 처리하지 않음
  - ThresholdSplitOutput 처리 부분에서 Apply Threshold 모듈 제외 (Apply Threshold는 DataPreview를 사용)
  - Threshold 미만 데이터 탭 제거 (Apply Threshold는 threshold 이상 데이터만 처리)
- Apply Threshold 모듈에 히스토그램 탭 추가
  - 기존: table 탭만 존재
  - 변경: table 탭 + Histogram 탭 추가
  - Histogram 탭에서 Select Data 모듈과 동일하게 히스토그램 및 통계량 표시
  - amount_column 파라미터를 사용하여 히스토그램 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - Apply Threshold 모듈 처리 수정 및 히스토그램 탭 추가
  - ThresholdSplitOutput 처리 부분에서 Apply Threshold 모듈 제외
  - activeTab 타입에 'histogram' 추가
  - Apply Threshold 모듈에 Histogram 탭 버튼 추가
  - Histogram 탭 콘텐츠 추가 (Select Data 모듈과 동일한 구조)

**Reason:**
- Apply Threshold 모듈의 특성에 맞게 View Details 수정 (Threshold 미만 데이터 미처리)
- 사용자 요청에 따라 히스토그램을 별도 탭으로 분리하여 표시
- Select Data 모듈과 동일한 방식으로 히스토그램 표시하여 일관성 유지

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 03:00:00

### feat(ui): Add histogram visualization to Select Data and Apply Threshold modules

**Description:**
- Select Data 모듈의 View Details에 히스토그램 추가
  - table 탭에서 컬럼 선택 시 오른쪽에 히스토그램과 통계량을 함께 표시
  - 기존: 통계량만 표시
  - 변경: 히스토그램 + 통계량 표시
- Apply Threshold 모듈의 View Details에 히스토그램 추가
  - Threshold < threshold 탭: 상단 통계량 → 히스토그램 → 아래 전체 데이터 테이블 레이아웃
  - Threshold >= threshold 탭: 상단 통계량 → 히스토그램 → 아래 전체 데이터 테이블 레이아웃
  - 금액 컬럼을 자동으로 찾아서 히스토그램 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - Select Data 및 Apply Threshold 모듈에 히스토그램 추가
  - Select Data 모듈: table 탭에서 선택한 컬럼에 대한 히스토그램 표시 추가
  - Apply Threshold 모듈: belowThreshold와 aboveThreshold 각각에 히스토그램 추가
  - HistogramPlot 컴포넌트를 사용하여 금액 컬럼의 분포를 시각화

**Reason:**
- 사용자 요청에 따라 Select Data와 Apply Threshold 모듈의 View Details에서 데이터 분포를 시각적으로 확인할 수 있도록 히스토그램 추가
- 데이터 분석 시 분포를 한눈에 파악할 수 있어 분석 효율성 향상

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 02:30:00

### style(ui): Adjust View Details font size and ensure consistent layout

**Description:**
- View Details의 모든 테이블 글자 크기 조정
  - 테이블 헤더 및 셀 패딩: py-2 px-3 → py-1 px-2, py-1.5 px-3 → py-1 px-2, py-1.5 px-2 → py-1 px-2
  - Apply Threshold 모듈의 테이블 헤더 패딩 조정
  - XOL Calculator 모듈의 테이블 헤더 패딩 조정
  - 일반 테이블의 헤더 및 셀 패딩 조정
- Apply Threshold와 XOL Calculator 모듈 레이아웃 확인
  - 두 모듈 모두 상단 통계량 테이블 + 아래 전체 데이터 테이블 레이아웃으로 이미 구성되어 있음

**Files Affected:**
- `components/DataPreviewModal.tsx` - View Details 테이블의 모든 패딩 크기 조정
  - Apply Threshold 모듈: th 패딩 py-1.5 px-2 → py-1 px-2
  - XOL Calculator 모듈: th 패딩 py-1.5 px-2 → py-1 px-2
  - 일반 테이블: th 패딩 py-2 px-3 → py-1 px-2, td 패딩 py-1.5 px-3 → py-1 px-2

**Reason:**
- 사용자 요청에 따라 View Details의 글자 크기를 더 작게 조정하여 더 많은 정보를 한 화면에 표시
- 일관된 패딩 크기로 UI 통일성 향상

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 02:00:00

### feat(ui): Redesign Apply Threshold and XoOL Calculator View Details layout

**Description:**
- Apply Threshold 모듈의 View Details 레이아웃 재설계
  - 기존: 왼쪽 테이블 + 오른쪽 그래프/통계량 레이아웃
  - 변경: 상단 통계량 테이블 + 아래 전체 데이터 테이블 레이아웃
  - 글자 크기를 현재보다 약간 작게 조정 (text-sm → text-xs, py-2 px-3 → py-1 px-2)
- XoOL Calculator 모듈의 View Details 첫 번째 탭에도 동일한 레이아웃 적용
  - 상단 통계량 테이블 + 아래 전체 데이터 테이블 구조 유지
  - 글자 크기를 현재보다 약간 작게 조정
- StatsTable 컴포넌트 글자 크기 조정
  - 제목: text-lg → text-sm
  - 테이블: text-sm → text-xs
  - 셀 패딩: py-1.5 px-3 → py-1 px-2

**Files Affected:**
- `components/DataPreviewModal.tsx` - Apply Threshold 및 XoOL Calculator 모듈의 View Details 레이아웃 변경
  - Apply Threshold 모듈: ThresholdSplitOutput 처리 부분을 상단 통계량 + 아래 전체 테이블 구조로 변경
  - XoOL Calculator 모듈: 테이블 글자 크기 조정 (text-sm → text-xs, py-2 px-3 → py-1 px-2)
- `components/SplitDataPreviewModal.tsx` - StatsTable 컴포넌트 글자 크기 조정
  - 제목, 테이블, 셀 패딩 크기 축소

**Reason:**
- 사용자 요청에 따라 View Details 레이아웃을 더 직관적이고 일관된 구조로 개선
- 통계량과 데이터 테이블을 명확하게 구분하여 가독성 향상
- 글자 크기 축소로 더 많은 정보를 한 화면에 표시 가능

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 01:00:00

### feat(modules): Add Excel file support and direct input for Load Claim Data module

**Description:**
- Load Claim Data 모듈에 엑셀 파일(.xlsx, .xls) 지원 추가
  - SheetJS(xlsx) 라이브러리를 사용하여 엑셀 파일을 CSV 형식으로 변환
  - 파일 선택 시 CSV와 엑셀 파일 모두 선택 가능
  - 엑셀 파일의 첫 번째 시트를 자동으로 읽어서 처리
- 엑셀 데이터 직접 입력 기능 추가
  - "엑셀 데이터 직접 입력" 버튼 추가
  - 텍스트 영역에 엑셀에서 복사한 데이터(탭으로 구분)를 붙여넣기 가능
  - 붙여넣은 데이터를 CSV 형식으로 변환하여 처리
- 파일 타입 및 시트 정보 표시
  - 엑셀 파일 로드 시 시트 이름 표시
  - 파일 타입 정보를 파라미터에 저장

**Files Affected:**
- `package.json` - xlsx 라이브러리 의존성 추가
- `components/PropertiesPanel.tsx` - LoadClaimData 케이스에 엑셀 파일 지원 및 직접 입력 기능 추가
  - SheetJS import 추가
  - 엑셀 파일을 CSV로 변환하는 함수 추가
  - 파일 선택 시 엑셀 파일 처리 로직 추가
  - 직접 입력 모드 UI 및 처리 로직 추가
  - handleFileChange 함수에 LoadClaimData 엑셀 파일 처리 추가
  - file input의 accept 속성을 LoadClaimData에서 .csv,.xlsx,.xls로 확장

**Reason:**
- 사용자가 엑셀 파일을 직접 로드하거나 복사-붙여넣기로 데이터를 입력할 수 있도록 편의성 향상
- 기존 CSV 파일 로드 기능과 호환성 유지

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-16 00:00:00

### fix(modules): Update module names and fix Select Data module behavior

**Description:**
- Simulate Freq-Sev Table 모듈 이름 변경 반영
  - 에러 메시지에서 "Simulate Freq-Sev Table"로 명확히 표시
  - Properties Panel에서 모듈 이름 및 입력 타입 확인 로직 수정
- Select Data 모듈 동작 수정
  - 열을 선택하지 않으면 빈 결과 반환 (모든 열이 선택 해제된 경우)
  - 기본값이 모든 열을 선택한 상태로 설정
  - 열을 선택해도 원본 데이터 타입(number, string 등)이 그대로 유지되도록 수정

**Files Affected:**
- `App.tsx` - Select Data 모듈 로직 수정, Simulate Freq-Sev Table 에러 메시지 업데이트
- `components/PropertiesPanel.tsx` - Select Data 기본값 수정, Simulate Freq-Sev Table 이름 반영

**Reason:**
- Select Data 모듈의 예상치 못한 동작 수정
- 모듈 이름 변경에 따른 일관성 유지

**Commit Hash:** a7b6418

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-12-14 19:00:00

### fix(build): Add missing utility files to fix Vercel build error

**Description:**
- 누락된 유틸리티 파일 생성
  - `utils/fileOperations.ts`: `savePipeline`, `loadPipeline` 함수 구현
  - `utils/samples.ts`: `loadSampleFromFolder`, `loadFolderSamples` 함수 구현
- App.tsx의 import 경로 수정
  - `../shared/utils/fileOperations` → `./utils/fileOperations`
  - `../shared/utils/samples` → `./utils/samples`
- Vercel 빌드 오류 해결
  - "Could not resolve '../shared/utils/fileOperations'" 오류 수정

**Files Affected:**
- `utils/fileOperations.ts` - 새로 생성 (파이프라인 저장/로드 기능)
- `utils/samples.ts` - 새로 생성 (샘플 파일 로드 기능)
- `App.tsx` - import 경로 수정

**Reason:**
- Vercel 배포 실패 원인 해결
- 존재하지 않는 경로에서 파일을 import하던 문제 수정

**Commit Hash:** 0dc5664

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 0dc5664

# Or direct recovery
git reset --hard 0dc5664
```

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
