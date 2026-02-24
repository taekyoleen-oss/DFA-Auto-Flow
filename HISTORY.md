# Change History

## 2026-02-24 (현재 작업)

### feat(samples): Samples 데이터 Supabase 이관 (대분류 DFA)

**Description:**
- Life Matrix Flow와 동일한 방식으로 Samples를 Supabase로 이관
- 대분류(app_section): **DFA**
- Supabase 우선 로드, 없으면 서버 API → samples-list.json 폴백
- 단건 로드: sampleId(string) 시 Supabase에서 file_content 조회

**Files Affected:**
- `lib/supabase.ts` - Supabase 클라이언트 (신규)
- `utils/supabase-samples.ts` - Supabase Samples API, app_section=DFA (신규)
- `supabase/README.md` - 시드 안내 (신규)
- `scripts/seed-dfa-samples-to-supabase.mjs` - public/samples-list.json → Supabase 시드 (신규)
- `App.tsx` - loadFolderSamplesLocal: Supabase 우선, handleLoadSample: sampleId(string) 시 Supabase 단건 조회, 폴더 샘플 클릭 시 id 전달
- `package.json` - @supabase/supabase-js 추가

**Reason:**
- 사용자 요청: Samples를 Supabase로 이관, Life Matrix Flow와 동일한 방법 적용, 대분류 DFA

**Commit Hash:** (커밋 후 기록)

**Recovery Command:**
```bash
git stash push -u -m "백업"
git reset --hard <커밋해시>
# Or direct: git reset --hard <커밋해시>
```

---

## 2025-01-17 01:00:00

### feat(modules): Add AnalysisThreshold module and improve ThresholdAnalysis View Details with candidate threshold selection

**Description:**
- AnalysisThreshold 모듈 추가
  - 3개의 탭으로 구성: 데이터 분포, 경험적 분포, Mean Excess Plot
  - 데이터 분포 탭: 클레임 크기에 따라 건수를 히스토그램으로 표시
  - 경험적 분포 탭: Histogram, ECDF, QQ-Plot을 통해 "꼬리가 바뀌는 지점" 찾기
  - Mean Excess Plot 탭: e(u) 계산하여 일정 범위 이후 선형적으로 보이는 구간을 tail로 식별
- ThresholdAnalysis View Details 수정
  - View Details가 열리지 않던 문제 해결
  - DataPreviewModal 렌더링 조건에 ThresholdAnalysisOutput 타입 추가
  - thresholdAnalysisOutput이 있을 때 data가 없어도 모달이 열리도록 수정
- 각 탭에 기준점 선택 기능 추가
  - ECDF 탭: CDF가 완만해지거나 직선 형태로 변하는 지점을 자동 계산하여 표시
  - QQ-Plot 탭: 직선에서 벗어나 Tail이 두꺼워지는 지점을 자동 계산하여 표시
  - Mean Excess Plot 탭: 직선 형태가 시작하는 지점을 자동 계산하여 표시
  - 각 탭에서 기준점 버튼 클릭 또는 차트의 기준선 클릭으로 선택 가능
  - 선택된 기준점은 빨간색으로 강조 표시되고 설명 메시지 표시
- 프로젝트 이름 및 경로 변경
  - 프로젝트 이름: insure-auto-flow → dfa-auto-flow
  - 프로젝트 경로: C:\cursor\insure-auto-flow → C:\cursor\00 My Project\DFA-Auto-Flow
  - 모든 파일의 하드코딩된 경로 업데이트
- concurrently 패키지 추가
  - dev:full 스크립트로 여러 서버를 동시에 실행 가능
  - pnpm run dev:full로 split-data-server, samples-server, dev 서버 동시 실행

**Files Affected:**
- `types.ts` - AnalysisThresholdOutput 타입 추가, ThresholdAnalysisOutput에 candidateThresholds 필드 추가
- `constants.ts` - AnalysisThreshold 모듈 정의 추가
- `components/PropertiesPanel.tsx` - AnalysisThreshold 속성 패널 추가
- `components/AnalysisThresholdPreviewModal.tsx` - 새로 생성, AnalysisThreshold 전용 View Details 모달
- `components/DataPreviewModal.tsx` - ThresholdAnalysis View Details 수정, 각 차트 컴포넌트에 기준점 선택 기능 추가
- `App.tsx` - AnalysisThreshold 모듈 실행 로직 추가, ThresholdAnalysisOutput 처리 추가, AnalysisThresholdPreviewModal 렌더링 추가
- `codeSnippets.ts` - AnalysisThreshold Python 코드 추가, ThresholdAnalysis에 기준점 계산 로직 추가
- `package.json` - 프로젝트 이름 변경, concurrently 패키지 추가, dev:full 스크립트 추가
- `history` - 모든 경로 업데이트
- `data_analysis_modules.py` - 프로젝트 이름 업데이트
- `PYTHON_ANALYSIS_README.md` - 프로젝트 이름 업데이트

**Reason:**
- 사용자 요청에 따라 AnalysisThreshold 모듈 추가 및 ThresholdAnalysis View Details 개선
- 각 탭에서 기준점을 자동으로 계산하여 사용자가 쉽게 선택할 수 있도록 개선
- 프로젝트 이름 및 경로 변경에 따른 모든 참조 업데이트

**Commit Hash:** 5903132a189e1f38e7b5730cb93589d79079a599

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 5903132a189e1f38e7b5730cb93589d79079a599

# Or direct recovery
git reset --hard 5903132a189e1f38e7b5730cb93589d79079a599
```

## 2025-01-17 00:35:00

### refactor(ui): Restore terminal as tab in Properties Panel and add protection rules

**Description:**
- Properties Panel의 터미널을 탭으로 구성하도록 복원
  - Terminal 탭 추가: Properties, Preview, Code, Terminal (4개 탭)
  - 하단에 있던 별도 터미널 패널 제거
  - 터미널 리사이즈 기능 제거 (탭 내부에서 표시되므로 불필요)
  - 탭 아이콘 크기: 모든 탭 아이콘을 `w-5 h-5` (20px x 20px)로 유지
  - 탭 순서: Properties → Preview → Code → Terminal
  - Terminal 탭 아이콘: CommandLineIcon 사용
- Properties Panel 탭 구조 보호 규칙 추가
  - .cursorrules에 Properties Panel 탭 구조 보호 규칙 추가
  - 탭 구성, 아이콘 크기, 탭 순서 변경 금지 규칙 명시
  - 향후 앱 변경 시 다른 로직은 변경하지 못하도록 보호

**Files Affected:**
- `components/PropertiesPanel.tsx` - Terminal 탭 추가, 하단 터미널 패널 제거, 리사이즈 로직 제거
- `.cursorrules` - Properties Panel 탭 구조 보호 규칙 추가

**Reason:**
- 사용자 요청에 따라 터미널을 탭으로 구성하도록 복원
- 탭 아이콘 및 크기를 원래대로 유지
- 향후 변경 시 탭 구조가 보호되도록 Rules에 명시

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:34:00

### feat(ui): Sync Split By Threshold threshold parameter with Setting Threshold selection

**Description:**
- Setting Threshold 모듈의 View Details에서 "선택된 Threshold" 콤보박스 변경 시, 연결된 Split By Threshold 모듈의 threshold 속성도 동일한 값으로 자동 업데이트
  - Setting Threshold의 threshold_out 포트에 연결된 Split By Threshold 모듈 찾기
  - 연결된 모듈의 parameters.threshold 값을 Setting Threshold의 선택된 값으로 업데이트
  - ApplyThreshold 모듈도 동일하게 처리
  - onThresholdChange 콜백에서 한 번의 setModules 호출로 모든 업데이트 처리

**Files Affected:**
- `App.tsx` - SettingThresholdPreviewModal의 onThresholdChange 콜백에서 연결된 모듈의 threshold 파라미터 업데이트 로직 개선

**Reason:**
- 사용자 요청에 따라 Setting Threshold에서 선택한 값이 연결된 Split By Threshold 모듈의 속성에도 반영되도록 개선
- 사용자 편의성 향상 및 데이터 일관성 유지

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:33:00

### refactor(python): Redesign Split By Threshold output ports logic

**Description:**
- Split By Threshold 모듈의 출력 포트 로직 재설계
  - 첫 번째 출력 포트: Year Column을 기준으로 GroupBy하고 Amount Column의 합계 계산
    - `groupby(year_column)[amount_column].sum()` 수행
    - Year Column이 없으면 원본 데이터 반환
  - 두 번째 출력 포트: Threshold보다 크거나 같은 금액의 원본 데이터 반환
    - 원본 레이아웃 유지, 모든 컬럼 포함
  - 불필요한 datetime 변환 및 year 컬럼 추가 로직 제거
  - Year Column을 직접 사용하여 GroupBy 수행

**Files Affected:**
- `utils/pyodideRunner.ts` - splitByThresholdPython 함수의 로직 재설계
- `codeSnippets.ts` - SplitByThreshold 템플릿 코드 업데이트

**Reason:**
- 사용자 요청에 따라 출력 포트 계산 로직을 명확하게 재설계
- 첫 번째 출력은 Year Column 기준 집계, 두 번째 출력은 원본 데이터 유지
- 불필요한 데이터 변환 제거로 성능 및 명확성 향상

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:32:00

### fix(ui): Remove duplicate aggregation logic in Split By Threshold View Details

**Description:**
- Split By Threshold 모듈의 View Details에서 중복된 연도별 집계 로직 제거
  - Python 코드에서 이미 연도별 집계를 수행하므로 View Details에서는 원본 데이터를 그대로 표시
  - 첫 번째 탭은 첫 번째 출력 포트(belowThreshold)의 테이블을 그대로 표시
  - 두 번째 탭은 두 번째 출력 포트(aboveThreshold)의 테이블을 그대로 표시
  - Python의 splitByThresholdPython 함수가 year_column이 있을 때 연도별 집계를 수행하여 반환

**Files Affected:**
- `components/DataPreviewModal.tsx` - thresholdSplitData useMemo에서 연도별 집계 로직 제거, 원본 데이터 반환으로 변경

**Reason:**
- 사용자 요청에 따라 첫 번째 탭이 첫 번째 출력 포트의 테이블을 표시하도록 수정
- Python에서 이미 연도별 집계를 수행하므로 중복 처리 제거
- 출력 포트의 원본 데이터를 정확히 표시

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:31:00

### feat(ui): Add yearly claim aggregation for Split By Threshold below tab

**Description:**
- Split By Threshold 모듈의 View Details 첫 번째 탭("< Threshold")에 연도별 클레임 합계 기능 추가
  - 첫 번째 탭(below)에서 모듈 속성의 Year Column과 Amount Column을 사용하여 연도별 집계
  - Year Column을 기준으로 Amount Column의 금액을 합산하여 표시
  - 연도별로 정렬하여 표시
  - 기본 탭을 'below'로 변경하여 첫 번째 탭이 기본으로 선택되도록 함
  - 두 번째 탭(above)은 기존대로 원본 데이터 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - thresholdSplitData useMemo에 연도별 집계 로직 추가, 기본 탭을 'below'로 변경

**Reason:**
- 사용자 요청에 따라 첫 번째 탭에서 연도별 클레임 합계를 표시하여 데이터 분석 편의성 향상
- 모듈 속성에서 설정한 Year Column과 Amount Column을 사용하여 유연성 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:30:00

### feat(ui): Update Split By Threshold View Details tab names with threshold values

**Description:**
- Split By Threshold 모듈의 View Details 탭 이름을 threshold 값을 포함하도록 변경
  - 첫 번째 탭: "Below Threshold" → "< Threshold" (threshold 값 포함)
  - 두 번째 탭: "Above Threshold" → ">= Threshold" (threshold 값 포함)
  - threshold 값은 모듈의 parameters에서 가져와서 천단위 구분자와 함께 표시
  - 탭 순서 변경: 첫 번째 탭이 below, 두 번째 탭이 above

**Files Affected:**
- `components/DataPreviewModal.tsx` - SplitByThreshold 모듈의 탭 이름과 순서 변경

**Reason:**
- 사용자 요청에 따라 탭 이름에 threshold 값을 표시하여 더 명확하게 표시
- "< Threshold"와 ">= Threshold" 형식으로 threshold 기준을 명확히 표현

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:29:00

### refactor(ui): Change Split By Threshold Year Column to match Apply Inflation module

**Description:**
- Split By Threshold 모듈의 Year Column 속성을 Apply Inflation 모듈과 동일하게 변경
  - 파라미터 이름을 `date_column`에서 `year_column`으로 변경
  - 라벨을 "Year Column"으로 변경 (Apply Inflation과 동일)
  - 모든 컬럼을 옵션으로 제공 (날짜 필터링 제거, Apply Inflation과 동일)
  - 기본값을 "연도"로 설정 (Apply Inflation과 동일)
  - useEffect 자동 설정 로직을 year_column에 맞게 수정
  - App.tsx, pyodideRunner.ts, codeSnippets.ts에서 파라미터 이름 변경

**Files Affected:**
- `components/PropertiesPanel.tsx` - SplitByThreshold 케이스의 Year Column 구현을 Apply Inflation과 동일하게 변경, useEffect 수정
- `App.tsx` - date_column을 year_column으로 변경
- `utils/pyodideRunner.ts` - splitByThresholdPython 함수의 파라미터를 year_column으로 변경
- `codeSnippets.ts` - SplitByThreshold 템플릿의 파라미터를 year_column으로 변경

**Reason:**
- 사용자 요청에 따라 Apply Inflation 모듈의 Year Column과 동일한 방식으로 구현
- 일관성 있는 UI/UX 제공
- 연도 컬럼 선택을 더 직관적으로 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:28:00

### style(ui): Rename Date Column label to "Year" in Split By Threshold module

**Description:**
- Split By Threshold 모듈의 Date Column(연도) 라벨을 "Year"로 변경
  - PropertiesPanel.tsx에서 라벨을 "Date Column (연도)"에서 "Year"로 변경
  - 앞의 모듈(왼쪽으로 연결된 모듈)에서 열을 불러오는 기능은 이미 구현되어 있음
  - getConnectedDataSource 함수를 통해 연결된 모듈의 출력 데이터에서 컬럼 정보를 가져옴

**Files Affected:**
- `components/PropertiesPanel.tsx` - SplitByThreshold 케이스의 Date Column 라벨을 "Year"로 변경

**Reason:**
- 사용자 요청에 따라 라벨을 더 간결하고 명확하게 변경
- "Year"라는 이름이 연도 컬럼 선택의 목적을 더 명확하게 표현

**Commit Hash:** afebf692820ca761b7921e866c3af4f32b3f7be2

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2

# Or direct recovery
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2
```

## 2025-01-17 00:27:00

### feat(ui): Add auto-select for Date Column in Split By Threshold module

**Description:**
- Split By Threshold 모듈의 Date Column 속성에 자동 선택 기능 추가
  - 입력 데이터에서 날짜 컬럼을 자동으로 감지하여 콤보박스에 표시
  - 가장 유사한 컬럼을 기본값으로 자동 설정
  - 우선순위: "연도" > "year" > "날짜" > "date" > 첫 번째 날짜 컬럼
  - useEffect를 사용하여 date_column이 없을 때 자동으로 설정
  - 사용자가 명시적으로 선택하지 않은 경우에만 자동 설정

**Files Affected:**
- `components/PropertiesPanel.tsx` - SplitByThreshold 케이스에 자동 선택 로직 추가, useEffect로 자동 설정

**Reason:**
- 사용자 요청에 따라 Date Column 선택을 자동화하여 사용 편의성 향상
- 가장 유사한 컬럼을 자동으로 찾아 기본값으로 설정

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:26:00

### feat(ui): Add Date Column parameter to Split By Threshold module

**Description:**
- Split By Threshold 모듈의 속성에 연도(날짜) 컬럼 선택 옵션 추가
  - PropertiesPanel.tsx에 Date Column 선택 UI 추가
  - FormatChange 모듈과 동일한 날짜 컬럼 필터링 로직 적용
  - App.tsx에서 SplitByThreshold 실행 시 date_column 파라미터 전달
  - splitByThresholdPython 함수는 이미 date_column 파라미터를 지원하므로 추가 수정 불필요
  - date_column이 있으면 연도별 합계로 처리, 없으면 원본 레이아웃 유지

**Files Affected:**
- `components/PropertiesPanel.tsx` - SplitByThreshold 케이스에 Date Column 선택 UI 추가
- `App.tsx` - SplitByThreshold 실행 시 date_column 파라미터 전달

**Reason:**
- 사용자 요청에 따라 Split By Threshold 모듈에 연도 컬럼 선택 기능 추가
- 연도별 집계 기능을 활용할 수 있도록 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:25:00

### fix(ui): Restore Split By Threshold View Details

**Description:**
- Split By Threshold 모듈의 View Details 복원
  - PropertiesPanel.tsx에 ThresholdSplitOutput을 visualizableTypes에 추가하여 View Details 버튼 표시
  - DataPreviewModal.tsx에 ThresholdSplitOutput 처리 로직 추가
  - 탭 구성 추가: "Above Threshold"와 "Below Threshold" 탭으로 데이터 분리 표시
  - 각 탭에서 해당 데이터(aboveThreshold/belowThreshold)를 테이블로 표시
  - 숫자 형식 천단위 구분 및 오른쪽 정렬 적용 (연도 컬럼 제외)
  - 통계량 표시 기능 추가

**Files Affected:**
- `components/PropertiesPanel.tsx` - ThresholdSplitOutput을 visualizableTypes에 추가
- `components/DataPreviewModal.tsx` - ThresholdSplitOutput 처리 로직 및 탭 UI 추가

**Reason:**
- Split By Threshold 모듈의 View Details가 보이지 않는 문제 해결
- 기존 커밋된 내용을 참고하여 복원

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:24:00

### chore(backup): Update XoL Calculator View Details backup to latest version

**Description:**
- XoL Calculator의 View Details를 PreviewBackup 폴더에 최신 내용으로 업데이트
  - 기존 백업 파일을 최신 내용으로 교체
  - "건별 XoL 적용" 탭: 테이블(10행, 가로 스크롤), 통계량, 히스토그램(보험금, XoL Claim(Incl. Limit))
  - "연도별 XoL 적용" 탭: 테이블(text-sm), 통계 정보, 통계량 + 그래프
  - YearlyAmountBarPlot 컴포넌트, getXolData 함수, 히스토그램 렌더링 함수 포함
  - README.md에 상세 내용 추가

**Files Affected:**
- `PreviewBackup/XoLCalculator_ViewDetails_Backup.tsx` - 최신 내용으로 업데이트
- `PreviewBackup/README.md` - XoL Calculator 백업 파일 상세 내용 추가

**Reason:**
- 사용자 요청에 따라 XoL Calculator의 View Details를 최신 상태로 백업하여 복구 가능하도록 함

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:23:00

### style(ui): Increase font size in XoL Calculator "연도별 XoL 적용" tab table

**Description:**
- XoL Calculator의 "연도별 XoL 적용" 탭 테이블 폰트 크기 증가
  - 테이블 폰트 크기를 `text-xs`에서 `text-sm`으로 변경 (한 단계 증가)
  - 가독성 향상을 위해 폰트 크기 조정

**Files Affected:**
- `components/DataPreviewModal.tsx` - "연도별 XoL 적용" 탭 테이블 폰트 크기 변경

**Reason:**
- 사용자 요청에 따라 테이블의 폰트 크기를 한 단계 올려 가독성을 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:22:00

### feat(ui): Add horizontal scroll to XoL Calculator "건별 XoL 적용" tab table

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭 테이블에 가로 스크롤 추가
  - 테이블 컨테이너에 `overflow-x-auto` 추가하여 가로 스크롤 가능하도록 수정
  - `overflow-hidden`을 `overflow-x-auto overflow-y-hidden`으로 변경
  - 테이블에 `minWidth: 'max-content'` 스타일 추가하여 컬럼이 많을 때 가로 스크롤 활성화
  - 세로 스크롤은 제거하고 가로 스크롤만 활성화

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 테이블에 가로 스크롤 추가

**Reason:**
- 사용자 요청에 따라 테이블에 가로 스크롤을 추가하여 컬럼이 많을 때도 모든 데이터를 볼 수 있도록 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:21:00

### feat(ui): Reorder layout in XoL Calculator "건별 XoL 적용" tab

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭 레이아웃 순서 변경
  - 테이블을 가장 위로 이동
  - 통계량을 테이블 아래로 이동
  - 기존에는 통계량이 위, 테이블이 아래였으나 순서 변경
  - 히스토그램은 여전히 가장 아래에 위치

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 레이아웃 순서 변경

**Reason:**
- 사용자 요청에 따라 테이블을 가장 위에 배치하고 통계량을 그 아래에 배치하여 정보를 더 명확하게 표시

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:20:00

### fix(ui): Fix React Hooks rule violation in XoL Calculator histogram

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭에서 React Hooks 규칙 위반 에러 수정
  - 조건부 렌더링 안에서 `useMemo`를 사용하던 것을 일반 변수 계산으로 변경
  - "Rendered fewer hooks than expected" 에러 해결
  - 보험금과 XoL Claim(Incl. Limit) 히스토그램 데이터 계산을 `useMemo` 대신 일반 변수로 변경
  - React Hooks 규칙 준수: hooks는 항상 같은 순서로 호출되어야 함

**Files Affected:**
- `components/DataPreviewModal.tsx` - 조건부 렌더링 안에서 useMemo 제거, 일반 변수 계산으로 변경

**Reason:**
- React Hooks 규칙 위반으로 인한 에러를 해결하여 애플리케이션이 정상적으로 작동하도록 수정

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:19:00

### fix(ui): Fix "연도별 XoL 적용" tab not showing data

**Description:**
- XoL Calculator의 "연도별 XoL 적용" 탭에서 데이터가 없을 때 안내 메시지 표시 추가
  - 연도 컬럼이 없거나 데이터가 없을 때 명확한 안내 메시지 표시
  - xolData.rows가 없거나 columns가 없을 때 조건부 렌더링 추가
  - 사용자가 왜 데이터가 보이지 않는지 이해할 수 있도록 개선

**Files Affected:**
- `components/DataPreviewModal.tsx` - "연도별 XoL 적용" 탭에 데이터 없을 때 안내 메시지 추가

**Reason:**
- 사용자 요청에 따라 "연도별 XoL 적용" 탭이 보이지 않는 문제를 해결하고, 데이터가 없을 때 명확한 안내를 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:18:00

### fix(ui): Fix histogram display in XoL Calculator "건별 XoL 적용" tab

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭에서 히스토그램이 표시되지 않는 문제 수정
  - Setting Threshold의 Graph 탭 방식을 참조하여 SVG 기반 히스토그램 구현
  - 보험금(claim_column)과 XoL Claim(Incl. Limit) 히스토그램을 각각 useMemo로 계산
  - 30개 bins 사용, Setting Threshold와 동일한 스타일 적용
  - X축 눈금 6개 표시, Y축 눈금 5개 표시
  - SVG로 직접 렌더링하여 히스토그램 바, 축, 라벨 표시
  - 각 히스토그램에 제목과 데이터 없음 메시지 추가

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 히스토그램 구현 수정

**Reason:**
- 사용자 요청에 따라 Setting Threshold의 Graph 탭을 참조하여 히스토그램이 정상적으로 표시되도록 수정

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:17:00

### feat(ui): Redesign XoL Calculator "건별 XoL 적용" tab layout

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭 레이아웃 재구성
  - 테이블을 10개의 행만 보이도록 제한 (slice(0, 10))
  - 테이블이 가로 전체를 차지하도록 수정 (기존에는 통계량과 나란히 배치)
  - 테이블 상단에 선택된 컬럼의 통계량 표시 (ColumnStatistics)
  - 맨 하단에 속성에서 정한 대상(claim_column)에 대한 보험금과 XoL Claim(Incl. Limit)를 히스토그램으로 각각 상하로 표시
  - 히스토그램은 HistogramPlot 컴포넌트 사용
  - 테이블 하단에 전체 행 수 표시 (Showing 10 of X rows)

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 레이아웃 재구성, 히스토그램 추가

**Reason:**
- 사용자 요청에 따라 테이블을 10행으로 제한하고 가로 전체를 차지하도록 하며, 상단에 통계량, 하단에 히스토그램을 배치하여 정보를 더 명확하게 표시

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:16:00

### feat(ui): Increase table height and enable full scroll in XoL Calculator "건별 XoL 적용" tab

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭에서 테이블 높이 증가 및 전체 스크롤 개선
  - 테이블과 통계량 영역에 최소 높이 설정 (min-h-[600px])으로 테이블 높이 증가
  - 테이블 영역에 overflow-y-auto 적용하여 테이블 내부 스크롤 가능
  - overflow-hidden 제거하여 전체 컨테이너의 세로 스크롤이 작동하도록 수정
  - 전체 스크롤을 통해 아래의 그래프(Plot)를 볼 수 있도록 개선
  - 테이블 영역에 border 추가로 영역 구분 명확화

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 테이블 높이 증가 및 스크롤 구조 개선

**Reason:**
- 사용자 요청에 따라 테이블의 위아래 길이를 늘리고, 전체적인 세로 스크롤을 통해 아래 그래프를 볼 수 있도록 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:15:00

### feat(ui): Improve XoL Calculator View Details layout for "건별 XoL 적용" tab

**Description:**
- XoL Calculator의 "건별 XoL 적용" 탭 레이아웃 개선
  - 왼쪽에 테이블 배치, 오른쪽에 통계량 표시
  - 테이블 내에서 연도를 제외한 숫자 형식은 천단위 구분 적용 및 오른쪽 정렬
  - Plot는 테이블과 통계량 아래에 표시 (기존에는 오른쪽에 표시)
  - View Details에 스크롤 추가 (overflow-y-auto)
  - 테이블 헤더도 숫자 컬럼은 오른쪽 정렬
  - 통계량 영역에 배경색(gray-50) 적용

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator "건별 XoL 적용" 탭 레이아웃 수정, 숫자 형식 처리 개선, 스크롤 추가

**Reason:**
- 사용자 요청에 따라 XoL Calculator의 View Details 레이아웃을 개선하여 테이블과 통계량을 더 명확하게 구분하고, Plot를 아래에 배치하여 가독성 향상

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:14:00

### fix(ui): Fix XoL Contract View Details not opening

**Description:**
- XoL Contract 모듈의 View Details가 열리지 않는 문제 수정
  - `PropertiesPanel.tsx`의 `visualizableTypes` 배열에 `XolContractOutput` 추가
  - `canVisualize()` 함수에 `DefineXolContract` 모듈 타입 체크 추가
  - `App.tsx`에서 `DataPreviewModal` 렌더링 조건에 `XolContractOutput` 및 `ModuleType.DefineXolContract` 추가
  - 이제 XoL Contract 모듈에서 View Details 버튼이 정상적으로 표시되고 모달이 열림

**Files Affected:**
- `components/PropertiesPanel.tsx` - visualizableTypes에 XolContractOutput 추가, canVisualize 함수에 DefineXolContract 체크 추가
- `App.tsx` - DataPreviewModal 렌더링 조건에 XolContractOutput 및 DefineXolContract 모듈 타입 추가

**Reason:**
- 사용자 요청에 따라 XoL Contract의 View Details가 정상적으로 열리도록 수정

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:13:00

### feat(ui): Enhance XoL Contract View Details with comprehensive input data summary

**Description:**
- XoL Contract 모듈의 View Details에 입력 데이터를 XoL에 맞게 상세 요약 표시 기능 추가 및 개선
  - XoL Contract 파라미터 표시: Deductible, Limit, Reinstatements, Aggregate Deductible, Expense Ratio, Reinstatement Premiums
  - 기본 통계 (Basic Statistics):
    - Total Claims, Total Amount, Mean Amount, Median Amount, Std Dev, Min Amount, Max Amount
  - XoL 적용 결과 (XoL Application Results):
    - Total Retention / Mean Retention (Deductible 이하 금액)
    - Total Excess / Mean Excess (Deductible 초과 금액)
    - Total XoL Claim / Mean XoL Claim (XoL 적용 금액)
    - XoL Ratio (XoL 적용 비율)
  - XoL 적용 건수 통계 (XoL Application Counts):
    - Claims Exceeding Deductible (Deductible 초과 건수 및 비율)
    - Claims Hitting Limit (Limit 도달 건수 및 비율)
    - Claims Fully Covered (XoL 적용 건수 및 비율)
    - Claims Below Deductible (Deductible 미만 건수 및 비율)
  - 연도별 통계 테이블 (Yearly Statistics):
    - Year, Count, Total Amount, Mean Amount
    - Total Retention, Total Excess, Total XoL Claim, XoL Ratio
  - 입력 데이터가 연결되지 않은 경우 안내 메시지 표시
  - 입력 데이터에서 클레임 금액 컬럼과 연도 컬럼 자동 감지
  - XoL 적용 계산: Deductible과 Limit을 적용한 Retention, Excess, XoL Claim 계산

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Contract 전용 View Details 추가 및 개선, 상세 통계 계산 로직 추가
- `App.tsx` - XolContractOutput 타입일 때 DataPreviewModal 열도록 수정

**Reason:**
- 사용자 요청에 따라 XoL Contract의 입력 데이터를 XoL에 맞게 더 상세하게 요약하여 표시하여 계약 조건과 데이터의 관계를 명확히 파악할 수 있도록 함
- Retention, Excess, XoL Claim을 구분하여 표시하여 XoL 적용 결과를 더 명확하게 이해할 수 있도록 함

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:12:00

### style(ui): Remove thousand separators from year columns in View Details

**Description:**
- View Details에서 연도 컬럼에 천단위 구분 제거
  - 연도 컬럼 식별: 컬럼 이름이 '연도', 'year', 'Year' 또는 소문자 'year'인 경우
  - Load Claim Data, Format Change, Apply Inflation, Select Data, Apply Threshold의 Detail 탭 테이블에서 연도는 천단위 구분 없이 표시
  - Split By Freq-Sve의 Frequency/Severity 탭에서 연도는 천단위 구분 없이 표시
  - 연도 컬럼은 숫자형이어도 천단위 구분을 적용하지 않음

**Files Affected:**
- `components/DataPreviewModal.tsx` - Load Claim Data 등의 Detail 탭에서 연도 컬럼 처리 추가
- `components/SplitFreqServPreviewModal.tsx` - 연도 표시 확인 (이미 천단위 구분 없이 표시됨)

**Reason:**
- 사용자 요청에 따라 연도는 천단위 구분 없이 표시하여 가독성 향상

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:11:00

### feat(ui): Split By Freq-Sve View Details into Frequency and Severity tabs

**Description:**
- Split By Freq-Sve 모듈의 View Details를 두 개의 탭으로 분리
  - 첫 번째 탭: "Frequency" - 빈도 모델 표시 (Yearly Frequency 데이터 및 통계량)
  - 두 번째 탭: "Severity" - 심도 모델 표시 (Yearly Severity 데이터 및 Amount 데이터)
- Download CSV 버튼을 아이콘만으로 표시
  - 텍스트 제거하고 ArrowDownTrayIcon만 표시
  - hover 효과 및 title 속성 추가
  - 스타일을 아이콘 버튼 형태로 변경 (p-1.5, rounded, hover:bg-gray-100)

**Files Affected:**
- `components/SplitFreqServPreviewModal.tsx` - 탭 구조 추가, Download CSV 버튼 스타일 변경

**Reason:**
- 사용자 요청에 따라 빈도 모델과 심도 모델을 탭으로 분리하여 더 명확한 UI 제공
- Download CSV 버튼을 아이콘만으로 표시하여 공간 절약 및 깔끔한 UI

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:10:00

### feat(ui): Apply Load Claim Data View Details to Apply Threshold module

**Description:**
- Apply Threshold 모듈에 Load Claim Data와 동일한 View Details 적용
  - Detail 탭: 테이블 (숫자 천단위 구분, 우측 정렬) + 통계량 (테두리 없음, 옅은 회색 배경)
  - Graphs 탭: 히스토그램 (Setting Threshold 스타일)
  - activeLoadClaimDataTab 상태 관리
  - graphColumn 상태 관리
  - 히스토그램 데이터 계산 useMemo
  - 기존 View Details를 Load Claim Data 스타일로 대체

**Files Affected:**
- `components/DataPreviewModal.tsx` - Apply Threshold에 Load Claim Data와 동일한 View Details 적용

**Reason:**
- 사용자 요청에 따라 Apply Threshold 모듈에도 Load Claim Data와 동일한 View Details를 적용하여 일관성 있는 UI 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:09:00

### chore(backup): Backup Setting Threshold View Details to PreviewBackup folder

**Description:**
- Setting Threshold의 View Details를 PreviewBackup 폴더에 백업
  - `PreviewBackup/SettingThreshold_ViewDetails_Backup.tsx` 파일 생성
  - SettingThresholdPreviewModal.tsx의 전체 내용 백업
  - HistogramChart 컴포넌트 포함
  - 분석 탭 및 분포 탭 전체 구현 백업
  - Spread View 및 CSV 다운로드 기능 포함

**Files Affected:**
- `PreviewBackup/SettingThreshold_ViewDetails_Backup.tsx` - Setting Threshold View Details 백업 파일 생성

**Reason:**
- 사용자 요청에 따라 Setting Threshold의 View Details를 백업하여 복구 가능하도록 보존

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:08:00

### feat(ui): Apply Load Claim Data View Details to Format Change, Apply Inflation, and Select Data modules

**Description:**
- Load Claim Data의 View Details를 PreviewBackup에 백업
  - `PreviewBackup/LoadClaimData_ViewDetails_Backup.tsx` 파일 생성
  - Detail 탭과 Graphs 탭의 전체 구현 백업
- Format Change, Apply Inflation, Select Data 모듈에 Load Claim Data와 동일한 View Details 적용
  - Detail 탭: 테이블 (숫자 천단위 구분, 우측 정렬) + 통계량 (테두리 없음, 옅은 회색 배경)
  - Graphs 탭: 히스토그램 (Setting Threshold 스타일)
  - activeLoadClaimDataTab 상태 관리
  - graphColumn 상태 관리
  - 히스토그램 데이터 계산 useMemo
  - 기존 View Details를 Load Claim Data 스타일로 대체

**Files Affected:**
- `PreviewBackup/LoadClaimData_ViewDetails_Backup.tsx` - Load Claim Data View Details 백업 파일 생성
- `components/DataPreviewModal.tsx` - Format Change, Apply Inflation, Select Data에 Load Claim Data와 동일한 View Details 적용

**Reason:**
- 사용자 요청에 따라 Load Claim Data의 View Details를 다른 모듈에도 동일하게 적용하여 일관성 있는 UI 제공

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:07:00

### style(ui): Format numbers with thousand separators and right-align in Load Claim Data Detail tab table

**Description:**
- Load Claim Data의 View Details의 Detail 탭 테이블에서 숫자 포맷팅 및 정렬 개선
  - 숫자형 컬럼의 값에 천단위 구분 적용 (toLocaleString 사용, ko-KR 로케일)
  - 숫자형 컬럼을 우측 정렬 (text-right 클래스 추가)
  - 숫자 포맷: 최대 소수점 2자리, 최소 소수점 0자리
  - 숫자가 아닌 값은 기존대로 왼쪽 정렬 유지

**Files Affected:**
- `components/DataPreviewModal.tsx` - Load Claim Data Detail 탭 테이블 셀 렌더링 로직 수정

**Reason:**
- 사용자 요청에 따라 숫자 가독성 향상 및 정렬 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:06:00

### style(ui): Remove border and add light gray background to Load Claim Data Detail tab statistics

**Description:**
- Load Claim Data의 View Details의 Detail 탭에서 통계량 부분 스타일 수정
  - 통계량 부분의 테두리 제거 (ColumnStatistics 컴포넌트에 noBorder prop 추가)
  - 통계량 부분의 배경을 옅은 회색(bg-gray-50)으로 변경
  - ColumnStatistics 컴포넌트에 스타일 커스터마이징을 위한 optional props 추가 (noBorder, backgroundColor)

**Files Affected:**
- `components/DataPreviewModal.tsx` - ColumnStatistics 컴포넌트에 스타일 prop 추가, Load Claim Data Detail 탭 통계량 부분 스타일 수정

**Reason:**
- 사용자 요청에 따라 통계량 부분의 테두리 제거 및 배경색 변경

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:05:00

### style(ui): Remove border from Load Claim Data View Details and add X-axis labels to Graphs histogram

**Description:**
- Load Claim Data의 View Details에서 테두리 제거
  - Detail 탭의 테이블 테두리 제거 (border border-gray-200 rounded-lg 클래스 제거)
- Graphs 탭의 히스토그램에 가로축 숫자 표시 추가
  - X축에 6개의 숫자를 넓은 간격으로 표시
  - 최소값부터 최대값까지 균등하게 분배
  - tick 라인과 숫자 라벨 추가
  - 숫자 포맷: toLocaleString 사용, 최대 소수점 2자리

**Files Affected:**
- `components/DataPreviewModal.tsx` - Load Claim Data Detail 탭 테두리 제거, Graphs 탭 X축 라벨 추가

**Reason:**
- 사용자 요청에 따라 테두리 제거 및 가로축 숫자 표시 개선

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:04:00

### feat(ui): Update Load Claim Data Graphs tab to match Setting Threshold style

**Description:**
- Load Claim Data의 View Details의 Graphs 탭을 Setting Threshold의 Graphs 탭과 동일한 형태로 수정
  - Setting Threshold의 HistogramChart 스타일 적용
  - SVG 기반 히스토그램 차트로 변경 (800x400 크기)
  - Y축 라벨 (0, 0.25, 0.5, 0.75, 1.0 비율)
  - X축 라벨: "값 (Value)", Y축 라벨: "빈도 (Frequency)"
  - 히스토그램 바 색상: #3b82f6, opacity 0.7
  - 30개 bins 사용
  - 제목: "데이터 분포 히스토그램"
  - 동일한 패딩 및 스타일 적용

**Files Affected:**
- `components/DataPreviewModal.tsx` - Load Claim Data Graphs 탭 히스토그램 차트 스타일 수정

**Reason:**
- 사용자 요청에 따라 Load Claim Data의 Graphs 탭을 Setting Threshold와 동일한 형태로 통일

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:03:00

### feat(ui): Modify Load Claim Data View Details with Detail and Graphs tabs

**Description:**
- Load Claim Data 모듈의 View Details를 탭 구조로 변경
  - "Detail" 탭 추가:
    - 왼쪽: 현재 테이블 표시
    - 오른쪽: 선택된 열의 통계량 표시 (ColumnStatistics)
    - 그래프 제거 (히스토그램 제거)
  - "Graphs" 탭 추가:
    - 콤보박스로 숫자형 열 선택
    - 선택된 열에 대한 히스토그램 표시 (HistogramPlot)
- Load Claim Data 모듈 전용 탭 상태 관리 추가 (activeLoadClaimDataTab, graphColumn)

**Files Affected:**
- `components/DataPreviewModal.tsx` - Load Claim Data 모듈 전용 탭 및 UI 로직 추가

**Reason:**
- 사용자 요청에 따라 Load Claim Data의 View Details를 Detail/Graphs 탭으로 분리하여 테이블과 그래프를 별도로 관리

**Commit Hash:** (will be updated after commit)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2025-01-17 00:02:00

### fix(toolbox): Update Toolbox structure to match yesterday evening commit (8d4f7a5)

**Description:**
- Toolbox의 카테고리 구조를 어제 저녁 커밋(8d4f7a5)에 맞게 수정
  - 카테고리 구조 변경:
    - "Data Preprocess" (preprocessTypes에 LoadClaimData, SettingThreshold 추가)
    - "DFA Analysis" (dfaTypes 사용)
    - "XoL Analysis" (xolPricingTypes 사용, 이름 변경: "XoL Pricing" → "XoL Analysis")
  - 제거된 카테고리: "Data Analysis", "Tradition Analysis", "Reinsurance Analysis"
  - preprocessTypes 업데이트: LoadClaimData, SettingThreshold 추가, TransformData, TransitionData 제거

**Files Affected:**
- `components/Toolbox.tsx` - categorizedModules 구조 및 preprocessTypes 수정

**Reason:**
- 어제 저녁 커밋(8d4f7a5)에 맞게 Toolbox 구조 업데이트

**Commit Hash:** afebf692820ca761b7921e866c3af4f32b3f7be2

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2

# Or direct recovery
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2
```

## 2025-01-17 00:01:00

### fix(toolbox): Update Toolbox module list to match recent commits

**Description:**
- Toolbox의 XoL Pricing 카테고리 모듈 목록 수정
  - 제거: XoL Loading, Calculate Ceded Loss, Price XoL Contract
  - 추가: XoL Calculator, XoL Pricing
  - 카테고리 이름 변경: "XoL Reinsurance Pricing" → "XoL Pricing"

**Files Affected:**
- `components/Toolbox.tsx` - xolPricingTypes 배열 및 카테고리 이름 수정

**Reason:**
- 최근 커밋(c959f76)에 맞게 Toolbox 모듈 목록 업데이트

**Commit Hash:** afebf692820ca761b7921e866c3af4f32b3f7be2

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2

# Or direct recovery
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2
```

## 2025-01-17 00:00:00

### fix(ui): Remove PCA Visualization View Details and restore XoL Calculator View Details with tabs

**Description:**
- PCA Visualization의 View Details 제거
  - PropertiesPanel.tsx에서 PCAOutput을 visualizableTypes에서 제거하여 PCA 모듈의 View Details 버튼 비활성화
  - DataPreviewModal.tsx에서 PCA Visualization 탭 제거
- XoL Calculator의 View Details를 과거 버전으로 복원
  - "건별 XoL 적용" 및 "연도별 XoL 적용" 탭 추가
  - 연도별 집계 기능 복원 (연도별 XoL Claim 및 XoL Premium Rate 계산)
  - 통계 정보 섹션 추가: XoL Claim 평균, 표준편차, XoL Premium Rate 평균, Reluctance Factor
  - 연도별 XoL Claim(Incl. Agg/Reinst) 그래프 표시
  - YearlyAmountBarPlot 컴포넌트 추가

**Files Affected:**
- `components/PropertiesPanel.tsx` - PCAOutput을 visualizableTypes에서 제거
- `components/DataPreviewModal.tsx` - PCA Visualization 탭 제거, XoL Calculator View Details 복원 (건별/연도별 탭)
- `App.tsx` - DataPreviewModal 호출 시 allModules와 allConnections 전달

**Reason:**
- 사용자 요청에 따라 PCA Visualization의 View Details 제거
- XoL Calculator의 View Details를 과거 버전(건별 XoL 및 연도별 XoL 탭)으로 복원

**Commit Hash:** afebf692820ca761b7921e866c3af4f32b3f7be2

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2

# Or direct recovery
git reset --hard afebf692820ca761b7921e866c3af4f32b3f7be2
```

## 2025-12-19 07:27:15

### feat(ui): Update Save button icon and XoL Loss Model sample

**Description:**
- Save 버튼 아이콘 변경
  - Save 버튼의 아이콘을 CodeBracketIcon에서 ArrowDownTrayIcon으로 변경하여 저장 기능에 맞게 수정
- XoL Loss Model 샘플 업데이트
  - ApplyThreshold 모듈 추가 (임계값 1,000,000 적용)
  - 모듈 위치 및 파라미터 업데이트
  - 연결 관계 업데이트 (ApplyThreshold가 ApplyInflation과 SplitByFreqServ 사이에 위치)
  - 파일 내용에 포함된 클레임 데이터 사용

**Files Affected:**
- `App.tsx` - Save 버튼 아이콘 변경 (ArrowDownTrayIcon import 및 사용)
- `savedSamples.ts` - XoL Loss Model 샘플 업데이트

**Reason:**
- 사용자 요청에 따라 Save 버튼 아이콘을 저장 기능에 맞게 수정
- XoL Loss Model 샘플을 최신 내용으로 업데이트

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 09:00:00

### feat(xol): Add XoL Pricing module and improve XoL Calculator UI

**Description:**
- XoL Pricing 모듈 추가
  - XoL Pricing 카테고리에 새로운 모듈 추가
  - XoL Calculator의 출력 데이터를 입력으로 받아 Net Premium 및 Gross Premium 계산
  - Expense Rate 파라미터 지원 (기본값: 0.2)
  - View Details에 입력값, 계산 수식, 계산 결과 표시
- XoL Calculator View Details 개선
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
- `components/DataPreviewModal.tsx` - XoL Calculator View Details 레이아웃 개선, XoL Premium Rate 계산 추가
- `components/XolPricingPreviewModal.tsx` - 새로 생성, View Details 모달 구현

**Reason:**
- 사용자 요청에 따라 XoL Pricing 모듈 추가 및 XoL Calculator UI 개선
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

### feat(ui): Add horizontal scroll to XoL Calculator table and change histogram to bar chart

**Description:**
- XoL Calculator 모듈의 View Details 테이블에 좌우 스크롤 추가
  - 테이블 컨테이너에 `overflow-x-auto` 추가하여 좌우 스크롤 가능
  - 테이블에 `minWidth: 'max-content'` 스타일 추가하여 컬럼이 많을 때 스크롤 활성화
- XoL Claim(Incl. Agg/Reinst) 탭의 히스토그램을 연도별 막대그래프로 변경
  - 기존: 선택된 열의 히스토그램 표시
  - 변경: 연도를 가로축, 숫자형 열(클레임 금액_infl, XoL Claim(Incl. Limit), XoL Claim(Incl. Agg/Reinst))을 세로축으로 하는 막대그래프 표시
  - 각 숫자형 열마다 별도의 막대그래프 표시
  - YearlyAmountBarPlot 컴포넌트를 사용하여 연도별 값 표시
  - XoL Claim(Incl. Limit) 탭은 기존 히스토그램 유지

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator 테이블 스크롤 및 그래프 변경
  - 테이블 컨테이너에 overflow-x-auto 추가
  - 테이블에 minWidth 스타일 추가
  - XoL Claim(Incl. Agg/Reinst) 탭의 히스토그램을 연도별 막대그래프로 변경

**Reason:**
- 사용자 요청에 따라 테이블의 좌우 스크롤 기능 추가로 많은 컬럼을 가진 테이블도 편리하게 탐색 가능
- 연도별 집계 데이터를 막대그래프로 시각화하여 트렌드를 더 쉽게 파악 가능

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 07:00:00

### feat(ui): Add XoL Claim(Incl. Agg/Reinst) column to XoL Calculator second tab

**Description:**
- XoL Calculator 모듈의 두 번째 탭(XoL Claim(Incl. Agg/Reinst))에 새로운 열 추가
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 06:30:00

### feat(ui): Add year aggregation for XoL Calculator second tab with specific columns

**Description:**
- XoL Calculator 모듈의 두 번째 탭(XoL Claim(Incl. Agg/Reinst))에 연도별 집계 기능 추가
  - 두 번째 탭 선택 시 기존 테이블을 연도별로 집계하여 표시
  - 테이블 열 구성: 연도, 클레임 금액_infl, XoL Claim(Incl. Limit)
  - 연도 컬럼(연도, year 등)을 자동으로 찾아서 연도별로 집계
  - 클레임 금액_infl과 XoL Claim(Incl. Limit) 컬럼을 연도별로 합계 계산
  - 집계된 데이터는 연도, 클레임 금액_infl(합계), XoL Claim(Incl. Limit)(합계)만 포함
  - 첫 번째 탭과 동일한 레이아웃 유지 (왼쪽 테이블, 오른쪽 통계량, 아래 히스토그램)
  - 연도 컬럼이 없는 경우 빈 데이터 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator 모듈의 getXolData 함수 수정
  - 두 번째 탭(aggreinst) 선택 시 연도별 집계 로직 추가
  - 연도별 합계 계산 및 새로운 데이터 구조 생성

**Reason:**
- 사용자 요청에 따라 XoL Calculator 모듈의 두 번째 탭에서 연도별 집계 데이터를 확인할 수 있도록 개선
- 연도별 트렌드를 쉽게 파악할 수 있도록 집계 기능 제공

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 05:30:00

### fix(ui): Remove top statistics table from XoL Calculator View Details

**Description:**
- XoL Calculator 모듈의 View Details에서 상단 통계량 테이블 제거
  - 기존: 상단에 StatsTable 표시
  - 변경: 상단 통계량 테이블 제거, 테이블과 통계량/히스토그램만 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator 모듈의 상단 StatsTable 제거

**Reason:**
- 사용자 요청에 따라 XoL Calculator 모듈의 View Details에서 상단 통계량 테이블 제거
- 선택된 열의 통계량과 히스토그램만 표시하여 더 간결한 레이아웃 제공

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 05:00:00

### feat(ui): Update XoL Calculator View Details with table, statistics, and histogram layout

**Description:**
- XoL Calculator 모듈의 View Details 레이아웃 수정
  - 왼쪽: 데이터 테이블 (컬럼 클릭 가능)
  - 오른쪽: 선택된 열의 통계량 표시
  - 아래: 선택된 열의 히스토그램 표시 (숫자형 컬럼인 경우)
  - 상하 스크롤 가능하도록 flex-grow와 overflow-auto 적용
  - 컬럼 클릭 시 선택된 컬럼에 대한 통계량과 히스토그램 자동 표시

**Files Affected:**
- `components/DataPreviewModal.tsx` - XoL Calculator 모듈 View Details 레이아웃 수정
  - 왼쪽/오른쪽 레이아웃으로 변경 (테이블 + 통계량)
  - 선택된 열의 히스토그램을 아래에 추가
  - 상하 스크롤 가능하도록 컨테이너 구조 변경

**Reason:**
- 사용자 요청에 따라 XoL Calculator 모듈의 View Details를 더 직관적인 레이아웃으로 개선
- 테이블, 통계량, 히스토그램을 한 화면에서 확인할 수 있도록 개선
- 상하 스크롤을 통해 모든 정보를 쉽게 탐색 가능

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
```

## 2025-01-16 02:30:00

### style(ui): Adjust View Details font size and ensure consistent layout

**Description:**
- View Details의 모든 테이블 글자 크기 조정
  - 테이블 헤더 및 셀 패딩: py-2 px-3 → py-1 px-2, py-1.5 px-3 → py-1 px-2, py-1.5 px-2 → py-1 px-2
  - Apply Threshold 모듈의 테이블 헤더 패딩 조정
  - XoL Calculator 모듈의 테이블 헤더 패딩 조정
  - 일반 테이블의 헤더 및 셀 패딩 조정
- Apply Threshold와 XoL Calculator 모듈 레이아웃 확인
  - 두 모듈 모두 상단 통계량 테이블 + 아래 전체 데이터 테이블 레이아웃으로 이미 구성되어 있음

**Files Affected:**
- `components/DataPreviewModal.tsx` - View Details 테이블의 모든 패딩 크기 조정
  - Apply Threshold 모듈: th 패딩 py-1.5 px-2 → py-1 px-2
  - XoL Calculator 모듈: th 패딩 py-1.5 px-2 → py-1 px-2
  - 일반 테이블: th 패딩 py-2 px-3 → py-1 px-2, td 패딩 py-1.5 px-3 → py-1 px-2

**Reason:**
- 사용자 요청에 따라 View Details의 글자 크기를 더 작게 조정하여 더 많은 정보를 한 화면에 표시
- 일관된 패딩 크기로 UI 통일성 향상

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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

**Commit Hash:** 84e0fe566c4dba419d4c94438bd6cc12423a2a86

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86

# Or direct recovery
git reset --hard 84e0fe566c4dba419d4c94438bd6cc12423a2a86
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
