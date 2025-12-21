# Preview Backup

이 폴더는 View Details의 내용을 백업하기 위한 폴더입니다.

## 백업된 파일

### XoLCalculator_ViewDetails_Backup.tsx
- XoL Calculator 모듈의 View Details 관련 코드 백업
- 포함된 내용:
  - `YearlyAmountBarPlot` 컴포넌트
  - `getXolData` 함수
  - `XoLCalculatorViewDetails` 컴포넌트 (UI 렌더링 부분)

## 복구 방법

1. `DataPreviewModal.tsx` 파일을 엽니다.
2. XoL Calculator 관련 코드를 찾습니다.
3. 백업 파일의 내용을 참조하여 복구합니다.

## 주의사항

- 백업 파일은 참조용입니다. 직접 import하여 사용할 수 없습니다.
- `ColumnStatistics`와 `HistogramPlot` 컴포넌트는 `DataPreviewModal.tsx`에 정의되어 있어야 합니다.
- `ChevronUpIcon`과 `ChevronDownIcon`은 `components/icons.tsx`에서 import해야 합니다.

## 백업 일자

- XoL Calculator: 2025-01-17 (최신 업데이트)

## XoLCalculator_ViewDetails_Backup.tsx 상세 내용

**최신 업데이트:** 2025-01-17

**포함된 내용:**
- `YearlyAmountBarPlot` 컴포넌트: 연도별 금액을 표시하는 Bar Plot
- `getXolData` 함수: XoL Calculator 데이터 필터링 및 연도별 집계 로직
- `XoLCalculatorViewDetails` 컴포넌트: 전체 View Details UI
  - **"건별 XoL 적용" 탭:**
    - 테이블 (10행, 가로 스크롤, 숫자 천단위 구분, 오른쪽 정렬, 연도는 천단위 구분 없음)
    - 통계량 (테이블 아래, ColumnStatistics 컴포넌트 사용)
    - 히스토그램 (보험금, XoL Claim(Incl. Limit) - SVG 기반, Setting Threshold 스타일)
  - **"연도별 XoL 적용" 탭:**
    - 테이블 (text-sm, 가로/세로 스크롤, 숫자 천단위 구분)
    - 통계 정보 (XoL Claim 평균, 표준편차, XoL Premium Rate 평균, Reluctance Factor)
    - 통계량 + XoL Claim(Incl. Agg/Reinst) 그래프 (YearlyAmountBarPlot)
    - 데이터 없을 때 안내 메시지 표시

