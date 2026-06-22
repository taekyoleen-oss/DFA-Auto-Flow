// utils/dataOverview.ts
//
// 데이터 개요/요약 (Phase 3, 작업 1) — 읽기 전용·순수 TypeScript 집계.
//
// 이미 클라이언트에 로드된 미리보기 객체(columns + rows)에서만 계산한다.
// Pyodide/Python 을 호출하지 않으며, 모듈 실행·연결·시각화 동작을 전혀 건드리지 않는다.
// LoadData/LoadClaimData·codeSnippets·generatePipelineCode 와 무관한 추가(additive) 요약이다.

import type { ColumnInfo } from '../types';

export type InferredColumnType = 'numeric' | 'categorical';

export interface ColumnOverview {
  /** 열 이름 */
  name: string;
  /** DataPreview.ColumnInfo 가 보고한 원본 타입 문자열(있으면) */
  declaredType: string;
  /** 미리보기 값으로 추론한 타입 (수치형/범주형) */
  inferredType: InferredColumnType;
  /** 한글 표기: '수치형' | '범주형' */
  inferredTypeLabel: string;
  /** 미리보기 행 기준 결측/빈값 수 (null·undefined·빈 문자열) */
  missingCount: number;
  /** 미리보기 행 수 대비 결측 비율(0~1) */
  missingRatio: number;
  /** 결측이 1개 이상인지 (강조용) */
  hasMissing: boolean;
}

export interface DataOverview {
  /** 미리보기 객체가 보고한 전체 행 수(totalRowCount)가 있으면 그것, 없으면 표본 행 수 */
  rowCount: number;
  /** 요약 계산에 실제로 사용된 미리보기 표본 행 수 */
  sampleRowCount: number;
  /** rowCount 가 표본(전체의 일부)인지 여부 */
  isSample: boolean;
  /** 열 수 */
  columnCount: number;
  /** 열별 요약 */
  columns: ColumnOverview[];
  /** 결측이 있는 열 수 */
  columnsWithMissing: number;
}

/** DataPreview 와 호환되는 최소 입력 형태 (실제 DataPreview 객체를 그대로 받을 수 있음). */
export interface DataOverviewInput {
  columns?: ColumnInfo[] | null;
  rows?: Record<string, any>[] | null;
  totalRowCount?: number | null;
}

const isMissing = (v: any): boolean =>
  v === null || v === undefined || (typeof v === 'string' && v.trim() === '');

const isNumericValue = (v: any): boolean => {
  if (typeof v === 'number') return Number.isFinite(v);
  if (typeof v === 'string') {
    const t = v.trim();
    if (t === '') return false;
    const n = Number(t);
    return Number.isFinite(n);
  }
  return false;
};

const looksNumericByDeclaredType = (declaredType: string): boolean => {
  const t = (declaredType || '').toLowerCase();
  return (
    t.includes('int') ||
    t.includes('float') ||
    t.includes('double') ||
    t.includes('number') ||
    t.includes('numeric') ||
    t.includes('decimal') ||
    t.includes('real')
  );
};

/**
 * 미리보기 객체에서 데이터 개요를 계산한다 (순수 함수, 부작용 없음).
 * columns 가 비어 있으면 null 을 반환해 "표시할 것 없음" 을 신호한다.
 */
export function computeDataOverview(input: DataOverviewInput | null | undefined): DataOverview | null {
  if (!input) return null;
  const columns = Array.isArray(input.columns) ? input.columns : [];
  const rows = Array.isArray(input.rows) ? input.rows : [];

  if (columns.length === 0) return null;

  const sampleRowCount = rows.length;
  const declaredTotal =
    typeof input.totalRowCount === 'number' && input.totalRowCount >= 0
      ? input.totalRowCount
      : null;
  const rowCount = declaredTotal !== null ? declaredTotal : sampleRowCount;
  const isSample = declaredTotal !== null && declaredTotal > sampleRowCount;

  const columnOverviews: ColumnOverview[] = columns.map((col) => {
    const name = col?.name ?? '';
    const declaredType = col?.type ?? '';

    let missingCount = 0;
    let numericNonMissing = 0;
    let nonMissing = 0;

    for (const row of rows) {
      const value = row ? row[name] : undefined;
      if (isMissing(value)) {
        missingCount++;
        continue;
      }
      nonMissing++;
      if (isNumericValue(value)) numericNonMissing++;
    }

    // 타입 추론: 표본 데이터가 있으면 값 기반(비결측 중 90% 이상이 수치 → 수치형),
    // 표본이 없으면 declaredType 으로 폴백.
    let inferredType: InferredColumnType;
    if (nonMissing > 0) {
      inferredType = numericNonMissing / nonMissing >= 0.9 ? 'numeric' : 'categorical';
    } else {
      inferredType = looksNumericByDeclaredType(declaredType) ? 'numeric' : 'categorical';
    }

    const denom = sampleRowCount > 0 ? sampleRowCount : 0;
    const missingRatio = denom > 0 ? missingCount / denom : 0;

    return {
      name,
      declaredType,
      inferredType,
      inferredTypeLabel: inferredType === 'numeric' ? '수치형' : '범주형',
      missingCount,
      missingRatio,
      hasMissing: missingCount > 0,
    };
  });

  return {
    rowCount,
    sampleRowCount,
    isSample,
    columnCount: columns.length,
    columns: columnOverviews,
    columnsWithMissing: columnOverviews.filter((c) => c.hasMissing).length,
  };
}
