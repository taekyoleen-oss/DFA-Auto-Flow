---
name: python-codegen-standards
description: DFA-Auto-Flow 모듈별 파이썬 코드 스니펫(codeSnippets.ts) 작성 표준. 새 모듈 추가/기존 모듈 코드 수정, 파라미터 옵션 노출, 재현성(seed)·라이브러리 호환 보장 시 사용. "모듈 코드", "파이썬 스니펫", "템플릿" 관련 작업에 트리거.
---

# Python Codegen Standards (모듈 단위 코드)

## 핵심 구분
- **모듈 단위 코드** = 다양한 입력값·옵션을 보여준다(학습·탐색용). ← 이 스킬의 대상.
- **전체 파이프라인 코드** = 실행 흐름만 깔끔하게. ← `pipeline-export-reproducibility` 스킬 대상.
혼동하지 말 것.

## 템플릿 작성 규칙 (`codeSnippets.ts` templates)
- 플레이스홀더 `{{paramName}}`는 `module.parameters`에서 치환. 누락 시 합리적 기본값으로 동작하게 작성.
- 옵션 노출: 대안 방법은 주석으로 함께 제시(예: `# fillna 옵션: 'mean'|'median'|'mode'|상수`). 사용자가 파라미터를 바꾸면 코드가 그에 맞게 바뀌도록.
- 내부 출력 변수명은 `MODULE_OUTPUT_VAR`(generatePipelineCode.ts)와 정확히 일치(예: HandleMissingValues→`cleaned_data`).

## 재현성·호환성 (절대 규칙)
- 난수: `random_state=42` / `np.random.seed(42)` 명시. seed 없는 무작위 금지.
- 라이브러리: pandas/numpy/scikit-learn/statsmodels만. Pyodide+CPython 양쪽 동작 확인된 API만.
- dtype·정렬을 명시해 환경 간 결과 흔들림 방지.

## 새 모듈 추가 체크리스트
1. `templates[ModuleType]`에 파이썬 코드 추가(옵션 주석 포함).
2. `MODULE_OUTPUT_VAR`에 출력 변수명 등록(generatePipelineCode.ts).
3. `getModuleCode` 특수 분기 필요 여부 확인(ResultModel/EvaluateStat 처럼).
4. parity-qa로 로컬 CPython 실행 검증.

## 검증
변경한 모듈 코드를 단독 파이썬 파일로 만들어 로컬 실행 → 에러 없이 기대 출력 확인. 동일 파라미터로 Pyodide와 결과 비교.
