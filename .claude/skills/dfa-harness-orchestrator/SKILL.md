---
name: dfa-harness-orchestrator
description: DFA-Auto-Flow(InsureAutoFlow) 앱의 파이썬 코드 생성·재현성·Pyodide 실행·AI 기능 작업을 조율하는 오케스트레이터. 모듈 파이썬 스니펫 추가/수정, 전체 파이프라인 코드 내보내기·standalone 재현성, Pyodide 실행 일치, AI 기능(로컬 API 키·코드 설명·결과 해석·자동완성)·멀티 프로바이더 작업 시 사용. "전체코드", "파이썬 재현", "모듈 코드", "API 키", "AI 기능" 관련 요청에 반드시 트리거. 후속 요청("다시 실행", "재실행", "업데이트", "수정", "보완", "이전 결과 기반", "OO만 다시")에도 트리거.
---

# DFA Harness Orchestrator

DFA-Auto-Flow의 핵심 가치(모듈↔파이썬 1:1, 외부 재현성)를 지키며 작업을 조율한다. 상세 원칙은 메모리 [[python-module-codegen-is-core]] 참조.

## 실행 모드
**에이전트 팀**이 기본. 작업이 단일 영역(예: 한 모듈 스니펫만 수정)이면 해당 에이전트를 서브로 직접 호출. 단, **사용자가 명시적으로 팀/병렬 실행을 요청한 경우에만** 팀을 구성한다. 그 외에는 메인이 직접 편집하고 이 스킬의 원칙을 따른다.

## 에이전트 구성
- `python-codegen-engineer` — 모듈별 파이썬 스니펫(옵션 풍부). `codeSnippets.ts`.
- `pipeline-export-engineer` — 전체코드(실행 흐름만 + 데이터 임베드 standalone). `generatePipelineCode.ts`.
- `ai-feature-engineer` — 멀티 프로바이더 로컬 API 키 + AI 기능. `utils/aiClient.ts`.
- `pyodide-runtime-engineer` — Pyodide↔CPython 결과 일치. `utils/pyodideRunner.ts`.
- `python-parity-qa` — 생성 코드의 외부 CPython 실행 동치 검증 (general-purpose).

## Phase 0: 컨텍스트 확인
1. `_workspace/` 존재 + 부분 수정 요청 → **부분 재실행**(해당 에이전트만).
2. `_workspace/` 존재 + 새 입력 → **새 실행**(기존을 `_workspace_prev/`로 이동).
3. `_workspace/` 미존재 → **초기 실행**.

## Phase 1: 분류 → 라우팅
| 요청 유형 | 담당 | 검증 |
|----------|------|------|
| 모듈 코드 추가/수정 | python-codegen-engineer | parity-qa |
| 전체코드 내보내기·클린화·재현성 | pipeline-export-engineer | parity-qa |
| API 키·AI 기능·프로바이더 | ai-feature-engineer | 수동/UI |
| Pyodide 실행·패키지·성능 | pyodide-runtime-engineer | parity-qa |

## Phase 2: 변경 → 검증 (생성-검증 패턴)
모든 파이썬 코드 변경은 `python-parity-qa`가 **로컬 CPython(Python 3.13 + pandas/numpy/sklearn/statsmodels)에서 실제 실행**해 동치를 확인하기 전엔 완료로 보지 않는다. 데이터 임베드 코드는 파일 의존 없이 단독 실행돼야 한다.

## 데이터 전달
- 파일 기반: 중간 산출물은 `_workspace/`(`{phase}_{agent}_{artifact}.{ext}`), 최종만 실제 경로.
- 메시지 기반: 에이전트 간 규약(변수명·패키지·출력형식) 공유.

## 에러 핸들링
1회 재시도 후 재실패 시 해당 변경을 건너뛰고 리포트에 누락 명시. parity 불일치는 삭제하지 말고 원인 에이전트에 출처와 함께 전달.

## 불변 가드레일
- 새 모듈/기능엔 반드시 대응 파이썬 코드를 함께 만든다(UI만 추가 금지).
- API 키는 번들에 넣지 않는다(localStorage 런타임 로드).
- 표시 코드 ≡ 실행 코드(로직 분기 최소화).

## 테스트 시나리오
- **정상:** "SplitData 모듈 코드에 stratify 옵션 추가" → codegen이 템플릿 수정 → parity-qa가 로컬 실행으로 train/test shape 동일 확인 → 통과.
- **에러:** parity-qa가 Pyodide엔 있고 CPython엔 다른 결과 발견 → runtime-engineer에 리포트 → seed/정렬 고정 → 재검증.
