# DFA-Auto-Flow

보험·계리(XoL 재보험 프라이싱 / DFA) 비주얼 파이프라인 빌더. Vite + React 19 + Pyodide.

## 불변 원칙 (최우선)
모든 캔버스 모듈은 파이썬으로 구축되며, 앱이 생성하는 파이썬 코드로 **외부 환경에서 동일 결과를 재현**할 수 있어야 한다. 새 모듈/기능엔 반드시 대응 파이썬 코드를 함께 만든다. 이 1:1 대응이 앱의 존재 이유다.

## 하네스: DFA 파이썬 코드 생성·재현·AI

**목표:** 모듈↔파이썬 1:1 대응과 외부 재현성을 지키며 코드 생성·Pyodide 실행·AI 기능을 강화한다.

**트리거:** 모듈 파이썬 스니펫, 전체 파이프라인 코드/내보내기/재현성, Pyodide 실행, AI 기능(로컬 API 키·코드 설명·결과 해석·자동완성·멀티 프로바이더) 관련 작업 요청 시 `dfa-harness-orchestrator` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**에이전트/스킬:** `.claude/agents/`, `.claude/skills/`에서 관리. 목록은 오케스트레이터 스킬 참조.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-06-07 | 초기 구성 (에이전트 5종 + 스킬 4종 + 오케스트레이터) | 전체 | 파이썬 재현성·로컬 API 키·전체코드 클린화 요구 |
| 2026-06-07 | API 키 로컬화(멀티 프로바이더) + 전체코드 데이터 임베드 재현성 + DFA 재현성 버그 4종 수정 | aiClient/AiSettingsModal/generatePipelineCode/codeSnippets | 1차 실행 |
| 2026-06-22 | Phase 4 — URL/원격 데이터 로더(데이터 입력 계층 전용) | server/split-data-server.js(GET /api/proxy-csv CORS 프록시), vite.config.ts(/api/proxy-csv 프록시), components/PropertiesPanel.tsx(UrlSourceLoader: LoadData/LoadClaimData에 URL 입력→프록시 fetch→업로드와 동일하게 fileContent 저장, sourceType:'url'), utils/generatePipelineCode.ts(sourceType==='url'일 때만 pd.read_csv(<url>) 부가 분기) | 공개 CSV URL 로드 지원. Pyodide 실행 경로 불변, 파일 소스 생성 코드 byte-identical 보장(sourceType 미설정 시 기존 분기). build 성공 |
| 2026-06-23 | RandomForest 전체코드 export 템플릿 신설(Elston 기반, 재현성 갭 해소) | codeSnippets.ts(RandomForest 템플릿=검증된 GradientBoosting 패턴 미러, create_random_forest와 정합), verify/pipelines/04_random_forest.json(dfa_claims_numeric 회귀 체인) | ML/JMDC와 동일하게 DFA에도 RandomForest 모듈은 있으나 전체코드 템플릿이 누락된 갭을 해소. 결정적(random_state=42). verify:pipelines 4/4 PASS(신규 04 포함), build 성공 |
| 2026-06-23 | ★재현성 원칙 점검·복구: LogisticRegression/SVM/NaiveBayes 템플릿 갭 수정 | codeSnippets.ts(3종 export 템플릿 신설; create_* 정합·결정적, NaiveBayes는 var_smoothing 파라미터), verify/datasets/dfa_claims_class.csv(claim_amount 중앙값 분할 파생 이진 라벨), verify/pipelines/05~07(분류 체인) | 이 3모델 배치 시 전체코드 빈 블록→NameError 잠복 버그 복구(원칙: 분석 모듈은 검증 가능한 Python으로 export). verify:pipelines 7/7 PASS, build 성공. (추가 발견: DFA 군집 패밀리 KMeans/DBSCAN/Hierarchical은 TrainClusteringModel/ClusteringData 부재로 미완성 → ML Auto Flow docs/azure_ml_book/05 문서에 기록, 완성/제거는 별도 결정) |
