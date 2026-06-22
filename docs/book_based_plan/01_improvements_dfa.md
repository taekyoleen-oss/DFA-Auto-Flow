# DFA-Auto-Flow — 책 기반 개선 사항 및 추가 기능 (앱별 계획)

> 원형: `ML Auto Flow/docs/azure_ml_book/01_book_based_improvements.md` (Jeff Barnes, *Microsoft Azure Essentials: Azure Machine Learning*, 2015 기반).
> 횡단 공통 I/O: `ML Auto Flow/docs/cross_app_io_improvements.md` 참조.
> 대상: **DFA-Auto-Flow** — 보험계리 **Dynamic Financial Analysis / XoL 재보험 프라이싱** 노드 파이프라인. 본 문서는 **계획서**이며 구현은 승인 후 별도 진행.

---

## 0. 한눈에 보기 — 같은 엔진, 다른 도메인

DFA-Auto-Flow는 ML Auto Flow와 **동일한 패턴**을 공유한다(탐색 확인): 캔버스(`components/Canvas.tsx`), 모듈 enum(`types.ts`), 팔레트(`constants.ts`의 `TOOLBOX_MODULES` 75+), Python 코드 생성(`codeSnippets.ts`, `utils/generatePipelineCode.ts`), Pyodide 실행(`utils/pyodideRunner.ts`), 참조 구현(`data_analysis_modules.py`), 멀티프로바이더 AI(`utils/aiClient.ts`). 기 보유 모듈에 `LoadClaimData`/`LoadData`, `SplitData`, `TrainModel`, `ScoreModel`, `EvaluateModel`, 회귀군(Linear/Logistic/Poisson…), 군집(KMeans/DBSCAN/Hierarchical/PCA), statsmodels(OLS/Poisson/NegBin…), **재보험군**(FitLossDistribution, PriceXoLLayer, XolPricing…), **DFA군**(Aggregate/Frequency/Severity 모델, SimulateAggDist 등)이 있다.

따라서 01의 개선안은 **Pyodide·sklearn 기반이므로 대부분 적용 가능**하되, "책 예제"는 ML 교과서가 아니라 **DFA/재보험 계리 맥락**으로 재해석한다. dev 포트 3001, 선택적 Supabase 샘플 연동, `.claude` 하네스(python-parity-qa 등) 보유.

---

## 1. 개선 항목 (기존 기능 강화)

### 1-1. EvaluateModel / EvaluateStat "계리 수준" 강화 (01의 2-1) ★높음
- DFA의 모델 적합도·예측 평가를 강화: 분류엔 ROC/AUC·혼동행렬, 회귀/GLM엔 잔차·이탈도(deviance)·정보기준(AIC/BIC). 재보험 빈도-심도 적합엔 Q-Q/PP plot·적합도 검정.
- 영향: `data_analysis_modules.py`, `codeSnippets.ts`, `components/EvaluationPreviewModal.tsx`, `components/EvaluateStat*` 상당물.
- 재현성: 결정적 → 불변식 영향 없음. 템플릿 정합 유지.

### 1-2. 회귀/GLM 평가지표 정합 (01의 2-2)
- RMSE·MAE·상대오차 + GLM 표준(이탈도·피어슨 χ²) 명시 노출.
- 영향: 동상.

### 1-3. 데이터 개요/요약 패널 (01의 2-3 → 공통 문서 작업1)
- `LoadClaimData` 직후 열별 타입·결측·분포 요약(claim 금액 분포 강조).
- 영향: `components/PropertiesPanel.tsx`(미리보기 탭), `components/DataPreviewModal.tsx`.

---

## 2. 추가 기능 (신규 모듈/기능)

### 2-1. 그래디언트 부스팅 트리 (01의 3-1) — 지도학습군 확장 ★최우선
- **근거:** 이미 회귀·트리·랜덤포레스트 보유. claim 심도/사고 예측에 GBM은 강력. 기존 지도학습 모듈 패턴을 그대로 복제.
- **구현:** `sklearn.ensemble.GradientBoostingRegressor/Classifier` 또는 `HistGradientBoosting*`.
- **영향:** `types.ts`(enum), `constants.ts`(`TOOLBOX_MODULES`), `codeSnippets.ts`(`random_state=42`), `data_analysis_modules.py`, `components/PropertiesPanel.tsx`(n_estimators·learning_rate·max_depth), `components/ComponentRenderer.tsx`.
- **재현성:** `random_state=42` 고정 + verify 픽스처.

### 2-2. URL/공개 저장소 claim 데이터 로더 (01의 3-2 → 공통 문서 작업2) ★높음
- `LoadClaimData`/`LoadData`에 URL 옵션. Pyodide CORS는 `server/split-data-server.js` 프록시 경유.
- 영향: `codeSnippets.ts`, `components/PropertiesPanel.tsx`, `server/`.
- 재현성: URL 데이터 가변 → 검증용 로컬 스냅샷 권장.

### 2-3. 번들 레퍼런스 파이프라인 확장 (01의 3-3 → 공통 문서 작업3) ★난이도 낮음
- DFA 대표 시나리오를 로드 즉시 실행 가능한 레퍼런스로: ①빈도-심도 적합→집계분포 시뮬, ②XoL 레이어 프라이싱, ③임계값(threshold) 분석.
- 영향: `sampleData.ts`, `savedSamples.ts`, `samples/`(DFA Model.mla / GLM_Model.mla 확장), `localSamples.json`.

### 2-4. 샘플 메타데이터 스키마 강화 (공통 문서 작업4)
- `localSamples.json`/`utils/samples.ts`에 분포가정·재보험 레이어·기대 결과 속성 추가.

### 2-5. 재현성 verify 하네스 정식화 (공통 문서 작업5) ★고가치
- 현재 `.claude` 하네스(python-parity-qa)만 존재 → **ML Auto Flow의 `verify/run-verification.mjs` 패턴을 정식 이식**: `verify/pipelines/*.json` + 외부 Python 2회 byte-identical. 시뮬레이션 모듈(SimulateAggDist 등)은 seed 고정 후 픽스처.
- 영향: 신규 `verify/`, `utils/generatePipelineCode.ts`(검증 코드 생성 경로).

### 2-6. 모델/프라이싱 스코어링 내보내기 (01의 3-6 → 공통 문서 작업6) ★중
- 적합 모델·프라이싱 함수를 `joblib` 저장 + FastAPI/Flask 스코어링 스니펫 + 요청/응답 JSON으로 내보내기. 고급기능 게이트 대상.
- 영향: `utils/generatePipelineCode.ts`, `codeSnippets.ts`, PipelineCode 상당 모달.

### 2-7. 재학습/지속학습 (01의 3-7) ★낮음(장기)
- 신규 사고연도 데이터 주입 → 재적합 → 버전 저장. Supabase 샘플 연동 활용 여지.

> **범위 외:** 01의 **3-5 추천 모듈**은 DFA 도메인과 직접 연관이 낮아 **해당 없음**(필요 시 별도 검토).

---

## 3. 우선순위 요약 (DFA)

| 우선순위 | 항목 | 난이도 | 비고 |
|---|---|---|---|
| 1 | 2-1 그래디언트 부스팅 | 중 | 지도학습군 미러링 |
| 2 | 2-3 레퍼런스 파이프라인 + 2-4 메타 | 낮음 | DFA/XoL 시나리오 |
| 3 | 1-1 Evaluate 강화 | 중 | 계리 적합도 |
| 4 | 2-2 URL claim 로더 | 중 | CORS 프록시 |
| 5 | 2-5 verify 정식화 | 중 | 재현성 보증 |
| 6 | 2-6 스코어링 내보내기 | 중 | 고급 게이트 |
| 7 | 1-2 지표 정합 / 2-7 재학습 | 낮음/상 | |

> **불변식:** `data_analysis_modules.py` ↔ `codeSnippets.ts` 정합, 무작위·시뮬 단계 seed 고정, verify 픽스처로 byte-identical 재현(DFA CLAUDE.md의 재현성 원칙과 정합).


---

## 부록: 구현 결과 (2026-06-22)

> DFA 도메인에 맞춰 개선안을 구현·검증했습니다(재현성 verify 하네스 신설 포함). 재학습은 후속.

### 항목별 구현 상태
| 항목 | 상태 | 비고 |
|---|---|---|
| 1-1 Evaluate/EvaluateStat 강화 | ✅ | 분류 ROC-AUC·Average Precision, GLM Deviance·Pearson χ²·AIC/BIC |
| 1-2 회귀/GLM 지표 정합 | ✅ | RMSE/MAE/RSE/RAE |
| 1-3 데이터 개요 패널 | ✅ | utils/dataOverview.ts |
| 2-1 그래디언트 부스팅 | ✅ | 지도학습군 확장, random_state=42, 픽스처 03 |
| 2-2 URL claim 로더 | ✅ | 입력층 fetch+/api/proxy-csv |
| 2-3 레퍼런스 파이프라인 / 2-4 메타 | ✅ | 빈도-심도·XoL·임계값 3종 + 메타 |
| 2-5 재현성 verify 하네스 정식화 | ✅ | verify/ 신설 + verify:pipelines 스크립트 (+OLS 예제 LoadData→LoadClaimData 버그 수정) |
| 2-6 스코어링 내보내기 | ✅ | utils/scoringExport.ts(joblib+FastAPI/Flask) |
| 2-7 재학습/지속학습 | ⏳ 후속 | 장기 |
| 3-5 추천 | — 범위 외 | DFA 도메인 무관(해당 없음) |

### 검증
- `npm run verify:pipelines` → **3/3 PASS** (01_ols_resultmodel, 02_linreg_chain, 03_gradient_boosting; 외부 Python 2회 byte-identical).
- `vite build` 성공. 모든 변경 가산적, 기존 실행/연결/시각화 불변.
