# DFA-Auto-Flow — 앱 설명 책자 제작 방향 및 가능성

> 원형: `ML Auto Flow/docs/azure_ml_book/02_app_booklet_direction.md`.
> 목표: Jeff Barnes의 Azure ML 책과 **유사한 형식**으로, **DFA-Auto-Flow(보험계리 DFA/XoL 재보험 파이프라인)를 설명하는 책자**를 만든다. 본 문서는 **방향·가능성·구조·작업량** 기획서(작성은 승인 후).

---

## 1. 가능성 평가 — 결론: **높음 / 난이도 낮음**

책자화 원천 자료가 앱 안에 풍부하다. "재구성·정리"가 주 작업이다.

| 책자 구성요소 | 이미 존재하는 자산(DFA) | 비고 |
|---|---|---|
| 모듈별 설명 | `constants.ts`(`TOOLBOX_MODULES` 75+), 모듈별 미리보기 모달 40+ | 거의 전부 |
| 앱 개요·사용법 | `README.md`, `CLAUDE.md`(원칙) | 1~2장 |
| Python 분석/재현 | `PYTHON_ANALYSIS_README.md`, `PYODIDE_DEVELOPMENT_GUIDE.md`, `data_analysis_modules.py` | 차별화 강점 장 |
| 코드 예제 | `codeSnippets.ts`, `utils/generatePipelineCode.ts`(임베드 데이터 standalone) | 실행·재현 |
| 재보험/DFA 예제 | `samples/`(DFA Model.mla, GLM_Model.mla), `sampleData.ts` | 도메인 예제 |
| 변경 이력 | `HISTORY.md`(상세) | 부록 |
| 모듈 사용법 | `DiversionChecker_사용법.md` 등 | 부록 |

→ 책과 달리 **클라우드·과금 불필요 + 브라우저 Pyodide 즉시 실행 + standalone 재현성**을 부각.

---

## 2. 책자 목차(안) — 책 8장을 DFA 도메인으로 매핑

| 장 | 제목 | 책 대응 | 핵심 내용 | 주 자산 |
|---|---|---|---|---|
| 1 | 보험계리 데이터 분석 입문 | Ch1 | 손해 분석·DFA·재보험 개념 | 신규 서술 |
| 2 | DFA-Auto-Flow 시작하기 | Ch3 전반 | 캔버스·모듈·연결·실행, 클라우드 불필요 | `README.md` |
| 3 | 데이터 적재·전처리 | Ch3 | LoadClaimData·인플레이션 적용·포맷 변환·열 선택 | `codeSnippets.ts` |
| 4 | 회귀·GLM 모델 | Ch3/Ch5 | Linear/Logistic/Poisson/NegBin, statsmodels | 모듈 모달 |
| 5 | 빈도-심도 손해 모델링 | (책 너머) | Fit Frequency/Severity → 집계분포 시뮬(SimulateAggDist) | DFA군 모듈 |
| 6 | XoL 재보험 프라이싱 | (책 너머) | FitLossDistribution→Exposure Curve→PriceXoLLayer→CededLoss | 재보험군 모듈 |
| 7 | 임계값·집계 분석 | (책 너머) | SettingThreshold/AnalysisThreshold, CombineLossModel | DFA군 모듈 |
| 8 | **Python 코드 내보내기와 재현성** | (앱 차별화) | standalone 실행·byte-identical, verify(정식화 후) | `PYTHON_ANALYSIS_README.md`, `utils/generatePipelineCode.ts` |
| 9 | **AI 보조 기능** | (앱 차별화) | 목표/데이터 기반 파이프라인 생성, 코드 해설 | `utils/aiClient.ts`, `utils/moduleInsights.ts` |
| 부록 A | 모듈 레퍼런스 | — | 75+ 모듈 카드(입출력·파라미터) | `constants.ts`, 모달 |
| 부록 B | 변경 이력 | — | 연혁 | `HISTORY.md` |

### 부각할 강점(책 대비)
1. **클라우드·과금 불필요** — 브라우저만으로.
2. **브라우저 내 Python(Pyodide)** — 설치 없는 즉시 실행.
3. **standalone 재현성** — 데이터 임베드 코드가 외부에서 동일 결과.
4. **AI 보조** — 멀티프로바이더(Gemini/OpenAI/Anthropic) 파이프라인 생성·해설.
5. **재보험·DFA 실무 모듈** — 빈도-심도·집계분포·XoL 프라이싱 일괄.

---

## 3. 제작 경로 / 포맷

| 옵션 | 도구/스킬 | 권장도 |
|---|---|---|
| Markdown → PDF | `make-pdf` | ★ 주 산출물 |
| Word(.docx) | `report-builder` | 보조 |
| PPT(.pptx) | `app-doc-ppt` | 발표용 |

**권장:** Markdown SSOT → `make-pdf` → 필요 시 Word/PPT.

---

## 4. 시각자료 & 작업량

- **스크린샷 자동화:** Playwright(MCP)로 앱(`npm run dev`, 포트 3001) 띄워 각 장 예제 캔버스·모달 캡처. `samples/`의 .mla 로드로 동일 화면 재현.
- **코드 일관성:** 책자 코드는 verify(2-5 정식화 후) 검증 코드만 사용.

| 단계 | 범위 | 산출물 | 분량 |
|---|---|---|---|
| MVP | 1~4장 + 부록 A(핵심 모듈) | PDF | 40~60p |
| 확장판 | 5~9장(재보험/DFA/AI) + 부록 B | PDF | +40~60p |
| 발표본 | 핵심 요약 | PPTX | 20~30 슬라이드 |

> **권고:** 개선 계획(`01_improvements_dfa.md`)의 verify 정식화·레퍼런스 파이프라인을 먼저 완성하면 5~7장 예제·코드·스크린샷을 자동 확보해 중복 작업을 최소화한다.
