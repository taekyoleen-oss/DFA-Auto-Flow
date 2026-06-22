# 전체 파이프라인 코드 검증 (verify/) — DFA-Auto-Flow

이 앱의 핵심 불변식을 **자동 회귀 검증**한다:

> "전체 파이프라인 코드"(PipelineCodeModal/Panel에서 복사하는 코드)를 외부 Python에
> 그대로 가져가면 **바로 실행되고, 매번 동일한 결과**가 나온다.

> 원형: `ML Auto Flow/verify/`의 패턴을 DFA-Auto-Flow로 정식 이식한 것이다
> (cross_app_io_improvements.md 작업5 / docs/book_based_plan/01_improvements_dfa.md 2-5).

## 실행

```bash
pnpm run verify:pipelines      # 또는: node verify/run-verification.mjs
```

각 픽스처에 대해:
1. 픽스처(.json)를 앱과 동일하게 hydrate하여 `generateFullPipelineCode`로 전체코드(.py) 생성
2. 필요한 데이터셋 CSV를 임시 디렉토리에 복사
3. 실제 `python`으로 그 .py를 **2회** 실행
4. **(a)** 두 번 모두 exit 0 (외부 실행 가능)  **(b)** 정규화 후 stdout이 byte-identical (동일 결과 재현)

하나라도 실패하면 종료코드 1 → CI에서 회귀를 잡는다.

## 구성

- `generate.ts` — 픽스처 → 전체코드(.py) 생성 엔트리 (esbuild로 번들되어 실행). 앱의
  `utils/generatePipelineCode.ts`(`generateFullPipelineCode`)와 `constants.ts`(`DEFAULT_MODULES`)를
  **읽기 전용**으로 소비한다. 모듈 실행·코드 생성 로직은 건드리지 않는다.
- `run-verification.mjs` — 오케스트레이터(번들·생성·실행·단언·요약).
  - DFA-Auto-Flow는 `esbuild`를 직접 의존하지 않지만 vite(>=6)가 전이 의존으로 포함한다.
    러너의 `resolveEsbuild()`가 vite 경유로 esbuild를 해석하므로 별도 설치가 필요 없다.
- `pipelines/*.json` — 검증 픽스처(저장 파이프라인 포맷: `modules` + 인덱스 기반 `connections`)
  - 선택 필드 `requires: ["statsmodels", ...]` — 해당 파이썬 패키지가 없으면 그 픽스처는 **SKIP**(FAIL 아님)
- `datasets/*.csv` — 결정적 로컬 데이터셋. 정적으로 커밋된 파일이며 러너는 재생성하지 않는다.
  - 데이터셋은 `verify/datasets/`(우선) → `public/` → `Test/` → `samples/` 순으로 찾는다.

## 픽스처 (현재 2개, 둘 다 결정적)

| 파일 | 체인 | 데이터셋 | 결정성 근거 |
|------|------|----------|-------------|
| `01_ols_resultmodel.json` | LoadData → OLSModel → ResultModel | `dfa_claims_numeric.csv` | statsmodels OLS는 닫힌형(closed-form) 최소제곱 → 무작위성 없음. localSamples.json "Example: OLS Model" 레퍼런스를 결정적 데이터로 각색. `requires: statsmodels` |
| `02_linreg_chain.json` | LoadData → SplitData → LinearRegression → TrainModel → ScoreModel → EvaluateModel | `dfa_claims_numeric.csv` | sklearn `LinearRegression`은 무작위성 없음. `SplitData`는 `random_state=42` 고정. 평가지표는 `.6f` 고정 포맷 출력. `requires: sklearn` |

### 데이터셋: `dfa_claims_numeric.csv`
보험계리 맥락의 합성 클레임 데이터(200행, 6컬럼: exposure, prior_claims, vehicle_age,
driver_age, region_risk, claim_amount). `numpy` 고정 시드(`default_rng(42)`)로 **단 한 번**
생성해 정적 파일로 커밋했다. 러너는 이 파일을 읽기만 하고 절대 재생성하지 않으므로 byte-stable.

## 픽스처 추가하기 (새 모듈/체인 검증)

1. `pipelines/NN_이름.json`을 만든다. `modules[].type`/`parameters`와 인덱스 기반 `connections`를 채운다.
2. `LoadData.parameters.source`(또는 `LoadClaimData`)는 `verify/datasets/`에 있는 CSV 파일명으로 둔다.
3. 특정 파이썬 패키지가 필요하면 최상위 `requires`에 적는다.
4. `pnpm run verify:pipelines`로 PASS를 확인한다. 실패하면 생성기/템플릿의 export 버그다 — 고친다.
5. **무작위·시뮬레이션 단계는 seed를 고정**하고, 2회 실행이 byte-identical 함을 확인하기 전에는 추가하지 않는다.

## 알려진 제한 (의도적 범위 외)

- **DFA 몬테카를로 시뮬레이션 모듈**(`SimulateAggDist`, `SimulateFreqServ`, `CombineLossModel` 등)은
  본 하네스에서 **픽스처로 추가하지 않았다**. 이 모듈들의 전체코드 재현성은 두 가지 이유로 불확실하다:
  1. `utils/generatePipelineCode.ts`의 `MODULE_OUTPUT_VAR` 주석이 명시하듯, `Simulate*` 계열은
     변수 배선만으로 재현되지 않고 모듈 상태 파라미터(`frequency_params` 등)에 의존하며,
     `FitSeverityModel`/`CombineLossModel`은 전체코드 템플릿이 부재하다.
  2. 시뮬레이션 자체가 난수에 의존하므로 seed를 코드 경로 끝까지 완전히 고정했음을 증명하기 전에는
     flaky 픽스처가 될 위험이 있다.
  → 가이드라인(작업5)에 따라 **byte-identical을 보장할 수 없는 확률적 픽스처는 제외**했다.
  추후 `Simulate*` 전체코드 템플릿이 seed 고정과 함께 정식화되면 그때 결정적 픽스처를 추가한다.
- 따라서 현재 하네스는 **결정적 통계/회귀 파이프라인**(OLS, sklearn 회귀)만 검증한다 — 이는
  DFA의 적합(fit) 단계 재현성을 보증하기에 충분하며, 핵심 코드 생성 경로를 회귀 방지한다.
