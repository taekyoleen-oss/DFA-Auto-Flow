// Static descriptions shown by the 설명(ⓘ) popup on each module (DFA-Auto-Flow).
// 키워드: 역할(무엇을), 언제 사용, 입력(소비 데이터), 결과(산출물),
//        파라미터, 권장 연결(앞→뒤 모듈), 흔한 오류, 비고.
// DFA(동적재무분석)/재보험 도메인: 손해 클레임 데이터, 빈도-심도 모델, XoL 계약/계산,
// 몬테카를로 시뮬레이션, 임계값(threshold) 꼬리 분석, 전통 통계 GLM 등에 맞춰 작성.
// beginner: 비유·일상어 2~4문장 / analysisMethod: 내부 알고리즘을 평이하게 2~4문장.

import { ModuleType } from "./types";

export interface ModuleDescription {
  title: string;
  category?: string;
  /** 🔰 초보자용 쉬운 설명 — 비유·일상어로 이 모듈이 무엇을 하는지(전문용어 최소화).
   *  모달 최상단에 강조 표시된다. 처음 보는 사용자가 한눈에 이해하도록 작성. */
  beginner?: string;
  /** 📊 분석 방법 — 이 모듈이 내부적으로 어떤 분석/계산/알고리즘을 수행하는지 글로 풀어 설명
   *  (라이브러리·수식 언급 가능하되 평이하게). 초보자가 "왜/어떻게"를 이해하도록. */
  analysisMethod?: string;
  role: string;
  input: string;
  output: string;
  parameters?: string;
  /** 언제 이 모듈을 쓰면 좋은지 — 사용 시점/판단 기준 */
  whenToUse?: string;
  /** 권장 연결 — 일반적인 업스트림(앞) → 다운스트림(뒤) 모듈 */
  connections?: string;
  /** 흔한 오류 — 자주 막히는 지점과 해결법 (경고 톤으로 표시) */
  commonErrors?: string;
  notes?: string;
}

export const MODULE_DESCRIPTIONS: Partial<
  Record<ModuleType, ModuleDescription>
> = {
  // ===== Data I/O / Preprocess =====
  [ModuleType.LoadClaimData]: {
    title: "Load Claim Data",
    category: "데이터 입출력",
    beginner:
      "분석에 쓸 보험 클레임(손해) 데이터를 불러오는 '출발점' 블록입니다. CSV 파일을 선택하면 종목·날짜·클레임 금액 같은 항목이 표(행과 열) 형태로 앱에 들어옵니다. 모든 DFA·재보험 분석은 여기서 시작하므로 파이프라인 맨 앞에 두고 ▶ 버튼으로 먼저 실행하세요.",
    analysisMethod:
      "선택한 CSV의 첫 줄을 '열 이름'으로, 나머지를 클레임 행으로 읽어 표(판다스 DataFrame)로 만듭니다. 파일이 없을 때는 데모용 합성 클레임을 생성할 수도 있습니다(금액은 로그정규분포로, 연도 범위에 걸쳐). 이 단계는 계산을 하지 않고 '데이터를 표로 정리'만 하며, 이후 모듈이 이 표를 이어받습니다.",
    role: "클레임 CSV를 업로드해 DFA/재보험 파이프라인의 시작점으로 삼습니다. Pyodide 메모리에 DataFrame으로 로드됩니다.",
    input: "사용자가 선택한 로컬 .csv / .xlsx 파일(종목구분·날짜·클레임 금액 등). 헤더 1줄 + 데이터 N줄.",
    output: "전체 클레임 DataFrame. 후속 모듈(Statistics, ApplyInflation, FormatChange, SplitBy* 등)에 연결.",
    parameters: "fileContent: 파일 내용(자동) · source: 표시용 경로 · start_year/end_year/claims_per_year(합성 생성 시).",
    whenToUse: "모든 클레임 분석 파이프라인의 첫 모듈. 손해 데이터를 가져올 때 가장 먼저 배치합니다.",
    connections: "(시작점) → Statistics · ApplyInflation · FormatChange · SettingThreshold · SplitByThreshold/SplitByFreqServ.",
    commonErrors: "파일 미실행 상태에서 하위 모듈을 돌리면 'LoadClaimData를 먼저 실행' 오류 — ▶로 먼저 실행하세요. 금액·날짜 컬럼명이 후속 모듈 설정과 맞아야 합니다.",
    notes: "파일 크기가 너무 크면 Pyodide 메모리 한계로 중단될 수 있습니다.",
  },
  [ModuleType.LoadData]: {
    title: "Load Data",
    category: "데이터 입출력",
    beginner:
      "일반 CSV/엑셀 데이터를 불러오는 출발점 블록입니다(클레임 전용이 아닌 범용 로더). 파일을 선택하면 표 형태로 앱에 들어오고, 모든 분석이 여기서 시작합니다.",
    analysisMethod:
      "선택한 파일의 첫 줄을 열 이름으로, 나머지를 데이터 행으로 읽어 DataFrame을 만듭니다. 엑셀은 첫 시트를 CSV처럼 변환합니다. 계산 없이 '표로 정리'만 합니다.",
    role: "CSV/Excel 파일을 업로드해 파이프라인의 시작점으로 삼습니다.",
    input: "로컬 .csv / .xlsx / .xls 파일.",
    output: "전체 DataFrame. 후속 전처리·모델 모듈에 연결.",
    whenToUse: "클레임이 아닌 일반 표 데이터를 가져올 때.",
    connections: "(시작점) → Statistics · SelectData · 전처리 → SplitData/모델링.",
    commonErrors: "먼저 ▶로 실행해야 하위 모듈이 데이터를 받습니다. 인코딩이 깨지면 UTF-8로 저장 후 재업로드.",
  },
  [ModuleType.Statistics]: {
    title: "Statistics",
    category: "탐색적 분석",
    beginner:
      "데이터의 '건강검진 요약표'를 만들어 주는 블록입니다. 각 열의 평균·결측·분포, 열 사이의 상관관계를 한눈에 보여 줘서, 본격적인 분석 전에 '내 데이터가 어떻게 생겼는지' 살펴볼 때 씁니다.",
    analysisMethod:
      "각 수치형 열의 평균·중앙값·표준편차·최소/최대·결측 비율과 왜도·첨도를 계산하고, 열들 사이의 상관계수(−1~+1)를 행렬로 구해 히트맵으로 보여 줍니다. 데이터를 바꾸지 않고 요약 정보만 산출합니다.",
    role: "수치형/범주형 컬럼별 기술통계(평균·중앙값·분산·결측·왜도·첨도)와 상관행렬·히트맵을 생성합니다.",
    input: "DataFrame 1개.",
    output: "기술통계표 + 상관계수 행렬 + 시각화. '결과 보기'로 확인.",
    whenToUse: "분석 초반 데이터의 분포·결측·상관 구조를 한눈에 파악하고 싶을 때(EDA 첫 단계).",
    connections: "LoadClaimData/LoadData → Statistics. (데이터를 변형하지 않는 확인용 분기.)",
    commonErrors: "수치형 컬럼이 2개 미만이면 상관행렬이 비어 보일 수 있습니다.",
  },
  [ModuleType.SelectData]: {
    title: "Select Data",
    category: "데이터 가공",
    beginner:
      "표에서 '필요한 열만 골라 담는' 블록입니다. 큰 표에서 분석에 쓸 열만 체크하고 나머지는 버립니다. 각 열이 예측에 쓸 입력(피처)인지 맞히려는 정답(타깃)인지 역할도 미리 지정할 수 있어요.",
    analysisMethod:
      "선택한 컬럼명만 남기고 DataFrame을 잘라내며(df[선택열]), 각 컬럼의 역할·자료형 정보를 메타데이터로 함께 전달합니다. 값 자체는 바꾸지 않습니다.",
    role: "분석에 사용할 컬럼만 골라내고, 각 컬럼의 역할(피처/타깃)이나 자료형을 지정합니다.",
    input: "DataFrame 1개.",
    output: "선택된 컬럼만 남은 DataFrame + 컬럼 메타데이터.",
    parameters: "columnSelections: {컬럼명: {selected, type, role}}",
    whenToUse: "불필요한 컬럼을 떼어내거나 피처/타깃 역할을 명시해 후속 모델링을 단순화할 때.",
    connections: "Load* → SelectData → 전처리/모델링.",
    commonErrors: "타깃 컬럼을 실수로 제외하면 모델 모듈이 라벨을 찾지 못합니다.",
  },
  [ModuleType.ApplyInflation]: {
    title: "Apply Inflation",
    category: "데이터 전처리",
    beginner:
      "과거에 발생한 클레임 금액을 '오늘 가치(목표 연도 기준)로 환산'해 주는 블록입니다. 10년 전의 1억 원과 올해의 1억 원은 가치가 다르듯, 물가상승률을 반영해 모든 클레임을 같은 기준 연도로 맞춰 공정하게 비교·합산할 수 있게 합니다.",
    analysisMethod:
      "각 클레임에 대해 발생 연도와 목표 연도의 차이만큼 인플레이션을 복리로 곱합니다: 환산금액 = 원금액 × (1 + 인플레이션율)^(목표연도 − 발생연도). 결과를 원래 금액 컬럼명에 '_infl'을 붙인 새 컬럼으로 추가합니다.",
    role: "연간 인플레이션율로 클레임 금액을 목표 연도 가치로 보정(on-leveling)합니다.",
    input: "DataFrame 1개 (금액 컬럼 + 연도/날짜 컬럼).",
    output: "인플레이션 보정 금액이 추가된 DataFrame(원컬럼명_infl).",
    parameters: "amount_column · year_column · inflation_rate(연 %, 예: 3) · target_year(기준 연도)",
    whenToUse: "여러 해에 걸친 클레임을 같은 화폐가치로 맞춰(트렌딩) 분포 적합·요율 산출의 기준을 일관되게 할 때.",
    connections: "LoadClaimData → (FormatChange로 연도 추출) → ApplyInflation → 분포 적합/임계값 분석.",
    commonErrors: "year_column이 연도(정수)가 아니면 보정이 틀어집니다 — 날짜만 있으면 먼저 FormatChange로 '연도'를 만드세요. 인플레이션율은 % 단위(3=3%)로 입력합니다.",
  },
  [ModuleType.FormatChange]: {
    title: "Format Change",
    category: "데이터 전처리",
    beginner:
      "날짜 열(예: 2018-05-03)에서 '연도(2018)'만 뽑아 새 열로 추가해 주는 블록입니다. 연도별로 클레임을 묶거나(빈도 집계), 인플레이션을 적용하려면 연도 정보가 필요한데, 이 블록이 그 준비를 해 줍니다.",
    analysisMethod:
      "지정한 날짜 컬럼을 datetime으로 해석한 뒤 연(year)을 추출해 '연도' 컬럼을 날짜 옆에 끼워 넣습니다. 다른 값은 그대로 두고 컬럼 하나만 추가하는 결정적 변환입니다.",
    role: "날짜 컬럼에서 연도를 추출해 새 '연도' 컬럼을 추가합니다.",
    input: "DataFrame 1개 (날짜 컬럼 포함).",
    output: "'연도' 컬럼이 추가된 DataFrame.",
    parameters: "date_column(연도를 뽑을 날짜 컬럼)",
    whenToUse: "연도별 집계·인플레이션 보정·빈도 분석에 필요한 연도 열을 만들 때.",
    connections: "LoadClaimData → FormatChange → ApplyInflation / SplitByFreqServ / SettingThreshold.",
    commonErrors: "date_column이 날짜로 해석되지 않으면(형식 불일치) 추출이 실패합니다 — YYYY-MM-DD 등 표준 형식을 권장합니다.",
  },
  [ModuleType.NormalizeData]: {
    title: "Normalize Data",
    category: "데이터 전처리",
    beginner:
      "단위가 제각각인 숫자 열들을 '같은 눈금'으로 맞춰 주는 블록입니다. 예컨대 클레임 건수(0~10)와 금액(수억 단위)을 그대로 쓰면 금액이 계산을 압도하므로, 모든 변수를 비슷한 크기로 줄여 거리·경사 기반 모델이 한 변수에 휘둘리지 않게 합니다.",
    analysisMethod:
      "선택한 방식으로 수치형 컬럼을 변환합니다: MinMax는 최솟값~최댓값을 0~1로, StandardScaler는 (값−평균)/표준편차로 평균0·분산1로, RobustScaler는 중앙값과 IQR을 써서 이상치에 덜 민감하게 변환합니다. 학습 기준(스케일러)을 저장해 동일하게 적용합니다.",
    role: "수치형 변수를 MinMax / StandardScaler / RobustScaler로 정규화·표준화합니다.",
    input: "DataFrame 1개.",
    output: "스케일링된 DataFrame + 스케일러 정보.",
    parameters: "method: MinMax(0~1) / StandardScaler(평균0,분산1) / RobustScaler(중앙값,IQR) · columnSelections",
    whenToUse: "거리/경사 기반 알고리즘(KNN, SVM, 신경망, 군집) 전에는 거의 필수.",
    connections: "전처리 → NormalizeData → 거리/경사 기반 모델.",
    commonErrors: "트리 계열(DecisionTree, RandomForest)에는 효과가 없습니다. 이상치가 많으면 MinMax가 왜곡되니 RobustScaler를 고려.",
  },
  [ModuleType.TransitionData]: {
    title: "Transition Data",
    category: "데이터 전처리",
    beginner:
      "한쪽으로 길게 치우친(꼬리가 긴) 클레임 금액 같은 데이터를 '대칭에 가깝게 펴 주는' 블록입니다. 소수의 큰 손해가 분포를 끌고 가는 경우, log 같은 변환으로 큰 값을 눌러 균형을 맞춥니다. 그러면 분포 적합·통계검정의 가정이 더 잘 맞습니다.",
    analysisMethod:
      "선택한 변환식을 컬럼에 적용해 새 컬럼({원컬럼}_{변환명})을 만듭니다: Log/Square Root는 큰 값을 압축해 오른쪽 꼬리를 줄이고, Min-Log/Min-Square Root는 최솟값을 보정한 뒤 변환해 0 이하 값 문제를 완화합니다. 결정적 변환입니다.",
    role: "Log / Square Root / Min-Log / Min-Square Root 등 분포 변환을 적용합니다.",
    input: "DataFrame 1개 + 변환할 컬럼.",
    output: "변환된 새 컬럼이 추가된 DataFrame.",
    parameters: "transformations: {컬럼명: 변환방식(Log/Square Root/Min-Log/Min-Square Root)}",
    whenToUse: "치우친(skewed) 클레임/손해 분포를 펴서 분포 적합·검정·선형모델 가정을 만족시키고 싶을 때.",
    connections: "Statistics(진단) → TransitionData → 분포 적합/통계 모델.",
    commonErrors: "순수 log/sqrt는 0 이하 값에서 실패합니다 — 음수·0이 있으면 Min-Log/Min-Square Root를 쓰세요.",
  },
  [ModuleType.PythonScript]: {
    title: "Python Script (Advanced)",
    category: "고급 (커스텀 코드)",
    beginner:
      "기본 블록만으로 부족할 때, 직접 작성한 파이썬 코드를 끼워 넣는 '만능' 블록입니다. 들어온 표를 코드로 자유롭게 가공해 결과 표로 내보냅니다. 강력한 만큼 잘못된 코드가 들어갈 수 있어 고급 사용자용입니다.",
    analysisMethod:
      "입력 데이터가 'dataframe' 변수로 주어지고, 사용자가 만든 결과를 'scripted_data'에 담으면 그것이 출력됩니다(없으면 입력을 그대로 통과). 작성한 코드는 내보낸 전체 코드에도 그대로 포함되며, Pyodide 샌드박스에서 실행됩니다.",
    role: "사용자 정의 Python 코드를 실행합니다. 'dataframe'을 입력으로 받아 'scripted_data'로 출력합니다.",
    input: "data_in: DataFrame(코드에서 'dataframe'으로 접근).",
    output: "'scripted_data'에 담은 결과(없으면 입력 통과).",
    parameters: "code: 사용자 Python 코드(시드 고정 권장)",
    whenToUse: "기본 모듈로 표현 못 하는 커스텀 전처리·계산이 필요할 때.",
    connections: "임의 데이터 모듈 → PythonScript → 후속 분석.",
    commonErrors: "출력은 반드시 'scripted_data'에 담아야 합니다. 무한 루프·과도한 메모리는 타임아웃됩니다. 재현성을 위해 난수는 시드를 고정하세요.",
    notes: "내보낸 코드에 verbatim 삽입됩니다 — 외부 실행 시에도 동일 동작해야 합니다.",
  },
  [ModuleType.SplitData]: {
    title: "Split Data",
    category: "데이터 가공",
    beginner:
      "데이터를 '연습 문제'와 '실전 시험'으로 나누는 블록입니다. 모든 데이터로 학습한 뒤 같은 데이터로 평가하면 실력을 부풀려 보게 되므로, 일부(학습셋)로만 공부시키고 따로 둔 나머지(테스트셋)로 채점합니다.",
    analysisMethod:
      "train_test_split으로 행을 무작위로 학습/테스트 두 묶음으로 나눕니다(예: 80/20). random_state(기본 42)를 고정해 매번 같은 분할이 나오도록 재현성을 보장하며, train·test 두 포트로 나눠 내보냅니다.",
    role: "데이터를 학습/테스트로 분할합니다.",
    input: "DataFrame 1개.",
    output: "train_data_out, test_data_out 두 포트.",
    parameters: "train_size · shuffle · random_state(기본 42, 재현성)",
    whenToUse: "지도학습에서 과적합을 막고 일반화 성능을 정직하게 평가할 때.",
    connections: "전처리 → SplitData → (train)TrainModel · (test)ScoreModel/EvaluateModel.",
    commonErrors: "train·test 포트를 바꿔 연결하면 평가가 왜곡됩니다. random_state를 비우면 매 실행 결과가 달라집니다.",
  },

  // ===== DFA: Threshold / Split =====
  [ModuleType.SettingThreshold]: {
    title: "Setting Threshold",
    category: "DFA (임계값 분석)",
    beginner:
      "클레임을 '얼마 이상을 대형 손해로 볼지(임계값)' 정하는 걸 돕는 블록입니다. 여러 후보 금액선을 그어 보고, 그 선보다 큰 클레임이 몇 건·몇 %인지, 연도별로는 어떤지 표와 히스토그램으로 보여 줘서 적절한 분리선을 고르게 합니다.",
    analysisMethod:
      "지정한 여러 임계값마다 그 값을 넘는 행의 건수·비율·누적치를 계산하고, 연도별 건수와 전체 분포 히스토그램(약 50구간), 분위수 통계를 산출합니다. 데이터를 바꾸지 않고 '어디서 자르면 좋을지' 판단 근거만 만듭니다.",
    role: "여러 후보 임계값별로 초과 건수·비율·연도별 분포를 분석해 최적 분리 기준 선택을 돕습니다.",
    input: "DataFrame 1개 (대상 금액 컬럼).",
    output: "임계값별 건수/비율표 + 분위수 통계 + 연도별 카운트 + 히스토그램. (선택한 임계값을 threshold 포트로 전달 가능.)",
    parameters: "target_column(대상 금액 컬럼) · thresholds(후보 임계값 배열) · year_column(연도별 분석용)",
    whenToUse: "대형/소형 클레임을 나누는 임계값(attachment point)을 데이터에 근거해 정할 때.",
    connections: "LoadClaimData/FormatChange → SettingThreshold → SplitByThreshold(threshold 포트) / 분포 적합.",
    commonErrors: "thresholds가 데이터 범위 밖이면 모든 건수가 0 또는 전체가 됩니다 — 통계의 분위수를 참고해 후보를 정하세요.",
  },
  [ModuleType.ThresholdAnalysis]: {
    title: "Threshold Analysis",
    category: "DFA (임계값 분석)",
    beginner:
      "클레임 분포의 '꼬리(아주 큰 손해 영역)가 어디서부터 달라지는지'를 그래프로 찾아 주는 블록입니다. 평범한 손해와 극단적 대형 손해가 갈리는 지점을 통계적으로 짚어 줘서, 대형손해 모델링이나 재보험 attachment point 결정에 씁니다.",
    analysisMethod:
      "네 가지 진단을 수행합니다: ①히스토그램(구간 빈도), ②ECDF(누적분포의 기울기 변화로 상위 꼬리 식별), ③QQ-Plot(정규분포 대비 꼬리 이탈), ④평균초과그래프(Mean Excess Plot — 선형 구간이 시작되는 지점으로 GPD 적합 임계값 후보 제시). 각 그래프에서 꼬리가 두꺼워지는 후보 임계값을 표시합니다.",
    role: "경험적 분포(Histogram/ECDF/QQ)와 Mean Excess Plot으로 꼬리 거동 변화점(후보 임계값)을 식별합니다.",
    input: "DataFrame 1개 (대상 금액 컬럼).",
    output: "히스토그램·ECDF·QQ-Plot·Mean Excess Plot + 후보 임계값 + 분위수 통계.",
    parameters: "target_column(분석할 금액 컬럼)",
    whenToUse: "극단값 이론(EVT)·GPD 적합을 위한 임계값을 결정하거나, 꼬리 위험을 진단할 때.",
    connections: "LoadClaimData → ThresholdAnalysis → SettingThreshold/FitSeverityModel.",
    commonErrors: "표본이 작으면 Mean Excess Plot이 불안정해 후보가 흔들립니다 — 여러 그래프를 함께 보고 판단하세요.",
  },
  [ModuleType.AnalysisThreshold]: {
    title: "Analysis Threshold",
    category: "DFA (임계값 분석)",
    beginner:
      "클레임 데이터를 세 개의 탭으로 깊이 살펴보며 '꼬리가 바뀌는 지점'을 찾는 블록입니다. ①데이터 분포 ②경험적 분포(히스토그램·ECDF·QQ) ③평균초과그래프를 차례로 보여 줘서, 어디서부터 손해가 극단적으로 커지는지 종합 판단하게 돕습니다.",
    analysisMethod:
      "탭1은 클레임 크기별 건수 분포를, 탭2는 ECDF 기울기 변화·QQ 이탈·히스토그램 급변으로 꼬리 변화점을 표시하고, 탭3은 Mean Excess Plot의 선형 구간을 찾아 GPD 적합에 쓸 임계값 범위를 제시합니다. 데이터를 바꾸지 않는 진단용 시각화입니다.",
    role: "클레임 분포를 3탭(분포/경험적 분포/Mean Excess)으로 분석해 선형 꼬리 구간과 변화점을 식별합니다.",
    input: "DataFrame 1개 (클레임 금액 컬럼).",
    output: "분포 플롯 + 경험적 분포 3종 + Mean Excess Plot(선형 구간) + 분위수 통계.",
    parameters: "claim_column(분석할 클레임 컬럼)",
    whenToUse: "ThresholdAnalysis보다 더 상세히 꼬리 거동을 탐색하고 GPD 적합 구간을 정할 때.",
    connections: "LoadClaimData → AnalysisThreshold → SettingThreshold/FitSeverityModel.",
    commonErrors: "claim_column이 수치형이 아니면 분포 분석이 실패합니다. 0·음수가 섞이면 로그 기반 시각화가 왜곡될 수 있습니다.",
  },
  [ModuleType.SplitByThreshold]: {
    title: "Split By Threshold",
    category: "DFA (데이터 분할)",
    beginner:
      "정한 금액선을 기준으로 클레임을 '소형(아래)'과 '대형(위)' 두 갈래로 갈라 주는 블록입니다. 소형은 연도별 합계로 묶고, 대형은 개별 클레임 그대로 내보내, 각각에 알맞은 빈도/심도 모델로 따로 다룰 수 있게 합니다.",
    analysisMethod:
      "임계값보다 작은 행은 연도별로 그룹화해 금액을 합산(below 출력)하고, 임계값 이상인 행은 원래 클레임 단위 그대로 유지(above 출력)합니다. 임계값은 직접 입력하거나 SettingThreshold의 threshold 포트로 받습니다.",
    role: "클레임을 임계값 기준으로 두 출력(아래=연도별 합계, 위=개별 대형클레임)으로 분할합니다.",
    input: "DataFrame 1개 (+ 선택적으로 threshold 포트). 금액·연도 컬럼.",
    output: "below_threshold(연도별 합계 DataFrame) · above_threshold(개별 대형 클레임 DataFrame) 두 포트.",
    parameters: "threshold · amount_column · date_column(또는 year_column)",
    whenToUse: "어트리션(소형 다발) 손해와 대형 손해를 분리해 각각 다른 분포·모델로 다룰 때(재보험 layering의 사전 단계).",
    connections: "SettingThreshold(threshold) → SplitByThreshold → (below)FitAggregateModel · (above)FitSeverityModel/ApplyThreshold.",
    commonErrors: "임계값이 데이터 범위 밖이면 한쪽 출력이 비어 후속 모듈이 실패합니다. date_column이 연도화돼 있어야 below 집계가 맞습니다.",
  },
  [ModuleType.SplitByFreqServ]: {
    title: "Split By Freq-Sev",
    category: "DFA (데이터 분할)",
    beginner:
      "클레임 데이터를 '얼마나 자주 났는가(빈도)'와 '한 번에 얼마였나(심도)' 두 가지 관점으로 나눠 주는 블록입니다. 빈도는 연도별 사고 건수로, 심도는 개별 클레임 금액으로 분리해, 각각 따로 분포를 적합할 수 있게 준비합니다.",
    analysisMethod:
      "날짜(연도)별로 클레임 건수를 세어 빈도 테이블(연도, count)을 만들고, 개별 클레임 금액을 그대로 모아 심도 테이블을 만듭니다. 빈도-심도 분리는 집계 손해 모델링의 표준 전처리입니다.",
    role: "클레임을 빈도(연도별 건수)와 심도(개별 금액) 두 출력으로 분해합니다.",
    input: "DataFrame 1개 (금액·날짜 컬럼).",
    output: "frequency(연도별 건수 DataFrame) · severity(개별 클레임 금액 DataFrame) 두 포트.",
    parameters: "amount_column · date_column",
    whenToUse: "빈도·심도를 각각 분포로 적합해 합성하는 집계손해(collective risk) 모델을 만들 때.",
    connections: "LoadClaimData/FormatChange → SplitByFreqServ → (frequency)FitFrequencyModel · (severity)FitSeverityModel → SimulateFreqServ.",
    commonErrors: "date_column이 연도로 해석되지 않으면 빈도 집계가 틀어집니다 — 먼저 FormatChange를 적용하세요.",
  },

  // ===== DFA: Distribution Fitting & Simulation =====
  [ModuleType.FitAggregateModel]: {
    title: "Fit Aggregate Model",
    category: "DFA (분포 적합)",
    beginner:
      "연도별(또는 전체) '총 손해액'에 어떤 통계 분포가 가장 잘 맞는지 찾아 주는 블록입니다. 여러 후보 분포를 데이터에 맞춰 보고, 가장 잘 들어맞는 것을 자동으로 골라 줘서, 이후 미래 손해를 시뮬레이션할 토대를 만듭니다.",
    analysisMethod:
      "Lognormal·Exponential·Gamma·Pareto 등 선택한 분포들을 최대우도법으로 집계 손해액에 적합하고, AIC(작을수록 좋음)로 가장 적합한 분포를 선택합니다. QQ/PP 플롯과 적합 통계(AIC/BIC/KS)를 함께 제공합니다.",
    role: "집계 클레임 금액에 여러 분포를 적합·비교해 AIC 기준 최적 분포와 모수를 산출합니다.",
    input: "DataFrame 1개 (집계 금액 컬럼).",
    output: "분포별 적합 결과 + 최적 분포 + 모수(params, SimulateAggDist로 전달).",
    parameters: "amount_column · selected_distributions(Lognormal/Exponential/Gamma/Pareto 등)",
    whenToUse: "연도별 총 손해 같은 집계 손해의 분포를 추정해 시뮬레이션·요율 산출의 입력을 만들 때.",
    connections: "SplitByThreshold(below)/집계 데이터 → FitAggregateModel → SimulateAggDist.",
    commonErrors: "표본(연도 수)이 적으면 적합이 불안정합니다. 0/음수가 있으면 Lognormal·Gamma 적합이 실패할 수 있습니다.",
  },
  [ModuleType.SimulateAggDist]: {
    title: "Simulate Agg Table",
    category: "DFA (시뮬레이션)",
    beginner:
      "앞에서 고른 손해 분포로 '미래에 일어날 법한 총 손해를 수천 번 가상으로 뽑아 보는' 블록입니다. 주사위를 수천 번 굴리듯 시뮬레이션해, 평균은 물론 최악의 경우(상위 분위수)까지 손해가 어느 정도 분포로 나올지 보여 줍니다.",
    analysisMethod:
      "FitAggregateModel이 고른 분포·모수에서 난수를 simulation_count(기본 1000)만큼 추출해 집계 손해 표본을 만듭니다. 시드(random_state=42)를 고정해 매번 동일한 결과가 나오며, 평균·표준편차·5~99% 분위수 등 요약 통계를 산출합니다.",
    role: "적합된 집계 분포에서 몬테카를로로 손해를 추출해 분포·통계를 생성합니다(결정적).",
    input: "params(FitAggregateModel의 적합 분포·모수).",
    output: "시뮬레이션된 집계 손해 배열 + 분위수 통계. CombineLossModel로 연결 가능.",
    parameters: "simulation_count(기본 1000)/custom_count · random_state(기본 42, 결정성)",
    whenToUse: "집계 손해의 전체 분포·꼬리 위험(VaR 등)을 추정할 때.",
    connections: "FitAggregateModel → SimulateAggDist → CombineLossModel.",
    commonErrors: "시뮬레이션 횟수가 적으면 상위 분위수(꼬리) 추정이 부정확합니다. params 포트가 비면 실행되지 않습니다.",
  },
  [ModuleType.FitFrequencyModel]: {
    title: "Fit Frequency Model",
    category: "DFA (분포 적합)",
    beginner:
      "'사고가 1년에 몇 번 나는가(빈도)'에 가장 잘 맞는 분포를 찾아 주는 블록입니다. 연도별 사고 건수에 카운트 분포를 맞춰 보고, 흩어짐 정도까지 따져 적절한 분포를 골라, 미래 사고 횟수를 시뮬레이션할 준비를 합니다.",
    analysisMethod:
      "연도별 클레임 건수에 Poisson과 Negative Binomial을 적합하고(주로 적률법), 평균·분산·분산/평균비(과대분산 정도)와 AIC/BIC로 비교해 적합한 분포를 고릅니다. 분산이 평균보다 크면 Negative Binomial이 선택됩니다.",
    role: "연도별 빈도(건수)에 Poisson/Negative Binomial을 적합·비교해 최적 빈도 분포와 모수를 산출합니다.",
    input: "빈도 DataFrame (연도별 건수 컬럼).",
    output: "분포별 적합 결과 + frequency_params(type·lambda/n·p 등, SimulateFreqServ로 전달).",
    parameters: "count_column · selected_frequency_types(Poisson/NegativeBinomial 등)",
    whenToUse: "빈도-심도 합성 모델의 빈도 부분을 추정할 때.",
    connections: "SplitByFreqServ(frequency) → FitFrequencyModel → SimulateFreqServ.",
    commonErrors: "연도 수가 적으면 적합이 불안정합니다. count_column이 정수 건수가 아니면 결과가 왜곡됩니다.",
  },
  [ModuleType.FitSeverityModel]: {
    title: "Fit Severity Model",
    category: "DFA (분포 적합)",
    beginner:
      "'한 번 사고가 나면 얼마나 큰 손해인가(심도)'에 가장 잘 맞는 분포를 찾아 주는 블록입니다. 개별 클레임 금액에 여러 분포를 맞춰 보고, 특히 드물지만 큰 손해(꼬리)를 잘 설명하는 분포를 골라 시뮬레이션의 토대를 만듭니다.",
    analysisMethod:
      "개별 클레임 금액에 Normal·Lognormal·Pareto·Gamma·Exponential·Weibull·GeneralizedPareto·Burr 등 여러 분포를 scipy로 적합하고 AIC로 최적을 선택합니다. QQ/PP 플롯과 KS 통계로 적합도를 확인합니다.",
    role: "개별 클레임 금액(심도)에 8종 분포를 적합·비교해 AIC 기준 최적 분포와 모수를 산출합니다.",
    input: "심도 DataFrame (개별 클레임 금액 컬럼).",
    output: "분포별 적합 결과 + 최적 분포 + severity_params(SimulateFreqServ로 전달).",
    parameters: "amount_column · selected_severity_types(Lognormal/Pareto/Gamma/Weibull/GPD/Burr 등)",
    whenToUse: "빈도-심도 합성의 심도 부분을 추정하거나, 대형 손해의 꼬리 분포(GPD 등)를 적합할 때.",
    connections: "SplitByFreqServ(severity)/ThresholdAnalysis → FitSeverityModel → SimulateFreqServ.",
    commonErrors: "0/음수 금액은 Lognormal·Pareto 적합을 깨뜨립니다. 꼬리가 두꺼우면 Pareto/GPD/Burr를 후보에 포함하세요.",
  },
  [ModuleType.SimulateFreqServ]: {
    title: "Simulate Freq-Sev Table",
    category: "DFA (시뮬레이션)",
    beginner:
      "'1년에 몇 번 날까(빈도) × 한 번에 얼마일까(심도)'를 결합해 미래 총 손해를 수천 번 가상으로 만들어 보는 블록입니다. 먼저 사고 횟수를 뽑고, 그만큼 개별 손해액을 뽑아 더하는 과정을 반복해, 한 해 총 손해가 어떤 분포로 나올지 보여 줍니다.",
    analysisMethod:
      "각 시뮬레이션마다 빈도 분포에서 사고 건수 N을 뽑고, 심도 분포에서 N개의 손해액을 뽑아 합산하는 집단위험(collective risk) 몬테카를로를 simulation_count(기본 1000)만큼 반복합니다. 시드 고정으로 결정적이며, DFA 형식(집계 통계)과 XoL 형식(개별 사고) 둘 다 출력할 수 있습니다.",
    role: "빈도·심도 분포를 결합해 집계 손해를 몬테카를로로 시뮬레이션합니다(결정적).",
    input: "frequency_params + severity_params (Fit* 모듈 출력).",
    output: "집계 손해 분포·통계(DFA 형식) 또는 개별 사고 테이블(XoL 형식). CombineLossModel로 연결.",
    parameters: "frequency_type · severity_type · simulation_count(기본 1000)/custom_count · random_state(42) · output_format(dfa/xol)",
    whenToUse: "빈도와 심도를 따로 모델링한 뒤 합쳐 집계 손해 분포를 만들 때(집계법 대안).",
    connections: "FitFrequencyModel + FitSeverityModel → SimulateFreqServ → CombineLossModel / XoL 계산.",
    commonErrors: "빈도·심도 모수 포트를 둘 다 연결해야 합니다. 시뮬레이션 횟수가 적으면 꼬리 추정이 부정확합니다.",
  },
  [ModuleType.CombineLossModel]: {
    title: "Combine Loss Model",
    category: "DFA (시뮬레이션)",
    beginner:
      "서로 다른 방법(집계 시뮬레이션과 빈도-심도 시뮬레이션)으로 만든 손해를 '하나로 합쳐 총 위험을 보는' 블록입니다. 합친 분포에서 최악의 경우 얼마까지 손해날 수 있는지(VaR), 그 선을 넘는 경우 평균은 얼마인지(TVaR)를 계산합니다.",
    analysisMethod:
      "두 시뮬레이션 결과 배열을 시뮬레이션별로 더해(combined[i] = aggDist[i] + freqServ[i]) 종합 손해 분포를 만들고, 평균·표준편차와 함께 여러 신뢰수준(90/95/99/99.5/99.9%)의 VaR(분위수)과 TVaR(초과분 평균)을 산출합니다.",
    role: "두 손해 시뮬레이션 결과를 결합해 종합 분포와 VaR/TVaR을 산출합니다.",
    input: "agg_dist + freq_serv (SimulateAggDist·SimulateFreqServ의 손해 배열).",
    output: "종합 손해 통계 + 신뢰수준별 VaR/TVaR + 분위수.",
    parameters: "(별도 파라미터 없음 — 두 입력 포트를 합산)",
    whenToUse: "복수 경로로 추정한 손해를 합쳐 자본요건·꼬리위험(VaR/TVaR)을 평가할 때.",
    connections: "SimulateAggDist + SimulateFreqServ → CombineLossModel.",
    commonErrors: "두 입력 배열 길이(시뮬레이션 횟수)가 맞아야 원소별 합산이 됩니다. 분포 간 의존(상관)을 무시하면 꼬리 위험이 과소평가될 수 있습니다.",
  },

  // ===== Reinsurance: Distribution / Exposure / XoL =====
  [ModuleType.FitLossDistribution]: {
    title: "Fit Loss Distribution",
    category: "재보험 (분포)",
    beginner:
      "손해 금액에 Pareto나 Lognormal 같은 '꼬리가 두꺼운 분포'를 맞춰 주는 블록입니다. 대형 손해를 잘 설명하는 분포 모수를 추정해, 이후 노출곡선(exposure curve)과 XoL 요율 계산의 기초로 씁니다.",
    analysisMethod:
      "지정한 손해 컬럼에 Pareto 또는 Lognormal을 scipy로 적합해 (shape, loc, scale) 모수를 추정합니다. 보험·재보험에서 흔한 두꺼운 꼬리를 표현하는 데 적합합니다.",
    role: "손해 데이터에 Pareto/Lognormal을 적합해 분포 모수를 산출합니다.",
    input: "DataFrame 1개 (손해 컬럼).",
    output: "적합된 분포 모수(distribution + parameters). GenerateExposureCurve로 연결.",
    parameters: "loss_column · distribution_type(Pareto / Lognormal)",
    whenToUse: "노출곡선·XoL 레이어 요율을 분포 기반으로 산출하기 전 손해 분포를 추정할 때.",
    connections: "LoadClaimData → FitLossDistribution → GenerateExposureCurve → PriceXoLLayer.",
    commonErrors: "0/음수 손해가 있으면 적합이 실패합니다. 꼬리가 매우 두꺼우면 Pareto가 더 적절합니다.",
  },
  [ModuleType.GenerateExposureCurve]: {
    title: "Generate Exposure Curve",
    category: "재보험 (분포)",
    beginner:
      "'손해가 특정 금액선까지 쌓일 때 전체 손해의 몇 %가 그 아래에 있는가'를 곡선으로 그려 주는 블록입니다. 이 노출곡선(exposure curve)은 재보험에서 어느 구간(layer)이 전체 손해의 얼마를 차지하는지 가늠하는 기본 도구입니다.",
    analysisMethod:
      "적합된 손해 분포의 누적분포(CDF)를 0부터 최대 손해의 약 2배까지 100개 보유점(retention)에 대해 보간해, 각 보유점까지의 누적 손해 비율(loss %)을 계산합니다. 결과는 (retention, loss_pct) 쌍의 곡선입니다.",
    role: "적합 분포로부터 보유금액(retention) 대비 누적 손해비율 곡선을 생성합니다.",
    input: "FitLossDistribution의 분포 모수.",
    output: "노출곡선(retention, loss_pct) + 총 기대손해. PriceXoLLayer로 연결.",
    parameters: "(분포 모수에서 자동 계산)",
    whenToUse: "분포 기반(exposure rating)으로 XoL 레이어 요율을 산출하기 전 곡선을 만들 때.",
    connections: "FitLossDistribution → GenerateExposureCurve → PriceXoLLayer.",
    commonErrors: "분포 모수가 비현실적이면(과대 꼬리) 곡선이 왜곡됩니다 — 적합 품질을 먼저 확인하세요.",
  },
  [ModuleType.PriceXoLLayer]: {
    title: "Price XoL Layer",
    category: "재보험 (분포)",
    beginner:
      "재보험의 한 '구간(layer)'에 대한 보험료를 노출곡선으로 계산하는 블록입니다. '보유금액(deductible) 초과분 중 한도(limit)까지'에 해당하는 손해가 전체의 몇 %인지 곡선에서 읽어, 기대손해와 보험료(요율)를 산출합니다.",
    analysisMethod:
      "노출곡선에서 retention과 retention+limit 두 지점의 누적 손해비율을 보간해 그 차이를 레이어 손해비율로 삼고, 레이어 손해비율 × 총손해 = 기대 레이어 손해를 구합니다. 여기에 로딩(loading factor)을 곱해 보험료와 ROL(Rate on Line)을 계산합니다.",
    role: "노출곡선으로 XoL 레이어의 기대손해·보험료·ROL을 산출합니다(exposure rating).",
    input: "GenerateExposureCurve의 노출곡선.",
    output: "기대 레이어 손해 + 보험료 + ROL(Rate on Line).",
    parameters: "retention(보유금액) · limit(한도) · loading_factor(로딩)",
    whenToUse: "경험 데이터가 부족하거나 분포 기반 요율(exposure rating)이 필요할 때.",
    connections: "GenerateExposureCurve → PriceXoLLayer.",
    commonErrors: "retention/limit이 곡선의 금액 범위를 벗어나면 보간이 부정확합니다. loading_factor는 1보다 큰 값(예: 1.1)로 비용·마진을 반영합니다.",
  },
  [ModuleType.ApplyThreshold]: {
    title: "Apply Threshold",
    category: "재보험 (경험)",
    beginner:
      "정한 금액 이상의 '대형 클레임만 남기고' 작은 것은 걸러 내는 필터 블록입니다. 재보험은 보통 큰 손해만 책임지므로, 분석 대상을 대형 클레임으로 좁힐 때 씁니다.",
    analysisMethod:
      "지정한 손해 컬럼이 임계값 이상인 행만 골라냅니다(df[df[col] >= threshold]). 열은 그대로 두고 행만 줄이는 결정적 필터입니다.",
    role: "임계값 이상 클레임만 추출합니다(소형 손해 제거).",
    input: "DataFrame 1개 (손해 컬럼).",
    output: "임계값 이상 클레임만 남은 DataFrame.",
    parameters: "threshold · amount_column",
    whenToUse: "경험 기반 XoL 분석에서 보유금액 미만의 소형 클레임을 제외할 때.",
    connections: "LoadClaimData → ApplyThreshold → DefineXolContract/CalculateCededLoss.",
    commonErrors: "임계값이 너무 높으면 0행이 남아 후속 모듈이 실패합니다 — 필터 후 건수를 확인하세요.",
  },
  [ModuleType.DefineXolContract]: {
    title: "XoL Contract",
    category: "재보험 (경험)",
    beginner:
      "재보험 계약의 조건(보유금액·한도·복원 횟수 등)을 '정의해 담아 두는' 블록입니다. 실제 계산을 하기보다, 이후 손해 계산·요율 산출 모듈들이 참고할 계약 조건을 한곳에 정리해 두는 역할입니다.",
    analysisMethod:
      "보유금액(deductible)·한도(limit)·복원 횟수(reinstatements)·집계 공제(aggDeductible)·사업비율(expenseRatio)·복원료율 등을 저장하고, 각 복원에 대한 복원보험료(= 한도 × 복원료율)를 계산해 둡니다. 이 계약 조건을 contract 포트로 내보냅니다.",
    role: "XoL 재보험 계약 조건(deductible/limit/reinstatements 등)을 정의하고 복원보험료를 계산합니다.",
    input: "(계약 정의) UI 파라미터.",
    output: "계약 조건 객체(contract) + 복원보험료. CalculateCededLoss/PriceXolContract/XolCalculator로 연결.",
    parameters: "deductible(보유) · limit(한도) · reinstatements(복원 수) · aggDeductible(집계공제) · expenseRatio(사업비%) · defaultReinstatementRate(복원료%) · yearRates",
    whenToUse: "경험 기반 XoL 손해·요율 계산에 앞서 계약 구조를 명시할 때.",
    connections: "→ (contract)CalculateCededLoss/XolCalculator/PriceXolContract.",
    commonErrors: "이 모듈만으로는 손해가 계산되지 않습니다 — contract 포트를 계산 모듈에 연결해야 합니다. expenseRatio·요율은 % 단위로 입력합니다.",
  },
  [ModuleType.CalculateCededLoss]: {
    title: "Calculate Ceded Loss",
    category: "재보험 (경험)",
    beginner:
      "각 클레임에서 '재보험사가 책임지는 금액(출재손해)'을 계산해 주는 블록입니다. 보유금액을 넘는 부분 중 한도까지만 재보험이 부담하므로, 클레임마다 그 몫을 따로 떼어 새 열로 적어 줍니다.",
    analysisMethod:
      "각 클레임에 대해 출재손해 = min(한도, max(0, 손해 − 보유금액))을 계산합니다. 즉 보유금액 이하면 0, 초과분은 한도까지만 인정합니다. 계약 조건은 contract 포트에서 받습니다.",
    role: "계약 조건(보유/한도)으로 클레임별 출재손해(ceded loss)를 계산합니다.",
    input: "contract(계약 조건) + DataFrame(손해 컬럼).",
    output: "'ceded_loss' 컬럼이 추가된 DataFrame.",
    parameters: "loss_column · (contract에서 deductible·limit)",
    whenToUse: "경험 데이터로 재보험사가 부담할 손해를 산출할 때(burning cost 요율의 입력).",
    connections: "DefineXolContract(contract) + 클레임 데이터 → CalculateCededLoss → PriceXolContract.",
    commonErrors: "contract·data 포트를 둘 다 연결해야 합니다. loss_column이 보정 전 금액이면 결과가 달라지니 ApplyInflation 적용 여부를 확인하세요.",
  },
  [ModuleType.PriceXolContract]: {
    title: "Price XoL Contract",
    category: "재보험 (경험)",
    beginner:
      "과거 출재손해 경험으로 재보험 계약의 보험료를 매기는 블록입니다(버닝코스트 방식). 연도별 출재손해의 평균에 변동성 여유를 더하고, 사업비를 얹어 최종 보험료를 산출합니다.",
    analysisMethod:
      "출재손해를 연도별로 합산해 평균과 표준편차를 구하고, 순보험료 = 평균 연손해 + (연손해 표준편차 × 변동성 로딩%)으로 변동 여유를 반영합니다. 총보험료 = 순보험료 / (1 − 사업비율)로 사업비를 더합니다.",
    role: "연도별 출재손해 경험으로 버닝코스트 기반 XoL 보험료를 산출합니다.",
    input: "contract + 출재손해가 계산된 DataFrame(연도·ceded_loss 컬럼).",
    output: "순보험료·총보험료(gross premium).",
    parameters: "year_column · ceded_loss_column · volatility_loading(변동성 로딩 %) · (contract의 expenseRatio)",
    whenToUse: "충분한 경험 데이터로 경험 기반(experience rating) 요율을 산출할 때.",
    connections: "CalculateCededLoss → PriceXolContract.",
    commonErrors: "연도 수가 적으면 변동성 추정이 불안정합니다. ceded_loss_column·year_column을 정확히 지정하세요.",
  },
  [ModuleType.XolCalculator]: {
    title: "XoL Calculator",
    category: "재보험 (계산)",
    beginner:
      "계약 조건을 클레임에 적용해 '클레임마다 XoL로 회수되는 금액'을 계산해 주는 블록입니다. 보유금액 초과분을 한도 내에서 잘라, 각 클레임의 XoL 회수액과 전체 합계·비율을 보여 줍니다.",
    analysisMethod:
      "각 클레임에 대해 XoL 회수액 = max(min(손해 − 보유금액, 한도), 0)을 계산해 'XoL Claim(Incl. Limit)' 컬럼으로 추가하고, 총 회수액과 원손해 대비 XoL 비율을 산출합니다. 계약 조건은 contract 포트에서 받습니다.",
    role: "계약 조건으로 클레임별 XoL 회수액과 합계·비율을 계산합니다.",
    input: "contract + DataFrame(클레임 컬럼).",
    output: "'XoL Claim(Incl. Limit)' 컬럼이 추가된 DataFrame + 합계·비율.",
    parameters: "claim_column · (contract에서 deductible·limit)",
    whenToUse: "계약 조건별 XoL 손해를 클레임 단위로 산출해 XolPricing 입력으로 쓸 때.",
    connections: "DefineXolContract(contract) + 클레임 데이터 → XolCalculator → XolPricing.",
    commonErrors: "contract·data 포트를 둘 다 연결해야 합니다. CalculateCededLoss와 계산식은 사실상 동일하니(min/max layering) 워크플로에 맞는 쪽을 쓰세요.",
  },
  [ModuleType.XolPricing]: {
    title: "XoL Pricing",
    category: "재보험 (계산)",
    beginner:
      "XoL Calculator로 구한 회수액을 받아 '최종 재보험 요율·보험료'로 정리해 주는 블록입니다. 여러 계약/계산기 결과를 모아 순보험료에 사업비를 얹은 총보험료를 산출합니다.",
    analysisMethod:
      "XoL 회수액의 평균·표준편차에서 순보험료(기대 XoL 손해 + 변동성 여유)를 구하고, 사업비율을 반영해 총보험료를 계산합니다. 복수의 XolCalculator 입력을 각각 처리해 계약별 결과를 함께 제공합니다.",
    role: "XolCalculator 결과로 XoL 순/총보험료와 요율을 산출합니다.",
    input: "XolCalculator의 결과(계약별 XoL 회수).",
    output: "계약별 순보험료·총보험료·요율 + 합산 결과.",
    parameters: "expenseRate(사업비율 %)",
    whenToUse: "XoL 손해 계산 결과를 실제 청구 가능한 요율·보험료로 변환할 때.",
    connections: "XolCalculator → XolPricing.",
    commonErrors: "입력이 비어 있으면 산출되지 않습니다. expenseRate는 % 단위로, 1 미만 분모(1−사업비)가 되도록 합리적 범위로 입력하세요.",
  },
  [ModuleType.ExperienceModel]: {
    title: "Experience Model",
    category: "재보험 (경험)",
    beginner:
      "계약 조건을 클레임 데이터에 적용해 'XoL 손해 경험'을 계산하는 블록입니다. XoL Calculator와 비슷하게 보유·한도를 적용하되, 경험 기반 분석 흐름에서 데이터에 XoL 결과를 붙여 내보냅니다.",
    analysisMethod:
      "계약의 보유금액·한도를 각 클레임에 적용해 XoL 손해(layering: max(min(손해−보유, 한도), 0))를 계산하고, 그 결과를 데이터에 추가해 후속 경험 분석으로 넘깁니다.",
    role: "계약 조건을 클레임에 적용해 XoL 손해 경험을 계산·부착합니다.",
    input: "contract + DataFrame(클레임 컬럼).",
    output: "XoL 손해가 계산된 DataFrame.",
    parameters: "claim_column · (contract에서 deductible·limit)",
    whenToUse: "경험 기반 워크플로에서 계약 적용 결과를 데이터에 붙여 분석할 때.",
    connections: "DefineXolContract(contract) + 클레임 데이터 → ExperienceModel → 경험 분석/요율.",
    commonErrors: "contract·data 포트를 둘 다 연결해야 합니다. XolCalculator와 목적이 겹치니 파이프라인 일관성을 유지하세요.",
  },

  // ===== Supervised Learning Models =====
  [ModuleType.LinearRegression]: {
    title: "Linear Regression",
    category: "지도학습 (회귀)",
    beginner:
      "여러 입력값을 조합해 '숫자(손해액·요율 등)'를 예측하는 가장 기본적인 블록입니다. 각 입력이 결과에 얼마씩 기여하는지를 직선 식으로 학습해, 해석이 쉬워 가장 먼저 시도하는 출발 모델로 좋습니다.",
    analysisMethod:
      "타깃 = a + b1·x1 + b2·x2 + … 꼴의 직선식 계수를 최소제곱법으로 찾습니다. 모델 정의만 만들고 실제 적합은 TrainModel에서 수행합니다.",
    role: "연속형 타깃을 선형 결합으로 예측합니다. 해석성 우선 시 첫 베이스라인.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 모델 객체. TrainModel로 연결.",
    whenToUse: "타깃이 연속형이고 해석 가능한 베이스라인이 필요할 때.",
    connections: "(model_in)→ TrainModel ←(data)SplitData.train → ScoreModel → EvaluateModel.",
    commonErrors: "이 모듈만으로는 학습되지 않습니다 — 반드시 TrainModel에 연결하세요.",
  },
  [ModuleType.LogisticRegression]: {
    title: "Logistic Regression",
    category: "지도학습 (분류)",
    beginner:
      "'예/아니오'를 맞히는 분류의 기본 블록입니다. 이름은 회귀지만 실제로는 확률을 계산해 분류하며(예: 클레임 발생 여부), 해석이 쉬워 분류 문제의 첫 출발 모델로 많이 씁니다.",
    analysisMethod:
      "입력의 선형 결합을 시그모이드 함수에 넣어 0~1 확률로 변환하고, 보통 0.5를 넘으면 양성으로 분류합니다. 모델 정의만 만들고 적합은 TrainModel에서 수행합니다.",
    role: "이진/다항 분류기. 정규화(L1/L2) 지원.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 분류 모델.",
    whenToUse: "분류의 해석 가능한 베이스라인이 필요할 때.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "라벨이 연속형이면 분류가 되지 않습니다. 클래스 불균형 시 지표 해석에 주의.",
  },
  [ModuleType.PoissonRegression]: {
    title: "Poisson Regression",
    category: "지도학습 (카운트)",
    beginner:
      "'몇 번 일어났는가(건수)'를 예측하는 블록입니다. 보험 클레임 횟수처럼 0,1,2…로 세는 값에 맞춰진 모델로, 음수가 나오지 않게 설계돼 있습니다.",
    analysisMethod:
      "카운트의 로그 평균을 입력의 선형 결합으로 모델링하는 Poisson GLM(log link)입니다. '평균=분산'을 가정합니다.",
    role: "건수 데이터(클레임 횟수) 예측. (Deprecated: 추론 중심이면 Poisson Model 권장.)",
    input: "(모델 정의) 하이퍼파라미터만.",
    output: "학습되지 않은 GLM 모델.",
    whenToUse: "타깃이 음이 아닌 정수(횟수)일 때.",
    connections: "→ TrainModel → ScoreModel/EvaluateModel.",
    commonErrors: "과대분산이면 표준오차가 과소추정됩니다 — NegativeBinomial을 고려하세요.",
    notes: "Deprecated — 계수 추론·offset이 필요하면 전통 통계의 Poisson Model을 쓰세요.",
  },
  [ModuleType.NegativeBinomialRegression]: {
    title: "Negative Binomial Regression",
    category: "지도학습 (카운트)",
    beginner:
      "건수를 예측하되 값이 '들쭉날쭉 심하게 흩어진' 경우에 쓰는 블록입니다. 보통의 카운트 모델보다 흩어짐을 더 유연하게 다룹니다.",
    analysisMethod:
      "Poisson에 분산을 키우는 모수(α)를 더한 음이항 GLM으로, 과대분산을 흡수해 표준오차를 더 정확히 추정합니다.",
    role: "과대분산 카운트 데이터 예측. (Deprecated: 추론 중심이면 Negative Binomial Model 권장.)",
    input: "(모델 정의) 하이퍼파라미터만.",
    output: "학습되지 않은 NB GLM 모델.",
    whenToUse: "카운트 타깃의 분산이 평균보다 뚜렷이 클 때.",
    connections: "→ TrainModel → ScoreModel/EvaluateModel.",
    commonErrors: "과대분산이 없으면 Poisson이 더 단순·안정적입니다.",
    notes: "Deprecated — 계수 추론이 목적이면 전통 통계의 Negative Binomial Model을 쓰세요.",
  },
  [ModuleType.DecisionTree]: {
    title: "Decision Tree",
    category: "지도학습",
    beginner:
      "'스무고개'처럼 질문을 가지치며 결론에 도달하는 블록입니다. 규칙이 그림으로 보여 사람이 이해하기 쉽지만, 너무 깊어지면 본 데이터만 외우는 과적합에 빠집니다.",
    analysisMethod:
      "각 단계에서 데이터를 가장 잘 갈라 주는 컬럼·기준값을 골라(분류는 지니/엔트로피, 회귀는 분산 감소) 가지를 뻗습니다. max_depth 등으로 가지치기를 제한해 과적합을 막습니다. 적합은 TrainModel에서 수행합니다.",
    role: "단일 의사결정 트리. 해석성 강점, 단일 트리는 과적합 경향.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 트리 모델.",
    parameters: "model_purpose(classification/regression) · criterion · max_depth · min_samples_split/leaf",
    whenToUse: "규칙 기반의 해석 가능한 모델이 필요하거나 앙상블의 베이스라인으로.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "max_depth를 제한하지 않으면 쉽게 과적합합니다.",
  },
  [ModuleType.RandomForest]: {
    title: "Random Forest",
    category: "지도학습 (앙상블)",
    beginner:
      "'여러 명의 전문가에게 물어 다수결로 결정하는' 블록입니다. 조금씩 다른 트리를 수백 그루 만들어 분류는 투표로, 회귀는 평균으로 합칩니다. 한 그루보다 훨씬 안정적입니다.",
    analysisMethod:
      "데이터·변수를 무작위로 조금씩 다르게 뽑아 많은 트리를 학습하고(bagging) 예측을 평균/투표로 모읍니다(sklearn RandomForest, random_state=42로 결정적). 변수중요도를 제공하며, 적합은 TrainModel에서 실제 sklearn 모델로 수행합니다.",
    role: "다수 트리의 평균/투표. 강건한 일반화 + 변수중요도.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 RF 모델 + (학습 후) 변수중요도.",
    whenToUse: "표 형식 데이터에서 튜닝 없이 좋은 성능이 필요한 기본 선택지.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "트리가 많으면 학습이 느려집니다(Pyodide).",
  },
  [ModuleType.GradientBoosting]: {
    title: "Gradient Boosting",
    category: "지도학습 (앙상블)",
    beginner:
      "'앞 사람의 실수를 다음 사람이 고쳐 나가는' 방식으로 점점 똑똑해지는 블록입니다. 표 형식 데이터에서 종종 가장 높은 정확도를 내며, 클레임 빈도·심도 예측에 강합니다.",
    analysisMethod:
      "앞선 트리들의 예측 오차(잔차)를 줄이는 새 약한 트리를 순차적으로 더해 가는 부스팅 기법입니다(sklearn GradientBoosting, random_state=42로 결정적). 적합은 TrainModel에서 실제 모델로 수행합니다.",
    role: "약한 트리를 순차 부스팅. 강력한 예측 성능 + 변수중요도.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 GBM 모델 + (학습 후) 변수중요도.",
    parameters: "model_purpose · n_estimators(100) · learning_rate(0.1) · max_depth(3) · random_state=42",
    whenToUse: "정확도가 중요한 클레임 빈도·심도 예측에. RandomForest보다 종종 더 높은 성능.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "learning_rate가 크면 과적합·불안정. 순차 학습이라 RF보다 느릴 수 있습니다.",
  },
  [ModuleType.SVM]: {
    title: "Support Vector Machine",
    category: "지도학습",
    beginner:
      "두 집단 사이에 '가장 넓은 안전 간격을 두는 경계선'을 긋는 블록입니다. 커널로 휘어진 경계도 만들 수 있습니다.",
    analysisMethod:
      "마진이 최대가 되는 초평면을 찾고, 커널(rbf 등)로 비선형 경계도 표현합니다(sklearn SVC/SVR, probability·random_state=42). 거리 기반이라 스케일링이 필수입니다.",
    role: "마진 최대화 기반 분류/회귀. 커널로 비선형 확장.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 SVM 모델.",
    whenToUse: "표본이 크지 않고 경계가 복잡할 때. 스케일링 필수.",
    connections: "NormalizeData → SplitData → TrainModel(+SVM) → ScoreModel → EvaluateModel.",
    commonErrors: "대용량에서 매우 느립니다. 스케일링을 빠뜨리면 성능이 급락합니다.",
  },
  [ModuleType.LinearDiscriminantAnalysis]: {
    title: "Linear Discriminant Analysis (LDA)",
    category: "지도학습 (분류)",
    beginner:
      "여러 부류를 '가장 잘 갈라 보이는 방향'을 찾아 분류하는 블록입니다. 분류와 동시에 차원을 줄여 시각화하는 데도 쓸 수 있습니다.",
    analysisMethod:
      "모든 클래스의 공분산이 같다는 가정 아래, 집단 간 분산은 키우고 집단 내 분산은 줄이는 선형 판별 방향을 구해 결정경계를 만듭니다(sklearn LDA).",
    role: "공분산 동일 가정 하 선형 결정경계 학습 + 차원축소.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 LDA 모델.",
    whenToUse: "분류 + 시각화용 저차원 투영을 동시에 원할 때.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "클래스별 공분산이 크게 다르면 가정이 깨집니다. 피처 수>표본이면 불안정.",
  },
  [ModuleType.NaiveBayes]: {
    title: "Naive Bayes",
    category: "지도학습 (분류)",
    beginner:
      "확률을 이용해 '가장 그럴듯한 부류'를 빠르게 고르는 블록입니다. 각 단서를 따로 본 뒤 확률을 곱해 합산하는 단순한 방식이라 매우 빠릅니다.",
    analysisMethod:
      "베이즈 정리를 써서, 각 특징이 서로 독립이라는 가정 아래 클래스별 사후확률을 계산하고 가장 높은 클래스로 분류합니다(sklearn GaussianNB 등).",
    role: "독립 가정 기반 베이즈 분류. 빠른 베이스라인.",
    input: "(모델 정의) 하이퍼파라미터만. 학습은 TrainModel.",
    output: "학습되지 않은 NB 모델.",
    whenToUse: "빠른 베이스라인이 필요하거나 고차원 희소 피처일 때.",
    connections: "→ TrainModel → ScoreModel → EvaluateModel.",
    commonErrors: "강한 피처 상관이 있으면 독립 가정 위반으로 확률 보정이 나빠집니다.",
  },
  [ModuleType.KNN]: {
    title: "K-Nearest Neighbors",
    category: "지도학습",
    beginner:
      "'주변에서 나와 가장 비슷한 K명에게 물어보고 따라가는' 블록입니다. 새 데이터가 들어오면 가장 가까운 이웃 K개를 찾아 분류는 다수결, 회귀는 평균으로 답을 정합니다.",
    analysisMethod:
      "예측할 점에서 모든 학습 데이터까지 거리를 재 가장 가까운 K개 이웃을 골라 라벨을 다수결/평균으로 모읍니다(sklearn KNeighbors). 거리 기반이라 스케일링이 중요합니다.",
    role: "가까운 K개 이웃의 다수결/평균으로 예측. 학습 없음, 추론 비용 높음.",
    input: "(모델 정의) 하이퍼파라미터만.",
    output: "학습 데이터를 보유한 KNN 모델.",
    parameters: "n_neighbors(K) · weights · 거리 척도",
    whenToUse: "경계가 비선형이고 표본이 작을 때의 간단한 비모수 베이스라인.",
    connections: "NormalizeData → SplitData → TrainModel(+KNN) → ScoreModel → EvaluateModel.",
    commonErrors: "스케일링이 없으면 단위 큰 변수가 거리를 지배합니다. 표본이 크면 추론이 느립니다.",
  },

  // ===== Model Operations (ML) =====
  [ModuleType.TrainModel]: {
    title: "Train Model",
    category: "모델 연산",
    beginner:
      "모델에게 '예제 문제와 정답을 보여 주며 공부시키는' 블록입니다. 앞에서 고른 모델 종류와 학습 데이터를 받아 입력과 정답의 관계를 익히게 합니다.",
    analysisMethod:
      "model_in으로 들어온 모델 정의에 학습 데이터를 넣어 model.fit(X_train, y_train)을 실행합니다. feature_columns로 입력 열을, label_column으로 정답 열을 지정하며, RandomForest·GradientBoosting 등은 실제 sklearn 모델로 결정적으로 적합됩니다.",
    role: "모델 정의(model)와 학습 데이터를 받아 model.fit(X, y)를 실행합니다.",
    input: "model_in: 모델 객체 + data_in: X_train·y_train(SplitData.train).",
    output: "학습된 모델(trained_model).",
    parameters: "feature_columns · label_column",
    whenToUse: "지도학습 모델 정의를 실제로 적합시킬 때.",
    connections: "(model_in)모델정의 · (data_in)SplitData.train → TrainModel → ScoreModel/EvaluateModel.",
    commonErrors: "model_in·data_in 둘 다 연결해야 합니다. label_column 미지정 시 학습 실패. 피처에 결측·문자열이 남으면 fit 오류.",
  },
  [ModuleType.ScoreModel]: {
    title: "Score Model",
    category: "모델 연산",
    beginner:
      "공부를 마친 모델에게 '실제 문제를 풀게 해' 답안을 받아 오는 블록입니다. 학습된 모델과 새 데이터를 넣으면 각 행의 예측값(또는 확률)을 'Predict' 열로 붙여 줍니다.",
    analysisMethod:
      "학습된 모델로 입력 데이터에 model.predict(X)(분류는 필요 시 predict_proba)를 적용해 예측을 산출합니다. 학습 때 쓴 피처·인코딩·스케일링 기준이 입력에도 동일해야 결과가 맞습니다.",
    role: "학습된 모델로 새 데이터의 예측값/확률을 계산합니다.",
    input: "학습된 모델(trained_model) + X_test(또는 신규 데이터).",
    output: "예측 결과 DataFrame(원본 + Predict 컬럼).",
    whenToUse: "학습된 모델로 테스트셋/실데이터를 예측할 때.",
    connections: "TrainModel → ScoreModel ←(data)SplitData.test → EvaluateModel.",
    commonErrors: "학습에 쓰인 피처 컬럼과 추론 데이터의 컬럼이 일치해야 합니다.",
  },
  [ModuleType.EvaluateModel]: {
    title: "Evaluate Model",
    category: "모델 연산",
    beginner:
      "모델의 '시험 성적표'를 만들어 주는 블록입니다. 예측값과 실제 정답을 맞대어 얼마나 잘 맞혔는지 여러 점수로 보여 줘서, 모델 간 비교에 씁니다.",
    analysisMethod:
      "분류는 정확도·정밀도·재현율·F1과 혼동행렬을, 회귀는 MSE·RMSE·MAE·R² 등을 계산합니다. 고정된 예측·정답으로 계산하므로 같은 입력이면 같은 점수가 재현됩니다.",
    role: "분류는 Accuracy/Precision/Recall/F1, 회귀는 MSE/RMSE/MAE/R² 등을 산출합니다.",
    input: "예측 결과(scored_data) + 실제값(label).",
    output: "지표 표 + 혼동행렬 / 잔차 plot.",
    parameters: "model_type(classification/regression) · label_column · prediction_column",
    whenToUse: "모델 성능을 정량 지표로 확인하고 비교할 때.",
    connections: "ScoreModel → EvaluateModel.",
    commonErrors: "model_type(분류/회귀)을 실제 타깃과 맞춰야 지표가 맞습니다.",
  },

  // ===== Traditional Statistics (statsmodels GLM) =====
  [ModuleType.OLSModel]: {
    title: "OLS Model",
    category: "전통 통계 모델",
    beginner:
      "직선식으로 숫자를 설명하되 '각 변수가 통계적으로 유의한지'까지 알려 주는 블록입니다. 각 요인이 결과에 얼마씩 영향을 주는지, 그게 우연이 아닌지를 p-value로 함께 보여 줍니다.",
    analysisMethod:
      "최소제곱법으로 선형회귀 계수를 추정하고, 각 계수의 t-통계량·p-value·신뢰구간과 R²를 statsmodels로 산출합니다. 정의만 만들고 실제 적합은 ResultModel이 수행합니다.",
    role: "최소제곱 선형회귀. 계수·t-stat·p-value·R² 산출.",
    input: "(모델 정의) + 데이터·변수 지정.",
    output: "정의 객체. 적합 결과는 ResultModel이 생성.",
    whenToUse: "연속형 타깃의 선형 관계와 각 변수의 유의성을 해석할 때.",
    connections: "(model_in)OLSModel → ResultModel → EvaluateStat.",
    commonErrors: "이 모듈은 정의만 합니다 — 적합 결과는 ResultModel에 연결해야 나옵니다.",
  },
  [ModuleType.LogisticModel]: {
    title: "Logistic Model",
    category: "전통 통계 모델",
    beginner:
      "'발생/비발생' 같은 이진 결과에 대해 각 요인이 발생 가능성을 몇 배로 바꾸는지(오즈비) 해석하는 블록입니다.",
    analysisMethod:
      "이진 종속변수에 로지스틱 GLM을 적합해 계수를 추정하고, 지수변환한 오즈비와 Wald p-value·신뢰구간을 statsmodels로 산출합니다. 적합은 ResultModel이 수행합니다.",
    role: "이진 로지스틱 GLM. odds ratio·Wald p-value 산출.",
    input: "(모델 정의) + 이진 종속 변수.",
    output: "정의 객체. 적합 결과는 ResultModel.",
    whenToUse: "이진 결과에 대한 변수별 오즈비를 해석할 때.",
    connections: "(model_in)LogisticModel → ResultModel.",
    commonErrors: "완전분리가 있으면 계수가 발산합니다. 종속변수는 0/1 이진이어야 합니다.",
  },
  [ModuleType.PoissonModel]: {
    title: "Poisson Model",
    category: "전통 통계 모델",
    beginner:
      "발생 '건수'를 모델링하되 각 요인이 발생률에 주는 영향을 해석하는 블록입니다. 관찰 기간이 다른 경우 노출량(exposure)을 반영해 단위 기간당 발생률로 공정하게 비교합니다.",
    analysisMethod:
      "카운트 종속변수에 Poisson GLM(log link)을 적합하고, 관찰 규모 차이는 offset으로 반영합니다. 각 계수는 발생률 비로 해석되며 p-value와 함께 산출됩니다. 적합은 ResultModel이 수행합니다.",
    role: "Poisson GLM. 카운트 데이터 + offset(노출량) 지원.",
    input: "(모델 정의) + count 종속 변수(+ offset).",
    output: "정의 객체. 적합 결과는 ResultModel.",
    parameters: "max_iter",
    whenToUse: "발생 건수를 노출량 대비로 모델링할 때(발생률).",
    connections: "(model_in)PoissonModel → ResultModel. 과대분산이면 NegativeBinomialModel.",
    commonErrors: "과대분산을 무시하면 p-value가 과소추정됩니다 — DiversionChecker/EvaluateStat로 점검.",
  },
  [ModuleType.QuasiPoissonModel]: {
    title: "Quasi-Poisson Model",
    category: "전통 통계 모델",
    beginner:
      "카운트 모델인데 값의 흩어짐이 평균보다 조금~중간 정도 큰 경우, 그 흩어짐을 '한꺼번에 보정'해 주는 블록입니다. p-value가 너무 낙관적으로 나오지 않도록 표준오차를 키웁니다.",
    analysisMethod:
      "Poisson GLM을 적합하되 분산 = φ·평균으로 가정하고, 추정한 분산팽창계수 φ로 표준오차를 보정합니다. 계수 추정값은 Poisson과 같고 유의성 판정만 보수적이 됩니다.",
    role: "과대분산 보정 Poisson. 분산이 평균의 상수배라고 가정.",
    input: "(모델 정의) + count 종속.",
    output: "정의 객체. 적합 결과는 ResultModel.",
    parameters: "max_iter",
    whenToUse: "경미~중간 과대분산을 표준오차 보정으로 다루고 싶을 때.",
    connections: "(model_in)QuasiPoissonModel → ResultModel.",
    commonErrors: "강한 과대분산은 NegativeBinomial이 더 적절합니다.",
  },
  [ModuleType.NegativeBinomialModel]: {
    title: "Negative Binomial Model",
    category: "전통 통계 모델",
    beginner:
      "흩어짐이 평균보다 뚜렷이 큰 카운트 데이터를 해석하는 전통 통계 블록입니다. Poisson으로 부족한 큰 변동을 추가 모수로 흡수해 계수 유의성을 더 정확히 평가합니다.",
    analysisMethod:
      "음이항 GLM을 적합하며 과대분산 정도 α를 데이터에서 자유롭게 추정해 표준오차·p-value를 과소추정하지 않게 보정합니다. 적합은 ResultModel이 수행합니다.",
    role: "NB GLM. 과대분산 카운트에 적합(α 자유 추정).",
    input: "(모델 정의) + count 종속.",
    output: "정의 객체. 적합 결과는 ResultModel.",
    parameters: "max_iter · disp(분산)",
    whenToUse: "카운트 타깃의 분산이 평균보다 뚜렷이 클 때.",
    connections: "(model_in)NegativeBinomialModel → ResultModel.",
    commonErrors: "과대분산이 없으면 Poisson이 더 단순합니다.",
  },
  [ModuleType.StatModels]: {
    title: "Stat Models",
    category: "전통 통계 모델",
    beginner:
      "예측 정확도보다 '각 변수가 정말 의미 있는 영향을 주는지'를 따지는 전통 통계 모델 블록으로, 특히 Gamma·Tweedie 같은 보험 손해에 맞는 고급 분포를 다룹니다. 변수별 영향력과 신뢰도(p-value)를 표로 보여 줍니다.",
    analysisMethod:
      "statsmodels의 GLM으로 Gamma(양의 연속·손해 심도)나 Tweedie(0이 섞인 손해총액) 분포를 적합하고, 각 계수의 추정값·표준오차·p-value·신뢰구간을 산출합니다. 적합은 ResultModel이 수행합니다.",
    role: "Gamma/Tweedie 등 고급 GLM을 statsmodels로 정의. 계수 신뢰구간/p-value 산출.",
    input: "(모델 정의) + 데이터.",
    output: "정의 객체. 적합 결과는 ResultModel.",
    parameters: "model: Gamma / Tweedie",
    whenToUse: "손해 심도(Gamma)나 0 포함 손해총액(Tweedie)을 GLM으로 추론할 때.",
    connections: "(model_in)StatModels → ResultModel/EvaluateStat.",
    commonErrors: "model 종류를 타깃 분포와 맞춰야 합니다(양의 연속→Gamma, 0포함 총액→Tweedie).",
  },
  [ModuleType.ResultModel]: {
    title: "Result Model",
    category: "출력",
    beginner:
      "통계 모델 정의 블록(OLS·Poisson 등)을 받아 '실제로 적합시키고 결과표를 펼쳐 주는' 블록입니다. 정의만 해 둔 모델을 여기에 연결해야 비로소 계수·p-value·적합도 같은 진짜 결과를 볼 수 있습니다.",
    analysisMethod:
      "연결된 통계 모델 정의 종류에 맞춰 statsmodels로 적합(fit)을 수행하고(상수항을 더해), 계수표(추정값·표준오차·p-value·신뢰구간)와 적합도(AIC/BIC/deviance) 진단을 종합한 결과 뷰를 만듭니다.",
    role: "통계 모델 정의를 받아 실제 적합을 수행하고 계수표·진단을 종합 출력합니다.",
    input: "model_in: 통계 모델 정의 + data_in + feature_columns·label_column.",
    output: "적합 결과(계수·p-value·적합도) 통합 뷰. EvaluateStat/PredictModel로 연결.",
    whenToUse: "OLSModel 등 statsmodels 정의 모듈의 적합 결과를 실제로 산출할 때.",
    connections: "OLSModel/PoissonModel/… → ResultModel → EvaluateStat/PredictModel.",
    commonErrors: "model_in에 정의 모듈을 연결해야 합니다. label_column 미지정 시 적합 실패.",
  },
  [ModuleType.PredictModel]: {
    title: "Predict Model",
    category: "모델 연산",
    beginner:
      "적합된 통계 모델로 새 데이터를 예측하는 블록입니다. 학습 시와 같은 변수 구성으로 맞춘 뒤 예측값을 'Predict' 열로 붙여 줍니다.",
    analysisMethod:
      "ResultModel이 적합한 statsmodels 결과를 받아 신규 입력의 피처를 정렬하고 상수항을 더한 뒤 results.predict()를 호출해 예측값을 산출합니다. 신규 데이터의 전처리가 학습 시와 동일해야 합니다.",
    role: "적합된 통계 모델 + 신규 데이터로 예측을 산출합니다.",
    input: "model_in: 적합 결과 + data_in: 신규 데이터.",
    output: "'Predict' 컬럼이 추가된 DataFrame.",
    whenToUse: "적합된 GLM을 새 데이터에 적용해 예측을 산출할 때.",
    connections: "ResultModel → PredictModel ←(신규 데이터).",
    commonErrors: "신규 데이터의 피처/전처리가 학습 시와 동일해야 합니다.",
  },
  [ModuleType.DiversionChecker]: {
    title: "Diversion Checker",
    category: "데이터 품질",
    beginner:
      "카운트(건수) 데이터가 '흩어짐이 평균보다 큰지(과대분산)'를 점검하고, 어떤 카운트 회귀(Poisson/Quasi/음이항)를 써야 할지 추천해 주는 블록입니다. 잘못된 분포를 쓰면 p-value가 왜곡되므로 모델 선택 전에 진단합니다.",
    analysisMethod:
      "Poisson을 적합한 뒤 분산팽창계수 φ(Pearson 잔차 제곱합 ÷ 자유도)를 구해 φ<1.2면 Poisson, 1.2≤φ<2면 Quasi-Poisson, φ≥2면 Negative Binomial을 추천합니다. 추가로 Poisson vs NB의 AIC 비교와 Cameron–Trivedi 과대분산 검정을 수행합니다.",
    role: "카운트 데이터의 과대분산(φ)을 진단하고 적절한 회귀 모델을 추천합니다.",
    input: "DataFrame + feature_columns·label_column(count).",
    output: "φ + 추천 모델 + AIC 비교 + Cameron–Trivedi 검정 결과.",
    parameters: "feature_columns · label_column(카운트 타깃)",
    whenToUse: "카운트 회귀(클레임 횟수 등)를 적합하기 전 분포 가정을 점검할 때.",
    connections: "전처리 데이터 → DiversionChecker → PoissonModel/QuasiPoissonModel/NegativeBinomialModel 선택.",
    commonErrors: "label_column이 정수 카운트가 아니면 진단이 무의미합니다. 표본이 적으면 φ 추정이 불안정합니다.",
  },
  [ModuleType.EvaluateStat]: {
    title: "Evaluate Stat",
    category: "모델 연산",
    beginner:
      "전통 통계 모델의 '적합도 성적표'를 만들어 주는 블록입니다. 모델이 데이터를 얼마나 잘 설명하는지, 여러 모델 중 어느 쪽이 나은지를 통계 지표로 비교하고 잔차도 점검합니다.",
    analysisMethod:
      "적합된 statsmodels 결과에서 Log-Likelihood·AIC·BIC(작을수록 좋음, 모델 비교)·deviance·Pearson χ²와 계수표(coef·std err·z·p-value·CI)를 뽑아 종합합니다. AIC/BIC는 동일 데이터에 적합한 모델끼리만 비교가 유효합니다.",
    role: "전통 통계 모델의 적합도(LogLik/AIC/BIC/deviance/χ²)와 계수·잔차 진단을 수행합니다.",
    input: "적합된 statsmodels 결과(ResultModel).",
    output: "적합도 지표 + 계수표 + 잔차 진단.",
    parameters: "label_column · prediction_column · model_type",
    whenToUse: "GLM/회귀의 적합도를 비교하거나 가정 위반(잔차 패턴)을 진단할 때.",
    connections: "ResultModel → EvaluateStat.",
    commonErrors: "AIC/BIC는 동일 데이터에 적합한 모델끼리만 비교 가능합니다.",
  },
};
