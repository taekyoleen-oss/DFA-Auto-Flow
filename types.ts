// fix: Removed erroneous import of ModuleType from './App' to resolve circular dependency and declaration merge errors.
export enum ModuleType {
  LoadData = "LoadData",
  Statistics = "Statistics", // New module type
  SelectData = "SelectData",
  HandleMissingValues = "HandleMissingValues",
  TransformData = "TransformData",
  EncodeCategorical = "EncodeCategorical",
  NormalizeData = "NormalizeData",
  TransitionData = "TransitionData",
  ResampleData = "ResampleData",
  SplitData = "SplitData",

  // Supervised Learning Models
  LinearRegression = "LinearRegression",
  LogisticRegression = "LogisticRegression",
  PoissonRegression = "PoissonRegression",
  NegativeBinomialRegression = "NegativeBinomialRegression",
  DecisionTree = "DecisionTree",
  RandomForest = "RandomForest",
  SVM = "SVM",
  LinearDiscriminantAnalysis = "LinearDiscriminantAnalysis",
  NaiveBayes = "NaiveBayes",
  KNN = "KNN",

  // Model Operations
  TrainModel = "TrainModel",
  ScoreModel = "ScoreModel",
  EvaluateModel = "EvaluateModel",

  // Unsupervised Learning
  KMeans = "KMeans",
  HierarchicalClustering = "HierarchicalClustering",
  DBSCAN = "DBSCAN",
  PrincipalComponentAnalysis = "PrincipalComponentAnalysis",

  // Traditional Analysis - Statsmodels Models
  OLSModel = "OLSModel",
  LogisticModel = "LogisticModel",
  PoissonModel = "PoissonModel",
  QuasiPoissonModel = "QuasiPoissonModel",
  NegativeBinomialModel = "NegativeBinomialModel",
  DiversionChecker = "DiversionChecker",
  EvaluateStat = "EvaluateStat",

  // Legacy/StatModels - Keeping for advanced models (Gamma, Tweedie)
  StatModels = "StatModels",
  ResultModel = "ResultModel",
  PredictModel = "PredictModel",

  // Reinsurance Modules
  FitLossDistribution = "FitLossDistribution",
  GenerateExposureCurve = "GenerateExposureCurve",
  PriceXoLLayer = "PriceXoLLayer",
  ApplyThreshold = "ApplyThreshold",
  DefineXolContract = "DefineXolContract",
  CalculateCededLoss = "CalculateCededLoss",
  PriceXolContract = "PriceXolContract",
  XolCalculator = "XolCalculator",
  XolPricing = "XolPricing",
  ExperienceModel = "ExperienceModel",

  // DFA Modules
  LoadClaimData = "LoadClaimData",
  ApplyInflation = "ApplyInflation",
  FormatChange = "FormatChange",
  SplitByThreshold = "SplitByThreshold",
  SplitByFreqServ = "SplitByFreqServ",
  FitAggregateModel = "FitAggregateModel",
  SimulateAggDist = "SimulateAggDist",
  FitFrequencyModel = "FitFrequencyModel",
  FitSeverityModel = "FitSeverityModel",
  SimulateFreqServ = "SimulateFreqServ",
  CombineLossModel = "CombineLossModel",
  SettingThreshold = "SettingThreshold",

  // Deprecating these
  LogisticTradition = "LogisticTradition",

  // Shape Types
  TextBox = "TextBox",
  GroupBox = "GroupBox",
}

export enum ModuleStatus {
  Pending = "Pending",
  Running = "Running",
  Success = "Success",
  Error = "Error",
}

export interface Port {
  name: string;
  type:
    | "data"
    | "model"
    | "evaluation"
    | "distribution"
    | "curve"
    | "contract"
    | "handler"
    | "threshold";
}

export interface ColumnInfo {
  name: string;
  type: string;
}

export interface DataPreview {
  type: "DataPreview"; // Differentiator
  columns: ColumnInfo[];
  totalRowCount: number;
  rows?: Record<string, any>[];
}

// Types for the new Statistics module output
export interface DescriptiveStats {
  [columnName: string]: {
    count: number;
    mean: number;
    std: number;
    min: number;
    "25%": number;
    "50%": number; // median
    "75%": number;
    max: number;
    variance: number;
    nulls: number;
    mode: number | string;
    skewness: number;
    kurtosis: number;
  };
}

export interface CorrelationMatrix {
  [column1: string]: {
    [column2: string]: number;
  };
}

export interface StatisticsOutput {
  type: "StatisticsOutput"; // Differentiator
  stats: DescriptiveStats;
  correlation: CorrelationMatrix;
  columns: ColumnInfo[]; // Keep original column info
}

export interface SplitDataOutput {
  type: "SplitDataOutput";
  train: DataPreview;
  test: DataPreview;
}

export interface TuningCandidateScore {
  params: Record<string, number>;
  score: number;
}

export interface TuningSummary {
  enabled: boolean;
  strategy?: "grid";
  bestParams?: Record<string, number>;
  bestScore?: number;
  scoringMetric?: string;
  candidates?: TuningCandidateScore[];
}

export interface TrainedModelOutput {
  type: "TrainedModelOutput";
  modelType: ModuleType;
  modelPurpose?: "classification" | "regression";
  coefficients: Record<string, number>;
  intercept: number;
  metrics: Record<string, number>;
  featureColumns: string[];
  labelColumn: string;
  tuningSummary?: TuningSummary;
  statsModelsResult?: StatsModelsResultOutput; // statsmodels 결과 (포아송/음이항 회귀용)
}

export type StatsModelFamily =
  | "OLS"
  | "Logistic"
  | "Logit"
  | "Poisson"
  | "QuasiPoisson"
  | "NegativeBinomial"
  | "Gamma"
  | "Tweedie";

export interface ModelDefinitionOutput {
  type: "ModelDefinitionOutput";
  modelFamily: "statsmodels";
  modelType: StatsModelFamily;
  parameters: Record<string, any>;
}

export interface StatsModelsResultOutput {
  type: "StatsModelsResultOutput";
  modelType: StatsModelFamily;
  summary: {
    coefficients: Record<
      string,
      {
        coef: number;
        "std err": number;
        t?: number;
        z?: number;
        "P>|t|"?: number;
        "P>|z|"?: number;
        "[0.025": number;
        "0.975]": number;
      }
    >;
    metrics: Record<string, string | number>;
  };
  featureColumns: string[];
  labelColumn: string;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ThresholdMetric {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface EvaluationOutput {
  type: "EvaluationOutput";
  modelType: "classification" | "regression";
  metrics: Record<string, number | string>;
  confusionMatrix?: ConfusionMatrix;
  threshold?: number;
  thresholdMetrics?: ThresholdMetric[]; // 여러 threshold에 대한 precision/recall
}

export interface DiversionCheckerOutput {
  type: "DiversionCheckerOutput";
  phi: number;
  recommendation: "Poisson" | "QuasiPoisson" | "NegativeBinomial";
  poissonAic: number | null;
  negativeBinomialAic: number | null;
  aicComparison: string | null;
  cameronTrivediCoef: number;
  cameronTrivediPvalue: number;
  cameronTrivediConclusion: string;
  methodsUsed: string[];
  results: {
    phi: number;
    phi_interpretation: string;
    recommendation: string;
    poisson_aic: number | null;
    negative_binomial_aic: number | null;
    cameron_trivedi_coef: number;
    cameron_trivedi_pvalue: number;
    cameron_trivedi_conclusion: string;
  };
}

export interface EvaluateStatOutput {
  type: "EvaluateStatOutput";
  modelType: string;
  metrics: Record<string, number | string>;
  residuals?: number[];
  deviance?: number;
  pearsonChi2?: number;
  dispersion?: number;
  aic?: number;
  bic?: number;
  logLikelihood?: number;
}

// --- New Unsupervised Learning Outputs ---
export interface KMeansOutput {
  type: "KMeansOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
  centroids: Record<string, number>[];
  model: any; // To hold inertia_ or other model properties
}

export interface HierarchicalClusteringOutput {
  type: "HierarchicalClusteringOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
}

export interface DBSCANOutput {
  type: "DBSCANOutput";
  clusterAssignments: DataPreview; // Data with an added 'cluster' column
  n_clusters: number;
  n_noise: number;
}

export interface PCAOutput {
  type: "PCAOutput";
  transformedData: DataPreview;
  explainedVarianceRatio: number[];
}

// --- Reinsurance Module Outputs ---
export type LossDistributionType = "Pareto" | "Lognormal";

export interface FittedDistributionOutput {
  type: "FittedDistributionOutput";
  distributionType: LossDistributionType;
  parameters: Record<string, number>;
  lossColumn: string;
}

export interface ExposureCurveOutput {
  type: "ExposureCurveOutput";
  curve: { retention: number; loss_pct: number }[];
  totalExpectedLoss: number;
}

export interface XoLPriceOutput {
  type: "XoLPriceOutput";
  retention: number;
  limit: number;
  expectedLayerLoss: number;
  rateOnLinePct: number;
  premium: number;
}

export interface XolContractOutput {
  type: "XolContractOutput";
  deductible: number;
  limit: number;
  reinstatements: number;
  aggDeductible: number;
  expenseRatio: number;
  reinstatementPremiums: number[];
}

export interface FinalXolPriceOutput {
  type: "FinalXolPriceOutput";
  expectedLoss: number;
  stdDev: number;
  volatilityMargin: number;
  purePremium: number;
  expenseLoading: number;
  finalPremium: number;
}

export interface XolPricingOutput {
  type: "XolPricingOutput";
  xolClaimMean: number;
  xolClaimStdDev: number;
  xolPremiumRateMean: number;
  reluctanceFactor: number;
  expenseRate: number;
  netPremium: number;
  grossPremium: number;
  limit: number;
  deductible: number;
  reinstatements: number;
  aggDeductible: number;
  reinstatementPremiums: number[];
}

export interface MissingHandlerOutput {
  type: "MissingHandlerOutput";
  method: "remove_row" | "impute" | "knn";
  // For impute
  strategy?: "mean" | "median" | "mode";
  // For KNN
  n_neighbors?: number;
  metric?: string;
  // For all methods that are not row removal, we need the values computed from the training set
  imputation_values: Record<string, number | string>; // e.g. { 'Age': 29.5, 'Embarked': 'S' }
}

export interface EncoderOutput {
  type: "EncoderOutput";
  method: "label" | "one_hot" | "ordinal";
  mappings: Record<string, Record<string, number> | string[]>;
  columns_to_encode: string[];
  // one-hot params that are passed through
  drop?: "first" | "if_binary" | null;
  handle_unknown?: "error" | "ignore";
}

export interface NormalizerOutput {
  type: "NormalizerOutput";
  method: "MinMax" | "StandardScaler" | "RobustScaler";
  stats: Record<
    string,
    {
      min?: number;
      max?: number;
      mean?: number;
      stdDev?: number;
      median?: number;
      iqr?: number;
    }
  >;
}

// --- DFA Module Outputs ---
export interface ClaimDataOutput {
  type: "ClaimDataOutput";
  data: DataPreview;
}

export interface InflatedDataOutput {
  type: "InflatedDataOutput";
  data: DataPreview;
  targetYear: number;
  inflationRate: number;
}

export interface FormatChangeOutput {
  type: "FormatChangeOutput";
  data: DataPreview;
}

export interface ThresholdSplitOutput {
  type: "ThresholdSplitOutput";
  belowThreshold: DataPreview; // 연도별 합계만
  aboveThreshold: DataPreview; // 원본 레이아웃 유지
  threshold: number;
}

export interface AggregateModelFitResult {
  distributionType: "Lognormal" | "Exponential" | "Gamma" | "Pareto";
  parameters: Record<string, number>;
  fitStatistics: {
    aic?: number;
    bic?: number;
    logLikelihood?: number;
    ksStatistic?: number;
    ksPValue?: number;
  };
  qqPlot?: {
    theoreticalQuantiles: number[];
    sampleQuantiles: number[];
  };
  ppPlot?: {
    theoreticalCDF: number[];
    empiricalCDF: number[];
  };
  cumulativeDistribution?: {
    percentiles: number[];
    amounts: number[];
  };
  theoreticalCumulative?: {
    probabilities: number[];
    amounts: number[];
  };
}

export interface AggregateModelOutput {
  type: "AggregateModelOutput";
  results: AggregateModelFitResult[]; // 여러 분포의 적합 결과
  selectedDistribution?: "Lognormal" | "Exponential" | "Gamma" | "Pareto"; // 사용자가 선택한 분포
  yearlyAggregates: Array<{
    year: number;
    totalAmount: number;
  }>;
}

export interface SimulateAggDistOutput {
  type: "SimulateAggDistOutput";
  simulationCount: number;
  results: Array<{
    count: number;
    amount: number;
  }>;
  rawSimulations?: number[]; // 원본 시뮬레이션 결과 (최대 100개)
  statistics: {
    mean: number;
    std: number;
    min: number;
    max: number;
    percentile5: number;
    percentile25: number;
    percentile50: number;
    percentile75: number;
    percentile95: number;
    percentile99: number;
  };
  outputFormat?: "xol" | "dfa"; // 출력 형식
  claimLevelData?: Array<{ // XoL 형식일 때 개별 사고 데이터
    simulationNumber: number;
    claimAmount: number;
  }>;
}

export interface SimulateFreqServOutput {
  type: "SimulateFreqServOutput";
  outputFormat: "xol" | "dfa";
  // DFA 형식 (output_1): evaluation 타입
  dfaOutput?: SimulateAggDistOutput;
  // XoL 형식 (output_2): data 타입
  xolOutput?: DataPreview; // claimLevelData를 DataPreview로 변환
}

export interface SplitFreqServOutput {
  type: "SplitFreqServOutput";
  frequencyData: DataPreview; // 연도별 빈도 데이터
  severityData: DataPreview; // 개별 클레임 심도 데이터
  yearlyFrequency: Array<{ year: number; count: number }>;
  yearlySeverity: Array<{ year: number; totalAmount: number; count: number; meanAmount: number }>;
}

export interface FrequencyModelFitResult {
  distributionType: "Poisson" | "NegativeBinomial";
  parameters: Record<string, number>;
  fitStatistics: {
    aic?: number;
    bic?: number;
    logLikelihood?: number;
    mean?: number;
    variance?: number;
    dispersion?: number; // 분산/평균 비율
  };
  yearlyCounts?: Array<{ year: number; count: number }>;
  qqPlot?: {
    theoreticalQuantiles: number[];
    sampleQuantiles: number[];
  };
  ppPlot?: {
    theoreticalCDF: number[];
    empiricalCDF: number[];
  };
}

export interface FrequencyModelOutput {
  type: "FrequencyModelOutput";
  results: FrequencyModelFitResult[];
  selectedDistribution?: "Poisson" | "NegativeBinomial";
  yearlyCounts: Array<{ year: number; count: number }>;
}

export interface SeverityModelFitResult {
  distributionType: "Normal" | "Lognormal" | "Pareto" | "Gamma" | "Exponential" | "Weibull";
  parameters: Record<string, number>;
  fitStatistics: {
    aic?: number;
    bic?: number;
    logLikelihood?: number;
    ksStatistic?: number;
    ksPValue?: number;
  };
  qqPlot?: {
    theoreticalQuantiles: number[];
    sampleQuantiles: number[];
  };
  ppPlot?: {
    theoreticalCDF: number[];
    empiricalCDF: number[];
  };
  cumulativeDistribution?: {
    percentiles: number[];
    amounts: number[];
  };
  theoreticalCumulative?: {
    probabilities: number[];
    amounts: number[];
  };
}

export interface SeverityModelOutput {
  type: "SeverityModelOutput";
  results: SeverityModelFitResult[];
  selectedDistribution?: "Normal" | "Lognormal" | "Pareto" | "Gamma" | "Exponential" | "Weibull";
  originalData?: number[]; // 원본 보험금 데이터 (통계 계산용)
}

export interface FrequencySeverityModelOutput {
  type: "FrequencySeverityModelOutput";
  frequencyModel: {
    type: "Poisson" | "NegativeBinomial";
    parameters: Record<string, number>;
    fitStatistics: {
      aic?: number;
      bic?: number;
      logLikelihood?: number;
    };
  };
  severityModel: {
    type: "Normal" | "Lognormal" | "Pareto" | "Gamma" | "Exponential" | "Weibull";
    parameters: Record<string, number>;
    fitStatistics: {
      aic?: number;
      bic?: number;
      logLikelihood?: number;
      ksStatistic?: number;
      ksPValue?: number;
    };
  };
  aggregateDistribution: {
    mean: number;
    stdDev: number;
    percentiles: Record<string, number>;
  };
}

export interface CombineLossModelOutput {
  type: "CombineLossModelOutput";
  combinedStatistics: {
    mean: number;
    stdDev: number;
    min: number;
    max: number;
    skewness?: number;
    kurtosis?: number;
  };
  var: Record<string, number>; // VaR at different confidence levels (e.g., { "95": 1000000, "99": 2000000 })
  tvar: Record<string, number>; // TVaR (Conditional VaR) at different confidence levels
  percentiles: Record<string, number>; // Combined percentiles
  aggDistPercentiles?: Record<string, number>; // Aggregate Distribution percentiles
  freqServPercentiles?: Record<string, number>; // Frequency-Severity percentiles
  aggregateLossDistribution?: {
    percentiles: number[];
    amounts: number[];
  };
}

export interface SettingThresholdOutput {
  type: "SettingThresholdOutput";
  targetColumn: string;
  thresholds: number[];
  selectedThreshold?: number; // 선택된 threshold 값
  thresholdResults: Array<{
    threshold: number;
    count: number; // 해당 threshold보다 큰 행의 건수
    percentage: number; // 전체 대비 비율
    cumulativeCount: number; // 누적 건수 (가장 큰 threshold부터)
    cumulativePercentage: number; // 누적 비율
  }>;
  yearlyCounts?: Array<{
    year: number | string;
    counts: number[]; // 각 threshold별 건수
    totals?: {
      total: number;
      mean: number;
      std: number;
    };
  }>;
  dataDistribution?: {
    values: number[]; // 원본 데이터 값들 (히스토그램용)
    bins: number[]; // 히스토그램 bins
    frequencies: number[]; // 각 bin의 빈도
  };
  statistics: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
    q25: number;
    q75: number;
  };
}

export interface CanvasModule {
  id: string;
  name: string;
  type: ModuleType;
  position: { x: number; y: number };
  status: ModuleStatus;
  parameters: Record<string, any>;
  inputs: Port[];
  outputs: Port[];
  outputData?:
    | DataPreview
    | StatisticsOutput
    | SplitDataOutput
    | TrainedModelOutput
    | ModelDefinitionOutput
    | StatsModelsResultOutput
    | FittedDistributionOutput
    | ExposureCurveOutput
    | XoLPriceOutput
    | XolContractOutput
    | FinalXolPriceOutput
    | XolPricingOutput
    | EvaluationOutput
    | DiversionCheckerOutput
    | EvaluateStatOutput
    | KMeansOutput
    | HierarchicalClusteringOutput
    | PCAOutput
    | DBSCANOutput
    | MissingHandlerOutput
    | EncoderOutput
    | NormalizerOutput
    | ClaimDataOutput
    | InflatedDataOutput
    | FormatChangeOutput
    | ThresholdSplitOutput
    | SplitFreqServOutput
    | AggregateModelOutput
    | SimulateAggDistOutput
    | FrequencyModelOutput
    | SeverityModelOutput
    | FrequencySeverityModelOutput
    | CombineLossModelOutput
    | SettingThresholdOutput;
  // Shape-specific properties
  shapeData?: {
    // For TextBox
    text?: string;
    width?: number;
    height?: number;
    fontSize?: number;
    // For GroupBox
    moduleIds?: string[]; // IDs of modules in this group
    bounds?: { x: number; y: number; width: number; height: number }; // Bounding box of the group
  };
}

export interface Connection {
  id: string;
  from: { moduleId: string; portName: string };
  to: { moduleId: string; portName: string };
}
