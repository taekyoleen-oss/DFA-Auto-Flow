import { ModuleType, CanvasModule, ModuleStatus } from "./types";
// fix: Changed import from ChartBarIcon to BarChartIcon to match the exported member and added ModuleStatus for enum usage.
import {
  DatabaseIcon,
  TableCellsIcon,
  ScaleIcon,
  BarChartIcon,
  ShareIcon,
  CogIcon,
  CheckBadgeIcon,
  CalculatorIcon,
  BellCurveIcon,
  ChartCurveIcon,
  PriceTagIcon,
  FilterIcon,
  DocumentTextIcon,
  UsersIcon,
  BeakerIcon,
  HashtagIcon,
  PresentationChartLineIcon,
  CircleStackIcon,
  ShieldCheckIcon,
  ChartPieIcon,
  FingerPrintIcon,
} from "./components/icons";

export const TOOLBOX_MODULES = [
  // Data Preprocess
  {
    type: ModuleType.Statistics,
    name: "Statistics",
    icon: BarChartIcon,
    description: "Computes descriptive statistics for the dataset.",
  },
  {
    type: ModuleType.SelectData,
    name: "Select Data",
    icon: TableCellsIcon,
    description: "Selects or removes columns from the data.",
  },
  {
    type: ModuleType.LoadClaimData,
    name: "Load Claim Data",
    icon: DatabaseIcon,
    description: "Loads claim data from a CSV file for DFA analysis.",
  },
  {
    type: ModuleType.ApplyInflation,
    name: "Apply Inflation",
    icon: ChartCurveIcon,
    description: "Applies annual inflation rate to project claim amounts to target year.",
  },
  {
    type: ModuleType.FormatChange,
    name: "Format Change",
    icon: TableCellsIcon,
    description: "Extracts year from date column and adds it as a new column.",
  },
  {
    type: ModuleType.TransitionData,
    name: "Transition Data",
    icon: ScaleIcon,
    description: "Applies mathematical transformations to numeric columns.",
  },
  {
    type: ModuleType.TransformData,
    name: "Transform Data",
    icon: CogIcon,
    description:
      "Applies pre-trained handlers (e.g., for missing values, encoding) to a dataset.",
  },

  // Claim Analysis
  {
    type: ModuleType.SplitData,
    name: "Split Data",
    icon: ShareIcon,
    description: "Splits data into training and testing sets.",
  },
  {
    type: ModuleType.TrainModel,
    name: "Train Model",
    icon: CogIcon,
    description: "Trains a machine learning model with data.",
  },
  {
    type: ModuleType.ScoreModel,
    name: "Score Model",
    icon: CalculatorIcon,
    description: "Generates predictions on data using a trained model.",
  },
  {
    type: ModuleType.EvaluateModel,
    name: "Evaluate Model",
    icon: CheckBadgeIcon,
    description: "Evaluates the performance of a model's predictions.",
  },

  // Supervised Learning
  {
    type: ModuleType.LinearRegression,
    name: "Linear Regression",
    icon: BarChartIcon,
    description: "A regression algorithm for predicting continuous values.",
  },
  {
    type: ModuleType.LogisticRegression,
    name: "Logistic Regression",
    icon: PresentationChartLineIcon,
    description: "A classification algorithm for predicting binary outcomes.",
  },
  {
    type: ModuleType.PoissonRegression,
    name: "Poisson Regression",
    icon: HashtagIcon,
    description: "A regression model for count data. (Deprecated: Use Poisson Model instead)",
  },
  {
    type: ModuleType.NegativeBinomialRegression,
    name: "Negative Binomial",
    icon: HashtagIcon,
    description: "A regression model for overdispersed count data. (Deprecated: Use Negative Binomial Model instead)",
  },
  {
    type: ModuleType.DecisionTree,
    name: "Decision Tree",
    icon: ShareIcon,
    description:
      "A model using a tree-like structure for classification or regression.",
  },
  {
    type: ModuleType.RandomForest,
    name: "Random Forest",
    icon: ShareIcon,
    description:
      "An ensemble of decision trees for classification or regression.",
  },
  {
    type: ModuleType.SVM,
    name: "Support Vector Machine",
    icon: ShieldCheckIcon,
    description:
      "A model finding the optimal hyperplane for classification or regression.",
  },
  {
    type: ModuleType.LinearDiscriminantAnalysis,
    name: "LDA",
    icon: BeakerIcon,
    description: "A dimensionality reduction and classification technique.",
  },
  {
    type: ModuleType.NaiveBayes,
    name: "Naive Bayes",
    icon: BeakerIcon,
    description: "A probabilistic classifier based on Bayes' theorem.",
  },
  {
    type: ModuleType.KNN,
    name: "K-Nearest Neighbors",
    icon: CircleStackIcon,
    description:
      "An algorithm for classification or regression based on nearest neighbors.",
  },

  // Unsupervised Models
  {
    type: ModuleType.KMeans,
    name: "K-Means Clustering",
    icon: UsersIcon,
    description:
      "An unsupervised algorithm for partitioning data into K clusters.",
  },
  {
    type: ModuleType.HierarchicalClustering,
    name: "Hierarchical Clustering",
    icon: UsersIcon,
    description:
      "An unsupervised algorithm that builds a hierarchy of clusters.",
  },
  {
    type: ModuleType.DBSCAN,
    name: "DBSCAN",
    icon: FingerPrintIcon,
    description: "A density-based clustering algorithm.",
  },
  {
    type: ModuleType.PrincipalComponentAnalysis,
    name: "PCA",
    icon: ChartPieIcon,
    description: "A technique for dimensionality reduction.",
  },

  // Traditional Analysis - Statsmodels Models
  {
    type: ModuleType.OLSModel,
    name: "OLS Model",
    icon: BarChartIcon,
    description: "Ordinary Least Squares regression model.",
  },
  {
    type: ModuleType.LogisticModel,
    name: "Logistic Model",
    icon: PresentationChartLineIcon,
    description: "Logistic regression model for binary classification.",
  },
  {
    type: ModuleType.PoissonModel,
    name: "Poisson Model",
    icon: HashtagIcon,
    description: "Poisson regression model for count data.",
  },
  {
    type: ModuleType.QuasiPoissonModel,
    name: "Quasi-Poisson Model",
    icon: HashtagIcon,
    description: "Quasi-Poisson regression model for overdispersed count data.",
  },
  {
    type: ModuleType.NegativeBinomialModel,
    name: "Negative Binomial Model",
    icon: HashtagIcon,
    description: "Negative Binomial regression model for overdispersed count data.",
  },
  // Tradition Analysis - Advanced Models
  {
    type: ModuleType.StatModels,
    name: "Stat Models",
    icon: CogIcon,
    description: "Advanced statistical models (Gamma, Tweedie).",
  },
  {
    type: ModuleType.ResultModel,
    name: "Result Model",
    icon: CalculatorIcon,
    description: "Fits a statistical model and shows the summary.",
  },
  {
    type: ModuleType.PredictModel,
    name: "Predict Model",
    icon: CalculatorIcon,
    description: "Generates predictions using a fitted statistical model.",
  },
  {
    type: ModuleType.DiversionChecker,
    name: "Diversion Checker",
    icon: BeakerIcon,
    description: "Checks for overdispersion in count data and recommends appropriate regression models.",
  },
  {
    type: ModuleType.EvaluateStat,
    name: "Evaluate Stat",
    icon: CalculatorIcon,
    description: "Evaluates statistical model performance with various metrics.",
  },

  // Reinsurance (Distribution)
  {
    type: ModuleType.FitLossDistribution,
    name: "Fit Loss Distribution",
    icon: BellCurveIcon,
    description: "Fits a statistical distribution to loss data.",
  },
  {
    type: ModuleType.GenerateExposureCurve,
    name: "Generate Exposure Curve",
    icon: ChartCurveIcon,
    description: "Generates an exposure curve from a fitted distribution.",
  },
  {
    type: ModuleType.PriceXoLLayer,
    name: "Price XoL Layer",
    icon: PriceTagIcon,
    description: "Prices an Excess of Loss (XoL) layer from an exposure curve.",
  },

  // Reinsurance (Experience)
  {
    type: ModuleType.XolLoading,
    name: "XoL Loading",
    icon: DatabaseIcon,
    description: "Loads claims data for experience-based XoL pricing.",
  },
  {
    type: ModuleType.ApplyThreshold,
    name: "Apply Threshold",
    icon: FilterIcon,
    description: "Filters out claims below a specified threshold.",
  },
  {
    type: ModuleType.DefineXolContract,
    name: "XoL Contract",
    icon: DocumentTextIcon,
    description: "Defines the terms of an XoL reinsurance contract.",
  },
  {
    type: ModuleType.CalculateCededLoss,
    name: "Calculate Ceded Loss",
    icon: CalculatorIcon,
    description:
      "Calculates the ceded loss for claims based on contract terms.",
  },
  {
    type: ModuleType.PriceXolContract,
    name: "Price XoL Contract",
    icon: PriceTagIcon,
    description: "Prices an XoL contract using the burning cost method.",
  },
  {
    type: ModuleType.XolCalculator,
    name: "XOL Calculator",
    icon: CalculatorIcon,
    description: "Calculates XoL results using contract terms and claim data.",
  },
  {
    type: ModuleType.ExperienceModel,
    name: "Experience Model",
    icon: BeakerIcon,
    description: "Applies XoL contract terms to claim data to calculate XoL claims.",
  },

  // DFA Modules
  {
    type: ModuleType.SplitByThreshold,
    name: "Split By Threshold",
    icon: FilterIcon,
    description: "Splits claims data by threshold amount into two outputs.",
  },
  {
    type: ModuleType.FitAggregateModel,
    name: "Fit Agg Model",
    icon: BellCurveIcon,
    description: "Fits statistical distribution to aggregate claim amounts.",
  },
  {
    type: ModuleType.SimulateAggDist,
    name: "Simulate Agg Table",
    icon: PresentationChartLineIcon,
    description: "Performs Monte Carlo simulation using the selected distribution.",
  },
  {
    type: ModuleType.SplitByFreqServ,
    name: "Split By Freq-Sev",
    icon: ShareIcon,
    description: "Splits claim data into frequency and severity components.",
  },
  {
    type: ModuleType.FitFrequencyModel,
    name: "Fit Frequency Model",
    icon: BeakerIcon,
    description: "Fits and compares multiple frequency distributions.",
  },
  {
    type: ModuleType.FitSeverityModel,
    name: "Fit Severity Model",
    icon: BeakerIcon,
    description: "Fits and compares multiple severity distributions.",
  },
  {
    type: ModuleType.SimulateFreqServ,
    name: "Simulate Freq-Sev Table",
    icon: PresentationChartLineIcon,
    description: "Simulates aggregate loss using frequency and severity models.",
  },
  {
    type: ModuleType.CombineLossModel,
    name: "Combine Loss Model",
    icon: BarChartIcon,
    description: "Combines two simulation results and calculates VaR and TVaR.",
  },
];

// fix: Replaced all instances of status: 'Pending' with status: ModuleStatus.Pending to conform to the ModuleStatus enum type.
export const DEFAULT_MODULES: Omit<CanvasModule, "id" | "position" | "name">[] =
  [
    {
      type: ModuleType.LoadClaimData,
      status: ModuleStatus.Pending,
      parameters: { source: "claim_data.csv" },
      inputs: [],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.Statistics,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [],
    },
    {
      type: ModuleType.SelectData,
      status: ModuleStatus.Pending,
      parameters: { columnSelections: {} },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.HandleMissingValues,
      status: ModuleStatus.Pending,
      parameters: {
        method: "remove_row", // 'remove_row', 'impute', 'knn'
        strategy: "mean", // for 'impute'
        n_neighbors: 5, // for 'knn'
        metric: "nan_euclidean", // for 'knn'
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "handler_out", type: "handler" }],
    },
    {
      type: ModuleType.TransformData,
      status: ModuleStatus.Pending,
      parameters: { primary_exclude_column: "", exclude_columns: [] },
      inputs: [
        { name: "handler_in", type: "handler" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.EncodeCategorical,
      status: ModuleStatus.Pending,
      parameters: {
        method: "one_hot",
        columns: [],
        // one-hot params
        handle_unknown: "ignore",
        drop: "first",
        // ordinal params
        ordinal_mapping: "{}", // JSON string for easier UI
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "handler_out", type: "handler" }],
    },
    {
      type: ModuleType.NormalizeData,
      status: ModuleStatus.Pending,
      parameters: { method: "MinMax", columnSelections: {} },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "handler_out", type: "handler" }],
    },
    {
      type: ModuleType.TransitionData,
      status: ModuleStatus.Pending,
      parameters: { transformations: {} },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.ResampleData,
      status: ModuleStatus.Pending,
      parameters: { method: "SMOTE", target_column: null },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.SplitData,
      status: ModuleStatus.Pending,
      parameters: {
        train_size: 0.75,
        random_state: 43,
        shuffle: "True",
        stratify: "False",
        stratify_column: null,
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [
        { name: "train_data_out", type: "data" },
        { name: "test_data_out", type: "data" },
      ],
    },
    {
      type: ModuleType.LinearRegression,
      status: ModuleStatus.Pending,
      // Updated parameters for Linear Regression, Lasso, Ridge, ElasticNet
      parameters: {
        model_type: "LinearRegression",
        alpha: 1.0,
        l1_ratio: 0.5,
        fit_intercept: "True",
        tuning_enabled: "False",
        tuning_strategy: "GridSearch",
        alpha_candidates: "0.01,0.1,1,10",
        l1_ratio_candidates: "0.2,0.5,0.8",
        cv_folds: 5,
        scoring_metric: "neg_mean_squared_error",
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.LogisticRegression,
      status: ModuleStatus.Pending,
      parameters: {
        penalty: "l2",
        C: 1.0,
        solver: "lbfgs",
        max_iter: 100,
        tuning_enabled: "False",
        tuning_strategy: "GridSearch",
        c_candidates: "0.01,0.1,1,10,100",
        l1_ratio_candidates: "0.2,0.5,0.8",
        cv_folds: 5,
        scoring_metric: "accuracy",
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.SVM,
      status: ModuleStatus.Pending,
      parameters: {
        model_purpose: "classification",
        C: 1.0,
        kernel: "rbf",
        gamma: "scale",
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.RandomForest,
      status: ModuleStatus.Pending,
      parameters: {
        model_purpose: "classification",
        n_estimators: 100,
        criterion: "gini",
        max_depth: null,
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.PoissonRegression,
      status: ModuleStatus.Pending,
      parameters: { distribution_type: "Poisson", max_iter: 100 },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.NegativeBinomialRegression,
      status: ModuleStatus.Pending,
      parameters: {
        distribution_type: "NegativeBinomial",
        max_iter: 100,
        disp: 1.0,
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.LinearDiscriminantAnalysis,
      status: ModuleStatus.Pending,
      parameters: { solver: "svd", shrinkage: null },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.NaiveBayes,
      status: ModuleStatus.Pending,
      parameters: { var_smoothing: 1e-9 },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.KNN,
      status: ModuleStatus.Pending,
      parameters: {
        model_purpose: "classification",
        n_neighbors: 3,
        weights: "uniform",
        algorithm: "auto",
        metric: "minkowski",
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.DecisionTree,
      status: ModuleStatus.Pending,
      parameters: {
        model_purpose: "classification",
        criterion: "gini",
        max_depth: null,
        min_samples_split: 2,
        min_samples_leaf: 1,
      },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.KMeans,
      status: ModuleStatus.Pending,
      parameters: {
        n_clusters: 3,
        init: "k-means++",
        n_init: 10,
        max_iter: 300,
        random_state: 42,
        feature_columns: [],
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [
        { name: "data_out", type: "data" },
        { name: "model_out", type: "model" },
      ],
    },
    {
      type: ModuleType.HierarchicalClustering,
      status: ModuleStatus.Pending,
      parameters: {
        n_clusters: 3,
        affinity: "euclidean",
        linkage: "ward",
        feature_columns: [],
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.DBSCAN,
      status: ModuleStatus.Pending,
      parameters: { eps: 0.5, min_samples: 5, feature_columns: [] },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.PrincipalComponentAnalysis,
      status: ModuleStatus.Pending,
      parameters: { n_components: 2, feature_columns: [] },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.TrainModel,
      status: ModuleStatus.Pending,
      parameters: { feature_columns: [], label_column: null },
      inputs: [
        { name: "model_in", type: "model" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "trained_model_out", type: "model" }],
    },
    {
      type: ModuleType.ScoreModel,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [
        { name: "model_in", type: "model" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "scored_data_out", type: "data" }],
    },
    {
      type: ModuleType.EvaluateModel,
      status: ModuleStatus.Pending,
      parameters: {
        label_column: null,
        prediction_column: null,
        model_type: "regression",
        threshold: 0.5,
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "evaluation_out", type: "evaluation" }],
    },
    {
      type: ModuleType.OLSModel,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.LogisticModel,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.PoissonModel,
      status: ModuleStatus.Pending,
      parameters: { max_iter: 100 },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.QuasiPoissonModel,
      status: ModuleStatus.Pending,
      parameters: { max_iter: 100 },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.NegativeBinomialModel,
      status: ModuleStatus.Pending,
      parameters: { max_iter: 100, disp: 1.0 },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.StatModels,
      status: ModuleStatus.Pending,
      parameters: { model: "Gamma" },
      inputs: [],
      outputs: [{ name: "model_out", type: "model" }],
    },
    {
      type: ModuleType.ResultModel,
      status: ModuleStatus.Pending,
      parameters: { feature_columns: [], label_column: null },
      inputs: [
        { name: "model_in", type: "model" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "result_out", type: "evaluation" }],
    },
    {
      type: ModuleType.PredictModel,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [
        { name: "model_in", type: "evaluation" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "scored_data_out", type: "data" }],
    },
    {
      type: ModuleType.DiversionChecker,
      status: ModuleStatus.Pending,
      parameters: { feature_columns: [], label_column: null, max_iter: 100 },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "result_out", type: "evaluation" }],
    },
    {
      type: ModuleType.EvaluateStat,
      status: ModuleStatus.Pending,
      parameters: { label_column: null, prediction_column: null, model_type: null },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "result_out", type: "evaluation" }],
    },
    // Reinsurance (Distribution)
    {
      type: ModuleType.FitLossDistribution,
      status: ModuleStatus.Pending,
      parameters: { loss_column: null, distribution_type: "Pareto" },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "dist_out", type: "distribution" }],
    },
    {
      type: ModuleType.GenerateExposureCurve,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [
        { name: "dist_in", type: "distribution" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "curve_out", type: "curve" }],
    },
    {
      type: ModuleType.PriceXoLLayer,
      status: ModuleStatus.Pending,
      parameters: { retention: 1000000, limit: 5000000, loading_factor: 1.5 },
      inputs: [{ name: "curve_in", type: "curve" }],
      outputs: [{ name: "price_out", type: "evaluation" }],
    },
    // Reinsurance (Experience)
    {
      type: ModuleType.XolLoading,
      status: ModuleStatus.Pending,
      parameters: { source: "xol_claims_data.csv" },
      inputs: [],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.ApplyThreshold,
      status: ModuleStatus.Pending,
      parameters: { threshold: 1000000, amount_column: "클레임 금액" },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.DefineXolContract,
      status: ModuleStatus.Pending,
      parameters: {
        deductible: 3000000,
        limit: 2000000,
        reinstatements: 2,
        aggDeductible: 0,
        expenseRatio: 0.3,
        defaultReinstatementRate: 100,
        yearRates: [],
      },
      inputs: [],
      outputs: [{ name: "contract_out", type: "contract" }],
    },
    {
      type: ModuleType.CalculateCededLoss,
      status: ModuleStatus.Pending,
      parameters: { loss_column: "loss" },
      inputs: [
        { name: "data_in", type: "data" },
        { name: "contract_in", type: "contract" },
      ],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.PriceXolContract,
      status: ModuleStatus.Pending,
      parameters: {
        volatility_loading: 25.0,
        year_column: "year",
        ceded_loss_column: "ceded_loss",
      },
      inputs: [
        { name: "data_in", type: "data" },
        { name: "contract_in", type: "contract" },
      ],
      outputs: [{ name: "price_out", type: "evaluation" }],
    },
    {
      type: ModuleType.XolCalculator,
      status: ModuleStatus.Pending,
      parameters: {
        claim_column: null,
      },
      inputs: [
        { name: "contract_in", type: "contract" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.ExperienceModel,
      status: ModuleStatus.Pending,
      parameters: {
        claim_column: null,
      },
      inputs: [
        { name: "contract_in", type: "contract" },
        { name: "data_in", type: "data" },
      ],
      outputs: [{ name: "data_out", type: "data" }],
    },
    // DFA Modules
    {
      type: ModuleType.LoadClaimData,
      status: ModuleStatus.Pending,
      parameters: { source: "claim_data.csv" },
      inputs: [],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.ApplyInflation,
      status: ModuleStatus.Pending,
      parameters: {
        target_year: 2026,
        inflation_rate: 5.0,
        amount_column: "클레임 금액",
        year_column: "날짜",
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.FormatChange,
      status: ModuleStatus.Pending,
      parameters: {
        date_column: "날짜",
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "data" }],
    },
    {
      type: ModuleType.SplitByThreshold,
      status: ModuleStatus.Pending,
      parameters: {
        threshold: 1000000,
        amount_column: "클레임 금액",
        date_column: "날짜",
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [
        { name: "below_threshold_out", type: "data" },
        { name: "above_threshold_out", type: "data" },
      ],
    },
    {
      type: ModuleType.FitAggregateModel,
      status: ModuleStatus.Pending,
      parameters: {
        selected_distributions: ["Lognormal", "Exponential", "Gamma", "Pareto"],
        amount_column: "total_amount",
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "model_out", type: "evaluation" }],
    },
    {
      type: ModuleType.SimulateAggDist,
      status: ModuleStatus.Pending,
      parameters: {
        simulation_count: 10000,
        custom_count: "",
      },
      inputs: [{ name: "model_in", type: "evaluation" }],
      outputs: [{ name: "simulation_out", type: "evaluation" }],
    },
    {
      type: ModuleType.SplitByFreqServ,
      status: ModuleStatus.Pending,
      parameters: {
        amount_column: "클레임 금액_infl",
        date_column: "날짜",
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [
        { name: "frequency_out", type: "data" },
        { name: "severity_out", type: "data" },
      ],
    },
    {
      type: ModuleType.FitFrequencyModel,
      status: ModuleStatus.Pending,
      parameters: {
        count_column: "count",
        selected_frequency_types: ["Poisson", "NegativeBinomial"],
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "distribution" }],
    },
    {
      type: ModuleType.FitSeverityModel,
      status: ModuleStatus.Pending,
      parameters: {
        amount_column: "클레임 금액",
        selected_severity_types: ["Lognormal", "Exponential", "Gamma", "Pareto"],
      },
      inputs: [{ name: "data_in", type: "data" }],
      outputs: [{ name: "data_out", type: "distribution" }],
    },
    {
      type: ModuleType.SimulateFreqServ,
      status: ModuleStatus.Pending,
      parameters: {
        frequency_type: "Poisson",
        severity_type: "Lognormal",
        amount_column: "클레임 금액",
        date_column: "날짜",
        simulation_count: 10000,
        custom_count: "",
      },
      inputs: [
        { name: "frequency_in", type: "distribution" },
        { name: "severity_in", type: "distribution" },
      ],
      outputs: [{ name: "model_out", type: "evaluation" }],
    },
    {
      type: ModuleType.CombineLossModel,
      status: ModuleStatus.Pending,
      parameters: {},
      inputs: [
        { name: "agg_dist_in", type: "evaluation" },
        { name: "freq_serv_in", type: "evaluation" },
      ],
      outputs: [{ name: "combined_out", type: "evaluation" }],
    },
  ];

export const SAMPLE_MODELS: any[] = [];
