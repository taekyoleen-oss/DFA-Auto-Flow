import { CanvasModule } from "./types";

const replacePlaceholders = (
  template: string,
  params: Record<string, any>
): string => {
  let code = template;
  for (const key in params) {
    const placeholder = new RegExp(`{${key}}`, "g");
    let value = params[key];
    // Stringify only if it's not already a string that looks like code
    if (value === null) {
      value = "None";
    } else if (typeof value !== "string" || !isNaN(Number(value))) {
      value = JSON.stringify(value);
    } else {
      value = `'${value}'`; // Wrap strings in quotes for Python
    }
    code = code.replace(placeholder, value);
  }
  return code;
};

const templates: Record<string, string> = {
  LoadData: `
import pandas as pd

# CSV 파일을 불러와서 DataFrame으로 반환합니다.
# Parameters from UI
file_path = {source}

# Execution
dataframe = pd.read_csv(file_path)
`,

  Statistics: `
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_statistics(df: pd.DataFrame):
    """
    데이터프레임에 대한 기술 통계량과 상관관계 분석을 수행합니다.
    """
    print("=" * 60)
    print("기술 통계량 분석")
    print("=" * 60)
    
    # 기술 통계량
    desc_stats = df.describe()
    print(desc_stats)
    
    # 결측치 정보
    print("\\n결측치 정보:")
    print(df.isnull().sum())
    
    # 상관관계 분석
    print("\\n" + "=" * 60)
    print("상관관계 행렬")
    print("=" * 60)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        corr_matrix = numeric_df.corr()
        print(corr_matrix)
        
        # 상관관계 히트맵 시각화
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
            plt.title('상관관계 히트맵')
            plt.tight_layout()
            plt.show()
    else:
        print("수치형 컬럼이 없어 상관관계 분석을 수행할 수 없습니다.")
        corr_matrix = None
    
    return desc_stats, corr_matrix

# Assuming 'dataframe' is passed from the previous step
# Execution
# descriptive_statistics, correlation_matrix = analyze_statistics(dataframe)
`,

  SelectData: `
import pandas as pd

def select_data(df: pd.DataFrame, columns: list):
    """
    지정된 컬럼만 선택합니다.
    """
    print(f"컬럼 선택: {columns}")
    selected_df = df[columns].copy()
    print(f"선택 완료. Shape: {selected_df.shape}")
    return selected_df

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
# columnSelections is a dict: {column_name: {selected: bool, type: str}}
column_selections = {columnSelections}
selected_columns = [col for col, sel in column_selections.items() if sel.get('selected', True)]

# Execution
# selected_data = select_data(dataframe, selected_columns)
`,
  HandleMissingValues: `
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_values(df: pd.DataFrame, method: str = 'remove_row', 
                         strategy: str = 'mean', columns: list = None,
                         n_neighbors: int = 5):
    """
    결측치를 처리합니다.
    """
    print(f"결측치 처리 방법: {method}")
    df_processed = df.copy()
    
    if method == 'remove_row':
        original_shape = df_processed.shape
        df_processed = df_processed.dropna()
        print(f"행 제거 완료. {original_shape[0]} -> {df_processed.shape[0]} 행")
    
    elif method == 'impute':
        cols_to_impute = columns if columns else df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in cols_to_impute:
            if col not in df_processed.columns:
                continue
            if df_processed[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_processed[col].mean()
                elif strategy == 'median':
                    fill_value = df_processed[col].median()
                elif strategy == 'mode':
                    fill_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 0
                else:
                    fill_value = df_processed[col].mean()
                
                df_processed[col].fillna(fill_value, inplace=True)
                print(f"컬럼 '{col}' 결측치를 {strategy} 값({fill_value:.2f})으로 대체")
    
    elif method == 'knn':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            print(f"KNN 방법으로 결측치 처리 완료 (n_neighbors={n_neighbors})")
        else:
            print("경고: 수치형 컬럼이 없어 KNN 방법을 사용할 수 없습니다.")
    
    return df_processed

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_strategy = {strategy}
p_columns = {columns}
p_n_neighbors = {n_neighbors}

# Execution
# cleaned_data = handle_missing_values(dataframe, p_method, p_strategy, p_columns, p_n_neighbors)
`,
  EncodeCategorical: `
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df: pd.DataFrame, method: str = 'label', 
                      columns: list = None, drop: str = 'first',
                      handle_unknown: str = 'ignore', ordinal_mapping: dict = None):
    """
    범주형 변수를 인코딩합니다.
    """
    print(f"범주형 인코딩 방법: {method}")
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'label':
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                print(f"컬럼 '{col}'에 Label Encoding 적용")
    
    elif method == 'one_hot':
        for col in columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=(drop == 'first'))
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                print(f"컬럼 '{col}'에 One-Hot Encoding 적용")
    
    elif method == 'ordinal':
        for col in columns:
            if col in df_encoded.columns:
                if ordinal_mapping and col in ordinal_mapping:
                    mapping = {val: idx for idx, val in enumerate(ordinal_mapping[col])}
                    df_encoded[col] = df_encoded[col].map(mapping)
                    if handle_unknown == 'ignore':
                        df_encoded[col].fillna(-1, inplace=True)
                else:
                    # 알파벳 순서로 매핑
                    unique_vals = sorted(df_encoded[col].unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df_encoded[col] = df_encoded[col].map(mapping)
                print(f"컬럼 '{col}'에 Ordinal Encoding 적용")
    
    return df_encoded

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_columns = {columns}
p_drop = {drop}
p_handle_unknown = {handle_unknown}
p_ordinal_mapping = {ordinal_mapping}

# Execution
# encoded_data = encode_categorical(dataframe, p_method, p_columns, p_drop, p_handle_unknown, p_ordinal_mapping)
`,
  NormalizeData: `
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

def normalize_data(df: pd.DataFrame, method: str = 'MinMax', columns: list = None):
    """
    데이터를 정규화합니다.
    """
    print(f"데이터 정규화 방법: {method}")
    df_normalized = df.copy()
    
    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'MinMax':
        scaler = MinMaxScaler()
    elif method == 'StandardScaler':
        scaler = StandardScaler()
    elif method == 'RobustScaler':
        scaler = RobustScaler()
    else:
        print(f"알 수 없는 정규화 방법: {method}. MinMax를 사용합니다.")
        scaler = MinMaxScaler()
    
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    print(f"정규화 완료. 컬럼: {columns}")
    
    return df_normalized

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
# columnSelections is a dict: {column_name: {selected: bool}}
p_method = {method}
column_selections = {columnSelections}
p_columns = [col for col, sel in column_selections.items() if sel.get('selected', False) and col in dataframe.columns]

# Execution
# normalized_data = normalize_data(dataframe, p_method, p_columns)
`,
  TransitionData: `
import pandas as pd
import numpy as np

def transform_data(df: pd.DataFrame, transformations: dict):
    """
    수치형 컬럼에 수학적 변환을 적용합니다.
    """
    print("데이터 변환 적용 중...")
    df_transformed = df.copy()
    
    for col, method in transformations.items():
        if method == 'None' or col not in df_transformed.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df_transformed[col]):
            print(f"경고: 컬럼 '{col}'은 수치형이 아니므로 변환할 수 없습니다.")
            continue
        
        new_col_name = f"{col}_{method.lower().replace(' ', '_').replace('-', '_')}"
        print(f"  - 컬럼 '{col}'에 '{method}' 변환 적용 -> '{new_col_name}'")
        
        if method == 'Log':
            df_transformed[new_col_name] = np.log(df_transformed[col].apply(lambda x: x if x > 0 else np.nan))
            df_transformed[new_col_name].fillna(0, inplace=True)
        elif method == 'Square Root':
            df_transformed[new_col_name] = np.sqrt(df_transformed[col].apply(lambda x: x if x >= 0 else np.nan))
            df_transformed[new_col_name].fillna(0, inplace=True)
        elif method == 'Min-Log':
            min_val = df_transformed[col].min()
            df_transformed[new_col_name] = np.log((df_transformed[col] - min_val) + 1)
        elif method == 'Min-Square Root':
            min_val = df_transformed[col].min()
            df_transformed[new_col_name] = np.sqrt((df_transformed[col] - min_val) + 1)
    
    print("데이터 변환 완료.")
    return df_transformed

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI are captured in a dictionary
p_transformations = {transformations}

# Execution
# transformed_data = transform_data(dataframe, p_transformations)
`,
  ResampleData: `
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

def resample_data(df: pd.DataFrame, method: str = 'SMOTE', target_column: str = None):
    """
    클래스 불균형을 처리하기 위해 데이터를 리샘플링합니다.
    """
    if target_column is None:
        print("경고: 타겟 컬럼이 지정되지 않았습니다.")
        return df
    
    print(f"리샘플링 방법: {method}, 타겟 컬럼: {target_column}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"SMOTE 적용 완료. {len(X)} -> {len(X_resampled)} 샘플")
    elif method == 'NearMiss':
        near_miss = NearMiss(version=1)
        X_resampled, y_resampled = near_miss.fit_resample(X, y)
        print(f"NearMiss 적용 완료. {len(X)} -> {len(X_resampled)} 샘플")
    else:
        print(f"알 수 없는 리샘플링 방법: {method}")
        return df
    
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled
    
    return df_resampled

# Assuming 'dataframe' is passed from the previous step
# Parameters from UI
p_method = {method}
p_target_column = {target_column}

# Execution
# resampled_data = resample_data(dataframe, p_method, p_target_column)
`,
  SplitData: `
from sklearn.model_selection import train_test_split
import pandas as pd

# sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
# Assuming 'dataframe' is passed from the previous step

# DataFrame 인덱스를 명시적으로 0부터 시작하도록 리셋
# 이는 동일한 random_state로 항상 동일한 결과를 보장하기 위함입니다.
df = dataframe.copy()
df.index = range(len(df))

# Parameters from UI
p_train_size = {train_size}
p_random_state = {random_state}
p_shuffle = {shuffle} == 'True'
p_stratify = {stratify} == 'True'
p_stratify_column = {stratify_column}

# Stratify 배열 준비
stratify_array = None
if p_stratify and p_stratify_column and p_stratify_column != 'None':
    stratify_array = df[p_stratify_column]

# 데이터 분할
train_data, test_data = train_test_split(
    df,
    train_size=p_train_size,
    random_state=p_random_state,
    shuffle=p_shuffle,
    stratify=stratify_array
)
`,

  LinearRegression: `
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# This module creates a linear regression model instance.
# The model will be trained in the 'Train Model' module.
# Parameters from UI
p_model_type = {model_type}
p_fit_intercept = {fit_intercept} == 'True'
p_alpha = {alpha}
p_l1_ratio = {l1_ratio}

# Create model instance based on model type
if p_model_type == 'LinearRegression':
    model = LinearRegression(fit_intercept=p_fit_intercept)
elif p_model_type == 'Lasso':
    model = Lasso(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
elif p_model_type == 'Ridge':
    model = Ridge(alpha=p_alpha, fit_intercept=p_fit_intercept, random_state=42)
elif p_model_type == 'ElasticNet':
    model = ElasticNet(alpha=p_alpha, l1_ratio=p_l1_ratio, fit_intercept=p_fit_intercept, random_state=42)
else:
    print(f"Unknown model type: {p_model_type}. Defaulting to LinearRegression.")
    model = LinearRegression(fit_intercept=p_fit_intercept)

print(f"{p_model_type} model instance created successfully.")
print(f"  Fit Intercept: {p_fit_intercept}")
if p_model_type in ['Lasso', 'Ridge', 'ElasticNet']:
    print(f"  Alpha: {p_alpha}")
if p_model_type == 'ElasticNet':
    print(f"  L1 Ratio: {p_l1_ratio}")

# Note: The model is not fitted here. It will be fitted in the 'Train Model' module.
# model variable contains the model instance ready for training.
`,

  DecisionTree: `
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def create_decision_tree(model_purpose: str = 'classification', criterion: str = 'gini',
                        max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1):
    """
    의사결정나무 모델을 생성합니다.
    """
    print(f"의사결정나무 모델 생성 중 ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = DecisionTreeClassifier(
            criterion=criterion.lower(),
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    else:
        criterion_reg = 'squared_error' if criterion == 'mse' else 'absolute_error'
        model = DecisionTreeRegressor(
            criterion=criterion_reg,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    print("모델 생성 완료.")
    return model

# Parameters from UI
p_model_purpose = {model_purpose}
p_criterion = {criterion}
p_max_depth = {max_depth}
p_min_samples_split = {min_samples_split}
p_min_samples_leaf = {min_samples_leaf}

# Execution
# decision_tree_model = create_decision_tree(p_model_purpose, p_criterion, p_max_depth, p_min_samples_split, p_min_samples_leaf)
`,

  LogisticTradition: `
from sklearn.linear_model import LogisticRegression

def create_logistic_regression_model():
    """
    Creates a Logistic Regression model using scikit-learn.
    This model uses solvers like 'lbfgs' to find the optimal coefficients.
    It can handle regularization and is suitable for binary and multiclass classification.
    """
    print("Creating Logistic Regression model (sklearn.linear_model.LogisticRegression)...")

    # The parameters (e.g., penalty, C, solver) are hardcoded here for simplicity,
    # but could be exposed in the UI in a real application.
    model = LogisticRegression(random_state=42)
    
    print("Model created successfully.")
    return model

# This module defines the intent to use a scikit-learn LogisticRegression model.
# The actual training happens when this model is connected to a 'Train Model' module.
print("sklearn.linear_model.LogisticRegression model configured.")
`,

  TrainModel: `
import pandas as pd

# This module trains a model using the provided data.
# The model instance comes from a model definition module (e.g., LinearRegression module).
# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}

# Assuming 'model' (from LinearRegression module) and 'dataframe' (from data source) are available
# Extract features and label from dataframe
X_train = dataframe[p_feature_columns]
y_train = dataframe[p_label_column]

# Train the model
trained_model = model.fit(X_train, y_train)

# The trained_model is now ready for use in Score Model or Evaluate Model modules
`,
  ScoreModel: `
import pandas as pd

# This module applies a trained model to a second dataset to generate predictions.
# Parameters from UI (if needed for feature selection)
# Note: Feature columns are typically inferred from the trained model

# Assuming 'trained_model' (from TrainModel module) and 'second_data' (second dataset) are available
# Extract feature columns from the trained model (sklearn models store feature names)
if hasattr(trained_model, 'feature_names_in_'):
    feature_columns = list(trained_model.feature_names_in_)
else:
    # Fallback: use all numeric columns except the label column
    # This assumes the second_data has the same structure as training data
    feature_columns = second_data.select_dtypes(include=['number']).columns.tolist()

# Prepare the second dataset features
X_second = second_data[feature_columns]

# Apply model.predict() to the second data
predictions = trained_model.predict(X_second)

# Add predictions to the second dataset
scored_data = second_data.copy()
scored_data['Predict'] = predictions

# The scored_data now contains the original data plus predictions
`,
  EvaluateModel: `
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# This module evaluates model performance using the scored data from ScoreModel.
# The scored data should contain both actual values (label_column) and predictions (prediction_column).
# Parameters from UI
p_label_column = {label_column}
p_prediction_column = {prediction_column}
p_model_type = {model_type}

# Assuming 'scored_data' (from ScoreModel module) is available
# Extract actual values and predictions
y_true = scored_data[p_label_column]
y_pred = scored_data[p_prediction_column]

# Calculate evaluation metrics based on model type
if p_model_type == 'classification':
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("Classification Evaluation Metrics:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall: {recall:.6f}")
    print(f"  F1-Score: {f1:.6f}")
    
else:  # regression
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    evaluation_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print("Regression Evaluation Metrics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  R-squared (R²): {r2:.6f}")

# evaluation_metrics contains all calculated statistics
`,
  StatModels: `
# This module configures the type of statistical model to be used from the statsmodels library.
# The actual model fitting occurs in the 'Result Model' module.

# Parameters from UI
selected_model_type = {model}

print(f"Selected statsmodels type: {selected_model_type}")
`,

  ResultModel: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

def fit_count_regression_statsmodels(df: pd.DataFrame, distribution_type: str, feature_columns: list, label_column: str, 
                                     max_iter: int = 100, disp: float = 1.0):
    """
    statsmodels를 사용하여 포아송, 음이항, Quasi-Poisson 회귀 모델을 피팅합니다.
    """
    print(f"{distribution_type} 회귀 모델 피팅 중...")
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    # 결측치 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다. 결측치를 확인하세요.")
    
    X = sm.add_constant(X, prepend=True)
    
    try:
        if distribution_type == 'Poisson':
            model = sm.Poisson(y, X)
            results = model.fit(maxiter=max_iter)
        elif distribution_type == 'NegativeBinomial':
            model = sm.NegativeBinomial(y, X, loglike_method='nb2')
            results = model.fit(maxiter=max_iter, disp=disp)
        elif distribution_type == 'QuasiPoisson':
            # Quasi-Poisson은 GLM을 사용하여 구현
            model = sm.GLM(y, X, family=sm.families.Poisson())
            results = model.fit(maxiter=max_iter)
            # Quasi-Poisson은 분산을 과분산 파라미터로 조정
            mu = results.mu
            pearson_resid = (y - mu) / np.sqrt(mu)
            phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
            results.scale = phi
        else:
            raise ValueError(f"지원하지 않는 분포 타입: {distribution_type}")
        
        # 모델 요약 텍스트 생성
        summary_text = str(results.summary())
        print(f"\\n--- {distribution_type} 회귀 모델 결과 ---")
        print(summary_text)
        
        # 통계량 추출
        metrics = {}
        metrics['Log Likelihood'] = results.llf if hasattr(results, 'llf') else None
        metrics['AIC'] = results.aic if hasattr(results, 'aic') else None
        metrics['BIC'] = results.bic if hasattr(results, 'bic') else None
        metrics['Deviance'] = results.deviance if hasattr(results, 'deviance') else None
        metrics['Pearson chi2'] = results.pearson_chi2 if hasattr(results, 'pearson_chi2') else None
        
        # 음이항 회귀의 경우 dispersion 파라미터 추가
        if distribution_type == 'NegativeBinomial':
            if hasattr(model, 'alpha'):
                metrics['Dispersion (alpha)'] = model.alpha
            elif hasattr(results, 'alpha'):
                metrics['Dispersion (alpha)'] = results.alpha
        
        # Quasi-Poisson의 경우 과분산 파라미터 추가
        if distribution_type == 'QuasiPoisson':
            if hasattr(results, 'scale'):
                metrics['Dispersion (phi)'] = results.scale
        
        # 계수 정보 추출
        coefficients = {}
        if hasattr(results, 'params'):
            params = results.params
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            else:
                params_dict = {name: params.iloc[i] if hasattr(params, 'iloc') else params[i] 
                               for i, name in enumerate(results.model.exog_names)}
            
            # 표준 오차, z/t 통계량, p-value, 신뢰구간 추출
            if hasattr(results, 'bse'):
                bse = results.bse
                if hasattr(bse, 'to_dict'):
                    bse_dict = bse.to_dict()
                else:
                    bse_dict = {name: bse.iloc[i] if hasattr(bse, 'iloc') else bse[i] 
                               for i, name in enumerate(results.model.exog_names)}
            else:
                bse_dict = {name: 0.0 for name in params_dict.keys()}
            
            if hasattr(results, 'tvalues'):
                tvalues = results.tvalues
            elif hasattr(results, 'zvalues'):
                tvalues = results.zvalues
            else:
                tvalues = None
            
            if hasattr(results, 'pvalues'):
                pvalues = results.pvalues
            else:
                pvalues = None
            
            # 신뢰구간 추출
            conf_int = None
            if hasattr(results, 'conf_int'):
                conf_int = results.conf_int()
            
            for param_name in params_dict.keys():
                coef_value = params_dict[param_name]
                std_err = bse_dict.get(param_name, 0.0)
                z_value = tvalues[param_name] if tvalues is not None and param_name in tvalues.index else 0.0
                p_value = pvalues[param_name] if pvalues is not None and param_name in pvalues.index else 1.0
                
                conf_lower = conf_int.loc[param_name, 0] if conf_int is not None and param_name in conf_int.index else 0.0
                conf_upper = conf_int.loc[param_name, 1] if conf_int is not None and param_name in conf_int.index else 0.0
                
                coefficients[param_name] = {
                    'coef': float(coef_value),
                    'std err': float(std_err),
                    'z': float(z_value),
                    'P>|z|': float(p_value),
                    '[0.025': float(conf_lower),
                    '0.975]': float(conf_upper)
                }
        
        return {
            'results': results,
            'summary_text': summary_text,
            'metrics': metrics,
            'coefficients': coefficients,
            'distribution_type': distribution_type
        }
        
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_stats_model(df: pd.DataFrame, model_type: str, feature_columns: list, label_column: str):
    """
    statsmodels를 사용하여 통계 모델을 피팅합니다.
    Count regression 모델(Poisson, NegativeBinomial, QuasiPoisson)은 fit_count_regression_statsmodels를 사용합니다.
    """
    # Count regression 모델의 경우 fit_count_regression_statsmodels 사용
    if model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        max_iter = 100
        disp = 1.0
        model_results = fit_count_regression_statsmodels(
            df, model_type, feature_columns, label_column, max_iter, disp
        )
        
        # 통계량 출력
        print("\\n=== 모델 통계량 ===")
        for key, value in model_results['metrics'].items():
            if value is not None:
                print(f"{key}: {value:.6f}")
        
        print("\\n=== 계수 정보 ===")
        for param_name, coef_info in model_results['coefficients'].items():
            print(f"{param_name}:")
            print(f"  계수: {coef_info['coef']:.6f}")
            print(f"  표준 오차: {coef_info['std err']:.6f}")
            print(f"  z-통계량: {coef_info['z']:.6f}")
            print(f"  p-value: {coef_info['P>|z|']:.6f}")
            print(f"  신뢰구간: [{coef_info['[0.025']:.6f}, {coef_info['0.975]']:.6f}]")
        
        return model_results['results']
    
    # 다른 모델의 경우 기존 방식 사용
    print(f"{model_type} 모델 피팅 중...")
    
    X = df[feature_columns]
    y = df[label_column]
    X = sm.add_constant(X, prepend=True)
    
    if model_type == 'OLS':
        model = sm.OLS(y, X)
    elif model_type == 'Logit':
        model = sm.Logit(y, X)
    elif model_type == 'Gamma':
        model = sm.GLM(y, X, family=sm.families.Gamma())
    elif model_type == 'Tweedie':
        model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
    else:
        print(f"오류: 알 수 없는 모델 타입 '{model_type}'")
        return None
    
    try:
        results = model.fit()
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        return None

# Assuming 'dataframe' is passed from a data module.
# The 'model_type' would be passed from the connected 'Stat Models' module.
# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
# p_model_type = 'OLS'  # This would be set dynamically based on Stat Models output

# Execution
# model_results = run_stats_model(
#     dataframe,
#     p_model_type,
#     p_feature_columns,
#     p_label_column
# )
`,
  PredictModel: `
import pandas as pd
import statsmodels.api as sm

def predict_with_statsmodel(results, df: pd.DataFrame, feature_columns: list):
    """
    Applies a fitted statsmodels result object to a new dataset to generate predictions.
    
    Parameters:
    -----------
    results : 모델 결과 객체
        피팅된 모델 결과 (ResultModel에서 생성)
    df : pd.DataFrame
        예측할 데이터
    feature_columns : list
        특성 컬럼 리스트 (모델 피팅 시 사용한 컬럼과 동일해야 함)
    
    Returns:
    --------
    pd.DataFrame
        예측 결과가 추가된 데이터프레임
    """
    print("statsmodels 모델로 예측 수행 중...")
    
    # 특성 컬럼만 선택
    X = df[feature_columns].copy()
    
    # 상수항 추가 (모델 피팅 시와 동일한 방식)
    X = sm.add_constant(X, prepend=True, has_constant='add')
    
    # 모델의 특성 순서에 맞춰 정렬 (모델 피팅 시 사용한 순서와 일치해야 함)
    required_cols = results.model.exog_names
    X_aligned = X.reindex(columns=required_cols).fillna(0)
    
    # 예측 수행
    predictions = results.predict(X_aligned)
    
    # 원본 데이터프레임에 예측 결과 추가
    predict_df = df.copy()
    predict_df['Predict'] = predictions
    
    print("예측 완료. 'Predict' 컬럼이 추가되었습니다.")
    print(predict_df.head())
    
    return predict_df

# Assuming 'model_results' (from ResultModel) and a dataframe 'data_to_predict' are available
# Parameters from UI
p_feature_columns = {feature_columns}

# Execution
# predicted_data = predict_with_statsmodel(model_results, data_to_predict, p_feature_columns)
`,
  FitLossDistribution: `
from scipy import stats
import pandas as pd

def fit_loss_distribution(df: pd.DataFrame, loss_column: str, dist_type: str = 'Pareto'):
    """
    손실 데이터에 통계 분포를 피팅합니다.
    """
    print(f"{dist_type} 분포 피팅 중 (컬럼: {loss_column})...")
    loss_data = df[loss_column].dropna()
    
    if dist_type.lower() == 'pareto':
        params = stats.pareto.fit(loss_data, floc=0)
        print(f"Pareto 파라미터 (shape, loc, scale): {params}")
    elif dist_type.lower() == 'lognormal':
        params = stats.lognorm.fit(loss_data, floc=0)
        print(f"Lognormal 파라미터 (shape, loc, scale): {params}")
    else:
        print(f"오류: 지원하지 않는 분포 타입 '{dist_type}'")
        return None
    
    return params

# Assuming 'dataframe' is passed from a previous step
# Parameters from UI
p_loss_column = {loss_column}
p_dist_type = {distribution_type}

# Execution
# fitted_params = fit_loss_distribution(dataframe, p_loss_column, p_dist_type)
`,

  GenerateExposureCurve: `
import numpy as np
from scipy import stats

def generate_exposure_curve(dist_type: str, params: tuple, total_loss: float):
    """
    피팅된 분포로부터 노출 곡선을 생성합니다.
    """
    print("노출 곡선 생성 중...")
    
    if dist_type.lower() == 'pareto':
        dist = stats.pareto(b=params[0], loc=params[1], scale=params[2])
    elif dist_type.lower() == 'lognormal':
        dist = stats.lognorm(s=params[0], loc=params[1], scale=params[2])
    else:
        raise ValueError(f"지원하지 않는 분포: {dist_type}")
    
    max_retention = total_loss * 2  # Go beyond total loss for a full curve
    retention_points = np.linspace(0, max_retention, 100)
    loss_percentages = 1 - dist.cdf(retention_points)
    
    curve_data = list(zip(retention_points, loss_percentages))
    
    print("노출 곡선 생성 완료.")
    return curve_data

# Assuming 'fitted_params' and 'total_loss' are available
# p_dist_type = 'Pareto'  # From FitLossDistribution module
# p_total_loss = 50000000  # From input data
#
# Execution
# exposure_curve = generate_exposure_curve(p_dist_type, fitted_params, p_total_loss)
`,

  PriceXoLLayer: `
import numpy as np

def price_xol_layer(curve_data: list, total_loss: float, retention: float, 
                   limit: float, loading_factor: float = 1.5):
    """
    노출 곡선을 사용하여 XoL 레이어의 가격을 책정합니다.
    """
    print(f"레이어 가격 책정: {limit:,.0f} xs {retention:,.0f}")
    
    retentions, loss_pcts = zip(*curve_data)
    
    pct_at_retention = np.interp(retention, retentions, loss_pcts)
    pct_at_limit_plus_retention = np.interp(retention + limit, retentions, loss_pcts)
    
    layer_loss_pct = pct_at_retention - pct_at_limit_plus_retention
    expected_layer_loss = total_loss * layer_loss_pct
    rate_on_line = (expected_layer_loss / limit) * 100 if limit > 0 else 0
    final_premium = expected_layer_loss * loading_factor
    
    print(f"  - 예상 레이어 손실: {expected_layer_loss:,.2f}")
    print(f"  - Rate on Line (RoL): {rate_on_line:.2f}%")
    print(f"  - 최종 보험료 (로딩 팩터 {loading_factor}): {final_premium:,.2f}")
    
    return final_premium, expected_layer_loss, rate_on_line

# Assuming 'exposure_curve' and 'total_loss' are available
# Parameters from UI
p_retention = {retention}
p_limit = {limit}
p_loading_factor = {loading_factor}
# p_total_loss = 50000000  # from GenerateExposureCurve step

# Execution
# premium, _, _ = price_xol_layer(exposure_curve, p_total_loss, p_retention, p_limit, p_loading_factor)
`,
  XolLoading: `
import pandas as pd

# This is identical to the standard LoadData module but conceptually used for XoL data.
def load_xol_data(file_path: str):
    """
    Loads claims data from a CSV file, expecting columns like 'year', 'loss'.
    """
    print(f"Loading XoL claims data from: {file_path}")
    df = pd.read_csv(file_path)
    print("XoL data loaded successfully.")
    return df

# Parameters from UI
p_file_path = {source}

# Execution
# xol_dataframe = load_xol_data(p_file_path)
`,

  ApplyThreshold: `
import pandas as pd

def apply_loss_threshold(df: pd.DataFrame, threshold: float, loss_col: str):
    """
    Filters out claims that are below the specified threshold.
    """
    print(f"Applying threshold of {threshold:,.0f} to column '{loss_col}'...")
    original_rows = len(df)
    filtered_df = df[df[loss_col] >= threshold].copy()
    retained_rows = len(filtered_df)
    print(f"Retained {retained_rows} of {original_rows} claims.")
    return filtered_df

# Assuming 'xol_dataframe' is passed from the previous step
# Parameters from UI
p_threshold = {threshold}
p_loss_column = {loss_column}

# Execution
# large_claims_df = apply_loss_threshold(xol_dataframe, p_threshold, p_loss_column)
`,

  DefineXolContract: `
# This module defines the parameters for an Excess of Loss (XoL) reinsurance contract.
# These parameters are then used by downstream modules.

# Parameters from UI
p_deductible = {deductible}  # Also known as retention
p_limit = {limit}
p_reinstatements = {reinstatements}
p_agg_deductible = {aggDeductible}
p_expense_ratio = {expenseRatio}

contract_terms = {
    'deductible': p_deductible,
    'limit': p_limit,
    'reinstatements': p_reinstatements,
    'agg_deductible': p_agg_deductible,
    'expense_ratio': p_expense_ratio,
}

print("XoL Contract terms defined:")
print(contract_terms)
`,

  CalculateCededLoss: `
import pandas as pd

def calculate_ceded_loss(df: pd.DataFrame, deductible: float, limit: float, loss_col: str):
    """
    Calculates the ceded loss for each claim based on the contract's deductible and limit.
    """
    print(f"Calculating ceded loss for layer {limit:,.0f} xs {deductible:,.0f}...")
    
    # Ceded loss is the portion of the loss above the deductible, up to the limit.
    df['ceded_loss'] = df[loss_col].apply(
        lambda loss: min(limit, max(0, loss - deductible))
    )
    
    print("'ceded_loss' column added to the dataframe.")
    return df

# Assuming 'large_claims_df' and 'contract_terms' are passed from previous steps
# Parameters from UI
p_loss_column = {loss_column}
# contract_deductible = contract_terms['deductible']
# contract_limit = contract_terms['limit']

# Execution
# ceded_df = calculate_ceded_loss(large_claims_df, contract_deductible, contract_limit, p_loss_column)
`,

  PriceXolContract: `
import pandas as pd
import numpy as np

def price_xol_contract(df: pd.DataFrame, contract: dict, volatility_loading: float,
                       year_column: str, ceded_loss_column: str):
    """
    경험 기반 방법으로 XoL 계약의 가격을 책정합니다.
    """
    print("경험 기반 XoL 계약 가격 책정 중...")
    
    yearly_ceded_losses = df.groupby(year_column)[ceded_loss_column].sum()
    print("\\n연도별 인출 손실:")
    print(yearly_ceded_losses)
    
    expected_loss = yearly_ceded_losses.mean()
    loss_volatility = yearly_ceded_losses.std()
    
    volatility_margin = loss_volatility * (volatility_loading / 100)
    pure_premium = expected_loss + volatility_margin
    
    expense_ratio = contract.get('expense_ratio', 0.3)
    gross_premium = pure_premium / (1 - expense_ratio)
    
    print(f"\\n--- 가격 책정 요약 ---")
    print(f"평균 연도별 인출 손실 (예상 손실): {expected_loss:,.2f}")
    print(f"연도별 손실 표준편차 (변동성): {loss_volatility:,.2f}")
    print(f"변동성 마진 ({volatility_loading}%): {volatility_margin:,.2f}")
    print(f"순 보험료 (손실 + 변동성): {pure_premium:,.2f}")
    print(f"총 보험료 ({expense_ratio*100:.1f}% 비용 로딩): {gross_premium:,.2f}")
    
    return gross_premium

# Assuming 'ceded_df' and 'contract_terms' are passed from previous steps
# Parameters from UI
p_volatility_loading = {volatility_loading}
p_year_column = {year_column}
p_ceded_loss_column = {ceded_loss_column}

# Execution
# final_price = price_xol_contract(
#     ceded_df, 
#     contract_terms, 
#     p_volatility_loading, 
#     p_year_column, 
#     p_ceded_loss_column
# )
`,

  DiversionChecker: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

def fit_count_regression_statsmodels(df: pd.DataFrame, distribution_type: str, feature_columns: list, label_column: str, 
                                     max_iter: int = 100, disp: float = 1.0):
    """
    statsmodels를 사용하여 포아송, 음이항, Quasi-Poisson 회귀 모델을 피팅합니다.
    """
    print(f"{distribution_type} 회귀 모델 피팅 중...")
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    # 결측치 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        raise ValueError("유효한 데이터가 없습니다. 결측치를 확인하세요.")
    
    X = sm.add_constant(X, prepend=True)
    
    try:
        if distribution_type == 'Poisson':
            model = sm.Poisson(y, X)
            results = model.fit(maxiter=max_iter)
        elif distribution_type == 'NegativeBinomial':
            model = sm.NegativeBinomial(y, X, loglike_method='nb2')
            results = model.fit(maxiter=max_iter, disp=disp)
        elif distribution_type == 'QuasiPoisson':
            # Quasi-Poisson은 GLM을 사용하여 구현
            model = sm.GLM(y, X, family=sm.families.Poisson())
            results = model.fit(maxiter=max_iter)
            # Quasi-Poisson은 분산을 과분산 파라미터로 조정
            mu = results.mu
            pearson_resid = (y - mu) / np.sqrt(mu)
            phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
            results.scale = phi
        else:
            raise ValueError(f"지원하지 않는 분포 타입: {distribution_type}")
        
        # 모델 요약 텍스트 생성
        summary_text = str(results.summary())
        print(f"\\n--- {distribution_type} 회귀 모델 결과 ---")
        print(summary_text)
        
        # 통계량 추출
        metrics = {}
        metrics['Log Likelihood'] = results.llf if hasattr(results, 'llf') else None
        metrics['AIC'] = results.aic if hasattr(results, 'aic') else None
        metrics['BIC'] = results.bic if hasattr(results, 'bic') else None
        metrics['Deviance'] = results.deviance if hasattr(results, 'deviance') else None
        metrics['Pearson chi2'] = results.pearson_chi2 if hasattr(results, 'pearson_chi2') else None
        
        # 음이항 회귀의 경우 dispersion 파라미터 추가
        if distribution_type == 'NegativeBinomial':
            if hasattr(model, 'alpha'):
                metrics['Dispersion (alpha)'] = model.alpha
            elif hasattr(results, 'alpha'):
                metrics['Dispersion (alpha)'] = results.alpha
        
        # Quasi-Poisson의 경우 과분산 파라미터 추가
        if distribution_type == 'QuasiPoisson':
            if hasattr(results, 'scale'):
                metrics['Dispersion (phi)'] = results.scale
        
        # 계수 정보 추출
        coefficients = {}
        if hasattr(results, 'params'):
            params = results.params
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            else:
                params_dict = {name: params.iloc[i] if hasattr(params, 'iloc') else params[i] 
                               for i, name in enumerate(results.model.exog_names)}
            
            # 표준 오차, z/t 통계량, p-value, 신뢰구간 추출
            if hasattr(results, 'bse'):
                bse = results.bse
                if hasattr(bse, 'to_dict'):
                    bse_dict = bse.to_dict()
                else:
                    bse_dict = {name: bse.iloc[i] if hasattr(bse, 'iloc') else bse[i] 
                               for i, name in enumerate(results.model.exog_names)}
            else:
                bse_dict = {name: 0.0 for name in params_dict.keys()}
            
            if hasattr(results, 'tvalues'):
                tvalues = results.tvalues
            elif hasattr(results, 'zvalues'):
                tvalues = results.zvalues
            else:
                tvalues = None
            
            if hasattr(results, 'pvalues'):
                pvalues = results.pvalues
            else:
                pvalues = None
            
            # 신뢰구간 추출
            conf_int = None
            if hasattr(results, 'conf_int'):
                conf_int = results.conf_int()
            
            for param_name in params_dict.keys():
                coef_value = params_dict[param_name]
                std_err = bse_dict.get(param_name, 0.0)
                z_value = tvalues[param_name] if tvalues is not None and param_name in tvalues.index else 0.0
                p_value = pvalues[param_name] if pvalues is not None and param_name in pvalues.index else 1.0
                
                conf_lower = conf_int.loc[param_name, 0] if conf_int is not None and param_name in conf_int.index else 0.0
                conf_upper = conf_int.loc[param_name, 1] if conf_int is not None and param_name in conf_int.index else 0.0
                
                coefficients[param_name] = {
                    'coef': float(coef_value),
                    'std err': float(std_err),
                    'z': float(z_value),
                    'P>|z|': float(p_value),
                    '[0.025': float(conf_lower),
                    '0.975]': float(conf_upper)
                }
        
        return {
            'results': results,
            'summary_text': summary_text,
            'metrics': metrics,
            'coefficients': coefficients,
            'distribution_type': distribution_type
        }
        
    except Exception as e:
        print(f"모델 피팅 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

def dispersion_checker(df: pd.DataFrame, feature_columns: list, label_column: str, max_iter: int = 100):
    """
    과대산포를 측정하고 적합한 모델을 추천합니다.
    """
    print("=== 과대산포 검사 (Diversion Checker) ===\\n")
    
    # 1. 포아송 모델 적합
    print("1. 포아송 모델 적합 중...")
    poisson_result = fit_count_regression_statsmodels(
        df, 'Poisson', feature_columns, label_column, max_iter, 1.0
    )
    poisson_results = poisson_result['results']
    
    # 2. Dispersion φ 계산
    print("\\n2. Dispersion φ 계산 중...")
    y = df[label_column].copy()
    mask = ~(df[feature_columns].isnull().any(axis=1) | y.isnull())
    y = y[mask]
    mu = poisson_results.mu
    pearson_resid = (y - mu) / np.sqrt(mu)
    phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
    
    print(f"Dispersion φ = {phi:.6f}")
    
    # 3. 모델 추천
    print("\\n3. 모델 추천:")
    if phi < 1.2:
        recommendation = "Poisson"
        print(f"φ < 1.2 → Poisson 모델 추천")
    elif 1.2 <= phi < 2:
        recommendation = "QuasiPoisson"
        print(f"1.2 ≤ φ < 2 → Quasi-Poisson 모델 추천")
    else:
        recommendation = "NegativeBinomial"
        print(f"φ ≥ 2 → Negative Binomial 모델 추천")
    
    # 4. 포아송 vs 음이항 AIC 비교
    print("\\n4. 포아송 vs 음이항 AIC 비교 (보조 기준):")
    poisson_aic = poisson_result['metrics'].get('AIC', None)
    print(f"Poisson AIC: {poisson_aic:.6f}" if poisson_aic else "Poisson AIC: N/A")
    
    print("음이항 모델 적합 중...")
    nb_result = fit_count_regression_statsmodels(
        df, 'NegativeBinomial', feature_columns, label_column, max_iter, 1.0
    )
    nb_aic = nb_result['metrics'].get('AIC', None)
    print(f"Negative Binomial AIC: {nb_aic:.6f}" if nb_aic else "Negative Binomial AIC: N/A")
    
    aic_comparison = None
    if poisson_aic is not None and nb_aic is not None:
        if nb_aic < poisson_aic:
            aic_comparison = "Negative Binomial이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        else:
            aic_comparison = "Poisson이 더 낮은 AIC를 가짐 (더 나은 적합도)"
        print(f"AIC 비교: {aic_comparison}")
    
    # 5. Cameron–Trivedi test
    print("\\n5. Cameron–Trivedi test (최종 확인):")
    # Cameron–Trivedi test: (y - mu)^2 - y를 종속변수로 하는 회귀
    X = df[feature_columns].copy()
    X = X[mask]
    X = sm.add_constant(X, prepend=True)
    
    # 테스트 통계량 계산
    test_stat = (y - mu)**2 - y
    ct_model = sm.OLS(test_stat, X)
    ct_results = ct_model.fit()
    
    # 상수항의 계수와 p-value 확인
    const_coef = ct_results.params.get('const', ct_results.params.iloc[0] if len(ct_results.params) > 0 else 0)
    const_pvalue = ct_results.pvalues.get('const', ct_results.pvalues.iloc[0] if len(ct_results.pvalues) > 0 else 1.0)
    
    print(f"Cameron–Trivedi test 통계량 (상수항 계수): {const_coef:.6f}")
    print(f"Cameron–Trivedi test p-value: {const_pvalue:.6f}")
    
    if const_pvalue < 0.05:
        ct_conclusion = "과대산포가 통계적으로 유의함 (p < 0.05)"
        print(f"결론: {ct_conclusion}")
    else:
        ct_conclusion = "과대산포가 통계적으로 유의하지 않음 (p ≥ 0.05)"
        print(f"결론: {ct_conclusion}")
    
    # 최종 추천
    print("\\n=== 최종 추천 ===")
    print(f"추천 모델: {recommendation}")
    if aic_comparison:
        print(f"AIC 비교: {aic_comparison}")
    print(f"Cameron–Trivedi test: {ct_conclusion}")
    
    return {
        'phi': phi,
        'recommendation': recommendation,
        'poisson_aic': poisson_aic,
        'negative_binomial_aic': nb_aic,
        'aic_comparison': aic_comparison,
        'cameron_trivedi_coef': const_coef,
        'cameron_trivedi_pvalue': const_pvalue,
        'cameron_trivedi_conclusion': ct_conclusion,
        'methods_used': [
            '1. 포아송 모델 적합',
            '2. Dispersion φ 계산',
            '3. φ 기준 모델 추천',
            '4. 포아송 vs 음이항 AIC 비교',
            '5. Cameron–Trivedi test'
        ],
        'results': {
            'phi': phi,
            'phi_interpretation': f"φ = {phi:.6f}",
            'recommendation': recommendation,
            'poisson_aic': poisson_aic,
            'negative_binomial_aic': nb_aic,
            'cameron_trivedi_coef': const_coef,
            'cameron_trivedi_pvalue': const_pvalue,
            'cameron_trivedi_conclusion': ct_conclusion
        }
    }

# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
p_max_iter = {max_iter}

# Execution
result = dispersion_checker(dataframe, p_feature_columns, p_label_column, p_max_iter)
print("\\n=== 분석 완료 ===")
`,
};

export const getModuleCode = (module: CanvasModule | null): string => {
  if (!module) {
    return "# Select a module to view its Python code.";
  }
  const template =
    templates[module.type] || `# Code for ${module.name} is not available.`;
  return replacePlaceholders(template.trim(), module.parameters);
};
