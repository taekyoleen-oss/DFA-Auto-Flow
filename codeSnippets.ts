import { CanvasModule, Connection } from './types';

const replacePlaceholders = (template: string, params: Record<string, any>): string => {
  let code = template;
  for (const key in params) {
    const placeholder = new RegExp(`{${key}}`, 'g');
    let value = params[key];
    // Stringify only if it's not already a string that looks like code
    if (value === null) {
        value = 'None';
    } else if (typeof value !== 'string' || !isNaN(Number(value))) {
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

# This module creates a decision tree model instance.
# The model will be trained in the 'Train Model' module.
# Parameters from UI
p_model_purpose = {model_purpose}
p_criterion = {criterion}
p_max_depth = {max_depth} if {max_depth} else None
p_min_samples_split = {min_samples_split}
p_min_samples_leaf = {min_samples_leaf}

# Create model instance based on model purpose
if p_model_purpose == 'classification':
    criterion_clf = p_criterion.lower() if p_criterion else 'gini'
        model = DecisionTreeClassifier(
        criterion=criterion_clf,
        max_depth=p_max_depth,
        min_samples_split=p_min_samples_split,
        min_samples_leaf=p_min_samples_leaf,
            random_state=42
        )
    else:
    criterion_reg = 'squared_error' if p_criterion == 'mse' else 'absolute_error'
        model = DecisionTreeRegressor(
            criterion=criterion_reg,
        max_depth=p_max_depth,
        min_samples_split=p_min_samples_split,
        min_samples_leaf=p_min_samples_leaf,
            random_state=42
        )
    
print(f"Decision Tree model instance created successfully ({p_model_purpose}).")
print(f"  Criterion: {p_criterion}")
print(f"  Max Depth: {p_max_depth}")
print(f"  Min Samples Split: {p_min_samples_split}")
print(f"  Min Samples Leaf: {p_min_samples_leaf}")

# Note: The model is not fitted here. It will be fitted in the 'Train Model' module.
# model variable contains the model instance ready for training.
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
    OLSModel: `
import statsmodels.api as sm

# This module defines an OLS (Ordinary Least Squares) regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI: None (OLS has no configurable parameters)

print("OLS Model definition created successfully.")
print("This model will be instantiated and fitted using statsmodels.OLS in the Result Model module.")
`,
    LogisticModel: `
import statsmodels.api as sm

# This module defines a Logistic regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI: None (Logistic has no configurable parameters)

print("Logistic Model definition created successfully.")
print("This model will be instantiated and fitted using statsmodels.Logit in the Result Model module.")
`,
    PoissonModel: `
import statsmodels.api as sm

# This module defines a Poisson regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}

print(f"Poisson Model definition created successfully (max_iter={p_max_iter}).")
print("This model will be instantiated and fitted using statsmodels.Poisson in the Result Model module.")
`,
    QuasiPoissonModel: `
import statsmodels.api as sm

# This module defines a Quasi-Poisson regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}

print(f"Quasi-Poisson Model definition created successfully (max_iter={p_max_iter}).")
print("This model will be instantiated and fitted using statsmodels.GLM with Poisson family in the Result Model module.")
`,
    NegativeBinomialModel: `
import statsmodels.api as sm

# This module defines a Negative Binomial regression model.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
p_max_iter = {max_iter}
p_disp = {disp}

print(f"Negative Binomial Model definition created successfully (max_iter={p_max_iter}, disp={p_disp}).")
print("This model will be instantiated and fitted using statsmodels.NegativeBinomial in the Result Model module.")
`,
    StatModels: `
import statsmodels.api as sm

# This module configures advanced statistical models (Gamma, Tweedie) from the statsmodels library.
# The model instance will be created in the 'Result Model' module using this definition.
# Parameters from UI
selected_model_type = {model}

print(f"Stat Models definition created successfully (model type: {selected_model_type}).")
print("This model will be instantiated and fitted in the Result Model module.")
`,

    ResultModel: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_stats_model(df: pd.DataFrame, model_type: str, feature_columns: list, label_column: str, max_iter: int = 100, disp: float = 1.0):
    """
    statsmodels를 사용하여 통계 모델을 피팅합니다.
    """
    print(f"{model_type} 모델 피팅 중...")
    
    X = df[feature_columns]
    y = df[label_column]
    X = sm.add_constant(X, prepend=True)
    
    if model_type == 'OLS':
        model = sm.OLS(y, X)
    elif model_type == 'Logit' or model_type == 'Logistic':
        model = sm.Logit(y, X)
    elif model_type == 'Poisson':
        model = sm.Poisson(y, X)
        results = model.fit(maxiter=max_iter)
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    elif model_type == 'QuasiPoisson':
        model = sm.GLM(y, X, family=sm.families.Poisson())
        results = model.fit(maxiter=max_iter)
        # Quasi-Poisson은 분산을 과분산 파라미터로 조정
        mu = results.mu
        pearson_resid = (y - mu) / np.sqrt(mu)
        phi = np.sum(pearson_resid**2) / (len(y) - len(feature_columns) - 1)
        results.scale = phi
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
    elif model_type == 'NegativeBinomial':
        model = sm.NegativeBinomial(y, X, loglike_method='nb2')
        results = model.fit(maxiter=max_iter, disp=disp)
        print(f"\\n--- {model_type} 모델 결과 ---")
        print(results.summary())
        return results
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
# The 'model_type' would be passed from the connected model definition module.
# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
# p_model_type = 'OLS'  # This would be set dynamically based on model definition output

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

def predict_with_statsmodel(results, df: pd.DataFrame):
    """
    Applies a fitted statsmodels result object to a new dataset to generate predictions.
    """
    print("Generating predictions with the fitted statsmodels model...")
    
    # Ensure the 'const' column is present for the intercept
    df_with_const = sm.add_constant(df, prepend=True, has_constant='raise')
    
    # Ensure columns in the prediction data match the model's exog names
    # and are in the same order.
    required_cols = results.model.exog_names
    df_aligned = df_with_const.reindex(columns=required_cols).fillna(0)

    predictions = results.predict(df_aligned)
    
    predict_df = df.copy()
    predict_df['Predict'] = predictions
    
    print("Prediction complete. 'Predict' column added.")
    print(predict_df.head())
    
    return predict_df

# Assuming 'model_results' (from ResultModel) and a dataframe 'data_to_predict' are available
#
# Execution
# predicted_data = predict_with_statsmodel(model_results, data_to_predict)
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
p_default_reinstatement_rate = {defaultReinstatementRate}  # Default reinstatement rate (100% or 0%)
p_year_rates = {yearRates}  # List of {year: int, rate: float} for specific years

# Calculate reinstatement premiums (보장금액 × 비율(%) = 복원보험료)
# All reinstatements use default rate, except for years specified in yearRates
reinstatement_premium_rates = []
year_rate_dict = {{yr['year']: yr['rate'] for yr in p_year_rates}} if p_year_rates else {{}}

for i in range(1, p_reinstatements + 1):
    # Use year-specific rate if available, otherwise use default rate
    if i in year_rate_dict:
        rate = year_rate_dict[i] / 100.0
    else:
        rate = p_default_reinstatement_rate / 100.0
    reinstatement_premium_rates.append(rate)

# Calculate reinstatement premiums (보장금액 × 비율)
reinstatement_premiums = [p_limit * rate for rate in reinstatement_premium_rates]

contract_terms = {{
    'deductible': p_deductible,
    'limit': p_limit,
    'reinstatements': p_reinstatements,
    'agg_deductible': p_agg_deductible,
    'expense_ratio': p_expense_ratio,
    'default_reinstatement_rate': p_default_reinstatement_rate,
    'reinstatement_premium_rates': reinstatement_premium_rates,
    'reinstatement_premiums': reinstatement_premiums,
}}

print("XoL Contract terms defined:")
print(contract_terms)
print(f"Default Reinstatement Rate: {{p_default_reinstatement_rate}}%")
print(f"Year-specific Rates: {{year_rate_dict}}")
print(f"Reinstatement Premium Rates: {{reinstatement_premium_rates}}")
print(f"Reinstatement Premiums (보장금액 × 비율): {{reinstatement_premiums}}")
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

    XolCalculator: `
import pandas as pd
import numpy as np

def calculate_xol_claim(df: pd.DataFrame, contract: dict, claim_col: str):
    """
    Calculates XoL Claim (Incl. Limit) for each claim based on contract terms.
    Formula: Max(min(클레임-Deductible, Limit), 0)
    """
    deductible = contract.get('deductible', 0)
    limit = contract.get('limit', 0)
    
    print(f"Calculating XoL Claim (Incl. Limit) for layer {limit:,.0f} xs {deductible:,.0f}...")
    
    # Calculate XoL Claim (Incl. Limit): Max(min(클레임-Deductible, Limit), 0)
    df['XoL Claim(Incl. Limit)'] = df[claim_col].apply(
        lambda claim: max(min(claim - deductible, limit), 0)
    )
    
    # Calculate statistics
    total_claim = df[claim_col].sum()
    total_xol_claim = df['XoL Claim(Incl. Limit)'].sum()
    xol_ratio = (total_xol_claim / total_claim * 100) if total_claim > 0 else 0
    
    print(f"Total Claim: {total_claim:,.2f}")
    print(f"Total XoL Claim (Incl. Limit): {total_xol_claim:,.2f}")
    print(f"XoL Ratio: {xol_ratio:.2f}%")
    print("'XoL Claim(Incl. Limit)' column added to the dataframe.")
    
    return df

# Assuming 'dataframe' and 'contract_terms' are passed from previous steps
# Parameters from UI
p_claim_column = {claim_column}
# contract_terms = {{'deductible': 250000, 'limit': 1000000, ...}}

# Execution
# result_df = calculate_xol_claim(dataframe, contract_terms, p_claim_column)
`,
    DiversionChecker: `
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Parameters from UI
p_feature_columns = {feature_columns}
p_label_column = {label_column}
p_max_iter = {max_iter}

# Assuming 'dataframe' is passed from a data module.
# Extract features and label
X = dataframe[p_feature_columns]
y = dataframe[p_label_column]

# 과대산포 검사 (Diversion Checker) 실행
# 이 모듈은 dispersion_checker 함수를 사용하여 과대산포를 측정하고
# 적합한 회귀 모델을 추천합니다.

print("=== 과대산포 검사 (Diversion Checker) ===")
print("이 모듈은 다음을 수행합니다:")
print("1. 포아송 모델 적합")
print("2. Dispersion φ 계산")
print("3. φ 기준 모델 추천")
print("4. 포아송 vs 음이항 AIC 비교")
print("5. Cameron–Trivedi test")
print("\\n실제 실행은 'Run' 버튼을 클릭하면 수행됩니다.")
`,
    KNN: `
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def create_knn_model(model_purpose: str = 'classification', n_neighbors: int = 3,
                     weights: str = 'uniform', algorithm: str = 'auto', metric: str = 'minkowski'):
    """
    Creates a K-Nearest Neighbors model using scikit-learn.
    This model uses k nearest neighbors to make predictions.
    It can handle both classification and regression tasks.
    """
    print(f"Creating K-Nearest Neighbors model ({model_purpose})...")
    
    if model_purpose == 'classification':
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
    else:
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
    
    print("Model created successfully.")
    return model

# Parameters from UI
p_model_purpose = {model_purpose}
p_n_neighbors = {n_neighbors}
p_weights = {weights}
p_algorithm = {algorithm}
p_metric = {metric}

# Execution
# knn_model = create_knn_model(p_model_purpose, p_n_neighbors, p_weights, p_algorithm, p_metric)

# This module defines the intent to use a scikit-learn KNeighborsClassifier/KNeighborsRegressor model.
# The actual training happens when this model is connected to a 'Train Model' module.
print(f"sklearn.neighbors.KNeighbors{'Classifier' if p_model_purpose == 'classification' else 'Regressor'} model configured.")
`,

    LoadClaimData: `
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Parameters from UI
p_start_year = {start_year}
p_end_year = {end_year}
p_claims_per_year = {claims_per_year}

# 클레임 데이터 자동 생성
categories = ["자동차보험", "화재보험", "상해보험", "배상책임보험", "건강보험"]
accident_types = [
    "교통사고", "화재사고", "낙상사고", "물건손상", "도난사고",
    "상해사고", "질병", "재산손해", "배상책임", "기타사고"
]

rows = []
for year in range(p_start_year, p_end_year + 1):
    for i in range(p_claims_per_year):
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = datetime(year, month, day)
        category = random.choice(categories)
        base_amount = np.random.lognormal(mean=13.5, sigma=0.8)
        claim_amount = max(100000, int(base_amount))
        accident_content = random.choice(accident_types)
        
        rows.append({
            "종목구분": category,
            "날짜": date.strftime("%Y-%m-%d"),
            "클레임 금액": claim_amount,
            "기타": f"{accident_content} - {random.randint(1, 1000)}번 사고"
        })

dataframe = pd.DataFrame(rows)
print(f"생성된 클레임 데이터: {len(dataframe)}건")
print(dataframe.head())
`,

    ApplyInflation: `
import pandas as pd
import numpy as np
from datetime import datetime

# Parameters from UI
p_target_year = {target_year}
p_inflation_rate = {inflation_rate}  # %
p_amount_column = {amount_column}
p_year_column = {year_column}

# Assuming 'dataframe' is passed from previous step
# Year Column을 숫자로 변환 (이미 연도인 경우)
dataframe[p_year_column] = pd.to_numeric(dataframe[p_year_column], errors='coerce')

# 인플레이션 적용: Amount Column * (1 + Inflation Rate) ^ (Target Year - Year Column)
# 컬럼 이름: 기존 금액 컬럼명 + "_infl"
inflation_factor = (1 + p_inflation_rate / 100.0) ** (p_target_year - dataframe[p_year_column])
inflated_column_name = f"{p_amount_column}_infl"
dataframe[inflated_column_name] = (dataframe[p_amount_column] * inflation_factor).astype(int)

print(f"인플레이션 적용 완료 (목표 연도: {p_target_year}, 상승률: {p_inflation_rate}%)")
print(dataframe.head())
`,

    FormatChange: `
import pandas as pd
from datetime import datetime

def format_change(df: pd.DataFrame, date_column: str):
    # 날짜 컬럼을 datetime으로 변환
    df[date_column] = pd.to_datetime(df[date_column])
    
    # 연도 추출하여 새로운 컬럼 추가 (날짜 컬럼 옆에)
    year_column = "연도"
    df[year_column] = df[date_column].dt.year
    
    # 날짜 컬럼을 문자열로 변환 (원본 형식 유지)
    df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    
    # 컬럼 순서 재배치: 날짜 컬럼 다음에 연도 컬럼 배치
    cols = df.columns.tolist()
    date_idx = cols.index(date_column)
    cols.remove(year_column)
    cols.insert(date_idx + 1, year_column)
    df = df[cols]
    
    return df

# Execution
# Assuming 'dataframe' is passed from the previous step
formatted_dataframe = format_change(dataframe, {date_column})
print("날짜 형식 변경 완료")
print(formatted_dataframe.head())
`,

    SplitByThreshold: `
import pandas as pd
import numpy as np
from datetime import datetime

# Parameters from UI
p_threshold = {threshold}
p_amount_column = {amount_column}
p_year_column = {year_column}

# Assuming 'dataframe' is passed from previous step
# Threshold 기준으로 분리
below_threshold_df = dataframe[dataframe[p_amount_column] < p_threshold].copy()
above_threshold_df = dataframe[dataframe[p_amount_column] >= p_threshold].copy()

# 첫 번째 출력: Year Column을 기준으로 GroupBy하고 Amount Column의 합계 계산
if p_year_column and p_year_column != "" and p_year_column in below_threshold_df.columns:
    below_grouped = below_threshold_df.groupby(p_year_column)[p_amount_column].sum().reset_index()
    below_grouped.columns = [p_year_column, p_amount_column]
else:
    below_grouped = below_threshold_df.copy()

# 두 번째 출력: Threshold보다 크거나 같은 금액의 원본 데이터
above_result = above_threshold_df.copy()

print(f"Threshold 분리 완료 (기준: {p_threshold:,}원)")
print(f"첫 번째 출력 (Threshold 미만, 연도별 합계): {len(below_grouped)}건")
print(below_grouped)
print(f"\\n두 번째 출력 (Threshold 이상, 원본 데이터): {len(above_result)}건")
print(above_result.head())
`,

    SplitByFreqServ: `
import pandas as pd
import numpy as np
from datetime import datetime

# Parameters from UI
p_amount_column = {amount_column}
p_date_column = {date_column}

# Assuming 'dataframe' is passed from previous step
# 날짜 컬럼을 datetime으로 변환
dataframe[p_date_column] = pd.to_datetime(dataframe[p_date_column])
dataframe['year'] = dataframe[p_date_column].dt.year

# 양수 값만 사용
dataframe = dataframe[dataframe[p_amount_column] > 0].copy()

# 1. 빈도 데이터: 연도별 클레임 건수
yearly_frequency_df = dataframe.groupby('year').size().reset_index(name='count')

# 2. 심도 데이터: 개별 클레임 금액 (원본 데이터 유지)
severity_df = dataframe.copy()
if 'year' in severity_df.columns:
    severity_df = severity_df.drop(columns=['year'])

print(f"빈도-심도 분리 완료")
print(f"연도별 빈도 데이터: {len(yearly_frequency_df)}개 연도")
print(yearly_frequency_df)
print(f"\\n심도 데이터: {len(severity_df)}건")
print(severity_df.head())
`,

    FitAggregateModel: `
import pandas as pd
import numpy as np
from scipy import stats

# 선택된 열에서 데이터 가져오기
amounts = dataframe['{amount_column}'].values

# 양수 값만 사용
amounts = amounts[amounts > 0]

# 분포 적합
if "{distribution_type}" == "Lognormal":
    params = stats.lognorm.fit(amounts, floc=0)
elif "{distribution_type}" == "Exponential":
    params = stats.expon.fit(amounts, floc=0)
elif "{distribution_type}" == "Pareto":
    params = stats.pareto.fit(amounts, floc=0)
elif "{distribution_type}" == "Gamma":
    params = stats.gamma.fit(amounts, floc=0)
`,

    SimulateAggDist: `
import numpy as np
from scipy import stats

# 분포 객체 생성
if "{distribution_type}" == "Lognormal":
    dist = stats.lognorm(
        s={parameters}["shape"],
        scale={parameters}["scale"],
        loc={parameters}["loc"]
    )
elif "{distribution_type}" == "Exponential":
    dist = stats.expon(
        scale={parameters}["scale"],
        loc={parameters}["loc"]
    )
elif "{distribution_type}" == "Pareto":
    dist = stats.pareto(
        b={parameters}["shape"],
        scale={parameters}["scale"],
        loc={parameters}["loc"]
    )
elif "{distribution_type}" == "Gamma":
    dist = stats.gamma(
        a={parameters}["shape"],
        scale={parameters}["scale"],
        loc={parameters}["loc"]
    )

# 시뮬레이션
np.random.seed(42)
simulated_amounts = dist.rvs(size={n_simulations})
`,
    SimulateFreqServ: `
import numpy as np
from scipy import stats

# 빈도 분포 객체 생성
freq_type = frequency_params["type"]

if freq_type == "Poisson":
    lambda_param = frequency_params["lambda"]
    freq_dist = stats.poisson(lambda_param)

elif freq_type == "NegativeBinomial":
    n = frequency_params["n"]
    p = frequency_params["p"]
    freq_dist = stats.nbinom(n, p)

else:
    raise ValueError(f"Unknown frequency distribution: {freq_type}")

# 심도 분포 객체 생성
sev_type = severity_params["type"]

if sev_type == "Normal":
    mean = severity_params["mean"]
    std = severity_params["std"]
    sev_dist = stats.norm(loc=mean, scale=std)

elif sev_type == "Lognormal":
    s = severity_params["shape"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.lognorm(s=s, scale=scale, loc=loc)

elif sev_type == "Exponential":
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.expon(scale=scale, loc=loc)

elif sev_type == "Pareto":
    shape = severity_params["shape"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.pareto(b=shape, scale=scale, loc=loc)

elif sev_type == "Gamma":
    shape = severity_params["shape"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.gamma(a=shape, scale=scale, loc=loc)

elif sev_type == "Weibull":
    shape = severity_params["shape"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.weibull_min(c=shape, scale=scale, loc=loc)

elif sev_type == "GeneralizedPareto":
    shape = severity_params["shape"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.genpareto(c=shape, scale=scale, loc=loc)

elif sev_type == "Burr":
    c = severity_params["c"]
    d = severity_params["d"]
    scale = severity_params["scale"]
    loc = severity_params.get("loc", 0.0)
    sev_dist = stats.burr12(c=c, d=d, scale=scale, loc=loc)

else:
    raise ValueError(f"Unknown severity distribution: {sev_type}")

# 몬테카를로 시뮬레이션
np.random.seed(42)
aggregate_losses = []

for i in range(n_simulations):
    frequency_count = int(freq_dist.rvs())
    frequency_count = max(0, frequency_count)

    if frequency_count == 0:
        total_loss = 0.0
    else:
        severities = sev_dist.rvs(size=frequency_count)
        severities = np.maximum(severities, 0.0)
        total_loss = float(np.sum(severities))

    aggregate_losses.append(total_loss)

# 통계량 계산
mean_amount = np.mean(aggregate_losses)
std_amount = np.std(aggregate_losses)
min_amount = np.min(aggregate_losses)
max_amount = np.max(aggregate_losses)
percentiles = np.percentile(aggregate_losses, [5, 25, 50, 75, 95, 99])
`,
    SelectDist: `
# Parameters from UI
p_distribution_type = {distribution_type}

# Assuming 'aggregate_model_output' is passed from FitAggregateModel
# 선택된 분포의 결과 추출
selected_result = None
for result in aggregate_model_output.results:
    if result.distribution_type == p_distribution_type:
        selected_result = result
        break

if selected_result is None:
    raise ValueError(f"Distribution '{p_distribution_type}' not found in results")

print(f"=== Selected Distribution: {p_distribution_type} ===")
print(f"Parameters:")
for key, value in selected_result.parameters.items():
    print(f"  {key}: {value:.6f}")

print(f"\\nFit Statistics:")
print(f"  AIC: {selected_result.fit_statistics.get('aic', 'N/A')}")
print(f"  BIC: {selected_result.fit_statistics.get('bic', 'N/A')}")
print(f"  Log Likelihood: {selected_result.fit_statistics.get('log_likelihood', 'N/A')}")
`,

    FitFrequencyModel: `
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Parameters from UI
p_count_column = {count_column}
p_frequency_types = {selected_frequency_types}

# 선택된 열에서 건수 데이터 가져오기
df = pd.DataFrame(dataframe)
counts = pd.to_numeric(df[p_count_column], errors='coerce').dropna().values

# 양수 값만 사용 (건수는 0 이상이어야 함)
counts = counts[counts >= 0]

if len(counts) == 0:
    raise ValueError(f"No valid count data found in column '{p_count_column}'")

print(f"=== Frequency Model Fitting ===")
print(f"Count Column: {p_count_column}")
print(f"Number of observations: {len(counts)}")
print(f"Mean count: {np.mean(counts):.2f}")
print(f"Variance: {np.var(counts):.2f}")
print(f"Dispersion (variance/mean): {np.var(counts) / np.mean(counts) if np.mean(counts) > 0 else 'N/A':.2f}")
print()

all_results = []

for dist_type in p_frequency_types:
    print(f"--- Fitting {dist_type} Distribution ---")
    try:
        if dist_type == "Poisson":
            # 포아송 분포: lambda = mean
            lambda_param = np.mean(counts)
            if lambda_param <= 0:
                print(f"  Error: Mean count must be positive for Poisson distribution")
                all_results.append({{"distribution_type": "Poisson", "error": "Mean count must be positive"}})
                continue
            
            dist = stats.poisson(lambda_param)
            params = {{"lambda": float(lambda_param)}}
            
            # 통계량 계산
            log_likelihood = np.sum(dist.logpmf(counts))
            n_params = 1
            n_obs = len(counts)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n_obs) - 2 * log_likelihood
            
            print(f"  Lambda (λ): {lambda_param:.6f}")
            print(f"  Log Likelihood: {log_likelihood:.6f}")
            print(f"  AIC: {aic:.6f}")
            print(f"  BIC: {bic:.6f}")
            
            all_results.append({{
                "distribution_type": "Poisson",
                "parameters": params,
                "fit_statistics": {{
                    "aic": float(aic),
                    "bic": float(bic),
                    "log_likelihood": float(log_likelihood),
                    "mean": float(np.mean(counts)),
                    "variance": float(np.var(counts)),
                    "dispersion": float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0
                }}
            }})
            
        elif dist_type == "NegativeBinomial":
            # 음이항 분포: Method of Moments
            mean_count = np.mean(counts)
            var_count = np.var(counts)
            
            if var_count <= mean_count:
                print(f"  Error: Variance must be greater than mean for Negative Binomial (overdispersion required)")
                print(f"  Mean: {mean_count:.2f}, Variance: {var_count:.2f}")
                all_results.append({{"distribution_type": "NegativeBinomial", "error": "Variance must be greater than mean (overdispersion required)"}})
                continue
            
            # Method of moments estimators
            r_est = mean_count ** 2 / (var_count - mean_count)
            p_est = mean_count / var_count
            
            if r_est <= 0 or p_est <= 0 or p_est >= 1:
                print(f"  Error: Invalid parameters (r={r_est:.2f}, p={p_est:.2f})")
                all_results.append({{"distribution_type": "NegativeBinomial", "error": f"Invalid parameters: r={r_est:.2f}, p={p_est:.2f}"}})
                continue
            
            dist = stats.nbinom(r_est, p_est)
            params = {{"n": float(r_est), "p": float(p_est)}}
            
            # 통계량 계산
            log_likelihood = np.sum(dist.logpmf(counts))
            n_params = 2
            n_obs = len(counts)
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(n_obs) - 2 * log_likelihood
            
            print(f"  n (r): {r_est:.6f}")
            print(f"  p: {p_est:.6f}")
            print(f"  Log Likelihood: {log_likelihood:.6f}")
            print(f"  AIC: {aic:.6f}")
            print(f"  BIC: {bic:.6f}")
            
            all_results.append({{
                "distribution_type": "NegativeBinomial",
                "parameters": params,
                "fit_statistics": {{
                    "aic": float(aic),
                    "bic": float(bic),
                    "log_likelihood": float(log_likelihood),
                    "mean": float(np.mean(counts)),
                    "variance": float(np.var(counts)),
                    "dispersion": float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0
                }}
            }})
        else:
            print(f"  Error: Unknown distribution type '{dist_type}'")
            all_results.append({{"distribution_type": dist_type, "error": f"Unknown distribution type"}})
            
    except Exception as e:
        print(f"  Error fitting {dist_type}: {{str(e)}}")
        all_results.append({{"distribution_type": dist_type, "error": str(e)}})
    
    print()

# 결과 요약
print("=== Fitting Results Summary ===")
successful_results = [r for r in all_results if "error" not in r]
if successful_results:
    print(f"Successfully fitted {len(successful_results)} distribution(s)")
    for result in successful_results:
        dist_name = result["distribution_type"]
        aic = result["fit_statistics"]["aic"]
        print(f"  {dist_name}: AIC = {aic:.2f}")
    
    # AIC 기준으로 최적 분포 선택
    best_result = min(successful_results, key=lambda x: x["fit_statistics"]["aic"])
    print(f"\\nBest fit (lowest AIC): {best_result['distribution_type']}")
    print(f"  AIC: {best_result['fit_statistics']['aic']:.2f}")
    print(f"  Parameters: {best_result['parameters']}")
else:
    print("No distributions were successfully fitted")

# 결과 반환
result = {{
    "results": all_results,
    "yearly_counts": df[[p_count_column]].to_dict('records')
}}

result
`,

    FitFrequencySeverityModel: `
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Parameters from UI
p_frequency_type = {frequency_type}
p_severity_type = {severity_type}
p_amount_column = {amount_column}
p_date_column = {date_column}

# Assuming 'dataframe' is passed from previous step
# 날짜 컬럼을 datetime으로 변환
dataframe[p_date_column] = pd.to_datetime(dataframe[p_date_column])
dataframe['year'] = dataframe[p_date_column].dt.year

# 빈도 모델: 연도별 클레임 건수
yearly_counts = dataframe.groupby('year').size().reset_index(name='count')
counts = yearly_counts['count'].values

# 빈도 분포 적합
if p_frequency_type == "Poisson":
    lambda_param = np.mean(counts)
    freq_params = {"lambda": lambda_param}
    freq_dist = stats.poisson(lambda_param)
    freq_log_likelihood = np.sum(freq_dist.logpmf(counts))
elif p_frequency_type == "NegativeBinomial":
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    if var_count > mean_count:
        r = mean_count ** 2 / (var_count - mean_count)
        p = mean_count / var_count
    else:
        r = 10.0
        p = mean_count / (mean_count + r)
    freq_params = {"n": r, "p": p}
    freq_dist = stats.nbinom(r, p)
    freq_log_likelihood = np.sum(freq_dist.logpmf(counts))

print(f"=== 빈도 모델 ({p_frequency_type}) 적합 결과 ===")
for key, value in freq_params.items():
    print(f"{key}: {value:.6f}")
print(f"Log Likelihood: {freq_log_likelihood:.6f}")

# 심도 모델: 개별 클레임 금액
amounts = dataframe[p_amount_column].values
amounts = amounts[amounts > 0]

# 심도 분포 적합
if p_severity_type == "Normal":
    params = stats.norm.fit(amounts)
    sev_dist = stats.norm
elif p_severity_type == "Lognormal":
    log_amounts = np.log(amounts)
    params = stats.norm.fit(log_amounts)
    sev_dist = stats.lognorm
    params = (params[1], np.exp(params[0]), 0)
elif p_severity_type == "Pareto":
    params, _ = stats.pareto.fit(amounts, floc=0)
    sev_dist = stats.pareto
elif p_severity_type == "Gamma":
    params = stats.gamma.fit(amounts, floc=0)
    sev_dist = stats.gamma
elif p_severity_type == "Exponential":
    params = stats.expon.fit(amounts, floc=0)
    sev_dist = stats.expon
elif p_severity_type == "Weibull":
    params = stats.weibull_min.fit(amounts, floc=0)
    sev_dist = stats.weibull_min
elif p_severity_type == "GeneralizedPareto":
    params = stats.genpareto.fit(amounts, floc=0)
    sev_dist = stats.genpareto
elif p_severity_type == "Burr":
    params = stats.burr12.fit(amounts, floc=0)
    sev_dist = stats.burr12

print(f"\\n=== 심도 모델 ({p_severity_type}) 적합 결과 ===")
print(f"파라미터: {params}")

# 집계 분포 계산
if p_frequency_type == "Poisson":
    E_N = lambda_param
    Var_N = lambda_param
else:
    E_N = r * (1 - p) / p
    Var_N = r * (1 - p) / (p ** 2)

E_X = np.mean(amounts)
Var_X = np.var(amounts)

E_Total = E_N * E_X
Var_Total = E_N * Var_X + Var_N * (E_X ** 2)
Std_Total = np.sqrt(Var_Total)

print(f"\\n=== 집계 분포 ===")
print(f"기대값: {E_Total:,.2f}")
print(f"표준편차: {Std_Total:,.2f}")
`,

    SettingThreshold: `
import pandas as pd
import numpy as np
import json

# Parameters from UI
p_target_column = {target_column}
p_thresholds = {thresholds}  # List of threshold values
p_year_column = {year_column}  # Year column for yearly analysis

# Assuming 'dataframe' is passed from previous step
# Check if target column exists
if p_target_column not in dataframe.columns:
    raise ValueError(f"Column '{p_target_column}' not found in dataframe")

# Get the target column data (numeric only)
target_data = pd.to_numeric(dataframe[p_target_column], errors='coerce')
target_data = target_data.dropna()

if len(target_data) == 0:
    raise ValueError(f"No valid numeric data found in column '{p_target_column}'")

# Sort thresholds in ascending order
p_thresholds = sorted([float(t) for t in p_thresholds if t is not None and str(t).strip() != ''])

if len(p_thresholds) == 0:
    raise ValueError("At least one threshold value must be provided")

# Calculate statistics
statistics = {
    'min': float(target_data.min()),
    'max': float(target_data.max()),
    'mean': float(target_data.mean()),
    'median': float(target_data.median()),
    'std': float(target_data.std()),
    'q25': float(target_data.quantile(0.25)),
    'q75': float(target_data.quantile(0.75))
}

# Calculate threshold results
total_count = len(target_data)
threshold_results = []

# Sort thresholds descending for cumulative calculation
sorted_thresholds = sorted(p_thresholds, reverse=True)
cumulative_count = 0

for threshold in sorted_thresholds:
    count = int((target_data > threshold).sum())
    percentage = (count / total_count * 100) if total_count > 0 else 0.0
    cumulative_count += count
    cumulative_percentage = (cumulative_count / total_count * 100) if total_count > 0 else 0.0
    
    threshold_results.append({
        'threshold': float(threshold),
        'count': count,
        'percentage': float(percentage),
        'cumulativeCount': cumulative_count,
        'cumulativePercentage': float(cumulative_percentage)
    })

# Sort back to ascending for display
threshold_results = sorted(threshold_results, key=lambda x: x['threshold'])

# Prepare data distribution for histogram (sample up to 1000 points for performance)
sample_size = min(1000, len(target_data))
if sample_size < len(target_data):
    sampled_data = target_data.sample(n=sample_size, random_state=42)
else:
    sampled_data = target_data

# Create histogram bins
num_bins = 50
hist_values = sampled_data.values.tolist()
bins = np.linspace(float(target_data.min()), float(target_data.max()), num_bins + 1).tolist()
frequencies, _ = np.histogram(hist_values, bins=bins)
frequencies = frequencies.tolist()

data_distribution = {
    'values': hist_values[:100],  # Limit to 100 values for JSON size
    'bins': bins,
    'frequencies': frequencies
}

# Print results
print("=" * 60)
print(f"Setting Threshold Analysis: {p_target_column}")
print("=" * 60)
print(f"\\nTotal rows: {total_count:,}")
print(f"\\nStatistics:")
print(f"  Min: {statistics['min']:,.2f}")
print(f"  Max: {statistics['max']:,.2f}")
print(f"  Mean: {statistics['mean']:,.2f}")
print(f"  Median: {statistics['median']:,.2f}")
print(f"  Std: {statistics['std']:,.2f}")
print(f"  Q25: {statistics['q25']:,.2f}")
print(f"  Q75: {statistics['q75']:,.2f}")

print(f"\\nThreshold Results:")
print(f"{'Threshold':>15} {'Count':>10} {'Percentage':>12} {'Cumulative':>12} {'Cum %':>10}")
print("-" * 70)
for result in threshold_results:
    print(f"{result['threshold']:>15,.2f} {result['count']:>10,} {result['percentage']:>11.2f}% {result['cumulativeCount']:>12,} {result['cumulativePercentage']:>9.2f}%")

# Calculate yearly counts if year column is provided
yearly_counts = []
if p_year_column and p_year_column.strip() and p_year_column in dataframe.columns:
    # Extract year from year column (handle various formats)
    def extract_year(value):
        if pd.isna(value):
            return None
        value_str = str(value)
        # Try to extract 4-digit year
        import re
        year_match = re.search(r'\\d{4}', value_str)
        if year_match:
            return int(year_match.group())
        # Try to parse as integer
        try:
            year_int = int(float(value_str))
            if 1900 <= year_int <= 2100:
                return year_int
        except:
            pass
        return None
    
    dataframe['_extracted_year'] = dataframe[p_year_column].apply(extract_year)
    dataframe_with_year = dataframe[dataframe['_extracted_year'].notna()].copy()
    
    if len(dataframe_with_year) > 0:
        # Get numeric target data with year
        target_data_with_year = pd.to_numeric(dataframe_with_year[p_target_column], errors='coerce')
        dataframe_with_year['_target_value'] = target_data_with_year
        dataframe_with_year = dataframe_with_year[dataframe_with_year['_target_value'].notna()].copy()
        
        # Group by year and calculate counts for each threshold
        for year in sorted(dataframe_with_year['_extracted_year'].unique()):
            year_data = dataframe_with_year[dataframe_with_year['_extracted_year'] == year]
            year_target = year_data['_target_value']
            
            counts = []
            for threshold in sorted(p_thresholds):
                count = int((year_target >= threshold).sum())
                counts.append(count)
            
            # Calculate totals, mean, std for this year
            year_counts_array = np.array(counts)
            totals = {
                'total': int(year_counts_array.sum()),
                'mean': float(year_counts_array.mean()) if len(year_counts_array) > 0 else 0.0,
                'std': float(year_counts_array.std()) if len(year_counts_array) > 0 else 0.0
            }
            
            yearly_counts.append({
                'year': int(year) if isinstance(year, (int, float)) else str(year),
                'counts': counts,
                'totals': totals
            })
    
    dataframe = dataframe.drop(columns=['_extracted_year'], errors='ignore')

# Output JSON for frontend
output = {
    'type': 'SettingThresholdOutput',
    'targetColumn': p_target_column,
    'thresholds': p_thresholds,
    'selectedThreshold': p_thresholds[0] if len(p_thresholds) > 0 else None,  # 기본값: 첫 번째 threshold
    'thresholdResults': threshold_results,
    'yearlyCounts': yearly_counts if yearly_counts else None,
    'dataDistribution': data_distribution,
    'statistics': statistics
}

print("\\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
print("\\nOutput JSON:")
print(json.dumps(output, indent=2))

# output을 반환 (마지막 줄)
output
`,

    ThresholdAnalysis: `
import pandas as pd
import numpy as np
import json
from scipy import stats

# Parameters from UI
p_target_column = {target_column}

# Assuming 'dataframe' is passed from previous step
# Check if target column exists
if p_target_column not in dataframe.columns:
    raise ValueError(f"Column '{p_target_column}' not found in dataframe")

# Get the target column data (numeric only)
target_data = pd.to_numeric(dataframe[p_target_column], errors='coerce')
target_data = target_data.dropna()

if len(target_data) == 0:
    raise ValueError(f"No valid numeric data found in column '{p_target_column}'")

# Convert to numpy array and sort
data_values = np.sort(target_data.values)
n = len(data_values)

# Calculate basic statistics
statistics = {
    'min': float(data_values[0]),
    'max': float(data_values[-1]),
    'mean': float(np.mean(data_values)),
    'median': float(np.median(data_values)),
    'std': float(np.std(data_values)),
    'q25': float(np.percentile(data_values, 25)),
    'q75': float(np.percentile(data_values, 75)),
    'q90': float(np.percentile(data_values, 90)),
    'q95': float(np.percentile(data_values, 95)),
    'q99': float(np.percentile(data_values, 99))
}

# 1. Histogram
num_bins = 50
hist_values = data_values.tolist()
bins = np.linspace(float(data_values[0]), float(data_values[-1]), num_bins + 1).tolist()
frequencies, _ = np.histogram(hist_values, bins=bins)
frequencies = frequencies.tolist()

histogram = {
    'bins': bins,
    'frequencies': frequencies
}

# 2. ECDF (Empirical Cumulative Distribution Function)
sorted_values = data_values.tolist()
cumulative_probabilities = np.arange(1, n + 1) / n
cumulative_probabilities = cumulative_probabilities.tolist()

ecdf = {
    'sortedValues': sorted_values,
    'cumulativeProbabilities': cumulative_probabilities
}

# 3. QQ-Plot (Quantile-Quantile Plot)
# Normal distribution을 기준으로 QQ-Plot 생성
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
sample_quantiles = data_values

qq_plot = {
    'theoreticalQuantiles': theoretical_quantiles.tolist(),
    'sampleQuantiles': sample_quantiles.tolist()
}

# 4. Mean Excess Plot
# 여러 threshold 값에 대해 Mean Excess 계산
# threshold는 데이터의 percentile 기반으로 생성
percentiles = np.linspace(0, 95, 20)  # 0%부터 95%까지 20개 지점
thresholds = np.percentile(data_values, percentiles).tolist()
mean_excesses = []

for threshold in thresholds:
    excess_data = data_values[data_values > threshold] - threshold
    if len(excess_data) > 0:
        mean_excess = float(np.mean(excess_data))
    else:
        mean_excess = 0.0
    mean_excesses.append(mean_excess)

mean_excess_plot = {
    'thresholds': thresholds,
    'meanExcesses': mean_excesses
}

# Prepare output
output = {
    'type': 'ThresholdAnalysisOutput',
    'targetColumn': p_target_column,
    'data': data_values.tolist()[:1000],  # 최대 1000개만 저장 (성능 고려)
    'histogram': histogram,
    'ecdf': ecdf,
    'qqPlot': qq_plot,
    'meanExcessPlot': mean_excess_plot,
    'statistics': statistics
}

print("=" * 60)
print(f"Threshold Analysis: {p_target_column}")
print("=" * 60)
print(f"\\nTotal data points: {n:,}")
print(f"\\nStatistics:")
for key, value in statistics.items():
    print(f"  {key}: {value:,.2f}")
print(f"\\nHistogram bins: {len(bins)}")
print(f"ECDF points: {len(sorted_values)}")
print(f"QQ-Plot points: {len(theoretical_quantiles)}")
print(f"Mean Excess Plot thresholds: {len(thresholds)}")
print("\\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
print("\\nOutput JSON:")
print(json.dumps(output, indent=2))

# output을 반환 (마지막 줄)
output
`,
};

export const getModuleCode = (
    module: CanvasModule | null,
    allModules?: CanvasModule[],
    connections?: Connection[]
): string => {
    if (!module) {
        return "# Select a module to view its Python code.";
    }
    
    // EvaluateStat의 경우 generateEvaluateStatCode 사용
    if (module.type === "EvaluateStat") {
        return generateEvaluateStatCode(module);
    }
    
    // ResultModel의 경우 연결된 모델 타입에 따라 코드 생성
    if (module.type === "ResultModel" && allModules && connections) {
        const modelInputConnection = connections.find(
            (c) => c.to.moduleId === module.id && c.to.portName === "model_in"
        );
        
        if (modelInputConnection) {
            const modelSourceModule = allModules.find(
                (m) => m.id === modelInputConnection.from.moduleId
            );
            
            if (modelSourceModule) {
                // 모델 타입 확인
                let modelType: string | null = null;
                
                if (modelSourceModule.type === "OLSModel") {
                    modelType = "OLS";
                } else if (modelSourceModule.type === "LogisticModel") {
                    modelType = "Logit";
                } else if (modelSourceModule.type === "PoissonModel") {
                    modelType = "Poisson";
                } else if (modelSourceModule.type === "QuasiPoissonModel") {
                    modelType = "QuasiPoisson";
                } else if (modelSourceModule.type === "NegativeBinomialModel") {
                    modelType = "NegativeBinomial";
                } else if (modelSourceModule.type === "StatModels") {
                    modelType = modelSourceModule.parameters.model || "Gamma";
                } else if (modelSourceModule.outputData?.type === "ModelDefinitionOutput") {
                    modelType = modelSourceModule.outputData.modelType;
                }
                
                if (modelType) {
                    return generateResultModelCode(module, modelType, modelSourceModule.parameters);
                }
            }
        }
    }
    
    const template = templates[module.type] || `# Code for ${module.name} is not available.`;
    return replacePlaceholders(template.trim(), module.parameters);
};

/**
 * ResultModel의 모델 타입에 맞는 코드를 생성합니다
 */
function generateResultModelCode(
    module: CanvasModule,
    modelType: string,
    modelParams: Record<string, any>
): string {
    const { feature_columns, label_column } = module.parameters;
    const max_iter = modelParams.max_iter || 100;
    const disp = modelParams.disp || 1.0;
    
    let code = `import pandas as pd
import numpy as np
import statsmodels.api as sm

# Parameters from UI (Result Model)
p_feature_columns = ${JSON.stringify(feature_columns || [])}
p_label_column = ${label_column ? `'${label_column}'` : 'None'}

# Assuming 'dataframe' is passed from a data module and 'model_definition' is passed from the model definition module.
# Extract features and label
X = dataframe[p_feature_columns]
y = dataframe[p_label_column]
X = sm.add_constant(X, prepend=True)

# Create model instance based on the connected model definition module
`;

    // 모델 타입에 따라 코드 생성 (연결된 모델 정의 모듈의 타입에 따라)
    if (modelType === "OLS") {
        code += `# OLS 모델 인스턴스 생성 및 피팅
# (모델 정의는 OLS Model 모듈에서 제공됨)
model = sm.OLS(y, X)
results = model.fit()

print("\\n--- OLS 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Logit" || modelType === "Logistic") {
        code += `# Logistic 모델 인스턴스 생성 및 피팅
# (모델 정의는 Logistic Model 모듈에서 제공됨)
model = sm.Logit(y, X)
results = model.fit()

print("\\n--- Logistic 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Poisson") {
        code += `# Poisson 모델 인스턴스 생성 및 피팅
# (모델 정의는 Poisson Model 모듈에서 제공됨, max_iter=${max_iter})
model = sm.Poisson(y, X)
results = model.fit(maxiter=${max_iter})

print("\\n--- Poisson 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "QuasiPoisson") {
        code += `# Quasi-Poisson 모델 인스턴스 생성 및 피팅
# (모델 정의는 Quasi-Poisson Model 모듈에서 제공됨, max_iter=${max_iter})
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit(maxiter=${max_iter})

# Quasi-Poisson은 분산을 과분산 파라미터로 조정
mu = results.mu
pearson_resid = (y - mu) / np.sqrt(mu)
phi = np.sum(pearson_resid**2) / (len(y) - len(p_feature_columns) - 1)
results.scale = phi

print("\\n--- Quasi-Poisson 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "NegativeBinomial") {
        code += `# Negative Binomial 모델 인스턴스 생성 및 피팅
# (모델 정의는 Negative Binomial Model 모듈에서 제공됨, max_iter=${max_iter}, disp=${disp})
model = sm.NegativeBinomial(y, X, loglike_method='nb2')
results = model.fit(maxiter=${max_iter}, disp=${disp})

print("\\n--- Negative Binomial 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Gamma") {
        code += `# Gamma 모델 인스턴스 생성 및 피팅
# (모델 정의는 Stat Models 모듈에서 제공됨, model type: Gamma)
model = sm.GLM(y, X, family=sm.families.Gamma())
results = model.fit()

print("\\n--- Gamma 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else if (modelType === "Tweedie") {
        code += `# Tweedie 모델 인스턴스 생성 및 피팅
# (모델 정의는 Stat Models 모듈에서 제공됨, model type: Tweedie)
model = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5))
results = model.fit()

print("\\n--- Tweedie 모델 결과 ---")
print(results.summary())

model_results = results
`;
    } else {
        code += `# 알 수 없는 모델 타입: ${modelType}
print(f"오류: 알 수 없는 모델 타입 '${modelType}'")
model_results = None
`;
    }
    
    return code;
}

/**
 * EvaluateStat 모듈의 코드를 생성합니다
 */
function generateEvaluateStatCode(
    module: CanvasModule
): string {
    const { label_column, prediction_column, model_type } = module.parameters;
    
    let code = `import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Parameters from UI
p_label_column = ${label_column ? `'${label_column}'` : 'None'}
p_prediction_column = ${prediction_column ? `'${prediction_column}'` : 'None'}
p_model_type = ${model_type ? `'${model_type}'` : 'None'}

# Assuming 'dataframe' is passed from a data module.
# Extract actual and predicted values
y_true = dataframe[p_label_column].values
y_pred = dataframe[p_prediction_column].values

# 기본 통계량 계산 (전통적인 방법)
print("=" * 60)
print("통계 모델 평가 (Evaluate Stat)")
print("=" * 60)

# 기본 회귀 메트릭
mse = float(mean_squared_error(y_true, y_pred))
rmse = float(np.sqrt(mse))
mae = float(mean_absolute_error(y_true, y_pred))
r2 = float(r2_score(y_true, y_pred))

print(f"\\n--- 기본 통계량 ---")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared: {r2:.6f}")

# 잔차 계산
residuals = (y_true - y_pred).tolist()
residuals_array = np.array(residuals)

print(f"\\n--- 잔차 통계량 ---")
print(f"Mean Residual: {np.mean(residuals_array):.6f}")
print(f"Std Residual: {np.std(residuals_array):.6f}")
print(f"Min Residual: {np.min(residuals_array):.6f}")
print(f"Max Residual: {np.max(residuals_array):.6f}")

# 모델 타입별 특수 통계량 (선택적)
`;

    if (model_type) {
        code += `
# 모델 타입: ${model_type}
if p_model_type and p_model_type != '' and p_model_type != 'None':
    print(f"\\n--- ${model_type} 모델 특수 통계량 ---")
    
    if p_model_type in ['Poisson', 'NegativeBinomial', 'QuasiPoisson']:
        # Count regression 모델 통계량
        mu = np.maximum(y_pred, 1e-10)  # 0 방지
        deviance_val = 2 * np.sum(y_true * np.log(np.maximum(y_true, 1e-10) / mu) - (y_true - mu))
        deviance = float(deviance_val)
        
        # Pearson chi2
        pearson_resid = (y_true - mu) / np.sqrt(mu)
        pearson_chi2_val = np.sum(pearson_resid ** 2)
        pearson_chi2 = float(pearson_chi2_val)
        
        # Dispersion (phi)
        n = len(y_true)
        p = 1  # 간단히 1로 가정 (실제로는 모델의 파라미터 수)
        dispersion_val = pearson_chi2_val / (n - p) if (n - p) > 0 else 1.0
        dispersion = float(dispersion_val)
        
        if p_model_type == 'Poisson':
            # Log-likelihood (Poisson)
            log_likelihood_val = np.sum(stats.poisson.logpmf(y_true, mu))
            log_likelihood = float(log_likelihood_val)
            
            # AIC, BIC (근사치)
            aic = float(-2 * log_likelihood_val + 2 * p)
            bic = float(-2 * log_likelihood_val + np.log(n) * p)
            
            print(f"Deviance: {deviance:.6f}")
            print(f"Pearson chi²: {pearson_chi2:.6f}")
            print(f"Dispersion (φ): {dispersion:.6f}")
            print(f"Log-Likelihood: {log_likelihood:.6f}")
            print(f"AIC: {aic:.6f}")
            print(f"BIC: {bic:.6f}")
        else:
            print(f"Deviance: {deviance:.6f}")
            print(f"Pearson chi²: {pearson_chi2:.6f}")
            print(f"Dispersion (φ): {dispersion:.6f}")
    
    elif p_model_type in ['Logistic', 'Logit']:
        # Logistic regression 통계량
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        y_true_clipped = np.clip(y_true, 1e-10, 1 - 1e-10)
        deviance_val = -2 * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        deviance = float(deviance_val)
        
        # Pearson chi2
        pearson_resid = (y_true - y_pred) / np.sqrt(y_pred * (1 - y_pred) + 1e-10)
        pearson_chi2_val = np.sum(pearson_resid ** 2)
        pearson_chi2 = float(pearson_chi2_val)
        
        # Log-likelihood
        log_likelihood_val = np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        log_likelihood = float(log_likelihood_val)
        
        n = len(y_true)
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)
        
        print(f"Deviance: {deviance:.6f}")
        print(f"Pearson chi²: {pearson_chi2:.6f}")
        print(f"Log-Likelihood: {log_likelihood:.6f}")
        print(f"AIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")
    
    elif p_model_type == 'OLS':
        # OLS 통계량
        deviance_val = np.sum((y_true - y_pred) ** 2)
        deviance = float(deviance_val)
        
        # Log-likelihood (normal distribution)
        n = len(y_true)
        sigma2 = deviance_val / n if n > 0 else 1.0
        log_likelihood_val = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        log_likelihood = float(log_likelihood_val)
        
        p = 1
        aic = float(-2 * log_likelihood_val + 2 * p)
        bic = float(-2 * log_likelihood_val + np.log(n) * p)
        
        print(f"Deviance (Residual Sum of Squares): {deviance:.6f}")
        print(f"Log-Likelihood: {log_likelihood:.6f}")
        print(f"AIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")

print("\\n" + "=" * 60)
print("평가 완료")
print("=" * 60)
`;
    } else {
        code += `
print("\\n모델 타입이 지정되지 않아 기본 통계량만 계산되었습니다.")
print("모델 타입을 지정하면 추가 통계량(Deviance, AIC, BIC 등)을 계산할 수 있습니다.")

print("\\n" + "=" * 60)
print("평가 완료")
print("=" * 60)
`;
    }
    
    return code;
}