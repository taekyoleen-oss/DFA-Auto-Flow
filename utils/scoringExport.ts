import { CanvasModule, Connection, ModuleType } from '../types';

/**
 * 스코어링/배포 코드 내보내기 (additive)
 * --------------------------------------------------------------------------
 * 이 모듈은 기존 파이프라인 코드 생성(`utils/generatePipelineCode.ts`,
 * `codeSnippets.ts` 실행 템플릿)과 완전히 독립적인 *별도* 생성기다.
 * 적합된 모델/프라이싱 체인을 가진 파이프라인에 대해, 배포 가능한
 *   - joblib 저장/로드
 *   - 최소 FastAPI + Flask 스코어링 엔드포인트
 *   - 요청/응답 JSON 샘플
 * 을 생성한다. 기존 모듈 실행·연결·시각화·재현성 경로는 건드리지 않는다.
 */

export interface ScoringExportResult {
  /** 내보내기 가능 여부 (적합 모델/프라이싱 체인 존재) */
  available: boolean;
  /** 사용자에게 보여줄 사유(불가 시) */
  reason?: string;
  /** 감지된 내보내기 종류 */
  kind?: 'ml-model' | 'pricing';
  /** 대표 산출물(모델/프라이싱) 모듈 이름 */
  artifactName?: string;
  /** 생성된 전체 스니펫 텍스트 */
  code: string;
}

/** 감지된 적합 모델 정보 */
interface DetectedModel {
  moduleName: string;
  featureColumns: string[];
  labelColumn: string;
  modelPurpose: 'classification' | 'regression';
  modelType: string;
}

/** 감지된 프라이싱 산출물 정보 */
interface DetectedPricing {
  moduleName: string;
  moduleType: string;
  /** 요청 샘플에 쓸 입력 키 후보(파라미터 기반) */
  inputKeys: Array<{ key: string; value: number }>;
}

const ML_MODEL_OUTPUT = 'TrainedModelOutput';
const PRICING_MODULE_TYPES: ModuleType[] = [
  ModuleType.PriceXoLLayer,
  ModuleType.PriceXolContract,
  ModuleType.XolPricing,
  ModuleType.XolCalculator,
  ModuleType.FitLossDistribution,
];

/** outputData가 적합된 ML 모델인지 검사하고 정보를 추출한다. */
function detectMlModel(modules: CanvasModule[]): DetectedModel | null {
  // 우선 TrainModel/지도학습 모듈의 TrainedModelOutput을 찾는다.
  for (const m of modules) {
    const out = m.outputData as any;
    if (out && out.type === ML_MODEL_OUTPUT && Array.isArray(out.featureColumns)) {
      return {
        moduleName: m.name,
        featureColumns: out.featureColumns,
        labelColumn: out.labelColumn ?? 'target',
        modelPurpose: out.modelPurpose === 'classification' ? 'classification' : 'regression',
        modelType: String(out.modelType ?? m.type),
      };
    }
  }
  return null;
}

/** 프라이싱/분포적합 산출물을 찾아 정보를 추출한다. */
function detectPricing(modules: CanvasModule[]): DetectedPricing | null {
  for (const m of modules) {
    if (!PRICING_MODULE_TYPES.includes(m.type)) continue;
    const params = m.parameters || {};
    const inputKeys: Array<{ key: string; value: number }> = [];
    for (const [k, v] of Object.entries(params)) {
      if (typeof v === 'number' && Number.isFinite(v)) {
        inputKeys.push({ key: k, value: v });
      }
    }
    return {
      moduleName: m.name,
      moduleType: String(m.type),
      inputKeys,
    };
  }
  return null;
}

function pyStr(s: string): string {
  return JSON.stringify(String(s));
}

function jsonSample(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

/** ML 모델용 스코어링 스니펫 생성 */
function buildMlSnippet(model: DetectedModel): string {
  const feats = model.featureColumns.length
    ? model.featureColumns
    : ['feature_1', 'feature_2'];
  const featListPy = '[' + feats.map(pyStr).join(', ') + ']';
  // 요청 샘플: 각 피처에 0.0 placeholder
  const sampleFeatures: Record<string, number> = {};
  feats.forEach((f) => { sampleFeatures[f] = 0.0; });
  const requestSample = { instances: [sampleFeatures] };
  const isClf = model.modelPurpose === 'classification';
  const responseSample = isClf
    ? { predictions: [0], probabilities: [[0.7, 0.3]] }
    : { predictions: [0.0] };

  return `# =============================================================================
# 스코어링 / 배포 코드 — ${model.moduleName} (${model.modelType}, ${isClf ? '분류' : '회귀'})
# =============================================================================
# 이 스니펫은 앱의 파이프라인 코드와 별개로, "적합된 모델을 저장하고
# 서비스로 배포"하는 흐름을 보여줍니다.
#
#   1) 학습 스크립트(앱에서 내보낸 전체 파이프라인 코드)로 모델을 적합시킨 뒤
#      아래 [1] 저장 코드로 model.joblib 파일을 생성합니다.
#   2) [2] FastAPI 또는 [3] Flask 스니펫으로 스코어링 엔드포인트를 띄웁니다.
#   3) [4] 요청/응답 JSON 샘플로 호출을 검증합니다.
# =============================================================================

# 학습 파이프라인에서 만들어진 모델 변수명은 'model'로 가정합니다.
FEATURE_COLUMNS = ${featListPy}
LABEL_COLUMN = ${pyStr(model.labelColumn)}


# -----------------------------------------------------------------------------
# [1] 모델 저장 / 로드 (joblib)
# -----------------------------------------------------------------------------
import joblib
import pandas as pd

def save_model(model, path: str = "model.joblib") -> None:
    """적합된 sklearn 모델과 피처 스키마를 함께 저장."""
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, path)

def load_model(path: str = "model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_columns"]


def predict(model, feature_columns, instances: list[dict]):
    """요청 instance 목록 -> 예측 결과."""
    X = pd.DataFrame(instances)[feature_columns]
    preds = model.predict(X)
    result = {"predictions": [float(p) for p in preds]}
${isClf ? `    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["probabilities"] = [[float(v) for v in row] for row in proba]
` : ''}    return result


# -----------------------------------------------------------------------------
# [2] FastAPI 스코어링 엔드포인트
#     실행:  uvicorn scoring_service:app --host 0.0.0.0 --port 8000
#     의존성: pip install fastapi uvicorn joblib pandas scikit-learn
# -----------------------------------------------------------------------------
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI(title="Scoring Service — ${model.moduleName}")
_model, _feature_columns = joblib.load("model.joblib").values()

class ScoreRequest(BaseModel):
    instances: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(req: ScoreRequest):
    X = pd.DataFrame(req.instances)[_feature_columns]
    preds = _model.predict(X)
    out = {"predictions": [float(p) for p in preds]}
${isClf ? `    if hasattr(_model, "predict_proba"):
        out["probabilities"] = [[float(v) for v in r] for r in _model.predict_proba(X)]
` : ''}    return out
"""


# -----------------------------------------------------------------------------
# [3] Flask 스코어링 엔드포인트 (대안)
#     실행:  flask --app scoring_service run --port 8000
#     의존성: pip install flask joblib pandas scikit-learn
# -----------------------------------------------------------------------------
"""
from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
_bundle = joblib.load("model.joblib")
_model, _feature_columns = _bundle["model"], _bundle["feature_columns"]

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/score")
def score():
    payload = request.get_json(force=True)
    X = pd.DataFrame(payload["instances"])[_feature_columns]
    preds = _model.predict(X)
    out = {"predictions": [float(p) for p in preds]}
${isClf ? `    if hasattr(_model, "predict_proba"):
        out["probabilities"] = [[float(v) for v in r] for r in _model.predict_proba(X)]
` : ''}    return jsonify(out)
"""


# -----------------------------------------------------------------------------
# [4] 요청 / 응답 JSON 샘플
# -----------------------------------------------------------------------------
# POST /score  (Content-Type: application/json)
#
# Request:
# ${jsonSample(requestSample).split('\n').join('\n# ')}
#
# Response:
# ${jsonSample(responseSample).split('\n').join('\n# ')}
#
# curl 예시:
#   curl -X POST http://localhost:8000/score \\
#     -H "Content-Type: application/json" \\
#     -d '${JSON.stringify(requestSample)}'
`;
}

/** 프라이싱/분포적합용 스코어링 스니펫 생성 */
function buildPricingSnippet(pricing: DetectedPricing): string {
  const inputs = pricing.inputKeys.length
    ? pricing.inputKeys
    : [{ key: 'retention', value: 1_000_000 }, { key: 'limit', value: 5_000_000 }];
  const sampleInput: Record<string, number> = {};
  inputs.forEach(({ key, value }) => { sampleInput[key] = value; });
  const requestSample = { params: sampleInput };
  const responseSample = {
    premium: 0.0,
    expected_loss: 0.0,
    rate_on_line_pct: 0.0,
  };
  const paramKeysPy = '[' + inputs.map((i) => pyStr(i.key)).join(', ') + ']';
  const defaultsPy = '{' + inputs.map((i) => `${pyStr(i.key)}: ${i.value}`).join(', ') + '}';

  return `# =============================================================================
# 스코어링 / 배포 코드 — ${pricing.moduleName} (${pricing.moduleType}, 프라이싱)
# =============================================================================
# 적합된 프라이싱/분포 체인을 "재사용 가능한 프라이싱 함수"로 패키징하고
# 서비스로 배포하는 흐름입니다. price_layer()의 본문은 앱에서 내보낸 전체
# 파이프라인 코드의 프라이싱 로직(분포 적합 + 레이어 가격 산출)으로 채우세요.
# =============================================================================

EXPECTED_PARAM_KEYS = ${paramKeysPy}
DEFAULT_PARAMS = ${defaultsPy}


# -----------------------------------------------------------------------------
# [1] 프라이싱 모델 저장 / 로드 (joblib)
#     적합된 분포 파라미터(예: fitted_params) 등 상태를 함께 저장합니다.
# -----------------------------------------------------------------------------
import joblib

def save_pricing_model(state: dict, path: str = "pricing_model.joblib") -> None:
    """프라이싱에 필요한 적합 상태(분포 파라미터 등)를 저장."""
    joblib.dump(state, path)

def load_pricing_model(path: str = "pricing_model.joblib") -> dict:
    return joblib.load(path)


def price_layer(params: dict, state: dict | None = None) -> dict:
    """레이어/계약 파라미터 -> 프라이싱 결과.
    NOTE: 아래 본문을 앱의 전체 파이프라인 코드 프라이싱 로직으로 교체하세요.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    # TODO: 적합된 분포 + 레이어 정의로 expected_loss / premium 계산
    expected_loss = 0.0
    rate_on_line_pct = 0.0
    premium = 0.0
    return {
        "premium": float(premium),
        "expected_loss": float(expected_loss),
        "rate_on_line_pct": float(rate_on_line_pct),
    }


# -----------------------------------------------------------------------------
# [2] FastAPI 프라이싱 엔드포인트
#     실행:  uvicorn pricing_service:app --host 0.0.0.0 --port 8000
#     의존성: pip install fastapi uvicorn joblib numpy scipy
# -----------------------------------------------------------------------------
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Pricing Service — ${pricing.moduleName}")
_state = joblib.load("pricing_model.joblib")

class PriceRequest(BaseModel):
    params: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/price")
def price(req: PriceRequest):
    return price_layer(req.params, _state)
"""


# -----------------------------------------------------------------------------
# [3] Flask 프라이싱 엔드포인트 (대안)
#     실행:  flask --app pricing_service run --port 8000
#     의존성: pip install flask joblib numpy scipy
# -----------------------------------------------------------------------------
"""
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
_state = joblib.load("pricing_model.joblib")

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/price")
def price():
    payload = request.get_json(force=True)
    return jsonify(price_layer(payload.get("params", {}), _state))
"""


# -----------------------------------------------------------------------------
# [4] 요청 / 응답 JSON 샘플
# -----------------------------------------------------------------------------
# POST /price  (Content-Type: application/json)
#
# Request:
# ${jsonSample(requestSample).split('\n').join('\n# ')}
#
# Response:
# ${jsonSample(responseSample).split('\n').join('\n# ')}
#
# curl 예시:
#   curl -X POST http://localhost:8000/price \\
#     -H "Content-Type: application/json" \\
#     -d '${JSON.stringify(requestSample)}'
`;
}

/**
 * 파이프라인에서 스코어링 배포 코드를 생성한다.
 * ML 모델 체인을 우선 감지하고, 없으면 프라이싱/분포 체인을 감지한다.
 */
export function buildScoringExport(
  modules: CanvasModule[],
  _connections: Connection[],
): ScoringExportResult {
  if (!modules || modules.length === 0) {
    return {
      available: false,
      reason: '파이프라인이 비어 있습니다. 모듈을 추가하고 실행해 주세요.',
      code: '',
    };
  }

  const ml = detectMlModel(modules);
  if (ml) {
    return {
      available: true,
      kind: 'ml-model',
      artifactName: ml.moduleName,
      code: buildMlSnippet(ml),
    };
  }

  const pricing = detectPricing(modules);
  if (pricing) {
    return {
      available: true,
      kind: 'pricing',
      artifactName: pricing.moduleName,
      code: buildPricingSnippet(pricing),
    };
  }

  return {
    available: false,
    reason:
      '적합된 모델 또는 프라이싱 체인이 없습니다. TrainModel(지도학습) 또는 ' +
      'PriceXoLLayer/XolPricing/FitLossDistribution 등의 모듈을 실행한 뒤 다시 시도하세요.',
    code: '',
  };
}
