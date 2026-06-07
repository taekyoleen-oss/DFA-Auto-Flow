import { CanvasModule, Connection } from '../types';
import { getModuleCode } from '../codeSnippets';

/**
 * 모듈 id를 파이썬 식별자로 안전하게 변환한다. 전체 id를 사용해 고유성을 보장한다.
 * (id.slice(0,8)은 "module-<timestamp>-N" 형태에서 공통 접두사만 잘라 모든 모듈이
 *  같은 변수명으로 충돌하던 버그가 있었다.)
 */
function sanitizeId(id: string): string {
  const s = id.replace(/[^a-zA-Z0-9_]/g, '_');
  return /^[0-9]/.test(s) ? `m_${s}` : s;
}

function encodeBase64(str: string): string {
  const bytes = new TextEncoder().encode(str);
  // Process in 64 KiB chunks to avoid call-stack overflow on large data
  const chunkSize = 65536;
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += chunkSize) {
    chunks.push(String.fromCharCode(...bytes.subarray(i, i + chunkSize)));
  }
  return btoa(chunks.join(''));
}

/** LoadData 임베드 데이터의 추출 결과 */
interface EmbeddedData {
  funcName: string;
  helperDef: string;   // 부록에 들어갈 def 정의
  note?: string;       // 잘림 등 안내 (실행 흐름에 1줄 주석으로)
}

/** standalone 임베드 대상이 되는 데이터 소스(로드) 모듈 타입. */
const EMBEDDABLE_LOAD_TYPES = ['LoadData', 'LoadClaimData'];

/**
 * 모듈의 outputData에서 임베드 가능한 DataPreview를 꺼낸다.
 * - LoadData: outputData 자체가 DataPreview
 * - LoadClaimData: outputData가 ClaimDataOutput이고, 실제 미리보기는 outputData.data(중첩 DataPreview)
 */
function extractDataPreview(outputData: any): any | null {
  if (!outputData) return null;
  if (outputData.type === 'DataPreview' && outputData.rows?.length) return outputData;
  const nested = outputData.data;
  if (nested && nested.type === 'DataPreview' && nested.rows?.length) return nested;
  return null;
}

/**
 * 로드 모듈의 실행된 데이터를 standalone 임베드 헬퍼로 변환한다.
 * 외부 Python(Jupyter/스크립트)에 그대로 복붙해도 파일 없이 동일 결과가 나오도록,
 * 무거운 base64 데이터는 부록 헬퍼 함수 안에 격리하고, 실행 흐름에서는 호출만 한다.
 * 데이터가 없으면(미실행) null 반환 → 호출부가 read_csv 안내로 폴백.
 *
 * 특히 LoadClaimData는 원래 코드가 random으로 합성 데이터를 매번 새로 만들어
 * 재현 불가능했으나, 임베드로 전환하면 앱에서 실제 로드한 데이터를 결정적으로 재현한다.
 */
function buildEmbeddedData(module: CanvasModule): EmbeddedData | null {
  const preview = extractDataPreview(module.outputData as any);
  if (!preview) return null;

  const MAX_INJECT_ROWS = 5000;
  const allRows = preview.rows as Record<string, any>[];
  const rows = allRows.length > MAX_INJECT_ROWS ? allRows.slice(0, MAX_INJECT_ROWS) : allRows;
  const totalCount: number = preview.totalRowCount || allRows.length;
  const columns: string[] = (preview.columns as any[])?.map((c: any) => c.name)
    ?? Object.keys(rows[0] ?? {});
  const isTruncated = rows.length < allRows.length;
  const isPartial = allRows.length < totalCount;

  const colData: Record<string, any[]> = {};
  columns.forEach((col) => {
    colData[col] = rows.map((row) => row[col] ?? null);
  });

  const b64 = encodeBase64(JSON.stringify(colData));
  const safeId = sanitizeId(module.id);
  const funcName = `_embedded_${safeId}`;

  const note = isTruncated
    ? `${rows.length}행 임베드 (전체 ${totalCount}행 — 5000행 초과로 잘림, 전체는 외부 Python에서 원본 파일 사용)`
    : isPartial
    ? `${rows.length}행 임베드 (전체 ${totalCount}행 미리보기)`
    : `전체 ${rows.length}행 임베드`;

  const helperDef = [
    `def ${funcName}():`,
    `    """[${module.name}] 앱에서 로드된 데이터 — ${note}"""`,
    `    import base64, json`,
    `    _b64 = "${b64}"`,
    `    return pd.DataFrame(json.loads(base64.b64decode(_b64).decode("utf-8")))`,
  ].join('\n');

  return { funcName, helperDef, note };
}

/**
 * (구) Pyodide 인라인 주입 코드. 호환용으로 유지 — 부록 임베드 방식으로 통합되었으므로
 * 신규 코드 경로에서는 사용하지 않는다.
 */
export function generateLoadDataInjectionCode(module: CanvasModule): string {
  const source = String(module.parameters?.source || 'data.csv');
  const embedded = buildEmbeddedData(module);
  if (!embedded) {
    return [
      `# ⚠️  Pyodide 실행: 로컬 파일 접근 불가 (브라우저 환경 제한)`,
      `# 해결 방법: 캔버스에서 LoadData 모듈을 먼저 실행(▶)한 후 다시 시도하세요.`,
      `# 외부 Python(Jupyter, 스크립트)에서는: dataframe = pd.read_csv('${source}')`,
      `raise RuntimeError("LoadData 모듈을 먼저 실행해주세요. 모듈 선택 후 ▶ 버튼을 클릭하세요.")`,
    ].join('\n');
  }
  return [embedded.helperDef, `dataframe = ${embedded.funcName}()`].join('\n');
}

/**
 * 모듈 간 의존성을 기반으로 실행 순서를 결정합니다 (위상 정렬)
 */
function getExecutionOrder(
  modules: CanvasModule[],
  connections: Connection[]
): CanvasModule[] {
  const moduleMap = new Map<string, CanvasModule>();
  modules.forEach((m) => moduleMap.set(m.id, m));

  const dependencies = new Map<string, Set<string>>();
  modules.forEach((m) => dependencies.set(m.id, new Set()));

  connections.forEach((conn) => {
    const fromModule = moduleMap.get(conn.from.moduleId);
    const toModule = moduleMap.get(conn.to.moduleId);
    if (fromModule && toModule) {
      const deps = dependencies.get(toModule.id);
      if (deps) {
        deps.add(fromModule.id);
      }
    }
  });

  const ordered: CanvasModule[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function visit(moduleId: string) {
    if (visiting.has(moduleId)) return; // 순환 참조 방지
    if (visited.has(moduleId)) return;

    visiting.add(moduleId);
    const deps = dependencies.get(moduleId);
    if (deps) {
      deps.forEach((depId) => visit(depId));
    }
    visiting.delete(moduleId);
    visited.add(moduleId);

    const module = moduleMap.get(moduleId);
    if (module) {
      ordered.push(module);
    }
  }

  modules.forEach((m) => visit(m.id));
  return ordered;
}

/**
 * 모듈 타입별 내부 출력 변수명 매핑
 */
const MODULE_OUTPUT_VAR: Record<string, string> = {
  LoadData: 'dataframe',
  SelectData: 'selected_data',
  DataFiltering: 'dataframe',
  HandleMissingValues: 'cleaned_data',
  EncodeCategorical: 'encoded_data',
  ScalingTransform: 'normalized_data',
  NormalizeData: 'normalized_data',
  TransformData: 'transformed_data',
  TransitionData: 'transformed_data',
  ResampleData: 'resampled_data',
  TrainModel: 'trained_model',
  ScoreModel: 'scored_data',
  EvaluateModel: 'evaluation_metrics',
  ResultModel: 'model_results',
  PredictModel: 'predicted_data',
  Join: 'dataframe',
  Concat: 'dataframe',
  // ── DFA/XoL 도메인 모듈 (단일 출력 변환) ──
  LoadClaimData: 'dataframe',     // 임베드 데이터를 dataframe으로 로드
  ApplyInflation: 'dataframe',    // dataframe에 인플레이션 컬럼 추가
  FormatChange: 'formatted_dataframe',
  FitAggregateModel: 'params',    // 분포 적합 파라미터
  SimulateAggDist: 'simulated_amounts', // 집계분포 시뮬레이션 결과(CombineLoss.agg_dist_in)
  FitFrequencyModel: 'frequency_params', // 최적 빈도 분포 파라미터(SimulateFreqServ 입력 규약)
  FitSeverityModel: 'severity_params', // 최적 심도 분포 파라미터(SimulateFreqServ 입력 규약)
  CombineLossModel: 'result',     // 통합 손실 통계/VaR/TVaR
  // 주의: Simulate*(7,11,12)는 변수가 아닌 모듈 상태 파라미터(parameters/frequency_params)에
  //       의존하므로 변수 배선만으로는 재현 불가. FitSeverityModel/CombineLossModel은 템플릿 부재.
};

/**
 * 다중 출력 모듈: 출력 포트명 → 템플릿 내부 변수명.
 * 출력 포트마다 서로 다른 변수를 후속 모듈로 전달해야 하는 DFA 모듈에 사용한다.
 */
const MODULE_PORT_OUTPUT_VAR: Record<string, Record<string, string>> = {
  SplitByThreshold: {
    below_threshold_out: 'below_grouped',
    above_threshold_out: 'above_result',
  },
  SplitByFreqServ: {
    frequency_out: 'yearly_frequency_df',
    severity_out: 'severity_df',
  },
  SimulateFreqServ: {
    output_1: 'aggregate_losses',  // CombineLoss.freq_serv_in 입력
  },
};

/** 입력 포트명 → 템플릿이 읽는 표준 변수명. (data_in 계열은 모두 dataframe) */
const INPUT_PORT_VAR: Record<string, string> = {
  data_in: 'dataframe',
  agg_dist_in: 'agg_dist',     // CombineLossModel ← SimulateAggDist
  freq_serv_in: 'freq_serv',   // CombineLossModel ← SimulateFreqServ
  frequency_in: 'frequency_params', // SimulateFreqServ ← FitFrequencyModel
  severity_in: 'severity_params',   // SimulateFreqServ ← FitSeverityModel
};

/**
 * 모듈 타입별 입력 포트 변수 오버라이드.
 * 같은 포트명이라도 모듈 타입에 따라 의미가 다를 때 사용한다.
 * (예: model_in → ML은 trained_model, DFA SimulateAggDist는 적합 파라미터 params)
 */
const MODULE_INPUT_PORT_VAR: Record<string, Record<string, string>> = {
  SimulateAggDist: { model_in: 'params' },
};

/**
 * 전체 파이프라인의 Python 코드를 생성합니다
 */
export function generateFullPipelineCode(
  modules: CanvasModule[],
  connections: Connection[],
  forExecution = false
): string {
  if (modules.length === 0) {
    return '# 파이프라인이 비어있습니다.';
  }

  const executionOrder = getExecutionOrder(modules, connections);
  const moduleMap = new Map<string, CanvasModule>();
  modules.forEach((m) => moduleMap.set(m.id, m));

  const codeLines: string[] = [];
  codeLines.push('# ============================================================================');
  codeLines.push('# 전체 파이프라인 실행 코드 (standalone)');
  codeLines.push('# 이 코드는 Jupyter / Python 스크립트에 그대로 붙여넣으면 별도 데이터 파일 없이');
  codeLines.push('# 앱과 동일한 결과를 재현합니다. (데이터는 하단 [부록]에 자동 임베드)');
  codeLines.push('# ============================================================================');
  codeLines.push('');
  codeLines.push('import pandas as pd');
  codeLines.push('import numpy as np');
  codeLines.push('');
  // 부록(임베드 데이터 헬퍼) 삽입 위치 — 실행 흐름과 시각적으로 분리하되,
  // 파이썬은 def를 호출 전에 정의해야 하므로 import 직후·실행부 앞에 둔다.
  const appendixInsertIndex = codeLines.length;

  const embeddedHelpers: string[] = []; // LoadData standalone 임베드 헬퍼 모음
  const variableMap = new Map<string, string>(); // moduleId -> outputVarName (단일 출력 폴백)
  const portVariableMap = new Map<string, string>(); // `${moduleId}:${portName}` -> 출력 포트 변수

  executionOrder.forEach((module, index) => {
    // ── 모듈 섹션 헤더 ──────────────────────────────────────────────────────
    codeLines.push('# ============================================================================');
    codeLines.push(`# [모듈 ${index + 1}/${executionOrder.length}] ${module.name}`);
    codeLines.push(`# 타입: ${module.type}`);
    codeLines.push('# ============================================================================');
    codeLines.push('');

    // ── 모듈 코드 생성 ──────────────────────────────────────────────────────
    // LoadData는 실행된 데이터가 있으면 항상 standalone 임베드(표시/실행 동일).
    // 무거운 데이터는 하단 [부록] 헬퍼로 격리하고 여기서는 호출만 한다.
    let moduleCode: string;
    if (EMBEDDABLE_LOAD_TYPES.includes(module.type)) {
      const embedded = buildEmbeddedData(module);
      if (embedded) {
        embeddedHelpers.push(embedded.helperDef);
        const noteLine = embedded.note ? `  # ${embedded.note}` : '';
        moduleCode = `dataframe = ${embedded.funcName}()${noteLine}`;
      } else {
        const source = String(module.parameters?.source || 'data.csv');
        moduleCode = [
          `# ⚠️  LoadData 미실행 — 데이터가 임베드되지 않았습니다.`,
          `# 앱에서: LoadData 모듈을 먼저 실행(▶)하면 데이터가 자동 임베드되어 어디서나 재현됩니다.`,
          `dataframe = pd.read_csv('${source}')  # 외부 실행 시 원본 파일 경로로 수정`,
        ].join('\n');
      }
    } else {
      try {
        moduleCode = getModuleCode(module, modules, connections);
      } catch (e: any) {
        moduleCode = `# ${module.name} 코드 생성 실패: ${e?.message || String(e)}`;
      }
    }

    // ── 출력 변수명 결정 ────────────────────────────────────────────────────
    const safeId = sanitizeId(module.id);
    let outputVarName = '';
    if (module.outputs.length > 0) {
      const outputType = module.outputs[0].type;
      if (outputType === 'data') {
        outputVarName = `data_${safeId}`;
      } else if (outputType === 'model') {
        outputVarName = `model_${safeId}`;
      } else if (outputType === 'handler') {
        outputVarName = `handler_${safeId}`;
      } else if (outputType === 'evaluation') {
        outputVarName = `eval_${safeId}`;
      } else {
        outputVarName = `result_${safeId}`;
      }
    }

    // ── 입력 연결 처리: 이전 모듈 출력을 표준 변수명으로 받음 ───────────────
    const inputConnections = connections.filter((c) => c.to.moduleId === module.id);
    const inputPrefixLines: string[] = [];

    inputConnections.forEach((conn) => {
      const fromModule = moduleMap.get(conn.from.moduleId);
      // 출력 포트별 변수를 우선 사용(다중 출력 대응), 없으면 모듈 단위 폴백
      const prevVarName =
        portVariableMap.get(`${conn.from.moduleId}:${conn.from.portName}`) ||
        variableMap.get(conn.from.moduleId);
      if (!prevVarName || !fromModule) return;

      const toPort = conn.to.portName;
      const fromLabel = fromModule.name;

      // 모듈 타입별 입력 포트 오버라이드 우선(model_in 충돌 등)
      const moduleOverride = MODULE_INPUT_PORT_VAR[module.type]?.[toPort];
      if (moduleOverride) {
        inputPrefixLines.push(`${moduleOverride} = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (INPUT_PORT_VAR[toPort]) {
        inputPrefixLines.push(`${INPUT_PORT_VAR[toPort]} = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'data_in') {
        inputPrefixLines.push(`dataframe = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'model_in') {
        inputPrefixLines.push(`trained_model = ${prevVarName}  # ← [${fromLabel}] 출력`);
        inputPrefixLines.push(`second_data = dataframe  # ScoreModel용 데이터 (필요시 수정)`);
      } else if (toPort === 'handler_in') {
        inputPrefixLines.push(`handler = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'dist_in') {
        inputPrefixLines.push(`dist = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'curve_in') {
        inputPrefixLines.push(`curve = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'contract_in') {
        inputPrefixLines.push(`contract = ${prevVarName}  # ← [${fromLabel}] 출력`);
      } else if (toPort === 'eval_in') {
        inputPrefixLines.push(`scored_data = ${prevVarName}  # ← [${fromLabel}] 출력`);
      }
    });

    // 입력 연결이 있으면 모듈 코드 앞에 주입
    if (inputPrefixLines.length > 0) {
      codeLines.push('# 이전 모듈 출력을 입력으로 받음');
      inputPrefixLines.forEach((line) => codeLines.push(line));
      codeLines.push('');
    }

    codeLines.push(moduleCode);

    // ── 출력 변수 할당 + print + 다음 모듈용 등록 ──────────────────────────────
    const portMap = MODULE_PORT_OUTPUT_VAR[module.type];
    if (portMap) {
      // 다중 출력 DFA 모듈: 각 출력 포트를 고유 변수에 할당하고 포트별로 등록
      let firstVar = '';
      module.outputs.forEach((outPort) => {
        const internalVar = portMap[outPort.name];
        if (!internalVar) return;
        const portVar = `${internalVar}_${safeId}`;
        codeLines.push(`${portVar} = ${internalVar}  # 출력 포트: ${outPort.name}`);
        portVariableMap.set(`${module.id}:${outPort.name}`, portVar);
        if (!firstVar) firstVar = portVar;
      });
      module.outputs.forEach((outPort) => {
        const internalVar = portMap[outPort.name];
        if (!internalVar) return;
        const portVar = `${internalVar}_${safeId}`;
        codeLines.push(
          `print(f"[${module.name}/${outPort.name}] {getattr(${portVar}, 'shape', len(${portVar}) if hasattr(${portVar}, '__len__') else ${portVar})}")`
        );
      });
      codeLines.push('print()');
      if (firstVar) variableMap.set(module.id, firstVar);
    } else if (outputVarName) {
      const internalVar = MODULE_OUTPUT_VAR[module.type] || '';
      // 출력 변수를 실제로 할당했는지 추적 — 미할당 변수를 print하면 NameError가 난다.
      const assigned = module.type === 'SplitData' || !!internalVar;
      if (module.type === 'SplitData') {
        codeLines.push(`${outputVarName}_train = train_data`);
        codeLines.push(`${outputVarName}_test = test_data`);
      } else if (internalVar) {
        codeLines.push(`${outputVarName} = ${internalVar}`);
      }

      const outputType = module.outputs.length > 0 ? module.outputs[0].type : '';
      if (module.type === 'SplitData') {
        codeLines.push(`print(f"[${module.name}] train: {${outputVarName}_train.shape}, test: {${outputVarName}_test.shape}")`);
        codeLines.push(`print()`);
      } else if (assigned && outputType === 'data') {
        codeLines.push(`print(f"[${module.name}] shape: {${outputVarName}.shape}")`);
        codeLines.push(`print(${outputVarName}.head(5).to_string())`);
        codeLines.push(`print()`);
      } else if (assigned && outputType === 'model') {
        codeLines.push(`print(f"[${module.name}] 모델: {${outputVarName}}")`);
        codeLines.push(`print()`);
      } else if (assigned) {
        codeLines.push(`print(f"[${module.name}] 완료")`);
        codeLines.push(`print(${outputVarName})`);
        codeLines.push(`print()`);
      } else {
        // 출력 변수 규약 미정의 모듈: 모듈 코드만 실행하고 변수 참조 print는 생략
        codeLines.push(`print(f"[${module.name}] 완료")`);
        codeLines.push(`print()`);
      }

      // 할당된 경우에만 후속 모듈용으로 등록(미할당 변수를 전파하면 NameError 전염)
      if (assigned) {
        const primaryVar = module.type === 'SplitData' ? `${outputVarName}_train` : outputVarName;
        variableMap.set(module.id, primaryVar);
        if (module.outputs.length > 0) {
          portVariableMap.set(`${module.id}:${module.outputs[0].name}`, primaryVar);
        }
      }
    }

    codeLines.push('');
  });

  codeLines.push('# ============================================================================');
  codeLines.push('# 파이프라인 실행 완료');
  codeLines.push('# ============================================================================');

  // ── [부록] 임베드 데이터 헬퍼 주입 ────────────────────────────────────────
  // import 직후에 한 블록으로 모아 둔다. 실행 흐름(아래)에서는 호출만 하므로
  // 본문은 깔끔하게 유지되고, 데이터 블록은 한 번에 건너뛸 수 있다.
  if (embeddedHelpers.length > 0) {
    const appendix: string[] = [];
    appendix.push('# ──────────────────────────────────────────────────────────────────────────');
    appendix.push('# [부록] 임베드 데이터 (자동 생성) — 읽지 않아도 됩니다. 실행에만 사용됩니다.');
    appendix.push('# 외부 환경에서 원본 파일로 대체하려면 아래 함수를 pd.read_csv(...)로 바꾸세요.');
    appendix.push('# ──────────────────────────────────────────────────────────────────────────');
    embeddedHelpers.forEach((def) => {
      appendix.push(def);
      appendix.push('');
    });
    appendix.push('# ──────────────────────────────────────── [부록 끝] 실행 파이프라인 시작 ────');
    appendix.push('');
    codeLines.splice(appendixInsertIndex, 0, ...appendix);
  }

  return codeLines.join('\n');
}
