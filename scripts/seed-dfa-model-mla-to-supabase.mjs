/**
 * samples/DFA Model.mla 파일 하나를 Supabase에 등록하는 스크립트
 *
 * 사용법: node scripts/seed-dfa-model-mla-to-supabase.mjs
 *
 * .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY (또는 NEXT_PUBLIC_*) 가 있어야 합니다.
 * 제목: "DFA 모델", app_section "DFA", category "기타", 설명 등 적절히 입력합니다.
 */

import { createClient } from "@supabase/supabase-js";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

function loadEnv() {
  const envPath = resolve(projectRoot, ".env");
  if (!existsSync(envPath)) {
    console.error(".env 파일을 찾을 수 없습니다.");
    process.exit(1);
  }
  const content = readFileSync(envPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      const eq = trimmed.indexOf("=");
      if (eq > 0) {
        const key = trimmed.slice(0, eq).trim();
        const value = trimmed
          .slice(eq + 1)
          .trim()
          .replace(/^["']|["']$/g, "");
        process.env[key] = value;
      }
    }
  }
}

loadEnv();

const supabaseUrl =
  process.env.VITE_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey =
  process.env.VITE_SUPABASE_ANON_KEY ||
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error(
    "Supabase URL/Key가 없습니다. .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY를 설정하세요."
  );
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

const mlaPath = resolve(projectRoot, "samples", "DFA Model.mla");
if (!existsSync(mlaPath)) {
  console.error("samples/DFA Model.mla 파일을 찾을 수 없습니다.");
  process.exit(1);
}

const raw = readFileSync(mlaPath, "utf-8");
let data;
try {
  data = JSON.parse(raw);
} catch (e) {
  console.error("DFA Model.mla JSON 파싱 실패:", e.message);
  process.exit(1);
}

const rawModules = Array.isArray(data.modules) ? data.modules : [];
const connections = Array.isArray(data.connections) ? data.connections : [];

if (rawModules.length === 0) {
  console.error("modules가 비어 있습니다.");
  process.exit(1);
}

// Supabase 행 크기 제한을 위해 outputData 등 런타임 캐시는 제거하고 저장 (로드 후 Run으로 재실행 가능)
const modules = rawModules.map((m) => {
  const { outputData, ...rest } = m;
  return rest;
});

const MODEL_NAME = "DFA 모델";
const APP_SECTION = "DFA";
const CATEGORY = "기타";
const DESCRIPTION =
  "클레임 데이터를 로드하여 빈도/심각도 분리, 모델링하는 DFA 샘플 모델입니다.";
const DEVELOPER_EMAIL = null;

async function main() {
  console.log("DFA Model.mla → Supabase 등록 (제목: DFA 모델)");

  // 1) 입력 데이터: LoadClaimData 모듈의 fileContent가 있으면 sample_input_data에 등록
  let inputDataId = null;
  const loadModule = modules.find(
    (m) =>
      (m.type === "LoadClaimData" || m.type === "LoadData") &&
      m.parameters?.fileContent
  );
  if (loadModule?.parameters?.fileContent) {
    const content = String(loadModule.parameters.fileContent);
    const inputName = "claim_data";
    const { data: inputRow, error: inputErr } = await supabase
      .from("sample_input_data")
      .insert({ name: inputName, content })
      .select("id")
      .single();
    if (inputErr) {
      console.warn("sample_input_data 삽입 건너뜀:", inputErr.message);
    } else if (inputRow) {
      inputDataId = inputRow.id;
      console.log("  sample_input_data 등록:", inputName);
    }
  }

  // 2) sample_models: 모델명 "DFA 모델", file_content = { modules, connections }
  const { data: modelRow, error: modelErr } = await supabase
    .from("sample_models")
    .insert({
      name: MODEL_NAME,
      file_content: { modules, connections },
    })
    .select("id")
    .single();

  if (modelErr) {
    console.error("sample_models 삽입 실패:", modelErr.message);
    process.exit(1);
  }

  // 3) autoflow_samples: app_section DFA, category, description 등
  const { error: sampleErr } = await supabase.from("autoflow_samples").insert({
    app_section: APP_SECTION,
    category: CATEGORY,
    developer_email: DEVELOPER_EMAIL,
    model_id: modelRow.id,
    input_data_id: inputDataId,
    description: DESCRIPTION,
  });

  if (sampleErr) {
    console.error("autoflow_samples 삽입 실패:", sampleErr.message);
    process.exit(1);
  }

  console.log("  sample_models 등록:", MODEL_NAME);
  console.log("  autoflow_samples 등록 (app_section: DFA, category: 기타)");
  console.log("\n완료. DFA-Auto-Flow에서 Samples 메뉴를 열어 'DFA 모델'을 확인하세요.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
