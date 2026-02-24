/**
 * XoL Loss Model.mla, XoL Experience.mla 두 파일을 Supabase에 등록 (카테고리: XOL)
 *
 * 사용법: node scripts/seed-xol-mla-to-supabase.mjs
 *
 * .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY (또는 NEXT_PUBLIC_*) 가 있어야 합니다.
 * 파일 경로는 아래 ENTRIES 또는 환경에 맞게 수정하세요.
 */

import { createClient } from "@supabase/supabase-js";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

// 다운로드 폴더 경로 (Windows 기준, 필요 시 수정)
const downloadsDir =
  process.env.USERPROFILE || process.env.HOME
    ? resolve(process.env.USERPROFILE || process.env.HOME, "Downloads")
    : resolve(projectRoot, "samples");

const ENTRIES = [
  {
    path: resolve(downloadsDir, "XoL Loss Model.mla"),
    modelName: "XoL Loss Model",
    description:
      "XoL(Excess of Loss) 손해 모델 파이프라인. 클레임 데이터 로드 후 포맷 변환, 인플레이션, 빈도/심각도 분리 및 XoL 프라이싱까지 포함합니다.",
    inputDataName: "xol_loss_claim_data",
  },
  {
    path: resolve(downloadsDir, "XoL Experience.mla"),
    modelName: "XoL Experience",
    description:
      "XoL 경험 분석 파이프라인. 클레임 데이터 기반 포맷 변환, 인플레이션 적용, 빈도/심각도 분리 및 경험 분석을 수행합니다.",
    inputDataName: "xol_experience_claim_data",
  },
];

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

const APP_SECTION = "DFA";
const CATEGORY = "XOL";

async function registerOne(entry, index) {
  const { path: mlaPath, modelName, description, inputDataName } = entry;

  if (!existsSync(mlaPath)) {
    console.warn(`[${index + 1}] 파일 없음, 건너뜀: ${mlaPath}`);
    return;
  }

  let data;
  try {
    data = JSON.parse(readFileSync(mlaPath, "utf-8"));
  } catch (e) {
    console.error(`[${index + 1}] "${modelName}" JSON 파싱 실패:`, e.message);
    return;
  }

  const rawModules = Array.isArray(data.modules) ? data.modules : [];
  const connections = Array.isArray(data.connections) ? data.connections : [];

  if (rawModules.length === 0) {
    console.warn(`[${index + 1}] "${modelName}" modules 없음, 건너뜀`);
    return;
  }

  const modules = rawModules.map((m) => {
    const { outputData, ...rest } = m;
    return rest;
  });

  let inputDataId = null;
  const loadModule = modules.find(
    (m) =>
      (m.type === "LoadClaimData" || m.type === "LoadData") &&
      m.parameters?.fileContent
  );
  if (loadModule?.parameters?.fileContent) {
    const content = String(loadModule.parameters.fileContent);
    const { data: inputRow, error: inputErr } = await supabase
      .from("sample_input_data")
      .insert({ name: inputDataName, content })
      .select("id")
      .single();
    if (!inputErr && inputRow) {
      inputDataId = inputRow.id;
      console.log(`  sample_input_data: ${inputDataName}`);
    }
  }

  const { data: modelRow, error: modelErr } = await supabase
    .from("sample_models")
    .insert({
      name: modelName,
      file_content: { modules, connections },
    })
    .select("id")
    .single();

  if (modelErr) {
    console.error(`[${index + 1}] "${modelName}" sample_models 실패:`, modelErr.message);
    return;
  }

  const { error: sampleErr } = await supabase.from("autoflow_samples").insert({
    app_section: APP_SECTION,
    category: CATEGORY,
    developer_email: null,
    model_id: modelRow.id,
    input_data_id: inputDataId,
    description: description,
  });

  if (sampleErr) {
    console.error(`[${index + 1}] "${modelName}" autoflow_samples 실패:`, sampleErr.message);
    return;
  }

  console.log(`  [${index + 1}/${ENTRIES.length}] ${modelName} (category: ${CATEGORY})`);
}

async function main() {
  console.log(`XoL .mla 2건 → Supabase 등록 (category: ${CATEGORY})\n`);

  for (let i = 0; i < ENTRIES.length; i++) {
    await registerOne(ENTRIES[i], i);
  }

  console.log("\n완료. DFA-Auto-Flow Samples에서 카테고리 'XOL'로 확인하세요.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
