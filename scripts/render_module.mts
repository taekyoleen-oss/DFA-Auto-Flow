// 단일 모듈 코드를 실제 getModuleCode 경로로 렌더링한다 (템플릿 검증용).
// 사용: npx tsx scripts/render_module.mts '<moduleJson>'
import { getModuleCode } from '../codeSnippets';

const module = JSON.parse(process.argv[2]);
process.stdout.write(getModuleCode(module, [module], []));
