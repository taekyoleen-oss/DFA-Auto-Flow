// 실제 TS 생성기를 Node에서 구동해 전체 파이프라인 코드를 생성한다 (parity 검증용).
// 사용: npx tsx scripts/gen_check.mts "<model.mla>" "<out.py>"
import { readFileSync, writeFileSync } from 'fs';
import { generateFullPipelineCode } from '../utils/generatePipelineCode';

const [, , modelPath, outPath] = process.argv;
const model = JSON.parse(readFileSync(modelPath, 'utf-8'));
const code = generateFullPipelineCode(model.modules, model.connections, false);
writeFileSync(outPath, code, 'utf-8');

const loadTypes = model.modules.map((m: any) => m.type);
console.log('modules:', model.modules.length, '| types:', loadTypes.join(', '));
console.log('code lines:', code.split('\n').length, '| chars:', code.length);
console.log('embedded helpers:', (code.match(/_embedded_/g) || []).length);
console.log('read_csv occurrences:', (code.match(/read_csv/g) || []).length);
console.log('random/np.random (비결정):', (code.match(/random\.|np\.random/g) || []).length);
