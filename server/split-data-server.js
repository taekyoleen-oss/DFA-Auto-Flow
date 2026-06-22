/**
 * Node.js Express 서버 - SplitData API
 * Pyodide가 타임아웃되거나 실패할 때 사용하는 백엔드 서버
 */

import express from 'express';
import { spawn } from 'child_process';
import cors from 'cors';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

/**
 * CORS 프록시 — 공개 URL의 CSV를 서버가 대신 가져와 그대로 전달한다.
 * 브라우저(Pyodide/앱)는 CORS 때문에 외부 URL을 직접 fetch하지 못하므로,
 * 인앱 URL 로더가 이 엔드포인트를 경유해 CSV 텍스트를 받는다.
 *
 * 주의: 이 프록시는 "데이터 입력 계층"에서만 쓰이며, Pyodide 실행 경로는 변경하지 않는다.
 * 가져온 CSV는 업로드 파일과 동일하게 module.parameters.fileContent로 저장된다.
 */
app.get('/api/proxy-csv', async (req, res) => {
    const targetUrl = req.query.url;

    if (!targetUrl || typeof targetUrl !== 'string') {
        return res.status(400).json({ error: 'Missing required query parameter: url' });
    }

    // 기본 SSRF 방어: http/https만 허용
    let parsed;
    try {
        parsed = new URL(targetUrl);
    } catch {
        return res.status(400).json({ error: 'Invalid URL' });
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return res.status(400).json({ error: 'Only http/https URLs are allowed' });
    }

    try {
        // Node 18+/22 글로벌 fetch 사용
        const upstream = await fetch(targetUrl, {
            redirect: 'follow',
            headers: { 'Accept': 'text/csv, text/plain, */*' },
        });

        if (!upstream.ok) {
            return res.status(502).json({
                error: `Upstream responded with ${upstream.status} ${upstream.statusText}`,
            });
        }

        const text = await upstream.text();
        res.set('Content-Type', 'text/plain; charset=utf-8');
        res.status(200).send(text);
    } catch (error) {
        console.error('proxy-csv error:', error);
        res.status(502).json({ error: `Failed to fetch URL: ${error.message}` });
    }
});

app.post('/api/split-data', async (req, res) => {
    try {
        const { data, train_size, random_state, shuffle, stratify, stratify_column } = req.body;

        // Python 스크립트 실행
        const pythonScript = `
import sys
import json
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    # sklearn의 train_test_split을 사용하여 데이터를 분할합니다.
    input_data = json.loads(sys.stdin.read())
    dataframe = pd.DataFrame(input_data['data'])
    
    # DataFrame 인덱스를 명시적으로 0부터 시작하도록 리셋
    dataframe.index = range(len(dataframe))
    
    # Parameters from UI
    p_train_size = float(input_data['train_size'])
    p_random_state = int(input_data['random_state'])
    p_shuffle = bool(input_data['shuffle'])
    p_stratify = bool(input_data.get('stratify', False))
    p_stratify_column = input_data.get('stratify_column', None)
    
    # Stratify 배열 준비
    stratify_array = None
    if p_stratify and p_stratify_column and p_stratify_column != 'None' and p_stratify_column is not None:
        if p_stratify_column not in dataframe.columns:
            raise ValueError(f"Stratify column '{p_stratify_column}' not found in DataFrame")
        stratify_array = dataframe[p_stratify_column]
    
    # 데이터 분할
    train_data, test_data = train_test_split(
        dataframe,
        train_size=p_train_size,
        random_state=p_random_state,
        shuffle=p_shuffle,
        stratify=stratify_array
    )
    
    result = {
        'train_indices': train_data.index.tolist(),
        'test_indices': test_data.index.tolist(),
        'train_count': len(train_data),
        'test_count': len(test_data)
    }
    
    print(json.dumps(result))
except Exception as e:
    error_info = {
        'error': True,
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_traceback': ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    }
    print(json.dumps(error_info), file=sys.stderr)
    sys.exit(1)
`;

        const pythonProcess = spawn('python', ['-c', pythonScript], {
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const inputData = JSON.stringify({
            data,
            train_size: parseFloat(train_size),
            random_state: parseInt(random_state),
            shuffle: shuffle === true || shuffle === 'True',
            stratify: stratify === true || stratify === 'True',
            stratify_column: stratify_column || null
        });

        let output = '';
        let error = '';

        pythonProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });

        // Promise로 래핑하여 비동기 처리
        await new Promise<void>((resolve, reject) => {
            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    console.error('Python error:', error);
                    try {
                        // stderr에서 에러 정보 파싱 시도
                        const errorInfo = JSON.parse(error);
                        if (errorInfo.error) {
                            return reject(new Error(`Python execution failed: ${errorInfo.error_message || error}`));
                        }
                    } catch {
                        // JSON 파싱 실패 시 원본 에러 사용
                    }
                    return reject(new Error(`Python execution failed with code ${code}: ${error || output}`));
                }

                try {
                    const result = JSON.parse(output);
                    res.status(200).json(result);
                    resolve();
                } catch (e) {
                    console.error('JSON parse error:', e);
                    reject(new Error(`Failed to parse Python output: ${output}`));
                }
            });

            pythonProcess.on('error', (err) => {
                console.error('Python process error:', err);
                reject(new Error(`Failed to start Python process: ${err.message}`));
            });

            pythonProcess.stdin.write(inputData);
            pythonProcess.stdin.end();
        });

    } catch (error) {
        console.error('API error:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`SplitData 서버가 포트 ${PORT}에서 실행 중입니다.`);
    console.log(`http://localhost:${PORT}/api/split-data`);
    console.log(`http://localhost:${PORT}/api/proxy-csv?url=...`);
});
