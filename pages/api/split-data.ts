/**
 * Python의 sklearn.train_test_split과 동일한 결과를 반환하는 API
 * 
 * 이 API는 Python 백엔드를 호출하여 정확한 결과를 반환합니다.
 */

import type { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';
import path from 'path';

export default async function handler(
    req: NextApiRequest,
    res: NextApiResponse
) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

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

    } catch (error: any) {
        console.error('API error:', error);
        res.status(500).json({ error: 'Internal server error', details: error.message });
    }
}

