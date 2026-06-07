---
name: pipeline-export-reproducibility
description: DFA-Auto-Flow "전체 파이프라인 코드"의 외부 파이썬 standalone 재현성과 클린 출력 표준. 전체코드 내보내기, 데이터 임베드, 실행 흐름만 노출, 표시코드=실행코드 동치 보장 시 사용. "전체코드", "재현성", "내보내기", "standalone" 작업에 트리거.
---

# Pipeline Export Reproducibility (전체코드)

## 두 가지 동시 목표
1. **실행되는 내용만** — 모듈 코드의 풍부한 옵션/대안 주석을 걷어내고 실제 실행 경로만 남긴다.
2. **즉시 동일 결과** — 외부 Jupyter/스크립트에 복붙하면 파일 없이 단독 실행돼 앱과 같은 결과.

## 데이터 임베드 패턴 (standalone)
- LoadData 데이터를 코드에 압축 임베드하되 **실행부와 시각 분리**: 본문 상단 = 깔끔한 실행 흐름, 데이터 = 하단 부록 헬퍼.
- 형태:
  ```python
  # === 실행 파이프라인 ===
  dataframe = _load_embedded_data()
  ... 실제 처리 단계 ...
  # === 부록: 임베드 데이터 (자동 생성) ===
  def _load_embedded_data():
      import base64, json
      return pd.DataFrame(json.loads(base64.b64decode('...').decode()))
  ```
- 파이썬은 `def`를 호출 시점에 평가하므로 헬퍼를 하단에 둬도 상단에서 호출 가능(단, 호출은 함수 정의가 모듈 로드 후 실행). 안전하게: 임베드 헬퍼 정의를 상단 import 직후 1블록으로, 본문 실행은 그 아래. **시각적 분리 ≠ 실행 순서 위반**에 주의.
- 5000행 초과 등 잘림이 있으면 코드에 명시하고, 외부 전체 실행 안내를 남긴다.

## 클린화 규칙
- 전체코드에서 제거: 모듈 단위 코드의 "옵션 A/B/C" 대안 주석, 미사용 분기, 장황한 설명.
- 유지: 모듈 섹션 헤더(추적용), 실제 실행 라인, 결과 확인 print.
- 표시용(`forExecution=false`)과 실행용(`true`)의 **로직 동치**를 유지 — 차이는 데이터 주입 방식뿐.

## 검증 (parity)
1. 전체코드를 `_workspace/pipeline.py`로 저장.
2. 로컬 `python pipeline.py` 실행(데이터 파일 없이) → 성공해야 함.
3. 출력을 앱 Pyodide 출력과 비교 → shape·수치 동일(부동소수 허용오차 내) 확인.
4. 불일치 시 pyodide-runtime-engineer와 원인 조율.
