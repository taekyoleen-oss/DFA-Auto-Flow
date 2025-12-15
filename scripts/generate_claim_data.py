import pandas as pd
import numpy as np
from datetime import datetime
import random

# 시드 설정
random.seed(42)
np.random.seed(42)

# 종목구분 리스트
categories = ["자동차보험", "화재보험", "상해보험", "배상책임보험", "건강보험"]

# 사고내용 리스트
accident_types = [
    "교통사고", "화재사고", "낙상사고", "물건손상", "도난사고",
    "상해사고", "질병", "재산손해", "배상책임", "기타사고"
]

rows = []

# 2020-2024년, 연도별 20개씩 생성
for year in range(2020, 2025):
    for i in range(20):
        # 날짜 생성 (연도 내 랜덤)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = datetime(year, month, day)
        
        # 종목구분 랜덤 선택
        category = random.choice(categories)
        
        # 클레임 금액 생성 (로그정규분포 기반)
        # 평균 500만원, 표준편차 300만원 정도
        base_amount = np.random.lognormal(mean=13.5, sigma=0.8)
        claim_amount = max(100000, int(base_amount))  # 최소 10만원
        
        # 사고내용 랜덤 선택
        accident_content = random.choice(accident_types)
        
        rows.append({
            "종목구분": category,
            "날짜": date.strftime("%Y-%m-%d"),
            "클레임 금액": claim_amount,
            "기타": f"{accident_content} - {random.randint(1, 1000)}번 사고"
        })

df = pd.DataFrame(rows)

# CSV 파일로 저장
df.to_csv("public/claim_data.csv", index=False, encoding="utf-8-sig")

print(f"CSV 파일 생성 완료: {len(df)}건의 데이터")
print(f"컬럼: {list(df.columns)}")
print(f"\n처음 5개 행:")
print(df.head())







