# tkLeen 브랜드 가이드 (정본) — 다른 프로젝트용 이식 문서

> **목적**: 이 파일 하나만 프로젝트에 넣으면(또는 이 파일을 참조하면) tkLeen 브랜드를 **정확히** 적용할 수 있다.
> 로고 SVG 소스를 본문에 그대로 포함하므로 별도 이미지 파일이 없어도 렌더링 가능하다.
> 원본 자산: `tkleen-brand-assets/`(svg 11종·png 60종·preview). 본 문서는 `README.md`(Brand Assets v1.0) 기준.

---

## 1. 브랜드 이름
- **정식 표기**: `tkLeen` (camelCase). 가운데 **L만 대문자**, 나머지는 소문자.
- **금지 표기**: `TKLEEN`, `tkleen`, `TkLeen`, `Tkleen` 등 변형 금지.
- 발음: 티-케이-린.

## 2. 컬러 시스템 (고정)
| 이름 | HEX | 용도 |
|---|---|---|
| Ink | `#0A0A0A` | T · 본문 텍스트 · 1차 다크 |
| **Sky Blue** | `#4A90C2` | **K · 워드마크 'L' 액센트 · 브랜드 시그니처(절대 변경 금지)** |
| Deep Navy | `#1B2845` | 앱 아이콘 컨테이너 |
| Cream | `#FAFAF7` | 배경 · 네거티브 (테두리 `#E5E3DC`) |

## 3. 배경별 사용 규칙 (핵심)
- **흰색/밝은 배경** → 마크는 **Ink(#0A0A0A) + Sky Blue** 사용. → `01-primary-transparent` 또는 `05-app-icon-cream`.
- **흰색이 아닌/어두운 배경** → 마크는 **Cream(#FAFAF7) + Sky Blue** 사용. → `02-reversed-transparent` 또는 `03-app-icon-navy`.
- K(파란색)는 배경과 무관하게 항상 Sky Blue `#4A90C2` 고정.

## 4. 워드마크 규칙
- `tk` + `L`(Sky Blue) + `een`. **가운데 L만 Sky Blue**, `tk`·`een`은 Ink(밝은 배경) 또는 Cream(어두운 배경).
- 폰트: Fraunces (display, 300/400). 단순 텍스트 표기는 색 강조 없이 `tkLeen`만 써도 무방.
- HTML 스니펫:
```html
<span style="font-family:Fraunces,Georgia,serif;color:#0A0A0A">
  tk<span style="color:#4A90C2">L</span>een
</span>
```

## 5. 로고 마크 구조 (지오메트리, viewBox 0 0 200 200)
- **T**: 가로바 `x40 y40 80×20` + 세로 기둥 `x60 y60 20×40`.
- **k**: 기둥이 아래로 이어진 파란 `x60 y100 20×60` + 위 대각 팔 `(80,100)(100,80)(120,60)` + 아래 대각 팔 `(80,120)(100,140)(120,160)` (각 20×20).
- T와 k는 x=60 기둥에서 이어져 **맞물린다**.

## 6. 로고 SVG 소스 (그대로 복사·임베드)

### 6.1 밝은 배경용 — 메인 마크 (Ink+Blue, 투명)  `01-primary-transparent`
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect x="40" y="40" width="80" height="20" fill="#0A0A0A"/>
  <rect x="60" y="60" width="20" height="40" fill="#0A0A0A"/>
  <rect x="60" y="100" width="20" height="60" fill="#4A90C2"/>
  <rect x="80" y="100" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="80" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="60" width="20" height="20" fill="#4A90C2"/>
  <rect x="80" y="120" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="140" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="160" width="20" height="20" fill="#4A90C2"/>
</svg>
```

### 6.2 어두운 배경용 — 리버스 마크 (Cream+Blue, 투명)  `02-reversed-transparent`
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect x="40" y="40" width="80" height="20" fill="#FAFAF7"/>
  <rect x="60" y="60" width="20" height="40" fill="#FAFAF7"/>
  <rect x="60" y="100" width="20" height="60" fill="#4A90C2"/>
  <rect x="80" y="100" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="80" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="60" width="20" height="20" fill="#4A90C2"/>
  <rect x="80" y="120" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="140" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="160" width="20" height="20" fill="#4A90C2"/>
</svg>
```

### 6.3 기본 앱 아이콘 (Navy 컨테이너 + Cream/Blue)  `03-app-icon-navy`
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect x="0" y="0" width="200" height="200" rx="44" fill="#1B2845"/>
  <rect x="40" y="40" width="80" height="20" fill="#FAFAF7"/>
  <rect x="60" y="60" width="20" height="40" fill="#FAFAF7"/>
  <rect x="60" y="100" width="20" height="60" fill="#4A90C2"/>
  <rect x="80" y="100" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="80" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="60" width="20" height="20" fill="#4A90C2"/>
  <rect x="80" y="120" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="140" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="160" width="20" height="20" fill="#4A90C2"/>
</svg>
```

### 6.4 라이트 톤 앱 아이콘 (Cream 컨테이너 + Ink/Blue)  `05-app-icon-cream`
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect x="1" y="1" width="198" height="198" rx="43" fill="#FAFAF7" stroke="#E5E3DC" stroke-width="2"/>
  <rect x="40" y="40" width="80" height="20" fill="#0A0A0A"/>
  <rect x="60" y="60" width="20" height="40" fill="#0A0A0A"/>
  <rect x="60" y="100" width="20" height="60" fill="#4A90C2"/>
  <rect x="80" y="100" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="80" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="60" width="20" height="20" fill="#4A90C2"/>
  <rect x="80" y="120" width="20" height="20" fill="#4A90C2"/>
  <rect x="100" y="140" width="20" height="20" fill="#4A90C2"/>
  <rect x="120" y="160" width="20" height="20" fill="#4A90C2"/>
</svg>
```

## 7. 전체 자산 목록 (원본 폴더 tkleen-brand-assets/svg·png)
| # | 파일 | 용도 |
|---|---|---|
| 01 | primary-transparent | 메인 마크, 라이트 배경 전반 |
| 02 | reversed-transparent | 다크 배경용(T가 크림색) |
| 03 | app-icon-navy | 기본 앱 아이콘(iOS/Android/PWA) |
| 04 | app-icon-ink | 대안 앱 아이콘(모던 톤) |
| 05 | app-icon-cream | 라이트 톤 앱 아이콘 |
| 06 | outlined-frame | 뱃지·인장·인증 마크 |
| 07 | circle-avatar | 소셜 프로필 |
| 08 | monochrome-ink | 1색 인쇄·흑백·워터마크 |
| 09 | lockup-horizontal-light | 라이트 배경 가로 락업(마크+워드마크+태그라인) |
| 10 | lockup-horizontal-dark | 다크 배경 가로 락업 |
| 11 | favicon | 브라우저 탭 아이콘 |
- 태그라인: `AI WORKFLOWS · INSURANCE & FINANCE`.
- PNG 사이즈: 16/32/64/128/256/512px(락업 360/720/1440px).

## 8. 최소 사이즈·여백·금지
- 마크 단독 최소 16px, 락업 최소 너비 120px. 마크 주변 여백은 마크 너비의 25% 이상.
- **금지**: 비율 변형(stretch)·회전·K 색 변경·마크 위 그래픽 겹치기·브랜드명 표기 변경.

## 9. 문서(HTML/PDF)에서의 tkLeen 적용 기본값
- 폰트: Fraunces(디스플레이)·Pretendard/Noto Sans CJK KR(본문)·JetBrains Mono/DejaVu Sans Mono(코드·메타).
- 표지: 밝은(Cream) 배경 → `05-app-icon-cream`(또는 `01-primary`) 마크를 크게, **tkLeen 워드마크(L만 Sky Blue)는 우측 상단**.
- 렌더링: 단독 HTML(정본) + PDF(Playwright/Chromium), 한글 Noto CJK.

---
*tkLeen · 보험·금융 실무를 위한 AI 워크플로우 · ai4insurance.com · Brand Assets v1.0 기준*
