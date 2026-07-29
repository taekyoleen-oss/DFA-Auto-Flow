# BMW M 스타일 — 앱 적용 가이드

> 이 문서 하나로 **다른 앱에도 같은 스타일을 그대로 이식**할 수 있다.
> 원칙: **컴포넌트를 고치지 않는다.** 색·라운드·타이포를 "토큰"에서 재정의하면
> 기존 `dark:bg-gray-800` 같은 클래스가 자동으로 BMW M 표면으로 바뀐다.
> 실제 변경 파일은 **5개뿐**이다.
>
> 적용 완료: `ML Auto Flow` · `ML_Auto_Flow-JMDC` · `DFA-Auto-Flow` · `life matrix flow new` (2026-07-29)

---

## 1. 디자인 원칙 (요약)

| 원칙 | 내용 |
|---|---|
| 캔버스 | 다크에서는 순수 검정(`#000`) 바닥 + 흰 타입이 원본 시스템의 얼굴이다. 단, **이 앱들의 기본 진입은 라이트**이고 다크는 사용자가 토글로 선택한다(작업 도구 특성상 밝은 화면 요구가 있어 두 모드를 모두 유지). |
| 타이포 | 디스플레이 **700 + UPPERCASE**, 본문 **300(Light)**. 이 굵기 대비가 시그니처 — 절대 흐리지 말 것. |
| 라운드 | **거의 항상 0px**, 원형 아이콘 버튼만 `9999px`. 중간값 없음(이진 규칙). |
| 색 | 브랜드 색은 **M 트라이컬러뿐**(파랑→파랑→빨강). CTA 색으로 쓰지 않는다. 액션은 흑/백. |
| 버튼 | 사각 실루엣 + 대문자 + 자간(machined). 채움 없음이 기본, 흰 아웃라인이 곧 버튼. |
| 장식 | 그림자·그라디언트 없음. 유일한 장식은 4px M 스트라이프. |
| 여백 | 4px 기준, 섹션 96px / 히어로 64px / 카드 24px. |

---

## 2. 토큰

### 색
| 토큰 | 값 | 용도 |
|---|---|---|
| `canvas` | `#000000` | 페이지 바닥 |
| `surface-soft` | `#0d0d0d` | 패널·스펙 셀 |
| `surface-card` | `#1a1a1a` | 카드·보조 버튼 |
| `surface-elevated` | `#262626` | 중첩 카드 · 강한 헤어라인 |
| `carbon-gray` | `#2b2b2b` | 기술 스펙 카드 |
| `hairline` | `#3c3c3c` | 1px 구분선 |
| `on-dark` | `#ffffff` | 헤드라인·주요 텍스트 |
| `body` | `#bbbbbb` | 본문 |
| `body-strong` | `#e6e6e6` | 강조 본문 |
| `muted` | `#7e7e7e` | 캡션·푸터 |
| `m-blue-light` | `#0066b1` | M 스트라이프 1 |
| `m-blue-dark` / `bmw-blue` | `#1c69d4` | M 스트라이프 2 · 헤리티지 블루 |
| `m-red` | `#e22718` | M 스트라이프 3 |
| `warning` / `success` | `#f4b400` / `#0fa336` | semantic(절제해서) |

### 타이포
| 토큰 | 크기/굵기 | 자간 |
|---|---|---|
| display-xl / lg / md / sm | 80 / 56 / 40 / 32 · **700** | 0 |
| title-lg / md | 24 · 700 / 20 · 400 | 0 |
| label-uppercase / button | 14 · **700** | **1.5px** |
| body-md / sm | 16 / 14 · **300** | 0 |
| caption | 12 · 400 | 0.5px |

폰트: **BMW Type Next Latin**. 없으면 **Inter**(700/300). 웹폰트를 추가하지 않으려면
`-apple-system, Segoe UI, Roboto, Pretendard, Malgun Gothic` 시스템 스택으로 충분하다.

---

## 3. 이식 절차 — 파일 5개

### ① `tailwind.config.js` — 핵심 (이거 하나로 화면 전체가 바뀐다)

`theme: { extend: {} }` 를 아래로 교체한다. `content` / `safelist` 는 앱 것을 그대로 둔다.

```js
theme: {
  extend: {
    colors: {
      canvas: "#000000", "surface-soft": "#0d0d0d", "surface-card": "#1a1a1a",
      "surface-elevated": "#262626", "carbon-gray": "#2b2b2b", hairline: "#3c3c3c",
      "m-blue-light": "#0066b1", "m-blue-dark": "#1c69d4", "m-red": "#e22718",
      "bmw-blue": "#1c69d4",

      // ★ 기존 컴포넌트가 쓰는 gray 스케일을 BMW 표면으로 재매핑
      //   → dark:bg-gray-800 등이 컴포넌트 수정 없이 BMW 표면이 된다
      gray: {
        50:"#fafafa", 100:"#f2f2f2", 200:"#e6e6e6", 300:"#d4d4d4",
        400:"#bbbbbb",  // body
        500:"#7e7e7e",  // muted
        600:"#3c3c3c",  // hairline(보더)
        700:"#262626",  // surface-elevated
        800:"#0d0d0d",  // surface-soft(패널)
        900:"#000000", 950:"#000000",  // canvas
      },

      // 액션 색은 스펙 semantic으로 정렬 + 알록달록한 채움은 중립화
      blue:{600:"#1c69d4",700:"#0653b6"}, green:{600:"#0fa336",700:"#0c8a2d"},
      red:{600:"#e22718",700:"#c11f12"}, yellow:{500:"#f4b400"},
      purple:{600:"#262626",700:"#3c3c3c"}, indigo:{600:"#262626",700:"#3c3c3c"},
      cyan:{600:"#262626",700:"#3c3c3c"},  teal:{600:"#262626",700:"#3c3c3c"},
    },
    fontFamily: { sans: ["Inter","-apple-system","BlinkMacSystemFont",
      "Segoe UI Variable Text","Segoe UI","Roboto","Pretendard","Malgun Gothic","sans-serif"] },
    letterSpacing: { machined: "1.5px" },
  },
  // "거의 항상 0, 가끔 원형"
  borderRadius: { none:"0", sm:"0", DEFAULT:"0", md:"0", lg:"0", xl:"0",
                  "2xl":"0", "3xl":"0", full:"9999px" },
},
```

> ⚠️ **dev 서버는 반드시 재시작**해야 config 변경이 반영된다(HMR로는 안 잡힘).
> 확인: `curl -s http://localhost:PORT/index.css | grep -c "147 51 234"` → `0` 이어야 함(=기본 보라 사라짐).

### ② `index.css` — 토큰으로 표현 못 하는 3가지만

```css
@layer base {
  html.dark body { background-color:#000; color:#bbbbbb; }
  h1, h2, h3 { font-weight:700; text-transform:uppercase; letter-spacing:0; }
  button { text-transform:uppercase; letter-spacing:0.06em; }
  input, textarea, select, option, pre, code { text-transform:none; letter-spacing:normal; }
}
@layer components {
  .m-stripe {           /* M 트라이컬러 4px — 이 시스템의 유일한 장식 */
    height:4px;
    background:linear-gradient(90deg,#0066b1 0%,#0066b1 33.33%,
      #1c69d4 33.33%,#1c69d4 66.66%,#e22718 66.66%,#e22718 100%);
  }
}
```

> 한글 UI에서 `uppercase`는 무해하다(한글 미변환, 라틴 라벨만 대문자화).
> 단, **입력값·코드 영역은 반드시 제외**해야 사용자가 친 값이 대문자로 보이지 않는다.

### ③ `index.html` — 캔버스 바닥 · 스크롤바 · 폰트

```css
body { font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI Variable Text',
       'Segoe UI',Roboto,'Pretendard','Malgun Gothic',sans-serif; }
.canvas-bg { background-color:#000000; }        /* 기존 #111827 */
.dark .panel-scrollbar::-webkit-scrollbar-track { background:#0d0d0d; }
.panel-scrollbar::-webkit-scrollbar-thumb { background:#7e7e7e; border-radius:0; }
.dark .panel-scrollbar::-webkit-scrollbar-thumb { background:#3c3c3c; }
```

### ④ `contexts/ThemeContext.tsx` — 기본 테마 고정

`prefers-color-scheme` 분기를 지우고 기본값을 **`return 'light'`** 로 고정한다.
(원본 BMW M은 다크 전용이지만, 이 앱들은 작업 도구라 기본 진입을 라이트로 두고
다크는 사용자가 토글로 선택하면 localStorage에 저장돼 유지된다.
브랜드 톤을 그대로 쓰려면 이 값만 `'dark'`로 바꾸면 된다 — 나머지 토큰은 두 모드를 모두 지원한다.)

### ⑤ `App.tsx` — M 스트라이프 1줄 + 헤더 2색

`</header>` 바로 아래:

```tsx
<div className="m-stripe flex-shrink-0" aria-hidden />
```

모듈 최상단(import 아래)에 2색 상수:

```tsx
const BTN_PRIMARY =
  "bg-gray-900 dark:bg-white text-white dark:text-black border border-gray-900 dark:border-white hover:bg-gray-700 dark:hover:bg-gray-200";
const BTN_FLAT =
  "bg-transparent border border-gray-400 dark:border-gray-600 text-gray-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-800 hover:border-gray-900 dark:hover:border-white";
```

헤더 버튼 규칙 — **딱 두 가지만 쓴다**:

| 상태 | 클래스 | 대상 |
|---|---|---|
| 주요 액션 · 활성 | `BTN_PRIMARY` (흰 채움 + 검정 라벨) | 전체 실행, 열린 드롭다운, 고급기능 해제됨 |
| 그 외 전부 | `BTN_FLAT` (투명 + 1px 헤어라인 + 흰 라벨) | 샘플·내 작업·AI·PPT·코드·비교·보고서… |

치환 시 주의:
- `bg-*-600 text-white` → `BTN_PRIMARY` 로 바꿀 때 **`text-white`를 반드시 제거**(흰 배경 + 흰 글자 = 안 보임).
- 문자열 className은 템플릿 리터럴(`` className={`… ${BTN_FLAT}`} ``)로 바꿔야 상수를 넣을 수 있다.
- 진행바·배지도 흑백으로: `bg-green-500` → `bg-gray-900 dark:bg-white`.
- 워드마크(로고+제목)는 흰색: `text-blue-600 dark:text-blue-300` → `text-gray-900 dark:text-white`.

---

## 4. 건드리지 않는 것 (의도적)

- **모듈 카테고리 색(`*-400` / `*-500`)** — 툴박스/캔버스에서 종류를 구분하는 **기능적** 색이라 보존.
- **semantic 상태 색** — 에러(빨강)·경고(노랑)·성공(초록)은 `600`/`500` 단계에서 스펙 값으로만 정렬.
- **로직·상태·핸들러** — 순수 스타일 변경. Pyodide 실행·내보낸 Python·`verify:pipelines` 와 무관.

### ⑥ 필수 후속 — 텍스트 대비 점검 (놓치기 쉬움 · 두 모드 모두)

gray 재매핑으로 **회색 스케일의 의미가 바뀐다**. `600`은 더 이상 "중간 회색 글자"가 아니라
보더 톤(#3c3c3c)이고, `300/400`은 밝은 회색(#d4d4d4/#bbbbbb)이다.
그래서 **양쪽 모드에서 각각 반대 방향으로** 안 보이는 자리가 생긴다.
(실제로 캔버스 모듈 카드의 `Pending` 라벨이 다크·라이트 양쪽에서 순차적으로 문제가 됐다.)

```bash
# 다크 분기에 어두운 회색 글자 / 라이트 분기에 밝은 회색 글자 찾기
grep -rnE 'dark:text-gray-(600|700)|: "text-gray-(600|700)"' components/ *.tsx   # 다크에서 안 보임
grep -rnE "'light' \? 'text-gray-(300|400|500)'|\? \"text-gray-(300|400)\"" components/ *.tsx  # 라이트에서 안 보임
```

| 용도 | 다크 분기 | 라이트 분기 |
|---|---|---|
| 주요 라벨(모듈명, 상태) | `text-white` | `text-gray-900`(=#000) |
| 비활성/보조 라벨 | `text-gray-300` (대비 ≈ 13.9:1) | `text-gray-800`(=#0d0d0d) |
| 메타(실행시간, 캡션) | `text-gray-300` ~ `text-gray-200` | `text-gray-700`(=#262626) |

**테마 분기가 없는 정적 요소**(예: `statusIcons` 같은 모듈 레벨 상수 맵)는 `theme` 값을 못 받으므로
`dark:` 변형으로 처리한다 — `text-gray-700 dark:text-gray-300`.

## 5. 검증 체크리스트

```bash
npx tsc --noEmit          # 0
npm run build             # exit 0 (dist CSS 생성)
npm run verify:pipelines  # 기존 PASS 수 유지 (스타일은 재현성과 무관하지만 관례)
```

브라우저(dev 서버 **재시작 후**):
1. 새 프로필로 진입 → **라이트 기본**(다크로 토글하면 검정 캔버스)
2. 헤더 아래 M 스트라이프(파랑–파랑–빨강) 렌더
3. 헤더 버튼이 **흰 채움 / 아웃라인 2종**뿐
4. 모든 모서리 각짐(원형 아이콘 버튼만 예외)
5. 모달·패널에 보라/청록 채움이 남아있지 않음
6. **캔버스 모듈 카드의 모듈명·상태 라벨이 또렷하게 읽히는지**(비활성 상태 포함)
7. 다크 토글 → 정상 전환 + **다크에서도 6번 재확인**(모드마다 안 보이는 자리가 다르다)
8. console error 0

빠른 실측 방법(브라우저 콘솔):

```js
document.querySelectorAll('.module-header').forEach(h => {
  const c = h.parentElement;
  const b = [...c.querySelectorAll('button')].find(x => /Pending|결과 보기/.test(x.textContent));
  console.log(getComputedStyle(b).color, '/', getComputedStyle(c).backgroundColor);
});
// 다크: rgb(212,212,212) / rgb(13,13,13) · 라이트: rgb(13,13,13) / rgb(255,255,255)
```

## 6. 알려진 제약

- 웹폰트 미포함 → 실제 BMW Type Next Latin 대신 시스템 폰트. 필요하면 `@fontsource/inter` 추가(외부 CDN 금지).
- 사진 기반 히어로 밴드(원본 시스템의 핵심)는 도구형 앱이라 미적용 — 캔버스 자체가 히어로 역할.
- `text-transform: uppercase`는 라틴 라벨만 바꾼다. 한글 UI에서는 시각 효과가 제한적.
