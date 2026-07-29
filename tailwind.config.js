/** @type {import('tailwindcss').Config} */
// CDN(cdn.tailwindcss.com) → PostCSS 빌드 통합.
// darkMode는 기존 인라인 설정(index.html)과 동일하게 'class'.
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./contexts/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./utils/**/*.{ts,tsx}",
  ],
  // 동적 조립 클래스 방어(PropertiesPanel `text-${highlight.color}-400` —
  // 현재 color 값이 실제로 설정되는 곳은 없으나 향후 사용 대비).
  safelist: [
    "text-green-400",
    "text-red-400",
    "text-yellow-400",
    "text-blue-400",
    "text-purple-400",
  ],
  theme: {
    extend: {
      colors: {
        // BMW M 브랜드 토큰(신규 클래스로 직접 사용: bg-canvas, text-m-red 등)
        canvas: "#000000",
        "surface-soft": "#0d0d0d",
        "surface-card": "#1a1a1a",
        "surface-elevated": "#262626",
        "carbon-gray": "#2b2b2b",
        hairline: "#3c3c3c",
        "m-blue-light": "#0066b1",
        "m-blue-dark": "#1c69d4",
        "m-red": "#e22718",
        "bmw-blue": "#1c69d4",

        // 기존 컴포넌트가 쓰는 gray 스케일 → BMW M 표면/텍스트로 재매핑.
        // (다크 = 표면, 라이트 = 기존과 비슷한 밝기 유지 → 두 모드 모두 동작)
        gray: {
          50: "#fafafa",
          100: "#f2f2f2",
          200: "#e6e6e6", // body-strong / 라이트 표면
          300: "#d4d4d4", // 라이트 보더 / 다크 강조 텍스트
          400: "#bbbbbb", // body — 다크 보조 텍스트
          500: "#7e7e7e", // muted — 캡션/푸터
          600: "#3c3c3c", // hairline — 보더
          700: "#262626", // surface-elevated / hairline-strong
          800: "#0d0d0d", // surface-soft — 패널/카드
          900: "#000000", // canvas
          950: "#000000",
        },

        // 액션 색: BMW 헤리티지 블루 + 스펙의 semantic 값으로 정렬.
        // (모듈 카테고리 색상 400/500 단계는 식별용이라 그대로 둔다)
        blue: { 600: "#1c69d4", 700: "#0653b6" },
        green: { 600: "#0fa336", 700: "#0c8a2d" },
        red: { 600: "#e22718", 700: "#c11f12" },
        yellow: { 500: "#f4b400" },
        // 액션 채움으로 쓰이던 보라/남색/청록 계열은 중립 표면으로 통일
        // (버튼/탭이 알록달록해지지 않도록 — 브랜드 색은 M 트라이컬러뿐)
        purple: { 600: "#262626", 700: "#3c3c3c" },
        indigo: { 600: "#262626", 700: "#3c3c3c" },
        cyan: { 600: "#262626", 700: "#3c3c3c" },
        teal: { 600: "#262626", 700: "#3c3c3c" },
      },
      fontFamily: {
        // BMW Type Next Latin 대체 스택(외부 웹폰트 추가 없음 — 시스템 폰트로 700/300 대비)
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI Variable Text",
          "Segoe UI",
          "Roboto",
          "Pretendard",
          "Malgun Gothic",
          "sans-serif",
        ],
      },
      letterSpacing: {
        machined: "1.5px", // 대문자 라벨/버튼 트래킹
      },
    },
    // "거의 항상 0, 가끔 원형" — BMW M의 이진 라운드 규칙.
    borderRadius: {
      none: "0",
      sm: "0",
      DEFAULT: "0",
      md: "0",
      lg: "0",
      xl: "0",
      "2xl": "0",
      "3xl": "0",
      full: "9999px", // 원형 아이콘 버튼만 예외
    },
  },
  plugins: [],
};
