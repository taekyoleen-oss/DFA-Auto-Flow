// 모델 분석보고서(ModelAnalysisReport) — AI(generateAiText) 기반 HTML 생성 + 결정적 폴백.
// 문서화(메타) 기능: codeSnippets/export/verify와 무관(Python 재현성 불변식 비대상).
//
// DFA 적응점:
//  - ML은 lib/aiHelpers의 Claude SDK 스트리밍을 직접 호출하지만, DFA는 멀티프로바이더
//    공용 generateAiText(opts)를 사용한다(tier:"smart"=Sonnet급). 스트리밍 불필요.
//  - generateAiText는 키가 없으면 throw하므로 반드시 try/catch로 감싸 폴백으로 강등한다.

import { ReportContext } from "../types";
import { generateAiText } from "./aiClient";
import {
  buildModelReportHtmlFallback,
  sanitizeReportContextForPrompt,
} from "./modelReport";

/** AI가 생성한 HTML이 자기완결·안전한지 검증한다(<html> 포함, <script> 없음). */
function isValidReportHtml(html: string): boolean {
  if (!html || html.length < 200) return false;
  const lower = html.toLowerCase();
  if (!lower.includes("<html")) return false;
  if (lower.includes("<script")) return false; // 보안: 스크립트 금지
  return true;
}

/** AI 응답에서 코드펜스(```html …```)를 제거하고 HTML만 남긴다. */
function stripHtmlFence(text: string): string {
  let t = (text || "").trim();
  // ```html ... ``` 또는 ``` ... ``` 제거
  const fence = t.match(/```(?:html)?\s*([\s\S]*?)```/i);
  if (fence && fence[1]) t = fence[1].trim();
  // 앞부분 잡설이 있으면 <!DOCTYPE 또는 <html부터 시작.
  const idx = t.search(/<!doctype html|<html/i);
  if (idx > 0) t = t.slice(idx);
  return t.trim();
}

function buildModelReportPrompt(ctx: ReportContext): string {
  const styleHint =
    'CSS 변수(--ink/--muted/--line/--accent/--accent-soft/--th 등)와 .report/.cover/.badge/.callout/.callout.warn/.kpi-grid/.kpi/table(th,td,.num)/pre 클래스를 사용한 인라인 <style>를 head에 포함하라(예시 디자인과 동일한 깔끔한 문서 스타일).';
  return `너는 보험계리/머신러닝 모델 문서 작성가다. 아래 파이프라인 메타데이터(JSON)와 사용자 추가정보만으로, 한국어 **자기완결 HTML** 모델 분석보고서를 1개 작성한다.

[필수 규칙]
1) 메타데이터에 있는 수치만 사용한다(수치 창작 절대 금지). 없는 값은 비우거나 "(자료 없음)"으로 둔다.
2) 데이터셋/도메인 배경 서술에 일반 지식을 쓸 수 있으나, 그런 문장은 "(일반 지식 기반)"으로 표기해 실측과 구분한다.
3) 출력은 \`<!DOCTYPE html>\`로 시작하는 **완전한 HTML 1개**. 외부 CSS/JS/폰트 0, \`<script>\` 태그 절대 금지(보안). ${styleHint}
4) 섹션 구조: 표지(badge "모델 분석보고서")·1.요약(+KPI 그리드)·2.데이터셋 개요(표+표본)·3.변수(컬럼) 사전(사용/미사용 특성 명시)·4.타깃/클래스 또는 군집 분포·5.모델 개발 과정(파이프라인 다이어그램 pre + 단계별 파라미터)·6.분석 결과와 해석(혼동행렬·임계값·지표)·7.재현성·8.결론 및 한계.
5) 표·callout·KPI 카드를 적극 활용하고, 미사용 특성·클래스 불균형 등 한계를 정직하게 기술한다.
6) HTML 외에 다른 텍스트(설명·코드펜스)를 출력하지 마라.

[파이프라인 메타데이터 JSON]
${JSON.stringify(sanitizeReportContextForPrompt(ctx), null, 2)}

${
  ctx.extraInfo && ctx.extraInfo.trim()
    ? `[사용자 추가정보 — 최우선 근거로 반영]\n${ctx.extraInfo.trim()}`
    : "[사용자 추가정보 없음 — 데이터셋/도메인 배경은 일반 지식으로 보강하되 '(일반 지식 기반)'으로 표기하라]"
}`;
}

/**
 * 모델 분석보고서 HTML을 생성한다.
 * - generateAiText({ tier:"smart" })로 단일 호출(멀티프로바이더, Anthropic 기본=Sonnet급).
 *   긴 HTML을 위해 maxTokens=16000을 지정한다.
 * - 키 없음/네트워크/검증 실패 등 어떤 오류든 결정적 폴백으로 graceful degradation(throw 금지).
 * - AI 출력은 자기완결·안전(<html> 포함·<script> 없음) 검증을 통과해야 채택된다.
 */
export async function generateModelReportHtml(
  ctx: ReportContext
): Promise<{ html: string; source: "ai" | "fallback" }> {
  const prompt = buildModelReportPrompt(ctx);
  try {
    // generateAiText는 키가 없으면 MissingApiKeyError를 throw → catch에서 폴백.
    const raw = await generateAiText({ prompt, tier: "smart", maxTokens: 16000 });
    const html = stripHtmlFence(raw);
    if (isValidReportHtml(html)) {
      return { html, source: "ai" };
    }
  } catch (err) {
    console.warn("[generateModelReportHtml] AI 생성 실패, 결정적 폴백 사용:", err);
  }
  return { html: buildModelReportHtmlFallback(ctx), source: "fallback" };
}
