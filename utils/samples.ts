/**
 * Sample loading utilities
 */

export interface SampleModel {
  name: string;
  modules: any[];
  connections: any[];
  filename?: string;
  /**
   * 선택적 메타데이터 (모두 backward-compatible · additive).
   * 없으면 기존 동작과 완전히 동일하게 표시된다.
   */
  description?: string;
  category?: string;
  tags?: string[];
  expectedOutput?: string;
  /** DFA 도메인 속성: 분포가정 (예: "Poisson-LogNormal 집계분포") */
  distributionAssumption?: string;
  /** DFA 도메인 속성: 재보험 레이어 (예: "XoL 5M xs 5M") */
  reinsuranceLayer?: string;
}

/**
 * Load a sample from a folder (via server API)
 */
export async function loadSampleFromFolder(
  filename: string,
  baseUrl: string
): Promise<SampleModel | null> {
  try {
    const response = await fetch(`${baseUrl}/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to load sample: ${response.statusText}`);
    }
    const data = await response.json();
    return data as SampleModel;
  } catch (error) {
    console.error("Error loading sample from folder:", error);
    return null;
  }
}

/**
 * Load list of samples from a folder (via server API)
 */
export async function loadFolderSamples(
  listUrl: string
): Promise<SampleModel[]> {
  try {
    const response = await fetch(listUrl);
    if (!response.ok) {
      throw new Error(`Failed to load samples list: ${response.statusText}`);
    }
    const data = await response.json();
    return Array.isArray(data) ? data : [];
  } catch (error) {
    console.error("Error loading folder samples:", error);
    return [];
  }
}































