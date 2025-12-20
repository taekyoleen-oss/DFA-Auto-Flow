/**
 * Sample loading utilities
 */

export interface SampleModel {
  name: string;
  modules: any[];
  connections: any[];
  filename?: string;
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
























