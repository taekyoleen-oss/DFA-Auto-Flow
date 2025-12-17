/**
 * File operations for saving and loading pipeline files
 */

export interface SavePipelineOptions {
  extension: string;
  description?: string;
  onSuccess?: (fileName: string) => void;
  onError?: (error: Error) => void;
}

export interface LoadPipelineOptions {
  extension: string;
  onError?: (error: Error) => void;
}

export interface PipelineState {
  modules: any[];
  connections: any[];
  projectName?: string;
}

/**
 * Save pipeline state to a file
 */
export async function savePipeline(
  state: PipelineState,
  options: SavePipelineOptions
): Promise<void> {
  try {
    const jsonContent = JSON.stringify(state, null, 2);
    const blob = new Blob([jsonContent], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.projectName || "pipeline"}${options.extension}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    if (options.onSuccess) {
      options.onSuccess(link.download);
    }
  } catch (error: any) {
    if (options.onError) {
      options.onError(error instanceof Error ? error : new Error(String(error)));
    } else {
      throw error;
    }
  }
}

/**
 * Load pipeline state from a file
 */
export async function loadPipeline(
  options: LoadPipelineOptions
): Promise<PipelineState | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = options.extension;
    
    input.onchange = async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      
      try {
        const text = await file.text();
        const state = JSON.parse(text) as PipelineState;
        resolve(state);
      } catch (error: any) {
        if (options.onError) {
          options.onError(error instanceof Error ? error : new Error(String(error)));
        }
        resolve(null);
      }
    };
    
    input.oncancel = () => {
      resolve(null);
    };
    
    input.click();
  });
}












