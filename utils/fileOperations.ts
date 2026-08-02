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
  /** 시작 폴더(사용자가 OPEN > 폴더 설정으로 지정한 경우). 없으면 브라우저가 id로 마지막 폴더를 기억한다. */
  startIn?: FileSystemDirectoryHandle | null;
}

export interface PipelineState {
  modules: any[];
  connections: any[];
  projectName?: string;
}

/**
 * Save pipeline state to a file — 항상 로컬 다운로드로 저장 (브라우저 다운로드 폴더)
 */
export async function savePipeline(
  state: PipelineState,
  options: SavePipelineOptions
): Promise<void> {
  try {
    const fileName =
      (state.projectName && state.projectName.trim()
        ? state.projectName.trim()
        : "pipeline") + options.extension;
    const blob = new Blob([JSON.stringify(state, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    if (options.onSuccess) {
      options.onSuccess(fileName);
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
  // File System Access API가 있으면 그쪽을 쓴다 — id를 주면 브라우저가 이 용도의
  // '마지막에 열었던 폴더'를 기억해 다음에 거기서 연다. 없으면 기존 input 폴백.
  if ("showOpenFilePicker" in window) {
    try {
      const [fileHandle] = await (window as any).showOpenFilePicker({
        id: "dfa-pipeline",
        ...(options.startIn ? { startIn: options.startIn } : {}),
        types: [
          {
            description: "Pipeline Files",
            accept: {
              "application/json": [options.extension, ".mla", ".json"],
            },
          },
        ],
      });
      const file = await fileHandle.getFile();
      return JSON.parse(await file.text()) as PipelineState;
    } catch (error: any) {
      if (error?.name === "AbortError") return null;
      if (options.onError) {
        options.onError(
          error instanceof Error ? error : new Error(String(error))
        );
      }
      return null;
    }
  }

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































