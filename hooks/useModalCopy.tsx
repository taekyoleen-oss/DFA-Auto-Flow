import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useCopyOnCtrlC } from './useCopyOnCtrlC';

export const useModalCopy = () => {
  const contentRef = useRef<HTMLDivElement>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);

  useCopyOnCtrlC(contentRef);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY });
  }, []);

  const handleCopy = useCallback(async () => {
    if (contentRef.current) {
      const text = contentRef.current.innerText || contentRef.current.textContent || '';
      if (text.trim()) {
        try {
          await navigator.clipboard.writeText(text);
          setContextMenu(null);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      }
    }
  }, []);

  useEffect(() => {
    if (!contextMenu) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Element;
      if (contextMenu && !target.closest('.context-menu')) {
        setContextMenu(null);
      }
    };
    
    // 다음 틱에서 이벤트 리스너 추가 (버튼 클릭이 먼저 처리되도록)
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside, true);
      document.addEventListener('click', handleClickOutside, true);
    }, 0);
    
    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside, true);
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [contextMenu]);

  const ContextMenuComponent = contextMenu ? (
    <div
      className="context-menu fixed bg-white border border-gray-300 rounded-lg shadow-lg py-1 z-[100]"
      style={{ left: contextMenu.x, top: contextMenu.y }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <button
        type="button"
        onMouseDown={(e) => {
          e.stopPropagation();
        }}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          handleCopy();
        }}
        className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer focus:outline-none focus:bg-gray-100"
      >
        복사 (Ctrl+C)
      </button>
    </div>
  ) : null;

  return {
    contentRef,
    contextMenu,
    handleContextMenu,
    handleCopy,
    ContextMenuComponent,
  };
};

