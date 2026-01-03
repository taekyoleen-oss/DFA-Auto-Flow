import React, { useEffect, useRef, useCallback, useState } from 'react';

export const useCopyOnCtrlC = (contentRef: React.RefObject<HTMLElement>) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !e.shiftKey && !e.altKey) {
        const activeElement = document.activeElement;
        const isEditingText =
          activeElement &&
          (activeElement.tagName === "INPUT" ||
            activeElement.tagName === "TEXTAREA" ||
            (activeElement as HTMLElement).isContentEditable);
        
        if (!isEditingText && contentRef.current) {
          // 모달이 열려있는지 확인 - contentRef가 모달 내부에 있는지 확인
          const modalElement = contentRef.current.closest('.fixed.inset-0');
          if (modalElement && modalElement.classList.contains('z-50')) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            const text = contentRef.current.innerText || contentRef.current.textContent || '';
            if (text.trim()) {
              navigator.clipboard.writeText(text).then(() => {
                console.log('Copied to clipboard');
              }).catch((err) => {
                console.error('Failed to copy:', err);
              });
            }
            return false;
          }
        }
      }
    };

    // capture phase에서 이벤트를 잡아서 다른 핸들러보다 먼저 실행
    document.addEventListener('keydown', handleKeyDown, true);
    return () => document.removeEventListener('keydown', handleKeyDown, true);
  }, [contentRef]);
};

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
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenu && !(e.target as Element).closest('.context-menu')) {
        setContextMenu(null);
      }
    };
    document.addEventListener('click', handleClickOutside, true);
    return () => {
      document.removeEventListener('click', handleClickOutside, true);
    };
  }, [contextMenu]);

  return {
    contentRef,
    contextMenu,
    handleContextMenu,
    handleCopy,
  };
};





















