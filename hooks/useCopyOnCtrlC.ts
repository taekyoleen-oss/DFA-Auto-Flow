import { useEffect, useRef } from 'react';

export const useCopyOnCtrlC = (contentRef: React.RefObject<HTMLElement>) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        const activeElement = document.activeElement;
        const isEditingText =
          activeElement &&
          (activeElement.tagName === "INPUT" ||
            activeElement.tagName === "TEXTAREA" ||
            (activeElement as HTMLElement).isContentEditable);
        
        if (!isEditingText && contentRef.current) {
          const text = contentRef.current.innerText || contentRef.current.textContent || '';
          if (text.trim()) {
            navigator.clipboard.writeText(text).catch(console.error);
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [contentRef]);
};






