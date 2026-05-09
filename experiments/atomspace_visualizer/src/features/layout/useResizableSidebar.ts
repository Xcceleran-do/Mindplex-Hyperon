import { createEffect, createSignal, onCleanup } from 'solid-js';

export const useResizableSidebar = (initialWidth = 320) => {
  const [sidebarWidth, setSidebarWidth] = createSignal(initialWidth);
  const [isResizing, setIsResizing] = createSignal(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = createSignal(false);

  const stopResizing = () => {
    setIsResizing(false);
  };

  const resize = (event: MouseEvent) => {
    if (!isResizing()) {
      return;
    }
    setSidebarWidth(Math.max(250, Math.min(600, event.clientX - 20)));
  };

  const startResizing = (event: MouseEvent) => {
    setIsResizing(true);
    event.preventDefault();
  };

  createEffect(() => {
    if (!isResizing()) {
      return;
    }

    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);

    onCleanup(() => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    });
  });

  return {
    sidebarWidth,
    isSidebarCollapsed,
    setIsSidebarCollapsed,
    startResizing,
  };
};
