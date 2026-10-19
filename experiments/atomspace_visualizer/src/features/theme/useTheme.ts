import { createEffect, createSignal } from 'solid-js';

export type ThemeMode = 'auto' | 'light' | 'dark';

const STORAGE_KEY = 'theme';

export const useTheme = () => {
  const [theme, setTheme] = createSignal<ThemeMode>((localStorage.getItem(STORAGE_KEY) as ThemeMode) || 'auto');

  const applyTheme = (nextTheme: ThemeMode) => {
    if (nextTheme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }

    localStorage.setItem(STORAGE_KEY, nextTheme);
    setTheme(nextTheme);
  };

  createEffect(() => {
    applyTheme(theme());
  });

  return {
    theme,
    applyTheme,
  };
};
