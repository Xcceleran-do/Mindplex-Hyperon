export const env = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || '',
  ingestionEnabled: import.meta.env.VITE_BYPASS_INGESTION !== 'true',
  maxVisualizationArticles: Number(import.meta.env.VITE_MAX_VIS_ARTICLES || 1500),
} as const;

export const isEnabledNumber = (value: number) => Number.isFinite(value) && value > 0;
