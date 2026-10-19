const positiveInteger = (value: unknown, fallback: number) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
};

export const env = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || '',
  ingestionEnabled: import.meta.env.VITE_INGESTION_ENABLED !== 'false',
  maxVisualizationArticles: positiveInteger(import.meta.env.VITE_MAX_VIS_ARTICLES, 1500),
  defaultChainDepth: positiveInteger(import.meta.env.VITE_DEFAULT_CHAIN_DEPTH, 3),
} as const;

export const isEnabledNumber = (value: number) => Number.isFinite(value) && value > 0;
