// API Configuration
// Centralized API endpoint configuration using environment variables

export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || '',
  
  // API endpoints
  ENDPOINTS: {
    HEALTH: '/api/health',
    MINE: '/api/mine',
    INGEST: '/api/ingest',
    CHAT: '/api/chat',
    CHAT_ANALYZE: '/api/chat/analyze',
    CHAT_CLEAR: '/api/chat/clear',
  },
  
  // Helper functions
  getUrl: (endpoint: string) => {
    const baseUrl = API_CONFIG.BASE_URL;
    return `${baseUrl}${endpoint}`;
  },
  
  // Common headers
  HEADERS: {
    'Content-Type': 'application/json',
  },
} as const;

// Export commonly used URLs
export const API_URLS = {
  MINE: API_CONFIG.getUrl(API_CONFIG.ENDPOINTS.MINE),
  CHAT: API_CONFIG.getUrl(API_CONFIG.ENDPOINTS.CHAT),
  CHAT_ANALYZE: API_CONFIG.getUrl(API_CONFIG.ENDPOINTS.CHAT_ANALYZE),
  CHAT_CLEAR: API_CONFIG.getUrl(API_CONFIG.ENDPOINTS.CHAT_CLEAR),
} as const;