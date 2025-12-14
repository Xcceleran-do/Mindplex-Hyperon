#!/usr/bin/env python3
"""
Shared API Configuration for Experiments
Loads API URLs from environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env.local')

# Also try to load from current directory if in a subdirectory
if not os.path.exists('.env.local') and os.path.exists('../.env.local'):
    load_dotenv('../.env.local')

# API Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'https://super-duper-winner-97q9rxp6vvx9hxq5q-5000.app.github.dev')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'https://super-duper-winner-97q9rxp6vvx9hxq5q-3000.app.github.dev/')

# API Endpoints
API_ENDPOINTS = {
    'HEALTH': f'{API_BASE_URL}/api/health',
    'MINE': f'{API_BASE_URL}/api/mine',
    'CHAT': f'{API_BASE_URL}/api/chat',
    'CHAT_ANALYZE': f'{API_BASE_URL}/api/chat/analyze',
    'CHAT_CLEAR': f'{API_BASE_URL}/api/chat/clear',
}

def get_api_url(endpoint: str = '') -> str:
    """Get API URL with optional endpoint"""
    return f"{API_BASE_URL}{endpoint}"

# Export commonly used URLs
HEALTH_URL = API_ENDPOINTS['HEALTH']
MINE_URL = API_ENDPOINTS['MINE']
CHAT_URL = API_ENDPOINTS['CHAT']
CHAT_ANALYZE_URL = API_ENDPOINTS['CHAT_ANALYZE']