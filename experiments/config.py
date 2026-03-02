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
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

