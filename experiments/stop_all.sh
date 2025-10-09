#!/bin/bash
echo "🛑 Stopping services..."
lsof -ti:5000 | xargs kill -9 2>/dev/null && echo "✓ Backend stopped" || echo "Backend not running"
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "✓ Frontend stopped" || echo "Frontend not running"
pkill -f 'mining_api.py' 2>/dev/null
echo "✅ Done!"
