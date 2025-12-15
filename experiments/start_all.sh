#!/bin/bash
echo "🚀 Starting services..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 2
cd /workspaces/Mindplex-Hyperon/experiments
python3 mining_api.py > backend.log 2>&1 &
echo "Backend started on port 5000"
sleep 3
cd atomspace_visualizer
npm run dev > frontend.log 2>&1 &
echo "Frontend started on port 3000"
sleep 2
# Load URLs from config
API_URL=$(python3 -c "from config import API_BASE_URL; print(API_BASE_URL)" 2>/dev/null || echo "https://mindplex-hyperon-3.onrender.com")
FRONTEND_URL=$(python3 -c "from config import FRONTEND_URL; print(FRONTEND_URL)" 2>/dev/null || echo "https://mindplex-hyperon-4.onrender.com/")
echo "✅ Done! Backend: $API_URL, Frontend: $FRONTEND_URL"