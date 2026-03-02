# AtomSpace Visualizer with AI Chat Feature

## Overview

This enhanced atomspace visualizer includes an AI-powered chat interface that helps users understand mining results, analyze patterns, and visualize exact pattern matches on the graph.

## Features

### 1. **AI Chat Interface** 🤖
- Real-time chat with Gemini AI assistant
- Automatic analysis of mining results
- Pattern explanations and insights
- Function calling support for advanced interactions

### 2. **Pattern Mining** ⛏️
- Mine patterns with configurable conjunction counts
- Receive instant results with support values
- Automatic AI-generated summaries for each pattern

### 3. **Exact Pattern Visualization** 👁️
- Click "Visualize" button on any pattern
- Shows ONLY entities that match ALL conditions in the pattern
- Example: For pattern `((tone $x "Analytical") (length $x "low") (engagement_level $x "high"))`
  - Only shows topics that have tone="Analytical" AND length="low" AND engagement_level="high"
  - Not just any topic that matches one of these conditions

## Architecture

```
┌─────────────────────────────────────────────┐
│         Frontend (React/SolidJS)            │
│  - AtomSpace Visualizer (Port 3000)         │
│  - Chat Interface Component                 │
│  - Mining Interface Component               │
└────────────┬────────────────────────────────┘
             │
             ├──────────────┬─────────────────┐
             │              │                 │
             ▼              ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌─────────────┐
    │ Mining API   │ │  Chat API    │ │ Graph API   │
    │ (Port 5000)  │ │ (Port 5001)  │ │             │
    │              │ │              │ │             │
    │ - MeTTa      │ │ - Gemini AI  │ │ - D3.js     │
    │ - Pattern    │ │ - Function   │ │ - Force     │
    │   Miner      │ │   Calling    │ │   Layout    │
    └──────────────┘ └──────────────┘ └─────────────┘
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or pnpm

### Backend Setup

1. **Install Python Dependencies**
```bash
cd /workspaces/Mindplex-Hyperon/experiments
pip install -r requirements.txt
```

2. **Start Mining API Server (Port 5000)**
```bash
python3 mining_api.py
```

3. **Start Chat API Server (Port 5001)**
```bash
python3 chat_api.py
```

### Frontend Setup

1. **Install Node Dependencies**
```bash
cd /workspaces/Mindplex-Hyperon/experiments/atomspace_visualizer
npm install
```

2. **Start Development Server (Port 3000)**
```bash
npm run dev
```

3. **Open Browser**
Navigate to: `http://localhost:3000`

## Usage Guide

### Mining Patterns

1. **Set Conjunction Count**
   - Use the input field to set the number of conjunctions (1-10)
   - Higher numbers find more complex patterns

2. **Click "Mine Neural Gold"** ⛏️💎
   - The mining process starts
   - A fancy animation shows mining in progress
   - Results are automatically sent to the chat interface

3. **View Results in Chat**
   - Chat opens automatically when mining completes
   - Each pattern appears with:
     - AI-generated summary
     - Support value (how many times it appears)
     - "Visualize" button

### Chatting with AI

1. **Ask Questions**
   ```
   - "What patterns have been found?"
   - "Explain the most common pattern"
   - "What does this pattern mean?"
   - "Show me patterns with high engagement"
   ```

2. **Get Insights**
   - AI explains patterns in simple terms
   - Provides context and examples
   - Answers questions about the data

### Visualizing Patterns

1. **Click "Visualize" Button** 👁️
   - Button is on each pattern card in chat
   - Graph updates to show ONLY exact matches

2. **Understanding the Visualization**
   - **Before**: All nodes and edges shown
   - **After**: Only nodes matching ALL pattern conditions
   - Example for `((tone $x "Analytical") (length $x "low"))`:
     - Shows topic nodes (0, 1, 2, etc.)
     - Shows their tone and length property edges
     - ONLY if topic has BOTH tone="Analytical" AND length="low"

3. **Reset View**
   - Simply mine again or refresh the page

## API Endpoints

### Mining API (Port 5000)

#### `POST /api/mine`
Start a mining job
```json
{
  "conjunction_count": 3
}
```

Response:
```json
{
  "jobId": "uuid",
  "status": "finished",
  "conjunction count": 3,
  "message": "Mining job finished successfully",
  "result": [
    {
      "pattern": "((tone $x \"Analytical\") (length $x \"low\"))",
      "support": "3"
    }
  ]
}
```

#### `GET /api/mine/{jobId}`
Get job status (legacy, not used in new flow)

#### `GET /api/health`
Health check

### Chat API (Port 5001)

#### `POST /api/chat/analyze`
Analyze a specific pattern
```json
{
  "pattern": "((tone $x \"Analytical\") (length $x \"low\"))",
  "support": "3"
}
```

Response:
```json
{
  "summary": "📊 **Pattern Analysis**\n\n...",
  "pattern": "...",
  "support": "3"
}
```

#### `POST /api/chat`
Chat with AI assistant
```json
{
  "message": "What patterns have been found?",
  "history": [...],
  "session_id": "optional-session-id"
}
```

Response:
```json
{
  "response": "Here are the patterns I found...",
  "functionCall": null,
  "session_id": "session-id"
}
```

#### `POST /api/chat/clear`
Clear conversation history
```json
{
  "session_id": "session-id"
}
```

#### `GET /api/chat/health`
Health check

## Configuration

### Google Gemini API Key
Located in: `experiments/chat_api.py`
```python
GOOGLE_API_KEY = "AIzaSyChGxk4M-RrG4q7_Oi-sPQgGIRBx8snHcs"
```

### Port Configuration
- Frontend: `3000` (Vite dev server)
- Mining API: `5000` (Flask)
- Chat API: `5001` (Flask)

## Pattern Matching Algorithm

### How Exact Matching Works

1. **Parse Pattern**
   ```typescript
   // Input: "((tone $x \"Analytical\") (length $x \"low\"))"
   // Output: { tone: "Analytical", length: "low" }
   ```

2. **Find Matching Nodes**
   ```typescript
   // For each node (topic):
   //   1. Check if it has ALL required properties
   //   2. Check if each property has the exact required value
   //   3. Only include if ALL conditions match
   ```

3. **Build Filtered Graph**
   ```typescript
   // Include:
   //   - Matching topic nodes
   //   - Their property edges
   //   - Property value nodes
   // Exclude:
   //   - Non-matching topics
   //   - Unrelated properties
   ```

### Example

**Pattern**: `((tone $x "Analytical") (length $x "low") (engagement_level $x "high"))`

**Data**:
```
(topic 0 "AI")
(tone 0 "Analytical")
(length 0 "low")
(engagement_level 0 "high")

(topic 1 "Gardening")
(tone 1 "Analytical")
(length 1 "low")
(engagement_level 1 "high")

(topic 3 "Home Decor")
(tone 3 "Analytical")
(length 3 "low")
(engagement_level 3 "medium")  ← Different!
```

**Result**: Only topics 0 and 1 shown (topic 3 excluded because engagement_level is "medium" not "high")

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port
lsof -ti:5000 | xargs kill -9
lsof -ti:5001 | xargs kill -9
```

### Chat Not Working
1. Check if chat_api.py is running on port 5001
2. Verify Google API key is valid
3. Check browser console for CORS errors

### Mining Fails
1. Check if mining_api.py is running on port 5000
2. Verify MeTTa files are accessible
3. Check Python dependencies are installed

### Visualization Shows Too Many Nodes
- This means the pattern matching is working on OR logic instead of AND
- Check the `handleVisualize` function in `App.tsx`
- Ensure all property conditions are checked together

## File Structure

```
experiments/
├── mining_api.py                 # Mining API server
├── chat_api.py                   # Chat API with Gemini
├── requirements.txt              # Python dependencies
├── data/
│   └── data.metta         # Sample data
└── atomspace_visualizer/
    ├── src/
    │   ├── App.tsx              # Main app with visualization logic
    │   ├── components/
    │   │   ├── ChatInterface/
    │   │   │   ├── ChatInterface.tsx
    │   │   │   └── ChatInterface.css
    │   │   ├── MiningInterface/
    │   │   │   ├── MiningInterface.tsx
    │   │   │   └── MiningInterface.css
    │   │   └── ...
    │   └── ...
    └── package.json
```

## Technologies Used

- **Frontend**: SolidJS, TypeScript, Vite, D3.js
- **Backend**: Python, Flask, Flask-CORS
- **AI**: Google Gemini 1.5 Flash
- **Data**: MeTTa (Hyperon)
- **Pattern Mining**: Custom MeTTa pattern miner

## Future Enhancements

- [ ] Multi-pattern visualization comparison
- [ ] Pattern history and bookmarking
- [ ] Export chat conversations
- [ ] Advanced filtering with AI suggestions
- [ ] Real-time collaborative analysis
- [ ] Pattern recommendation system
- [ ] Custom pattern query builder
- [ ] Integration with more AI models

## License

MIT License (or as per your project's license)

## Contributors

- Your Name
- AI Assistant (GitHub Copilot)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API endpoints documentation
3. Check browser console and server logs
4. Open an issue on GitHub

---

**Happy Pattern Mining! ⛏️💎🤖**
