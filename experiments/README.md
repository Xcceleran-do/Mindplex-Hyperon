# AtomSpace Visualizer with AI Chat & Pattern Mining

**Complete system for visualizing knowledge graphs, mining patterns, and interacting with AI assistant.**

---

## 🚀 Quick Start

```bash
# 1. Start all services (backend + frontend)
cd /workspaces/Mindplex-Hyperon/experiments
./start_all.sh

# 2. Open in browser
# http://localhost:3000

# 3. Mine patterns, chat with AI, visualize exact matches!

# Stop services when done
./stop_all.sh
# AtomSpace Visualizer — Unified Chat + Mining Documentation

This README consolidates the project README and the chat debugging/implementation notes. It explains how the chat+mining features work, how to run and test them, and where to look when things go wrong.

## Quick start

1. Start services (backend + frontend):

```bash
cd experiments
./start_all.sh
```

2. Open the app: http://localhost:3000

3. Use the Mining panel to set conjunct count (2 or 3 recommended) and click "Mine Neural Gold".


## What the Chat + Mining flow does (short)

- When you click Mine, the frontend starts a mining job on the backend (port 5000).
- As patterns return, the UI displays pattern cards with support counts and a Visualize button.
- The Chat automatically opens and a system/user auto-message like "Mine rules with N patterns" is sent; the AI responds with summaries and insights.
- You can continue the conversation and ask the AI about patterns; the AI can return analysis and suggestions.


## Where to find endpoints

- Health: `GET http://localhost:5000/api/health`
- Start mining: `POST http://localhost:5000/api/mine` (body: `{ "conjunction_count": <n> }`)
- Analyze patterns: `POST http://localhost:5000/api/chat/analyze` (body: `{ "result": [{ pattern, support }, ...] }`)
- Chat: `POST http://localhost:5000/api/chat` (body: `{ message, history?, session_id? }`)


## Quick verification & test steps

1. Open DevTools (F12) and watch the Console.
2. Set conjunct count to 2 and click Mine.
3. Expected console logs (Chat):

```
ChatInterface rendered, props: {...}
ChatInterface: conjunctSize changed to: 2
ChatInterface: Opening chat and sending message for conjunct size: 2
```

4. Check the DOM (Elements tab) for `.chat-interface` and verify it has `display: flex` and `z-index: 9999`.


## Troubleshooting checklist (common issues)

1. Chat doesn't appear:
  - Ensure `ChatInterface` is mounted and receives `conjunctSize` from the parent.
  - Verify console logs described above.
  - Manually open with: `document.querySelector('.chat-toggle-btn')?.click()` in the console.

2. Chat opens but no UI visible:
  - Inspect `.chat-interface` styles: `bottom: 0`, `display: flex`, `z-index: 9999`.
  - Ensure mining button isn't covering the chat (z-index/position conflict).

3. Backend endpoints 404 or preflight errors:
  - Confirm backend is running on port 5000 and CORS enabled.
  - Test endpoints with curl:

```bash
curl http://localhost:5000/api/health
curl -X POST http://localhost:5000/api/chat/analyze -H 'Content-Type: application/json' -d '{"result":[]}'
```

4. HMR / dev server issues (frontend):
  - Ensure `vite` is running and `vite.config.ts` HMR host/port match your environment.


## Implementation summary (concise)

- Chat auto-opening: `ChatInterface` watches `conjunctSize`. On change it opens and sends an auto-message like "Mine rules with N patterns".
- Analysis endpoint: `/api/chat/analyze` accepts a list of `{pattern, support}` and returns a JSON summary with stats and insights (average/min/max support, top properties, top property-values).
- Visualization: Visualize filters the graph to show exact matches (AND across all properties in the pattern).


## Testing & debug commands

- Start/stop services:
```bash
./stop_all.sh && ./start_all.sh
```
- Check backend availability:
```bash
curl http://localhost:5000/api/health
```
- Force-open chat (dev):
```js
document.querySelector('.chat-toggle-btn')?.click()
```
- Inspect chat computed styles in console:
```js
const chat = document.querySelector('.chat-interface');
console.log(getComputedStyle(chat).display, getComputedStyle(chat).zIndex, getComputedStyle(chat).bottom);
```


## Status & notes

- Chat, mining, and analysis are implemented and integrated. If you run into errors, follow the troubleshooting checklist above and capture console logs / Elements screenshots.
- If 127.0.0.1 doesn't reach your backend (WSL), use `http://wsl.localhost:5000` or the WSL IP (e.g., `http://172.17.x.x:5000`) or enable WSL localhost forwarding by creating `C:\Users\<you>\.wslconfig` with `localhostForwarding=true` and running `wsl --shutdown`.


## Contact / Next steps

- If chat still fails to auto-open, paste the browser console logs and a screenshot of the Elements tree showing `.chat-interface`.
- Want me to rework the chat trigger or unify endpoints further? Tell me which behavior to change and I will update the code.

- All conditions must use the **same variable** for exact matching
- Property names must **match exactly** with data file
- Values are **case-sensitive** strings

### Performance Tips

- Reduce visualization load by filtering to relevant data first
- Use smaller datasets for testing and development
- Clear browser cache if UI behaves unexpectedly
- Restart services if memory usage grows too high

---

## 📞 Support & Contact

### Getting Help

1. **Check this README** - Most questions are answered here
2. **Review logs** - Check backend.log and frontend.log
3. **Run tests** - `python3 test_unified_backend.py`
4. **Restart services** - `./stop_all.sh && ./start_all.sh`
5. **Open GitHub issue** - For bugs or feature requests

### Reporting Issues

When reporting issues, please include:
- Error messages from logs
- Steps to reproduce
- Expected vs actual behavior
- Browser and OS information
- Screenshots if relevant

---

## 🎉 Summary

You now have a **complete, working system** that:

✅ **Mines patterns** from MeTTa knowledge graphs  
✅ **Analyzes patterns** with Google Gemini AI  
✅ **Visualizes exact matches** (ALL conditions must match)  
✅ **Provides chat interface** for AI interaction  
✅ **Includes all tools** and comprehensive documentation  

**Everything is ready to use!** 🚀

---

## 🏁 Getting Started Again

Forgot the commands? Here they are:

```bash
# Navigate to experiments folder
cd /workspaces/Mindplex-Hyperon/experiments

# Start everything
./start_all.sh

# Open browser
# http://localhost:3000

# Mine patterns, chat with AI, visualize exact matches!

# Stop when done
./stop_all.sh
```

---

**Happy Mining & Visualizing!** ⛏️💎🤖

*Last Updated: October 7, 2025*
