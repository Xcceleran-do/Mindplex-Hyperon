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

## MIND Benchmark Quick Run

Use this when you need an external, public dataset result you can share quickly.

1. Download and extract MIND (recommend **MINDsmall**) so the folder contains either:
  - `train/news.tsv`, `train/behaviors.tsv`, `valid/news.tsv`, `valid/behaviors.tsv`, or
  - `news.tsv` and `behaviors.tsv` directly.

2. Run the one-shot adapter (from repo root):

```bash
python experiments/ingestion/run_mind_benchmark.py --mind-dir <PATH_TO_MIND_FOLDER>
```

3. Outputs generated automatically:
  - `experiments/atomspace_visualizer/public/data.metta` (ready for existing miner/visualizer)
  - `experiments/reports/mind_preliminary_results.md` (shareable summary)
  - `experiments/reports/mind_preliminary_results.json` (raw stats)

4. Optional test:

```bash
python -m unittest experiments.ingestion.tests.mind_adapter_test
```

Notes:
- Engagement labels are derived from CTR buckets.
- Popularity is marked `Top_10` using the top 10% impression threshold.
- Tone/sentiment/expertise are heuristic, for preliminary benchmarking.

### If direct URL download fails (zip fallback)

If your environment cannot fetch MIND with direct links, download `MINDsmall_train.zip`
and `MINDsmall_dev.zip` manually from https://msnews.github.io/ (Download section), then run:

```bash
python3 experiments/ingestion/setup_mind_from_zips.py \
  --train-zip /path/to/MINDsmall_train.zip \
  --dev-zip /path/to/MINDsmall_dev.zip \
  --mind-root datasets/MIND \
  --min-articles 10000
```

This extracts train/valid and immediately generates:
- `experiments/atomspace_visualizer/public/data.metta`
- `experiments/reports/mind_preliminary_results.md`
- `experiments/reports/mind_preliminary_results.json`

This README consolidates the project README and the chat debugging/implementation notes. It explains how the chat+mining features work, how to run and test them, and where to look when things go wrong.

## Quick start

1. Start services (backend + frontend):

```bash
cd experiments
./start_all.sh
```

2. Open the app: http://localhost:3000

3. Use the Mining panel to set conjunct count (2 or 3 recommended) and click "Mine Neural Gold".

## Route Chat Through OmegaClaw

For trials where the UI chat should be handled by OmegaClaw, run the Mindplex API and OmegaClaw with the same queue directory:

```bash
export OMEGACLAW_MINDPLEX_QUEUE_DIR=/tmp/omegaclaw-mindplex
export MINDPLEX_CHAT_BACKEND=omegaclaw

# terminal 1: Mindplex backend/frontend as usual
cd experiments
./start_all.sh

# terminal 2: OmegaClaw reading Mindplex chat
cd PeTTa
metta run.metta commchannel=mindplex MP_QUEUE_DIR=/tmp/omegaclaw-mindplex MP_RESPONSE_TIMEOUT=110 maxNewInputLoops=1
```

Then use the existing UI chat. Messages are delivered to OmegaClaw, and OmegaClaw's `send` response is returned to the chat panel.

Typed chat requests are not intercepted by the frontend mining shortcut in this mode. For example, asking the chat to mine with a conjunction count and minimum support goes through `/api/chat` to OmegaClaw, then OmegaClaw calls `mindplex-mine`. Pattern summary and single-pattern analysis requests from the chat UI are also routed to OmegaClaw. The direct Mine button still uses the local `/api/mine` workflow.

By default, the bridge sends only the current chat message to OmegaClaw. Mindplex UI history stays out of the queue because OmegaClaw keeps its own history. To forward UI history for debugging, set `OMEGACLAW_MINDPLEX_FORWARD_HISTORY=1`.

When using `docker compose`, the `mining-api` service enables this redirect and mounts the project-local `.omegaclaw-mindplex` directory into the container at `/tmp/omegaclaw-mindplex`. Start OmegaClaw with the host-side path:

```bash
cd PeTTa
export OMEGACLAW_MINDPLEX_QUEUE_DIR="$(cd .. && pwd)/.omegaclaw-mindplex"
export MINDPLEX_API_BASE_URL=http://127.0.0.1:5000
metta run.metta commchannel=mindplex MP_QUEUE_DIR="$OMEGACLAW_MINDPLEX_QUEUE_DIR" MP_RESPONSE_TIMEOUT=110 maxNewInputLoops=1
```

OmegaClaw also has a trial miner skill:

```metta
(mindplex-mine "2" "3")
```

It calls the running Mindplex `/api/mine` pipeline, so data ingestion remains the same UI flow. When this skill is invoked from the Mindplex chat channel, the mining completion summary is sent back to the chat response automatically.


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
