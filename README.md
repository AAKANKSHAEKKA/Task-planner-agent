# TaskMind — Task-Planning Agent with Memory

A full-stack agentic app: describe a goal in chat, the agent breaks it into
tasks and tracks them, and it **remembers your goals, deadlines, and
preferences across sessions** using real long-term memory — not just chat
history.

## Architecture

```
 frontend/index.html          backend/server.py            backend/agent.py
 (vanilla JS, fetch) ───────▶ FastAPI (CORS, REST)  ─────▶ LangGraph ReAct agent
                                     │                         (Gemini / GPT-4o-mini)
                                     │                              │
                        ┌────────────┴────────────┐    decides which tool to call
                        ▼                          ▼
              backend/tasks.py            backend/memory.py
              Structured task store        Long-term semantic memory
              (SQLite: tasks.db)           (ChromaDB + local
              add/list/update/delete        sentence-transformer embeddings)
```

**Three kinds of memory, on purpose:**
- **Structured memory** (`tasks.py`, SQLite) — exact CRUD. "Mark task a3f9
  done" needs an exact match, not a similarity search.
- **Semantic memory** (`memory.py`, ChromaDB) — fuzzy recall. "What did I
  say about my sleep schedule?" needs embedding similarity.
- **Short-term memory** — LangGraph's `MemorySaver` checkpointer, keyed by
  `thread_id`, keeps one conversation coherent turn-to-turn.

The agent decides for itself, per message, which tools to call — that's
what makes it agentic rather than a fixed script: one message like *"help
me plan my thesis, due Sept 1, I focus best in the mornings"* can trigger
`search_memory`, several `create_task` calls, and `save_memory`, all in a
single turn.

The frontend's **Memory Trail** panel makes that visible — every time the
agent remembers or recalls something, it appears as a pinned note in real
time, instead of being invisible inside the chat response.

## Running it

**1. Backend**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GOOGLE_API_KEY (free tier: https://aistudio.google.com/apikey)
uvicorn server:app --reload --port 8000
```

**2. Frontend** — just open the file, no build step:
```bash
cd frontend
open index.html          # macOS
# or: python3 -m http.server 5500   then visit localhost:5500
```

Backend must be running first. If you deploy the backend somewhere other
than `localhost:8000`, update `API_BASE` at the top of `frontend/index.html`'s
`<script>` block.

## Try this to see the memory actually work

1. Send: *"I'm doing a final-year ML project, due in 6 weeks. I focus best
   in short 2-hour blocks."*
   Watch tasks appear in the right panel, and a plum "remembered" chip
   land in the Memory Trail.
2. Refresh the page (or open the file in a new tab — a fresh browser
   session gets a new `thread_id`, so there's no chat history carried
   over).
3. Send: *"What should I focus on for my project, and how should I
   schedule it?"*
   The agent calls `search_memory`, pulls back your deadline and work
   style from ChromaDB — which persisted on disk, independent of the
   conversation — and answers using it.

That's the actual point of this project: memory that survives a reload,
not just a longer context window.

## Notes on scope

- SQLite and ChromaDB both write to local files next to `server.py`
  (`tasks.db`, `chroma_memory/`) — fine for a single-user demo, not for
  multi-tenant production. For that, add auth and scope both stores by
  user id.
- CORS is wide open (`allow_origins=["*"]`) for local demo convenience —
  lock this to your actual frontend origin before deploying anywhere public.
- The Memory Trail panel only shows events from the current browser
  session (it's not fetched from history on load) — the underlying
  memories persist regardless, this is just a UI choice to keep the trail
  simple.

## Extending this

- Add a `reflect_on_completion` tool that fires when a task is marked
  done, prompting the LLM to write a short retrospective into long-term
  memory — closes the loop into a self-improving planner.
- Add a `web_search` tool (Tavily/SerpAPI) so the agent can research, not
  just track.
- Swap SQLite for Postgres + add auth for real multi-user support.
- Deploy the backend (Render/Railway/Fly.io) and the frontend (Vercel/
  Netlify/GitHub Pages) separately — it's already split cleanly for that.
