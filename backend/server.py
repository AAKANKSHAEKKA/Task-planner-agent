"""
server.py — FastAPI backend wrapping the TaskMind agent.

Endpoints:
  POST /api/chat        — send a message, get the agent's reply + fresh
                           task list + which memory tools it used
  GET  /api/tasks        — current task list
  PATCH /api/tasks/{id}  — update a task's status directly (no LLM round trip)
  GET  /api/new_thread   — get a fresh conversation thread id
  GET  /api/health       — liveness check
"""
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agent import build_agent
from tasks import list_tasks, update_status

load_dotenv()

app = FastAPI(title="TaskMind API")

# Demo-scope CORS. Lock this down to your actual frontend origin before
# deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = build_agent()

TRACKED_TOOLS = {"save_memory", "search_memory", "create_task"}


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    tasks: list[dict]
    memory_events: list[dict]


class StatusUpdate(BaseModel):
    status: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=req.message)]}, config=config
    )
    messages = result["messages"]

    memory_events = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                if call["name"] not in TRACKED_TOOLS:
                    continue
                output = next(
                    (
                        m.content
                        for m in messages[i + 1 :]
                        if isinstance(m, ToolMessage) and m.tool_call_id == call["id"]
                    ),
                    "",
                )
                memory_events.append(
                    {"type": call["name"], "input": call["args"], "output": output}
                )

    return ChatResponse(
        response=messages[-1].content,
        tasks=list_tasks(),
        memory_events=memory_events,
    )


@app.get("/api/tasks")
def get_tasks():
    return list_tasks()


@app.patch("/api/tasks/{task_id}")
def patch_task(task_id: str, body: StatusUpdate):
    task = update_status(task_id, body.status)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/new_thread")
def new_thread():
    return {"thread_id": str(uuid.uuid4())}


@app.get("/api/health")
def health():
    return {"status": "ok"}
