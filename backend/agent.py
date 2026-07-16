"""
agent.py — The task-planning agent.

Built as a LangGraph ReAct agent: it reasons about the user's message,
decides which tool(s) to call (task CRUD, memory search, memory save),
and responds. Short-term conversational memory comes from LangGraph's
checkpointer; long-term memory across sessions comes from our own
Chroma-backed store in memory.py.
"""
import os

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from memory import LongTermMemory
from tasks import add_task, delete_task, list_tasks, update_status

long_term = LongTermMemory()

SYSTEM_PROMPT = """You are TaskMind, a personal task-planning \
assistant with long-term memory.

Rules:
- Whenever the user states a goal, deadline, or a preference (e.g. "I work \
best in the mornings", "always break big tasks into sub-tasks under 2 \
hours"), call save_memory to remember it for future sessions.
- Before planning new tasks for the user, call search_memory to check for \
relevant past goals or preferences, and use them to personalize your plan.
- Break vague goals ("finish my thesis") into 3-6 concrete tasks using \
create_task, with sensible priorities and due dates if the user gives a \
deadline.
- Use show_tasks to check current state before claiming something is done \
or before re-adding a task.
- Be concise. Confirm actions in one or two lines; don't repeat the whole \
task list unless asked.
"""


@tool
def create_task(title: str, priority: str = "medium", due_date: str = "", notes: str = "") -> str:
    """Add a new task. priority is 'low', 'medium', or 'high'.
    due_date is a free-text date like '2026-07-20' (optional)."""
    task = add_task(title, priority, due_date, notes)
    return f"Added task {task['id']}: {task['title']} (priority={priority})"


@tool
def show_tasks(status_filter: str = "all") -> str:
    """List tasks. status_filter is 'all', 'pending', 'in-progress', or 'done'."""
    tasks = list_tasks(status_filter)
    if not tasks:
        return "No tasks found."
    lines = [f"[{t['id']}] ({t['status']}, {t['priority']}) {t['title']}" for t in tasks]
    return "\n".join(lines)


@tool
def mark_task(task_id: str, status: str) -> str:
    """Update a task's status. status is 'pending', 'in-progress', or 'done'."""
    task = update_status(task_id, status)
    if not task:
        return f"No task with id {task_id}."
    return f"Task {task_id} marked as {status}."


@tool
def remove_task(task_id: str) -> str:
    """Delete a task by id."""
    ok = delete_task(task_id)
    return "Deleted." if ok else f"No task with id {task_id}."


@tool
def save_memory(text: str, kind: str = "note") -> str:
    """Save a fact, goal, or preference to long-term memory.
    kind is 'goal', 'preference', or 'note'."""
    long_term.remember(text, kind=kind)
    return "Saved to memory."


@tool
def search_memory(query: str) -> str:
    """Search long-term memory for relevant past goals, preferences, or notes."""
    results = long_term.recall(query, k=4)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"- {r['text']} ({r['metadata'].get('kind', 'note')})" for r in results)


TOOLS = [create_task, show_tasks, mark_task, remove_task, save_memory, search_memory]


def build_llm():
    """Provider is chosen via LLM_PROVIDER env var: 'gemini' (default) or 'openai'."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0.3
    )


def build_agent():
    llm = build_llm()
    checkpointer = MemorySaver()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
