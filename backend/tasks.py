"""
tasks.py — Structured, persistent task store (SQLite-backed).

Swapped from JSON to SQLite for the web backend: multiple requests can
land concurrently once this is behind FastAPI, and a JSON read-modify-write
isn't safe for that. Semantics are identical to the CLI version.
"""
import sqlite3
import time
import uuid
from contextlib import contextmanager

DB_FILE = "tasks.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                priority TEXT NOT NULL,
                due_date TEXT,
                notes TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


_init_db()


def add_task(title: str, priority: str = "medium", due_date: str = "", notes: str = "") -> dict:
    task = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "priority": priority,
        "due_date": due_date,
        "notes": notes,
        "status": "pending",
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, priority, due_date, notes, status, created_at) "
            "VALUES (:id, :title, :priority, :due_date, :notes, :status, :created_at)",
            task,
        )
    return task


def list_tasks(status_filter: str = "all") -> list[dict]:
    with _connect() as conn:
        if status_filter == "all":
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC",
                (status_filter,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_status(task_id: str, status: str) -> dict | None:
    with _connect() as conn:
        cur = conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row)


def delete_task(task_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    return cur.rowcount > 0
