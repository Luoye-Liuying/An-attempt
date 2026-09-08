"""
数据库层 — SQLite + aiosqlite
表: tasks (任务级) + questions (题目级)
"""
import aiosqlite
from pathlib import Path
from loguru import logger

DB_PATH = Path(__file__).resolve().parent / "grading_history.db"

INIT_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    filename TEXT,
    mode TEXT,
    total_score REAL,
    max_total_score REAL,
    accuracy REAL,
    summary TEXT,
    question_count INTEGER,
    elapsed_seconds REAL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    question_index INTEGER,
    question_text TEXT,
    student_answer TEXT,
    reference_answer TEXT,
    is_correct INTEGER,
    score REAL,
    max_score REAL,
    analysis TEXT,
    suggestion TEXT,
    question_image TEXT
);

CREATE INDEX IF NOT EXISTS idx_questions_task ON questions(task_id);
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
"""


async def init_db():
    """初始化数据库表。"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(INIT_SQL)
        await db.commit()
    logger.info(f"数据库就绪: {DB_PATH}")


async def save_task(task_id: str, filename: str, mode: str,
                    total_score: float, max_total_score: float,
                    accuracy: float, summary: str, question_count: int,
                    elapsed_seconds: float):
    """保存批改任务。"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT OR REPLACE INTO tasks(id,filename,mode,total_score,max_total_score,accuracy,summary,question_count,elapsed_seconds) VALUES(?,?,?,?,?,?,?,?,?)",
            (task_id, filename, mode, total_score, max_total_score, accuracy, summary, question_count, elapsed_seconds))
        await db.commit()


async def save_question(task_id: str, q: dict):
    """保存单题批改结果。"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT INTO questions(task_id,question_index,question_text,student_answer,reference_answer,is_correct,score,max_score,analysis,suggestion,question_image) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, q.get("index"), q.get("question_text", ""), q.get("student_answer", ""),
                q.get("reference_answer", ""), int(q.get("is_correct", False)),
                q.get("score", 0), q.get("max_score", 0),
                q.get("analysis", ""), q.get("suggestion", ""),
                q.get("question_image", "")))
        await db.commit()


async def get_task(task_id: str) -> dict | None:
    """查询单个任务及其题目。"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        row = await db.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        task = await row.fetchone()
        if not task:
            return None
        t = dict(task)
        qs = await db.execute(
            "SELECT * FROM questions WHERE task_id=? ORDER BY question_index", (task_id,))
        t["questions"] = [dict(q) for q in await qs.fetchall()]
        return t


async def list_tasks(page: int = 1, size: int = 20) -> list[dict]:
    """分页查询历史列表（仅任务，不含题目详情）。"""
    offset = (page - 1) * size
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        rows = await db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?", (size, offset))
        return [dict(r) for r in await rows.fetchall()]


async def delete_task(task_id: str) -> bool:
    """删除历史记录（CASCADE 自动删关联题目）。"""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cur = await db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        await db.commit()
        return cur.rowcount > 0
