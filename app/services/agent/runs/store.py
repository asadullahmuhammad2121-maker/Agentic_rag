"""SQLite persistence for agent run history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger(__name__)

AgentRunStatus = Literal["success", "failure"]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'failure')),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    duration_ms INTEGER,
    tool_used TEXT,
    step_count INTEGER NOT NULL DEFAULT 0,
    citation_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    error_code TEXT,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started_at
    ON agent_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status
    ON agent_runs(status);
"""


@dataclass(frozen=True, slots=True)
class AgentRunSummary:
    """Summary row for agent run listing."""

    run_id: str
    query: str
    status: AgentRunStatus
    started_at: datetime
    completed_at: datetime
    duration_ms: int | None
    tool_used: str | None
    step_count: int
    citation_count: int
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunDetail(AgentRunSummary):
    """Full stored agent run payload."""

    answer: str | None = None
    citations: list[dict[str, Any]] | None = None
    steps: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentRunListPage:
    """Paginated agent run list."""

    runs: list[AgentRunSummary]
    total: int
    limit: int
    offset: int


class AgentRunStore:
    """Persist and query agent run history in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialize()

    def create_run_id(self) -> str:
        return str(uuid4())

    def save_success(
        self,
        *,
        run_id: str,
        query: str,
        started_at: datetime,
        completed_at: datetime,
        duration_ms: int,
        response_payload: dict[str, Any],
    ) -> None:
        tool_used = response_payload.get("tool_used")
        steps = response_payload.get("steps", [])
        metadata = response_payload.get("metadata", {})
        citations = response_payload.get("citations", [])
        step_count = metadata.get("step_count", len(steps) if isinstance(steps, list) else 0)
        citation_count = metadata.get("citation_count", len(citations) if isinstance(citations, list) else 0)
        self._insert(
            run_id=run_id,
            query=query,
            status="success",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            tool_used=str(tool_used) if tool_used else None,
            step_count=int(step_count),
            citation_count=int(citation_count),
            error_message=None,
            error_code=None,
            payload_json=json.dumps(response_payload),
        )

    def save_failure(
        self,
        *,
        run_id: str,
        query: str,
        started_at: datetime,
        completed_at: datetime,
        duration_ms: int,
        error_message: str,
        error_code: str,
    ) -> None:
        payload: dict[str, Any] = {
            "answer": None,
            "citations": [],
            "tool_used": None,
            "steps": [],
            "metadata": {
                "error_code": error_code,
            },
        }
        self._insert(
            run_id=run_id,
            query=query,
            status="failure",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            tool_used=None,
            step_count=0,
            citation_count=0,
            error_message=error_message,
            error_code=error_code,
            payload_json=json.dumps(payload),
        )

    def list_runs(
        self,
        *,
        search: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> AgentRunListPage:
        clauses: list[str] = []
        params: list[Any] = []

        if search:
            clauses.append("query LIKE ?")
            params.append(f"%{search.strip()}%")
        if status:
            clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        count_sql = f"SELECT COUNT(*) FROM agent_runs {where_sql}"
        list_sql = f"""
            SELECT run_id, query, status, started_at, completed_at, duration_ms,
                   tool_used, step_count, citation_count, error_message, error_code
            FROM agent_runs
            {where_sql}
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """

        with self._connect() as connection:
            total = connection.execute(count_sql, params).fetchone()[0]
            rows = connection.execute(list_sql, [*params, limit, offset]).fetchall()

        runs = [_row_to_summary(row) for row in rows]
        return AgentRunListPage(runs=runs, total=int(total), limit=limit, offset=offset)

    def get_run(self, run_id: str) -> AgentRunDetail | None:
        sql = """
            SELECT run_id, query, status, started_at, completed_at, duration_ms,
                   tool_used, step_count, citation_count, error_message, error_code,
                   payload_json
            FROM agent_runs
            WHERE run_id = ?
        """
        with self._connect() as connection:
            row = connection.execute(sql, (run_id,)).fetchone()
        if row is None:
            return None
        summary = _row_to_summary(row[:11])
        payload = json.loads(row[11])
        return AgentRunDetail(
            run_id=summary.run_id,
            query=summary.query,
            status=summary.status,
            started_at=summary.started_at,
            completed_at=summary.completed_at,
            duration_ms=summary.duration_ms,
            tool_used=summary.tool_used,
            step_count=summary.step_count,
            citation_count=summary.citation_count,
            error_message=summary.error_message,
            error_code=summary.error_code,
            answer=payload.get("answer"),
            citations=list(payload.get("citations", [])),
            steps=list(payload.get("steps", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    def _initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _insert(
        self,
        *,
        run_id: str,
        query: str,
        status: AgentRunStatus,
        started_at: datetime,
        completed_at: datetime,
        duration_ms: int,
        tool_used: str | None,
        step_count: int,
        citation_count: int,
        error_message: str | None,
        error_code: str | None,
        payload_json: str,
    ) -> None:
        sql = """
            INSERT INTO agent_runs (
                run_id, query, status, started_at, completed_at, duration_ms,
                tool_used, step_count, citation_count, error_message, error_code,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as connection:
            connection.execute(
                sql,
                (
                    run_id,
                    query,
                    status,
                    _format_dt(started_at),
                    _format_dt(completed_at),
                    duration_ms,
                    tool_used,
                    step_count,
                    citation_count,
                    error_message,
                    error_code,
                    payload_json,
                ),
            )
            connection.commit()
        logger.info(
            "agent_run_persisted",
            extra={
                "operation": "save_agent_run",
                "run_id": run_id,
                "status": status,
                "duration_ms": duration_ms,
            },
        )


def _format_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _row_to_summary(row: sqlite3.Row | tuple[Any, ...]) -> AgentRunSummary:
    return AgentRunSummary(
        run_id=str(row[0]),
        query=str(row[1]),
        status=str(row[2]),  # type: ignore[arg-type]
        started_at=_parse_dt(str(row[3])),
        completed_at=_parse_dt(str(row[4])),
        duration_ms=int(row[5]) if row[5] is not None else None,
        tool_used=str(row[6]) if row[6] is not None else None,
        step_count=int(row[7]),
        citation_count=int(row[8]),
        error_message=str(row[9]) if row[9] is not None else None,
        error_code=str(row[10]) if row[10] is not None else None,
    )
