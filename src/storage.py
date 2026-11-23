"""SQLite relational persistence, FTS5 full-text indexing, and JSON/SRT writers."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Dict, List

from .transcriber import TranscriptionResult, format_srt_timestamp, render_srt

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    model_name TEXT NOT NULL,
    language TEXT NOT NULL,
    total_words INTEGER NOT NULL,
    words_per_minute REAL NOT NULL,
    avg_confidence REAL NOT NULL,
    ingested_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    video_id TEXT NOT NULL,
    segment_idx INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    text TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    PRIMARY KEY (video_id, segment_idx),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    video_id UNINDEXED,
    segment_idx UNINDEXED,
    text
);
"""


def _build_safe_fts_query(raw_query: str) -> str:
    """Quotes search tokens safely so dots/hyphens (e.g. 802.1Q) do not break FTS5."""
    tokens = [tok.replace('"', "").strip() for tok in raw_query.split()]
    valid = [f'"{tok}"' for tok in tokens if tok]
    return " ".join(valid)


def init_database(db_path: Path) -> None:
    """Initializes SQLite tables and FTS5 virtual index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.executescript(SCHEMA_SQL)


def upsert_transcription(db_path: Path, result: TranscriptionResult) -> None:
    """Idempotently writes video metadata, segments, and FTS5 index entries."""
    init_database(db_path)
    ingested_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    vid = result.metadata.video_id

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        with conn:
            conn.execute("DELETE FROM segments_fts WHERE video_id = ?;", (vid,))
            conn.execute("DELETE FROM transcript_segments WHERE video_id = ?;", (vid,))

            conn.execute(
                """
                INSERT INTO videos (
                    video_id, title, source_url, duration_sec,
                    model_name, language, total_words,
                    words_per_minute, avg_confidence, ingested_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title = excluded.title,
                    source_url = excluded.source_url,
                    duration_sec = excluded.duration_sec,
                    model_name = excluded.model_name,
                    language = excluded.language,
                    total_words = excluded.total_words,
                    words_per_minute = excluded.words_per_minute,
                    avg_confidence = excluded.avg_confidence,
                    ingested_at_utc = excluded.ingested_at_utc;
                """,
                (
                    vid,
                    result.metadata.title,
                    result.metadata.source_url,
                    result.metadata.duration_sec,
                    result.model_name,
                    result.language,
                    result.total_words,
                    result.words_per_minute,
                    result.avg_confidence,
                    ingested_ts,
                ),
            )

            segment_rows = [
                (
                    vid,
                    seg.segment_idx,
                    seg.start_sec,
                    seg.end_sec,
                    seg.text,
                    seg.confidence_score,
                )
                for seg in result.segments
            ]
            conn.executemany(
                """
                INSERT INTO transcript_segments (
                    video_id, segment_idx, start_sec, end_sec, text, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                segment_rows,
            )

            fts_rows = [(vid, seg.segment_idx, seg.text) for seg in result.segments]
            conn.executemany(
                "INSERT INTO segments_fts (video_id, segment_idx, text) VALUES (?, ?, ?);",
                fts_rows,
            )


def export_artifacts(output_dir: Path, result: TranscriptionResult) -> Dict[str, Path]:
    """Persists JSON, SRT captions, and SQLite FTS5 records into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = result.metadata.video_id

    json_path = output_dir / f"{vid}.json"
    srt_path = output_dir / f"{vid}.srt"
    db_path = output_dir / "transcripts.db"

    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    srt_path.write_text(render_srt(result.segments), encoding="utf-8")
    upsert_transcription(db_path, result)

    return {
        "json_path": json_path,
        "srt_path": srt_path,
        "db_path": db_path,
    }


def search_segments(
    db_path: Path,
    query: str,
    limit: int = 10,
) -> List[Dict[str, object]]:
    """Searches indexed transcripts via SQLite FTS5 and returns exact timestamps."""
    if not db_path.is_file():
        return []

    safe_query = _build_safe_fts_query(query)
    if not safe_query:
        return []

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT
                s.video_id,
                v.title,
                s.segment_idx,
                s.start_sec,
                s.end_sec,
                s.text,
                s.confidence_score
            FROM segments_fts f
            JOIN transcript_segments s
              ON s.video_id = f.video_id AND s.segment_idx = f.segment_idx
            JOIN videos v
              ON v.video_id = s.video_id
            WHERE segments_fts MATCH ?
            ORDER BY rank
            LIMIT ?;
            """,
            (safe_query, max(1, int(limit))),
        ).fetchall()

    return [
        {
            "video_id": row[0],
            "title": row[1],
            "segment_idx": row[2],
            "start_sec": row[3],
            "end_sec": row[4],
            "timestamp_srt": format_srt_timestamp(float(row[3])),
            "text": row[5],
            "confidence_score": row[6],
        }
        for row in rows
    ]
