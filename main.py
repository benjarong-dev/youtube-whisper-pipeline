"""CLI entrypoint for YouTube Whisper ASR ingestion and SQLite FTS5 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import List, Optional

from src.extractor import download_youtube_audio, get_sample_demo_payload
from src.storage import export_artifacts, search_segments
from src.transcriber import normalize_segments, transcribe_audio_with_whisper


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YouTube audio, transcribe via Whisper, and index into SQLite FTS5."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="YouTube video URL to download and transcribe.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="OpenAI Whisper model size (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code override (e.g. en, th).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.20,
        help="Minimum segment confidence threshold [0.0-1.0] (default: 0.20).",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./outputs",
        help="Target directory for JSON, SRT, and SQLite database outputs.",
    )
    parser.add_argument(
        "--sample-demo",
        action="store_true",
        help="Run deterministic offline ETL pipeline without network or GPU dependencies.",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Full-text search query against outputs/transcripts.db (returns exact timestamps).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Retain temporary 16 kHz WAV file after Whisper transcription completes.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.search is not None:
        db_path = output_dir / "transcripts.db"
        matches = search_segments(db_path=db_path, query=args.search)
        sys.stdout.write(
            json.dumps(
                {"query": args.search, "match_count": len(matches), "matches": matches},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    if args.sample_demo:
        meta, raw_segments = get_sample_demo_payload()
        result = normalize_segments(
            metadata=meta,
            raw_segments=raw_segments,
            model_name="sample-fixture",
            language="en",
            min_confidence=args.min_confidence,
        )
        paths = export_artifacts(output_dir=output_dir, result=result)
        summary = {
            "status": "ok",
            "mode": "sample-demo",
            "video_id": result.metadata.video_id,
            "segments_indexed": len(result.segments),
            "total_words": result.total_words,
            "words_per_minute": result.words_per_minute,
            "avg_confidence": result.avg_confidence,
            "json_output": str(paths["json_path"]),
            "srt_output": str(paths["srt_path"]),
            "sqlite_db": str(paths["db_path"]),
        }
        sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return 0

    if args.url:
        work_dir = output_dir / "tmp_audio"
        try:
            meta = download_youtube_audio(url=args.url, work_dir=work_dir)
            result = transcribe_audio_with_whisper(
                metadata=meta,
                model_name=args.model,
                language=args.language,
                min_confidence=args.min_confidence,
            )
            paths = export_artifacts(output_dir=output_dir, result=result)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            sys.stderr.write(f"Pipeline error: {exc}\n")
            return 1
        finally:
            if not args.keep_audio and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

        summary = {
            "status": "ok",
            "mode": "youtube-whisper",
            "video_id": result.metadata.video_id,
            "segments_indexed": len(result.segments),
            "total_words": result.total_words,
            "words_per_minute": result.words_per_minute,
            "avg_confidence": result.avg_confidence,
            "json_output": str(paths["json_path"]),
            "srt_output": str(paths["srt_path"]),
            "sqlite_db": str(paths["db_path"]),
        }
        sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        return 0

    sys.stderr.write("Error: Provide --url <YOUTUBE_URL>, --sample-demo, or --search <QUERY>.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
