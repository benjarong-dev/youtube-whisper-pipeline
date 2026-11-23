"""Unit tests for URL validation, SRT rendering, SQLite upsert, and FTS5 search."""

from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from main import main as cli_main
from src.extractor import VideoMetadata, extract_youtube_video_id, get_sample_demo_payload
from src.storage import export_artifacts, search_segments
from src.transcriber import format_srt_timestamp, normalize_segments, render_srt


class WhisperPipelineTestSuite(unittest.TestCase):
    def test_extract_youtube_video_id_accepts_valid_formats_and_blocks_spoofed_hosts(self) -> None:
        self.assertEqual(
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ?t=15"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

        with self.assertRaises(ValueError):
            extract_youtube_video_id("https://evil-youtube.com/watch?v=dQw4w9WgXcQ")

        with self.assertRaises(ValueError):
            extract_youtube_video_id("https://www.youtube.com/watch?v=too_short")

    def test_srt_timestamp_and_caption_rendering(self) -> None:
        self.assertEqual(format_srt_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(3661.25), "01:01:01,250")

        meta, raw_segments = get_sample_demo_payload()
        result = normalize_segments(meta, raw_segments[:2])
        srt_text = render_srt(result.segments)

        self.assertIn("00:00:00,000 --> 00:00:11,200", srt_text)
        self.assertIn("802.1Q VLAN", srt_text)

    def test_normalize_segments_filters_low_confidence_and_computes_wpm(self) -> None:
        meta = VideoMetadata(
            video_id="TESTVID0001",
            title="Test Audio",
            source_url="https://www.youtube.com/watch?v=TESTVID0001",
            duration_sec=30.0,
        )
        raw = [
            {"start": 0.0, "end": 15.0, "text": "clean signal one two three", "avg_logprob": -0.10},
            {"start": 15.0, "end": 30.0, "text": "garbled noise artifact", "avg_logprob": -2.50},
        ]
        result = normalize_segments(meta, raw, min_confidence=0.30)

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.total_words, 5)
        self.assertEqual(result.words_per_minute, 10.0)

    def test_idempotent_sqlite_upsert_and_fts5_search_handles_special_tokens(self) -> None:
        meta, raw_segments = get_sample_demo_payload()
        result = normalize_segments(meta, raw_segments)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            export_artifacts(out_path, result)
            paths = export_artifacts(out_path, result)

            with closing(sqlite3.connect(paths["db_path"])) as conn:
                video_count = conn.execute("SELECT COUNT(*) FROM videos;").fetchone()[0]
                seg_count = conn.execute("SELECT COUNT(*) FROM transcript_segments;").fetchone()[0]

            self.assertEqual(video_count, 1)
            self.assertEqual(seg_count, 4)

            hits = search_segments(paths["db_path"], "802.1Q")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["timestamp_srt"], "00:00:00,000")

    def test_cli_sample_demo_and_search_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ingest_buf = io.StringIO()
            with redirect_stdout(ingest_buf):
                code_ingest = cli_main(["--sample-demo", "--output-dir", tmp_dir])
            self.assertEqual(code_ingest, 0)

            search_buf = io.StringIO()
            with redirect_stdout(search_buf):
                code_search = cli_main(["--output-dir", tmp_dir, "--search", "PostgreSQL"])
            self.assertEqual(code_search, 0)

            payload = json.loads(search_buf.getvalue())
            self.assertEqual(payload["match_count"], 1)
            self.assertEqual(payload["matches"][0]["timestamp_srt"], "00:00:34,000")


if __name__ == "__main__":
    unittest.main()
