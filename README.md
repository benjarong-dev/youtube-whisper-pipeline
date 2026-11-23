# YouTube Speech-to-Text & SQLite FTS5 Data Pipeline

Modular Python ETL pipeline that extracts mono `16 kHz` audio from YouTube URLs via `yt-dlp`, transcribes timestamped segments with **OpenAI Whisper**, exports structured **JSON** and **SRT** captions, and indexes utterances into a local **SQLite** database with **FTS5 full-text search**. Includes an offline `--sample-demo` mode and deterministic `unittest` coverage.

## Pipeline Architecture

```text
+-------------------+     yt-dlp + ffmpeg      +------------------------+
|  YouTube Video ID | =======================> | 16 kHz Mono WAV Stream |
+-------------------+   URL Validation         +-----------+------------+
                                                           |
                                                           v
+-------------------+   Idempotent Upsert      +------------------------+
| SQLite + FTS5 DB  | <======================= |  Whisper ASR & Metrics |
| (videos/segments) |   + JSON & SRT Exports   | (WPM, Confidence Filter)|
+-------------------+                          +------------------------+
```

## Repository Structure

```text
youtube-whisper-pipeline/
├── .gitignore                  # Excludes downloaded audio, outputs/, and SQLite DBs
├── LICENSE                     # MIT License
├── main.py                     # CLI entrypoint (ingest, --sample-demo, --search)
├── requirements.txt            # Pinned yt-dlp and openai-whisper dependencies
├── src/
│   ├── __init__.py
│   ├── extractor.py            # Strict YouTube ID parser & yt-dlp audio extractor
│   ├── storage.py              # SQLite relational schema, FTS5 index, JSON/SRT writers
│   └── transcriber.py          # Whisper model runner, SRT timestamps, and WPM metrics
└── tests/
    └── test_pipeline.py        # End-to-end ETL & SQLite FTS5 unit tests (zero cloud calls)
```

## Storage Schema (SQLite 3 + FTS5)

Each ingested video is normalized across two relational tables and one virtual full-text index:

| Table / Virtual Index | Primary Key / Type | Purpose |
|---|---|---|
| `videos` | `video_id TEXT PRIMARY KEY` | Stores title, source URL, model name, total duration, word count, and speaking rate (`words_per_minute`) |
| `transcript_segments` | `(video_id, segment_idx)` | Stores `start_sec`, `end_sec`, normalized `text`, and `confidence_score` |
| `segments_fts` | `FTS5 Virtual Table` | Enables sub-millisecond keyword lookup returning exact video timestamps |

## Local Verification & Usage

### 1. Run Automated Unit Tests (No GPU, Network, or Extra Packages Required)

```bash
git clone https://github.com/benjarong-dev/youtube-whisper-pipeline.git
cd youtube-whisper-pipeline

python3 -m unittest discover -s tests -v
```

### 2. Run Offline End-to-End ETL & Full-Text Search Demo

```bash
# Ingests a deterministic sample lecture into outputs/transcripts.db, JSON, and SRT
python3 main.py --sample-demo --output-dir ./outputs

# Query the SQLite FTS5 index for exact spoken timestamps
python3 main.py --output-dir ./outputs --search "vlan"
```

### 3. Transcribe a Live YouTube URL (Requires ffmpeg & Whisper Dependencies)

```bash
sudo apt install -y ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --model "base" --output-dir ./outputs
```

## License

Distributed under the MIT License. See `LICENSE` for details.
