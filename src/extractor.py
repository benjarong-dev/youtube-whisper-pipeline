"""Strict YouTube URL validation and 16 kHz mono WAV extraction via yt-dlp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}


@dataclass(frozen=True)
class VideoMetadata:
    """Normalized metadata for an ingested video source."""

    video_id: str
    title: str
    source_url: str
    duration_sec: float
    audio_path: Optional[Path] = None


def extract_youtube_video_id(url: str) -> str:
    """Extracts and validates an 11-character YouTube video ID from a URL."""
    if not url or not url.strip():
        raise ValueError("YouTube URL must not be empty.")

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in ALLOWED_HOSTS:
        raise ValueError(f"Unsupported or untrusted video host: {host or 'unknown'}")

    candidate_id = ""
    if host == "youtu.be":
        candidate_id = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        query_vals = parse_qs(parsed.query).get("v", [])
        candidate_id = query_vals[0] if query_vals else ""
    elif parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
        parts = [p for p in parsed.path.split("/") if p]
        candidate_id = parts[1] if len(parts) >= 2 else ""

    if not YOUTUBE_ID_PATTERN.match(candidate_id):
        raise ValueError("Invalid YouTube video ID format (expected 11 URL-safe characters).")

    return candidate_id


def download_youtube_audio(url: str, work_dir: Path) -> VideoMetadata:
    """Downloads YouTube audio and converts it to 16 kHz mono WAV for Whisper ASR."""
    video_id = extract_youtube_video_id(url)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(work_dir / f"{video_id}.%(ext)s")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "Package 'yt-dlp' is not installed. Install requirements.txt or use --sample-demo."
        ) from exc

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    wav_path = work_dir / f"{video_id}.wav"
    title = str(info.get("title") or f"YouTube Video {video_id}")
    duration_sec = float(info.get("duration") or 0.0)

    return VideoMetadata(
        video_id=video_id,
        title=title,
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        duration_sec=round(duration_sec, 2),
        audio_path=wav_path,
    )


def get_sample_demo_payload() -> Tuple[VideoMetadata, List[Dict[str, object]]]:
    """Deterministic technical briefing fixture for offline verification."""
    meta = VideoMetadata(
        video_id="DEMO8021QVL",
        title="Enterprise 802.1Q VLAN Segmentation and Firewall Architecture",
        source_url="https://www.youtube.com/watch?v=DEMO8021QVL",
        duration_sec=44.0,
        audio_path=None,
    )
    raw_segments: List[Dict[str, object]] = [
        {
            "start": 0.0,
            "end": 11.2,
            "text": " Segmenting enterprise networks into dedicated 802.1Q VLAN zones limits lateral movement.",
            "avg_logprob": -0.18,
        },
        {
            "start": 11.2,
            "end": 22.8,
            "text": " Management interfaces remain isolated inside VLAN 10 with strict SSH key enforcement.",
            "avg_logprob": -0.14,
        },
        {
            "start": 22.8,
            "end": 34.0,
            "text": " Unmanaged IoT cameras reside in VLAN 40 where stateful nftables rules block RFC1918 east-west traffic.",
            "avg_logprob": -0.21,
        },
        {
            "start": 34.0,
            "end": 44.0,
            "text": " Centralizing PostgreSQL telemetry logs enables sub-second brute-force and port-scan detection.",
            "avg_logprob": -0.16,
        },
    ]
    return meta, raw_segments
