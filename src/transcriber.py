"""Whisper ASR runner, confidence scoring, speaking-rate metrics, and SRT formatter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

from .extractor import VideoMetadata


@dataclass(frozen=True)
class TranscriptSegment:
    """Single timestamped utterance with normalized confidence."""

    segment_idx: int
    start_sec: float
    end_sec: float
    text: str
    confidence_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "segment_idx": self.segment_idx,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "confidence_score": self.confidence_score,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    """Aggregated video transcription artifact ready for storage and export."""

    metadata: VideoMetadata
    model_name: str
    language: str
    total_words: int
    words_per_minute: float
    avg_confidence: float
    segments: List[TranscriptSegment]

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": "1.0",
            "video_id": self.metadata.video_id,
            "title": self.metadata.title,
            "source_url": self.metadata.source_url,
            "duration_sec": self.metadata.duration_sec,
            "model_name": self.model_name,
            "language": self.language,
            "total_words": self.total_words,
            "words_per_minute": self.words_per_minute,
            "avg_confidence": self.avg_confidence,
            "segments": [seg.to_dict() for seg in self.segments],
        }


def format_srt_timestamp(seconds: float) -> str:
    """Formats seconds into SubRip (SRT) timestamp format HH:MM:SS,mmm."""
    safe_sec = max(0.0, float(seconds))
    total_ms = int(round(safe_sec * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(segments: List[TranscriptSegment]) -> str:
    """Renders normalized transcript segments into standard .srt caption text."""
    blocks: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = format_srt_timestamp(seg.start_sec)
        end_ts = format_srt_timestamp(seg.end_sec)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{seg.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def logprob_to_confidence(avg_logprob: float) -> float:
    """Converts Whisper natural log-probability (<= 0) into bounded [0.0, 1.0] score."""
    bounded_lp = min(0.0, float(avg_logprob))
    return round(max(0.0, min(1.0, math.exp(bounded_lp))), 3)


def normalize_segments(
    metadata: VideoMetadata,
    raw_segments: List[Dict[str, object]],
    model_name: str = "base",
    language: str = "en",
    min_confidence: float = 0.0,
) -> TranscriptionResult:
    """Cleans Whisper output segments and computes speaking rate & mean confidence."""
    cleaned: List[TranscriptSegment] = []

    for raw in raw_segments:
        text = " ".join(str(raw.get("text", "")).split())
        if not text:
            continue

        start_sec = round(max(0.0, float(raw.get("start", 0.0))), 2)
        end_sec = round(max(start_sec, float(raw.get("end", start_sec))), 2)
        avg_lp = float(raw.get("avg_logprob", -0.25))
        conf = logprob_to_confidence(avg_lp)

        if conf < min_confidence:
            continue

        cleaned.append(
            TranscriptSegment(
                segment_idx=len(cleaned),
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
                confidence_score=conf,
            )
        )

    total_words = sum(len(seg.text.split()) for seg in cleaned)
    inferred_duration = max(
        metadata.duration_sec,
        cleaned[-1].end_sec if cleaned else 0.0,
    )
    if inferred_duration > 0:
        wpm = round(total_words / (inferred_duration / 60.0), 1)
    else:
        wpm = 0.0

    avg_conf = (
        round(sum(seg.confidence_score for seg in cleaned) / len(cleaned), 3)
        if cleaned
        else 0.0
    )

    updated_meta = VideoMetadata(
        video_id=metadata.video_id,
        title=metadata.title,
        source_url=metadata.source_url,
        duration_sec=round(inferred_duration, 2),
        audio_path=metadata.audio_path,
    )

    return TranscriptionResult(
        metadata=updated_meta,
        model_name=model_name,
        language=language,
        total_words=total_words,
        words_per_minute=wpm,
        avg_confidence=avg_conf,
        segments=cleaned,
    )


def transcribe_audio_with_whisper(
    metadata: VideoMetadata,
    model_name: str = "base",
    language: Optional[str] = None,
    min_confidence: float = 0.20,
) -> TranscriptionResult:
    """Executes local OpenAI Whisper inference on extracted 16 kHz WAV audio."""
    if metadata.audio_path is None or not metadata.audio_path.is_file():
        raise FileNotFoundError("Audio WAV file is missing for Whisper inference.")

    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Package 'openai-whisper' is not installed. Install requirements.txt or use --sample-demo."
        ) from exc

    model = whisper.load_model(model_name)
    decode_opts: Dict[str, object] = {"fp16": False}
    if language:
        decode_opts["language"] = language

    output = model.transcribe(str(metadata.audio_path), **decode_opts)
    detected_lang = str(output.get("language") or language or "en")
    raw_segments = list(output.get("segments") or [])

    return normalize_segments(
        metadata=metadata,
        raw_segments=raw_segments,
        model_name=model_name,
        language=detected_lang,
        min_confidence=min_confidence,
    )
