from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from docx import Document


# ----------------------------
# Transcript heuristics
# ----------------------------

def looks_like_transcript(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Strong signatures
    speaker_colon = re.compile(r"^[^:\n]{1,60}:\s+\S+")  # "Speaker: text"
    teams_speaker_time = re.compile(
        r"^[^0-9:\n][^:\n]{1,90}\s+\d{1,2}:\d{2}(?::\d{2})?\s*$"  # "Name   0:05" or "Name  0:05:12"
    )
    cue_arrow = re.compile(r"-->\s*\d")  # VTT/SRT cue arrows
    timestamp_any = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
    teams_header = re.compile(r"(started transcription|stopped transcription)", re.I)

    speaker_hits = 0
    ts_hits = 0
    cue_hits = 0
    header_hits = 0

    for l in lines:
        if speaker_colon.match(l):
            speaker_hits += 1
        if teams_speaker_time.match(l):
            speaker_hits += 1
            ts_hits += 1
        if cue_arrow.search(l):
            cue_hits += 1
        if timestamp_any.search(l):
            ts_hits += 1
        if teams_header.search(l):
            header_hits += 1

    # Accept if it matches any strong transcript signature
    if speaker_hits >= 2:
        return True
    if cue_hits >= 1 and ts_hits >= 2:
        return True
    if header_hits >= 1 and (speaker_hits >= 1 or ts_hits >= 3):
        return True

    return False


# ----------------------------
# Normalizers
# ----------------------------

def normalize_teams_transcript(text: str) -> str:
    """
    Microsoft Teams transcripts often look like:
      Name, Last   0:05
      utterance line 1
      utterance line 2

    Convert to:
      Name, Last: utterance line 1 utterance line 2
    """
    lines = [l.rstrip() for l in text.splitlines()]

    header = re.compile(
        r"^\s*(?P<speaker>[^0-9:\n][^:\n]{1,90}?)\s+(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
    )
    noise = re.compile(r"(started transcription|stopped transcription)", re.I)

    out = []
    current_speaker = None
    buffer = []

    def flush():
        nonlocal buffer, current_speaker
        if current_speaker and buffer:
            utt = " ".join([b.strip() for b in buffer if b.strip()])
            utt = re.sub(r"\s+", " ", utt).strip()
            if utt:
                out.append(f"{current_speaker}: {utt}")
        buffer = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if noise.search(line):
            continue

        m = header.match(line)
        if m:
            flush()
            current_speaker = m.group("speaker").strip()
            # Remove trailing role tags like "(Guest)" "(Organizer)" etc.
            current_speaker = re.sub(r"\s*\(.*?\)\s*$", "", current_speaker).strip()
            buffer = []
            continue

        if current_speaker is None:
            current_speaker = "Unknown"

        buffer.append(line)

    flush()
    return "\n".join(out).strip()


def normalize_generic_transcript(text: str) -> str:
    """
    Normalize many transcript styles into 'Speaker: utterance' lines.
    Handles:
    - Teams: 'Name 0:05' headers + following lines
    - Speaker colon: 'Alice: ...'
    - Timestamp + speaker: '00:01:23 Alice: ...' or '[00:01] Alice: ...'
    - Slack-like logs: 'Alice 10:32 AM: ...' or 'Alice [10:32]: ...'
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Teams detection (before line normalization)
    teams_header = re.compile(r"^[^0-9:\n][^:\n]{1,90}\s+\d{1,2}:\d{2}(?::\d{2})?\s*$", re.M)
    if teams_header.search(t):
        t = normalize_teams_transcript(t)

    lines = [l.strip() for l in t.splitlines() if l.strip()]

    speaker_colon = re.compile(r"^(?P<speaker>[^:]{1,60}):\s*(?P<text>.+)$")
    ts_speaker_colon = re.compile(
        r"^(?:\[\s*)?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)(?:\s*\])?\s*(?P<speaker>[^:]{1,60}):\s*(?P<text>.+)$"
    )
    slackish = re.compile(
        r"^(?P<speaker>[A-Za-z0-9][A-Za-z0-9 ,._\-()]{1,60})\s+(?:\[\s*)?(?P<ts>\d{1,2}:\d{2}(?:\s*[AP]M)?)"
        r"(?:\s*\])?:\s*(?P<text>.+)$",
        re.I
    )

    out = []
    current_speaker = None
    buffer = []

    def flush():
        nonlocal buffer, current_speaker
        if current_speaker and buffer:
            utt = " ".join(buffer).strip()
            utt = re.sub(r"\s+", " ", utt)
            if utt:
                out.append(f"{current_speaker}: {utt}")
        buffer = []

    for line in lines:
        m = ts_speaker_colon.match(line)
        if m:
            flush()
            current_speaker = m.group("speaker").strip()
            buffer = [m.group("text").strip()]
            continue

        m = slackish.match(line)
        if m:
            flush()
            current_speaker = m.group("speaker").strip()
            buffer = [m.group("text").strip()]
            continue

        m = speaker_colon.match(line)
        if m:
            flush()
            current_speaker = m.group("speaker").strip()
            buffer = [m.group("text").strip()]
            continue

        if current_speaker is None:
            current_speaker = "Unknown"
        buffer.append(line)

    flush()
    return "\n".join(out).strip()


# ----------------------------
# Loaders
# ----------------------------

def load_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_docx(path: Path) -> str:
    doc = Document(path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def load_vtt(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        if "-->" in line:
            continue
        if line.strip().isdigit():
            continue
        if line.strip().startswith("WEBVTT"):
            continue
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def load_srt(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in text.splitlines():
        if "-->" in line:
            continue
        if line.strip().isdigit():
            continue
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def load_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    # Common patterns
    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                speaker = item.get("speaker", "")
                texts.append(f"{speaker}: {item['text']}")
        return "\n".join(texts)

    if isinstance(data, dict):
        if "transcript" in data and isinstance(data["transcript"], list):
            texts = []
            for item in data["transcript"]:
                speaker = item.get("speaker", "")
                texts.append(f"{speaker}: {item.get('text','')}")
            return "\n".join(texts)

    return json.dumps(data, indent=2)


def load_csv_file(path: Path) -> str:
    df = pd.read_csv(path)

    text_col = None
    speaker_col = None

    for c in df.columns:
        if c.lower() in ["text", "utterance", "content"]:
            text_col = c
        if c.lower() in ["speaker", "name"]:
            speaker_col = c

    if text_col is None:
        return df.to_string()

    lines = []
    for _, row in df.iterrows():
        speaker = row[speaker_col] if speaker_col else ""
        text = row[text_col]
        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


# ----------------------------
# Master loader
# ----------------------------

def load_transcript(path: Path) -> Tuple[str, bool, str]:
    ext = path.suffix.lower()

    if ext in [".txt", ".md"]:
        text = load_txt_md(path)
    elif ext == ".docx":
        text = load_docx(path)
    elif ext == ".vtt":
        text = load_vtt(path)
    elif ext == ".srt":
        text = load_srt(path)
    elif ext == ".json":
        text = load_json_file(path)
    elif ext == ".csv":
        text = load_csv_file(path)
    else:
        return "", False, f"Unsupported file type: {ext}"

    # Normalize across formats
    text = normalize_generic_transcript(text)

    is_valid = looks_like_transcript(text)
    return text, is_valid, ""