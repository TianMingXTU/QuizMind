from __future__ import annotations

import io
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from quizmind.logger import timed_event
from quizmind.models import Difficulty, KnowledgePoint, ParsedContent

MAX_SEGMENTS = max(20, int(os.getenv("QUIZMIND_MAX_SEGMENTS", "200")))
MAX_KNOWLEDGE_POINTS = max(6, int(os.getenv("QUIZMIND_MAX_KNOWLEDGE_POINTS", "24")))
SEGMENT_MAX_LEN = max(120, int(os.getenv("QUIZMIND_SEGMENT_MAX_LEN", "1200")))


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "is",
    "in",
    "on",
    "to",
    "of",
    "a",
    "an",
}


def load_text_from_upload(file_name: str, data: bytes) -> str:
    with timed_event("content.load_upload", file_name=file_name):
        suffix = Path(file_name).suffix.lower()
        if suffix in {".md", ".txt"}:
            return data.decode("utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".doc":
            raise ValueError("Legacy .doc is not supported. Please convert to .docx.")
        if suffix == ".docx":
            try:
                document = Document(io.BytesIO(data))
            except Exception as exc:
                raise ValueError(
                    "This .docx file is not a valid Word document package. "
                    "Please re-export it as a standard .docx and try again."
                ) from exc
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        raise ValueError(f"Unsupported file type: {suffix}")


def load_text_from_url(url: str) -> str:
    with timed_event("content.load_url", url=url):
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        return re.sub(r"\n{2,}", "\n", text).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_segments(text: str, max_len: int = SEGMENT_MAX_LEN) -> List[str]:
    raw_segments = re.split(r"\n+|(?<=[.!?。！？])", text)
    segments: List[str] = []
    buf = ""
    for part in raw_segments:
        clean = part.strip()
        if not clean:
            continue
        if len(buf) + len(clean) <= max_len:
            buf = f"{buf} {clean}".strip()
        else:
            if buf:
                segments.append(buf)
            buf = clean
    if buf:
        segments.append(buf)
    return segments[:MAX_SEGMENTS]


def extract_keywords(text: str, limit: int = 15) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    normalized = [token.lower() for token in tokens]
    filtered = [token for token in normalized if token not in STOP_WORDS]
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(limit)]


def infer_difficulty(segment: str) -> Difficulty:
    length = len(segment)
    if length < 60:
        return Difficulty.easy
    if length < 140:
        return Difficulty.medium
    return Difficulty.hard


def build_knowledge_points(segments: Iterable[str], keywords: List[str]) -> List[KnowledgePoint]:
    points: List[KnowledgePoint] = []
    for index, segment in enumerate(segments, start=1):
        local_keywords = [word for word in keywords if word in segment.lower()][:4]
        if not local_keywords:
            local_keywords = extract_keywords(segment, limit=4)
        point_name = local_keywords[0] if local_keywords else f"topic_{index}"
        points.append(
            KnowledgePoint(
                name=point_name,
                summary=segment,
                importance=max(1, min(5, len(local_keywords) + 1)),
                difficulty=infer_difficulty(segment),
                keywords=local_keywords,
            )
        )

    deduped: List[KnowledgePoint] = []
    seen = set()
    for point in points:
        if point.name in seen:
            continue
        deduped.append(point)
        seen.add(point.name)
    return deduped[:MAX_KNOWLEDGE_POINTS]


def fallback_parse_content(source: str, source_type: str) -> ParsedContent:
    with timed_event("content.fallback_parse", source_type=source_type):
        cleaned = normalize_text(source)
        segments = split_segments(cleaned)
        keywords = extract_keywords(cleaned)
        knowledge_points = build_knowledge_points(segments, keywords)
        first_line = next((line.strip() for line in cleaned.split("\n") if line.strip()), "")
        title = (first_line[:120] if first_line else (segments[0][:120] if segments else "untitled_content")).strip()
        return ParsedContent(
            title=title,
            source_type=source_type,  # type: ignore[arg-type]
            cleaned_text=cleaned,
            segments=segments,
            knowledge_points=knowledge_points,
            concepts=[point.name for point in knowledge_points],
        )
