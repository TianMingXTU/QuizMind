from __future__ import annotations

import io
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


STOP_WORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "与",
    "及",
    "并",
    "对",
    "将",
    "把",
    "为",
    "中",
    "通过",
    "一个",
    "可以",
    "进行",
    "如果",
    "我们",
    "需要",
    "使用",
    "以及",
    "内容",
    "系统",
    "用户",
}


def load_text_from_upload(file_name: str, data: bytes) -> str:
    with timed_event("content.load_upload", file_name=file_name):
        suffix = Path(file_name).suffix.lower()
        if suffix in {".md", ".txt"}:
            return data.decode("utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix in {".docx", ".doc"}:
            document = Document(io.BytesIO(data))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        raise ValueError(f"暂不支持的文件类型: {suffix}")


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


def split_segments(text: str, max_len: int = 260) -> List[str]:
    raw_segments = re.split(r"\n+|(?<=[。！？!?])", text)
    segments: List[str] = []
    buffer = ""
    for part in raw_segments:
        clean = part.strip()
        if not clean:
            continue
        if len(buffer) + len(clean) < max_len:
            buffer = f"{buffer}{clean}".strip()
        else:
            if buffer:
                segments.append(buffer)
            buffer = clean
    if buffer:
        segments.append(buffer)
    return segments[:50]


def extract_keywords(text: str, limit: int = 15) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    filtered = [token for token in tokens if token not in STOP_WORDS]
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
    for index, segment in enumerate(segments):
        local_keywords = [word for word in keywords if word in segment][:4]
        if not local_keywords:
            local_keywords = extract_keywords(segment, limit=4)
        points.append(
            KnowledgePoint(
                name=local_keywords[0] if local_keywords else f"知识点{index + 1}",
                summary=segment[:120],
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
    return deduped[:12]


def fallback_parse_content(source: str, source_type: str) -> ParsedContent:
    with timed_event("content.fallback_parse", source_type=source_type):
        cleaned = normalize_text(source)
        segments = split_segments(cleaned)
        keywords = extract_keywords(cleaned)
        knowledge_points = build_knowledge_points(segments, keywords)
        title = segments[0][:30] if segments else "未命名内容"
        return ParsedContent(
            title=title,
            source_type=source_type,  # type: ignore[arg-type]
            cleaned_text=cleaned,
            segments=segments,
            knowledge_points=knowledge_points,
            concepts=[point.name for point in knowledge_points],
        )
