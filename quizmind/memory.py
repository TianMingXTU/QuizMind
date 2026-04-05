from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from quizmind.content import fallback_parse_content
from quizmind.logger import log_event, timed_event
from quizmind.models import MemorySnapshot, ParsedContent


class HashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in text.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimension
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class MemoryStore:
    def __init__(self, root: str = ".quizmind_runtime/memory") -> None:
        self.root = Path(root)
        self.index_dir = self.root / "faiss_index"
        self.meta_file = self.root / "memory_meta.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.embeddings = self._build_embeddings()
        self.store = self._load_store()

    def _build_embeddings(self) -> Embeddings:
        from os import getenv

        api_key = getenv("SILICONFLOW_API_KEY") or getenv("OPENAI_API_KEY")
        embedding_model = getenv("SILICONFLOW_EMBEDDING_MODEL", "").strip()
        base_url = getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        if api_key and embedding_model:
            log_event("embedding.provider", provider="remote", model=embedding_model)
            return OpenAIEmbeddings(
                api_key=api_key,
                model=embedding_model,
                base_url=base_url,
            )
        log_event("embedding.provider", provider="hash_fallback")
        return HashEmbeddings()

    def _load_store(self) -> FAISS:
        with timed_event("memory.load_store"):
            if self.index_dir.exists():
                return FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            return FAISS.from_texts(
                ["QuizMind memory bootstrap"],
                self.embeddings,
                metadatas=[{"bootstrap": True}],
            )

    def save(self) -> None:
        with timed_event("memory.save"):
            self.store.save_local(str(self.index_dir))

    def add_parsed_content(self, parsed: ParsedContent) -> MemorySnapshot:
        with timed_event("memory.add_content", title=parsed.title, segments=len(parsed.segments)):
            documents = [
                Document(
                    page_content=segment,
                    metadata={
                        "title": parsed.title,
                        "source_type": parsed.source_type,
                        "concepts": ", ".join(parsed.concepts[:8]),
                    },
                )
                for segment in parsed.segments
                if segment.strip()
            ]
            if documents:
                self.store.add_documents(documents)
                self.save()

            snapshot = MemorySnapshot(
                title=parsed.title,
                chunks=len(documents),
                concepts=parsed.concepts[:8],
            )
            self._append_meta(snapshot)
            return snapshot

    def _append_meta(self, snapshot: MemorySnapshot) -> None:
        entries = self.list_snapshots()
        entries.append(snapshot)
        self.meta_file.write_text(
            json.dumps([entry.model_dump() for entry in entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_snapshots(self) -> List[MemorySnapshot]:
        if not self.meta_file.exists():
            return []
        raw = json.loads(self.meta_file.read_text(encoding="utf-8"))
        return [MemorySnapshot.model_validate(item) for item in raw]

    def build_memory_content(self, query: str = "", top_k: int = 4) -> ParsedContent:
        with timed_event("memory.build_content", query=query, top_k=top_k):
            documents = self.retrieve(query=query, top_k=top_k)
            if not documents:
                raise ValueError("Memory store is empty. Save some parsed content first.")
            sample_size = min(len(documents), max(2, min(top_k, len(documents))))
            selected = random.sample(documents, k=sample_size)
            merged_text = "\n".join(doc.page_content for doc in selected)
            return fallback_parse_content(merged_text, "memory")

    def retrieve(self, query: str = "", top_k: int = 4) -> List[Document]:
        with timed_event("memory.retrieve", query=query, top_k=top_k):
            if query.strip():
                docs = self.store.similarity_search(query, k=top_k)
                return [doc for doc in docs if not doc.metadata.get("bootstrap")]

            snapshots = self.list_snapshots()
            if not snapshots:
                return []
            seed = random.choice(snapshots)
            docs = self.store.similarity_search(seed.title, k=top_k + 2)
            return [doc for doc in docs if not doc.metadata.get("bootstrap")][:top_k]
