from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from quizmind.models import ParsedContent, Quiz, QuizConfig


class QuizBank:
    def __init__(self, root: str = ".quizmind_runtime/quiz_bank") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.root / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"

    def build_signature(self, source: str, source_type: str, config: QuizConfig) -> str:
        payload = {
            "source_type": source_type,
            "source": source.strip(),
            "config": config.model_dump(),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def find_by_signature(self, signature: str) -> Optional[Tuple[ParsedContent, Quiz, Dict[str, Any]]]:
        for item in reversed(self._load_index()):
            if item.get("signature") == signature:
                loaded = self._load_record(item)
                if loaded:
                    parsed, quiz = loaded
                    return parsed, quiz, item
        return None

    def save(
        self,
        signature: str,
        source_name: str,
        source_type: str,
        used_ai: bool,
        parsed: ParsedContent,
        quiz: Quiz,
    ) -> str:
        record_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        record_file = f"{record_id}.json"
        tags = [tag for tag in parsed.concepts[:10]]
        payload = {
            "id": record_id,
            "created_at": created_at,
            "source_name": source_name,
            "source_type": source_type,
            "used_ai": used_ai,
            "signature": signature,
            "tags": tags,
            "parsed": parsed.model_dump(),
            "quiz": quiz.model_dump(),
        }
        (self.root / record_file).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index = self._load_index()
        index.append(
            {
                "id": record_id,
                "created_at": created_at,
                "source_name": source_name,
                "source_type": source_type,
                "used_ai": used_ai,
                "signature": signature,
                "record_file": record_file,
                "question_count": len(quiz.questions),
                "title": quiz.title,
                "tags": tags,
            }
        )
        self._save_index(index)
        return record_id

    def list_recent(self, limit: int = 30) -> List[Dict[str, Any]]:
        index = self._load_index()
        return list(reversed(index[-limit:]))

    def search(
        self,
        file_name_keyword: str = "",
        date_from: str = "",
        date_to: str = "",
        tag_keyword: str = "",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        items = list(reversed(self._load_index()))
        result: List[Dict[str, Any]] = []
        file_kw = file_name_keyword.strip().lower()
        tag_kw = tag_keyword.strip().lower()
        for item in items:
            created_at = str(item.get("created_at", ""))
            source_name = str(item.get("source_name", "")).lower()
            tags = [str(tag).lower() for tag in item.get("tags", [])]
            if file_kw and file_kw not in source_name:
                continue
            if tag_kw and not any(tag_kw in tag for tag in tags):
                continue
            if date_from and created_at[:10] < date_from:
                continue
            if date_to and created_at[:10] > date_to:
                continue
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def get_by_id(self, record_id: str) -> Optional[Tuple[ParsedContent, Quiz, Dict[str, Any]]]:
        for item in self._load_index():
            if item.get("id") == record_id:
                loaded = self._load_record(item)
                if loaded:
                    parsed, quiz = loaded
                    return parsed, quiz, item
                return None
        return None

    def delete_by_id(self, record_id: str) -> bool:
        index = self._load_index()
        target = None
        for item in index:
            if item.get("id") == record_id:
                target = item
                break
        if not target:
            return False

        record_file = self.root / target["record_file"]
        if record_file.exists():
            record_file.unlink()
        index = [item for item in index if item.get("id") != record_id]
        self._save_index(index)
        return True

    def export_path(self, record_id: str, ext: str) -> Path:
        safe_ext = ext.strip(".").lower()
        return self.export_dir / f"{record_id}.{safe_ext}"

    def _load_record(self, item: Dict[str, Any]) -> Optional[Tuple[ParsedContent, Quiz]]:
        record_file = self.root / item["record_file"]
        if not record_file.exists():
            return None
        data = json.loads(record_file.read_text(encoding="utf-8"))
        parsed = ParsedContent.model_validate(data["parsed"])
        quiz = Quiz.model_validate(data["quiz"])
        return parsed, quiz

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.index_file.exists():
            return []
        return json.loads(self.index_file.read_text(encoding="utf-8"))

    def _save_index(self, index: List[Dict[str, Any]]) -> None:
        self.index_file.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

