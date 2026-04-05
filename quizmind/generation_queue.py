from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from quizmind.models import QuizConfig


class GenerationQueue:
    def __init__(self, root: str = ".quizmind_runtime/queue") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.root / "queue.json"

    def enqueue_items(self, items: List[Dict[str, str]]) -> int:
        queue = self._load()
        now = datetime.now().isoformat()
        for item in items:
            queue.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": item["name"],
                    "source_type": item["source_type"],
                    "content": item["content"],
                    "status": "pending",
                    "error": "",
                    "record_id": "",
                    "attempts": 0,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        self._save(queue)
        return len(items)

    def list_items(self) -> List[Dict[str, Any]]:
        return list(reversed(self._load()))

    def retry_failed(self) -> int:
        queue = self._load()
        count = 0
        now = datetime.now().isoformat()
        for item in queue:
            if item["status"] == "failed":
                item["status"] = "pending"
                item["error"] = ""
                item["updated_at"] = now
                count += 1
        self._save(queue)
        return count

    def process_pending(
        self,
        engine,
        config: QuizConfig,
        use_saved_first: bool,
        allow_ai_generation: bool,
        max_workers: int = 2,
    ) -> Dict[str, int]:
        queue = self._load()
        id_map = {item["id"]: item for item in queue}
        pending = [item for item in queue if item["status"] == "pending"]
        stats = {"processed": len(pending), "success": 0, "failed": 0}
        if not pending:
            return stats

        for item in pending:
            item["status"] = "processing"
            item["updated_at"] = datetime.now().isoformat()
        self._save(queue)

        worker_count = max(1, min(max_workers, 8))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(
                    self._process_one,
                    item=item,
                    engine=engine,
                    config=config,
                    use_saved_first=use_saved_first,
                    allow_ai_generation=allow_ai_generation,
                ): item["id"]
                for item in pending
            }
            for future in as_completed(future_map):
                item_id = future_map[future]
                item = id_map[item_id]
                try:
                    record_id = future.result()
                    item["status"] = "done"
                    item["record_id"] = record_id
                    item["error"] = ""
                    stats["success"] += 1
                except Exception as exc:
                    item["status"] = "failed"
                    item["error"] = str(exc)
                    stats["failed"] += 1
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["updated_at"] = datetime.now().isoformat()
                self._save(queue)
        return stats

    def clear_done(self) -> int:
        queue = self._load()
        before = len(queue)
        queue = [item for item in queue if item["status"] not in {"done"}]
        self._save(queue)
        return before - len(queue)

    def _process_one(
        self,
        item: Dict[str, Any],
        engine,
        config: QuizConfig,
        use_saved_first: bool,
        allow_ai_generation: bool,
    ) -> str:
        _, _, meta = engine.generate_or_load_from_source(
            source=item["content"],
            source_type=item["source_type"],
            source_name=item["name"],
            config=config,
            use_saved_first=use_saved_first,
            allow_ai_generation=allow_ai_generation,
        )
        return str(meta.get("record_id", ""))

    def _load(self) -> List[Dict[str, Any]]:
        if not self.queue_file.exists():
            return []
        return json.loads(self.queue_file.read_text(encoding="utf-8"))

    def _save(self, queue: List[Dict[str, Any]]) -> None:
        self.queue_file.write_text(
            json.dumps(queue, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

