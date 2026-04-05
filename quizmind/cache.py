from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonFileCache:
    def __init__(self, root: str = ".quizmind_runtime/cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def build_key(self, namespace: str, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{namespace}_{digest}"

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
