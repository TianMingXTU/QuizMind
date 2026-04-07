from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from quizmind.models import Question, Quiz


class UserFeatureStore:
    def __init__(self, root: str = ".quizmind_runtime/user_data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "features.json"

    def add_favorite(self, question: Question, source_title: str = "") -> str:
        data = self._load()
        fp = self.question_fingerprint(question)
        favorites = data.get("favorites", [])
        if any(item.get("fingerprint") == fp for item in favorites):
            return fp
        favorites.append(
            {
                "fingerprint": fp,
                "question": question.model_dump(),
                "source_title": source_title,
                "added_at": datetime.now().isoformat(),
            }
        )
        data["favorites"] = favorites
        self._save(data)
        return fp

    def remove_favorite(self, fingerprint: str) -> None:
        data = self._load()
        favorites = data.get("favorites", [])
        data["favorites"] = [
            item for item in favorites if item.get("fingerprint") != fingerprint
        ]
        self._save(data)

    def list_favorites(self) -> List[Dict[str, Any]]:
        items = self._load().get("favorites", [])
        return list(reversed(items))

    def has_favorite(self, question: Question) -> bool:
        fp = self.question_fingerprint(question)
        return any(item.get("fingerprint") == fp for item in self.list_favorites())

    def build_favorites_quiz(self, question_limit: int = 15) -> Quiz | None:
        favorites = self.list_favorites()
        if not favorites:
            return None
        questions: list[Question] = []
        for item in favorites[:question_limit]:
            raw = item.get("question")
            if not isinstance(raw, dict):
                continue
            try:
                questions.append(Question.model_validate(raw))
            except Exception:
                continue
        if not questions:
            return None
        return Quiz(
            title="我的收藏题单",
            source_summary="基于收藏题生成",
            questions=questions,
        )

    def add_quality_feedback(
        self,
        question: Question,
        verdict: str,
        detail: str = "",
        source_title: str = "",
    ) -> None:
        data = self._load()
        feedback = data.get("quality_feedback", [])
        feedback.append(
            {
                "fingerprint": self.question_fingerprint(question),
                "verdict": verdict.strip(),
                "detail": detail.strip(),
                "source_title": source_title,
                "tags": list(question.knowledge_tags or []),
                "question_type": str(question.question_type.value),
                "created_at": datetime.now().isoformat(),
            }
        )
        data["quality_feedback"] = feedback[-1000:]
        self._save(data)

    def blocked_tags(self, min_votes: int = 2) -> set[str]:
        data = self._load()
        counts: dict[str, int] = {}
        blocked_verdicts = {"题干不清晰", "疑似错误", "Unclear", "Possibly Wrong"}
        for item in data.get("quality_feedback", []):
            verdict = str(item.get("verdict", "")).strip()
            if verdict not in blocked_verdicts:
                continue
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                continue
            for tag in tags:
                key = str(tag).strip()
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
        return {tag for tag, n in counts.items() if n >= min_votes}

    def add_learning_session(
        self,
        *,
        source_name: str,
        quiz_title: str,
        overall_score: float,
        objective_accuracy: float,
        subjective_average: float,
        wrong_count: int,
        weak_topics: List[str],
    ) -> None:
        data = self._load()
        sessions = data.get("learning_sessions", [])
        sessions.append(
            {
                "created_at": datetime.now().isoformat(),
                "source_name": source_name,
                "quiz_title": quiz_title,
                "overall_score": float(overall_score),
                "objective_accuracy": float(objective_accuracy),
                "subjective_average": float(subjective_average),
                "wrong_count": int(wrong_count),
                "weak_topics": weak_topics[:10],
            }
        )
        data["learning_sessions"] = sessions[-2000:]
        self._save(data)

    def recent_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        items = self._load().get("learning_sessions", [])
        return list(reversed(items[-limit:]))

    def suggest_difficulty_mix(
        self, base_mix: Dict[str, int]
    ) -> Dict[str, int]:
        sessions = self.recent_sessions(limit=8)
        if not sessions:
            return dict(base_mix)

        avg_acc = sum(float(x.get("objective_accuracy", 0.0)) for x in sessions) / max(
            1, len(sessions)
        )
        easy = int(base_mix.get("easy", 30))
        medium = int(base_mix.get("medium", 50))
        hard = int(base_mix.get("hard", 20))

        if avg_acc >= 85:
            easy = max(5, easy - 10)
            hard = min(60, hard + 10)
        elif avg_acc <= 60:
            easy = min(70, easy + 15)
            hard = max(5, hard - 10)

        medium = max(0, 100 - easy - hard)
        return {"easy": easy, "medium": medium, "hard": hard}

    def weekly_dashboard(self, days: int = 7) -> Dict[str, Any]:
        sessions = self.recent_sessions(limit=500)
        now = datetime.now()
        start = (now - timedelta(days=max(1, days - 1))).date()

        by_day: dict[str, list[dict[str, Any]]] = {}
        for item in sessions:
            try:
                created = datetime.fromisoformat(str(item.get("created_at", "")))
            except Exception:
                continue
            day = created.date()
            if day < start:
                continue
            key = day.isoformat()
            by_day.setdefault(key, []).append(item)

        labels = [(start + timedelta(days=i)).isoformat() for i in range(days)]
        attempts = []
        avg_scores = []
        for key in labels:
            group = by_day.get(key, [])
            attempts.append(len(group))
            if group:
                avg = sum(float(x.get("overall_score", 0.0)) for x in group) / len(group)
            else:
                avg = 0.0
            avg_scores.append(round(avg, 2))

        total_attempts = sum(attempts)
        all_scores = [float(x.get("overall_score", 0.0)) for x in sessions[:days * 5]]
        avg_score = round(sum(all_scores) / max(1, len(all_scores)), 2)
        return {
            "labels": labels,
            "attempts": attempts,
            "avg_scores": avg_scores,
            "total_attempts": total_attempts,
            "avg_score": avg_score,
        }

    def question_fingerprint(self, question: Question) -> str:
        payload = {
            "type": str(question.question_type.value),
            "prompt": str(question.prompt).strip(),
            "tags": list(question.knowledge_tags or []),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> Dict[str, Any]:
        if not self.file.exists():
            return {"favorites": [], "quality_feedback": [], "learning_sessions": []}
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("favorites", [])
                data.setdefault("quality_feedback", [])
                data.setdefault("learning_sessions", [])
                return data
        except Exception:
            pass
        return {"favorites": [], "quality_feedback": [], "learning_sessions": []}

    def _save(self, data: Dict[str, Any]) -> None:
        self.file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
