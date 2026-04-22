from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from quizmind.models import ParsedContent, Question, Quiz


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
        knowledge_stats: List[Dict[str, Any]] | None = None,
        stage_key: str = "",
        stage_title: str = "",
    ) -> None:
        data = self._load()
        sessions = data.get("learning_sessions", [])
        snapshot = []
        for item in (knowledge_stats or [])[:30]:
            if not isinstance(item, dict):
                continue
            snapshot.append(
                {
                    "knowledge_point": str(item.get("knowledge_point", "")).strip(),
                    "avg_score": float(item.get("avg_score", 0.0)),
                    "accuracy": float(item.get("accuracy", 0.0)),
                    "status": str(item.get("status", "")).strip(),
                }
            )
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
                "knowledge_stats": snapshot,
                "stage_key": stage_key.strip(),
                "stage_title": stage_title.strip(),
            }
        )
        data["learning_sessions"] = sessions[-2000:]
        self._save(data)

    def recent_sessions(self, limit: int = 30) -> List[Dict[str, Any]]:
        items = self._load().get("learning_sessions", [])
        return list(reversed(items[-limit:]))

    def save_resume_context(
        self,
        *,
        parsed: ParsedContent,
        quiz: Quiz,
        source_name: str,
        origin_label: str,
        mode: str = "practice",
    ) -> None:
        data = self._load()
        data["resume_context"] = {
            "saved_at": datetime.now().isoformat(),
            "source_name": source_name,
            "origin_label": origin_label,
            "mode": mode,
            "parsed": parsed.model_dump(),
            "quiz": quiz.model_dump(),
        }
        self._save(data)

    def load_resume_context(
        self,
    ) -> tuple[ParsedContent, Quiz, Dict[str, Any]] | None:
        raw = self._load().get("resume_context")
        if not isinstance(raw, dict) or not raw:
            return None
        try:
            parsed = ParsedContent.model_validate(raw.get("parsed", {}))
            quiz = Quiz.model_validate(raw.get("quiz", {}))
        except Exception:
            return None
        return parsed, quiz, raw

    def clear_resume_context(self) -> None:
        data = self._load()
        if data.get("resume_context"):
            data["resume_context"] = {}
            self._save(data)

    def suggest_difficulty_mix(self, base_mix: Dict[str, int]) -> Dict[str, int]:
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
        all_scores = [float(x.get("overall_score", 0.0)) for x in sessions[: days * 5]]
        avg_score = round(sum(all_scores) / max(1, len(all_scores)), 2)
        return {
            "labels": labels,
            "attempts": attempts,
            "avg_scores": avg_scores,
            "total_attempts": total_attempts,
            "avg_score": avg_score,
        }

    def ability_profile(self, lookback_sessions: int = 120) -> List[Dict[str, Any]]:
        sessions = self.recent_sessions(limit=max(20, lookback_sessions))
        buckets: Dict[str, Dict[str, float]] = {}
        for session in sessions:
            for item in session.get("knowledge_stats", []) or []:
                if not isinstance(item, dict):
                    continue
                topic = str(item.get("knowledge_point", "")).strip()
                if not topic:
                    continue
                acc = float(item.get("accuracy", 0.0))
                score = float(item.get("avg_score", 0.0))
                box = buckets.setdefault(
                    topic,
                    {
                        "attempts": 0.0,
                        "score_sum": 0.0,
                        "acc_sum": 0.0,
                        "latest_score": 0.0,
                    },
                )
                box["attempts"] += 1.0
                box["score_sum"] += score
                box["acc_sum"] += acc
                box["latest_score"] = score

        profile: List[Dict[str, Any]] = []
        for topic, box in buckets.items():
            attempts = int(box["attempts"])
            if attempts <= 0:
                continue
            mastery = round(box["score_sum"] / attempts, 2)
            pass_rate = round(box["acc_sum"] / attempts, 2)
            status = "已掌握" if mastery >= 85 else "待巩固" if mastery >= 60 else "薄弱"
            profile.append(
                {
                    "knowledge_point": topic,
                    "mastery": mastery,
                    "pass_rate": pass_rate,
                    "attempts": attempts,
                    "status": status,
                    "latest_score": round(box["latest_score"], 2),
                }
            )
        profile.sort(key=lambda x: (x["mastery"], x["attempts"]))
        return profile

    def learning_trend(self, days: int = 7) -> Dict[str, Any]:
        sessions = self.recent_sessions(limit=500)
        now = datetime.now().date()
        start_day = now - timedelta(days=max(1, days - 1))
        in_range = []
        for item in sessions:
            try:
                created = datetime.fromisoformat(str(item.get("created_at", ""))).date()
            except Exception:
                continue
            if created < start_day:
                continue
            in_range.append(item)

        if not in_range:
            return {
                "overall_change": 0.0,
                "objective_change": 0.0,
                "subjective_change": 0.0,
                "best_improved_topic": "",
                "weakest_topic": "",
            }

        head = in_range[-1]
        tail = in_range[0]
        overall_change = round(
            float(tail.get("overall_score", 0.0)) - float(head.get("overall_score", 0.0)),
            2,
        )
        objective_change = round(
            float(tail.get("objective_accuracy", 0.0))
            - float(head.get("objective_accuracy", 0.0)),
            2,
        )
        subjective_change = round(
            float(tail.get("subjective_average", 0.0))
            - float(head.get("subjective_average", 0.0)),
            2,
        )

        topic_scores: Dict[str, List[float]] = {}
        for session in in_range:
            for item in session.get("knowledge_stats", []) or []:
                if not isinstance(item, dict):
                    continue
                topic = str(item.get("knowledge_point", "")).strip()
                if not topic:
                    continue
                topic_scores.setdefault(topic, []).append(float(item.get("avg_score", 0.0)))

        best_improved_topic = ""
        weakest_topic = ""
        best_delta = -10**9
        weakest_score = 10**9
        for topic, scores in topic_scores.items():
            if not scores:
                continue
            delta = scores[-1] - scores[0]
            avg = sum(scores) / len(scores)
            if delta > best_delta:
                best_delta = delta
                best_improved_topic = topic
            if avg < weakest_score:
                weakest_score = avg
                weakest_topic = topic

        return {
            "overall_change": overall_change,
            "objective_change": objective_change,
            "subjective_change": subjective_change,
            "best_improved_topic": best_improved_topic,
            "weakest_topic": weakest_topic,
        }

    def ability_radar_metrics(self, days: int = 14) -> List[Dict[str, Any]]:
        sessions = self.recent_sessions(limit=500)
        start = datetime.now().date() - timedelta(days=max(1, days - 1))
        in_range = []
        for item in sessions:
            try:
                created = datetime.fromisoformat(str(item.get("created_at", ""))).date()
            except Exception:
                continue
            if created >= start:
                in_range.append(item)

        if not in_range:
            return [
                {"metric": "基础准确率", "score": 0.0},
                {"metric": "主观表达", "score": 0.0},
                {"metric": "稳定性", "score": 0.0},
                {"metric": "训练频率", "score": 0.0},
                {"metric": "进步速度", "score": 0.0},
                {"metric": "薄弱修复力", "score": 0.0},
            ]

        overall = [float(x.get("overall_score", 0.0)) for x in in_range]
        objective = [float(x.get("objective_accuracy", 0.0)) for x in in_range]
        subjective = [float(x.get("subjective_average", 0.0)) for x in in_range]
        wrong = [float(x.get("wrong_count", 0.0)) for x in in_range]

        avg_objective = sum(objective) / max(1, len(objective))
        avg_subjective = sum(subjective) / max(1, len(subjective))
        avg_wrong = sum(wrong) / max(1, len(wrong))

        mean_overall = sum(overall) / max(1, len(overall))
        variance = sum((x - mean_overall) ** 2 for x in overall) / max(1, len(overall))
        std = math.sqrt(max(0.0, variance))
        stability = max(0.0, min(100.0, 100.0 - std * 2.2))

        freq_score = max(0.0, min(100.0, len(in_range) * (100.0 / max(6, days // 2))))
        trend = self.learning_trend(days=days)
        progress_speed = max(0.0, min(100.0, 50.0 + float(trend["overall_change"]) * 2.0))
        repair_power = max(0.0, min(100.0, 100.0 - avg_wrong * 10.0))

        return [
            {"metric": "基础准确率", "score": round(avg_objective, 2)},
            {"metric": "主观表达", "score": round(avg_subjective, 2)},
            {"metric": "稳定性", "score": round(stability, 2)},
            {"metric": "训练频率", "score": round(freq_score, 2)},
            {"metric": "进步速度", "score": round(progress_speed, 2)},
            {"metric": "薄弱修复力", "score": round(repair_power, 2)},
        ]

    def topic_trend_series(self, days: int = 7, top_n: int = 5) -> Dict[str, Any]:
        sessions = self.recent_sessions(limit=500)
        today = datetime.now().date()
        start = today - timedelta(days=max(1, days - 1))
        labels = [(start + timedelta(days=i)).isoformat() for i in range(days)]

        topic_day_scores: Dict[str, Dict[str, List[float]]] = {}
        for session in sessions:
            try:
                created = datetime.fromisoformat(str(session.get("created_at", ""))).date()
            except Exception:
                continue
            if created < start:
                continue
            day_key = created.isoformat()
            for item in session.get("knowledge_stats", []) or []:
                if not isinstance(item, dict):
                    continue
                topic = str(item.get("knowledge_point", "")).strip()
                if not topic:
                    continue
                score = float(item.get("avg_score", 0.0))
                topic_day_scores.setdefault(topic, {}).setdefault(day_key, []).append(score)

        if not topic_day_scores:
            return {"labels": labels, "series": {}}

        ranked_topics = sorted(
            topic_day_scores.keys(),
            key=lambda t: sum(len(v) for v in topic_day_scores[t].values()),
            reverse=True,
        )[: max(1, top_n)]

        series: Dict[str, List[float]] = {}
        for topic in ranked_topics:
            values = []
            last_val = 0.0
            for day in labels:
                bucket = topic_day_scores[topic].get(day, [])
                if bucket:
                    last_val = sum(bucket) / len(bucket)
                values.append(round(last_val, 2))
            series[topic] = values

        return {"labels": labels, "series": series}

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
            return {
                "favorites": [],
                "quality_feedback": [],
                "learning_sessions": [],
                "resume_context": {},
            }
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("favorites", [])
                data.setdefault("quality_feedback", [])
                data.setdefault("learning_sessions", [])
                data.setdefault("resume_context", {})
                for item in data["learning_sessions"]:
                    if isinstance(item, dict):
                        item.setdefault("knowledge_stats", [])
                        item.setdefault("stage_key", "")
                        item.setdefault("stage_title", "")
                return data
        except Exception:
            pass
        return {
            "favorites": [],
            "quality_feedback": [],
            "learning_sessions": [],
            "resume_context": {},
        }

    def _save(self, data: Dict[str, Any]) -> None:
        self.file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
