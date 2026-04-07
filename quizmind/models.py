from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


DIFFICULTY_ALIASES = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "simple": "easy",
    "normal": "medium",
    "difficult": "hard",
    "简单": "easy",
    "中等": "medium",
    "困难": "hard",
}

QUESTION_TYPE_ALIASES = {
    "single_choice": "single_choice",
    "multiple_choice": "multiple_choice",
    "fill_blank": "fill_blank",
    "short_answer": "short_answer",
    "true_false": "true_false",
    "single": "single_choice",
    "multiple": "multiple_choice",
    "single choice": "single_choice",
    "multiple choice": "multiple_choice",
    "fill in the blank": "fill_blank",
    "short": "short_answer",
    "true/false": "true_false",
    "singlechoice": "single_choice",
    "multi": "multiple_choice",
    "单选题": "single_choice",
    "多选题": "multiple_choice",
    "填空题": "fill_blank",
    "简答题": "short_answer",
    "判断题": "true_false",
}


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"

    @classmethod
    def normalize(cls, raw: object) -> str:
        if isinstance(raw, cls):
            return raw.value
        if isinstance(raw, Enum):
            raw = raw.value
        return DIFFICULTY_ALIASES.get(str(raw or "").strip().lower(), "medium")


class QuestionType(str, Enum):
    single_choice = "single_choice"
    multiple_choice = "multiple_choice"
    fill_blank = "fill_blank"
    short_answer = "short_answer"
    true_false = "true_false"

    @classmethod
    def normalize(cls, raw: object) -> str:
        if isinstance(raw, cls):
            return raw.value
        if isinstance(raw, Enum):
            raw = raw.value
        value = str(raw or "").strip().lower()
        if "." in value:
            value = value.split(".")[-1]
        return QUESTION_TYPE_ALIASES.get(value, "single_choice")


class QuizMode(str, Enum):
    practice = "practice"
    exam = "exam"

    @classmethod
    def normalize(cls, raw: object) -> str:
        value = str(raw or "").strip().lower()
        if value in {"exam", "考试模式"}:
            return "exam"
        return "practice"


class KnowledgePoint(BaseModel):
    name: str
    summary: str
    importance: int = Field(ge=1, le=5)
    difficulty: Difficulty
    keywords: List[str] = Field(default_factory=list)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: object) -> str:
        return Difficulty.normalize(value)


class ParsedContent(BaseModel):
    title: str
    source_type: Literal["text", "file", "url", "memory"]
    cleaned_text: str
    segments: List[str]
    knowledge_points: List[KnowledgePoint]
    concepts: List[str]


class Question(BaseModel):
    id: str
    question_type: QuestionType
    prompt: str
    options: List[str] = Field(default_factory=list)
    correct_answer: List[str]
    explanation: str
    knowledge_tags: List[str]
    difficulty: Difficulty
    reference_points: List[str] = Field(default_factory=list)

    @field_validator("question_type", mode="before")
    @classmethod
    def _normalize_question_type(cls, value: object) -> str:
        return QuestionType.normalize(value)

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: object) -> str:
        return Difficulty.normalize(value)


class Quiz(BaseModel):
    title: str
    source_summary: str
    questions: List[Question]


class UserAnswer(BaseModel):
    question_id: str
    answer: List[str]


class QuestionResult(BaseModel):
    question_id: str
    is_correct: bool
    score: float = Field(ge=0, le=100)
    max_score: float = 100.0
    user_answer: List[str]
    correct_answer: List[str]
    feedback: str
    missing_points: List[str] = Field(default_factory=list)


class KnowledgeStat(BaseModel):
    knowledge_point: str
    accuracy: float
    attempts: int
    avg_score: float
    status: str


class FeedbackReport(BaseModel):
    overall_score: float
    objective_accuracy: float
    subjective_average: float
    wrong_questions: List[QuestionResult]
    knowledge_stats: List[KnowledgeStat]
    review_recommendations: List[str]
    reinforcement_topics: List[str]
    reinforcement_quiz: Optional[Quiz] = None


class QuizConfig(BaseModel):
    question_count: int = Field(default=10, ge=1, le=50)
    difficulty_mix: Dict[str, int] = Field(
        default_factory=lambda: {"easy": 30, "medium": 50, "hard": 20}
    )
    type_mix: Dict[str, int] = Field(
        default_factory=lambda: {
            "single_choice": 35,
            "multiple_choice": 15,
            "fill_blank": 20,
            "short_answer": 20,
            "true_false": 10,
        }
    )

    @field_validator("difficulty_mix", mode="before")
    @classmethod
    def _normalize_difficulty_mix(cls, value: object) -> Dict[str, int]:
        if not isinstance(value, dict):
            return {"easy": 30, "medium": 50, "hard": 20}
        out: Dict[str, int] = {}
        for key, ratio in value.items():
            norm = Difficulty.normalize(key)
            out[norm] = int(ratio)
        return out

    @field_validator("type_mix", mode="before")
    @classmethod
    def _normalize_type_mix(cls, value: object) -> Dict[str, int]:
        if not isinstance(value, dict):
            return {
                "single_choice": 35,
                "multiple_choice": 15,
                "fill_blank": 20,
                "short_answer": 20,
                "true_false": 10,
            }
        out: Dict[str, int] = {}
        for key, ratio in value.items():
            norm = QuestionType.normalize(key)
            out[norm] = int(ratio)
        return out


class MemorySnapshot(BaseModel):
    title: str
    chunks: int
    concepts: List[str] = Field(default_factory=list)


class BatchSubjectiveGrade(BaseModel):
    question_id: str
    score: float = Field(ge=0, le=100)
    feedback: str
    missing_points: List[str] = Field(default_factory=list)


class SceneTurnResult(BaseModel):
    engineer_message: str
    should_end: bool = False
    is_passed: bool = False
    score: float = Field(default=0, ge=0, le=100)
    assessment: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
