from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    easy = "简单"
    medium = "中等"
    hard = "困难"


class QuestionType(str, Enum):
    single_choice = "单选题"
    multiple_choice = "多选题"
    fill_blank = "填空题"
    short_answer = "简答题"
    true_false = "判断题"


class QuizMode(str, Enum):
    practice = "练习模式"
    exam = "考试模式"


class KnowledgePoint(BaseModel):
    name: str
    summary: str
    importance: int = Field(ge=1, le=5)
    difficulty: Difficulty
    keywords: List[str] = Field(default_factory=list)


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
        default_factory=lambda: {"简单": 30, "中等": 50, "困难": 20}
    )
    type_mix: Dict[str, int] = Field(
        default_factory=lambda: {
            "单选题": 35,
            "多选题": 15,
            "填空题": 20,
            "简答题": 20,
            "判断题": 10,
        }
    )


class MemorySnapshot(BaseModel):
    title: str
    chunks: int
    concepts: List[str] = Field(default_factory=list)


class BatchSubjectiveGrade(BaseModel):
    question_id: str
    score: float = Field(ge=0, le=100)
    feedback: str
    missing_points: List[str] = Field(default_factory=list)
