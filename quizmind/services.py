from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence

from quizmind.content import fallback_parse_content
from quizmind.llm import LangChainQuizProvider
from quizmind.logger import log_event, timed_event
from quizmind.memory import MemoryStore
from quizmind.models import (
    Difficulty,
    FeedbackReport,
    KnowledgeStat,
    ParsedContent,
    Question,
    QuestionResult,
    QuestionType,
    Quiz,
    QuizConfig,
    UserAnswer,
)


class ContentService:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()

    def parse(self, source: str, source_type: str) -> ParsedContent:
        with timed_event("service.parse_content", source_type=source_type):
            return self.provider.parse_content(source, source_type)


class QuizEngine:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()
        self.memory_store = MemoryStore()

    def generate_quiz(self, parsed: ParsedContent, config: QuizConfig) -> Quiz:
        with timed_event("service.generate_quiz", title=parsed.title, question_count=config.question_count):
            llm_quiz = self.provider.generate_quiz(parsed, config)
            if llm_quiz:
                return llm_quiz
            log_event("service.generate_quiz.fallback_local", title=parsed.title)
            return self._generate_locally(parsed, config)

    def generate_from_source(self, source: str, source_type: str, config: QuizConfig) -> tuple[ParsedContent, Quiz]:
        with timed_event("service.generate_from_source", source_type=source_type):
            parsed, quiz = self.provider.generate_quiz_from_source(source, source_type, config)
            if quiz:
                return parsed, quiz
            fallback_parsed = fallback_parse_content(source, source_type)
            return fallback_parsed, self._generate_locally(fallback_parsed, config)

    def generate_from_memory(self, config: QuizConfig, query: str = "", top_k: int = 4) -> tuple[ParsedContent, Quiz]:
        parsed = self.memory_store.build_memory_content(query=query, top_k=top_k)
        return parsed, self.generate_quiz(parsed, config)

    def save_memory(self, parsed: ParsedContent):
        return self.memory_store.add_parsed_content(parsed)

    def list_memory(self):
        return self.memory_store.list_snapshots()

    def _generate_locally(self, parsed: ParsedContent, config: QuizConfig) -> Quiz:
        questions: List[Question] = []
        type_targets = self._distribution_targets(config.question_count, config.type_mix)
        difficulty_targets = self._distribution_targets(config.question_count, config.difficulty_mix)
        points = parsed.knowledge_points or []
        if not points:
            raise ValueError("没有足够内容可用于生成题目。")

        point_index = 0
        for qtype_name, count in type_targets.items():
            for _ in range(count):
                point = points[point_index % len(points)]
                difficulty = self._next_difficulty(difficulty_targets)
                questions.append(
                    self._build_question(
                        point=point,
                        question_type=QuestionType(qtype_name),
                        difficulty=difficulty,
                        qid=f"Q{len(questions) + 1:03d}",
                    )
                )
                point_index += 1

        return Quiz(
            title=f"{parsed.title} - 智能练习",
            source_summary="、".join(parsed.concepts[:6]) or parsed.title,
            questions=questions[: config.question_count],
        )

    def _distribution_targets(self, total: int, mix: Dict[str, int]) -> Dict[str, int]:
        counts = {name: max(0, math.floor(total * ratio / 100)) for name, ratio in mix.items()}
        while sum(counts.values()) < total:
            for name in sorted(mix, key=mix.get, reverse=True):
                counts[name] += 1
                if sum(counts.values()) >= total:
                    break
        return counts

    def _next_difficulty(self, targets: Dict[str, int]) -> Difficulty:
        for name in ("中等", "简单", "困难"):
            if targets.get(name, 0) > 0:
                targets[name] -= 1
                return Difficulty(name)
        return Difficulty.medium

    def _build_question(self, point, question_type: QuestionType, difficulty: Difficulty, qid: str) -> Question:
        keyword = point.keywords[0] if point.keywords else point.name
        distractors = self._make_distractors(point)
        if question_type == QuestionType.single_choice:
            options = [keyword, *distractors[:3]]
            random.shuffle(options)
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"根据学习内容，以下哪一项最符合“{point.name}”的核心概念？",
                options=options,
                correct_answer=[keyword],
                explanation=f"{point.name} 的核心说明是：{point.summary}",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )
        if question_type == QuestionType.multiple_choice:
            correct = point.keywords[:2] if len(point.keywords) >= 2 else [keyword, point.name]
            options = list(dict.fromkeys(correct + distractors[:2]))
            random.shuffle(options)
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"以下哪些选项与“{point.name}”直接相关？",
                options=options,
                correct_answer=correct,
                explanation=f"与 {point.name} 直接相关的关键点包括：{'、'.join(correct)}。",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )
        if question_type == QuestionType.fill_blank:
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"填空：{point.summary.replace(keyword, '____', 1)}",
                correct_answer=[keyword],
                explanation=f"该空应填写 {keyword}，因为它是该知识点的核心术语。",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )
        if question_type == QuestionType.true_false:
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"判断：{point.summary}",
                options=["正确", "错误"],
                correct_answer=["正确"],
                explanation="该陈述来自学习内容，因此判断为正确。",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )
        return Question(
            id=qid,
            question_type=question_type,
            prompt=f"简要说明“{point.name}”的核心内容，并至少提到两个关键点。",
            correct_answer=[point.summary],
            explanation=f"作答时应覆盖：{point.summary}",
            knowledge_tags=[point.name],
            difficulty=difficulty,
            reference_points=point.keywords,
        )

    def _make_distractors(self, point) -> List[str]:
        anchor = point.keywords[0] if point.keywords else point.name
        base = point.keywords[1:] + ["边界条件", "实现细节", "非核心概念", "外部因素"]
        deduped = [item for item in dict.fromkeys(base) if item != anchor]
        while len(deduped) < 4:
            deduped.append(f"干扰项{len(deduped) + 1}")
        return deduped


class GradingService:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()

    def grade(self, quiz: Quiz, answers: Sequence[UserAnswer]) -> FeedbackReport:
        with timed_event("service.grade", quiz_title=quiz.title, answers=len(answers)):
            answer_map = {item.question_id: item.answer for item in answers}
            question_map = {question.id: question for question in quiz.questions}
            results: List[QuestionResult] = []

            subjective_pairs = [
                (question, answer_map.get(question.id, []))
                for question in quiz.questions
                if question.question_type == QuestionType.short_answer
            ]
            subjective_grades = self.provider.grade_subjective_batch(subjective_pairs)
            if subjective_pairs:
                log_event(
                    "service.grade.subjective_batch",
                    question_count=len(subjective_pairs),
                    llm_graded=len(subjective_grades),
                )

            for question in quiz.questions:
                user_answer = answer_map.get(question.id, [])
                if question.question_type == QuestionType.short_answer:
                    result = self._grade_subjective(question, user_answer, subjective_grades.get(question.id))
                else:
                    result = self._grade_objective(question, user_answer)
                results.append(result)

            return self._build_report(question_map, results)

    def _grade_objective(self, question: Question, answer: List[str]) -> QuestionResult:
        normalized_user = {item.strip() for item in answer if item.strip()}
        normalized_correct = {item.strip() for item in question.correct_answer if item.strip()}
        is_correct = normalized_user == normalized_correct
        score = 100.0 if is_correct else 0.0
        feedback = "回答正确。" if is_correct else f"正确答案：{'、'.join(question.correct_answer)}。"
        return QuestionResult(
            question_id=question.id,
            is_correct=is_correct,
            score=score,
            user_answer=answer,
            correct_answer=question.correct_answer,
            feedback=feedback + question.explanation,
            missing_points=[] if is_correct else question.reference_points,
        )

    def _grade_subjective(self, question: Question, answer: List[str], batch_grade=None) -> QuestionResult:
        if batch_grade:
            return QuestionResult(
                question_id=question.id,
                is_correct=batch_grade.score >= 60,
                score=batch_grade.score,
                user_answer=answer,
                correct_answer=question.correct_answer,
                feedback=batch_grade.feedback,
                missing_points=batch_grade.missing_points,
            )

        text = " ".join(answer)
        tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text))
        reference = set(question.reference_points)
        overlap = len(tokens & reference)
        score = min(100.0, 40.0 + overlap * 20.0) if text.strip() else 0.0
        return QuestionResult(
            question_id=question.id,
            is_correct=score >= 60,
            score=score,
            user_answer=answer,
            correct_answer=question.correct_answer,
            feedback="覆盖较完整。" if score >= 80 else "答案覆盖不够完整，建议补充关键概念。",
            missing_points=sorted(reference - tokens),
        )

    def _build_report(
        self,
        question_map: Dict[str, Question],
        results: List[QuestionResult],
    ) -> FeedbackReport:
        wrong_questions = [result for result in results if not result.is_correct]
        overall_score = round(sum(result.score for result in results) / max(1, len(results)), 2)

        objective_scores = [
            result.score
            for result in results
            if question_map[result.question_id].question_type != QuestionType.short_answer
        ]
        subjective_scores = [
            result.score
            for result in results
            if question_map[result.question_id].question_type == QuestionType.short_answer
        ]
        objective_accuracy = round(sum(score == 100.0 for score in objective_scores) / max(1, len(objective_scores)) * 100, 2)
        subjective_average = round(sum(subjective_scores) / max(1, len(subjective_scores)), 2)

        knowledge_buckets: Dict[str, List[float]] = defaultdict(list)
        for result in results:
            for tag in question_map[result.question_id].knowledge_tags:
                knowledge_buckets[tag].append(result.score)

        knowledge_stats: List[KnowledgeStat] = []
        for tag, scores in knowledge_buckets.items():
            avg_score = round(sum(scores) / len(scores), 2)
            status = "已掌握" if avg_score >= 85 else "待巩固" if avg_score >= 60 else "薄弱"
            knowledge_stats.append(
                KnowledgeStat(
                    knowledge_point=tag,
                    accuracy=round(sum(score >= 60 for score in scores) / len(scores) * 100, 2),
                    attempts=len(scores),
                    avg_score=avg_score,
                    status=status,
                )
            )
        knowledge_stats.sort(key=lambda item: item.avg_score)

        reinforcement_topics = [item.knowledge_point for item in knowledge_stats if item.avg_score < 70][:4]
        recommendations = [
            f"优先复习 {topic}，并重新梳理其定义、关键特征和应用场景。"
            for topic in reinforcement_topics
        ] or ["整体掌握较好，建议开始混合难度模拟测试。"]

        return FeedbackReport(
            overall_score=overall_score,
            objective_accuracy=objective_accuracy,
            subjective_average=subjective_average,
            wrong_questions=wrong_questions,
            knowledge_stats=knowledge_stats,
            review_recommendations=recommendations,
            reinforcement_topics=reinforcement_topics,
        )


def build_reinforcement_quiz(parsed: ParsedContent, report: FeedbackReport, config: QuizConfig) -> Quiz:
    focus_points = [
        point for point in parsed.knowledge_points if point.name in set(report.reinforcement_topics)
    ] or parsed.knowledge_points[:3]
    focused = ParsedContent(
        title=f"{parsed.title} - 强化训练",
        source_type=parsed.source_type,
        cleaned_text=parsed.cleaned_text,
        segments=[point.summary for point in focus_points],
        knowledge_points=focus_points,
        concepts=[point.name for point in focus_points],
    )
    return QuizEngine().generate_quiz(
        focused,
        QuizConfig(
            question_count=min(max(5, len(focus_points) * 2), config.question_count),
            difficulty_mix={"简单": 20, "中等": 50, "困难": 30},
            type_mix={"单选题": 25, "多选题": 20, "填空题": 20, "简答题": 25, "判断题": 10},
        ),
    )
