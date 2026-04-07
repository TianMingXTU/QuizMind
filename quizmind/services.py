from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

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
from quizmind.quiz_bank import QuizBank


class ContentService:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()

    def parse(self, source: str, source_type: str) -> ParsedContent:
        with timed_event("service.parse_content", source_type=source_type):
            try:
                return self.provider.parse_content(source, source_type)
            except Exception as exc:
                log_event("service.parse_content.fallback", source_type=source_type, error=str(exc))
                return fallback_parse_content(source, source_type)

    def generate_interactive_html(self, parsed: ParsedContent, allow_ai_generation: bool = True) -> str:
        with timed_event("service.generate_interactive_html", title=parsed.title):
            if allow_ai_generation:
                html = self.provider.generate_interactive_html(parsed)
                if html and html.strip():
                    return html
            return self._build_local_interactive_html(parsed)

    def _build_local_interactive_html(self, parsed: ParsedContent) -> str:
        topics = parsed.knowledge_points[:6]
        if not topics:
            return "<div style='padding:12px;border:1px solid #ddd;border-radius:10px;'>暂无可展示知识点。</div>"

        buttons = []
        cards = []
        for idx, point in enumerate(topics):
            active = "active" if idx == 0 else ""
            safe_name = point.name.replace("'", "\\'")
            buttons.append(
                f"<button class='tab-btn {active}' onclick=\"showCard({idx})\">{safe_name}</button>"
            )
            cards.append(
                (
                    f"<div class='card {active}' id='card-{idx}'>"
                    f"<h3>{point.name}</h3>"
                    f"<p>{point.summary}</p>"
                    f"<div class='meta'>关键词：{'、'.join(point.keywords[:5]) or '无'}</div>"
                    f"</div>"
                )
            )

        quiz_data = []
        for idx, point in enumerate(topics[:3], start=1):
            correct = point.keywords[0] if point.keywords else point.name
            quiz_data.append(
                {
                    "id": idx,
                    "q": f"“{point.name}”最相关的关键词是？",
                    "opts": [correct, "边界条件", "外部因素"],
                    "a": correct,
                }
            )

        html = f"""
<div class="km-wrap">
  <h2>互动知识卡：{parsed.title}</h2>
  <div class="tabs">{''.join(buttons)}</div>
  <div class="cards">{''.join(cards)}</div>
  <div class="quiz">
    <h3>快速自测</h3>
    <div id="quiz-box"></div>
  </div>
</div>
<style>
.km-wrap{{font-family: "Microsoft YaHei", Arial, sans-serif; padding:12px; background:#f8fbff; border:1px solid #d9e7f5; border-radius:12px;}}
.tabs{{display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 12px;}}
.tab-btn{{border:1px solid #b7cde5; background:#fff; border-radius:999px; padding:6px 12px; cursor:pointer;}}
.tab-btn.active{{background:#1f6feb; color:#fff; border-color:#1f6feb;}}
.card{{display:none; background:#fff; border:1px solid #dbe7f3; border-radius:10px; padding:10px;}}
.card.active{{display:block;}}
.meta{{font-size:12px; color:#4b5563; margin-top:8px;}}
.quiz{{margin-top:14px; background:#fff; border:1px solid #dbe7f3; border-radius:10px; padding:10px;}}
.q-item{{margin:8px 0 12px;}}
.q-opt{{margin:4px 0;}}
.q-result{{font-size:12px; margin-top:4px;}}
</style>
<script>
const quizData = {quiz_data};
function showCard(i){{
  document.querySelectorAll('.card').forEach((el,idx)=>el.classList.toggle('active', idx===i));
  document.querySelectorAll('.tab-btn').forEach((el,idx)=>el.classList.toggle('active', idx===i));
}}
function renderQuiz(){{
  const box = document.getElementById('quiz-box');
  box.innerHTML = '';
  quizData.forEach((item, idx)=>{{
    const div = document.createElement('div');
    div.className = 'q-item';
    div.innerHTML = `<div><strong>${{idx+1}}.</strong> ${{item.q}}</div>`;
    item.opts.forEach(opt=>{{
      const row = document.createElement('div');
      row.className = 'q-opt';
      const id = `q${{item.id}}-${{opt}}`;
      row.innerHTML = `<label><input type="radio" name="q${{item.id}}" value="${{opt}}"> ${{opt}}</label>`;
      div.appendChild(row);
    }});
    const btn = document.createElement('button');
    btn.textContent = '检查答案';
    btn.onclick = ()=>{{
      const checked = div.querySelector(`input[name="q${{item.id}}"]:checked`);
      const result = div.querySelector('.q-result') || document.createElement('div');
      result.className='q-result';
      if(!checked){{ result.textContent='请先选择答案'; result.style.color='#b45309'; }}
      else if(checked.value===item.a){{ result.textContent='回答正确'; result.style.color='#166534'; }}
      else {{ result.textContent=`回答错误，正确答案：${{item.a}}`; result.style.color='#b91c1c'; }}
      if(!div.querySelector('.q-result')) div.appendChild(result);
    }};
    div.appendChild(btn);
    box.appendChild(div);
  }});
}}
renderQuiz();
</script>
"""
        return html


class QuizEngine:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()
        self.memory_store = MemoryStore()
        self.quiz_bank = QuizBank()

    def _repair_quiz(self, parsed: ParsedContent, quiz: Quiz, config: QuizConfig) -> Quiz:
        if not quiz or not isinstance(getattr(quiz, "questions", None), list):
            return self._generate_locally(parsed, config)

        sanitized_questions: list[Question] = []
        used_ids: set[str] = set()
        used_signatures: set[str] = set()
        for idx, question in enumerate(quiz.questions, start=1):
            repaired = self._repair_question(question, idx)
            signature = self._question_signature(repaired)
            if repaired.id in used_ids or signature in used_signatures:
                continue
            used_ids.add(repaired.id)
            used_signatures.add(signature)
            sanitized_questions.append(repaired)

        if not sanitized_questions:
            return self._generate_locally(parsed, config)

        repaired_quiz = Quiz(
        title=(quiz.title or f"{parsed.title} - 智能练习").strip(),
            source_summary=(quiz.source_summary or parsed.title).strip(),
            questions=sanitized_questions,
        )

        should_pad = len(repaired_quiz.questions) < config.question_count
        should_pad = should_pad or (not self._quiz_matches_type_targets(repaired_quiz, config))
        if not should_pad:
            return repaired_quiz.model_copy(update={"questions": repaired_quiz.questions[: config.question_count]})

        filler: list[Question] = []
        for _ in range(3):
            try:
                llm_quiz = self.provider.generate_quiz(parsed, config)
            except Exception:
                llm_quiz = None
            if not llm_quiz or not llm_quiz.questions:
                continue
            for idx, item in enumerate(llm_quiz.questions, start=1):
                repaired = self._repair_question(item, idx)
                filler.append(repaired)
            if len(filler) >= config.question_count:
                break

        if not filler:
            filler = self._generate_locally(parsed, config).questions

        existing_ids = {q.id for q in repaired_quiz.questions}
        existing_sigs = {self._question_signature(q) for q in repaired_quiz.questions}
        merged = list(repaired_quiz.questions)
        for item in filler:
            if len(merged) >= config.question_count:
                break
            sig = self._question_signature(item)
            if item.id in existing_ids or sig in existing_sigs:
                continue
            merged.append(item)
            existing_ids.add(item.id)
            existing_sigs.add(sig)

        return repaired_quiz.model_copy(update={"questions": merged[: config.question_count]})

    def _question_signature(self, question: Question) -> str:
        prompt = re.sub(r"\s+", "", (question.prompt or "").strip().lower())
        prompt = re.sub(r"[^\w\u4e00-\u9fff]", "", prompt)
        qtype = str(question.question_type.value)
        options = "|".join(sorted(str(o).strip().lower() for o in (question.options or [])))
        return f"{qtype}::{prompt}::{options}"

    def _normalize_quiz_for_display(self, parsed: ParsedContent, quiz: Quiz) -> Quiz:
        """Normalize loaded quiz structure for consistent UI rendering without altering quiz size."""
        if not quiz or not isinstance(getattr(quiz, "questions", None), list):
            return Quiz(title=parsed.title, source_summary=parsed.title, questions=[])

        sanitized_questions: list[Question] = []
        used_ids: set[str] = set()
        for idx, question in enumerate(quiz.questions, start=1):
            repaired = self._repair_question(question, idx)
            if repaired.id in used_ids:
                repaired = repaired.model_copy(update={"id": f"{repaired.id}_{idx}"})
            used_ids.add(repaired.id)
            sanitized_questions.append(repaired)

        return Quiz(
            title=(quiz.title or f"{parsed.title} - 智能练习").strip(),
            source_summary=(quiz.source_summary or parsed.title).strip(),
            questions=sanitized_questions,
        )

    def _repair_question(self, question: Question, index: int) -> Question:
        data = question.model_dump()
        qtype = QuestionType.normalize(data.get("question_type"))
        data["question_type"] = qtype

        qid = str(data.get("id", "")).strip() or f"Q{index:03d}"
        data["id"] = qid

        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            tags = data.get("knowledge_tags") or []
            fallback_topic = str(tags[0]).strip() if tags else qid
            data["prompt"] = f"请回答与“{fallback_topic}”相关的问题。"
        else:
            data["prompt"] = prompt

        options = data.get("options") or []
        if not isinstance(options, list):
            options = [str(options)]
        options = [str(opt).strip() for opt in options if str(opt).strip()]

        correct_answer = data.get("correct_answer") or []
        if not isinstance(correct_answer, list):
            correct_answer = [str(correct_answer)]
        correct_answer = [str(ans).strip() for ans in correct_answer if str(ans).strip()]

        if qtype == QuestionType.true_false.value:
            if set(options) != {"正确", "错误"}:
                options = ["正确", "错误"]
            if not correct_answer or correct_answer[0] not in {"正确", "错误"}:
                correct_answer = ["正确"]
        elif qtype in {QuestionType.single_choice.value, QuestionType.multiple_choice.value}:
            if not options:
                seed = correct_answer[0] if correct_answer else "閫夐」A"
                options = [seed, "閫夐」B", "閫夐」C", "閫夐」D"]
            if not correct_answer:
                correct_answer = [options[0]]
        else:
            if not correct_answer:
                correct_answer = [str(data.get("explanation", "")).strip() or "（参考答案略）"]

        data["options"] = options
        data["correct_answer"] = correct_answer
        data["difficulty"] = Difficulty.normalize(data.get("difficulty"))

        knowledge_tags = data.get("knowledge_tags") or []
        if not isinstance(knowledge_tags, list):
            knowledge_tags = [str(knowledge_tags)]
        knowledge_tags = [str(tag).strip() for tag in knowledge_tags if str(tag).strip()]
        data["knowledge_tags"] = knowledge_tags or ["untagged"]

        reference_points = data.get("reference_points") or []
        if not isinstance(reference_points, list):
            reference_points = [str(reference_points)]
        data["reference_points"] = [str(item).strip() for item in reference_points if str(item).strip()]

        data["explanation"] = str(data.get("explanation", "")).strip()

        return Question.model_validate(data)

    def generate_or_load_from_source(
        self,
        source: str,
        source_type: str,
        source_name: str,
        config: QuizConfig,
        use_saved_first: bool = True,
        allow_ai_generation: bool = True,
    ) -> Tuple[ParsedContent, Quiz, Dict[str, object]]:
        signature = self.quiz_bank.build_signature(source, source_type, config)
        if use_saved_first:
            found = self.quiz_bank.find_by_signature(signature)
            if found:
                parsed, quiz, item = found
                repaired = self._repair_quiz(parsed, quiz, config)
                if self._quiz_matches_type_targets(repaired, config) and len(repaired.questions) >= config.question_count:
                    log_event("quiz_bank.hit", source_name=source_name, record_id=item.get("id", ""))
                    return parsed, repaired, {"from_saved": True, "record_id": item.get("id", "")}
                log_event(
                    "quiz_bank.hit_incompatible",
                    source_name=source_name,
                    record_id=item.get("id", ""),
                    reason="question_type_distribution_mismatch",
                )

        if allow_ai_generation:
            parsed, quiz = self.generate_from_source(source, source_type, config)
            used_ai = True
        else:
            parsed = fallback_parse_content(source, source_type)
            quiz = self._generate_locally(parsed, config)
            used_ai = False

        quiz = self._repair_quiz(parsed, quiz, config)
        record_id = self.quiz_bank.save(
            signature=signature,
            source_name=source_name,
            source_type=source_type,
            used_ai=used_ai,
            parsed=parsed,
            quiz=quiz,
        )
        return parsed, quiz, {"from_saved": False, "record_id": record_id}

    def generate_quiz(
        self,
        parsed: ParsedContent,
        config: QuizConfig,
        allow_ai_generation: bool = True,
    ) -> Quiz:
        with timed_event("service.generate_quiz", title=parsed.title, question_count=config.question_count):
            if allow_ai_generation:
                try:
                    llm_quiz = self.provider.generate_quiz(parsed, config)
                    if llm_quiz and llm_quiz.questions:
                        return self._repair_quiz(parsed, llm_quiz, config)
                except Exception as exc:
                    log_event("service.generate_quiz.provider_error", title=parsed.title, error=str(exc))
            log_event("service.generate_quiz.fallback_local", title=parsed.title)
            return self._generate_locally(parsed, config)

    def generate_from_source(
        self,
        source: str,
        source_type: str,
        config: QuizConfig,
    ) -> tuple[ParsedContent, Quiz]:
        with timed_event("service.generate_from_source", source_type=source_type):
            try:
                parsed, quiz = self.provider.generate_quiz_from_source(source, source_type, config)
                if quiz and quiz.questions:
                    return parsed, self._repair_quiz(parsed, quiz, config)
            except Exception as exc:
                log_event("service.generate_from_source.provider_error", source_type=source_type, error=str(exc))
            fallback_parsed = fallback_parse_content(source, source_type)
            return fallback_parsed, self._generate_locally(fallback_parsed, config)

    def generate_from_memory(
        self,
        config: QuizConfig,
        query: str = "",
        top_k: int = 4,
        allow_ai_generation: bool = True,
    ) -> tuple[ParsedContent, Quiz]:
        parsed = self.memory_store.build_memory_content(query=query, top_k=top_k)
        return parsed, self.generate_quiz(parsed, config, allow_ai_generation=allow_ai_generation)

    def save_memory(self, parsed: ParsedContent):
        return self.memory_store.add_parsed_content(parsed)

    def list_memory(self):
        return self.memory_store.list_snapshots()

    def list_saved_quizzes(self, limit: int = 30):
        return self.quiz_bank.list_recent(limit=limit)

    def search_saved_quizzes(
        self,
        file_name_keyword: str = "",
        date_from: str = "",
        date_to: str = "",
        tag_keyword: str = "",
        limit: int = 200,
    ):
        return self.quiz_bank.search(
            file_name_keyword=file_name_keyword,
            date_from=date_from,
            date_to=date_to,
            tag_keyword=tag_keyword,
            limit=limit,
        )

    def load_saved_quiz(self, record_id: str) -> tuple[ParsedContent, Quiz] | None:
        found = self.quiz_bank.get_by_id(record_id)
        if not found:
            return None
        parsed, quiz, _ = found
        return parsed, self._normalize_quiz_for_display(parsed, quiz)

    def delete_saved_quiz(self, record_id: str) -> bool:
        return self.quiz_bank.delete_by_id(record_id)

    def _generate_locally(self, parsed: ParsedContent, config: QuizConfig) -> Quiz:
        questions: List[Question] = []
        type_targets = self._distribution_targets(config.question_count, config.type_mix)
        diff_targets = self._distribution_targets(config.question_count, config.difficulty_mix)
        points = parsed.knowledge_points or []
        if not points:
            raise ValueError("可用于出题的内容不足，请补充学习材料。")

        point_index = 0
        for qtype_name, count in type_targets.items():
            for _ in range(count):
                point = points[point_index % len(points)]
                difficulty = self._next_difficulty(diff_targets)
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
            source_summary="；".join(parsed.concepts[:6]) or parsed.title,
            questions=questions[: config.question_count],
        )

    def _quiz_matches_type_targets(self, quiz: Quiz, config: QuizConfig) -> bool:
        targets = self._distribution_targets(config.question_count, config.type_mix)
        required_types = {name for name, count in targets.items() if count > 0}
        actual_types = {q.question_type.value for q in quiz.questions}
        if not required_types:
            return True
        return required_types.issubset(actual_types)

    def _distribution_targets(self, total: int, mix: Dict[str, int]) -> Dict[str, int]:
        counts = {name: max(0, math.floor(total * ratio / 100)) for name, ratio in mix.items()}
        while sum(counts.values()) < total:
            for name in sorted(mix, key=mix.get, reverse=True):
                counts[name] += 1
                if sum(counts.values()) >= total:
                    break
        return counts

    def _next_difficulty(self, targets: Dict[str, int]) -> Difficulty:
        for name in ("medium", "easy", "hard"):
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
                explanation=f"{point.name} 的核心解释：{point.summary}",
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
                explanation=f"直接相关的关键点包括：{'；'.join(correct)}",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )

        if question_type == QuestionType.fill_blank:
            if keyword and keyword in point.summary:
                blank_prompt = point.summary.replace(keyword, "____", 1)
            else:
                blank_prompt = f"{point.name} 的核心关键词是：____。"
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"填空：{blank_prompt}",
                correct_answer=[keyword],
                explanation=f"该空应为“{keyword}”，它是该知识点的关键术语。",
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )

        if question_type == QuestionType.true_false:
            false_keyword = distractors[0] if distractors else "无关概念"
            if keyword and keyword in point.summary:
                false_statement = point.summary.replace(keyword, false_keyword, 1)
            else:
                false_statement = f"{point.name} 的核心关键词是 {false_keyword}。"
            use_false = (hash(qid) % 2) == 0
            statement = false_statement if use_false else point.summary
            answer = "错误" if use_false else "正确"
            explanation = (
                f"该陈述故意将关键术语替换为“{false_keyword}”，与原文不一致。"
                if use_false
                else "该陈述来自学习内容，因此判断为正确。"
            )
            return Question(
                id=qid,
                question_type=question_type,
                prompt=f"判断：{statement}",
                options=["正确", "错误"],
                correct_answer=[answer],
                explanation=explanation,
                knowledge_tags=[point.name],
                difficulty=difficulty,
                reference_points=point.keywords,
            )

        return Question(
            id=qid,
            question_type=question_type,
            prompt=f"请简要说明“{point.name}”的核心内容，并至少提到两个关键点。",
            correct_answer=[point.summary],
            explanation=f"作答建议覆盖：{point.summary}",
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

            all_pairs = [
                (question, answer_map.get(question.id, []))
                for question in quiz.questions
            ]
            try:
                llm_grades = self.provider.grade_subjective_batch(all_pairs)
            except Exception as exc:
                log_event("service.grade.llm_batch_error", error=str(exc))
                llm_grades = {}
            if all_pairs:
                log_event(
                    "service.grade.llm_batch",
                    question_count=len(all_pairs),
                    llm_graded=len(llm_grades),
                )

            for question in quiz.questions:
                user_answer = answer_map.get(question.id, [])
                llm_grade = llm_grades.get(question.id)
                if llm_grade:
                    objective = question.question_type != QuestionType.short_answer
                    threshold = 99.0 if objective else 60.0
                    result = QuestionResult(
                        question_id=question.id,
                        is_correct=llm_grade.score >= threshold,
                        score=llm_grade.score,
                        user_answer=user_answer,
                        correct_answer=question.correct_answer,
                        feedback=llm_grade.feedback or question.explanation,
                        missing_points=llm_grade.missing_points,
                    )
                elif question.question_type == QuestionType.short_answer:
                    result = self._grade_subjective(question, user_answer, None)
                else:
                    result = self._grade_objective(question, user_answer)
                results.append(result)

            return self._build_report(question_map, results)

    def _grade_objective(self, question: Question, answer: List[str]) -> QuestionResult:
        normalized_user = {item.strip() for item in answer if item.strip()}
        normalized_correct = {item.strip() for item in question.correct_answer if item.strip()}
        is_correct = normalized_user == normalized_correct
        score = 100.0 if is_correct else 0.0
        feedback = "回答正确。" if is_correct else f"正确答案：{', '.join(question.correct_answer)}。"
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

    def _build_report(self, question_map: Dict[str, Question], results: List[QuestionResult]) -> FeedbackReport:
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
            f"优先复习 {topic}：梳理定义、关键特征和应用场景。"
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
    focus_points = [point for point in parsed.knowledge_points if point.name in set(report.reinforcement_topics)] or parsed.knowledge_points[:3]
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
            difficulty_mix={"easy": 20, "medium": 50, "hard": 30},
            type_mix={
                "single_choice": 25,
                "multiple_choice": 20,
                "fill_blank": 20,
                "short_answer": 25,
                "true_false": 10,
            },
        ),
        allow_ai_generation=True,
    )


def build_targeted_quiz(
    parsed: ParsedContent,
    focus_topics: list[str],
    config: QuizConfig,
    question_count: int | None = None,
    allow_ai_generation: bool = True,
) -> Quiz:
    focus_set = {str(item).strip() for item in focus_topics if str(item).strip()}
    if focus_set:
        focus_points = [
            point
            for point in parsed.knowledge_points
            if point.name in focus_set
            or any(keyword in focus_set for keyword in point.keywords)
        ]
    else:
        focus_points = []
    if not focus_points:
        focus_points = parsed.knowledge_points[:4]

    focused = ParsedContent(
        title=f"{parsed.title} - 定向练习",
        source_type=parsed.source_type,
        cleaned_text=parsed.cleaned_text,
        segments=[point.summary for point in focus_points],
        knowledge_points=focus_points,
        concepts=[point.name for point in focus_points],
    )
    total = question_count or min(max(6, len(focus_points) * 3), config.question_count + 4)
    targeted_config = QuizConfig(
        question_count=max(5, min(30, total)),
        difficulty_mix={"easy": 25, "medium": 50, "hard": 25},
        type_mix={
            "single_choice": 35,
            "multiple_choice": 25,
            "fill_blank": 20,
            "short_answer": 15,
            "true_false": 5,
        },
    )
    return QuizEngine().generate_quiz(
        focused,
        targeted_config,
        allow_ai_generation=allow_ai_generation,
    )

