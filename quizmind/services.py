from __future__ import annotations

import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    SceneTurnResult,
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
        self._memory_store: MemoryStore | None = None
        self.quiz_bank = QuizBank()
        self.last_quiz_origin = ""
        self.strict_ai_generation = (
            str(os.getenv("QUIZMIND_STRICT_AI_GENERATION", "false")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if self.strict_ai_generation:
            # In strict mode, fail fast instead of stacking long upstream timeouts.
            self.provider.quiz_gen_attempts = 1
            self.provider.quiz_gen_concurrency = 1

    @property
    def memory_store(self) -> MemoryStore:
        # Lazy initialization to avoid FAISS/embedding setup on every app cold start.
        if self._memory_store is None:
            self._memory_store = MemoryStore()
        return self._memory_store

    def _origin_label(self, used_ai: bool, from_saved: bool = False) -> str:
        if used_ai:
            return "缓存AI" if from_saved else "AI生成"
        return "本地规则"

    def _result_meta(
        self,
        used_ai: bool,
        from_saved: bool,
        record_id: str = "",
    ) -> dict[str, object]:
        origin_label = self._origin_label(used_ai=used_ai, from_saved=from_saved)
        self.last_quiz_origin = origin_label
        return {
            "from_saved": from_saved,
            "record_id": record_id,
            "origin_label": origin_label,
        }

    def _repair_quiz(
        self,
        parsed: ParsedContent,
        quiz: Quiz,
        config: QuizConfig,
        allow_local_fallback: bool = True,
        allow_ai_generation: bool = True,
    ) -> Quiz:
        if not quiz or not isinstance(getattr(quiz, "questions", None), list):
            if allow_local_fallback:
                return self._generate_locally(parsed, config)
            return Quiz(
                title=f"{parsed.title} - 智能练习",
                source_summary=parsed.title,
                questions=[],
            )

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
            if allow_local_fallback:
                return self._generate_locally(parsed, config)
            return Quiz(
                title=f"{parsed.title} - 智能练习",
                source_summary=parsed.title,
                questions=[],
            )

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
        if allow_ai_generation and self.provider.llm:
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

        if not filler and allow_local_fallback:
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

        # In local-only mode, strict dedup can leave too few questions for sparse source text.
        # Top up with explicit local variants to satisfy configured question_count.
        if len(merged) < config.question_count and allow_local_fallback:
            relaxed_pool = filler or self._generate_locally(parsed, config).questions
            idx = 0
            while relaxed_pool and len(merged) < config.question_count:
                base = self._repair_question(
                    relaxed_pool[idx % len(relaxed_pool)],
                    idx + 1,
                )
                variant_no = len(merged) + 1
                merged.append(
                    base.model_copy(
                        update={
                            "id": f"{base.id}_R{variant_no}",
                            "prompt": f"{base.prompt}（变体{variant_no}）",
                        }
                    )
                )
                idx += 1

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
                seed = correct_answer[0] if correct_answer else "选项A"
                options = [seed, "选项B", "选项C", "选项D"]
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
        learning_style: str = "teacher",
    ) -> Tuple[ParsedContent, Quiz, Dict[str, object]]:
        local_fallback_allowed = not (allow_ai_generation and self.strict_ai_generation)
        signature = self.quiz_bank.build_signature(
            source,
            source_type,
            config,
            learning_style=learning_style,
        )
        if use_saved_first:
            found = self.quiz_bank.find_by_signature(signature)
            if found:
                parsed, quiz, item = found
                if (
                    allow_ai_generation
                    and self.strict_ai_generation
                    and not bool(item.get("used_ai", False))
                ):
                    log_event(
                        "quiz_bank.skip_non_ai_record",
                        source_name=source_name,
                        record_id=item.get("id", ""),
                    )
                else:
                    repaired = self._repair_quiz(
                        parsed,
                        quiz,
                        config,
                        allow_local_fallback=local_fallback_allowed,
                        allow_ai_generation=allow_ai_generation,
                    )
                    if (
                        self._quiz_matches_type_targets(repaired, config)
                        and len(repaired.questions) >= config.question_count
                        and self._quiz_passes_quality_baseline(repaired)
                        and self._quiz_passes_ai_review(repaired, parsed.cleaned_text, config)
                    ):
                        log_event("quiz_bank.hit", source_name=source_name, record_id=item.get("id", ""))
                        return parsed, repaired, self._result_meta(
                            used_ai=bool(item.get("used_ai", False)),
                            from_saved=True,
                            record_id=str(item.get("id", "")),
                        )
                    log_event(
                        "quiz_bank.hit_incompatible",
                        source_name=source_name,
                        record_id=item.get("id", ""),
                        reason="question_quality_or_type_mismatch",
                    )

        if allow_ai_generation:
            # More robust than one-shot "parse+quiz" generation: split into two AI stages.
            # This avoids hard failure when the combined call times out under strict mode.
            try:
                parsed = self.provider.parse_content(source, source_type)
            except Exception as exc:
                log_event(
                    "service.generate_or_load.parse_fallback",
                    source_name=source_name,
                    error=str(exc),
                )
                parsed = fallback_parse_content(source, source_type)
            quiz = self.generate_quiz(
                parsed,
                config,
                allow_ai_generation=True,
                learning_style=learning_style,
            )
            used_ai = self.last_quiz_origin != "本地规则"
        else:
            parsed = fallback_parse_content(source, source_type)
            quiz = self._generate_locally(parsed, config)
            used_ai = False
            self.last_quiz_origin = "本地规则"

        quiz = self._repair_quiz(
            parsed,
            quiz,
            config,
            allow_local_fallback=local_fallback_allowed,
            allow_ai_generation=allow_ai_generation,
        )
        if allow_ai_generation and self._quiz_passes_quality_baseline(quiz):
            quiz = self._repair_quiz_by_ai_review(
                parsed,
                quiz,
                config,
                learning_style,
            )
        if allow_ai_generation and (
            not self._quiz_passes_quality_baseline(quiz)
            or not self._quiz_passes_ai_review(quiz, parsed.cleaned_text, config)
        ):
            try:
                improved = self.generate_quiz(
                    parsed,
                    config,
                    allow_ai_generation=True,
                    learning_style=learning_style,
                )
                quiz = self._repair_quiz(
                    parsed,
                    improved,
                    config,
                    allow_local_fallback=local_fallback_allowed,
                    allow_ai_generation=True,
                )
            except Exception as exc:
                log_event(
                    "quiz.quality_retry.failed",
                    source_name=source_name,
                    error=str(exc),
                )
        final_quality_ok = self._quiz_passes_quality_baseline(quiz)
        if allow_ai_generation and final_quality_ok:
            final_quality_ok = self._quiz_passes_ai_review(
                quiz, parsed.cleaned_text, config
            )
        if not final_quality_ok:
            if allow_ai_generation and self.strict_ai_generation:
                raise RuntimeError(
                    "AI题目质量未达标，已拒绝回退到本地规则出题。请检查模型配置后重试。"
                )
            log_event(
                "quiz.save.skipped_low_quality",
                source_name=source_name,
                question_count=len(quiz.questions),
            )
            return parsed, quiz, self._result_meta(
                used_ai=used_ai,
                from_saved=False,
            )

        record_id = self.quiz_bank.save(
            signature=signature,
            source_name=source_name,
            source_type=source_type,
            used_ai=used_ai,
            parsed=parsed,
            quiz=quiz,
        )
        return parsed, quiz, self._result_meta(
            used_ai=used_ai,
            from_saved=False,
            record_id=record_id,
        )

    def generate_quiz(
        self,
        parsed: ParsedContent,
        config: QuizConfig,
        allow_ai_generation: bool = True,
        learning_style: str = "teacher",
    ) -> Quiz:
        local_fallback_allowed = not (allow_ai_generation and self.strict_ai_generation)
        with timed_event("service.generate_quiz", title=parsed.title, question_count=config.question_count):
            if allow_ai_generation:
                if self.strict_ai_generation and not self.provider.llm:
                    raise RuntimeError("未检测到可用LLM，严格AI出题模式下无法生成题目。")
                try:
                    llm_quiz = self.provider.generate_quiz(
                        parsed,
                        config,
                        learning_style=learning_style,
                    )
                    if llm_quiz and llm_quiz.questions:
                        repaired = self._repair_quiz(
                            parsed,
                            llm_quiz,
                            config,
                            allow_local_fallback=local_fallback_allowed,
                            allow_ai_generation=allow_ai_generation,
                        )
                        if self._quiz_passes_quality_baseline(repaired):
                            repaired = self._repair_quiz_by_ai_review(
                                parsed,
                                repaired,
                                config,
                                learning_style,
                            )
                        if self._quiz_passes_quality_baseline(repaired) and self._quiz_passes_ai_review(
                            repaired, parsed.cleaned_text, config
                        ):
                            self.last_quiz_origin = "AI生成"
                            return repaired
                        log_event(
                            "service.generate_quiz.low_quality_retry",
                            title=parsed.title,
                            reason="baseline_not_passed",
                        )
                    parallel_quiz = self._generate_quiz_by_parallel_questions(
                        parsed,
                        config,
                        learning_style,
                    )
                    if parallel_quiz and parallel_quiz.questions:
                        repaired = self._repair_quiz(
                            parsed,
                            parallel_quiz,
                            config,
                            allow_local_fallback=local_fallback_allowed,
                            allow_ai_generation=False,
                        )
                        if self._quiz_passes_quality_baseline(repaired):
                            self.last_quiz_origin = "AI生成"
                            return repaired
                except Exception as exc:
                    log_event("service.generate_quiz.provider_error", title=parsed.title, error=str(exc))
                    if self.strict_ai_generation:
                        raise RuntimeError(f"AI出题失败：{exc}") from exc
            if allow_ai_generation and self.strict_ai_generation:
                raise RuntimeError("AI题目质量未达标，严格AI出题模式下已终止。")
            log_event("service.generate_quiz.fallback_local", title=parsed.title)
            self.last_quiz_origin = "本地规则"
            return self._generate_locally(parsed, config)

    def generate_from_source(
        self,
        source: str,
        source_type: str,
        config: QuizConfig,
        learning_style: str = "teacher",
    ) -> tuple[ParsedContent, Quiz]:
        with timed_event("service.generate_from_source", source_type=source_type):
            if self.strict_ai_generation and not self.provider.llm:
                raise RuntimeError("未检测到可用LLM，严格AI出题模式下无法从源内容生成题目。")
            try:
                parsed, quiz = self.provider.generate_quiz_from_source(
                    source,
                    source_type,
                    config,
                    learning_style=learning_style,
                )
                if quiz and quiz.questions:
                    self.last_quiz_origin = "AI生成"
                    return parsed, self._repair_quiz(
                        parsed,
                        quiz,
                        config,
                        allow_local_fallback=not self.strict_ai_generation,
                        allow_ai_generation=True,
                    )
            except Exception as exc:
                log_event("service.generate_from_source.provider_error", source_type=source_type, error=str(exc))
                if self.strict_ai_generation:
                    raise RuntimeError(f"AI出题失败：{exc}") from exc
            if self.strict_ai_generation:
                raise RuntimeError("AI未返回有效题目，严格AI出题模式下已终止。")
            fallback_parsed = fallback_parse_content(source, source_type)
            self.last_quiz_origin = "本地规则"
            return fallback_parsed, self._generate_locally(fallback_parsed, config)

    def generate_from_memory(
        self,
        config: QuizConfig,
        query: str = "",
        top_k: int = 4,
        allow_ai_generation: bool = True,
        learning_style: str = "teacher",
    ) -> tuple[ParsedContent, Quiz]:
        parsed = self.memory_store.build_memory_content(query=query, top_k=top_k)
        return parsed, self.generate_quiz(
            parsed,
            config,
            allow_ai_generation=allow_ai_generation,
            learning_style=learning_style,
        )

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

    def _quiz_passes_quality_baseline(self, quiz: Quiz) -> bool:
        if not quiz or not quiz.questions:
            return False

        prompts = []
        for q in quiz.questions:
            prompt = str(q.prompt or "").strip()
            explanation = str(q.explanation or "").strip()
            if len(prompt) < 10 or len(explanation) < 10:
                return False
            if q.question_type.value in {
                QuestionType.single_choice.value,
                QuestionType.multiple_choice.value,
            }:
                options = [str(x).strip() for x in (q.options or []) if str(x).strip()]
                if len(options) < 4:
                    return False
                generic = sum(
                    1
                    for x in options
                    if re.search(r"^(选项|option)\s*[A-D]?$", x, flags=re.IGNORECASE)
                )
                if generic >= 2:
                    return False
            prompts.append(re.sub(r"[^\w\u4e00-\u9fff]", "", prompt.lower()))

        dedup_ratio = len(set(prompts)) / max(1, len(prompts))
        return dedup_ratio >= 0.8

    def _quiz_passes_ai_review(
        self,
        quiz: Quiz,
        source_text: str,
        config: QuizConfig,
    ) -> bool:
        review = self._run_ai_review(quiz, source_text, config)
        return bool(review is None or review.get("pass", False))

    def _run_ai_review(
        self,
        quiz: Quiz,
        source_text: str,
        config: QuizConfig,
    ) -> dict | None:
        if not self.provider.ai_quality_review:
            return None
        try:
            review = self.provider.review_quiz_quality(quiz, source_text, config)
        except Exception as exc:
            log_event("quiz.ai_review.error", error=str(exc))
            return None

        passed = bool(review.get("pass", False))
        if not passed:
            log_event(
                "quiz.ai_review.rejected",
                score=review.get("overall_score", 0),
                issue_count=len(review.get("issues", [])),
                summary=review.get("summary", ""),
            )
        return review

    def _regenerate_single_question(
        self,
        parsed: ParsedContent,
        question: Question,
        learning_style: str,
    ) -> Question | None:
        if not self.provider.llm:
            return None
        focus_points = _select_focus_points(parsed, list(question.knowledge_tags or []), fallback_count=2)
        focused = _build_focused_parsed_content(parsed, focus_points, f"题目修复 {question.id}")
        type_mix = {
            QuestionType.single_choice.value: 0,
            QuestionType.multiple_choice.value: 0,
            QuestionType.fill_blank.value: 0,
            QuestionType.short_answer.value: 0,
            QuestionType.true_false.value: 0,
        }
        type_mix[question.question_type.value] = 100
        diff_mix = {"easy": 0, "medium": 0, "hard": 0}
        diff_mix[question.difficulty.value] = 100
        single_config = QuizConfig(
            question_count=1,
            difficulty_mix=diff_mix,
            type_mix=type_mix,
        )
        candidate = self.provider.generate_quiz(
            focused,
            single_config,
            learning_style=learning_style,
        )
        if not candidate or not candidate.questions:
            return None
        picked = next(
            (item for item in candidate.questions if item.question_type == question.question_type),
            candidate.questions[0],
        )
        repaired = self._repair_question(picked, 1)
        return repaired.model_copy(update={"id": question.id})

    def _generate_single_question_candidate(
        self,
        parsed: ParsedContent,
        question_type: QuestionType,
        difficulty: Difficulty,
        question_id: str,
        point_index: int,
        learning_style: str,
    ) -> Question | None:
        if not self.provider.llm:
            return None
        points = parsed.knowledge_points or []
        focus_names: list[str] = []
        if points:
            focus_names.append(points[point_index % len(points)].name)
            if len(points) > 1:
                focus_names.append(points[(point_index + 1) % len(points)].name)
        focus_points = _select_focus_points(parsed, focus_names, fallback_count=2)
        focused = _build_focused_parsed_content(parsed, focus_points, f"单题生成 {question_id}")
        type_mix = {member.value: 0 for member in QuestionType}
        type_mix[question_type.value] = 100
        diff_mix = {member.value: 0 for member in Difficulty}
        diff_mix[difficulty.value] = 100
        single_config = QuizConfig(
            question_count=1,
            difficulty_mix=diff_mix,
            type_mix=type_mix,
        )
        candidate = self.provider.generate_quiz(
            focused,
            single_config,
            learning_style=learning_style,
        )
        if not candidate or not candidate.questions:
            return None
        picked = next(
            (item for item in candidate.questions if item.question_type == question_type),
            candidate.questions[0],
        )
        repaired = self._repair_question(picked, 1)
        return repaired.model_copy(update={"id": question_id, "difficulty": difficulty})

    def _generate_quiz_by_parallel_questions(
        self,
        parsed: ParsedContent,
        config: QuizConfig,
        learning_style: str,
    ) -> Quiz | None:
        if not self.provider.llm:
            return None
        type_targets = self._distribution_targets(config.question_count, config.type_mix)
        diff_targets = self._distribution_targets(config.question_count, config.difficulty_mix)
        plan: list[tuple[str, QuestionType, Difficulty]] = []
        question_no = 1
        for qtype_name, count in type_targets.items():
            for _ in range(count):
                plan.append(
                    (
                        f"Q{question_no:03d}",
                        QuestionType(qtype_name),
                        self._next_difficulty(diff_targets),
                    )
                )
                question_no += 1
        if not plan:
            return None

        questions: list[Question | None] = [None] * len(plan)
        max_workers = min(4, len(plan))
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="quiz-single"
        ) as pool:
            future_map = {
                pool.submit(
                    self._generate_single_question_candidate,
                    parsed,
                    qtype,
                    difficulty,
                    question_id,
                    index,
                    learning_style,
                ): index
                for index, (question_id, qtype, difficulty) in enumerate(plan)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    questions[index] = future.result()
                except Exception as exc:
                    log_event(
                        "quiz.parallel_single.error",
                        question_id=plan[index][0],
                        error=str(exc),
                    )

        built_questions = [question for question in questions if question]
        if not built_questions:
            return None
        log_event(
            "quiz.parallel_single.complete",
            requested=len(plan),
            generated=len(built_questions),
        )
        return Quiz(
            title=f"{parsed.title} - 智能练习",
            source_summary=parsed.title,
            questions=built_questions,
        )

    def _repair_quiz_by_ai_review(
        self,
        parsed: ParsedContent,
        quiz: Quiz,
        config: QuizConfig,
        learning_style: str,
    ) -> Quiz:
        review = self._run_ai_review(quiz, parsed.cleaned_text, config)
        if not review or bool(review.get("pass", False)):
            return quiz

        issue_ids = [
            str(item.get("question_id", "")).strip()
            for item in (review.get("issues", []) or [])
            if str(item.get("question_id", "")).strip()
        ]
        target_ids = list(dict.fromkeys(issue_ids))
        if not target_ids:
            return quiz

        question_map = {q.id: q for q in quiz.questions}
        replacements: dict[str, Question] = {}
        max_workers = min(3, len(target_ids))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quiz-fix") as pool:
            future_map = {
                pool.submit(
                    self._regenerate_single_question,
                    parsed,
                    question_map[qid],
                    learning_style,
                ): qid
                for qid in target_ids
                if qid in question_map
            }
            for future in as_completed(future_map):
                qid = future_map[future]
                try:
                    repaired = future.result()
                except Exception as exc:
                    log_event("quiz.question_regen.error", question_id=qid, error=str(exc))
                    repaired = None
                if repaired:
                    replacements[qid] = repaired

        if not replacements:
            return quiz

        updated_questions = [replacements.get(q.id, q) for q in quiz.questions]
        repaired_quiz = Quiz(
            title=quiz.title,
            source_summary=quiz.source_summary,
            questions=updated_questions,
        )
        followup_review = self._run_ai_review(repaired_quiz, parsed.cleaned_text, config)
        if followup_review and not bool(followup_review.get("pass", False)):
            return quiz
        return repaired_quiz

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

    def grade(
        self,
        quiz: Quiz,
        answers: Sequence[UserAnswer],
        learning_style: str = "teacher",
    ) -> FeedbackReport:
        with timed_event("service.grade", quiz_title=quiz.title, answers=len(answers)):
            answer_map = {item.question_id: item.answer for item in answers}
            question_map = {question.id: question for question in quiz.questions}
            results: List[QuestionResult] = []

            all_pairs = [
                (question, answer_map.get(question.id, []))
                for question in quiz.questions
            ]
            try:
                llm_grades = self.provider.grade_subjective_batch(
                    all_pairs,
                    learning_style=learning_style,
                )
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
                    is_correct = llm_grade.score >= threshold
                    error_category = self._normalize_error_category(
                        llm_grade.error_category,
                        question=question,
                        answer=user_answer,
                        is_correct=is_correct,
                    )
                    score_breakdown = (
                        llm_grade.score_breakdown
                        if llm_grade.score_breakdown
                        else self._default_breakdown(
                            score=llm_grade.score,
                            objective=objective,
                        )
                    )
                    structured_explanation = (
                        llm_grade.structured_explanation.strip()
                        if llm_grade.structured_explanation
                        else self._build_structured_explanation(
                            question=question,
                            user_answer=user_answer,
                            is_correct=is_correct,
                            feedback=llm_grade.feedback or question.explanation,
                            error_category=error_category,
                            learning_style=learning_style,
                        )
                    )
                    result = QuestionResult(
                        question_id=question.id,
                        is_correct=is_correct,
                        score=llm_grade.score,
                        user_answer=user_answer,
                        correct_answer=question.correct_answer,
                        feedback=llm_grade.feedback or question.explanation,
                        missing_points=llm_grade.missing_points,
                        error_category=error_category,
                        score_breakdown=score_breakdown,
                        structured_explanation=structured_explanation,
                    )
                elif question.question_type == QuestionType.short_answer:
                    result = self._grade_subjective(
                        question,
                        user_answer,
                        None,
                        learning_style=learning_style,
                    )
                else:
                    result = self._grade_objective(
                        question,
                        user_answer,
                        learning_style=learning_style,
                    )
                results.append(result)

            return self._build_report(question_map, results)

    def _grade_objective(
        self,
        question: Question,
        answer: List[str],
        learning_style: str = "teacher",
    ) -> QuestionResult:
        normalized_user = {item.strip() for item in answer if item.strip()}
        normalized_correct = {item.strip() for item in question.correct_answer if item.strip()}
        is_correct = normalized_user == normalized_correct
        score = 100.0 if is_correct else 0.0
        feedback = "回答正确。" if is_correct else f"正确答案：{', '.join(question.correct_answer)}。"
        error_category = self._normalize_error_category(
            "",
            question=question,
            answer=answer,
            is_correct=is_correct,
        )
        return QuestionResult(
            question_id=question.id,
            is_correct=is_correct,
            score=score,
            user_answer=answer,
            correct_answer=question.correct_answer,
            feedback=feedback + question.explanation,
            missing_points=[] if is_correct else question.reference_points,
            error_category=error_category,
            score_breakdown=self._default_breakdown(score=score, objective=True),
            structured_explanation=self._build_structured_explanation(
                question=question,
                user_answer=answer,
                is_correct=is_correct,
                feedback=feedback + question.explanation,
                error_category=error_category,
                learning_style=learning_style,
            ),
        )

    def _grade_subjective(
        self,
        question: Question,
        answer: List[str],
        batch_grade=None,
        learning_style: str = "teacher",
    ) -> QuestionResult:
        if batch_grade:
            is_correct = batch_grade.score >= 60
            error_category = self._normalize_error_category(
                batch_grade.error_category,
                question=question,
                answer=answer,
                is_correct=is_correct,
            )
            return QuestionResult(
                question_id=question.id,
                is_correct=is_correct,
                score=batch_grade.score,
                user_answer=answer,
                correct_answer=question.correct_answer,
                feedback=batch_grade.feedback,
                missing_points=batch_grade.missing_points,
                error_category=error_category,
                score_breakdown=batch_grade.score_breakdown
                or self._default_breakdown(score=batch_grade.score, objective=False),
                structured_explanation=batch_grade.structured_explanation
                or self._build_structured_explanation(
                    question=question,
                    user_answer=answer,
                    is_correct=is_correct,
                    feedback=batch_grade.feedback,
                    error_category=error_category,
                    learning_style=learning_style,
                ),
            )

        text = " ".join(answer)
        tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text))
        reference = set(question.reference_points)
        overlap = len(tokens & reference)
        score = min(100.0, 40.0 + overlap * 20.0) if text.strip() else 0.0
        is_correct = score >= 60
        feedback = "覆盖较完整。" if score >= 80 else "答案覆盖不够完整，建议补充关键概念。"
        error_category = self._normalize_error_category(
            "",
            question=question,
            answer=answer,
            is_correct=is_correct,
        )
        return QuestionResult(
            question_id=question.id,
            is_correct=is_correct,
            score=score,
            user_answer=answer,
            correct_answer=question.correct_answer,
            feedback=feedback,
            missing_points=sorted(reference - tokens),
            error_category=error_category,
            score_breakdown=self._default_breakdown(score=score, objective=False),
            structured_explanation=self._build_structured_explanation(
                question=question,
                user_answer=answer,
                is_correct=is_correct,
                feedback=feedback,
                error_category=error_category,
                learning_style=learning_style,
            ),
        )

    def _normalize_error_category(
        self,
        raw_category: str,
        question: Question,
        answer: List[str],
        is_correct: bool,
    ) -> str:
        if is_correct:
            return "none"

        aliases = {
            "concept_unclear": "concept_unclear",
            "概念不清": "concept_unclear",
            "careless_mistake": "careless_mistake",
            "粗心错误": "careless_mistake",
            "reasoning_error": "reasoning_error",
            "推理错误": "reasoning_error",
            "knowledge_forgotten": "knowledge_forgotten",
            "知识遗忘": "knowledge_forgotten",
            "expression_issue": "expression_issue",
            "表达问题": "expression_issue",
            "none": "none",
        }
        normalized = aliases.get(str(raw_category or "").strip().lower(), "")
        if normalized:
            return normalized
        return self._rule_based_error_category(question, answer)

    def _rule_based_error_category(self, question: Question, answer: List[str]) -> str:
        text = " ".join(answer).strip()
        if not text:
            return "knowledge_forgotten"

        answer_token_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))
        if question.question_type == QuestionType.short_answer:
            if answer_token_count <= 5:
                return "knowledge_forgotten"
            ref = set(question.reference_points or [])
            if ref:
                tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text))
                hit = len(tokens & ref)
                if hit == 0:
                    return "concept_unclear"
                if hit < max(1, len(ref) // 2):
                    return "reasoning_error"
            return "expression_issue"

        normalized_user = {item.strip() for item in answer if item.strip()}
        normalized_correct = {item.strip() for item in question.correct_answer if item.strip()}
        if normalized_user and normalized_user != normalized_correct:
            if question.question_type in {
                QuestionType.single_choice,
                QuestionType.true_false,
            }:
                return "careless_mistake"
            return "reasoning_error"
        return "concept_unclear"

    def _default_breakdown(self, score: float, objective: bool) -> dict[str, float]:
        if objective:
            return {
                "correctness": 10.0 if score >= 99 else 0.0,
                "completeness": 10.0 if score >= 99 else 0.0,
                "clarity": 8.0 if score >= 99 else 3.0,
            }
        scaled = max(0.0, min(100.0, score)) / 10.0
        completeness = min(10.0, round(scaled + 0.5, 1))
        clarity = min(10.0, round(max(2.0, scaled), 1))
        return {
            "correctness": round(scaled, 1),
            "completeness": completeness,
            "clarity": clarity,
        }

    def _build_structured_explanation(
        self,
        question: Question,
        user_answer: List[str],
        is_correct: bool,
        feedback: str,
        error_category: str,
        learning_style: str = "teacher",
    ) -> str:
        user_text = "、".join(user_answer) if user_answer else "未作答"
        correct_text = "、".join(question.correct_answer) if question.correct_answer else "无"
        options_text = "；".join(question.options) if question.options else "本题无选项"
        common_pitfall = {
            "concept_unclear": "容易把相近概念混为一谈，导致关键定义判断出错。",
            "careless_mistake": "容易因为忽略限定词或选项细节而误选。",
            "reasoning_error": "中间推理链不完整，结论与题干证据不一致。",
            "knowledge_forgotten": "关键知识点回忆不足，导致无法完成作答。",
            "expression_issue": "思路可能正确，但表达不完整，导致得分受限。",
            "none": "本题已掌握，注意保持稳定准确率。",
        }.get(error_category, "建议回到知识点定义与典型例题，做一次对照复盘。")
        transfer = (
            f"把题干中的核心概念替换为同类场景，再判断结论是否仍成立。"
            if question.question_type != QuestionType.short_answer
            else "尝试用“定义-条件-结论”三段式重写一次答案。"
        )
        style = (learning_style or "").strip().lower()
        why_other = "本题选项：{options}。除正确答案外，其余选项与题干关键条件不完全匹配。"
        if style == "concise":
            feedback = feedback.split("。")[0].strip() + "。"
            common_pitfall = common_pitfall.split("，")[0] + "。"
            why_other = "错误选项不满足题干关键条件。"
            transfer = "再做1题同类型变式，确保不再犯同类错误。"
        elif style == "interviewer":
            why_other = "其余选项要么违反题干约束，要么缺少可验证依据，不能用于生产级决策。"
            common_pitfall = f"{common_pitfall} 需要给出可验证依据与边界条件。"
            transfer = "给出更严谨的边界条件与反例，并说明为何你的结论依旧成立。"
        return (
            f"【答案】\n正确答案是：{correct_text}\n\n"
            f"【为什么选这个】\n你的作答：{user_text}。{feedback}\n\n"
            f"【为什么其他选项错】\n{why_other.format(options=options_text)}\n\n"
            f"【常见误区】\n{common_pitfall}\n\n"
            f"【知识点总结】\n本题考查：{'、'.join(question.knowledge_tags) or '核心知识点'}。\n\n"
            f"【举一反三】\n{transfer}"
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


class SceneInterviewService:
    def __init__(self) -> None:
        self.provider = LangChainQuizProvider()

    def next_turn(
        self,
        scene_description: str,
        transcript: list[dict[str, str]],
        max_rounds: int = 12,
        interview_mode: str = "guided",
    ) -> SceneTurnResult:
        result = self.provider.run_engineer_scene_turn(
            scene_description=scene_description,
            transcript=transcript,
            max_rounds=max_rounds,
            interview_mode=interview_mode,
        )
        try:
            normalized = SceneTurnResult.model_validate(result)
        except Exception as exc:
            log_event("scene.turn.invalid_payload", error=str(exc))
            normalized = SceneTurnResult(
                engineer_message="本轮生成失败，请重试。",
                should_end=False,
                is_passed=False,
                score=0,
            )

        # 无模型可用时不进入“继续追问”循环，直接结束并提示配置。
        if normalized.assessment == "模型不可用" or any(
            "未配置可用模型" in item for item in (normalized.weaknesses or [])
        ):
            return normalized.model_copy(update={"should_end": True, "is_passed": False})

        # 只在“明确通过”时结束；未通过时持续追问。
        if not normalized.is_passed:
            message = normalized.engineer_message.strip()
            if "？" not in message and "?" not in message:
                message = (
                    f"{message} 请继续说明你的核心设计权衡、故障兜底和可观测性方案。"
                ).strip()
            normalized = normalized.model_copy(
                update={
                    "should_end": False,
                    "assessment": normalized.assessment or "尚未通过，继续追问。",
                    "engineer_message": message,
                }
            )
        else:
            normalized = normalized.model_copy(update={"should_end": True})
        return normalized


def _build_auxiliary_engine(strict_ai_generation: bool) -> QuizEngine:
    engine = QuizEngine()
    engine.strict_ai_generation = strict_ai_generation
    return engine


def _build_focused_parsed_content(
    parsed: ParsedContent,
    focus_points: list,
    title_suffix: str,
) -> ParsedContent:
    return ParsedContent(
        title=f"{parsed.title} - {title_suffix}",
        source_type=parsed.source_type,
        cleaned_text=parsed.cleaned_text,
        segments=[point.summary for point in focus_points],
        knowledge_points=focus_points,
        concepts=[point.name for point in focus_points],
    )


def _select_focus_points(
    parsed: ParsedContent,
    focus_topics: list[str],
    fallback_count: int,
) -> list:
    focus_set = {str(item).strip() for item in focus_topics if str(item).strip()}
    if focus_set:
        focus_points = [
            point
            for point in parsed.knowledge_points
            if point.name in focus_set
            or any(keyword in focus_set for keyword in point.keywords)
        ]
        if focus_points:
            return focus_points
    return parsed.knowledge_points[:fallback_count]


def build_reinforcement_quiz(
    parsed: ParsedContent,
    report: FeedbackReport,
    config: QuizConfig,
    learning_style: str = "teacher",
    strict_ai_generation: bool = True,
) -> Quiz:
    focus_points = _select_focus_points(parsed, report.reinforcement_topics, fallback_count=3)
    focused = _build_focused_parsed_content(parsed, focus_points, "强化训练")
    engine = _build_auxiliary_engine(strict_ai_generation)
    return engine.generate_quiz(
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
        learning_style=learning_style,
    )


def build_targeted_quiz(
    parsed: ParsedContent,
    focus_topics: list[str],
    config: QuizConfig,
    question_count: int | None = None,
    allow_ai_generation: bool = True,
    difficulty_mix: dict[str, int] | None = None,
    type_mix: dict[str, int] | None = None,
    learning_style: str = "teacher",
    strict_ai_generation: bool = True,
) -> Quiz:
    focus_points = _select_focus_points(parsed, focus_topics, fallback_count=4)
    focused = _build_focused_parsed_content(parsed, focus_points, "定向练习")
    total = question_count or min(max(6, len(focus_points) * 3), config.question_count + 4)
    targeted_config = QuizConfig(
        question_count=max(5, min(30, total)),
        difficulty_mix=difficulty_mix or {"easy": 25, "medium": 50, "hard": 25},
        type_mix=type_mix
        or {
            "single_choice": 35,
            "multiple_choice": 25,
            "fill_blank": 20,
            "short_answer": 15,
            "true_false": 5,
        },
    )
    engine = _build_auxiliary_engine(strict_ai_generation)
    return engine.generate_quiz(
        focused,
        targeted_config,
        allow_ai_generation=allow_ai_generation,
        learning_style=learning_style,
    )

