from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quizmind.cache import JsonFileCache
from quizmind.content import fallback_parse_content
from quizmind.logger import log_event, timed_event
from quizmind.prompt_center import (
    ENGINEER_SCENE_INTERVIEW_PROMPT,
    ENGINEER_SCENE_MODE_GUIDED_PROMPT,
    ENGINEER_SCENE_MODE_STRICT_PROMPT,
    SOURCE_CHUNK_SUMMARY_PROMPT,
    SOURCE_MERGE_SUMMARY_PROMPT,
    generate_from_source_system_prompt,
    generate_quiz_guidance,
    generate_quiz_system_prompt,
    grade_batch_system_prompt,
    parse_content_system_prompt,
    quiz_quality_review_system_prompt,
)
from quizmind.models import (
    BatchSubjectiveGrade,
    Difficulty,
    ParsedContent,
    Question,
    QuestionType,
    Quiz,
    QuizConfig,
    SceneTurnResult,
)


load_dotenv()

TRUE_LABEL = "\u6b63\u786e"
FALSE_LABEL = "\u9519\u8bef"

PARSE_CONTENT_SYSTEM_PROMPT = parse_content_system_prompt()
GENERATE_QUIZ_SYSTEM_PROMPT = generate_quiz_system_prompt()
GENERATE_FROM_SOURCE_SYSTEM_PROMPT = generate_from_source_system_prompt()


class LangChainQuizProvider:
    def __init__(self) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3.2")
        raw_fallback_models = os.getenv(
            "QUIZMIND_FALLBACK_MODELS",
            "Pro/zai-org/GLM-5.1,Qwen/Qwen3.5-397B-A17B",
        )
        self.fallback_models = [
            item.strip()
            for item in raw_fallback_models.split(",")
            if item.strip() and item.strip() != self.model
        ]
        self.fast_mode = self._safe_bool_env("QUIZMIND_FAST_MODE", False)
        self.quality_first = self._safe_bool_env("QUIZMIND_QUALITY_FIRST", True)
        self.ai_quality_review = self._safe_bool_env("QUIZMIND_AI_QUALITY_REVIEW", True)
        self.ai_quality_threshold = max(
            40, min(95, self._safe_int_env("QUIZMIND_AI_QUALITY_THRESHOLD", 70))
        )
        # Keep failures fast: long upstream timeouts are worse than quick fallback for this app.
        self.llm_timeout = max(8, min(120, self._safe_int_env("QUIZMIND_LLM_TIMEOUT", 60)))
        self.llm_max_retries = max(0, min(3, self._safe_int_env("QUIZMIND_LLM_MAX_RETRIES", 2)))
        default_attempts = 1 if self.fast_mode else 2
        self.quiz_gen_attempts = max(
            1,
            min(6, self._safe_int_env("QUIZMIND_QUIZ_GEN_ATTEMPTS", default_attempts)),
        )
        self.source_char_limit = max(0, self._safe_int_env("QUIZMIND_SOURCE_CHAR_LIMIT", 2500))
        self.source_use_summary = self._safe_bool_env("QUIZMIND_SOURCE_USE_SUMMARY", False)
        self.source_context_max_chars = max(
            1200, min(5000, self._safe_int_env("QUIZMIND_SOURCE_CONTEXT_MAX_CHARS", 2500))
        )
        self.source_summary_chunk_size = max(
            1200, self._safe_int_env("QUIZMIND_SOURCE_SUMMARY_CHUNK_SIZE", 3200)
        )
        self.source_summary_max_chunks = max(
            2, min(24, self._safe_int_env("QUIZMIND_SOURCE_SUMMARY_MAX_CHUNKS", 10))
        )
        self.source_summary_concurrency = max(
            1, min(8, self._safe_int_env("QUIZMIND_SOURCE_SUMMARY_CONCURRENCY", 2))
        )
        self.max_output_tokens = max(
            512, min(4096, self._safe_int_env("QUIZMIND_MAX_OUTPUT_TOKENS", 2200))
        )
        self.generation_source_context_max_chars = max(
            800,
            min(
                3000,
                self._safe_int_env(
                    "QUIZMIND_GENERATION_SOURCE_CONTEXT_MAX_CHARS", 1600
                ),
            ),
        )
        self.review_source_context_max_chars = max(
            1200,
            min(
                6000,
                self._safe_int_env(
                    "QUIZMIND_REVIEW_SOURCE_CONTEXT_MAX_CHARS", 3200
                ),
            ),
        )
        self.quiz_points_limit = max(
            3,
            min(
                12,
                self._safe_int_env(
                    "QUIZMIND_QUIZ_POINTS_LIMIT", 6 if self.fast_mode else 10
                ),
            ),
        )
        self.quiz_segments_limit = max(
            3,
            min(
                12,
                self._safe_int_env(
                    "QUIZMIND_QUIZ_SEGMENTS_LIMIT", 6 if self.fast_mode else 10
                ),
            ),
        )
        default_concurrency = 1
        self.quiz_gen_concurrency = max(
            1,
            min(
                6,
                self._safe_int_env(
                    "QUIZMIND_QUIZ_GEN_CONCURRENCY", default_concurrency
                ),
            ),
        )
        self.parallel_model_generation = self._safe_bool_env(
            "QUIZMIND_PARALLEL_MODEL_GENERATION", True
        )

        self.cache = JsonFileCache()
        self.llm: Optional[ChatOpenAI] = None
        self.llm_clients: list[tuple[str, ChatOpenAI]] = []
        self.last_used_model: str = self.model
        self.last_model_chain: list[str] = [self.model]
        self.model_cooldown_seconds = max(
            30, self._safe_int_env("QUIZMIND_MODEL_COOLDOWN_SECONDS", 60)
        )
        self.model_cooldowns: dict[str, float] = {}
        self.generation_model_priority = [
            item.strip()
            for item in (
                self.fallback_models + [self.model]
            )
            if item.strip()
        ]
        if api_key:
            self.llm = self._build_llm_client(self.model, api_key, base_url)
            self.llm_clients.append((self.model, self.llm))
            for fallback_model in self.fallback_models:
                self.llm_clients.append(
                    (fallback_model, self._build_llm_client(fallback_model, api_key, base_url))
                )

    def _build_llm_client(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
    ) -> ChatOpenAI:
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.4,
            timeout=self.llm_timeout,
            max_retries=self.llm_max_retries,
        )

    def _ordered_llm_clients(self) -> list[tuple[str, ChatOpenAI]]:
        now = time.time()
        return [
            (name, client)
            for name, client in self.llm_clients
            if self.model_cooldowns.get(name, 0.0) <= now
        ]

    def _clients_for_operation(self, operation: str) -> list[tuple[str, ChatOpenAI]]:
        clients = self._ordered_llm_clients()
        if not clients and self.llm_clients:
            rescue_model = self.model
            if operation.startswith("generate_quiz") or operation == "review_quiz_quality":
                rescue_model = self.model
            for name, client in self.llm_clients:
                if name == rescue_model:
                    return [(name, client)]
            return [self.llm_clients[0]]
        if not clients:
            return []
        if operation.startswith("generate_quiz") or operation == "review_quiz_quality":
            preferred_index = {
                name: idx for idx, name in enumerate(self.generation_model_priority)
            }
            clients.sort(key=lambda item: preferred_index.get(item[0], len(preferred_index)))
        return clients

    def _mark_model_timeout(self, model_name: str) -> None:
        self.model_cooldowns[model_name] = time.time() + self.model_cooldown_seconds
        log_event(
            "llm.model_cooldown",
            model=model_name,
            cooldown_seconds=self.model_cooldown_seconds,
        )

    def parse_content(self, source: str, source_type: str) -> ParsedContent:
        if not self.llm:
            return fallback_parse_content(source, source_type)

        payload = {
            "source_type": source_type,
            "content": self._prepare_source_content(source),
        }
        response = self._invoke_json(
            operation="parse_content",
            temperature=0.2,
            system_prompt=PARSE_CONTENT_SYSTEM_PROMPT,
            human_prompt="Please parse the following learning content:\n{payload}",
            payload=payload,
            use_cache=False,
        )
        if not response:
            return fallback_parse_content(source, source_type)

        try:
            normalized = self._normalize_parsed_payload(response, source_type)
            normalized["cleaned_text"] = source.strip()
            return ParsedContent.model_validate(normalized)
        except Exception as exc:
            log_event("parse_content.fallback", reason=str(exc))
            return fallback_parse_content(source, source_type)

    def generate_quiz(
        self,
        parsed: ParsedContent,
        config: QuizConfig,
        learning_style: str = "teacher",
    ) -> Optional[Quiz]:
        if not self.llm:
            return None

        payload = {
            "title": parsed.title,
            "knowledge_points": [
                point.model_dump()
                for point in parsed.knowledge_points[
                    : min(self.quiz_points_limit, max(4, config.question_count))
                ]
            ],
            "segments": parsed.segments[
                : min(self.quiz_segments_limit, max(4, config.question_count // 2 + 2))
            ],
            "source_context": self._build_source_context(
                parsed.cleaned_text,
                max_chars=self.generation_source_context_max_chars,
            ),
            "config": config.model_dump(),
        }

        if self.fast_mode:
            temperatures = [0.38, 0.45] if self.quality_first else [0.45]
        else:
            temperatures = (
                [0.35, 0.45, 0.3, 0.5, 0.4, 0.55]
                if self.quality_first
                else [0.45, 0.6, 0.35, 0.7, 0.5, 0.3]
            )
        best_quiz: Optional[Quiz] = None
        best_score = -1
        if self.quiz_gen_attempts <= 1 or self.quiz_gen_concurrency <= 1:
            for attempt in range(self.quiz_gen_attempts):
                candidate = self._generate_quiz_candidate(
                    parsed=parsed,
                    config=config,
                    payload=payload,
                    attempt=attempt,
                    temperatures=temperatures,
                    learning_style=learning_style,
                )
                if not candidate:
                    continue
                quiz, score = candidate
                if not self._quiz_is_grounded(quiz, parsed.cleaned_text):
                    continue
                if score > best_score:
                    best_quiz = quiz
                    best_score = score
                if self.fast_mode or self._quiz_quality_good(quiz, config):
                    return quiz
            return best_quiz

        max_workers = min(self.quiz_gen_attempts, self.quiz_gen_concurrency)
        futures: list[Future] = []
        executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="quiz-gen"
        )
        try:
            for attempt in range(self.quiz_gen_attempts):
                futures.append(
                    executor.submit(
                        self._generate_quiz_candidate,
                        parsed,
                        config,
                        payload,
                        attempt,
                        temperatures,
                        learning_style,
                    )
                )

            for future in as_completed(futures):
                candidate = future.result()
                if not candidate:
                    continue
                quiz, score = candidate
                if not self._quiz_is_grounded(quiz, parsed.cleaned_text):
                    continue
                if score > best_score:
                    best_quiz = quiz
                    best_score = score
                if self.fast_mode or self._quiz_quality_good(quiz, config):
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    return quiz
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return best_quiz

    def _generate_quiz_candidate(
        self,
        parsed: ParsedContent,
        config: QuizConfig,
        payload: dict,
        attempt: int,
        temperatures: list[float],
        learning_style: str = "teacher",
    ) -> tuple[Quiz, int] | None:
        response = self._invoke_json(
            operation=f"generate_quiz_try_{attempt + 1}",
            temperature=temperatures[attempt % len(temperatures)],
            system_prompt=GENERATE_QUIZ_SYSTEM_PROMPT,
            human_prompt=(
                "Generate a quiz from the parsed result below.\n"
                f"Additional requirements:\n{generate_quiz_guidance(TRUE_LABEL, FALSE_LABEL, learning_style)}\n"
                "{payload}"
            ),
            payload=payload,
            use_cache=False,
        )
        if not response:
            return None
        try:
            candidate = Quiz.model_validate(
                self._normalize_quiz_payload(response, parsed.title)
            )
        except Exception:
            return None
        return candidate, self._quiz_quality_score(candidate, config)

    def generate_quiz_from_source(
        self,
        source: str,
        source_type: str,
        config: QuizConfig,
        learning_style: str = "teacher",
    ) -> tuple[ParsedContent, Optional[Quiz]]:
        if not self.llm:
            parsed = fallback_parse_content(source, source_type)
            return parsed, None

        payload = {
            "source_type": source_type,
            "content": self._prepare_source_content(source),
            "source_context": self._build_source_context(source),
            "config": config.model_dump(),
        }
        response = self._invoke_json(
            operation="generate_quiz_from_source",
            temperature=0.5,
            system_prompt=GENERATE_FROM_SOURCE_SYSTEM_PROMPT,
            human_prompt=(
                "Process the raw content below and output parsed_content + quiz.\n"
                f"Additional requirements:\n{generate_quiz_guidance(TRUE_LABEL, FALSE_LABEL, learning_style)}\n"
                "{payload}"
            ),
            payload=payload,
            use_cache=False,
        )
        if not response:
            parsed = fallback_parse_content(source, source_type)
            return parsed, None

        try:
            parsed_data = self._normalize_parsed_payload(
                response.get("parsed_content", {}), source_type
            )
            parsed_data["cleaned_text"] = source.strip()
            quiz_data = self._normalize_quiz_payload(
                response.get("quiz", {}),
                parsed_data.get("title", "Quiz"),
            )
            parsed_obj = ParsedContent.model_validate(parsed_data)
            quiz_obj = Quiz.model_validate(quiz_data)

            if not self._quiz_is_grounded(quiz_obj, parsed_obj.cleaned_text):
                improved = self.generate_quiz(parsed_obj, config, learning_style)
                return parsed_obj, improved
            if self.fast_mode:
                return parsed_obj, quiz_obj
            if self._quiz_quality_good(quiz_obj, config):
                return parsed_obj, quiz_obj

            improved = self.generate_quiz(parsed_obj, config, learning_style)
            return parsed_obj, (improved or quiz_obj)
        except Exception as exc:
            log_event("generate_quiz_from_source.fallback", reason=str(exc))
            parsed = fallback_parse_content(source, source_type)
            return parsed, self.generate_quiz(parsed, config, learning_style)

    def grade_subjective_batch(
        self,
        questions_and_answers: list[tuple[Question, List[str]]],
        learning_style: str = "teacher",
    ) -> dict[str, BatchSubjectiveGrade]:
        if not self.llm or not questions_and_answers:
            return {}

        payload = {
            "items": [
                {
                    "question_id": question.id,
                    "question_type": question.question_type.value,
                    "prompt": question.prompt,
                    "options": question.options,
                    "correct_answer": question.correct_answer,
                    "reference_points": question.reference_points,
                    "user_answer": answer,
                }
                for question, answer in questions_and_answers
            ]
        }

        response = self._invoke_json(
            operation="grade_subjective_batch",
            temperature=0.1,
            system_prompt=grade_batch_system_prompt(learning_style),
            human_prompt="Grade the following answers in batch (objective and subjective):\n{payload}",
            payload=payload,
            use_cache=False,
        )
        if not response:
            return {}

        grades = response.get("grades", [])
        if not isinstance(grades, list):
            grades = []
        result: dict[str, BatchSubjectiveGrade] = {}
        for item in grades:
            if not isinstance(item, dict):
                continue
            try:
                score_value = float(item.get("score", 0))
            except (TypeError, ValueError):
                score_value = 0.0
            raw_breakdown = item.get("score_breakdown", {})
            breakdown: dict[str, float] = {}
            if isinstance(raw_breakdown, dict):
                for key in ("correctness", "completeness", "clarity"):
                    try:
                        breakdown[key] = float(raw_breakdown.get(key, 0))
                    except (TypeError, ValueError):
                        breakdown[key] = 0.0
            grade = BatchSubjectiveGrade.model_validate(
                {
                    "question_id": str(item.get("question_id", "")),
                    "score": score_value,
                    "feedback": str(item.get("feedback", "")),
                    "missing_points": [
                        str(point) for point in item.get("missing_points", [])
                    ],
                    "error_category": str(item.get("error_category", "")),
                    "score_breakdown": breakdown,
                    "structured_explanation": str(
                        item.get("structured_explanation", "")
                    ),
                }
            )
            result[grade.question_id] = grade
        return result

    def review_quiz_quality(
        self,
        quiz: Quiz,
        source_text: str,
        config: QuizConfig,
    ) -> dict[str, Any]:
        if not self.llm:
            return {"pass": True, "overall_score": 100.0, "issues": [], "summary": "LLM unavailable"}

        payload = {
            "config": config.model_dump(),
            "source_context": self._build_source_context(
                source_text,
                max_chars=self.review_source_context_max_chars,
            ),
            "quiz": quiz.model_dump(),
        }
        response = self._invoke_json(
            operation="review_quiz_quality",
            temperature=0.05,
            system_prompt=quiz_quality_review_system_prompt(),
            human_prompt=(
                "Audit the quiz quality against source and config.\n"
                "Return strict quality review JSON only.\n"
                "{payload}"
            ),
            payload=payload,
            use_cache=False,
        )
        if not isinstance(response, dict):
            return {"pass": True, "overall_score": 100.0, "issues": [], "summary": "review skipped"}

        issues = response.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        normalized_issues = []
        for item in issues:
            if not isinstance(item, dict):
                continue
            normalized_issues.append(
                {
                    "question_id": str(item.get("question_id", "")),
                    "severity": str(item.get("severity", "low")).lower(),
                    "category": str(item.get("category", "")),
                    "message": str(item.get("message", "")).strip(),
                }
            )
        try:
            overall_score = float(response.get("overall_score", 0))
        except (TypeError, ValueError):
            overall_score = 0.0

        has_high_issue = any(x.get("severity") == "high" for x in normalized_issues)
        passed = bool(response.get("pass", False))
        if overall_score < self.ai_quality_threshold or has_high_issue:
            passed = False

        return {
            "pass": passed,
            "overall_score": overall_score,
            "issues": normalized_issues,
            "summary": str(response.get("summary", "")),
        }

    def run_engineer_scene_turn(
        self,
        scene_description: str,
        transcript: list[dict[str, str]],
        max_rounds: int = 12,
        interview_mode: str = "guided",
    ) -> dict:
        if not self.llm:
            return SceneTurnResult(
                engineer_message="当前未配置可用大模型，无法启动场景拷问模式。",
                should_end=True,
                is_passed=False,
                score=0,
                assessment="模型不可用",
                strengths=[],
                weaknesses=["未配置可用模型"],
                recommendations=["请先配置 OPENAI_API_KEY 或 SILICONFLOW_API_KEY。"],
            ).model_dump()

        normalized_scene = (scene_description or "").strip()
        mode = str(interview_mode or "guided").strip().lower()
        if mode not in {"guided", "strict"}:
            mode = "guided"
        mode_prompt = (
            ENGINEER_SCENE_MODE_STRICT_PROMPT
            if mode == "strict"
            else ENGINEER_SCENE_MODE_GUIDED_PROMPT
        )
        turns = transcript[-24:] if isinstance(transcript, list) else []
        payload = {
            "scene_description": normalized_scene,
            "interview_mode": mode,
            "transcript": turns,
            # Internal guard only; end decision is based on pass status.
            "max_rounds": max_rounds,
            "current_round": max(0, len([x for x in turns if x.get("role") == "engineer"])),
        }
        response = self._invoke_json(
            operation="engineer_scene_turn",
            temperature=0.35,
            system_prompt=f"{ENGINEER_SCENE_INTERVIEW_PROMPT}\n{mode_prompt}",
            human_prompt=(
                "Based on the scenario and transcript below, produce the next interviewer turn.\n"
                "If transcript is empty, start the interview with a concise opening and first question.\n"
                "Only end when the candidate is clearly passed.\n"
                "{payload}"
            ),
            payload=payload,
            use_cache=False,
        )
        if not isinstance(response, dict):
            return SceneTurnResult(
                engineer_message="本轮生成失败，请重试。",
                should_end=False,
                is_passed=False,
                score=0,
                assessment="",
                strengths=[],
                weaknesses=[],
                recommendations=[],
            ).model_dump()

        raw_message = str(response.get("engineer_message", "")).strip()
        should_end = bool(response.get("should_end", False))
        is_passed = bool(response.get("is_passed", False))
        if not is_passed:
            should_end = False
        if should_end and is_passed and "通过" not in raw_message and "认可" not in raw_message:
            raw_message = f"{raw_message} 我认可你通过了这个场景。".strip()

        try:
            score = float(response.get("score", 0))
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))

        result = SceneTurnResult(
            engineer_message=raw_message,
            should_end=should_end,
            is_passed=is_passed,
            score=score,
            assessment=str(response.get("assessment", "")).strip(),
            strengths=[
                str(x).strip()
                for x in response.get("strengths", [])
                if str(x).strip()
            ],
            weaknesses=[
                str(x).strip()
                for x in response.get("weaknesses", [])
                if str(x).strip()
            ],
            recommendations=[
                str(x).strip()
                for x in response.get("recommendations", [])
                if str(x).strip()
            ],
        )
        return result.model_dump()

    def _invoke_json(
        self,
        operation: str,
        temperature: float,
        system_prompt: str,
        human_prompt: str,
        payload: dict,
        use_cache: bool = True,
    ) -> dict | None:
        clients = self._clients_for_operation(operation)
        if not clients:
            return None
        self.last_used_model = clients[0][0]
        self.last_model_chain = [name for name, _ in clients]

        cache_key = self.cache.build_key(
            operation,
            {
                "models": [name for name, _ in clients],
                "temperature": temperature,
                "payload": payload,
            },
        )
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                log_event("llm.cache_hit", operation=operation, key=cache_key)
                return cached

        log_event("llm.cache_miss", operation=operation, key=cache_key)
        parser = JsonOutputParser()
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            include_format_instructions=True,
        )
        message_vars = {
            "payload": json.dumps(payload, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
        }

        fallback_prompt = self._build_prompt(
            system_prompt=system_prompt,
            human_prompt="{format_instructions}\n\n" + human_prompt,
            include_format_instructions=False,
        )
        fast_fail_on_timeout = operation == "parse_content"
        parallel_models = (
            self.parallel_model_generation
            and len(clients) > 1
            and (
                operation.startswith("generate_quiz")
                or operation == "review_quiz_quality"
            )
        )
        if parallel_models:
            return self._invoke_json_parallel(
                operation=operation,
                temperature=temperature,
                prompt=prompt,
                fallback_prompt=fallback_prompt,
                parser=parser,
                message_vars=message_vars,
                clients=clients,
                cache_key=cache_key,
                use_cache=use_cache,
            )
        last_error: Exception | None = None
        for index, (model_name, llm_client) in enumerate(clients):
            if index > 0:
                self.last_model_chain = [name for name, _ in clients[: index + 1]]
                log_event(
                    "llm.model_fallback",
                    operation=operation,
                    from_model=clients[index - 1][0],
                    to_model=model_name,
                )
            try:
                data = self._invoke_json_chain(
                    operation=operation,
                    temperature=temperature,
                    prompt=prompt,
                    parser=parser,
                    message_vars=message_vars,
                    llm_client=llm_client,
                    model_name=model_name,
                )
                if data is not None:
                    self.last_used_model = model_name
                    if use_cache:
                        self.cache.set(cache_key, data)
                    log_event("llm.invoke.success", operation=operation, model=model_name)
                    return data
            except Exception as exc:
                last_error = exc
                log_event(
                    "llm.invoke_json_chain_failed",
                    operation=operation,
                    model=model_name,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
                if self._is_timeout_like(exc):
                    self._mark_model_timeout(model_name)
                    if fast_fail_on_timeout:
                        break
                    continue

            try:
                fallback_text = self._invoke_text_chain(
                    operation=operation,
                    temperature=temperature,
                    prompt=fallback_prompt,
                    message_vars=message_vars,
                    llm_client=llm_client,
                    model_name=model_name,
                )
                if not fallback_text:
                    continue
                data = self._parse_json(fallback_text)
                self.last_used_model = model_name
                if use_cache:
                    self.cache.set(cache_key, data)
                log_event("llm.invoke.success", operation=operation, model=model_name)
                return data
            except Exception as exc:
                last_error = exc
                log_event(
                    "llm.invoke_failed",
                    operation=operation,
                    model=model_name,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
        if last_error:
            log_event(
                "llm.all_models_failed",
                operation=operation,
                model_chain=" -> ".join(name for name, _ in clients),
                error_type=last_error.__class__.__name__,
                error=str(last_error),
            )
        return None

    def _invoke_text(
        self,
        operation: str,
        temperature: float,
        system_prompt: str,
        human_prompt: str,
        payload: dict,
    ) -> Optional[str]:
        clients = self._clients_for_operation(operation)
        if not clients:
            return None
        self.last_used_model = clients[0][0]
        self.last_model_chain = [name for name, _ in clients]

        cache_key = self.cache.build_key(
            operation,
            {
                "models": [name for name, _ in clients],
                "temperature": temperature,
                "payload": payload,
            },
        )
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            log_event("llm.cache_hit", operation=operation, key=cache_key)
            return str(cached["text"])

        log_event("llm.cache_miss", operation=operation, key=cache_key)
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            include_format_instructions=False,
        )
        last_error: Exception | None = None
        for index, (model_name, llm_client) in enumerate(clients):
            if index > 0:
                self.last_model_chain = [name for name, _ in clients[: index + 1]]
                log_event(
                    "llm.model_fallback",
                    operation=operation,
                    from_model=clients[index - 1][0],
                    to_model=model_name,
                )
            try:
                text = self._invoke_text_chain(
                    operation=operation,
                    temperature=temperature,
                    prompt=prompt,
                    message_vars={"payload": json.dumps(payload, ensure_ascii=False)},
                    llm_client=llm_client,
                    model_name=model_name,
                )
                if text is None:
                    continue
                text = text.strip()

                if text.startswith("```"):
                    text = re.sub(r"^```(?:html)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                self.last_used_model = model_name
                self.cache.set(cache_key, {"text": text})
                log_event("llm.invoke.success", operation=operation, model=model_name)
                return text
            except Exception as exc:
                last_error = exc
                if self._is_timeout_like(exc):
                    self._mark_model_timeout(model_name)
                log_event(
                    "llm.invoke_failed",
                    operation=operation,
                    model=model_name,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
        if last_error:
            log_event(
                "llm.all_models_failed",
                operation=operation,
                model_chain=" -> ".join(name for name, _ in clients),
                error_type=last_error.__class__.__name__,
                error=str(last_error),
            )
        return None

    def _invoke_json_parallel(
        self,
        operation: str,
        temperature: float,
        prompt: ChatPromptTemplate,
        fallback_prompt: ChatPromptTemplate,
        parser: JsonOutputParser,
        message_vars: dict[str, Any],
        clients: list[tuple[str, ChatOpenAI]],
        cache_key: str,
        use_cache: bool,
    ) -> dict | None:
        log_event(
            "llm.parallel_models_start",
            operation=operation,
            model_chain=" -> ".join(name for name, _ in clients),
        )
        last_error: Exception | None = None
        pool = ThreadPoolExecutor(
            max_workers=len(clients), thread_name_prefix="llm-parallel"
        )
        try:
            future_map = {
                pool.submit(
                    self._invoke_json_model_worker,
                    operation,
                    temperature,
                    prompt,
                    fallback_prompt,
                    parser,
                    message_vars,
                    llm_client,
                    model_name,
                ): model_name
                for model_name, llm_client in clients
            }
            for future in as_completed(future_map):
                model_name = future_map[future]
                try:
                    data = future.result()
                except Exception as exc:
                    last_error = exc
                    if self._is_timeout_like(exc):
                        self._mark_model_timeout(model_name)
                    log_event(
                        "llm.parallel_model_failed",
                        operation=operation,
                        model=model_name,
                        error_type=exc.__class__.__name__,
                        error=str(exc),
                    )
                    continue
                if not data:
                    continue
                self.last_used_model = model_name
                self.last_model_chain = [name for name, _ in clients]
                if use_cache:
                    self.cache.set(cache_key, data)
                log_event("llm.invoke.success", operation=operation, model=model_name)
                for pending in future_map:
                    if not pending.done():
                        pending.cancel()
                return data
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if last_error:
            log_event(
                "llm.all_models_failed",
                operation=operation,
                model_chain=" -> ".join(name for name, _ in clients),
                error_type=last_error.__class__.__name__,
                error=str(last_error),
            )
        return None

    def _invoke_json_model_worker(
        self,
        operation: str,
        temperature: float,
        prompt: ChatPromptTemplate,
        fallback_prompt: ChatPromptTemplate,
        parser: JsonOutputParser,
        message_vars: dict[str, Any],
        llm_client: ChatOpenAI,
        model_name: str,
    ) -> dict | None:
        try:
            return self._invoke_json_chain(
                operation=operation,
                temperature=temperature,
                prompt=prompt,
                parser=parser,
                message_vars=message_vars,
                llm_client=llm_client,
                model_name=model_name,
            )
        except Exception as exc:
            log_event(
                "llm.invoke_json_chain_failed",
                operation=operation,
                model=model_name,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            if self._is_timeout_like(exc):
                raise
        fallback_text = self._invoke_text_chain(
            operation=operation,
            temperature=temperature,
            prompt=fallback_prompt,
            message_vars=message_vars,
            llm_client=llm_client,
            model_name=model_name,
        )
        if not fallback_text:
            return None
        return self._parse_json(fallback_text)

    def _build_prompt(
        self,
        system_prompt: str,
        human_prompt: str,
        include_format_instructions: bool,
    ) -> ChatPromptTemplate:
        safe_system_prompt = self._escape_prompt_template(system_prompt)
        safe_human_prompt = self._escape_prompt_template(human_prompt)
        if include_format_instructions:
            human = "{format_instructions}\n\n" + safe_human_prompt
        else:
            human = safe_human_prompt
        return ChatPromptTemplate.from_messages(
            [("system", safe_system_prompt), ("human", human)]
        )

    @staticmethod
    def _escape_prompt_template(
        text: str,
        allowed_vars: tuple[str, ...] = ("payload", "format_instructions"),
    ) -> str:
        # Escape all braces so JSON examples in prompts are treated as plain text.
        escaped = text.replace("{", "{{").replace("}", "}}")
        # Re-enable only the placeholders we intentionally pass to ChatPromptTemplate.
        for var in allowed_vars:
            escaped = escaped.replace(f"{{{{{var}}}}}", f"{{{var}}}")
        return escaped

    def _invoke_json_chain(
        self,
        operation: str,
        temperature: float,
        prompt: ChatPromptTemplate,
        parser: JsonOutputParser,
        message_vars: dict[str, Any],
        llm_client: ChatOpenAI,
        model_name: str,
    ) -> dict | None:
        chain = (
            prompt
            | llm_client.bind(**self._llm_bind_kwargs(temperature))
            | parser
        )
        with timed_event("llm.invoke", operation=operation, model=model_name):
            parsed = chain.invoke(message_vars)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return None

    def _invoke_text_chain(
        self,
        operation: str,
        temperature: float,
        prompt: ChatPromptTemplate,
        message_vars: dict[str, Any],
        llm_client: ChatOpenAI,
        model_name: str,
    ) -> str | None:
        chain = (
            prompt
            | llm_client.bind(**self._llm_bind_kwargs(temperature))
            | StrOutputParser()
        )
        with timed_event("llm.invoke", operation=operation, model=model_name):
            return str(chain.invoke(message_vars))

    def _parse_json(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
            raise json.JSONDecodeError("Unsupported JSON root type", cleaned, 0)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(cleaned):
                if char not in "{[":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
                if isinstance(candidate, list):
                    return {"items": candidate}
            raise

    def _normalize_parsed_payload(self, data: dict, source_type: str) -> dict:
        points = []
        for item in data.get("knowledge_points", []):
            if isinstance(item, str):
                points.append(
                    {
                        "name": item,
                        "summary": item,
                        "importance": 3,
                        "difficulty": Difficulty.medium.value,
                        "keywords": [item],
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            points.append(
                {
                    "name": str(item.get("name", "Untitled Knowledge Point")),
                    "summary": str(item.get("summary", "")),
                    "importance": self._normalize_importance(item.get("importance", 3)),
                    "difficulty": Difficulty.normalize(item.get("difficulty")),
                    "keywords": self._normalize_keywords(item.get("keywords", [])),
                }
            )

        segments = self._normalize_segments(data.get("segments", []))
        concepts = self._normalize_concepts(data.get("concepts", []))
        if not concepts:
            concepts = [point["name"] for point in points[:10]]

        cleaned_text = str(data.get("cleaned_text", "")).strip()
        if not cleaned_text and segments:
            cleaned_text = "\n".join(segments)

        return {
            "title": str(data.get("title", "Untitled Content")),
            "source_type": source_type,
            "cleaned_text": cleaned_text,
            "segments": segments,
            "knowledge_points": points,
            "concepts": concepts,
        }

    def _normalize_quiz_payload(self, data: dict, fallback_title: str) -> dict:
        questions = []
        raw_questions = data.get("questions")
        if not isinstance(raw_questions, list):
            items = data.get("items")
            raw_questions = items if isinstance(items, list) else []

        for index, question in enumerate(raw_questions, start=1):
            if not isinstance(question, dict):
                continue
            normalized = dict(question)
            normalized["id"] = str(normalized.get("id", f"Q{index:03d}"))

            raw_qtype = (
                normalized.get("question_type")
                or normalized.get("type")
                or normalized.get("kind")
                or normalized.get("questionType")
                or normalized.get("\u9898\u578b")
            )
            normalized["question_type"] = QuestionType.normalize(raw_qtype)
            normalized["difficulty"] = Difficulty.normalize(
                normalized.get("difficulty")
            )

            opts = normalized.get("options", [])
            normalized["options"] = (
                [str(o) for o in opts]
                if isinstance(opts, list)
                else ([str(opts)] if opts else [])
            )

            answer_value = (
                question.get("correct_answer")
                if "correct_answer" in question
                else question.get("answer")
            )
            tags = normalized.get("knowledge_tags", [])
            refs = normalized.get("reference_points", [])
            normalized["knowledge_tags"] = (
                [str(t) for t in tags if str(t).strip()]
                if isinstance(tags, list)
                else ([str(tags)] if tags is not None and str(tags).strip() else [])
            )
            normalized["reference_points"] = (
                [str(r) for r in refs if str(r).strip()]
                if isinstance(refs, list)
                else ([str(refs)] if refs is not None and str(refs).strip() else [])
            )
            normalized["prompt"] = str(
                normalized.get("prompt")
                or normalized.get("question")
                or normalized.get("stem")
                or ""
            )
            normalized["explanation"] = str(
                normalized.get("explanation")
                or normalized.get("analysis")
                or normalized.get("reason")
                or ""
            )
            normalized["correct_answer"] = self._normalize_correct_answer(
                answer_value,
                normalized["options"],
                normalized["question_type"],
            )

            if (
                normalized["question_type"] == QuestionType.true_false.value
                and not normalized["options"]
            ):
                normalized["options"] = [TRUE_LABEL, FALSE_LABEL]

            questions.append(normalized)

        return {
            "title": str(data.get("title", f"{fallback_title} - Quiz Practice")),
            "source_summary": str(data.get("source_summary", fallback_title)),
            "questions": questions,
        }

    def _safe_int_env(self, key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    def _safe_bool_env(self, key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _is_timeout_like(exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        text = str(exc).lower()
        return "timeout" in name or "timed out" in text or "timeout" in text

    def _llm_bind_kwargs(self, temperature: float) -> dict:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if self.max_output_tokens > 0:
            kwargs["max_tokens"] = self.max_output_tokens
        return kwargs

    def _prepare_source_content(self, source: str) -> str:
        cleaned = source.strip()
        if self.source_char_limit <= 0 or len(cleaned) <= self.source_char_limit:
            return cleaned

        if self.source_use_summary:
            summarized = self._summarize_source_content(cleaned)
            if summarized:
                return summarized

        head_len = int(self.source_char_limit * 0.65)
        tail_len = self.source_char_limit - head_len
        omitted = len(cleaned) - head_len - tail_len
        return (
            f"{cleaned[:head_len]}\n\n"
            f"[...middle part omitted {omitted} chars for context budgeting...]\n\n"
            f"{cleaned[-tail_len:]}"
        )

    def _summarize_source_content(self, source: str) -> str | None:
        if not self.llm_clients or self.source_char_limit <= 0:
            return None

        chunks = self._split_for_summary(
            source,
            chunk_size=self.source_summary_chunk_size,
            max_chunks=self.source_summary_max_chunks,
        )
        if not chunks:
            return None

        if len(chunks) == 1:
            single = self._summarize_chunk(chunks[0], 1, 1)
            return self._fit_to_limit(single or "", self.source_char_limit)

        summaries: list[str] = [""] * len(chunks)
        workers = min(self.source_summary_concurrency, len(chunks))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="source-sum"
        ) as pool:
            future_map: dict[Future, int] = {}
            for idx, chunk in enumerate(chunks, start=1):
                future = pool.submit(self._summarize_chunk, chunk, idx, len(chunks))
                future_map[future] = idx - 1
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    text = future.result()
                except Exception:
                    text = ""
                summaries[index] = (text or chunks[index]).strip()

        merged_input = "\n\n".join(
            f"[Part {i+1}]\n{item}" for i, item in enumerate(summaries) if item.strip()
        )
        if not merged_input.strip():
            return None

        merged = self._invoke_text(
            operation="source_summary_merge",
            temperature=0.1,
            system_prompt=SOURCE_MERGE_SUMMARY_PROMPT,
            human_prompt=(
                "Merge these summaries into one source for downstream parsing.\n"
                f"Target length: <= {self.source_char_limit} characters.\n"
                "{payload}"
            ),
            payload={"summaries": merged_input},
        )
        final_text = (merged or merged_input).strip()
        return self._fit_to_limit(final_text, self.source_char_limit)

    def _summarize_chunk(self, chunk: str, index: int, total: int) -> str:
        text = self._invoke_text(
            operation=f"source_chunk_summary_{index}",
            temperature=0.1,
            system_prompt=SOURCE_CHUNK_SUMMARY_PROMPT,
            human_prompt=(
                f"Summarize chunk {index}/{total} for later parsing.\n"
                "Keep facts and key details; avoid fluff.\n"
                "{payload}"
            ),
            payload={"chunk": chunk},
        )
        return (text or "").strip()

    def _split_for_summary(
        self, text: str, chunk_size: int, max_chunks: int
    ) -> list[str]:
        parts = [seg.strip() for seg in re.split(r"\n{2,}", text) if seg.strip()]
        if not parts:
            parts = [text]

        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 <= chunk_size:
                current = f"{current}\n\n{part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)

        if not chunks:
            return []

        if len(chunks) > max_chunks:
            # Preserve all content by merging adjacent chunks instead of dropping chunks.
            merged: list[str] = []
            group_size = (len(chunks) + max_chunks - 1) // max_chunks
            for i in range(0, len(chunks), group_size):
                merged.append("\n\n".join(chunks[i : i + group_size]))
            chunks = merged[:max_chunks]
        return chunks

    def _fit_to_limit(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit]

    def _build_source_context(self, source: str, max_chars: int | None = None) -> str:
        if max_chars is None:
            max_chars = self.source_context_max_chars
        text = (source or "").strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text

        # Keep multi-window slices to avoid overfitting on head-only context.
        windows = 4
        window_size = max(400, max_chars // windows)
        step = max(1, (len(text) - window_size) // max(1, windows - 1))
        chunks: list[str] = []
        for i in range(windows):
            start = i * step
            end = min(len(text), start + window_size)
            snippet = text[start:end].strip()
            if snippet:
                chunks.append(snippet)
        merged = "\n\n".join(chunks)
        return merged[:max_chars]

    def _quiz_is_grounded(self, quiz: Quiz, source_text: str) -> bool:
        source = (source_text or "").strip()
        if not source:
            return False

        normalized_source = self._normalize_match_text(source)
        grounded = 0
        for question in quiz.questions:
            anchors: list[str] = []
            anchors.extend(question.reference_points or [])
            anchors.extend(question.knowledge_tags or [])
            anchors.extend(
                question.correct_answer[:2] if question.correct_answer else []
            )
            anchors = [a.strip() for a in anchors if a and a.strip()]
            if not anchors:
                continue

            matched = False
            for anchor in anchors:
                if len(anchor) < 2:
                    continue
                if anchor in source:
                    matched = True
                    break
                normalized_anchor = self._normalize_match_text(anchor)
                if normalized_anchor and normalized_anchor in normalized_source:
                    matched = True
                    break
            if matched:
                grounded += 1

        min_required = max(1, int(len(quiz.questions) * 0.6))
        return grounded >= min_required

    def _normalize_match_text(self, text: str) -> str:
        lowered = text.lower()
        return re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)

    def _normalize_importance(self, value: Any) -> int:
        if isinstance(value, bool):
            return 3
        if isinstance(value, int):
            return max(1, min(5, value))
        if isinstance(value, float):
            return max(1, min(5, int(round(value))))

        text = str(value or "").strip().lower()
        if not text:
            return 3

        aliases = {
            "low": 2,
            "medium": 3,
            "high": 5,
            "very high": 5,
            "very low": 1,
            "\u4f4e": 2,
            "\u8f83\u4f4e": 2,
            "\u4e2d": 3,
            "\u4e2d\u7b49": 3,
            "\u4e00\u822c": 3,
            "\u9ad8": 5,
            "\u8f83\u9ad8": 4,
            "\u5f88\u9ad8": 5,
            "\u91cd\u8981": 4,
            "\u975e\u5e38\u91cd\u8981": 5,
            "\u6b21\u8981": 2,
        }
        if text in aliases:
            return aliases[text]

        digit = re.search(r"[1-5]", text)
        if digit:
            return int(digit.group(0))

        cn_digit_map = {"\u4e00": 1, "\u4e8c": 2, "\u4e09": 3, "\u56db": 4, "\u4e94": 5}
        for key, mapped in cn_digit_map.items():
            if key in text:
                return mapped
        return 3

    def _normalize_keywords(self, raw: Any) -> List[str]:
        if isinstance(raw, list):
            values = raw
        elif isinstance(raw, str):
            values = re.split(r"[,;\\s]+", raw)
        else:
            values = []
        out = [str(item).strip() for item in values if str(item).strip()]
        return out[:8]

    def _normalize_segments(self, raw: Any) -> List[str]:
        if isinstance(raw, str):
            values: list[Any] = [raw]
        elif isinstance(raw, list):
            values = raw
        elif isinstance(raw, dict):
            values = list(raw.values())
        else:
            values = []

        segments: List[str] = []
        for item in values:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(
                    item.get("content")
                    or item.get("text")
                    or item.get("summary")
                    or item.get("reference")
                    or ""
                ).strip()
            else:
                text = str(item).strip()
            if text:
                segments.append(text)
        return segments[:50]

    def _normalize_concepts(self, raw: Any) -> List[str]:
        if isinstance(raw, dict):
            values = raw.keys()
        elif isinstance(raw, list):
            values = raw
        elif isinstance(raw, str):
            values = re.split(r"[,;\\n]+", raw)
        else:
            values = []
        concepts = [str(item).strip() for item in values if str(item).strip()]
        return concepts[:20]

    def _normalize_correct_answer(
        self,
        raw: Any,
        options: List[str],
        question_type: str,
    ) -> List[str]:
        def _map_letter_to_option(text: str) -> str:
            candidate = text.strip()
            if not re.fullmatch(r"[A-Za-z]", candidate):
                return candidate
            idx = ord(candidate.upper()) - ord("A")
            if 0 <= idx < len(options):
                return str(options[idx]).strip()
            return candidate

        if isinstance(raw, list):
            values = [str(item).strip() for item in raw if str(item).strip()]
        elif raw is None:
            values = []
        else:
            values = [str(raw).strip()]

        lowered = [value.lower() for value in values]
        if question_type == QuestionType.true_false.value:
            mapped: List[str] = []
            for value in lowered:
                if value in {
                    "true",
                    "t",
                    "1",
                    "yes",
                    "y",
                    "\u6b63\u786e",
                    "\u5bf9",
                    "\u662f",
                }:
                    mapped.append(TRUE_LABEL)
                elif value in {
                    "false",
                    "f",
                    "0",
                    "no",
                    "n",
                    "\u9519\u8bef",
                    "\u9519",
                    "\u5426",
                }:
                    mapped.append(FALSE_LABEL)
            if mapped:
                return list(dict.fromkeys(mapped))

        mapped_values = [_map_letter_to_option(value) for value in values]
        mapped_values = [value for value in mapped_values if value]
        return list(dict.fromkeys(mapped_values))

    def _distribution_targets(self, total: int, mix: dict[str, int]) -> dict[str, int]:
        counts = {name: max(0, int(total * ratio / 100)) for name, ratio in mix.items()}
        while sum(counts.values()) < total:
            for name in sorted(mix, key=mix.get, reverse=True):
                counts[name] = counts.get(name, 0) + 1
                if sum(counts.values()) >= total:
                    break
        return counts

    def _quiz_quality_score(self, quiz: Quiz, config: QuizConfig) -> int:
        targets = self._distribution_targets(config.question_count, config.type_mix)
        required = [name for name, count in targets.items() if count > 0]
        if not required:
            return 0

        actual = Counter(question.question_type.value for question in quiz.questions)
        coverage = sum(1 for name in required if actual.get(name, 0) > 0)
        score = coverage * 20
        score -= abs(len(quiz.questions) - config.question_count) * 2

        if "fill_blank" in required:
            fill_ok = any(
                q.question_type.value == "fill_blank"
                and q.correct_answer
                and ("____" in q.prompt or "\u586b\u7a7a" in q.prompt.lower())
                for q in quiz.questions
            )
            score += 15 if fill_ok else 0

        if "true_false" in required:
            tf_ok = any(
                q.question_type.value == "true_false"
                and set(q.options) >= {TRUE_LABEL, FALSE_LABEL}
                and q.correct_answer
                and q.correct_answer[0] in {TRUE_LABEL, FALSE_LABEL}
                for q in quiz.questions
            )
            score += 15 if tf_ok else 0

        if "short_answer" in required:
            short_ok = any(
                q.question_type.value == "short_answer"
                and q.correct_answer
                and bool("".join(q.correct_answer).strip())
                for q in quiz.questions
            )
            score += 15 if short_ok else 0

        # Penalize duplicate prompts to prioritize diversity.
        prompt_keys = [
            re.sub(r"[^\w\u4e00-\u9fff]", "", (q.prompt or "").lower())
            for q in quiz.questions
        ]
        dedup = len(set(prompt_keys))
        duplicates = max(0, len(prompt_keys) - dedup)
        score -= duplicates * 10
        score += self._quiz_content_quality_score(quiz)
        return score

    def _quiz_quality_good(self, quiz: Quiz, config: QuizConfig) -> bool:
        targets = self._distribution_targets(config.question_count, config.type_mix)
        required = [name for name, count in targets.items() if count > 0]
        actual = Counter(question.question_type.value for question in quiz.questions)

        if any(actual.get(name, 0) <= 0 for name in required):
            return False
        if len(quiz.questions) < config.question_count:
            return False

        if "fill_blank" in required:
            if not any(
                q.question_type.value == "fill_blank"
                and q.correct_answer
                and ("____" in q.prompt or "\u586b\u7a7a" in q.prompt.lower())
                for q in quiz.questions
            ):
                return False

        if "true_false" in required:
            if not any(
                q.question_type.value == "true_false"
                and set(q.options) >= {TRUE_LABEL, FALSE_LABEL}
                and q.correct_answer
                and q.correct_answer[0] in {TRUE_LABEL, FALSE_LABEL}
                for q in quiz.questions
            ):
                return False

        if "short_answer" in required:
            if not any(
                q.question_type.value == "short_answer"
                and q.correct_answer
                and bool("".join(q.correct_answer).strip())
                for q in quiz.questions
            ):
                return False

        # Keep a minimum semantic bar to avoid fast but low-quality quizzes.
        min_quality_score = max(30, len(quiz.questions) * 4)
        if self._quiz_content_quality_score(quiz) < min_quality_score:
            return False

        return True

    def _quiz_content_quality_score(self, quiz: Quiz) -> int:
        score = 0
        for q in quiz.questions:
            prompt = (q.prompt or "").strip()
            explanation = (q.explanation or "").strip()
            refs = [str(item).strip() for item in (q.reference_points or []) if str(item).strip()]

            score += 2 if len(prompt) >= 16 else -3
            score += 2 if len(explanation) >= 20 else -2
            score += 1 if refs else -1

            if q.question_type.value in {
                QuestionType.single_choice.value,
                QuestionType.multiple_choice.value,
            }:
                options = [str(item).strip() for item in (q.options or []) if str(item).strip()]
                unique_options = set(options)
                if len(options) >= 4:
                    score += 2
                else:
                    score -= 3
                if len(unique_options) == len(options):
                    score += 1
                else:
                    score -= 2

                generic_like = sum(
                    1
                    for item in options
                    if re.search(r"^(选项|option)\s*[A-D]?$", item.strip(), flags=re.IGNORECASE)
                )
                if generic_like >= max(1, len(options) // 2):
                    score -= 4

                has_answer_in_options = any(
                    ans.strip() in unique_options for ans in (q.correct_answer or [])
                )
                if not has_answer_in_options:
                    score -= 4

            if q.question_type.value == QuestionType.short_answer.value:
                answer_len = len(" ".join(q.correct_answer or []).strip())
                if answer_len < 18:
                    score -= 3
                else:
                    score += 2

        # Penalize near-duplicate prompt bodies.
        keys = [
            re.sub(r"[^\w\u4e00-\u9fff]", "", (q.prompt or "").lower())
            for q in quiz.questions
        ]
        duplicates = max(0, len(keys) - len(set(keys)))
        score -= duplicates * 8
        return score
