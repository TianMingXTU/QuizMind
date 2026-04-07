from __future__ import annotations

import json
import os
import re
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
from quizmind.models import (
    BatchSubjectiveGrade,
    Difficulty,
    ParsedContent,
    Question,
    QuestionType,
    Quiz,
    QuizConfig,
)


load_dotenv()

TRUE_LABEL = "\u6b63\u786e"
FALSE_LABEL = "\u9519\u8bef"

PARSE_CONTENT_SYSTEM_PROMPT = (
    "You are a learning-content parser. Return JSON only.\n"
    "Required keys: title, source_type, cleaned_text, segments, knowledge_points, concepts.\n"
    "Each knowledge_points item must include: name, summary, importance(1-5), "
    "difficulty(easy|medium|hard), keywords.\n"
    "Prefer Simplified Chinese for natural-language fields. If unstable, output English.\n"
    "Do not output mojibake."
)

GENERATE_QUIZ_SYSTEM_PROMPT = (
    "You are a professional quiz generator. Return JSON only.\n"
    "Required keys: title, source_summary, questions.\n"
    "Each question item must include: id, question_type(single_choice|multiple_choice|"
    "fill_blank|short_answer|true_false), prompt, options(list), correct_answer(list), "
    "explanation, knowledge_tags(list), difficulty(easy|medium|hard), reference_points(list).\n"
    "Prefer Simplified Chinese for natural-language fields. If unstable, output English.\n"
    "Do not output mojibake."
    "Every question must be grounded in the provided source content.\n"
    "Do not invent facts, terms, numbers, formulas, people, or events not present in source."
)

GENERATE_QUIZ_GUIDANCE = (
    "Follow config type distribution as strictly as possible. Ensure usability of "
    "fill_blank, true_false, short_answer.\n"
    "fill_blank must be answerable; true_false must include options "
    f"'{TRUE_LABEL}' and '{FALSE_LABEL}'; short_answer must provide complete reference answers."
    "\nEach question must include reference_points copied from source wording whenever possible."
)

GENERATE_FROM_SOURCE_SYSTEM_PROMPT = (
    "You are the core engine of an adaptive quiz system. In one response, complete both "
    "content parsing and quiz generation.\n"
    "Return JSON only with keys: parsed_content and quiz.\n"
    "Prefer Simplified Chinese for natural-language fields. If unstable, output English.\n"
    "Do not output mojibake."
)

GRADE_BATCH_SYSTEM_PROMPT = (
    "You are a strict but supportive grader. Return JSON only.\n"
    "Output schema: grades (list). Each grade item includes: question_id, score(0-100), "
    "feedback, missing_points.\n"
    "For objective questions (single_choice, multiple_choice, true_false, fill_blank), "
    "use exact-match style grading and prefer 0 or 100.\n"
    "For short_answer, score by semantic completeness against reference points.\n"
    "Prefer Simplified Chinese for natural-language fields. If unstable, output English.\n"
    "Do not output mojibake."
)

GENERATE_INTERACTIVE_HTML_PROMPT = (
    "You are an assistant for generating interactive learning pages.\n"
    "Return one complete embeddable HTML snippet (including style and script).\n"
    "Requirements: Chinese copy preferred, clear structure, card layout, clickable topic switch, "
    "mini quiz (1-3 single-choice questions) with instant feedback.\n"
    "No external CDN dependency. Do not output markdown code fences.\n"
    "If Chinese output becomes unstable, use English and never output mojibake."
)

SOURCE_CHUNK_SUMMARY_PROMPT = (
    "You summarize a chunk of learning material.\n"
    "Return plain text only.\n"
    "Keep key facts, definitions, processes, formulas, examples, and constraints.\n"
    "Prefer Simplified Chinese."
)

SOURCE_MERGE_SUMMARY_PROMPT = (
    "You merge multiple chunk summaries into one concise learning source.\n"
    "Return plain text only.\n"
    "Keep coverage complete and remove repetition.\n"
    "Preserve key terms and technical details.\n"
    "Prefer Simplified Chinese."
)


class LangChainQuizProvider:
    def __init__(self) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3.2")
        self.fast_mode = self._safe_bool_env("QUIZMIND_FAST_MODE", False)
        self.llm_timeout = max(10, self._safe_int_env("QUIZMIND_LLM_TIMEOUT", 45))
        self.llm_max_retries = max(0, self._safe_int_env("QUIZMIND_LLM_MAX_RETRIES", 2))
        default_attempts = 1 if self.fast_mode else 3
        self.quiz_gen_attempts = max(
            1,
            min(6, self._safe_int_env("QUIZMIND_QUIZ_GEN_ATTEMPTS", default_attempts)),
        )
        self.source_char_limit = self._safe_int_env("QUIZMIND_SOURCE_CHAR_LIMIT", 0)
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
            0, self._safe_int_env("QUIZMIND_MAX_OUTPUT_TOKENS", 0)
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
        default_concurrency = 1 if self.fast_mode else 2
        self.quiz_gen_concurrency = max(
            1,
            min(
                6,
                self._safe_int_env(
                    "QUIZMIND_QUIZ_GEN_CONCURRENCY", default_concurrency
                ),
            ),
        )

        self.cache = JsonFileCache()
        self.llm: Optional[ChatOpenAI] = None
        if api_key:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.4,
                timeout=self.llm_timeout,
                max_retries=self.llm_max_retries,
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
        self, parsed: ParsedContent, config: QuizConfig
    ) -> Optional[Quiz]:
        if not self.llm:
            return None

        payload = {
            "title": parsed.title,
            "knowledge_points": [
                point.model_dump()
                for point in parsed.knowledge_points[: self.quiz_points_limit]
            ],
            "segments": parsed.segments[: self.quiz_segments_limit],
            "source_context": self._build_source_context(parsed.cleaned_text),
            "config": config.model_dump(),
        }

        temperatures = [0.45] if self.fast_mode else [0.45, 0.6, 0.35, 0.7, 0.5, 0.3]
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
    ) -> tuple[Quiz, int] | None:
        response = self._invoke_json(
            operation=f"generate_quiz_try_{attempt + 1}",
            temperature=temperatures[attempt % len(temperatures)],
            system_prompt=GENERATE_QUIZ_SYSTEM_PROMPT,
            human_prompt=(
                "Generate a quiz from the parsed result below.\n"
                f"Additional requirements:\n{GENERATE_QUIZ_GUIDANCE}\n"
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
            human_prompt="Process the raw content below and output parsed_content + quiz:\n{payload}",
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
                improved = self.generate_quiz(parsed_obj, config)
                return parsed_obj, improved
            if self.fast_mode:
                return parsed_obj, quiz_obj
            if self._quiz_quality_good(quiz_obj, config):
                return parsed_obj, quiz_obj

            improved = self.generate_quiz(parsed_obj, config)
            return parsed_obj, (improved or quiz_obj)
        except Exception as exc:
            log_event("generate_quiz_from_source.fallback", reason=str(exc))
            parsed = fallback_parse_content(source, source_type)
            return parsed, self.generate_quiz(parsed, config)

    def grade_subjective_batch(
        self,
        questions_and_answers: list[tuple[Question, List[str]]],
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
            system_prompt=GRADE_BATCH_SYSTEM_PROMPT,
            human_prompt="Grade the following answers in batch (objective and subjective):\n{payload}",
            payload=payload,
            use_cache=False,
        )
        if not response:
            return {}

        grades = response.get("grades", [])
        result: dict[str, BatchSubjectiveGrade] = {}
        for item in grades:
            grade = BatchSubjectiveGrade.model_validate(
                {
                    "question_id": str(item.get("question_id", "")),
                    "score": float(item.get("score", 0)),
                    "feedback": str(item.get("feedback", "")),
                    "missing_points": [
                        str(point) for point in item.get("missing_points", [])
                    ],
                }
            )
            result[grade.question_id] = grade
        return result

    def generate_interactive_html(self, parsed: ParsedContent) -> Optional[str]:
        if not self.llm:
            return None

        payload = {
            "title": parsed.title,
            "concepts": parsed.concepts[:8],
            "knowledge_points": [
                point.model_dump() for point in parsed.knowledge_points[:8]
            ],
        }

        return self._invoke_text(
            operation="generate_interactive_html",
            temperature=0.3,
            system_prompt=GENERATE_INTERACTIVE_HTML_PROMPT,
            human_prompt="Generate an interactive learning webpage from the parsed result below:\n{payload}",
            payload=payload,
        )

    def _invoke_json(
        self,
        operation: str,
        temperature: float,
        system_prompt: str,
        human_prompt: str,
        payload: dict,
        use_cache: bool = True,
    ) -> dict | None:
        if not self.llm:
            return None

        cache_key = self.cache.build_key(
            operation,
            {"model": self.model, "temperature": temperature, "payload": payload},
        )
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                log_event("llm.cache_hit", operation=operation, key=cache_key)
                return cached

        log_event("llm.cache_miss", operation=operation, key=cache_key)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{format_instructions}\n\n" + human_prompt),
            ]
        )
        parser = JsonOutputParser()
        message_vars = {
            "payload": json.dumps(payload, ensure_ascii=False),
            "format_instructions": parser.get_format_instructions(),
        }

        try:
            chain = (
                prompt
                | self.llm.bind(**self._llm_bind_kwargs(temperature))
                | StrOutputParser()
            )
            with timed_event("llm.invoke", operation=operation, model=self.model):
                text = chain.invoke(message_vars)
        except Exception as exc:
            log_event(
                "llm.invoke_failed",
                operation=operation,
                model=self.model,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            return None

        data: dict | None = None
        try:
            parsed = parser.parse(text)
            if isinstance(parsed, dict):
                data = parsed
            elif isinstance(parsed, list):
                data = {"items": parsed}
        except Exception:
            data = None

        if data is None:
            try:
                data = self._parse_json(text)
            except Exception as exc:
                log_event(
                    "llm.invoke_failed",
                    operation=operation,
                    model=self.model,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
                return None

        if use_cache:
            self.cache.set(cache_key, data)
        log_event("llm.invoke.success", operation=operation, model=self.model)
        return data

    def _invoke_text(
        self,
        operation: str,
        temperature: float,
        system_prompt: str,
        human_prompt: str,
        payload: dict,
    ) -> Optional[str]:
        if not self.llm:
            return None

        cache_key = self.cache.build_key(
            operation,
            {"model": self.model, "temperature": temperature, "payload": payload},
        )
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            log_event("llm.cache_hit", operation=operation, key=cache_key)
            return str(cached["text"])

        log_event("llm.cache_miss", operation=operation, key=cache_key)
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", human_prompt)]
            )
            chain = (
                prompt
                | self.llm.bind(**self._llm_bind_kwargs(temperature))
                | StrOutputParser()
            )
            with timed_event("llm.invoke", operation=operation, model=self.model):
                text = chain.invoke(
                    {"payload": json.dumps(payload, ensure_ascii=False)}
                ).strip()

            if text.startswith("```"):
                text = re.sub(r"^```(?:html)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            self.cache.set(cache_key, {"text": text})
            log_event("llm.invoke.success", operation=operation, model=self.model)
            return text
        except Exception as exc:
            log_event(
                "llm.invoke_failed",
                operation=operation,
                model=self.model,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            return None

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

    def _llm_bind_kwargs(self, temperature: float) -> dict:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if self.max_output_tokens > 0:
            kwargs["max_tokens"] = self.max_output_tokens
        return kwargs

    def _prepare_source_content(self, source: str) -> str:
        cleaned = source.strip()
        if self.source_char_limit <= 0 or len(cleaned) <= self.source_char_limit:
            return cleaned

        # For long input, prefer AI summarization over hard truncation.
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
        if not self.llm or self.source_char_limit <= 0:
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

    def _build_source_context(self, source: str, max_chars: int = 5000) -> str:
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

        return True
