from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quizmind.cache import JsonFileCache
from quizmind.content import fallback_parse_content
from quizmind.logger import log_event, timed_event
from quizmind.models import BatchSubjectiveGrade, Difficulty, ParsedContent, Question, QuestionType, Quiz, QuizConfig


load_dotenv()


class LangChainQuizProvider:
    def __init__(self) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self.cache = JsonFileCache()
        self.llm: Optional[ChatOpenAI] = None
        if api_key:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.4,
                timeout=30,
                max_retries=2,
            )

    def parse_content(self, source: str, source_type: str) -> ParsedContent:
        if not self.llm:
            return fallback_parse_content(source, source_type)

        payload = {"source_type": source_type, "content": source[:8000]}
        response = self._invoke_json(
            operation="parse_content",
            temperature=0.2,
            system_prompt=(
                "你是学习内容解析助手。只返回 JSON。"
                "必须包含字段：title, source_type, cleaned_text, segments, knowledge_points, concepts。"
                "knowledge_points 每项包含：name, summary, importance(1-5), difficulty(easy|medium|hard), keywords。"
                "所有自然语言内容请使用简体中文；若无法稳定输出中文，请改用英文，不要输出乱码。"
            ),
            human_prompt="请解析以下学习内容：\n{payload}",
            payload=payload,
        )
        if not response:
            return fallback_parse_content(source, source_type)
        try:
            return ParsedContent.model_validate(self._normalize_parsed_payload(response, source_type))
        except Exception as exc:
            log_event("parse_content.fallback", reason=str(exc))
            return fallback_parse_content(source, source_type)

    def generate_quiz(self, parsed: ParsedContent, config: QuizConfig) -> Optional[Quiz]:
        if not self.llm:
            return None

        payload = {
            "title": parsed.title,
            "knowledge_points": [point.model_dump() for point in parsed.knowledge_points[:10]],
            "segments": parsed.segments[:10],
            "config": config.model_dump(),
        }
        response = self._invoke_json(
            operation="generate_quiz",
            temperature=0.5,
            system_prompt=(
                "你是专业出题助手。只返回 JSON。"
                "输出结构包含字段：title、source_summary、questions。"
                "每题字段：id, question_type(single_choice|multiple_choice|fill_blank|short_answer|true_false), "
                "prompt, options(list), correct_answer(list), explanation, knowledge_tags(list), "
                "difficulty(easy|medium|hard), reference_points(list)。"
                "所有自然语言内容请使用简体中文；若无法稳定输出中文，请改用英文，不要输出乱码。"
            ),
            human_prompt="请根据以下解析结果生成题目：\n{payload}",
            payload=payload,
        )
        if not response:
            return None
        return Quiz.model_validate(self._normalize_quiz_payload(response, parsed.title))

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
            "content": source[:8000],
            "config": config.model_dump(),
        }
        response = self._invoke_json(
            operation="generate_quiz_from_source",
            temperature=0.5,
            system_prompt=(
                "你是智能练习系统的核心引擎。"
                "一次性完成内容解析与出题。只返回 JSON，必须包含 parsed_content 和 quiz。"
                "所有自然语言内容请使用简体中文；若无法稳定输出中文，请改用英文，不要输出乱码。"
            ),
            human_prompt="请处理以下原始内容并输出 parsed_content + quiz：\n{payload}",
            payload=payload,
        )
        if not response:
            parsed = fallback_parse_content(source, source_type)
            return parsed, None

        try:
            parsed_data = self._normalize_parsed_payload(response.get("parsed_content", {}), source_type)
            quiz_data = self._normalize_quiz_payload(response.get("quiz", {}), parsed_data.get("title", "Quiz"))
            return ParsedContent.model_validate(parsed_data), Quiz.model_validate(quiz_data)
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
                    "prompt": question.prompt,
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
            system_prompt=(
                "你是严格且友好的阅卷老师。只返回 JSON。"
                "输出结构包含字段 grades，且每项包含 question_id、score、feedback、missing_points。"
                "score 范围是 0-100。feedback 和 missing_points 请使用简体中文；若无法稳定中文，请改用英文，不要输出乱码。"
            ),
            human_prompt="请批量评分以下主观题作答：\n{payload}",
            payload=payload,
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
                    "missing_points": [str(point) for point in item.get("missing_points", [])],
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
            "knowledge_points": [point.model_dump() for point in parsed.knowledge_points[:8]],
        }
        return self._invoke_text(
            operation="generate_interactive_html",
            temperature=0.3,
            system_prompt=(
                "你是前端交互学习页面生成助手。"
                "请输出一个完整可嵌入的 HTML 片段（包含 style 和 script），"
                "用于帮助用户理解知识点。"
                "要求：中文文案、结构清晰、卡片式布局、可点击切换知识点、"
                "包含一个小测验区（1-3题，单选）和即时反馈。"
                "中文优先；若无法稳定中文，请改用英文，不要输出乱码。"
                "不要依赖外部 CDN，不要输出 markdown 代码块。"
            ),
            human_prompt="请基于以下解析结果生成互动学习网页：\n{payload}",
            payload=payload,
        )

    def _invoke_json(
        self,
        operation: str,
        temperature: float,
        system_prompt: str,
        human_prompt: str,
        payload: dict,
    ) -> dict | None:
        if not self.llm:
            return None

        cache_key = self.cache.build_key(
            operation,
            {"model": self.model, "temperature": temperature, "payload": payload},
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_event("llm.cache_hit", operation=operation, key=cache_key)
            return cached

        log_event("llm.cache_miss", operation=operation, key=cache_key)
        try:
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
            chain = prompt | self.llm.bind(temperature=temperature) | StrOutputParser()
            with timed_event("llm.invoke", operation=operation, model=self.model):
                text = chain.invoke({"payload": json.dumps(payload, ensure_ascii=False)})
            data = self._parse_json(text)
            self.cache.set(cache_key, data)
            return data
        except Exception as exc:
            log_event(
                "llm.invoke_failed",
                operation=operation,
                model=self.model,
                error_type=exc.__class__.__name__,
                error=str(exc),
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
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
            chain = prompt | self.llm.bind(temperature=temperature) | StrOutputParser()
            with timed_event("llm.invoke", operation=operation, model=self.model):
                text = chain.invoke({"payload": json.dumps(payload, ensure_ascii=False)}).strip()

            if text.startswith("```"):
                text = re.sub(r"^```(?:html)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            self.cache.set(cache_key, {"text": text})
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
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

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
                    "name": str(item.get("name", "未命名知识点")),
                    "summary": str(item.get("summary", "")),
                    "importance": int(item.get("importance", 3)),
                    "difficulty": Difficulty.normalize(item.get("difficulty")),
                    "keywords": [str(k) for k in item.get("keywords", [])],
                }
            )

        segments = [str(seg) for seg in data.get("segments", []) if str(seg).strip()]
        concepts = [str(c) for c in data.get("concepts", []) if str(c).strip()]
        return {
            "title": str(data.get("title", "未命名内容")),
            "source_type": source_type,
            "cleaned_text": str(data.get("cleaned_text", "")),
            "segments": segments,
            "knowledge_points": points,
            "concepts": concepts,
        }

    def _normalize_quiz_payload(self, data: dict, fallback_title: str) -> dict:
        questions = []
        for index, question in enumerate(data.get("questions", []), start=1):
            normalized = dict(question or {})
            normalized["id"] = str(normalized.get("id", f"Q{index:03d}"))
            normalized["question_type"] = QuestionType.normalize(normalized.get("question_type"))
            normalized["difficulty"] = Difficulty.normalize(normalized.get("difficulty"))

            ans = normalized.get("correct_answer", [])
            normalized["correct_answer"] = [str(a) for a in ans] if isinstance(ans, list) else [str(ans)]

            opts = normalized.get("options", [])
            normalized["options"] = [str(o) for o in opts] if isinstance(opts, list) else ([str(opts)] if opts else [])

            tags = normalized.get("knowledge_tags", [])
            refs = normalized.get("reference_points", [])
            normalized["knowledge_tags"] = [str(t) for t in tags] if isinstance(tags, list) else [str(tags)]
            normalized["reference_points"] = [str(r) for r in refs] if isinstance(refs, list) else [str(refs)]
            normalized["prompt"] = str(normalized.get("prompt", ""))
            normalized["explanation"] = str(normalized.get("explanation", ""))
            questions.append(normalized)

        return {
            "title": str(data.get("title", f"{fallback_title} - 智能练习")),
            "source_summary": str(data.get("source_summary", fallback_title)),
            "questions": questions,
        }
