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
from quizmind.models import BatchSubjectiveGrade, ParsedContent, Question, Quiz, QuizConfig


load_dotenv()


class LangChainQuizProvider:
    def __init__(self) -> None:
        api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.model = os.getenv("SILICONFLOW_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2")
        self.cache = JsonFileCache()
        self.llm: Optional[ChatOpenAI] = None
        if api_key:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.4,
            )

    def parse_content(self, source: str, source_type: str) -> ParsedContent:
        if not self.llm:
            return fallback_parse_content(source, source_type)

        payload = {"source_type": source_type, "content": source[:8000]}
        response = self._invoke_json(
            operation="parse_content",
            temperature=0.2,
            system_prompt=(
                "你是知识整理助手。请从用户内容中提取结构化学习信息。"
                "必须只输出 JSON。字段必须包括：title、source_type、cleaned_text、segments、knowledge_points、concepts。"
                "knowledge_points 中每个对象必须包括：name、summary、importance、difficulty、keywords。"
                "difficulty 只能是：简单、中等、困难。"
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
                "你是专业出题助手。请基于学习内容生成高质量练习题。"
                "必须只输出 JSON。返回结构必须是一个 Quiz 对象，字段包含：title、source_summary、questions。"
                "每道题必须包含：id、question_type、prompt、options、correct_answer、explanation、"
                "knowledge_tags、difficulty、reference_points。"
                "question_type 只能是：单选题、多选题、填空题、简答题、判断题。"
                "difficulty 只能是：简单、中等、困难。"
            ),
            human_prompt=(
                "请根据以下解析结果生成题目：\n{payload}\n"
                "要求：题量和题型分布尽量贴近配置，解析具体，简答题提供可评分要点。"
            ),
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
                "请一次性完成内容解析与出题，并只输出 JSON。"
                "返回字段必须包含 parsed_content 和 quiz。"
                "parsed_content 结构：title、source_type、cleaned_text、segments、knowledge_points、concepts。"
                "quiz 结构：title、source_summary、questions。"
                "每道题必须包含：id、question_type、prompt、options、correct_answer、explanation、knowledge_tags、difficulty、reference_points。"
                "difficulty 只能是：简单、中等、困难。"
            ),
            human_prompt=(
                "请基于以下原始内容，一次性完成解析和出题：\n{payload}\n"
                "要求：尽量减少冗余字段，题目数量和比例贴近配置。"
            ),
            payload=payload,
        )
        if not response:
            parsed = fallback_parse_content(source, source_type)
            return parsed, None

        try:
            parsed_data = self._normalize_parsed_payload(response.get("parsed_content", {}), source_type)
            quiz_data = self._normalize_quiz_payload(
                response.get("quiz", {}),
                parsed_data.get("title", "智能练习"),
            )
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
                "你是严格但友好的阅卷老师。"
                "请批量对多道简答题评分，并只输出 JSON。"
                "返回字段必须为 grades，且 grades 是数组。"
                "数组中每项必须包含：question_id、score、feedback、missing_points。"
                "score 范围为 0 到 100。"
            ),
            human_prompt="请对以下作答批量评分：\n{payload}",
            payload=payload,
        )
        if not response:
            return {}

        grades = response.get("grades", [])
        result: dict[str, BatchSubjectiveGrade] = {}
        for item in grades:
            normalized = {
                "question_id": str(item.get("question_id", "")),
                "score": float(item.get("score", 0)),
                "feedback": str(item.get("feedback", "")),
                "missing_points": [str(point) for point in item.get("missing_points", [])],
            }
            grade = BatchSubjectiveGrade.model_validate(normalized)
            result[grade.question_id] = grade
        return result

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
            {
                "model": self.model,
                "temperature": temperature,
                "payload": payload,
            },
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_event("llm.cache_hit", operation=operation, key=cache_key)
            return cached

        log_event("llm.cache_miss", operation=operation, key=cache_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        chain = prompt | self.llm.bind(temperature=temperature) | StrOutputParser()
        with timed_event("llm.invoke", operation=operation, model=self.model):
            text = chain.invoke({"payload": json.dumps(payload, ensure_ascii=False)})
        data = self._parse_json(text)
        self.cache.set(cache_key, data)
        return data

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
        knowledge_points = []
        for item in data.get("knowledge_points", []):
            if isinstance(item, str):
                knowledge_points.append(
                    {
                        "name": item,
                        "summary": item,
                        "importance": 3,
                        "difficulty": "中等",
                        "keywords": [item],
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            knowledge_points.append(
                {
                    "name": str(item.get("name", "未命名知识点")),
                    "summary": str(item.get("summary", "")),
                    "importance": int(item.get("importance", 3)),
                    "difficulty": self._normalize_difficulty(item.get("difficulty")),
                    "keywords": [str(keyword) for keyword in item.get("keywords", [])],
                }
            )

        raw_segments = data.get("segments", [])
        raw_concepts = data.get("concepts", [])
        segments = [str(segment) for segment in raw_segments] if isinstance(raw_segments, list) else []
        concepts = [str(concept) for concept in raw_concepts] if isinstance(raw_concepts, list) else []
        segments = [segment for segment in segments if segment.strip()]
        concepts = [concept for concept in concepts if concept.strip()]
        return {
            "title": str(data.get("title", "未命名内容")),
            "source_type": source_type,
            "cleaned_text": str(data.get("cleaned_text", "")),
            "segments": segments,
            "knowledge_points": knowledge_points,
            "concepts": concepts,
        }

    def _normalize_quiz_payload(self, data: dict, fallback_title: str) -> dict:
        questions = []
        for index, question in enumerate(data.get("questions", []), start=1):
            normalized = dict(question)
            normalized["id"] = str(normalized.get("id", f"Q{index:03d}"))
            normalized["question_type"] = self._normalize_question_type(normalized.get("question_type"))
            normalized["difficulty"] = self._normalize_difficulty(normalized.get("difficulty"))

            answer = normalized.get("correct_answer", [])
            if isinstance(answer, list):
                normalized["correct_answer"] = [str(item) for item in answer]
            elif answer is None:
                normalized["correct_answer"] = []
            else:
                normalized["correct_answer"] = [str(answer)]

            options = normalized.get("options", [])
            if isinstance(options, list):
                normalized["options"] = [str(item) for item in options]
            elif options is None:
                normalized["options"] = []
            else:
                normalized["options"] = [str(options)]

            tags = normalized.get("knowledge_tags", [])
            refs = normalized.get("reference_points", [])
            normalized["knowledge_tags"] = [str(item) for item in tags] if isinstance(tags, list) else [str(tags)]
            normalized["reference_points"] = [str(item) for item in refs] if isinstance(refs, list) else [str(refs)]
            normalized["prompt"] = str(normalized.get("prompt", ""))
            normalized["explanation"] = str(normalized.get("explanation", ""))
            questions.append(normalized)

        return {
            "title": str(data.get("title", f"{fallback_title} - 智能练习")),
            "source_summary": str(data.get("source_summary", fallback_title)),
            "questions": questions,
        }

    def _normalize_question_type(self, raw: object) -> str:
        value = str(raw or "单选题")
        aliases = {
            "单选": "单选题",
            "单项选择题": "单选题",
            "多选": "多选题",
            "多项选择题": "多选题",
            "填空": "填空题",
            "简答": "简答题",
            "判断": "判断题",
        }
        return aliases.get(value, value)

    def _normalize_difficulty(self, raw: object) -> str:
        value = str(raw or "中等")
        aliases = {
            "容易": "简单",
            "简单题": "简单",
            "普通": "中等",
            "中等题": "中等",
            "困难题": "困难",
        }
        return aliases.get(value, value)
