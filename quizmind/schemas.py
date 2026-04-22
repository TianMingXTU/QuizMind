from __future__ import annotations

import json
from typing import Any, Dict


PARSED_CONTENT_SCHEMA: Dict[str, Any] = {
    "title": "string",
    "source_type": "text|file|url|memory",
    "cleaned_text": "string",
    "segments": ["string"],
    "knowledge_points": [
        {
            "name": "string",
            "summary": "string",
            "importance": "int(1-5)",
            "difficulty": "easy|medium|hard",
            "keywords": ["string"],
        }
    ],
    "concepts": ["string"],
}


QUIZ_SCHEMA: Dict[str, Any] = {
    "title": "string",
    "source_summary": "string",
    "questions": [
        {
            "id": "string",
            "question_type": "single_choice|multiple_choice|fill_blank|short_answer|true_false",
            "prompt": "string",
            "options": ["string"],
            "correct_answer": ["string"],
            "explanation": "string",
            "knowledge_tags": ["string"],
            "difficulty": "easy|medium|hard",
            "reference_points": ["string"],
        }
    ],
}


BATCH_GRADE_SCHEMA: Dict[str, Any] = {
    "grades": [
        {
            "question_id": "string",
            "score": "number(0-100)",
            "score_breakdown": {
                "correctness": "number(0-10)",
                "completeness": "number(0-10)",
                "clarity": "number(0-10)",
            },
            "feedback": "string",
            "missing_points": ["string"],
            "error_category": "concept_unclear|careless_mistake|reasoning_error|knowledge_forgotten|expression_issue|none",
            "structured_explanation": "string",
        }
    ]
}


SCENE_TURN_SCHEMA: Dict[str, Any] = {
    "engineer_message": "string",
    "should_end": "bool",
    "is_passed": "bool",
    "score": "number(0-100)",
    "assessment": "string",
    "strengths": ["string"],
    "weaknesses": ["string"],
    "recommendations": ["string"],
}


QUIZ_QUALITY_REVIEW_SCHEMA: Dict[str, Any] = {
    "pass": "bool",
    "overall_score": "number(0-100)",
    "issues": [
        {
            "question_id": "string",
            "severity": "low|medium|high",
            "category": "grounding|clarity|difficulty|options|answer_key|explanation|duplication",
            "message": "string",
        }
    ],
    "summary": "string",
}


def schema_text(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)
