from __future__ import annotations

import json
import re
import html
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List

import streamlit as st
import streamlit.components.v1 as components

from quizmind.content import load_text_from_upload, load_text_from_url
from quizmind.logger import log_event
from quizmind.models import Question, QuestionType, Quiz, QuizConfig, QuizMode, UserAnswer
from quizmind.services import (
    ContentService,
    GradingService,
    QuizEngine,
    build_targeted_quiz,
)
from quizmind.user_store import UserFeatureStore


st.set_page_config(page_title="QuizMind", page_icon="🧠", layout="wide")


@st.cache_resource
def get_services() -> (
    tuple[
        ContentService,
        QuizEngine,
        GradingService,
        UserFeatureStore,
    ]
):
    return (
        ContentService(),
        QuizEngine(),
        GradingService(),
        UserFeatureStore(),
    )


@st.cache_data(show_spinner=False)
def read_uploaded_text(file_name: str, data: bytes) -> str:
    return load_text_from_upload(file_name, data)












def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --qm-bg: #0d1117;
            --qm-surface: #161b22;
            --qm-surface-muted: #21262d;
            --qm-border: #30363d;
            --qm-text: #c9d1d9;
            --qm-subtle: #8b949e;
            --qm-brand: #58a6ff;
            --qm-accent-hover: #79c0ff;
        }
        .stApp {
            background: var(--qm-bg);
            color: var(--qm-text);
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: var(--qm-bg);
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 980px;
            padding-top: 1rem;
            padding-bottom: 1.4rem;
        }
        [data-testid="stSidebar"] {
            background: var(--qm-surface);
            border-right: 1px solid var(--qm-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--qm-text);
        }
        .qm-page-title {
            font-size: 26px;
            line-height: 1.2;
            font-weight: 700;
            color: var(--qm-text);
            margin-bottom: 2px;
        }
        .qm-page-subtitle {
            color: var(--qm-subtle);
            font-size: 14px;
            margin-bottom: 8px;
        }
        .qm-section-title {
            font-size: 18px;
            font-weight: 600;
            color: var(--qm-text);
            margin-bottom: 2px;
        }
        .qm-section-subtitle {
            color: var(--qm-subtle);
            font-size: 13px;
            margin-bottom: 8px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--qm-border);
            border-radius: 10px;
            background: var(--qm-surface);
            box-shadow: none;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 500;
            color: var(--qm-text);
        }
        div.stButton > button {
            border-radius: 8px;
            border: 1px solid var(--qm-border);
            background: var(--qm-surface-muted);
            color: var(--qm-text);
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        div.stButton > button:hover {
            border-color: var(--qm-accent-hover);
            background: #262c36;
        }
        div.stButton > button:focus {
            outline: 2px solid var(--qm-brand) !important;
            outline-offset: 1px !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            background: var(--qm-surface-muted) !important;
            color: var(--qm-text) !important;
            border-color: var(--qm-border) !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        textarea:focus,
        input:focus {
            border-color: var(--qm-accent-hover) !important;
            box-shadow: 0 0 0 1px var(--qm-accent-hover) inset !important;
        }
        .stRadio label, .stCheckbox label, .stSelectbox label, .stTextInput label, .stTextArea label {
            color: var(--qm-text) !important;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--qm-subtle) !important;
        }
        .stMarkdown a {
            color: #58a6ff !important;
        }
        .stMarkdown a:hover {
            color: #79c0ff !important;
            text-decoration: underline;
        }
        .stMarkdown p, .stMarkdown li, .stMarkdown div {
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .stCodeBlock, pre, code {
            background: #0d1117 !important;
            border: 1px solid var(--qm-border) !important;
            border-radius: 6px !important;
            color: var(--qm-text) !important;
            white-space: pre-wrap !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }
        .qm-wrap-text {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
            line-height: 1.55;
        }
        .qm-origin-badge {
            display: inline-block;
            padding: 2px 10px;
            border: 1px solid var(--qm-border);
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            color: #79c0ff;
            background: rgba(56, 139, 253, 0.16);
            margin-bottom: 8px;
        }
        .qm-question-card {
            border: 1px solid var(--qm-border);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 8px;
            background: var(--qm-surface);
        }
        .qm-question-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
        }
        .qm-question-title {
            font-weight: 700;
            color: var(--qm-text);
        }
        .qm-question-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 6px;
        }
        .qm-question-prompt {
            color: var(--qm-text);
            line-height: 1.6;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .qm-question {
            border: 1px solid var(--qm-border);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            background: var(--qm-surface);
        }
        .qm-chip {
            display: inline-block;
            margin-right: 6px;
            margin-bottom: 6px;
            padding: 2px 10px;
            border-radius: 999px;
            background: #1f2937;
            color: #79c0ff;
            border: 1px solid var(--qm-border);
            font-size: 12px;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--qm-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_section_head(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="qm-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="qm-section-subtitle">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )





IMPORT_ARCHIVE_DIR = Path(".quizmind_runtime/import_papers")
IMPORT_ANSWER_SNAPSHOT_DIR = Path(".quizmind_runtime/import_answers")

IMPORT_EXAM_TEMPLATE = {
    "title": "导入试卷示例",
    "source_summary": "由外部大模型生成",
    "questions": [
        {
            "id": "Q1",
            "question_type": "single_choice",
            "prompt": "Python 中用于定义函数的关键字是？",
            "options": ["A. func", "B. def", "C. lambda", "D. class"],
            "difficulty": "easy"
        },
        {
            "id": "Q2",
            "question_type": "essay",
            "prompt": "简述 HTTP 与 HTTPS 的差异，并说明明文 HTTP 的两个风险。",
            "difficulty": "medium"
        },
        {
            "id": "Q3",
            "question_type": "case_analysis",
            "prompt": "```python\n# 示例伪代码\nif latency_ms > 500:\n    trigger_alert()\n```\n请根据以上信息给出排障方案。",
            "difficulty": "hard"
        },
        {
            "id": "Q4",
            "question_type": "calculation",
            "prompt": "已知 $$f(x)=x^2+2x+1$$，求 $$f(3)$$。",
            "difficulty": "easy"
        },
        {
            "id": "Q5",
            "question_type": "true_false",
            "prompt": "Protocol 只支持名义类型系统（nominal typing）。",
            "options": ["A. 正确", "B. 错误"],
            "difficulty": "easy"
        },
        {
            "id": "Q6",
            "question_type": "coding",
            "prompt": "实现一个函数，返回列表中的前 K 大元素。请给出 Python 代码。",
            "difficulty": "hard"
        },
        {
            "id": "Q7",
            "question_type": "multiple_choice",
            "prompt": "以下哪些属于 Python 常见内置数据类型？",
            "options": ["A. list", "B. dict", "C. tuple", "D. interface"],
            "difficulty": "easy"
        },
        {
            "id": "Q8",
            "question_type": "fill_blank",
            "prompt": "请填写：Python 中用于匿名函数的关键字是 ____，用于定义类的关键字是 ____。",
            "blank_count": 2,
            "difficulty": "easy"
        }
    ]
}

IMPORT_QUESTION_TYPE_MAP = {
    "single_choice": (QuestionType.single_choice.value, "单选题"),
    "single": (QuestionType.single_choice.value, "单选题"),
    "radio": (QuestionType.single_choice.value, "单选题"),
    "单选": (QuestionType.single_choice.value, "单选题"),
    "single_select": (QuestionType.single_choice.value, "单选题"),
    "multiple_choice": (QuestionType.multiple_choice.value, "多选题"),
    "multiple": (QuestionType.multiple_choice.value, "多选题"),
    "multi_select": (QuestionType.multiple_choice.value, "多选题"),
    "multi_choice": (QuestionType.multiple_choice.value, "多选题"),
    "multiple_select": (QuestionType.multiple_choice.value, "多选题"),
    "checkbox": (QuestionType.multiple_choice.value, "多选题"),
    "多选": (QuestionType.multiple_choice.value, "多选题"),
    "多选题": (QuestionType.multiple_choice.value, "多选题"),
    "多项选择": (QuestionType.multiple_choice.value, "多选题"),
    "多项选择题": (QuestionType.multiple_choice.value, "多选题"),
    "不定项选择题": (QuestionType.multiple_choice.value, "多选题"),
    "不定项选择": (QuestionType.multiple_choice.value, "多选题"),
    "true_false": (QuestionType.true_false.value, "判断题"),
    "boolean": (QuestionType.true_false.value, "判断题"),
    "tf": (QuestionType.true_false.value, "判断题"),
    "判断": (QuestionType.true_false.value, "判断题"),
    "fill_blank": (QuestionType.fill_blank.value, "填空题"),
    "blank": (QuestionType.fill_blank.value, "填空题"),
    "cloze_test": (QuestionType.fill_blank.value, "完形填空"),
    "填空": (QuestionType.fill_blank.value, "填空题"),
    "填空题": (QuestionType.fill_blank.value, "填空题"),
    "完形填空题": (QuestionType.fill_blank.value, "完形填空"),
    "多空填空": (QuestionType.fill_blank.value, "填空题"),
    "完形填空": (QuestionType.fill_blank.value, "完形填空"),
    "short_answer": (QuestionType.short_answer.value, "简答题"),
    "qa": (QuestionType.short_answer.value, "问答题"),
    "question_answer": (QuestionType.short_answer.value, "问答题"),
    "essay": (QuestionType.short_answer.value, "论述题"),
    "discussion": (QuestionType.short_answer.value, "论述题"),
    "case_analysis": (QuestionType.short_answer.value, "案例分析题"),
    "material": (QuestionType.short_answer.value, "材料分析题"),
    "calculation": (QuestionType.short_answer.value, "计算题"),
    "proof": (QuestionType.short_answer.value, "证明题"),
    "matching": (QuestionType.short_answer.value, "匹配题"),
    "ordering": (QuestionType.short_answer.value, "排序题"),
    "coding": (QuestionType.short_answer.value, "编程题"),
    "代码": (QuestionType.short_answer.value, "编程题"),
    "程序设计": (QuestionType.short_answer.value, "编程题"),
    "cloze": (QuestionType.fill_blank.value, "完形填空"),
    "阅读理解": (QuestionType.short_answer.value, "阅读理解题"),
    "应用题": (QuestionType.short_answer.value, "应用题"),
}


def clear_import_answer_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("import_answer_"):
            del st.session_state[key]
    st.session_state.import_answers_store = {}
    st.session_state.import_answers_store_digest = ""


def _sanitize_name(name: str, default_suffix: str = ".json") -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    if not base:
        base = "imported_paper_json"
    if "." not in base:
        base += default_suffix
    return base


def _persist_import_source(raw_text: str, source_name: str, source_kind: str) -> str:
    IMPORT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_name(source_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:10]
    save_path = IMPORT_ARCHIVE_DIR / f"{stamp}_{source_kind}_{content_hash}_{safe_name}"
    save_path.write_text(raw_text, encoding="utf-8")
    return str(save_path)


def _import_answer_snapshot_path(exam_hash: str) -> Path:
    IMPORT_ANSWER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    safe_hash = re.sub(r"[^a-fA-F0-9]", "", str(exam_hash or "").strip())[:64]
    if not safe_hash:
        safe_hash = "unknown"
    return IMPORT_ANSWER_SNAPSHOT_DIR / f"{safe_hash}.json"


def _persist_import_answers_snapshot(exam_hash: str) -> None:
    clean_hash = str(exam_hash or "").strip()
    if not clean_hash:
        return
    store = st.session_state.get("import_answers_store", {})
    if not isinstance(store, dict):
        return
    store_text = json.dumps(store, ensure_ascii=False, sort_keys=True)
    store_digest = hashlib.sha256(store_text.encode("utf-8")).hexdigest()
    if st.session_state.get("import_answers_store_digest") == store_digest:
        return
    payload = {
        "exam_hash": clean_hash,
        "updated_at": datetime.now().isoformat(),
        "answers_store": store,
    }
    _import_answer_snapshot_path(clean_hash).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.session_state.import_answers_store_digest = store_digest


def _load_import_answers_snapshot(exam_hash: str) -> dict[str, object]:
    clean_hash = str(exam_hash or "").strip()
    if not clean_hash:
        return {}
    file_path = _import_answer_snapshot_path(clean_hash)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    answers_store = payload.get("answers_store")
    if isinstance(answers_store, dict):
        store_text = json.dumps(answers_store, ensure_ascii=False, sort_keys=True)
        st.session_state.import_answers_store_digest = hashlib.sha256(store_text.encode("utf-8")).hexdigest()
        return answers_store
    return {}


def _sync_import_store_from_widgets(quiz: Quiz) -> None:
    for idx, question in enumerate(quiz.questions, start=1):
        key = import_answer_key(question, idx)
        widget_key = f"{key}_widget"
        qtype = QuestionType.normalize(getattr(question, "question_type", ""))
        options = [str(item) for item in (question.options or []) if str(item).strip()]

        if qtype == QuestionType.multiple_choice.value and options:
            items = _choice_items(options)
            full_options = [_choice_label(item) for item in items]
            cb_keys = [f"{widget_key}_cb_{i}" for i in range(1, len(full_options) + 1)]
            if any(cb in st.session_state for cb in cb_keys):
                selected = [
                    opt
                    for opt, cb in zip(full_options, cb_keys)
                    if bool(st.session_state.get(cb))
                ]
                _import_store_set(key, selected)
            continue

        if qtype == QuestionType.fill_blank.value and _import_blank_count(question) > 1:
            blank_count = _import_blank_count(question)
            blank_keys = [f"{widget_key}_blank_{i}" for i in range(1, blank_count + 1)]
            if any(bk in st.session_state for bk in blank_keys):
                _import_store_set(
                    key,
                    [str(st.session_state.get(bk, "") or "") for bk in blank_keys],
                )
            continue

        if widget_key in st.session_state:
            _import_store_set(key, st.session_state.get(widget_key))


def _normalize_import_options(raw: object) -> list[str]:
    if isinstance(raw, list):
        normalized: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                key = str(item.get("key") or item.get("label") or item.get("id") or "").strip()
                val = str(item.get("text") or item.get("value") or item.get("content") or "").strip()
                if not val:
                    continue
                normalized.append(f"{key}. {val}" if key else val)
            else:
                text = str(item).strip()
                if text:
                    normalized.append(text)
        return normalized
    if isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            k = str(key).strip()
            v = str(value).strip()
            if not v:
                continue
            items.append(f"{k}. {v}" if k else v)
        return items
    if isinstance(raw, str) and raw.strip():
        lines = [line.strip("- ").strip() for line in raw.splitlines()]
        return [line for line in lines if line]
    return []


def _resolve_import_type(raw_type: object, options: list[str]) -> tuple[str, str, str]:
    raw = str(raw_type or "").strip()
    token = raw.lower().replace(" ", "_").replace("-", "_")
    ui_type, label = IMPORT_QUESTION_TYPE_MAP.get(token, ("", ""))
    if not ui_type:
        if options:
            ui_type = QuestionType.multiple_choice.value if len(options) > 1 else QuestionType.single_choice.value
            label = "选择题"
        else:
            ui_type = QuestionType.short_answer.value
            label = "主观题"
    return ui_type, label, raw or label


def _extract_import_prompt(item: dict[str, Any]) -> str:
    candidates = [
        item.get("prompt"),
        item.get("question"),
        item.get("stem"),
        item.get("body"),
        item.get("content"),
        item.get("title"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _build_import_question(item: dict[str, Any], index: int) -> Question:
    options = _normalize_import_options(item.get("options") or item.get("choices"))
    qtype, type_label, raw_type = _resolve_import_type(item.get("question_type") or item.get("type"), options)
    if qtype == QuestionType.true_false.value and not options:
        options = ["正确", "错误"]

    blank_count = 0
    raw_blank_count = item.get("blank_count")
    if raw_blank_count is None:
        raw_blank_count = item.get("blanks")
    if raw_blank_count is None:
        raw_blank_count = item.get("slots")
    if isinstance(raw_blank_count, int):
        blank_count = max(0, raw_blank_count)
    elif isinstance(raw_blank_count, list):
        blank_count = len(raw_blank_count)
    elif isinstance(raw_blank_count, str) and raw_blank_count.strip().isdigit():
        blank_count = int(raw_blank_count.strip())
    if qtype == QuestionType.fill_blank.value and blank_count <= 0:
        prompt_text = _extract_import_prompt(item)
        inferred = len(re.findall(r"_{3,}|\[\s*\]|\(\s*\)", prompt_text))
        blank_count = inferred if inferred > 0 else 1

    reference_points = [str(v) for v in (item.get("reference_points") or []) if str(v).strip()]
    reference_points.extend([
        f"import_raw_type:{raw_type}",
        f"import_type_label:{type_label}",
    ])
    if blank_count > 0:
        reference_points.append(f"import_blank_count:{blank_count}")

    return Question(
        id=str(item.get("id") or item.get("qid") or item.get("no") or f"Q{index}"),
        question_type=qtype,
        prompt=_extract_import_prompt(item),
        options=options,
        correct_answer=[str(v) for v in (item.get("correct_answer") or []) if str(v).strip()],
        explanation=str(item.get("explanation") or ""),
        knowledge_tags=[str(v) for v in (item.get("knowledge_tags") or []) if str(v).strip()],
        difficulty=str(item.get("difficulty") or "medium"),
        reference_points=reference_points,
    )


def parse_import_exam_text(raw_text: str, file_name: str = "") -> Quiz:
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""

    if suffix == "jsonl":
        questions_raw: list[dict[str, Any]] = []
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            row = line.strip()
            if not row:
                continue
            try:
                item = json.loads(row)
            except Exception as exc:
                raise ValueError(f"JSONL 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL 第 {line_no} 行必须是对象。")
            questions_raw.append(item)
        title = "导入试卷"
        source_summary = "外部模型试卷（JSONL）"
    else:
        try:
            payload = json.loads(raw_text)
        except Exception as exc:
            raise ValueError(f"JSON 文件格式错误：{exc}") from exc
        if isinstance(payload, list):
            questions_raw = payload
            title = "导入试卷"
            source_summary = "外部模型试卷"
        elif isinstance(payload, dict):
            questions_raw = payload.get("questions")
            title = str(payload.get("title") or "导入试卷")
            source_summary = str(payload.get("source_summary") or "外部模型试卷")
        else:
            raise ValueError("JSON 根节点必须是对象或数组。")

    if not isinstance(questions_raw, list) or not questions_raw:
        raise ValueError("未读取到题目，请检查 JSON 的 `questions` 字段或 JSONL 内容。")

    questions: list[Question] = []
    for idx, item in enumerate(questions_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 题必须是对象。")
        question = _build_import_question(item, idx)
        if not question.prompt:
            raise ValueError(f"第 {idx} 题缺少题干字段（prompt/question/stem/body/content）。")
        questions.append(question)

    return Quiz(title=title, source_summary=source_summary, questions=questions)


def import_answer_key(question: Question, index: int) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(getattr(question, "id", f"q{index}")).strip() or f"q{index}")
    return f"import_answer_{index}_{safe_id}"


def _import_store_get(answer_key: str) -> object:
    store = st.session_state.get("import_answers_store", {})
    if isinstance(store, dict):
        return store.get(answer_key)
    return None


def _import_store_set(answer_key: str, value: object) -> None:
    store = st.session_state.setdefault("import_answers_store", {})
    if not isinstance(store, dict):
        store = {}
        st.session_state.import_answers_store = store
    if store.get(answer_key) == value:
        return
    store[answer_key] = value
    _persist_import_answers_snapshot(str(st.session_state.get("import_exam_hash", "") or ""))


def _import_meta(question: Question) -> tuple[str, str]:
    raw_type = ""
    type_label = ""
    for item in question.reference_points or []:
        line = str(item)
        if line.startswith("import_raw_type:"):
            raw_type = line.split(":", 1)[1].strip()
        elif line.startswith("import_type_label:"):
            type_label = line.split(":", 1)[1].strip()
    if not type_label:
        type_label = qtype_label(question.question_type)
    return raw_type or str(question.question_type), type_label


def _import_blank_count(question: Question) -> int:
    for item in question.reference_points or []:
        line = str(item)
        if line.startswith("import_blank_count:"):
            raw = line.split(":", 1)[1].strip()
            if raw.isdigit():
                return max(1, int(raw))
    return 1


def _count_import_answered(quiz: Quiz) -> int:
    answered = 0
    for idx, question in enumerate(quiz.questions, start=1):
        if _is_import_question_answered(question, idx):
            answered += 1
    return answered


def _first_unanswered_index(quiz: Quiz) -> int | None:
    for idx, question in enumerate(quiz.questions, start=1):
        if not _is_import_question_answered(question, idx):
            return idx
    return None


def _is_import_question_answered(question: Question, index: int) -> bool:
    key = import_answer_key(question, index)
    qtype = QuestionType.normalize(getattr(question, "question_type", ""))
    value = _import_store_get(key)
    if value is None:
        value = st.session_state.get(key)

    if qtype == QuestionType.fill_blank.value:
        blank_count = _import_blank_count(question)
        if blank_count <= 1:
            return isinstance(value, str) and bool(value.strip())
        if isinstance(value, list):
            valid = [str(v).strip() for v in value if str(v).strip()]
            return len(valid) >= blank_count
        return False

    if isinstance(value, list):
        return len([str(v).strip() for v in value if str(v).strip()]) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return False


def _import_type_progress(quiz: Quiz) -> list[tuple[str, int, int]]:
    progress: dict[str, dict[str, int]] = {}
    for idx, question in enumerate(quiz.questions, start=1):
        _, type_label = _import_meta(question)
        if type_label not in progress:
            progress[type_label] = {"total": 0, "answered": 0}
        progress[type_label]["total"] += 1
        if _is_import_question_answered(question, idx):
            progress[type_label]["answered"] += 1
    return sorted(
        [(label, item["answered"], item["total"]) for label, item in progress.items()],
        key=lambda x: x[0],
    )


def render_import_answer_sheet(quiz: Quiz, page_size: int) -> None:
    total = len(quiz.questions)
    if total <= 0:
        return

    answered = _count_import_answered(quiz)
    first_unanswered = _first_unanswered_index(quiz)
    total_pages = max(1, (total + page_size - 1) // page_size)
    current_page = int(st.session_state.get("import_page", 1) or 1)
    current_page = max(1, min(total_pages, current_page))
    start = (current_page - 1) * page_size + 1
    end = min(total, current_page * page_size)

    st.markdown("**作答记录**")
    st.progress(answered / max(1, total), text=f"已作答 {answered}/{total}")
    st.caption(f"当前页：{current_page}/{total_pages}（第 {start}-{end} 题）")
    if first_unanswered is not None:
        st.caption(f"首个未作答：第 {first_unanswered} 题")
    else:
        st.caption("当前全部题目已作答。")

    answered_numbers: list[int] = []
    page_status_lines: list[str] = []
    for idx, question in enumerate(quiz.questions, start=1):
        done = _is_import_question_answered(question, idx)
        if done:
            answered_numbers.append(idx)
        if start <= idx <= end:
            page_status_lines.append(f"{idx}. {'✅' if done else '⬜'}")

    preview = "、".join(str(i) for i in answered_numbers[:120])
    if len(answered_numbers) > 120:
        preview += f" ...（共 {len(answered_numbers)} 题）"
    if preview:
        st.caption("已作答题号")
        st.markdown(preview)
    else:
        st.caption("已作答题号：暂无")

    st.caption("当前页作答状态")
    st.markdown("  \n".join(page_status_lines) if page_status_lines else "暂无")


def _render_rich_text(text: str) -> None:
    content = str(text or "")
    pattern = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```|\$\$(.*?)\$\$", re.DOTALL)
    cursor = 0
    for match in pattern.finditer(content):
        prefix = content[cursor:match.start()]
        if prefix.strip():
            st.markdown(prefix)
        code_lang = match.group(1)
        code_body = match.group(2)
        latex_body = match.group(3)
        if code_body is not None:
            st.code(code_body.strip("\n"), language=(code_lang or None))
        elif latex_body is not None:
            latex = latex_body.strip()
            if latex:
                try:
                    st.latex(latex)
                except Exception:
                    st.markdown(f"$$\n{latex}\n$$")
        cursor = match.end()
    suffix = content[cursor:]
    if suffix.strip():
        st.markdown(suffix)


def _escape_markdown_inline(text: str) -> str:
    value = str(text or "")
    # Escape Markdown control chars so exported answer text keeps exact literal form.
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", value)


def _markdown_fence_wrap(text: str) -> list[str]:
    content = str(text or "")
    max_run = 0
    for match in re.finditer(r"`+", content):
        max_run = max(max_run, len(match.group(0)))
    fence = "`" * max(3, max_run + 1)
    return [f"{fence}text", content, fence]


def _choice_tokens(count: int) -> list[str]:
    base = [chr(ord("A") + i) for i in range(26)]
    if count <= 26:
        return base[:count]
    tokens = base[:]
    idx = 27
    while len(tokens) < count:
        tokens.append(f"O{idx}")
        idx += 1
    return tokens


def _choice_items(options: list[str]) -> list[dict[str, str]]:
    default_tokens = _choice_tokens(len(options))
    items: list[dict[str, str]] = []
    used_tokens: set[str] = set()
    pattern = re.compile(r"^\s*([A-Za-z0-9]+)[\.\)、:：]\s*(.+)\s*$")
    for idx, raw_opt in enumerate(options):
        raw = str(raw_opt).strip()
        token = default_tokens[idx]
        body = raw
        matched = pattern.match(raw)
        if matched:
            parsed_token = matched.group(1).strip().upper()
            parsed_body = matched.group(2).strip()
            if parsed_body:
                token = parsed_token
                body = parsed_body
        if token in used_tokens:
            token = default_tokens[idx]
        used_tokens.add(token)
        items.append({"token": token, "body": body})
    return items


def _choice_label(item: dict[str, str]) -> str:
    body = re.sub(r"\s+", " ", item["body"]).strip()
    return f"{item['token']}. {body}"


def _render_checkbox_group(
    base_key: str,
    options: list[str],
    existing_values: set[str] | None = None,
) -> list[str]:
    selected_values: list[str] = []
    existing = existing_values or set()
    for opt_idx, opt in enumerate(options, start=1):
        cb_key = f"{base_key}_cb_{opt_idx}"
        checked = st.checkbox(opt, key=cb_key, value=(opt in existing))
        if checked:
            selected_values.append(opt)
    return selected_values


def _convert_choice_answer(raw_value: object, options: list[str]) -> list[str]:
    items = _choice_items(options)
    mapping = {item["token"]: f"{item['token']}. {item['body']}" for item in items}

    def _convert_one(value: str) -> str:
        v = value.strip()
        if v in mapping:
            return mapping[v]
        for token, full in mapping.items():
            if v == full:
                return full
            if v.endswith(full):
                return full
        return v

    if isinstance(raw_value, list):
        return [_convert_one(str(v)) for v in raw_value if str(v).strip()]
    if isinstance(raw_value, str) and raw_value.strip():
        return [_convert_one(raw_value)]
    return []


def render_import_exam_questions(quiz: Quiz, page: int, page_size: int) -> None:
    total = len(quiz.questions)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(max(1, page), total_pages)
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    st.caption(f"当前显示第 {start + 1}-{end} 题，共 {total} 题（第 {page}/{total_pages} 页）")

    for idx in range(start, end):
        question = quiz.questions[idx]
        index = idx + 1
        key = import_answer_key(question, index)
        widget_key = f"{key}_widget"
        qtype = QuestionType.normalize(getattr(question, "question_type", ""))
        options = [str(item) for item in (question.options or []) if str(item).strip()]
        raw_type, type_label = _import_meta(question)
        stored_value = _import_store_get(key)

        st.markdown(f"**第 {index} 题（{type_label}）**")
        st.caption(f"题型：{type_label}")
        prompt_box = st.container(border=True)
        with prompt_box:
            _render_rich_text(question.prompt)

        if qtype == QuestionType.single_choice.value:
            if options:
                items = _choice_items(options)
                full_options = [_choice_label(item) for item in items]
                selected_index = None
                if isinstance(stored_value, str) and stored_value in full_options:
                    selected_index = full_options.index(stored_value)
                elif isinstance(stored_value, str):
                    converted = _convert_choice_answer(stored_value, options)
                    if converted and converted[0] in full_options:
                        selected_index = full_options.index(converted[0])
                st.radio(
                    "选择答案",
                    full_options,
                    key=widget_key,
                    index=selected_index,
                    label_visibility="collapsed",
                )
                _import_store_set(key, st.session_state.get(widget_key))
            else:
                st.text_input(
                    "作答",
                    key=widget_key,
                    value=str(stored_value or ""),
                    label_visibility="collapsed",
                )
                _import_store_set(key, st.session_state.get(widget_key, ""))
        elif qtype == QuestionType.multiple_choice.value:
            if options:
                items = _choice_items(options)
                full_options = [_choice_label(item) for item in items]
                existing_values: list[str] = []
                if isinstance(stored_value, list):
                    existing_values = [str(v).strip() for v in stored_value if str(v).strip()]
                elif isinstance(stored_value, str) and stored_value.strip():
                    existing_values = _convert_choice_answer(stored_value, options)
                selected = _render_checkbox_group(
                    base_key=widget_key,
                    options=full_options,
                    existing_values=set(existing_values),
                )
                _import_store_set(key, selected)
            else:
                st.text_input(
                    "作答（多个答案用逗号分隔）",
                    key=widget_key,
                    value=str(stored_value or ""),
                    label_visibility="collapsed",
                )
                raw = str(st.session_state.get(widget_key, "") or "")
                _import_store_set(key, [part.strip() for part in raw.split(",") if part.strip()])
        elif qtype == QuestionType.true_false.value:
            tf_options = options or ["正确", "错误"]
            selected_index = None
            if isinstance(stored_value, str) and stored_value in tf_options:
                selected_index = tf_options.index(stored_value)
            st.radio(
                "判断",
                tf_options,
                key=widget_key,
                index=selected_index,
                horizontal=True,
                label_visibility="collapsed",
            )
            _import_store_set(key, st.session_state.get(widget_key))
        elif qtype == QuestionType.fill_blank.value:
            blank_count = _import_blank_count(question)
            if blank_count <= 1:
                default_text = ""
                if isinstance(stored_value, list):
                    default_text = str(stored_value[0]) if stored_value else ""
                elif isinstance(stored_value, str):
                    default_text = stored_value
                st.text_input(
                    "填写答案",
                    key=widget_key,
                    value=default_text,
                    label_visibility="collapsed",
                )
                _import_store_set(key, st.session_state.get(widget_key, ""))
            else:
                values: list[str] = []
                for blank_idx in range(blank_count):
                    blank_key = f"{widget_key}_blank_{blank_idx + 1}"
                    default_text = ""
                    if isinstance(stored_value, list) and blank_idx < len(stored_value):
                        default_text = str(stored_value[blank_idx] or "")
                    text = st.text_input(
                        f"第 {blank_idx + 1} 空",
                        key=blank_key,
                        value=default_text,
                    )
                    values.append(text)
                _import_store_set(key, values)
        else:
            height = 200 if type_label in {"论述题", "案例分析题", "材料分析题", "编程题"} else 120
            st.text_area(
                "作答",
                key=widget_key,
                value=str(stored_value or ""),
                height=height,
                label_visibility="collapsed",
            )
            _import_store_set(key, st.session_state.get(widget_key, ""))


def collect_import_answers(quiz: Quiz) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, question in enumerate(quiz.questions, start=1):
        key = import_answer_key(question, idx)
        qtype = QuestionType.normalize(getattr(question, "question_type", ""))
        raw_type, type_label = _import_meta(question)
        value = _import_store_get(key)
        if value is None:
            value = st.session_state.get(key)
        options = [str(item) for item in (question.options or []) if str(item).strip()]
        answered_flag = _is_import_question_answered(question, idx)

        if qtype == QuestionType.multiple_choice.value and options:
            answer = _convert_choice_answer(value, options)
        elif qtype == QuestionType.single_choice.value and options:
            answer = _convert_choice_answer(value, options)
        elif qtype == QuestionType.fill_blank.value:
            blank_count = _import_blank_count(question)
            if isinstance(value, list):
                answer = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str):
                if blank_count > 1:
                    answer = [part.strip() for part in re.split(r"[;\n]+", value) if part.strip()]
                else:
                    answer = [value.strip()] if value.strip() else []
            else:
                answer = []
        elif qtype in {QuestionType.single_choice.value, QuestionType.true_false.value}:
            answer = [value] if isinstance(value, str) and value.strip() else []
        elif qtype == QuestionType.multiple_choice.value:
            if isinstance(value, list):
                answer = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str):
                answer = [part.strip() for part in value.split(",") if part.strip()]
            else:
                answer = []
        else:
            answer = [value.strip()] if isinstance(value, str) and value.strip() else []

        rows.append(
            {
                "index": idx,
                "question_id": question.id,
                "mapped_question_type": qtype,
                "display_type": type_label,
                "original_question_type": raw_type,
                "prompt": question.prompt,
                "answered": answered_flag,
                "answer": answer,
            }
        )
    return rows


def _refresh_import_source(raw_text: str, source_name: str, source_kind: str) -> None:
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if st.session_state.get("import_exam_hash") == content_hash:
        if not st.session_state.get("import_answers_store"):
            restored_answers = _load_import_answers_snapshot(content_hash)
            if restored_answers:
                st.session_state.import_answers_store = restored_answers
        return
    saved_path = _persist_import_source(raw_text, source_name, source_kind)
    st.session_state.import_exam_hash = content_hash
    st.session_state.import_source_text = raw_text
    st.session_state.import_source_name = source_name
    st.session_state.import_source_saved_path = saved_path
    st.session_state.import_quiz_hash = ""
    st.session_state.import_quiz_data = None
    st.session_state.import_page = 1
    st.session_state.import_show_answer_record = False
    st.session_state.import_answer_json = ""
    st.session_state.import_answer_md = ""
    clear_import_answer_state()
    restored_answers = _load_import_answers_snapshot(content_hash)
    if restored_answers:
        st.session_state.import_answers_store = restored_answers


def render_import_exam_mode() -> None:
    render_section_head(
        "试卷导入作答台",
        "导入整份试卷后分页作答，支持代码与 LaTeX 显示，并导出完整答案供外部大模型评估。",
    )

    st.markdown("**支持格式**")
    st.markdown("- `JSON`：根节点可为对象（含 `questions`）或题目数组")
    st.markdown("- `JSONL`：每行一个题目对象，适合大题量")
    st.markdown("- 支持上传文件或直接粘贴 JSON/JSONL 文本")

    with st.expander("查看试卷格式", expanded=False):
        st.code(json.dumps(IMPORT_EXAM_TEMPLATE, ensure_ascii=False, indent=2), language="json")
        st.caption("每题至少包含：`id` + 题干字段（`prompt/question/stem/body/content` 之一）。")
        st.caption("题干与选项支持：Markdown、代码块（```）和 LaTeX（$$...$$）。")
        st.caption(
            "支持题型（示例）：single_choice、multiple_choice、true_false、fill_blank、short_answer、essay、case_analysis、material、calculation、proof、matching、ordering、coding，以及中文别名（单选/多选/判断/填空/编程等）。"
        )

    st.download_button(
        "下载 JSON 模板",
        data=json.dumps(IMPORT_EXAM_TEMPLATE, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="quizmind_import_template.json",
        mime="application/json",
        width="stretch",
        key="download_import_template",
        on_click="ignore",
    )

    input_mode = st.radio(
        "输入来源",
        ["上传文件", "粘贴 JSON/JSONL 文本"],
        horizontal=True,
        key="import_input_mode",
    )

    if input_mode == "上传文件":
        uploaded = st.file_uploader("上传试卷文件（JSON / JSONL）", type=["json", "jsonl"], key="import_exam_file")
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = raw_bytes.decode("utf-8-sig", errors="replace")
            _refresh_import_source(raw_text, uploaded.name, "upload")
    else:
        paste_text = st.text_area(
            "粘贴试卷 JSON/JSONL 文本",
            key="import_pasted_text",
            height=220,
            placeholder="请粘贴完整 JSON 或 JSONL 内容...",
        )
        if st.button("加载粘贴内容", width="stretch", key="btn_load_pasted_exam"):
            if not paste_text.strip():
                st.warning("请先粘贴 JSON/JSONL 文本。")
            else:
                _refresh_import_source(paste_text.strip(), "pasted_exam.json", "paste")
                st.success("已加载并归档粘贴内容。")

    source_text = str(st.session_state.get("import_source_text", "") or "")
    source_name = str(st.session_state.get("import_source_name", "") or "")
    saved_path = str(st.session_state.get("import_source_saved_path", "") or "")

    if not source_text:
        st.info("请先上传试卷文件或粘贴 JSON/JSONL 文本。")
        return

    if saved_path:
        st.caption(f"已归档源文件：`{saved_path}`")

    try:
        if st.session_state.get("import_quiz_hash") != st.session_state.get("import_exam_hash"):
            quiz_obj = parse_import_exam_text(source_text, source_name)
            st.session_state.import_quiz_data = quiz_obj.model_dump()
            st.session_state.import_quiz_hash = st.session_state.get("import_exam_hash")
        quiz = Quiz.model_validate(st.session_state.import_quiz_data)
    except Exception as exc:
        st.error(f"试卷解析失败：{exc}")
        return

    _sync_import_store_from_widgets(quiz)
    total = len(quiz.questions)
    answered = _count_import_answered(quiz)
    page_size = int(st.session_state.get("import_page_size", 50) or 50)
    with st.sidebar:
        st.markdown("### 刷题辅助")
        st.toggle(
            "显示作答记录面板",
            key="import_show_answer_record",
            help="默认隐藏，开启后显示实时作答记录。",
        )

    st.markdown(f"### {quiz.title}")
    st.caption(f"题目数量：{total}")
    st.progress(answered / max(1, total), text=f"已作答 {answered} / {total}")
    type_progress = _import_type_progress(quiz)
    if type_progress:
        st.caption(
            "题型进度："
            + " | ".join(f"{label} {done}/{all_count}" for label, done, all_count in type_progress)
        )

    nav1, nav2 = st.columns(2)
    with nav1:
        page_size = st.selectbox("每页题数", [20, 50, 100, 200], index=1, key="import_page_size")
    with nav2:
        total_pages = max(1, (total + page_size - 1) // page_size)
        current_page = int(st.session_state.get("import_page", 1) or 1)
        current_page = max(1, min(total_pages, current_page))
        page = int(
            st.number_input(
                "页码",
                min_value=1,
                max_value=total_pages,
                step=1,
                value=current_page,
            )
        )
        st.session_state.import_page = page

    with st.sidebar:
        if st.session_state.get("import_show_answer_record", False):
            with st.expander("作答记录面板", expanded=True):
                render_import_answer_sheet(quiz, page_size=page_size)

    render_import_exam_questions(quiz, page=page, page_size=page_size)
    _persist_import_answers_snapshot(str(st.session_state.get("import_exam_hash", "") or ""))

    _sync_import_store_from_widgets(quiz)
    answers = collect_import_answers(quiz)
    payload = {
        "title": quiz.title,
        "source_summary": quiz.source_summary,
        "source_name": source_name,
        "source_archive_path": saved_path,
        "exported_at": datetime.now().isoformat(),
        "answered_count": answered,
        "question_count": total,
        "type_progress": [
            {"type": label, "answered": done, "total": all_count}
            for label, done, all_count in type_progress
        ],
        "answers": answers,
    }
    st.session_state.import_answer_json = json.dumps(payload, ensure_ascii=False, indent=2)

    answer_md_lines = [
        f"# {quiz.title} - 作答记录",
        "",
        f"- 导出时间: {payload['exported_at']}",
        f"- 来源: {source_name}",
        f"- 归档路径: {saved_path}",
        f"- 已作答: {answered}/{total}",
        "",
    ]
    for item in answers:
        answer_text = (
            "；".join(_escape_markdown_inline(x) for x in item["answer"])
            if item["answer"]
            else "(未作答)"
        )
        answer_md_lines.append(f"## 第 {item['index']} 题 ({item['question_id']})")
        answer_md_lines.append(
            f"- 题型: {_escape_markdown_inline(item['display_type'])} | 原始类型: {_escape_markdown_inline(item['original_question_type'])}"
        )
        answer_md_lines.extend(_markdown_fence_wrap(str(item["prompt"])))
        answer_md_lines.append(f"- 作答: {answer_text}")
        answer_md_lines.append("")

    st.session_state.import_answer_md = "\n".join(answer_md_lines)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "下载答案（JSON）",
        data=st.session_state.import_answer_json.encode("utf-8"),
        file_name=f"quizmind_answers_{stamp}.json",
        mime="application/json",
        width="stretch",
        key="download_import_answers_json",
        on_click="ignore",
    )
    st.download_button(
        "下载答案（Markdown）",
        data=st.session_state.import_answer_md.encode("utf-8"),
        file_name=f"quizmind_answers_{stamp}.md",
        mime="text/markdown",
        width="stretch",
        key="download_import_answers_md",
        on_click="ignore",
    )
def show_model_fallback_notice(provider) -> None:
    chain = list(getattr(provider, "last_model_chain", []) or [])
    used_model = str(getattr(provider, "last_used_model", "") or "").strip()
    if len(chain) <= 1 or not used_model or used_model == chain[0]:
        return
    st.warning(
        f"本次大模型请求已自动切换到备用模型：{used_model}（尝试链路：{' -> '.join(chain)}）"
    )




def init_state() -> None:
    defaults = {
        "parsed": None,
        "quiz": None,
        "report": None,
        "reinforcement_quiz": None,
        "memory_snapshot": None,
        "source_text": "",
        "source_type": "text",
        "source_name": "当前输入",
        "exam_deadline": None,
        "exam_timeout_processed": False,
        "last_generation_mode": QuizMode.practice.value,
        "flow_step": 1,
        "last_session_logged_key": "",
        "learning_style": "老师模式",
        "quiz_origin_label": "",
        "import_answer_json": "",
        "import_answer_md": "",
        "import_exam_hash": "",
        "import_source_text": "",
        "import_source_name": "",
        "import_source_saved_path": "",
        "import_quiz_hash": "",
        "import_quiz_data": None,
        "import_answers_store": {},
        "import_answers_store_digest": "",
        "import_pasted_text": "",
        "import_input_mode": "上传文件",
        "import_page": 1,
        "import_page_size": 50,
        "import_show_answer_record": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.flow_step = max(1, min(4, int(st.session_state.flow_step)))



def goto_step(step: int) -> None:
    st.session_state.flow_step = max(1, min(4, step))


def clear_answer_state() -> None:
    to_delete = [k for k in st.session_state.keys() if k.startswith("answer_")]
    for key in to_delete:
        del st.session_state[key]


def set_current_quiz(quiz, origin_label: str | None = None) -> None:
    clear_answer_state()
    st.session_state.quiz = quiz
    if origin_label is not None:
        st.session_state.quiz_origin_label = str(origin_label).strip()


def default_origin_label(engine: QuizEngine, allow_ai_generation: bool) -> str:
    return engine.last_quiz_origin or ("AI生成" if allow_ai_generation else "本地规则")


def apply_quiz_to_session(
    quiz: Quiz | None,
    parsed,
    config: QuizConfig,
    engine: QuizEngine,
    user_store: UserFeatureStore,
    origin_label: str,
) -> None:
    guarded_quiz = apply_quality_guard(
        quiz,
        parsed,
        config,
        engine,
        user_store,
    )
    set_current_quiz(guarded_quiz, origin_label=origin_label)
    persist_resume_context(user_store, parsed, guarded_quiz, origin_label)


def reset_result_state() -> None:
    st.session_state.report = None
    st.session_state.reinforcement_quiz = None


def persist_resume_context(
    user_store: UserFeatureStore,
    parsed,
    quiz: Quiz | None,
    origin_label: str,
) -> None:
    if parsed is None or quiz is None:
        return
    user_store.save_resume_context(
        parsed=parsed,
        quiz=quiz,
        source_name=st.session_state.get("source_name", "当前输入"),
        origin_label=origin_label,
        mode=str(st.session_state.get("last_generation_mode", QuizMode.practice.value)),
    )


def ensure_reinforcement_parsed(
    engine: QuizEngine,
    active_config: QuizConfig,
    allow_ai_generation: bool,
    query: str,
) -> object:
    parsed = st.session_state.parsed
    if parsed is not None:
        return parsed
    parsed, _ = engine.generate_from_memory(
        config=active_config,
        query=query,
        top_k=4,
        allow_ai_generation=allow_ai_generation,
        learning_style=learning_style_key(),
    )
    st.session_state.parsed = parsed
    return parsed


def quiz_session_key(quiz: Quiz, report) -> str:
    question_fingerprint = "|".join(
        f"{q.id}:{re.sub(r'\\s+', ' ', q.prompt or '').strip()[:80]}"
        for q in (quiz.questions or [])
    )
    raw = json.dumps(
        {
            "source_name": st.session_state.get("source_name", "当前输入"),
            "quiz_title": quiz.title,
            "overall_score": float(report.overall_score),
            "wrong_count": len(report.wrong_questions),
            "question_fingerprint": question_fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_smart_reinforcement_quiz(
    engine: QuizEngine,
    active_config: QuizConfig,
    allow_ai_generation: bool,
):
    quiz = st.session_state.quiz
    report = st.session_state.report
    if not quiz or not report:
        return None, None

    method = learning_method_strategy(report, quiz)
    parsed = ensure_reinforcement_parsed(
        engine=engine,
        active_config=active_config,
        allow_ai_generation=allow_ai_generation,
        query=" ".join(report.reinforcement_topics or []),
    )
    dominant = str(method["dominant_cause"])
    focus_topics = list(method["focus_topics"])
    diff_mix = dict(method["difficulty_mix"])
    type_mix = dict(method["type_mix"])
    question_count = min(
        max(int(method["question_count"]), len(report.wrong_questions) * 2),
        20,
    )
    reinforcement_quiz = build_targeted_quiz(
        parsed,
        focus_topics,
        active_config,
        question_count=question_count,
        allow_ai_generation=allow_ai_generation,
        difficulty_mix=diff_mix,
        type_mix=type_mix,
        learning_style=learning_style_key(),
        strict_ai_generation=engine.strict_ai_generation,
    )
    return dominant, reinforcement_quiz






def collect_source() -> tuple[str, str]:
    input_mode = st.radio(
        "输入方式",
        ["粘贴文本", "上传文件（可多选）", "网页 URL"],
        horizontal=True,
    )
    if input_mode == "粘贴文本":
        text = st.text_area(
            "学习内容",
            height=220,
            placeholder="把课程笔记、知识点、题目解析粘贴到这里...",
        )
        st.session_state.source_name = "粘贴文本"
        return text, "text"

    if input_mode == "上传文件（可多选）":
        files = st.file_uploader(
            "上传 PDF / DOCX / Markdown / TXT",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
        )
        if not files:
            return "", "file"

        build_mode = st.radio(
            "生成方式",
            ["综合生成（合并多个文件）", "单文件生成（从列表选择）"],
            horizontal=True,
        )

        loaded: list[tuple[str, str]] = []
        errors: list[str] = []
        for file in files:
            try:
                text = read_uploaded_text(file.name, file.getvalue())
                if text.strip():
                    loaded.append((file.name, text))
                else:
                    errors.append(f"{file.name}: 内容为空")
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")

        if errors:
            st.warning("以下文件读取失败：")
            for line in errors:
                st.caption(f"- {line}")

        if not loaded:
            return "", "file"

        st.caption(f"已成功读取 {len(loaded)} 个文件")
        names = [name for name, _ in loaded]
        text_map = {name: text for name, text in loaded}

        if build_mode == "综合生成（合并多个文件）":
            merged = "\n\n".join(
                [f"### 文件：{name}\n{text_map[name]}" for name in names]
            )
            st.session_state.source_name = (
                f"综合文件({len(names)}): "
                + ", ".join(names[:3])
                + ("..." if len(names) > 3 else "")
            )
            return merged, "file"

        selected_name = st.selectbox(
            "选择用于本次生成的文件", names, key="single_source_file"
        )
        st.session_state.source_name = selected_name
        return text_map[selected_name], "file"

    url = st.text_input("网页地址", placeholder="https://example.com/article")
    st.session_state.source_name = "网页 URL"
    return (load_text_from_url(url.strip()), "url") if url.strip() else ("", "url")


def render_sidebar(
    engine: QuizEngine,
) -> tuple[QuizConfig, str, str, int, bool]:
    st.sidebar.markdown("### 核心配置")
    source_mode = st.sidebar.radio(
        "内容模式",
        ["current", "memory"],
        index=0,
        format_func=lambda x: "当前输入" if x == "current" else "记忆库",
    )
    allow_ai_generation = st.sidebar.checkbox("启用 AI 出题", value=True)
    question_count = st.sidebar.slider("题目数量", 5, 20, 10, 1)
    difficulty_preset = st.sidebar.selectbox(
        "难度",
        ["均衡", "偏基础", "偏进阶"],
        index=0,
    )

    difficulty_map = {
        "均衡": {"easy": 30, "medium": 50, "hard": 20},
        "偏基础": {"easy": 50, "medium": 35, "hard": 15},
        "偏进阶": {"easy": 15, "medium": 45, "hard": 40},
    }
    mix = difficulty_map[difficulty_preset]

    memory_query = ""
    memory_top_k = 4
    if source_mode == "memory":
        memory_query = st.sidebar.text_input("检索词", placeholder="例如：Python 面试")
        memory_top_k = st.sidebar.slider("召回片段", 2, 8, 4, 1)
        snapshots = engine.list_memory()
        st.sidebar.caption(f"记忆快照：{len(snapshots)}")
        if not snapshots:
            st.sidebar.info("记忆库为空，请先在输入区保存内容。")

    config = QuizConfig(
        question_count=question_count,
        difficulty_mix=mix,
        type_mix={
            "single_choice": 40,
            "multiple_choice": 15,
            "fill_blank": 20,
            "short_answer": 15,
            "true_false": 10,
        },
    )
    return (config, source_mode, memory_query, memory_top_k, allow_ai_generation)


def reset_exam_state() -> None:
    st.session_state.exam_deadline = None
    st.session_state.exam_timeout_processed = False


def start_exam_if_needed(quiz_mode: QuizMode, exam_minutes: int) -> None:
    st.session_state.last_generation_mode = quiz_mode.value
    if quiz_mode == QuizMode.exam:
        deadline = datetime.now() + timedelta(minutes=exam_minutes)
        st.session_state.exam_deadline = deadline.isoformat()
        st.session_state.exam_timeout_processed = False
        log_event(
            "exam.start",
            deadline=st.session_state.exam_deadline,
            duration_minutes=exam_minutes,
        )
    else:
        reset_exam_state()


def load_answers_from_state() -> List[UserAnswer]:
    quiz = st.session_state.quiz
    if not quiz:
        return []
    answers: List[UserAnswer] = []
    for idx, question in enumerate(quiz.questions, start=1):
        key = answer_state_key(question, idx)
        qtype = QuestionType.normalize(getattr(question, "question_type", ""))
        value = st.session_state.get(key)
        if qtype in {QuestionType.single_choice.value, QuestionType.true_false.value}:
            answer = [value] if value else []
        elif qtype == QuestionType.multiple_choice.value:
            answer = value if isinstance(value, list) else []
        else:
            answer = [value] if isinstance(value, str) and value.strip() else []
        answers.append(UserAnswer(question_id=question.id, answer=answer))
    return answers


def answered_count() -> int:
    return sum(1 for item in load_answers_from_state() if item.answer)


def learning_style_key() -> str:
    return {
        "简洁模式": "concise",
        "老师模式": "teacher",
        "面试官模式": "interviewer",
    }.get(st.session_state.get("learning_style", "老师模式"), "teacher")


def process_exam_timeout(grader: GradingService) -> None:
    if st.session_state.last_generation_mode != QuizMode.exam.value:
        return
    if (
        not st.session_state.quiz
        or st.session_state.report
        or not st.session_state.exam_deadline
    ):
        return
    if st.session_state.exam_timeout_processed:
        return

    deadline = datetime.fromisoformat(st.session_state.exam_deadline)
    if datetime.now() < deadline:
        return

    st.session_state.report = grader.grade(
        st.session_state.quiz,
        load_answers_from_state(),
        learning_style=learning_style_key(),
    )
    st.session_state.exam_timeout_processed = True
    goto_step(4)
    log_event("exam.auto_submit", quiz_title=st.session_state.quiz.title)


def render_exam_timer() -> None:
    if (
        st.session_state.last_generation_mode != QuizMode.exam.value
        or not st.session_state.exam_deadline
    ):
        return
    if st.session_state.report:
        return

    deadline = datetime.fromisoformat(st.session_state.exam_deadline)
    remaining = deadline - datetime.now()
    remaining_seconds = max(0, int(remaining.total_seconds()))
    mins, secs = divmod(remaining_seconds, 60)
    st.error(f"考试倒计时：{mins:02d}:{secs:02d}（到时自动交卷）")
    components.html(
        f"""
        <script>
        const deadline = new Date("{deadline.isoformat()}").getTime();
        function tick() {{
          if (deadline - Date.now() <= 0) {{
            window.parent.location.reload();
          }}
        }}
        setInterval(tick, 1000);
        </script>
        """,
        height=0,
    )








def apply_quality_guard(
    quiz: Quiz | None,
    parsed,
    config: QuizConfig,
    engine: QuizEngine,
    user_store: UserFeatureStore,
) -> Quiz | None:
    if not quiz or not getattr(quiz, "questions", None):
        return quiz
    blocked_tags = user_store.blocked_tags(min_votes=2)
    if not blocked_tags:
        return quiz

    original_total = len(quiz.questions)
    filtered = [
        q
        for q in quiz.questions
        if not (set(q.knowledge_tags or []) & blocked_tags)
    ]
    dropped = original_total - len(filtered)
    if dropped <= 0:
        return quiz

    fallback = engine.generate_quiz(
        parsed,
        config,
        allow_ai_generation=True,
        learning_style=learning_style_key(),
    )
    existing = {q.id for q in filtered}
    existing_sig = {
        re.sub(r"[^\w\u4e00-\u9fff]", "", (q.prompt or "").strip().lower())
        for q in filtered
    }
    for item in fallback.questions:
        if len(filtered) >= original_total:
            break
        prompt_sig = re.sub(r"[^\w\u4e00-\u9fff]", "", (item.prompt or "").strip().lower())
        if item.id in existing:
            continue
        if prompt_sig in existing_sig:
            continue
        if set(item.knowledge_tags or []) & blocked_tags:
            continue
        filtered.append(item)
        existing.add(item.id)
        existing_sig.add(prompt_sig)

    st.info(f"已根据历史反馈替换 {dropped} 道低质量倾向题目。")
    if len(filtered) < original_total:
        st.warning(f"质量过滤后仅保留 {len(filtered)} 道高质量题，建议重新生成。")
    return quiz.model_copy(update={"questions": filtered})


def log_learning_session_if_needed(user_store: UserFeatureStore) -> None:
    report = st.session_state.get("report")
    quiz = st.session_state.get("quiz")
    if not report or not quiz:
        return
    method = learning_method_strategy(report, quiz)
    session_key = quiz_session_key(quiz, report)
    if st.session_state.get("last_session_logged_key") == session_key:
        return
    user_store.add_learning_session(
        source_name=st.session_state.get("source_name", "当前输入"),
        quiz_title=quiz.title,
        overall_score=report.overall_score,
        objective_accuracy=report.objective_accuracy,
        subjective_average=report.subjective_average,
        wrong_count=len(report.wrong_questions),
        weak_topics=list(report.reinforcement_topics),
        knowledge_stats=[item.model_dump() for item in report.knowledge_stats],
        stage_key=str(method.get("stage_key", "")),
        stage_title=str(method.get("stage_title", "")),
    )
    user_store.clear_resume_context()
    st.session_state.last_session_logged_key = session_key


def _learning_path_plan(user_store: UserFeatureStore, report) -> list[str]:
    method = learning_method_strategy(report, st.session_state.get("quiz"))
    weak_now = [item for item in (report.knowledge_stats or []) if item.avg_score < 70]
    weak_now.sort(key=lambda item: item.avg_score)
    recent = user_store.recent_sessions(limit=40)
    topic_counts: dict[str, int] = {}
    for session in recent:
        for topic in session.get("weak_topics", []) or []:
            key = str(topic).strip()
            if key:
                topic_counts[key] = topic_counts.get(key, 0) + 1
    recurring = [k for k, _ in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
    aligned_focus = _daily_training_focus(
        user_store,
        report,
        st.session_state.get("quiz"),
    )

    steps: list[str] = []
    steps.append(f"当前阶段：{method['stage_title']}，{method['tagline']}")
    if aligned_focus:
        steps.append(f"本轮聚焦：{'、'.join(aligned_focus[:3])}")
    elif weak_now:
        first = weak_now[0]
        steps.append(f"本轮聚焦：{first.knowledge_point}（当前均分 {first.avg_score:.1f}）")
    if aligned_focus:
        steps.append(
            f"下一轮训练：{'、'.join(aligned_focus[:2])}，建议 {int(method['question_count'])} 题，按“{method['stage_title']}”方式推进。"
        )
    elif len(weak_now) > 1:
        second = weak_now[1]
        steps.append(
            f"下一轮训练：{second.knowledge_point}，建议 {int(method['question_count'])} 题，按“{method['stage_title']}”方式推进。"
        )
    elif recurring:
        steps.append(
            f"下一轮训练：{recurring[0]}，建议 {int(method['question_count'])} 题，覆盖不同题型。"
        )
    if recurring:
        steps.append(
            f"连续巩固：{recurring[0]} + {recurring[1] if len(recurring) > 1 else recurring[0]}，连续 3 天更容易稳住。"
        )
    if not steps:
        steps.append("保持：当前整体掌握较好，建议继续每日训练 10 题。")
    return steps


def _daily_training_focus(user_store: UserFeatureStore, report, quiz: Quiz | None = None) -> list[str]:
    topics: list[str] = []
    if report and quiz:
        method_topics = [
            str(item).strip()
            for item in learning_method_strategy(report, quiz).get("focus_topics", [])
            if str(item).strip()
        ]
        topics.extend(method_topics)
    if report and getattr(report, "reinforcement_topics", None):
        topics.extend([str(x).strip() for x in report.reinforcement_topics if str(x).strip()])
    recent = user_store.recent_sessions(limit=40)
    counts: dict[str, int] = {}
    for session in recent:
        for topic in session.get("weak_topics", []) or []:
            key = str(topic).strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    ranked = [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
    for item in ranked:
        if item not in topics:
            topics.append(item)
        if len(topics) >= 4:
            break
    return topics[:4]


def _error_cause_training_strategy(report, quiz: Quiz | None) -> tuple[str, list[str], dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    qmap = {q.id: q for q in (quiz.questions if quiz else [])}
    cause_topics: dict[str, list[str]] = {}

    for wrong in report.wrong_questions:
        cause = str(getattr(wrong, "error_category", "") or "reasoning_error").strip()
        counts[cause] = counts.get(cause, 0) + 1
        q = qmap.get(wrong.question_id)
        if not q:
            continue
        cause_topics.setdefault(cause, []).extend(q.knowledge_tags or [])

    if not counts:
        return (
            "none",
            [],
            {"easy": 25, "medium": 50, "hard": 25},
            {
                "single_choice": 35,
                "multiple_choice": 25,
                "fill_blank": 20,
                "short_answer": 15,
                "true_false": 5,
            },
        )

    dominant = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
    raw_topics = cause_topics.get(dominant, [])
    focus_topics = [x for x in dict.fromkeys([str(i).strip() for i in raw_topics if str(i).strip()])]

    by_cause = {
        "concept_unclear": (
            {"easy": 50, "medium": 40, "hard": 10},
            {"single_choice": 35, "multiple_choice": 15, "fill_blank": 30, "short_answer": 10, "true_false": 10},
        ),
        "knowledge_forgotten": (
            {"easy": 55, "medium": 35, "hard": 10},
            {"single_choice": 40, "multiple_choice": 15, "fill_blank": 25, "short_answer": 10, "true_false": 10},
        ),
        "reasoning_error": (
            {"easy": 20, "medium": 45, "hard": 35},
            {"single_choice": 20, "multiple_choice": 30, "fill_blank": 15, "short_answer": 30, "true_false": 5},
        ),
        "careless_mistake": (
            {"easy": 35, "medium": 50, "hard": 15},
            {"single_choice": 45, "multiple_choice": 15, "fill_blank": 20, "short_answer": 10, "true_false": 10},
        ),
        "expression_issue": (
            {"easy": 30, "medium": 50, "hard": 20},
            {"single_choice": 15, "multiple_choice": 20, "fill_blank": 10, "short_answer": 50, "true_false": 5},
        ),
    }
    diff_mix, type_mix = by_cause.get(
        dominant,
        (
            {"easy": 25, "medium": 50, "hard": 25},
            {"single_choice": 35, "multiple_choice": 25, "fill_blank": 20, "short_answer": 15, "true_false": 5},
        ),
    )
    return dominant, focus_topics, diff_mix, type_mix


def learning_method_strategy(report, quiz: Quiz | None) -> dict[str, object]:
    dominant, focus_topics, diff_mix, type_mix = _error_cause_training_strategy(report, quiz)
    strategy_map = {
        "concept_unclear": {
            "stage_key": "comprehend",
            "stage_title": "理解建图",
            "tagline": "先把知识点之间的边界和定义彻底分开。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "先做低到中难度题，确保概念判断稳定。",
            "question_count": 8,
        },
        "knowledge_forgotten": {
            "stage_key": "recall",
            "stage_title": "主动回忆",
            "tagline": "优先唤醒记忆，而不是继续被动阅读。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "通过填空、判断和短题重新激活记忆。",
            "question_count": 8,
        },
        "reasoning_error": {
            "stage_key": "reason",
            "stage_title": "推理纠偏",
            "tagline": "重点不是记住答案，而是修正推理链。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "增加中高难度和简答占比，逼出完整推理。",
            "question_count": 10,
        },
        "careless_mistake": {
            "stage_key": "calibrate",
            "stage_title": "细节校准",
            "tagline": "你的核心问题不是不会，而是不够稳。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "用高相似度题目训练审题和排错。",
            "question_count": 8,
        },
        "expression_issue": {
            "stage_key": "express",
            "stage_title": "表达固化",
            "tagline": "思路有了，接下来要把答案说完整、说清楚。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "增加简答题和结构化作答训练。",
            "question_count": 8,
        },
        "none": {
            "stage_key": "advance",
            "stage_title": "综合进阶",
            "tagline": "当前不是修漏洞，而是提升稳定性和上限。",
            "method_name": "QuizMind CORE 学习法",
            "next_action": "进入综合混合训练，扩大题型覆盖。",
            "question_count": 10,
        },
    }
    picked = strategy_map.get(dominant, strategy_map["reasoning_error"]).copy()
    picked.update(
        {
            "dominant_cause": dominant,
            "focus_topics": focus_topics,
            "difficulty_mix": diff_mix,
            "type_mix": type_mix,
            "question_count": min(max(int(picked["question_count"]), 6), 20),
        }
    )
    return picked


def learning_stage_progress(
    user_store: UserFeatureStore,
    report,
    quiz: Quiz | None,
) -> dict[str, object] | None:
    if not report or not quiz:
        return None

    method = learning_method_strategy(report, quiz)
    stage_order = [
        ("comprehend", "理解建图"),
        ("recall", "主动回忆"),
        ("reason", "推理纠偏"),
        ("calibrate", "细节校准"),
        ("express", "表达固化"),
        ("advance", "综合进阶"),
    ]
    stage_titles = {key: title for key, title in stage_order}
    stage_index = {key: idx for idx, (key, _) in enumerate(stage_order)}
    current_key = str(method.get("stage_key", "reason"))
    current_idx = stage_index.get(current_key, stage_index["reason"])

    recent = user_store.recent_sessions(limit=12)
    stage_history: list[tuple[str, str]] = []
    for item in recent:
        key = str(item.get("stage_key", "")).strip()
        title = str(item.get("stage_title", "")).strip()
        if key or title:
            stage_history.append((key, title))

    previous_key = ""
    previous_title = ""
    for key, title in stage_history:
        normalized_key = key or current_key
        normalized_title = title or stage_titles.get(normalized_key, "")
        if normalized_key == current_key and normalized_title == str(method["stage_title"]):
            continue
        previous_key = normalized_key
        previous_title = normalized_title
        break

    same_streak = 0
    for key, title in stage_history:
        normalized_key = key or current_key
        normalized_title = title or stage_titles.get(normalized_key, "")
        if normalized_key == current_key and normalized_title == str(method["stage_title"]):
            same_streak += 1
        else:
            break

    overall_score = float(getattr(report, "overall_score", 0.0) or 0.0)
    wrong_count = len(getattr(report, "wrong_questions", []) or [])
    if current_key == "advance":
        target_key = "advance"
    elif overall_score >= 85 and wrong_count <= 2:
        target_key = stage_order[min(len(stage_order) - 1, current_idx + 1)][0]
    else:
        target_key = current_key

    target_title = stage_titles.get(target_key, stage_titles["advance"])
    phase_status = (
        f"连续 {same_streak + 1} 轮处于当前阶段，建议先把这一阶段练稳。"
        if same_streak >= 1 and target_key == current_key
        else f"如果本轮继续稳定，下一步可以推进到「{target_title}」。"
        if target_key != current_key
        else "当前阶段还没练透，建议继续按这个阶段推进。"
    )
    previous_text = previous_title or "起点评估"
    journey = f"{previous_text} -> {method['stage_title']} -> {target_title}"

    return {
        "current_key": current_key,
        "current_title": str(method["stage_title"]),
        "previous_title": previous_text,
        "target_key": target_key,
        "target_title": target_title,
        "journey": journey,
        "status": phase_status,
    }


def stage_feedback_profile(method: dict[str, object]) -> dict[str, object]:
    stage_key = str(method.get("stage_key", "reason"))
    profiles = {
        "comprehend": {
            "review_title": "本阶段复盘重点：概念边界",
            "coach_tip": "先分清概念定义、适用条件和相似概念差异，再继续刷题。",
            "self_checks": [
                "能不能用自己的话重新定义这个知识点。",
                "能不能说出它和相近概念的关键区别。",
                "看到选项时，能不能先排除边界不符合的干扰项。",
            ],
        },
        "recall": {
            "review_title": "本阶段复盘重点：主动回忆",
            "coach_tip": "先回忆再看答案，目标是把知识从“见过”变成“能提取”。",
            "self_checks": [
                "做题前能不能先口头回忆核心结论。",
                "错题里遗漏的是定义、步骤还是关键词。",
                "下一轮是否能在不看材料的情况下答出 70% 以上。",
            ],
        },
        "reason": {
            "review_title": "本阶段复盘重点：推理链",
            "coach_tip": "不要只记答案，要把题干条件、推理步骤和结论重新串起来。",
            "self_checks": [
                "能不能按“条件 -> 推理 -> 结论”复述一遍。",
                "中间哪一步跳得太快，导致结论失真。",
                "如果题目换个场景，这条推理链还成不成立。",
            ],
        },
        "calibrate": {
            "review_title": "本阶段复盘重点：审题与排错",
            "coach_tip": "你更需要稳定性训练，要抓限定词、单位、范围和否定表达。",
            "self_checks": [
                "这题的限定词和关键条件有没有漏看。",
                "错误发生在审题、选项比较还是最后确认。",
                "下一题能否先做 3 秒核对再提交。",
            ],
        },
        "express": {
            "review_title": "本阶段复盘重点：表达完整度",
            "coach_tip": "思路可能已经对了，接下来要把答案组织成可得分的结构。",
            "self_checks": [
                "答案里有没有明确的结论句。",
                "关键要点是否分点说清楚，而不是堆成一段。",
                "有没有把因果、条件和结果写完整。",
            ],
        },
        "advance": {
            "review_title": "本阶段复盘重点：综合稳定性",
            "coach_tip": "现在不是补漏洞，而是保持稳定并扩大覆盖面。",
            "self_checks": [
                "高频题型下是否还能保持稳定得分。",
                "陌生题型出现时，是否能快速迁移已有方法。",
                "能否在更高难度下继续保持清晰表达。",
            ],
        },
    }
    return profiles.get(stage_key, profiles["reason"])


def stage_answering_profile(method: dict[str, object] | None, qtype: str) -> dict[str, str]:
    stage_key = str((method or {}).get("stage_key", "reason"))
    by_stage = {
        "comprehend": {
            "hint": "先判断这题考的是哪个概念，再区分它和相近概念的边界。",
            "short_placeholder": "先写定义，再写适用条件，最后补一句和相近概念的区别。",
            "check": "提交前确认：我答的是这个概念本身，而不是相近概念。",
        },
        "recall": {
            "hint": "先不看解析，尝试先把关键结论从记忆里提出来。",
            "short_placeholder": "先默写你记得的关键词，再补充完整答案。",
            "check": "提交前确认：我是在回忆知识，而不是看到题干后临时猜测。",
        },
        "reason": {
            "hint": "按“条件 -> 推理 -> 结论”顺序思考，不要直接跳到答案。",
            "short_placeholder": "先写已知条件，再写推理过程，最后写结论。",
            "check": "提交前确认：中间推理链有没有断点或跳步。",
        },
        "calibrate": {
            "hint": "这类训练重点是稳，先圈出限定词、范围和否定表达再作答。",
            "short_placeholder": "先写你抓到的限定条件，再给出最终答案。",
            "check": "提交前确认：单位、范围、限定词、否定词有没有漏看。",
        },
        "express": {
            "hint": "你的目标不是想到就停，而是把答案组织成可得分的结构。",
            "short_placeholder": "用“结论 + 要点1/2/3”来组织答案。",
            "check": "提交前确认：结论句、分点和因果关系是否表达完整。",
        },
        "advance": {
            "hint": "保持稳定并扩大覆盖面，优先用你最稳的方法完成作答。",
            "short_placeholder": "写出最稳妥、最完整的答案，尽量覆盖不同角度。",
            "check": "提交前确认：这题如果换个场景，我是否还能稳定答对。",
        },
    }
    profile = by_stage.get(stage_key, by_stage["reason"]).copy()
    if qtype != QuestionType.short_answer.value:
        profile["short_placeholder"] = ""
    return profile


def _feynman_wrong_question_tasks(report, quiz: Quiz) -> list[dict[str, str]]:
    qmap = {q.id: q for q in (quiz.questions or [])}
    cause_cn = {
        "concept_unclear": "概念不清",
        "careless_mistake": "粗心错误",
        "reasoning_error": "推理错误",
        "knowledge_forgotten": "知识遗忘",
        "expression_issue": "表达问题",
        "none": "未分类",
    }
    tasks: list[dict[str, str]] = []
    for wrong in list(report.wrong_questions or [])[:5]:
        q = qmap.get(wrong.question_id)
        if not q:
            continue
        topic = (q.knowledge_tags or ["该知识点"])[0]
        cause = cause_cn.get(str(getattr(wrong, "error_category", "") or "none"), "未分类")
        tasks.append(
            {
                "question_id": str(wrong.question_id),
                "topic": str(topic),
                "cause": cause,
                "prompt": str(q.prompt),
                "template": f"请用 6 岁小朋友能听懂的话解释「{topic}」：它是什么、什么时候用、为什么这题容易错。",
            }
        )
    return tasks


def render_hardcore_mastery_panel(report, quiz: Quiz, user_store: UserFeatureStore) -> None:
    st.markdown("**硬核掌握闭环（费曼 + 主动回忆 + 交错训练 + 间隔复习）**")
    st.markdown("1. 费曼复述：把错题知识点讲给初学者，暴露自己讲不清的地方。")
    st.markdown("2. 主动回忆：遮住答案，先写定义、条件、步骤，再核对。")
    st.markdown("3. 交错训练：混合不同题型，避免只会做同套路题。")
    st.markdown("4. 间隔复习：按 24 小时、72 小时、7 天安排短测。")

    tasks = _feynman_wrong_question_tasks(report, quiz)
    if not tasks:
        st.caption("本轮无错题，建议直接进入迁移应用训练。")
        return

    st.markdown("**费曼复述任务卡（按错题生成）**")
    for idx, task in enumerate(tasks, start=1):
        with st.expander(f"任务 {idx}：{task['topic']}（{task['cause']}）", expanded=idx == 1):
            st.markdown(f"题号：`{task['question_id']}`")
            st.markdown(task["prompt"])
            st.caption(task["template"])
            st.text_area(
                "我的复述",
                key=f"feynman_note_{task['question_id']}",
                height=100,
                placeholder="按“定义 -> 条件 -> 常见误区 -> 例子”来讲。",
            )

    focus = _daily_training_focus(user_store, report, quiz)
    if focus:
        st.markdown("**下一轮交错训练建议**")
        st.caption(
            "将以下主题交错练习，每次 8-12 题，主客观题混合："
            + "、".join(focus[:4])
        )







def qtype_label(qtype: QuestionType | str) -> str:
    normalized = QuestionType.normalize(qtype)
    mapping = {
        QuestionType.single_choice.value: "单选题",
        QuestionType.multiple_choice.value: "多选题",
        QuestionType.fill_blank.value: "填空题",
        QuestionType.short_answer.value: "简答题",
        QuestionType.true_false.value: "判断题",
    }
    return mapping.get(normalized, normalized)


def diff_label(value: str) -> str:
    return {"easy": "简单", "medium": "中等", "hard": "困难"}.get(value, value)


def answer_state_key(question: Question, index: int) -> str:
    safe_id = str(getattr(question, "id", f"q{index}")).strip() or f"q{index}"
    return f"answer_{safe_id}"


def render_question_widget(
    question: Question,
    index: int,
    user_store: UserFeatureStore,
    source_title: str,
    is_favorited: bool,
    method: dict[str, object] | None = None,
) -> None:
    key = answer_state_key(question, index)
    qtype = QuestionType.normalize(getattr(question, "question_type", ""))
    answer_profile = stage_answering_profile(method, qtype)
    options = [str(item) for item in (question.options or []) if str(item).strip()]
    if qtype == QuestionType.true_false.value:
        if not ({"正确", "错误"}.issubset(set(options))):
            options = ["正确", "错误"]

    prompt = (question.prompt or "").strip() or "（题干缺失，请重新生成该题）"
    difficulty_value = getattr(question.difficulty, "value", str(question.difficulty))
    tags = question.knowledge_tags[:2] if question.knowledge_tags else ["未标注"]
    safe_id = html.escape(str(question.id))
    safe_type = html.escape(qtype_label(question.question_type))
    safe_diff = html.escape(diff_label(str(difficulty_value)))
    safe_tags = html.escape("/".join(tags))
    st.markdown(
        f'<div class="qm-question-card">'
        f'<div class="qm-question-head"><div class="qm-question-title">{safe_id} · {safe_type}</div></div>'
        f'<div class="qm-question-meta"><span class="qm-chip">{safe_diff}</span>'
        f'<span class="qm-chip">{safe_tags}</span></div></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        _render_rich_text(prompt)
    st.caption(answer_profile["hint"])
    c1, c2 = st.columns([1, 2])
    with c1:
        if is_favorited:
            if st.button("取消收藏", key=f"unfav_{key}", width="stretch"):
                user_store.remove_favorite(user_store.question_fingerprint(question))
                st.rerun()
        else:
            if st.button("收藏这题", key=f"fav_{key}", width="stretch"):
                user_store.add_favorite(question, source_title=source_title)
                st.success("已加入收藏题单。")
    with c2:
        feedback_choice = st.selectbox(
            "题目反馈",
            ["暂不反馈", "太简单", "太难", "题干不清晰", "疑似错误"],
            key=f"feedback_choice_{key}",
            label_visibility="collapsed",
        )
        if feedback_choice != "暂不反馈":
            if st.button("提交反馈", key=f"fb_submit_{key}", width="stretch"):
                user_store.add_quality_feedback(
                    question,
                    verdict=feedback_choice,
                    source_title=source_title,
                )
                st.success("反馈已记录，后续将用于题目质量优化。")

    if qtype == QuestionType.single_choice.value:
        if options:
            st.radio("选择一个", options, key=key, index=None, label_visibility="collapsed")
        else:
            st.text_input("作答", key=key, label_visibility="collapsed")
            st.caption("选项缺失，已切换为文本作答。")
    elif qtype == QuestionType.multiple_choice.value:
        if options:
            existing = set(st.session_state.get(key, []))
            st.session_state[key] = _render_checkbox_group(
                base_key=key,
                options=options,
                existing_values=existing,
            )
        else:
            st.text_input("作答", key=key, label_visibility="collapsed")
            st.caption("选项缺失，已切换为文本作答。")
    elif qtype == QuestionType.true_false.value:
        st.radio(
            "判断正误", options, key=key, index=None, label_visibility="collapsed"
        )
    else:
        st.text_area(
            "作答",
            key=key,
            height=120,
            label_visibility="collapsed",
            placeholder=answer_profile["short_placeholder"],
        )
    st.caption(answer_profile["check"])


def render_all_questions(quiz, user_store: UserFeatureStore) -> None:
    origin = str(st.session_state.get("quiz_origin_label", "")).strip()
    report = st.session_state.get("report")
    current_method = learning_method_strategy(report, quiz) if report and quiz else None
    if origin:
        st.markdown(
            f'<div class="qm-origin-badge">题目来源：{html.escape(origin)}</div>',
            unsafe_allow_html=True,
        )
    total = len(quiz.questions)
    answered = answered_count()
    st.progress(answered / max(1, total), text=f"已作答 {answered} / {total}")
    favorite_set = {
        str(item.get("fingerprint"))
        for item in user_store.list_favorites()
        if item.get("fingerprint")
    }

    type_counter: dict[str, int] = {}
    for question in quiz.questions:
        t = QuestionType.normalize(getattr(question, "question_type", ""))
        type_counter[t] = type_counter.get(t, 0) + 1
    st.caption("题型分布: " + ", ".join(f"{qtype_label(k)}={v}" for k, v in sorted(type_counter.items())))

    for idx, question in enumerate(quiz.questions, start=1):
        st.markdown(f"**第 {idx} 题**")
        try:
            render_question_widget(
                question,
                idx,
                user_store=user_store,
                source_title=st.session_state.get("source_name", "当前输入"),
                is_favorited=user_store.question_fingerprint(question) in favorite_set,
                method=current_method or st.session_state.get("daily_training_method"),
            )
        except Exception as exc:
            log_event(
                "ui.render_question.error",
                question_id=str(getattr(question, "id", idx)),
                question_type=str(getattr(question, "question_type", "")),
                error=str(exc),
            )
            st.error(f"第 {idx} 题渲染失败: {exc}")


def render_parsed_summary() -> None:
    parsed = st.session_state.parsed
    if not parsed:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("知识点", len(parsed.knowledge_points))
    c2.metric("文本分段", len(parsed.segments))
    c3.metric("核心概念", len(parsed.concepts))
    for point in parsed.knowledge_points[:10]:
        st.markdown(
            f"- **{point.name}** | 难度：{diff_label(point.difficulty.value)} | 关键词：{', '.join(point.keywords) or '无'}  \n"
            f"  {point.summary}"
        )
    with st.expander("查看完整解析原文", expanded=False):
        st.text_area(
            "完整解析内容",
            value=parsed.cleaned_text,
            height=280,
            disabled=True,
            key=f"parsed_full_text_{parsed.title}",
        )




def render_report(user_store: UserFeatureStore) -> None:
    report = st.session_state.report
    quiz = st.session_state.quiz
    if not report or not quiz:
        return
    method = learning_method_strategy(report, quiz)
    stage_progress = learning_stage_progress(user_store, report, quiz)
    feedback_profile = stage_feedback_profile(method)

    c1, c2, c3 = st.columns(3)
    c1.metric("综合得分", f"{report.overall_score:.1f}")
    c2.metric("客观题正确率", f"{report.objective_accuracy:.1f}%")
    c3.metric("主观题均分", f"{report.subjective_average:.1f}")

    achievement = (
        "掌握稳定，可以进入更高强度训练。"
        if report.overall_score >= 85
        else "已完成一轮有效训练，下一轮重点应该更聚焦。"
        if report.overall_score >= 60
        else "这轮暴露出了真实薄弱点，正适合立刻进入强化训练。"
    )
    st.markdown(
        (
            '<div class="qm-mission-card">'
            '<div class="qm-mission-title">本轮结论</div>'
            f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;margin-top:0;">{html.escape(achievement)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    focus_topics = list(method["focus_topics"])
    focus_text = "、".join(focus_topics[:3]) if focus_topics else "综合薄弱点"
    st.markdown(
        (
            '<div class="qm-mission-card">'
            '<div class="qm-mission-title">QuizMind CORE 学习法</div>'
            f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;margin-top:0;">当前阶段：{html.escape(str(method["stage_title"]))} | '
            f'聚焦：{html.escape(focus_text)}</div>'
            f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;">{html.escape(str(method["tagline"]))}</div>'
            f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;">下一轮训练：{html.escape(str(method["next_action"]))}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if stage_progress:
        st.markdown(
            (
                '<div class="qm-mission-card">'
                '<div class="qm-mission-title">阶段推进</div>'
                f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;margin-top:0;">'
                f'学习轨迹：{html.escape(str(stage_progress["journey"]))}</div>'
                f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;">'
                f'{html.escape(str(stage_progress["status"]))}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        (
            '<div class="qm-mission-card">'
            f'<div class="qm-mission-title">{html.escape(str(feedback_profile["review_title"]))}</div>'
            f'<div class="qm-card-hint" style="font-size:14px;color:#c9d1d9;margin-top:0;">'
            f'{html.escape(str(feedback_profile["coach_tip"]))}</div>'
            '<ul class="qm-mission-list">'
            + "".join(
                f"<li>{html.escape(str(item))}</li>"
                for item in list(feedback_profile["self_checks"])[:3]
            )
            + "</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("**知识点掌握情况**")
    for item in report.knowledge_stats:
        st.markdown(
            f"- **{item.knowledge_point}** | 状态：{item.status} | 平均分：{item.avg_score:.1f} | 通过率：{item.accuracy:.1f}%"
        )

    error_category_cn = {
        "concept_unclear": "概念不清",
        "careless_mistake": "粗心错误",
        "reasoning_error": "推理错误",
        "knowledge_forgotten": "知识遗忘",
        "expression_issue": "表达问题",
        "none": "无",
    }

    st.markdown("**错题集**")
    qmap = {q.id: q for q in quiz.questions}
    if not report.wrong_questions:
        st.success("本轮没有错题，掌握得不错。")
    for wrong in report.wrong_questions:
        q = qmap.get(wrong.question_id)
        if not q:
            continue
        breakdown = wrong.score_breakdown or {}
        breakdown_text = " | ".join(
            [
                f"正确性 {breakdown.get('correctness', 0):.1f}/10",
                f"完整性 {breakdown.get('completeness', 0):.1f}/10",
                f"清晰度 {breakdown.get('clarity', 0):.1f}/10",
            ]
        )
        structured = (wrong.structured_explanation or "").strip()
        st.markdown(
            f"- **{wrong.question_id} {q.prompt}** \n"
            f"你的答案：{', '.join(wrong.user_answer) or '未作答'}  \n"
            f"正确答案：{', '.join(wrong.correct_answer)}  \n"
            f"错因：{error_category_cn.get(wrong.error_category, wrong.error_category or '未分类')}  \n"
            f"评分维度：{breakdown_text}  \n"
            f"点评：{wrong.feedback}  \n"
            f"本阶段任务：{list(feedback_profile['self_checks'])[0]}"
        )
        if structured:
            safe_structured = html.escape(structured).replace("\n", "<br>")
            st.markdown(
                f'<div class="qm-wrap-text">{safe_structured}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("**复习建议**")
    st.caption(str(feedback_profile["coach_tip"]))
    for line in report.review_recommendations:
        st.markdown(f"- {line}")

    st.markdown("**建议学习路径**")
    for idx, line in enumerate(_learning_path_plan(user_store, report), start=1):
        st.markdown(f"{idx}. {line}")
    render_hardcore_mastery_panel(report, quiz, user_store)





def render_smart_mode_status() -> None:
    parsed_ready = st.session_state.get("parsed") is not None
    quiz = st.session_state.get("quiz")
    quiz_ready = bool(quiz and getattr(quiz, "questions", None))
    report_ready = st.session_state.get("report") is not None
    total = len(quiz.questions) if quiz_ready else 0
    answered = answered_count() if quiz_ready else 0

    steps = [
        ("1 输入", parsed_ready),
        ("2 组卷", quiz_ready),
        ("3 作答", report_ready or (quiz_ready and answered > 0)),
        ("4 复盘", report_ready),
    ]
    status_line = " | ".join(f"{name}{' [ok]' if done else ''}" for name, done in steps)
    st.caption(f"流程状态：{status_line}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("到输入", width="stretch", key="jump_step_1"):
            goto_step(1)
            st.rerun()
    with c2:
        if st.button("到组卷", width="stretch", key="jump_step_2"):
            goto_step(2)
            st.rerun()
    with c3:
        if st.button("到作答", width="stretch", key="jump_step_3"):
            goto_step(3)
            st.rerun()
    with c4:
        if st.button("到复盘", width="stretch", key="jump_step_4"):
            goto_step(4)
            st.rerun()

    if quiz_ready:
        st.progress(answered / max(1, total), text=f"作答进度 {answered}/{total}")

def main() -> None:
    init_state()
    inject_styles()
    content_service, engine, grader, user_store = get_services()

    st.markdown('<div class="qm-page-title">QuizMind 极简练习</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qm-page-subtitle">输入内容 -> 生成题目 -> 作答 -> 复盘</div>',
        unsafe_allow_html=True,
    )
    app_mode = st.radio(
        "模式",
        ["智能练习模式", "试卷导入作答台"],
        horizontal=True,
        key="app_mode",
    )
    if app_mode == "试卷导入作答台":
        render_import_exam_mode()
        return

    config, source_mode, memory_query, memory_top_k, allow_ai_generation = render_sidebar(engine)
    quiz_mode = QuizMode.practice
    exam_minutes = 0
    use_saved_first = True
    active_config = config
    st.caption("左侧仅保留核心参数，页面仅展示主流程。")
    render_smart_mode_status()

    with st.expander("1) 输入与解析", expanded=st.session_state.flow_step == 1):
        render_section_head("输入与解析", "建议先解析再出题，也支持一键解析并出题。")
        if source_mode == "current":
            source_text, source_type = collect_source()
            st.session_state.source_text = source_text
            st.session_state.source_type = source_type
        else:
            source_text, source_type = "", "memory"
            st.info("记忆模式将从向量库召回内容后自动组卷。")

        c1, c2 = st.columns(2)
        with c1:
            quick_action_label = "从记忆库出题" if source_mode == "memory" else "一键解析并出题"
            if st.button(quick_action_label, type="primary", width="stretch"):
                if source_mode == "memory":
                    if not engine.list_memory():
                        st.warning("记忆库为空，请先保存一份学习内容。")
                    else:
                        try:
                            with st.spinner("正在从记忆库召回内容并组卷..."):
                                parsed, quiz = engine.generate_from_memory(
                                    config=active_config,
                                    query=memory_query,
                                    top_k=memory_top_k,
                                    allow_ai_generation=allow_ai_generation,
                                    learning_style=learning_style_key(),
                                )
                                st.session_state.parsed = parsed
                                apply_quiz_to_session(
                                    quiz=quiz,
                                    parsed=parsed,
                                    config=active_config,
                                    engine=engine,
                                    user_store=user_store,
                                    origin_label=default_origin_label(engine, allow_ai_generation),
                                )
                                show_model_fallback_notice(engine.provider)
                                reset_result_state()
                                start_exam_if_needed(quiz_mode, exam_minutes)
                                goto_step(3)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"记忆出题失败：{exc}")
                elif not source_text.strip():
                    st.warning("请先输入学习内容。")
                else:
                    try:
                        with st.spinner("正在执行一键流程：组卷 -> 入库..."):
                            st.session_state.source_text = source_text
                            st.session_state.source_type = source_type
                            parsed, quiz, meta = engine.generate_or_load_from_source(
                                source=source_text,
                                source_type=source_type,
                                source_name=st.session_state.get("source_name", "当前输入"),
                                config=active_config,
                                use_saved_first=use_saved_first,
                                allow_ai_generation=allow_ai_generation,
                                learning_style=learning_style_key(),
                            )
                            st.session_state.parsed = parsed
                            apply_quiz_to_session(
                                quiz=quiz,
                                parsed=parsed,
                                config=active_config,
                                engine=engine,
                                user_store=user_store,
                                origin_label=str(meta.get("origin_label", "") or engine.last_quiz_origin),
                            )
                            show_model_fallback_notice(engine.provider)
                            reset_result_state()
                            try:
                                snapshot = engine.save_memory(parsed)
                                st.session_state.memory_snapshot = snapshot
                            except Exception as exc:
                                log_event("flow.one_click.save_memory_failed", error=str(exc))
                            start_exam_if_needed(quiz_mode, exam_minutes)
                            goto_step(3)
                            st.success("已生成题目并进入作答区。")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"一键出题失败：{exc}")
        with c2:
            if st.button("保存到记忆库", width="stretch"):
                if source_mode == "memory":
                    st.warning("记忆模式会直接从记忆库组卷，无需重复保存。")
                elif not source_text.strip():
                    st.warning("请先输入学习内容。")
                else:
                    parsed_to_save = content_service.parse(source_text, source_type)
                    st.session_state.parsed = parsed_to_save
                    snapshot = engine.save_memory(parsed_to_save)
                    st.session_state.memory_snapshot = snapshot
                    st.success(f"已保存：{snapshot.title}（{snapshot.chunks} 段）")
        render_parsed_summary()
    with st.expander("2) 组卷", expanded=st.session_state.flow_step == 2):
        render_section_head("组卷", "根据当前配置生成题目，可切换 AI/本地策略。")
        can_generate = (
            (source_mode == "memory" and bool(engine.list_memory()))
            or (source_mode == "current" and bool(st.session_state.parsed or source_text.strip()))
        )
        if st.button("生成题目", width="stretch", disabled=not can_generate):
            try:
                if source_mode == "current":
                    if st.session_state.parsed is not None:
                        generated = engine.generate_quiz(
                            st.session_state.parsed,
                            active_config,
                            allow_ai_generation=allow_ai_generation,
                            learning_style=learning_style_key(),
                        )
                        apply_quiz_to_session(
                            quiz=generated,
                            parsed=st.session_state.parsed,
                            config=active_config,
                            engine=engine,
                            user_store=user_store,
                            origin_label=default_origin_label(engine, allow_ai_generation),
                        )
                        show_model_fallback_notice(engine.provider)
                        st.success("已基于解析结果生成题目。")
                    elif source_text.strip():
                        parsed, quiz, meta = engine.generate_or_load_from_source(
                            source=source_text,
                            source_type=source_type,
                            source_name=st.session_state.get("source_name", "当前输入"),
                            config=active_config,
                            use_saved_first=use_saved_first,
                            allow_ai_generation=allow_ai_generation,
                            learning_style=learning_style_key(),
                        )
                        st.session_state.parsed = parsed
                        apply_quiz_to_session(
                            quiz=quiz,
                            parsed=parsed,
                            config=active_config,
                            engine=engine,
                            user_store=user_store,
                            origin_label=str(meta.get("origin_label", "") or engine.last_quiz_origin),
                        )
                        show_model_fallback_notice(engine.provider)
                        st.success("已生成并保存新题目。")
                    else:
                        st.warning("请先输入学习内容。")
                        return
                else:
                    if not engine.list_memory():
                        st.warning("记忆库为空，请先在“输入与解析”中解析并保存内容。")
                        return
                    parsed, quiz = engine.generate_from_memory(
                        config=active_config,
                        query=memory_query,
                        top_k=memory_top_k,
                        allow_ai_generation=allow_ai_generation,
                        learning_style=learning_style_key(),
                    )
                    st.session_state.parsed = parsed
                    apply_quiz_to_session(
                        quiz=quiz,
                        parsed=parsed,
                        config=active_config,
                        engine=engine,
                        user_store=user_store,
                        origin_label=default_origin_label(engine, allow_ai_generation),
                    )
                    show_model_fallback_notice(engine.provider)
                reset_result_state()
                start_exam_if_needed(quiz_mode, exam_minutes)
                goto_step(3)
                st.rerun()
            except Exception as exc:
                st.error(f"生成题目失败：{exc}")

    with st.expander("3) 作答", expanded=st.session_state.flow_step == 3):
        render_section_head("作答", "按题型作答后提交，考试模式支持倒计时自动交卷。")
        process_exam_timeout(grader)
        if not st.session_state.quiz:
            st.info("还没有题目，请先完成组卷。")
        else:
            current_mode = (
                "考试模式"
                if st.session_state.last_generation_mode == QuizMode.exam.value
                else "练习模式"
            )
            st.markdown(f"当前模式：`{current_mode}`")
            st.caption(f"当前讲解风格：{st.session_state.get('learning_style', '老师模式')}")
            st.caption("硬核作答法：先回忆再作答，最后补一句“为什么是这个答案”。")
            render_exam_timer()
            if not getattr(st.session_state.quiz, "questions", None):
                st.warning("当前题目为空，可能是生成或加载失败。")
                if st.session_state.parsed and st.button("重新生成题目", type="primary", width="stretch"):
                    generated = engine.generate_quiz(
                        st.session_state.parsed,
                        active_config,
                        allow_ai_generation=allow_ai_generation,
                        learning_style=learning_style_key(),
                    )
                    apply_quiz_to_session(
                        quiz=generated,
                        parsed=st.session_state.parsed,
                        config=active_config,
                        engine=engine,
                        user_store=user_store,
                        origin_label=default_origin_label(engine, allow_ai_generation),
                    )
                    show_model_fallback_notice(engine.provider)
                    reset_result_state()
                    reset_exam_state()
                    start_exam_if_needed(quiz_mode, exam_minutes)
                    st.rerun()
            else:
                render_all_questions(st.session_state.quiz, user_store)

            submit_label = (
                "提交练习"
                if st.session_state.last_generation_mode == QuizMode.practice.value
                else "交卷"
            )
            total_questions = len(getattr(st.session_state.quiz, "questions", []) or [])
            answered_questions = answered_count() if total_questions else 0
            if total_questions and answered_questions < total_questions:
                st.caption(f"未作答题数：{total_questions - answered_questions}")
            can_submit = total_questions > 0 and answered_questions > 0
            if not st.session_state.report and st.button(
                submit_label,
                type="primary",
                width="stretch",
                disabled=not can_submit,
            ):
                st.session_state.report = grader.grade(
                    st.session_state.quiz,
                    load_answers_from_state(),
                    learning_style=learning_style_key(),
                )
                st.session_state.exam_timeout_processed = True
                goto_step(4)
                st.rerun()

    with st.expander("4) 结果复盘", expanded=st.session_state.flow_step == 4):
        render_section_head("结果复盘", "查看错题与知识点掌握，生成针对性强化训练。")
        log_learning_session_if_needed(user_store)
        render_report(user_store)
        if st.session_state.report and st.session_state.quiz:
            method = learning_method_strategy(
                st.session_state.report,
                st.session_state.quiz,
            )
            dominant_cause = str(method["dominant_cause"])
            cause_cn = {
                "concept_unclear": "概念不清",
                "careless_mistake": "粗心错误",
                "reasoning_error": "推理错误",
                "knowledge_forgotten": "知识遗忘",
                "expression_issue": "表达问题",
                "none": "无",
            }
            st.caption(
                f"当前主要错因：{cause_cn.get(dominant_cause, dominant_cause)} | 当前学习阶段：{method['stage_title']}"
            )
            next_action_label = f"继续学习：{method['stage_title']}"
            if st.button(next_action_label, type="primary", width="stretch"):
                try:
                    dominant, reinforcement_quiz = build_smart_reinforcement_quiz(
                        engine=engine,
                        active_config=active_config,
                        allow_ai_generation=allow_ai_generation,
                    )
                    if not reinforcement_quiz:
                        st.warning("当前没有可继续强化的内容。")
                    else:
                        st.session_state.reinforcement_quiz = reinforcement_quiz
                        st.session_state.source_name = "智能续练"
                        st.session_state.last_generation_mode = QuizMode.practice.value
                        set_current_quiz(
                            reinforcement_quiz,
                            origin_label=f"智能续练·{cause_cn.get(dominant or 'none', '综合')}",
                        )
                        persist_resume_context(
                            user_store,
                            st.session_state.parsed,
                            reinforcement_quiz,
                            f"智能续练·{cause_cn.get(dominant or 'none', '综合')}",
                        )
                        show_model_fallback_notice(engine.provider)
                        reset_result_state()
                        reset_exam_state()
                        goto_step(3)
                        st.success("已为你生成下一轮个性化训练。")
                        st.rerun()
                except Exception as exc:
                    st.error(f"智能续练失败：{exc}")

if __name__ == "__main__":
    main()


