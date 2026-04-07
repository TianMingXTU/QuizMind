from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List

import streamlit as st
import streamlit.components.v1 as components

from quizmind.content import load_text_from_upload, load_text_from_url
from quizmind.exporter import QuizExporter
from quizmind.generation_queue import GenerationQueue
from quizmind.logger import log_event, read_recent_logs
from quizmind.models import Question, QuestionType, Quiz, QuizConfig, QuizMode, UserAnswer
from quizmind.services import (
    ContentService,
    GradingService,
    QuizEngine,
    build_targeted_quiz,
    build_reinforcement_quiz,
)
from quizmind.user_store import UserFeatureStore


st.set_page_config(page_title="QuizMind", page_icon="🧠", layout="wide")


@st.cache_resource
def get_services() -> (
    tuple[
        ContentService,
        QuizEngine,
        GradingService,
        GenerationQueue,
        QuizExporter,
        UserFeatureStore,
    ]
):
    return (
        ContentService(),
        QuizEngine(),
        GradingService(),
        GenerationQueue(),
        QuizExporter(),
        UserFeatureStore(),
    )


@st.cache_data(show_spinner=False)
def read_uploaded_text(file_name: str, data: bytes) -> str:
    return load_text_from_upload(file_name, data)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .qm-question {
            border: 1px solid #d8e3f0;
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
            background: #ffffff;
        }
        .qm-chip {
            display: inline-block;
            margin-right: 6px;
            margin-bottom: 6px;
            padding: 2px 10px;
            border-radius: 999px;
            background: #e2e8f0;
            color: #1e293b;
            font-size: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
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
        "selected_record_id": "",
        "interactive_html": "",
        "interactive_key": "",
        "quality_guard_applied": False,
        "last_session_logged_key": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.flow_step == 5:
        st.session_state.flow_step = 4
    elif st.session_state.flow_step == 4:
        st.session_state.flow_step = 3


def goto_step(step: int) -> None:
    st.session_state.flow_step = max(1, min(4, step))


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
) -> tuple[QuizConfig, str, str, int, QuizMode, int, bool, bool, bool, bool, bool]:
    st.sidebar.markdown("### 练习配置")
    source_mode = st.sidebar.radio("内容来源", ["当前内容", "记忆模式"], index=0)
    mode_cn = st.sidebar.radio("作答模式", ["练习模式", "考试模式"], index=0)
    quiz_mode = QuizMode.exam if mode_cn == "考试模式" else QuizMode.practice
    exam_minutes = (
        st.sidebar.slider("考试时长（分钟）", 5, 120, 30, 5)
        if quiz_mode == QuizMode.exam
        else 0
    )

    st.sidebar.markdown("### 生成策略")
    use_saved_first = st.sidebar.checkbox("优先使用已保存题目", value=False, disabled=True)
    allow_ai_generation = st.sidebar.checkbox("允许 AI 生成新题", value=True, disabled=True)
    enable_interactive_knowledge = st.sidebar.checkbox(
        "启用知识点互动网页（可选）", value=False
    )

    guided_mode = st.sidebar.checkbox("新手引导模式", value=True)
    adaptive_difficulty = st.sidebar.checkbox("启用自适应难度", value=True)

    memory_query = ""
    memory_top_k = 4
    if source_mode == "记忆模式":
        memory_query = st.sidebar.text_input(
            "记忆检索词", placeholder="例如：Python / 操作系统 / 面试"
        )
        memory_top_k = st.sidebar.slider("召回片段数", 2, 8, 4, 1)
        snapshots = engine.list_memory()
        st.sidebar.caption(f"记忆快照数：{len(snapshots)}")
        for item in snapshots[-5:]:
            st.sidebar.markdown(f"- {item.title}（{item.chunks} 段）")

    question_count = st.sidebar.slider("题目数量", 5, 30, 10, 1)
    st.sidebar.caption("难度比例")
    easy = st.sidebar.slider("简单 %", 0, 100, 30, 5)
    medium = st.sidebar.slider("中等 %", 0, 100, 50, 5)
    hard = max(0, 100 - easy - medium)
    st.sidebar.write(f"困难 %: {hard}")

    st.sidebar.caption("题型比例")
    single = st.sidebar.slider("单选 %", 0, 100, 35, 5)
    multiple = st.sidebar.slider("多选 %", 0, 100, 15, 5)
    fill = st.sidebar.slider("填空 %", 0, 100, 20, 5)
    short = st.sidebar.slider("简答 %", 0, 100, 20, 5)
    true_false = max(0, 100 - single - multiple - fill - short)
    st.sidebar.write(f"判断 %: {true_false}")

    config = QuizConfig(
        question_count=question_count,
        difficulty_mix={"easy": easy, "medium": medium, "hard": hard},
        type_mix={
            "single_choice": single,
            "multiple_choice": multiple,
            "fill_blank": fill,
            "short_answer": short,
            "true_false": true_false,
        },
    )
    return (
        config,
        source_mode,
        memory_query,
        memory_top_k,
        quiz_mode,
        exam_minutes,
        use_saved_first,
        allow_ai_generation,
        enable_interactive_knowledge,
        guided_mode,
        adaptive_difficulty,
    )


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
        st.session_state.quiz, load_answers_from_state()
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


def summarize_logs() -> dict[str, int]:
    lines = read_recent_logs(limit=100)
    summary = {"cache_hit": 0, "cache_miss": 0, "llm_ok": 0, "batch_grade": 0}
    for line in lines:
        summary["cache_hit"] += int('"event": "llm.cache_hit"' in line)
        summary["cache_miss"] += int('"event": "llm.cache_miss"' in line)
        summary["llm_ok"] += int('"event": "llm.invoke.success"' in line)
        summary["batch_grade"] += int(
            '"event": "service.grade.subjective_batch"' in line
        )
    return summary


def render_runtime_panel() -> None:
    summary = summarize_logs()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("缓存命中", summary["cache_hit"])
    c2.metric("缓存未命中", summary["cache_miss"])
    c3.metric("LLM 调用成功", summary["llm_ok"])
    c4.metric("简答批量评分", summary["batch_grade"])


def render_beginner_guide(guided_mode: bool) -> None:
    if not guided_mode:
        return
    source_ready = bool((st.session_state.get("source_text") or "").strip())
    parsed_ready = st.session_state.get("parsed") is not None
    quiz_ready = st.session_state.get("quiz") is not None
    report_ready = st.session_state.get("report") is not None

    with st.expander("新手引导", expanded=not report_ready):
        st.write("推荐流程：输入内容 -> 解析/出题 -> 作答提交 -> 结果复盘。")
        st.markdown(
            f"- 输入内容：{'已完成' if source_ready else '未完成'}\n"
            f"- 解析完成：{'已完成' if parsed_ready else '未完成'}\n"
            f"- 题目就绪：{'已完成' if quiz_ready else '未完成'}\n"
            f"- 已提交结果：{'已完成' if report_ready else '未完成'}"
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("去第1步", width="stretch", key="guide_to_step1"):
                goto_step(1)
                st.rerun()
        with c2:
            if st.button("去第2步", width="stretch", key="guide_to_step2"):
                goto_step(2)
                st.rerun()
        with c3:
            if st.button("去第3步", width="stretch", key="guide_to_step3"):
                goto_step(3)
                st.rerun()
        with c4:
            if st.button("去第4步", width="stretch", key="guide_to_step4"):
                goto_step(4)
                st.rerun()


def apply_quality_guard(
    quiz: Quiz | None,
    parsed,
    config: QuizConfig,
    engine: QuizEngine,
    allow_ai_generation: bool,
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
    )
    existing = {q.id for q in filtered}
    for item in fallback.questions:
        if len(filtered) >= original_total:
            break
        if item.id in existing:
            continue
        if set(item.knowledge_tags or []) & blocked_tags:
            continue
        filtered.append(item)
        existing.add(item.id)

    if len(filtered) < original_total:
        filtered.extend(quiz.questions[len(filtered):original_total])

    st.session_state.quality_guard_applied = True
    st.info(f"已根据历史反馈替换 {dropped} 道低质量倾向题目。")
    return quiz.model_copy(update={"questions": filtered[:original_total]})


def build_active_config(
    config: QuizConfig,
    adaptive_difficulty: bool,
    user_store: UserFeatureStore,
) -> QuizConfig:
    if not adaptive_difficulty:
        return config
    mixed = user_store.suggest_difficulty_mix(config.difficulty_mix)
    if mixed == config.difficulty_mix:
        return config
    st.caption(
        f"自适应难度已生效：easy={mixed['easy']}%, medium={mixed['medium']}%, hard={mixed['hard']}%"
    )
    return config.model_copy(update={"difficulty_mix": mixed})


def log_learning_session_if_needed(user_store: UserFeatureStore) -> None:
    report = st.session_state.get("report")
    quiz = st.session_state.get("quiz")
    if not report or not quiz:
        return
    session_key = f"{quiz.title}|{report.overall_score}|{len(report.wrong_questions)}"
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
    )
    st.session_state.last_session_logged_key = session_key


def render_learning_dashboard(user_store: UserFeatureStore) -> None:
    board = user_store.weekly_dashboard(days=7)
    st.markdown("**学习进度（7天）**")
    c1, c2 = st.columns(2)
    c1.metric("近7天练习次数", board["total_attempts"])
    c2.metric("近期平均分", f"{board['avg_score']:.1f}")
    st.line_chart(
        {"day": board["labels"], "avg_score": board["avg_scores"]},
        x="day",
        y="avg_score",
    )
    st.bar_chart(
        {"day": board["labels"], "attempts": board["attempts"]},
        x="day",
        y="attempts",
    )


def render_favorites_panel(user_store: UserFeatureStore) -> None:
    with st.expander("我的收藏题单", expanded=False):
        favorites = user_store.list_favorites()
        st.caption(f"已收藏 {len(favorites)} 道题。")
        if not favorites:
            st.info("先在作答页收藏题目，之后可在这里集中复练。")
            return
        preview = [
            {
                "added_at": item.get("added_at", ""),
                "source": item.get("source_title", ""),
                "prompt": str((item.get("question") or {}).get("prompt", ""))[:90],
                "tags": ", ".join((item.get("question") or {}).get("knowledge_tags", [])[:3]),
            }
            for item in favorites[:50]
        ]
        st.dataframe(preview, width="stretch", hide_index=True)
        if st.button("加载收藏题到当前会话", width="stretch"):
            quiz = user_store.build_favorites_quiz(question_limit=15)
            if quiz:
                st.session_state.quiz = quiz
                st.session_state.report = None
                st.session_state.reinforcement_quiz = None
                st.session_state.last_generation_mode = QuizMode.practice.value
                goto_step(3)
                st.rerun()


def qtype_label(qtype: QuestionType | str) -> str:
    normalized = QuestionType.normalize(qtype)
    mapping = {
        QuestionType.single_choice.value: "Single Choice",
        QuestionType.multiple_choice.value: "Multiple Choice",
        QuestionType.fill_blank.value: "Fill Blank",
        QuestionType.short_answer.value: "Short Answer",
        QuestionType.true_false.value: "True False",
    }
    return mapping.get(normalized, normalized)


def diff_label(value: str) -> str:
    return {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(value, value)


def answer_state_key(question: Question, index: int) -> str:
    safe_id = str(getattr(question, "id", f"q{index}")).strip() or f"q{index}"
    return f"answer_{safe_id}"


def render_question_widget(
    question: Question,
    index: int,
    user_store: UserFeatureStore,
    source_title: str,
) -> None:
    key = answer_state_key(question, index)
    qtype = QuestionType.normalize(getattr(question, "question_type", ""))
    options = [str(item) for item in (question.options or []) if str(item).strip()]
    if qtype == QuestionType.true_false.value:
        if not ({"正确", "错误"}.issubset(set(options))):
            options = ["正确", "错误"]

    prompt = (
        question.prompt or ""
    ).strip() or "(Missing prompt: please regenerate this question)"
    difficulty_value = getattr(question.difficulty, "value", str(question.difficulty))
    tags = question.knowledge_tags[:2] if question.knowledge_tags else ["untagged"]

    st.markdown(
        f'<div class="qm-question"><div style="font-weight:700;">{question.id} 路 {qtype_label(question.question_type)}</div>'
        f'<div><span class="qm-chip">{diff_label(str(difficulty_value))}</span>'
        f'<span class="qm-chip">{"/".join(tags)}</span></div>'
        f'<div style="margin-top:8px;">{prompt}</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        if user_store.has_favorite(question):
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
            st.radio(
                "Select one", options, key=key, index=None, label_visibility="collapsed"
            )
        else:
            st.text_input("Answer", key=key, label_visibility="collapsed")
            st.caption("Question options missing, switched to text input.")
    elif qtype == QuestionType.multiple_choice.value:
        if options:
            st.multiselect(
                "Select multiple", options, key=key, label_visibility="collapsed"
            )
        else:
            st.text_input("Answer", key=key, label_visibility="collapsed")
            st.caption("Question options missing, switched to text input.")
    elif qtype == QuestionType.true_false.value:
        st.radio(
            "True or False", options, key=key, index=None, label_visibility="collapsed"
        )
    else:
        st.text_area("Answer", key=key, height=120, label_visibility="collapsed")


def render_all_questions(quiz, user_store: UserFeatureStore) -> None:
    total = len(quiz.questions)
    st.progress(
        answered_count() / max(1, total), text=f"已作答 {answered_count()} / {total}"
    )

    type_counter: dict[str, int] = {}
    for question in quiz.questions:
        t = QuestionType.normalize(getattr(question, "question_type", ""))
        type_counter[t] = type_counter.get(t, 0) + 1
    st.caption(
        "题型分布: "
        + ", ".join(f"{qtype_label(k)}={v}" for k, v in sorted(type_counter.items()))
    )

    for idx, question in enumerate(quiz.questions, start=1):
        st.markdown(f"**第 {idx} 题**")
        try:
            render_question_widget(
                question,
                idx,
                user_store=user_store,
                source_title=st.session_state.get("source_name", "当前输入"),
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


def render_interactive_knowledge_panel(
    content_service: ContentService,
    allow_ai_generation: bool,
    enabled: bool,
) -> None:
    if not enabled:
        return
    parsed = st.session_state.parsed
    if not parsed:
        return

    current_key = (
        f"{parsed.title}|{len(parsed.knowledge_points)}|{','.join(parsed.concepts[:8])}"
    )
    if st.session_state.get("interactive_key") != current_key:
        st.session_state.interactive_html = ""
        st.session_state.interactive_key = current_key

    with st.expander("互动知识网页（可选）", expanded=False):
        st.caption("点击生成后，会在当前页面嵌入一个可交互的小网页，帮助理解知识点。")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "生成互动网页", width="stretch", key="btn_generate_interactive"
            ):
                with st.spinner("正在生成互动网页..."):
                    st.session_state.interactive_html = (
                        content_service.generate_interactive_html(
                            parsed,
                            allow_ai_generation=allow_ai_generation,
                        )
                    )
        with c2:
            if st.button("清空互动网页", width="stretch", key="btn_clear_interactive"):
                st.session_state.interactive_html = ""

        if st.session_state.interactive_html:
            components.html(
                st.session_state.interactive_html, height=620, scrolling=True
            )
        else:
            st.info("当前未生成互动网页。")


def render_report() -> None:
    report = st.session_state.report
    quiz = st.session_state.quiz
    if not report or not quiz:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("综合得分", f"{report.overall_score:.1f}")
    c2.metric("客观题正确率", f"{report.objective_accuracy:.1f}%")
    c3.metric("主观题均分", f"{report.subjective_average:.1f}")

    st.markdown("**知识点掌握情况**")
    for item in report.knowledge_stats:
        st.markdown(
            f"- **{item.knowledge_point}** | 状态：{item.status} | 平均分：{item.avg_score:.1f} | 通过率：{item.accuracy:.1f}%"
        )

    st.markdown("**错题本**")
    qmap = {q.id: q for q in quiz.questions}
    if not report.wrong_questions:
        st.success("本轮没有错题，掌握得不错。")
    for wrong in report.wrong_questions:
        q = qmap.get(wrong.question_id)
        if not q:
            continue
        st.markdown(
            f"- **{wrong.question_id} {q.prompt}** \n"
            f"你的答案：{', '.join(wrong.user_answer) or '未作答'}  \n"
            f"正确答案：{', '.join(wrong.correct_answer)}  \n"
            f"点评：{wrong.feedback}"
        )

    st.markdown("**复习建议**")
    for line in report.review_recommendations:
        st.markdown(f"- {line}")


def render_quiz_bank_panel(engine: QuizEngine, exporter: QuizExporter) -> None:
    with st.expander("题库管理：搜索 / 删除 / 导出", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            file_kw = st.text_input("文件名关键词", key="bank_file_kw")
        with col2:
            tag_kw = st.text_input("标签关键词", key="bank_tag_kw")
        with col3:
            date_from = st.date_input(
                "开始日期", value=date.today().replace(day=1), key="bank_date_from"
            )
        with col4:
            date_to = st.date_input("结束日期", value=date.today(), key="bank_date_to")

        records = engine.search_saved_quizzes(
            file_name_keyword=file_kw,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            tag_keyword=tag_kw,
            limit=200,
        )
        if not records:
            st.caption("没有匹配记录")
            return

        options = {
            f"{item['created_at']} | {item['source_name']} | {item['title']} | 标签:{','.join(item.get('tags', [])[:3])}": item["id"]
            for item in records
        }
        chosen = st.selectbox("选择记录", list(options.keys()))
        st.session_state.selected_record_id = options[chosen]

        action1, action2, action3, action4 = st.columns(4)
        with action1:
            if st.button("加载到当前会话", width="stretch"):
                loaded = engine.load_saved_quiz(st.session_state.selected_record_id)
                if loaded:
                    parsed, quiz = loaded
                    st.session_state.parsed = parsed
                    st.session_state.quiz = quiz
                    st.session_state.report = None
                    st.session_state.reinforcement_quiz = None
                    st.session_state.last_generation_mode = QuizMode.practice.value
                    goto_step(3)
                    st.rerun()
        with action2:
            if st.button("删除记录", width="stretch"):
                ok = engine.delete_saved_quiz(st.session_state.selected_record_id)
                if ok:
                    st.success("删除成功")
                    st.rerun()
                st.error("删除失败")
        with action3:
            if st.button("导出 JSON", width="stretch"):
                loaded = engine.load_saved_quiz(st.session_state.selected_record_id)
                if loaded:
                    parsed, quiz = loaded
                    target = engine.quiz_bank.export_path(
                        st.session_state.selected_record_id, "json"
                    )
                    exporter.export_json(target, parsed, quiz)
                    st.download_button(
                        "下载 JSON",
                        target.read_bytes(),
                        file_name=target.name,
                        mime="application/json",
                        width="stretch",
                    )
        with action4:
            export_fmt = st.selectbox("导出格式", ["md", "pdf"], key="export_fmt")
            if st.button("导出并下载", width="stretch"):
                loaded = engine.load_saved_quiz(st.session_state.selected_record_id)
                if loaded:
                    parsed, quiz = loaded
                    target = engine.quiz_bank.export_path(
                        st.session_state.selected_record_id, export_fmt
                    )
                    if export_fmt == "md":
                        exporter.export_markdown(target, parsed, quiz)
                        mime = "text/markdown"
                    else:
                        exporter.export_pdf(target, parsed, quiz)
                        mime = "application/pdf"
                    st.download_button(
                        f"下载 {export_fmt.upper()}",
                        target.read_bytes(),
                        file_name=target.name,
                        mime=mime,
                        width="stretch",
                    )

        st.dataframe(
            [
                {
                    "创建时间": item["created_at"],
                    "来源文件": item["source_name"],
                    "标题": item["title"],
                    "题目数": item["question_count"],
                    "标签": ", ".join(item.get("tags", [])),
                }
                for item in records[:100]
            ],
            width="stretch",
            hide_index=True,
        )


def render_queue_panel(
    queue: GenerationQueue,
    engine: QuizEngine,
    config: QuizConfig,
    use_saved_first: bool,
    allow_ai_generation: bool,
) -> None:
    with st.expander("批量生成队列", expanded=False):
        files = st.file_uploader(
            "批量上传文件并加入队列",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            key="batch_files",
        )
        w1, w2 = st.columns(2)
        with w1:
            max_workers = st.slider("并发处理数", 1, 8, 2, 1)
        with w2:
            st.caption("并发越高处理越快，但 API 压力也更高。")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("加入队列", width="stretch"):
                if not files:
                    st.warning("请先选择文件")
                else:
                    items = []
                    for file in files:
                        try:
                            content = read_uploaded_text(file.name, file.getvalue())
                            items.append(
                                {
                                    "name": file.name,
                                    "source_type": "file",
                                    "content": content,
                                }
                            )
                        except Exception as exc:
                            st.error(f"{file.name} 读取失败：{exc}")
                    if items:
                        count = queue.enqueue_items(items)
                        st.success(f"已加入 {count} 个文件")
        with b2:
            if st.button("处理队列", width="stretch"):
                stats = queue.process_pending(
                    engine=engine,
                    config=config,
                    use_saved_first=use_saved_first,
                    allow_ai_generation=allow_ai_generation,
                    max_workers=max_workers,
                )
                st.success(
                    f"处理完成：共 {stats['processed']}，成功 {stats['success']}，失败 {stats['failed']}"
                )
        with b3:
            if st.button("失败重试", width="stretch"):
                count = queue.retry_failed()
                st.success(f"已重置 {count} 条失败任务")
        with b4:
            if st.button("清理已完成", width="stretch"):
                removed = queue.clear_done()
                st.success(f"已清理 {removed} 条完成任务")

        items = queue.list_items()
        if not items:
            st.caption("队列为空")
            return

        st.dataframe(
            [
                {
                    "文件": item["name"],
                    "状态": item["status"],
                    "尝试次数": item.get("attempts", 0),
                    "记录ID": item["record_id"],
                    "错误": item["error"],
                    "更新时间": item["updated_at"],
                }
                for item in items[:100]
            ],
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    init_state()
    inject_styles()
    content_service, engine, grader, queue, exporter, user_store = get_services()

    st.title("QuizMind 智能练习与考试系统")
    st.caption("单页流程：输入解析 -> 自动出题 -> 作答评测 -> 强化训练")

    (
        config,
        source_mode,
        memory_query,
        memory_top_k,
        quiz_mode,
        exam_minutes,
        use_saved_first,
        allow_ai_generation,
        enable_interactive_knowledge,
        guided_mode,
        adaptive_difficulty,
    ) = render_sidebar(engine)
    render_beginner_guide(guided_mode)
    active_config = build_active_config(config, adaptive_difficulty, user_store)
    render_runtime_panel()
    st.divider()

    with st.expander("1) 输入与解析", expanded=st.session_state.flow_step == 1):
        if source_mode == "当前内容":
            source_text, source_type = collect_source()
            st.session_state.source_text = source_text
            st.session_state.source_type = source_type
        else:
            source_text, source_type = "", "memory"
            st.info("记忆模式将从向量库召回内容后自动组卷。")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("解析内容", width="stretch"):
                if source_mode == "记忆模式":
                    st.warning("记忆模式无需单独解析，请直接生成题目。")
                elif not source_text.strip():
                    st.warning("请先输入学习内容。")
                else:
                    st.session_state.parsed = content_service.parse(source_text, source_type)
                    st.session_state.quiz = None
                    st.session_state.report = None
                    st.session_state.reinforcement_quiz = None
                    st.session_state.interactive_html = ""
                    st.session_state.interactive_key = ""
                    reset_exam_state()
                    goto_step(2)
                    st.rerun()
        with c2:
            if st.button("保存到记忆库", width="stretch"):
                if not st.session_state.parsed:
                    st.warning("请先解析内容。")
                else:
                    snapshot = engine.save_memory(st.session_state.parsed)
                    st.session_state.memory_snapshot = snapshot
                    st.success(f"已保存：{snapshot.title}（{snapshot.chunks} 段）")
        with c3:
            if st.button("一键解析并出题", type="primary", width="stretch"):
                if source_mode == "记忆模式":
                    st.warning("记忆模式请在第 2 步直接生成题目。")
                elif not source_text.strip():
                    st.warning("请先输入学习内容。")
                else:
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
                        )
                        st.session_state.parsed = parsed
                        st.session_state.quiz = apply_quality_guard(
                            quiz,
                            parsed,
                            active_config,
                            engine,
                            allow_ai_generation,
                            user_store,
                        )
                        st.session_state.report = None
                        st.session_state.reinforcement_quiz = None
                        st.session_state.interactive_html = ""
                        st.session_state.interactive_key = ""
                        try:
                            snapshot = engine.save_memory(parsed)
                            st.session_state.memory_snapshot = snapshot
                        except Exception as exc:
                            log_event("flow.one_click.save_memory_failed", error=str(exc))
                        start_exam_if_needed(quiz_mode, exam_minutes)
                        goto_step(3)
                        if meta.get("from_saved"):
                            st.info("已复用历史题目并进入作答区。")
                        else:
                            st.success("已生成新题并进入作答区。")
                    st.rerun()

        render_parsed_summary()
        render_interactive_knowledge_panel(
            content_service=content_service,
            allow_ai_generation=allow_ai_generation,
            enabled=enable_interactive_knowledge,
        )

    with st.expander("2) 组卷与题库/队列", expanded=st.session_state.flow_step == 2):
        st.caption("优先复用题库可减少 API 调用；关闭 AI 时使用本地规则生成。")
        if st.button("生成题目", width="stretch"):
            try:
                if source_mode == "当前内容":
                    if st.session_state.parsed is not None:
                        generated = engine.generate_quiz(
                            st.session_state.parsed,
                            active_config,
                            allow_ai_generation=allow_ai_generation,
                        )
                        st.session_state.quiz = apply_quality_guard(
                            generated,
                            st.session_state.parsed,
                            active_config,
                            engine,
                            allow_ai_generation,
                            user_store,
                        )
                        st.success("已基于解析结果生成题目。")
                    elif source_text.strip():
                        parsed, quiz, meta = engine.generate_or_load_from_source(
                            source=source_text,
                            source_type=source_type,
                            source_name=st.session_state.get("source_name", "当前输入"),
                            config=active_config,
                            use_saved_first=use_saved_first,
                            allow_ai_generation=allow_ai_generation,
                        )
                        st.session_state.parsed = parsed
                        st.session_state.quiz = apply_quality_guard(
                            quiz,
                            parsed,
                            active_config,
                            engine,
                            allow_ai_generation,
                            user_store,
                        )
                        if meta.get("from_saved"):
                            st.info("本次使用了已保存题目。")
                        else:
                            st.success("已生成并保存新题目。")
                    else:
                        st.warning("请先输入学习内容。")
                        return
                else:
                    parsed, quiz = engine.generate_from_memory(
                        config=active_config,
                        query=memory_query,
                        top_k=memory_top_k,
                        allow_ai_generation=allow_ai_generation,
                    )
                    st.session_state.parsed = parsed
                    st.session_state.quiz = apply_quality_guard(
                        quiz,
                        parsed,
                        active_config,
                        engine,
                        allow_ai_generation,
                        user_store,
                    )
                st.session_state.report = None
                st.session_state.reinforcement_quiz = None
                start_exam_if_needed(quiz_mode, exam_minutes)
                goto_step(3)
                st.rerun()
            except Exception as exc:
                st.error(f"生成题目失败：{exc}")

        render_quiz_bank_panel(engine, exporter)
        render_favorites_panel(user_store)
        render_queue_panel(queue, engine, active_config, use_saved_first, allow_ai_generation)

    with st.expander("3) 作答", expanded=st.session_state.flow_step == 3):
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
            render_exam_timer()
            if not getattr(st.session_state.quiz, "questions", None):
                st.warning("当前题目为空，可能是生成/加载失败。")
                if st.session_state.parsed and st.button("重新生成题目", type="primary", width="stretch"):
                    generated = engine.generate_quiz(
                        st.session_state.parsed,
                        active_config,
                        allow_ai_generation=allow_ai_generation,
                    )
                    st.session_state.quiz = apply_quality_guard(
                        generated,
                        st.session_state.parsed,
                        active_config,
                        engine,
                        allow_ai_generation,
                        user_store,
                    )
                    st.session_state.report = None
                    st.session_state.reinforcement_quiz = None
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
            if not st.session_state.report and st.button(submit_label, type="primary", width="stretch"):
                st.session_state.report = grader.grade(
                    st.session_state.quiz, load_answers_from_state()
                )
                st.session_state.exam_timeout_processed = True
                goto_step(4)
                st.rerun()

    with st.expander("4) 结果复盘", expanded=st.session_state.flow_step == 4):
        log_learning_session_if_needed(user_store)
        render_report()
        if st.session_state.report and st.button("错题重练（变体）", width="stretch"):
            try:
                quiz = st.session_state.quiz
                report = st.session_state.report
                parsed = st.session_state.parsed
                qmap = {q.id: q for q in (quiz.questions if quiz else [])}
                focus_topics: list[str] = []
                for wrong in report.wrong_questions:
                    q = qmap.get(wrong.question_id)
                    if q:
                        focus_topics.extend(q.knowledge_tags or [])
                st.session_state.reinforcement_quiz = build_targeted_quiz(
                    parsed,
                    focus_topics,
                    active_config,
                    question_count=min(max(6, len(focus_topics) * 2), 20),
                    allow_ai_generation=allow_ai_generation,
                )
                st.success("已生成错题定向变体练习。")
            except Exception as exc:
                st.error(f"错题重练生成失败：{exc}")
        if st.session_state.report and st.button("生成针对性强化题", width="stretch"):
            try:
                st.session_state.reinforcement_quiz = build_reinforcement_quiz(
                    st.session_state.parsed,
                    st.session_state.report,
                    active_config,
                )
            except Exception as exc:
                st.error(f"强化题生成失败：{exc}")

        if st.session_state.reinforcement_quiz:
            st.markdown("#### 强化题预览")
            for question in st.session_state.reinforcement_quiz.questions:
                st.markdown(
                    f"- **{question.id} {qtype_label(question.question_type)}** | 知识点：{', '.join(question.knowledge_tags)}  \n"
                    f"{question.prompt}"
                )
        if st.session_state.reinforcement_quiz and st.button("加载强化题到作答区", width="stretch"):
            st.session_state.quiz = st.session_state.reinforcement_quiz
            st.session_state.report = None
            st.session_state.last_generation_mode = QuizMode.practice.value
            goto_step(3)
            st.rerun()

        render_learning_dashboard(user_store)

        with st.expander("最近运行日志", expanded=False):
            lines = read_recent_logs(limit=25)
            if lines:
                st.code("\n".join(lines), language="text")
            else:
                st.caption("暂无日志")


if __name__ == "__main__":
    main()
