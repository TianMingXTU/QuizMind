from __future__ import annotations
from datetime import datetime, timedelta
from typing import List

import streamlit as st
import streamlit.components.v1 as components

from quizmind.content import load_text_from_upload, load_text_from_url
from quizmind.logger import log_event, read_recent_logs
from quizmind.models import QuestionType, QuizConfig, QuizMode, UserAnswer
from quizmind.services import ContentService, GradingService, QuizEngine, build_reinforcement_quiz


st.set_page_config(page_title="QuizMind", page_icon="🧠", layout="wide")


@st.cache_resource
def get_services() -> tuple[ContentService, QuizEngine, GradingService]:
    return ContentService(), QuizEngine(), GradingService()


def init_state() -> None:
    defaults = {
        "parsed": None,
        "quiz": None,
        "report": None,
        "reinforcement_quiz": None,
        "memory_snapshot": None,
        "source_text": "",
        "source_type": "text",
        "exam_deadline": None,
        "exam_timeout_processed": False,
        "last_generation_mode": QuizMode.practice.value,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def collect_source() -> tuple[str, str]:
    input_mode = st.radio("输入方式", ["粘贴文本", "上传文件", "网页 URL"], horizontal=True)
    if input_mode == "粘贴文本":
        text = st.text_area("学习内容", height=280, placeholder="把要学习的知识内容贴到这里...")
        return text, "text"
    if input_mode == "上传文件":
        file = st.file_uploader("上传 PDF / Word / Markdown / TXT", type=["pdf", "docx", "doc", "md", "txt"])
        if file is None:
            return "", "file"
        return load_text_from_upload(file.name, file.getvalue()), "file"
    url = st.text_input("网页地址", placeholder="https://example.com/article")
    return (load_text_from_url(url.strip()), "url") if url.strip() else ("", "url")


def render_sidebar(engine: QuizEngine) -> tuple[QuizConfig, str, str, int, QuizMode, int]:
    st.sidebar.title("出题设置")
    source_mode = st.sidebar.radio("练习来源", ["当前内容", "记忆模式"], index=0)
    quiz_mode = QuizMode(
        st.sidebar.radio("答题模式", [QuizMode.practice.value, QuizMode.exam.value], index=0)
    )
    exam_minutes = st.sidebar.slider("考试时长（分钟）", 5, 120, 30, 5) if quiz_mode == QuizMode.exam else 0

    memory_query = ""
    memory_top_k = 4
    if source_mode == "记忆模式":
        memory_query = st.sidebar.text_input("记忆检索词", placeholder="例如：Python / 计算机网络 / 面试")
        memory_top_k = st.sidebar.slider("记忆召回片段数", 2, 8, 4, 1)
        snapshots = engine.list_memory()
        st.sidebar.caption(f"记忆库条目：{len(snapshots)}")
        for item in snapshots[-5:]:
            st.sidebar.markdown(f"- {item.title} ({item.chunks} 段)")

    question_count = st.sidebar.slider("题目数量", 5, 30, 10, 1)

    st.sidebar.caption("难度比例")
    easy = st.sidebar.slider("简单 %", 0, 100, 30, 5)
    medium = st.sidebar.slider("中等 %", 0, 100, 50, 5)
    hard = max(0, 100 - easy - medium)
    st.sidebar.write(f"困难 %: {hard}")

    st.sidebar.caption("题型比例")
    single = st.sidebar.slider("单选题 %", 0, 100, 35, 5)
    multiple = st.sidebar.slider("多选题 %", 0, 100, 15, 5)
    fill = st.sidebar.slider("填空题 %", 0, 100, 20, 5)
    short = st.sidebar.slider("简答题 %", 0, 100, 20, 5)
    true_false = max(0, 100 - single - multiple - fill - short)
    st.sidebar.write(f"判断题 %: {true_false}")

    return (
        QuizConfig(
            question_count=question_count,
            difficulty_mix={"简单": easy, "中等": medium, "困难": hard},
            type_mix={
                "单选题": single,
                "多选题": multiple,
                "填空题": fill,
                "简答题": short,
                "判断题": true_false,
            },
        ),
        source_mode,
        memory_query,
        memory_top_k,
        quiz_mode,
        exam_minutes,
    )


def load_answers_from_state() -> List[UserAnswer]:
    quiz = st.session_state.quiz
    if not quiz:
        return []
    answers: List[UserAnswer] = []
    for question in quiz.questions:
        key = f"answer_{question.id}"
        value = st.session_state.get(key)
        if question.question_type in {QuestionType.single_choice, QuestionType.true_false}:
            answer = [value] if value else []
        elif question.question_type == QuestionType.multiple_choice:
            answer = value if isinstance(value, list) else []
        else:
            answer = [value] if isinstance(value, str) and value.strip() else []
        answers.append(UserAnswer(question_id=question.id, answer=answer))
    return answers


def reset_exam_state() -> None:
    st.session_state.exam_deadline = None
    st.session_state.exam_timeout_processed = False


def start_exam_if_needed(quiz_mode: QuizMode, exam_minutes: int) -> None:
    st.session_state.last_generation_mode = quiz_mode.value
    if quiz_mode == QuizMode.exam:
        deadline = datetime.now() + timedelta(minutes=exam_minutes)
        st.session_state.exam_deadline = deadline.isoformat()
        st.session_state.exam_timeout_processed = False
        log_event("exam.start", deadline=st.session_state.exam_deadline, duration_minutes=exam_minutes)
    else:
        reset_exam_state()


def render_exam_banner() -> None:
    if st.session_state.last_generation_mode != QuizMode.exam.value or not st.session_state.exam_deadline:
        return
    if st.session_state.report:
        return

    deadline = datetime.fromisoformat(st.session_state.exam_deadline)
    remaining = deadline - datetime.now()
    remaining_seconds = max(0, int(remaining.total_seconds()))
    mins, secs = divmod(remaining_seconds, 60)
    st.warning(f"考试剩余时间：{mins:02d}:{secs:02d}")
    components.html(
        f"""
        <div id="quizmind-timer" style="font-size:14px;color:#b45309;">考试倒计时加载中...</div>
        <script>
        const deadline = new Date("{deadline.isoformat()}").getTime();
        const timer = document.getElementById("quizmind-timer");
        function tick() {{
          const diff = deadline - Date.now();
          if (diff <= 0) {{
            timer.innerText = "考试时间已到，正在自动交卷...";
            window.parent.location.reload();
            return;
          }}
          const seconds = Math.floor(diff / 1000);
          const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
          const ss = String(seconds % 60).padStart(2, "0");
          timer.innerText = `考试倒计时：${{mm}}:${{ss}}`;
        }}
        tick();
        setInterval(tick, 1000);
        </script>
        """,
        height=45,
    )


def process_exam_timeout(grader: GradingService) -> None:
    if st.session_state.last_generation_mode != QuizMode.exam.value:
        return
    if not st.session_state.quiz or st.session_state.report or not st.session_state.exam_deadline:
        return
    if st.session_state.exam_timeout_processed:
        return

    deadline = datetime.fromisoformat(st.session_state.exam_deadline)
    if datetime.now() < deadline:
        return

    answers = load_answers_from_state()
    st.session_state.report = grader.grade(st.session_state.quiz, answers)
    st.session_state.exam_timeout_processed = True
    log_event("exam.auto_submit", quiz_title=st.session_state.quiz.title)


def summarize_logs() -> dict[str, int]:
    lines = read_recent_logs(limit=80)
    summary = {"cache_hit": 0, "cache_miss": 0, "llm_success": 0, "batch_grade": 0}
    for line in lines:
        summary["cache_hit"] += int('"event": "llm.cache_hit"' in line)
        summary["cache_miss"] += int('"event": "llm.cache_miss"' in line)
        summary["llm_success"] += int('"event": "llm.invoke.success"' in line)
        summary["batch_grade"] += int('"event": "service.grade.subjective_batch"' in line)
    return summary


def render_runtime_panel() -> None:
    summary = summarize_logs()
    cols = st.columns(4)
    cols[0].metric("缓存命中", summary["cache_hit"])
    cols[1].metric("缓存未命中", summary["cache_miss"])
    cols[2].metric("LLM 调用完成", summary["llm_success"])
    cols[3].metric("主观题批量评分", summary["batch_grade"])

    with st.expander("查看最近运行日志"):
        lines = read_recent_logs(limit=25)
        if not lines:
            st.caption("暂无日志。")
        else:
            st.code("\n".join(lines), language="text")


def render_parsed_summary() -> None:
    parsed = st.session_state.parsed
    if not parsed:
        return
    st.subheader("内容解析")
    cols = st.columns(3)
    cols[0].metric("知识点", len(parsed.knowledge_points))
    cols[1].metric("分段数", len(parsed.segments))
    cols[2].metric("核心概念", len(parsed.concepts))
    for point in parsed.knowledge_points:
        st.markdown(
            f"- **{point.name}** | 难度：{point.difficulty.value} | 关键词：{'、'.join(point.keywords) or '无'}  \n"
            f"{point.summary}"
        )


def render_quiz() -> None:
    quiz = st.session_state.quiz
    if not quiz:
        return

    st.subheader("开始答题")
    for question in quiz.questions:
        with st.container(border=True):
            st.markdown(f"**{question.id} · {question.question_type.value} · {question.difficulty.value}**")
            st.write(question.prompt)
            key = f"answer_{question.id}"
            if question.question_type == QuestionType.single_choice:
                st.radio("请选择一个答案", question.options, key=key, index=None)
            elif question.question_type == QuestionType.multiple_choice:
                st.multiselect("可多选", question.options, key=key)
            elif question.question_type == QuestionType.true_false:
                st.radio("请选择", question.options, key=key, index=None)
            else:
                st.text_area("请输入答案", key=key, height=120)


def render_report() -> None:
    report = st.session_state.report
    quiz = st.session_state.quiz
    if not report or not quiz:
        return

    st.subheader("学习反馈")
    cols = st.columns(3)
    cols[0].metric("综合得分", f"{report.overall_score:.1f}")
    cols[1].metric("客观题正确率", f"{report.objective_accuracy:.1f}%")
    cols[2].metric("主观题均分", f"{report.subjective_average:.1f}")

    st.write("**知识点掌握情况**")
    for item in report.knowledge_stats:
        st.markdown(
            f"- **{item.knowledge_point}** | 状态：{item.status} | 平均分：{item.avg_score:.1f} | 通过率：{item.accuracy:.1f}%"
        )

    st.write("**错题本**")
    question_map = {question.id: question for question in quiz.questions}
    if not report.wrong_questions:
        st.success("本轮没有错题，整体掌握不错。")
    for wrong in report.wrong_questions:
        question = question_map[wrong.question_id]
        st.markdown(
            f"- **{wrong.question_id} {question.prompt}**  \n"
            f"你的答案：{'、'.join(wrong.user_answer) or '未作答'}  \n"
            f"正确答案：{'、'.join(wrong.correct_answer)}  \n"
            f"点评：{wrong.feedback}  \n"
            f"缺失点：{'、'.join(wrong.missing_points) or '无'}"
        )

    st.write("**复习建议**")
    for line in report.review_recommendations:
        st.markdown(f"- {line}")


def main() -> None:
    init_state()
    content_service, engine, grader = get_services()

    st.title("QuizMind 智能刷题系统")
    st.caption("日志更清晰、调用更省、支持考试时限的 LangChain 智能练习系统")

    config, source_mode, memory_query, memory_top_k, quiz_mode, exam_minutes = render_sidebar(engine)
    render_runtime_panel()

    if source_mode == "当前内容":
        source_text, source_type = collect_source()
        st.session_state.source_text = source_text
        st.session_state.source_type = source_type
    else:
        source_text, source_type = "", "memory"
        st.info("当前处于记忆模式：系统会从向量库中随机或按检索词召回历史学习内容，再自动生成题目。")

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("1. 解析内容", use_container_width=True):
            if not source_text.strip():
                st.warning("请先输入学习内容或上传文件。")
            else:
                try:
                    st.session_state.parsed = content_service.parse(source_text, source_type)
                    st.session_state.quiz = None
                    st.session_state.report = None
                    st.session_state.reinforcement_quiz = None
                    reset_exam_state()
                except Exception as exc:
                    st.error(f"内容解析失败：{exc}")

    with action_cols[1]:
        if st.button("2. 保存到记忆库", use_container_width=True):
            if not st.session_state.parsed:
                st.warning("请先解析内容，再保存到向量记忆库。")
            else:
                try:
                    snapshot = engine.save_memory(st.session_state.parsed)
                    st.session_state.memory_snapshot = snapshot
                    st.success(f"已写入记忆库：{snapshot.title}，共 {snapshot.chunks} 段。")
                except Exception as exc:
                    st.error(f"保存记忆失败：{exc}")

    with action_cols[2]:
        if st.button("3. 生成题目", use_container_width=True):
            try:
                if source_mode == "当前内容":
                    if st.session_state.parsed:
                        st.session_state.quiz = engine.generate_quiz(st.session_state.parsed, config)
                    elif source_text.strip():
                        st.session_state.parsed, st.session_state.quiz = engine.generate_from_source(
                            source_text,
                            source_type,
                            config,
                        )
                        st.info("本次使用了单次调用的“解析+出题”合并策略，以节省时间和成本。")
                    else:
                        st.warning("请先输入学习内容。")
                        return
                else:
                    st.session_state.parsed, st.session_state.quiz = engine.generate_from_memory(
                        config=config,
                        query=memory_query,
                        top_k=memory_top_k,
                    )
                st.session_state.report = None
                st.session_state.reinforcement_quiz = None
                start_exam_if_needed(quiz_mode, exam_minutes)
            except Exception as exc:
                st.error(f"生成题目失败：{exc}")

    process_exam_timeout(grader)
    render_exam_banner()
    render_parsed_summary()
    render_quiz()

    if st.session_state.quiz and not st.session_state.report:
        submit_label = "4. 提交并批改" if st.session_state.last_generation_mode == QuizMode.practice.value else "4. 交卷"
        if st.button(submit_label, type="primary", use_container_width=True):
            if (
                st.session_state.last_generation_mode == QuizMode.exam.value
                and st.session_state.exam_deadline
                and datetime.now() > datetime.fromisoformat(st.session_state.exam_deadline)
            ):
                st.warning("考试时间已结束，系统将按当前作答自动交卷。")
            st.session_state.report = grader.grade(st.session_state.quiz, load_answers_from_state())
            st.session_state.exam_timeout_processed = True

    render_report()

    if st.session_state.report and st.button("5. 生成强化题", use_container_width=True):
        try:
            st.session_state.reinforcement_quiz = build_reinforcement_quiz(
                st.session_state.parsed,
                st.session_state.report,
                config,
            )
        except Exception as exc:
            st.error(f"强化题生成失败：{exc}")

    if st.session_state.reinforcement_quiz:
        st.subheader("针对性强化题")
        for question in st.session_state.reinforcement_quiz.questions:
            st.markdown(
                f"- **{question.id} {question.question_type.value}** | 知识点：{'、'.join(question.knowledge_tags)}  \n"
                f"{question.prompt}"
            )


if __name__ == "__main__":
    main()
