from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from quizmind.models import ParsedContent, Quiz


class QuizExporter:
    def export_json(self, target: Path, parsed: ParsedContent, quiz: Quiz) -> Path:
        payload = {"parsed": parsed.model_dump(), "quiz": quiz.model_dump()}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def export_markdown(self, target: Path, parsed: ParsedContent, quiz: Quiz) -> Path:
        lines = [f"# {quiz.title}", "", f"- Source: {parsed.title}", f"- Summary: {quiz.source_summary}", ""]
        for i, q in enumerate(quiz.questions, start=1):
            lines.append(f"## {i}. {q.prompt}")
            lines.append(f"- Type: {q.question_type.value}")
            lines.append(f"- Difficulty: {q.difficulty.value}")
            lines.append(f"- Tags: {', '.join(q.knowledge_tags)}")
            if q.options:
                lines.append("- Options:")
                for option in q.options:
                    lines.append(f"  - {option}")
            lines.append(f"- Correct answer: {', '.join(q.correct_answer)}")
            lines.append(f"- Explanation: {q.explanation}")
            lines.append("")
        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    def export_pdf(self, target: Path, parsed: ParsedContent, quiz: Quiz) -> Path:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        c = canvas.Canvas(str(target), pagesize=A4)
        c.setFont("STSong-Light", 11)
        _, height = A4
        x = 15 * mm
        y = height - 15 * mm
        line_height = 6 * mm

        def draw_line(text: str) -> None:
            nonlocal y
            if y < 18 * mm:
                c.showPage()
                c.setFont("STSong-Light", 11)
                y = height - 15 * mm
            c.drawString(x, y, text[:1000])
            y -= line_height

        draw_line(f"Quiz: {quiz.title}")
        draw_line(f"Source: {parsed.title}")
        draw_line(f"Summary: {quiz.source_summary}")
        y -= 2 * mm
        for i, q in enumerate(quiz.questions, start=1):
            draw_line(f"{i}. {q.prompt}")
            draw_line(f"Type: {q.question_type.value} | Difficulty: {q.difficulty.value}")
            if q.options:
                for option in q.options:
                    draw_line(f"Option: {option}")
            draw_line(f"Correct answer: {', '.join(q.correct_answer)}")
            draw_line(f"Explanation: {q.explanation}")
            y -= 1 * mm

        c.save()
        return target
