"""
Evaluation report generation (Markdown + JSON).

Takes one or more ``ArmResult`` objects and writes a comparison report. Only
measured values are reported; nothing is fabricated. Representative sample
outputs and known limitations are included.
"""
from __future__ import annotations

import json
from pathlib import Path

from kathapal.evaluation.evaluator import ArmResult


def build_json_report(arms: list[ArmResult], meta: dict | None = None) -> dict:
    return {
        "meta": meta or {},
        "arms": [arm.aggregate() for arm in arms],
    }


def build_markdown_report(arms: list[ArmResult], meta: dict | None = None) -> str:
    lines: list[str] = []
    lines.append("# KathaPal-SLM Evaluation Report")
    lines.append("")
    if meta:
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Comparison (measured proxies only)")
    lines.append("")
    lines.append("| Arm | n | fallback | avg words | repetition | topic relevance | Q present | safety pass |")
    lines.append("|-----|---|----------|-----------|------------|-----------------|-----------|-------------|")
    for arm in arms:
        a = arm.aggregate()
        safety = a["safety"]["pass_rate"] if a["safety"] else "n/a"
        lines.append(
            f"| {a['name']} | {a['n']} | {a['fallback_count']} | {a['avg_word_count']} | "
            f"{a['avg_repetition']} | {a['avg_topic_relevance']} | "
            f"{a['question_present_rate']} | {safety} |"
        )
    lines.append("")

    # Representative sample from each arm.
    lines.append("## Representative outputs")
    lines.append("")
    for arm in arms:
        if arm.responses:
            r = arm.responses[0]
            lines.append(f"### {arm.name}")
            lines.append(f"**{r.story_title}** (topic: {r.topic})")
            lines.append("")
            lines.append(r.story)
            lines.append("")
            lines.append(f"*Q:* {r.question}  *A:* {r.answer}")
            lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Metrics are structural proxies (length, repetition, keyword overlap), "
        "not measures of true story quality or child-appropriateness."
    )
    lines.append(
        "- Safety pass-rate is from rule-based checks and does NOT guarantee safety."
    )
    lines.append(
        "- A high score for the template baseline would indicate the fine-tuned "
        "model must clearly beat it to justify its added complexity."
    )
    return "\n".join(lines)


def write_reports(arms: list[ArmResult], out_dir: str | Path, meta: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = build_markdown_report(arms, meta)
    js = build_json_report(arms, meta)
    (out / "evaluation_report.md").write_text(md, encoding="utf-8")
    (out / "evaluation_report.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    return {"markdown": str(out / "evaluation_report.md"), "json": str(out / "evaluation_report.json")}
