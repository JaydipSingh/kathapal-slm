"""Tests for evaluation metrics, baselines, and report generation (offline)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.evaluation.evaluator import (  # noqa: E402
    evaluate_adapter,
    evaluate_template_baseline,
)
from kathapal.evaluation.metrics import compute_story_metrics  # noqa: E402
from kathapal.evaluation.report import write_reports  # noqa: E402
from kathapal.models.base import GenerationParams, StubModelAdapter  # noqa: E402
from kathapal.models.prompting import StoryRequest  # noqa: E402
from kathapal.safety.fallback import get_fallback  # noqa: E402


def _requests():
    return [
        StoryRequest(age=4, language="en", topic="animals", learning_goal="kindness"),
        StoryRequest(age=5, language="en", topic="kindness", learning_goal="sharing"),
        StoryRequest(age=3, language="en", topic="nature", learning_goal="curiosity"),
    ]


def test_metrics_on_fallback():
    m = compute_story_metrics(get_fallback("animals"))
    assert m.schema_valid
    assert m.word_count >= 20
    assert m.has_question
    assert 0.0 <= m.repetition_ratio <= 1.0


def test_template_baseline_arm():
    arm = evaluate_template_baseline(_requests())
    agg = arm.aggregate()
    assert agg["n"] == 3
    assert agg["safety"]["pass_rate"] == 1.0  # curated fallbacks are safe


def test_adapter_arm_with_stub():
    arm = evaluate_adapter(
        "stub-model", StubModelAdapter(), _requests(), GenerationParams(deterministic=True)
    )
    agg = arm.aggregate()
    assert agg["n"] == 3
    # stub returns a valid animals story; all three should parse (no fallback)
    assert agg["fallback_count"] == 0


def test_report_writes_files(tmp_path):
    template_arm = evaluate_template_baseline(_requests())
    model_arm = evaluate_adapter(
        "stub-model", StubModelAdapter(), _requests(), GenerationParams(deterministic=True)
    )
    paths = write_reports([model_arm, template_arm], tmp_path, meta={"run": "test"})
    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    js = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert "Evaluation Report" in md
    assert "Limitations" in md
    assert len(js["arms"]) == 2
