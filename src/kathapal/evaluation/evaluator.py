"""
Evaluator: run a set of story requests through a generation function and
aggregate story-quality metrics + a safety summary.

Provides three comparable arms (per the spec's requirement to compare the
fine-tuned model against the base model AND a simple baseline):

  * ``evaluate_adapter`` — any ModelAdapter (fine-tuned, or unmodified base).
  * ``template_baseline_generate`` — a curated-template generator (no model).

Running all three on the same request set answers the honest question: does the
fine-tuned SLM add enough value over the base model and over a simple template?
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kathapal.data.schema import StoryResponse
from kathapal.evaluation.metrics import StoryMetrics, compute_story_metrics
from kathapal.evaluation.safety_checks import SafetySummary, summarize_safety
from kathapal.models.base import GenerationParams, ModelAdapter
from kathapal.models.generation import generate_story
from kathapal.models.prompting import StoryRequest
from kathapal.safety.fallback import get_fallback


@dataclass
class ArmResult:
    name: str
    responses: list[StoryResponse] = field(default_factory=list)
    metrics: list[StoryMetrics] = field(default_factory=list)
    fallback_count: int = 0
    safety: SafetySummary | None = None

    def aggregate(self) -> dict:
        n = len(self.metrics) or 1
        return {
            "name": self.name,
            "n": len(self.metrics),
            "fallback_count": self.fallback_count,
            "avg_word_count": round(sum(m.word_count for m in self.metrics) / n, 1),
            "avg_repetition": round(sum(m.repetition_ratio for m in self.metrics) / n, 4),
            "avg_topic_relevance": round(sum(m.topic_relevance for m in self.metrics) / n, 4),
            "question_present_rate": round(sum(m.has_question for m in self.metrics) / n, 4),
            "avg_word_length": round(sum(m.avg_word_length for m in self.metrics) / n, 2),
            "safety": self.safety.to_dict() if self.safety else None,
        }


def template_baseline_generate(req: StoryRequest) -> StoryResponse:
    """Curated-template 'generator': returns the reviewed fallback for the topic."""
    return get_fallback(req.topic)


def evaluate_template_baseline(requests: list[StoryRequest]) -> ArmResult:
    arm = ArmResult(name="template-baseline")
    for req in requests:
        resp = template_baseline_generate(req)
        arm.responses.append(resp)
        arm.metrics.append(compute_story_metrics(resp))
    arm.safety = summarize_safety(arm.responses)
    return arm


def evaluate_adapter(
    name: str,
    adapter: ModelAdapter,
    requests: list[StoryRequest],
    params: GenerationParams,
    *,
    max_retries: int = 2,
) -> ArmResult:
    arm = ArmResult(name=name)
    for req in requests:
        outcome = generate_story(adapter, req, params, max_retries=max_retries)
        arm.responses.append(outcome.response)
        arm.metrics.append(compute_story_metrics(outcome.response))
        if outcome.used_fallback:
            arm.fallback_count += 1
    arm.safety = summarize_safety(arm.responses)
    return arm
