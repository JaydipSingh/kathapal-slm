"""
Typer CLI for KathaPal-SLM.

Thin wrappers around the same functions the scripts/ use, exposed as a single
`kathapal` command (see pyproject [project.scripts]). Heavy/optional imports are
deferred into each command so the CLI loads without the training extras.
"""
from __future__ import annotations

import typer

from kathapal.logging_config import configure_logging

app = typer.Typer(help="KathaPal-SLM: preschool storytelling SLM toolkit.")


@app.command()
def validate(config: str = typer.Option(None), data: str = typer.Option(None)) -> None:
    """Validate a dataset and print the report."""
    from kathapal.config import load_training_config
    from kathapal.data.validation import validate_dataset

    log = configure_logging()
    path = data or (load_training_config(config).dataset_path if config else None)
    if not path:
        raise typer.BadParameter("provide --config or --data")
    report = validate_dataset(path)
    log.info("records=%d ok=%s", report.total_records, report.ok)
    raise typer.Exit(code=0 if report.ok else 1)


@app.command()
def generate(
    topic: str = typer.Option(...),
    age: int = typer.Option(...),
    language: str = typer.Option("en"),
    stub: bool = typer.Option(False, help="use offline stub adapter"),
) -> None:
    """Generate one story (safe fallback guaranteed)."""
    from kathapal.models.base import GenerationParams, StubModelAdapter
    from kathapal.models.generation import generate_story
    from kathapal.models.prompting import StoryRequest

    log = configure_logging()
    req = StoryRequest(age=age, language=language, topic=topic)
    adapter = StubModelAdapter() if stub else StubModelAdapter()  # real adapter via scripts/generate on Mac
    outcome = generate_story(adapter, req, GenerationParams(deterministic=True))
    r = outcome.response
    typer.echo(f"{r.story_title}\n\n{r.story}\n\nQ: {r.question}\nA: {r.answer}")
    if outcome.used_fallback:
        log.warning("used fallback: %s", outcome.reason)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
