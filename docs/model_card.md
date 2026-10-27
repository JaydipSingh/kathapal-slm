# Model Card: KathaPal-SLM

## Overview

KathaPal-SLM is a LoRA fine-tuned small language model that generates short,
structured stories (plus one comprehension question and answer) for children
aged 3–5, from a structured request (age, language, topic, length, learning
goal). It is a **research prototype**, not a production or clinical system.

## Intended use

- Prototyping a preschool storytelling toy / app under **adult supervision**.
- Research and teaching on small, domain-specialized, safety-wrapped LMs.

## Out-of-scope / excluded uses

- Unsupervised interaction with children.
- Any use implying the output is guaranteed safe, factual, or age-appropriate.
- Open-ended chat, web search, child profiles, recording, or data collection.
- Commercial deployment without independent base-model licence verification and
  a far larger, human-reviewed dataset.

## Base model

- Configurable (`configs/model.yaml`). Default candidate: a small instruct model
  (e.g. SmolLM2-360M-Instruct).
- **Licence is UNVERIFIED by this project.** You must verify the base model's
  licence and acceptable-use policy yourself before any commercial use. Set
  `license_verified: true` in config only after doing so.

## Training data

- A small set of **original, hand-authored** examples (`data/samples/`), licensed
  CC-BY-4.0, with recorded provenance. The starter set (~12 examples) is for
  pipeline validation only — **not** sufficient for production quality.
- Reaching usable quality requires a much larger (hundreds+), diverse,
  human-reviewed dataset. The data tooling supports this; the data work is the
  main remaining effort and the main determinant of quality.

## Evaluation

- Structural proxy metrics (schema validity, length, repetition, topic-keyword
  relevance, Q/A presence, vocabulary simplicity).
- Rule-based safety pass-rate (a baseline proxy, **not** a safety guarantee).
- The fine-tuned model is compared against the unmodified base model and a
  curated-template baseline. If the template baseline is competitive, the
  fine-tuned model must clearly beat it to justify its complexity.

## Safety

- Layered: topic allowlist, age/length bounds, strict output schema, rule-based
  content checks, and a curated fallback story library. Any failure returns a
  safe fallback rather than raw output.
- **Limitation:** none of these guarantee safety. Rule-based filters have false
  negatives. Adult supervision is required during any testing with children.

## Limitations

- Small model + tiny starter dataset → limited quality until data is expanded.
- English only initially (architecture allows adding languages later).
- Metrics are proxies, not true quality or safety measures.

## Licence

Software: MIT (see `LICENSE`). Base model: separate, verify independently.
