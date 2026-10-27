# Safety and Privacy

## This is a prototype — read first

KathaPal-SLM is a research prototype. **Model outputs can still fail safety
checks.** Rule-based filters and a small fine-tuned model **do not guarantee**
child-appropriate content. **Adult supervision is required at all times during
any testing with children.** Do not deploy to children unsupervised.

## Layered safety design

Safety does not rely on the model alone:

1. **Topic allowlist** — only a fixed set of preschool-appropriate topics.
2. **Age / length bounds** — enforced by the request and output schema.
3. **Strict output schema** — malformed output is rejected.
4. **Rule-based content checks** — flag violence, frightening content, sexual
   content, hate, dangerous instructions, personal-information requests, and
   adult themes. These are transparent keyword/pattern heuristics with known
   false negatives and false positives.
5. **Curated fallback library** — any failure returns a reviewed, safe story
   instead of raw output.
6. **Adult-controlled flow** — the intended product keeps an adult in control.

## Deliberately NOT included (initial version)

- Open-ended autonomous agents or tool use.
- Unrestricted web search.
- Child profiles, persistent recording, or remote surveillance.

## Privacy

- The system must never encourage a child to disclose their full name, address,
  school, phone number, or other personal information. The rule checker flags
  such requests.
- Logging must not record children's personal data. Do not pass raw end-user
  content into logs at INFO or above.

## Honest limitations

- Keyword-based checks cannot guarantee safety; they are a baseline layer.
- A small model on a small dataset may produce incoherent or off-topic text; the
  schema + fallback layers contain this but cannot make a weak model good.
- Treat every output as requiring human review before it reaches a child.
