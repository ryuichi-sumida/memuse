# MemUse — Evaluation Harness

The judge prompts and reference scorer for the three MemUse metrics, kept here
so the benchmark is bit-exact reproducible without consulting the paper PDF.

## Files

- `prompts/natural_integration.txt` — verbatim user prompt for the **primary**
  metric (binary judgment that the system's natural reply demonstrates memory
  of the referenced topic).
- `prompts/direct_qa.txt` — verbatim user prompt for both **Direct QA** and
  **Reference** (auxiliary). Reference is the same prompt applied to the
  natural-conversation reply instead of a separate QA-style reply.
- `metadata.json` — judge config (model, temperature, max tokens, parse rule,
  validation κ) and the prompt → metric mapping with slot sources.
- `run_judge.py` — minimal async reference scorer (~150 LOC) for the positives.
- `prompts/unprompted_recall.txt`, `prompts/recall_grounding.txt`,
  `run_judge_negatives.py` — false-positive scorer for the **negatives** split
  (`data/negatives.test.jsonl`): *unprompted recall rate* (did the reply bring up a
  prior conversation although nothing was cued?) and *fabricated recall rate*
  (…and is that content absent from the user's real history? — the headline
  false-positive number). Responses file: `{"natural": {"neg_001": "..."}}`.
- `human_validation/` — the paper's 56-item Natural Integration judge-validation
  set (Appendix B.4): items with two human labels and the GPT-5.4-nano label, for
  checking a replacement judge's agreement with humans. See its README.

## Judge — paper default

GPT-5.4-nano · temperature 0 · `max_completion_tokens=5` · parse:
`response.strip().lower().startswith("yes")` → 1, else 0.
System prompt: `Answer ONLY 'yes' or 'no'.` (Direct QA adds: `Does the response
show knowledge of the stated fact?`). Validated against humans at
Cohen's κ = 0.57 (95% bootstrap CI [0.34, 0.78]).

## Inputs

Your system produces a `responses.json`:

```jsonc
{
  "natural": {                      // required — used by Natural Integration and Reference
    "pos_001": "Glad you went to that seminar — I remember you were dreading it.",
    "pos_002": "..."
  },
  "direct_qa": {                    // optional — required only to score Direct QA
    "pos_001": [
      "A lecture/seminar for her work.",
      "She hated/dreaded it ...",
      "It would take a long time ...",
      "She was still recovering from the flu ..."
    ]
  }
}
```

If `direct_qa` is omitted, only Natural Integration (primary) and Reference are
scored — enough for the headline number. To produce `direct_qa` replies, prompt
your system with each `target.sub_questions[i].question` against the **same
reconstructed context** as the natural reply.

## Running

Positives:

```bash
export OPENAI_API_KEY=...
uv run --with openai python run_judge.py --responses my_responses.json
```

Outputs `report.json` with the three rates plus per-instance / per-sub-question
breakdowns.

Negatives (false positives):

```bash
uv run --with openai python run_judge_negatives.py --responses my_neg_responses.json
```

## Using a different judge

If you want to swap the judge (e.g. for a non-OpenAI model, or for a human
annotation pipeline), `run_judge.py` exposes a `JudgeFn` type:

```python
from run_judge import score, swap_judge

async def my_judge(system_prompt: str, user_prompt: str) -> bool:
    # ... call your model, return True for "yes"
    ...

report = await score(instances, natural_responses, direct_qa_responses,
                     judge=swap_judge(my_judge))
```

The paper's reported numbers use the OpenAI default. Other judges may
introduce judge bias — validate against humans before reporting
(`human_validation/ni_pilot.jsonl` has the paper's human labels).

## Reproducibility notes

- The Natural Integration prompt truncates `target.ground_truth` to the first
  300 characters before substitution (matches `analysis/phase4_benchmark.py` in
  the paper's analysis code).
- The same prompts are used across the LLM judge and the human annotators in
  the paper's judge-validation study (Appendix).
- `metadata.json` `schema_version` will be bumped if any prompt or judge config
  changes; pin to a specific version when reporting results.
