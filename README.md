<div align="center">

# MemUse

### Does your memory system actually *use* what it remembers?

[![Paper](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-b31b1b)](https://github.com/ryuichi-sumida/memuse)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RuiSumida%2Fmemuse-ffcc4d)](https://huggingface.co/datasets/RuiSumida/memuse)
[![Code](https://img.shields.io/badge/GitHub-ryuichi--sumida%2Fmemuse-181717?logo=github)](https://github.com/ryuichi-sumida/memuse)
[![Data license](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Code license](https://img.shields.io/badge/code-MIT-green)](https://opensource.org/licenses/MIT)

<img src="assets/fig_story.png" alt="Retrieval rises with memory capacity; integration and satisfaction stay flat" width="900">

*Across 7 randomized memory conditions in a 4-month deployment, retrieval-style scores climb steeply with capacity while natural integration — and user satisfaction — stay flat.*

</div>

MemUse evaluates long-term conversational memory the way users experience it: not *"can the model answer a fact question about last week?"* but *"when the user brings something up, does the reply show it remembers?"*

It is built from a real 4-month deployment of an AI diary companion (40 people, 1,872 sessions, daily use) and packages the rare, naturally occurring moments where a user cued the system's memory — *"as I wrote yesterday…"*, *"do you remember…?"* — together with the full conversation history leading up to each one.

> **Headline finding.** The same GPT-4.1-mini that answers **78.8 %** of MemUse fact questions correctly when asked directly mentions those facts in its *natural* reply only **7.9 %** of the time. Retrieval and integration are dissociable — and in the deployment, only integration was associated with user satisfaction.

<table>
<tr><td width="50%" valign="top">

**One real instance**

🗨️ **User (session 5):** *"I'm thinking I'll try to relax by listening to the music Luke recommended the other day."*

🔎 **Prior context (sessions 2–3):** the system had recommended Bruno Mars, Ryuichi Sakamoto, Hikaru Utada, Yumi Matsutoya, Haruka Nakamura.

</td><td width="50%" valign="top">

❌ **Deployed reply (full history in context):** *"That sounds like a perfect way to unwind. I hope the music brings you some calm…"* — generic, **0/4 facts referenced**

✅ **Same model, asked directly** *"What Japanese artists did you recommend?"* → *"Ryuichi Sakamoto, Hikaru Utada, Yumi Matsutoya, Haruka Nakamura."* — **correct**

</td></tr>
</table>

📄 **Paper:** *MemUse: Moving Memory Evaluation from Direct QA to Natural Integration in Long-Term Human-AI Conversation* — EMNLP 2026 (Main) · 🤗 **Dataset (all configs incl. the 1,872-session corpus):** https://huggingface.co/datasets/RuiSumida/memuse

This repository holds the evaluation harness, baseline runners and per-instance outputs; the benchmark and negatives JSONL are mirrored under `data/`.

---

## What's inside

| config | rows | what it is | use it for |
|---|---|---|---|
| `benchmark` *(default)* | 72 | user-cued memory moments (62 re-provisions + 10 explicit probes), full history inline, gold facts + 316 fact questions | the three core metrics |
| `negatives` | 72 | matched user turns where **no** prior-session memory is needed | false-positive checks (over-eager / fabricated recall) |
| `corpus` | 1,872 | every session of the deployment (40 users, 21,575 turns), with per-session 1–7 satisfaction ratings | your own analyses, new splits |
| `users` | 40 | per-user aggregates + monthly LLM summaries | context / long-term-memory baselines |

Each benchmark instance carries its **entire prior history** (mean ≈ 306 turns / ≈ 13k tokens) cut exactly at the trigger turn, so any memory system — long context, RAG, Mem0/Letta-style agents — can be plugged in without touching the corpus.

---

## Quickstart

**1. Load**

```python
from datasets import load_dataset

bench = load_dataset("RuiSumida/memuse", split="test")                 # 72 positives
negs  = load_dataset("RuiSumida/memuse", "negatives", split="test")    # 72 negatives

ex = bench[0]
history = ex["input"]["sessions"]          # all prior sessions, verbatim
current = ex["input"]["current_session"]   # current session up to the trigger
trigger = ex["input"]["trigger_quote"]     # the user turn your system must reply to
```

**2. Generate** — feed `history` + `current` to your system and reply to `trigger`. Save replies as

```jsonc
{"natural": {"pos_001": "…your reply…", "pos_002": "…"},
 "direct_qa": {"pos_001": ["…reply to sub_question 0…", "…"]}}   // optional
```

**3. Score**

```bash
export OPENAI_API_KEY=...
uv run --with openai python eval/run_judge.py            --responses my_responses.json      # positives
uv run --with openai python eval/run_judge_negatives.py  --responses my_neg_responses.json  # negatives
```

The judge is GPT-5.4-nano at temperature 0, validated against human annotators (details below). `eval/run_judge.py` exposes `swap_judge()` if you want to use another model or human raters.

---

## Repository layout

```
data/        instances.test.jsonl (72 positives) · negatives.test.jsonl (72 negatives)
             → the full corpus / users configs live on the Hugging Face dataset
eval/        run_judge.py · run_judge_negatives.py · prompts/ · metadata.json
baselines/   run_fullcontext.py (full-history / no-memory floors) · run_mem0.py · run_letta.py
             results/{mem0,letta}/                 positives: responses.json + report.json
             results/negatives/{full,none,mem0}/   negatives: responses.json + report_negatives.json
```

Install `pip install openai` for the judge (plus `mem0ai` / `letta` for those baselines) and set `OPENAI_API_KEY`.

---

## Metrics

**On the 72 positives** (same reply, same reconstructed context):

| metric | level | question the judge answers |
|---|---|---|
| **Natural Integration** *(primary)* | per instance | Does the natural reply demonstrate memory of the topic the user just referenced? |
| **Direct QA** | per fact question (316) | Asked literally, can the system answer the fact? *(the classic benchmark format — for comparison)* |
| **Reference** | per fact question (316) | Does that fact actually appear in the natural reply? |

*Direct QA − Reference* is the **retrieval–integration gap**: facts the system can retrieve on demand but never surfaces in conversation.

**On the 72 negatives** (nothing was cued):

| metric | question the judge answers |
|---|---|
| **Unprompted recall rate** | Does the reply bring up a specific prior conversation anyway? |
| **Fabricated recall rate** *(headline false-positive number)* | …and is that recalled content absent from the user's real history? |

A system that always says *"as you mentioned before…"* scores well on Natural Integration but gets caught here.

---

## Reference results

Natural reply generated by the named system; judge fixed at GPT-5.4-nano. Values in %.

**Positives** (from the paper, 73 detected instances; the released 72 drop one contaminated probe, ≤ 1.4 pp effect):

| system | memory | Direct QA ↑ | Reference ↑ | **Natural Integration ↑** |
|---|---|---|---|---|
| GPT-4.1-mini | summary only | 44.9 | 7.9 | 23.6 |
| GPT-4.1-mini | full history (LC-100 %) | 78.8 | 7.9 | 22.2 |
| GPT-5.5 | summary only | 50.0 | 15.4 | 53.4 |
| GPT-5.5 | full history | 82.9 | 13.6 | 52.1 |
| Gemini 3.1 Pro | summary only | 37.9 | 8.2 | 32.9 |
| Gemini 3.1 Pro | full history | 72.3 | 7.6 | 37.0 |
| GPT-4.1-mini + **Mem0** | extract/store/retrieve | 41.5 | 7.3 | **58.3** |
| GPT-4.1-mini + **Letta (MemGPT)** | archival memory agent | 61.4 | 10.8 | **56.9** |

**Negatives** (false positives; lower is better):

| system | memory | unprompted recall ↓ | **fabricated recall ↓** |
|---|---|---|---|
| GPT-4.1-mini | none (current session only) | 0.0 | 0.0 |
| GPT-4.1-mini | full history | 0.0 | 0.0 |

Per-instance outputs and the exact runner scripts are in the GitHub repo (`baselines/`). Mem0 / Letta rows for the negatives will be added as they finish.

---

## Data format

<details>
<summary><b>benchmark</b> row (click to expand)</summary>

```jsonc
{
  "instance_id": "pos_001",
  "user_id": "annotator_11",
  "event_type": "user_re_provision",        // or "user_memory_probe"
  "trigger_session_idx": 4,
  "input": {
    "sessions": [                            // ALL sessions before the trigger session, verbatim
      {"session_idx": 0, "date": "2025/November/05",
       "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    ],
    "current_session": {                     // the trigger session, truncated at the trigger turn
      "session_idx": 4, "date": "2025/November/09",
      "turns": [/* ... */, {"role": "user", "content": "As I wrote yesterday, ..."}]
    },
    "trigger_quote": "As I wrote yesterday, ..."
  },
  "target": {
    "ground_truth": "In session 3 (the previous day) the user ...",   // the referenced prior context
    "sub_questions": [{"question": "...", "answer": "..."}]            // 3–5 fact questions (316 total)
  }
}
```
The assistant's original reply is deliberately **not** included — it is not a gold answer (the deployed system often failed), and it would invite string-matching. Dates matter: many triggers are temporal ("yesterday", "the other day").
</details>

<details>
<summary><b>negatives</b> row</summary>

Same `input` schema. `target` is
```jsonc
{"memory_needed": false, "pcd_score": 0,
 "prior_session_summaries": [{"session_idx": 0, "date": "...", "summary": "..."}, ...]}  // for the grounding judge
```
plus `matched_positive` (the positive it was paired with). Selection: a session in which the memory-event detector found nothing, a substantive user turn, and a GPT-5.4 prior-context-dependence check = 0 ("a good reply needs only the current session"); 70/72 come from the same user as their positive, 2 from the nearest user by history length.
</details>

<details>
<summary><b>corpus</b> row (one session)</summary>

```jsonc
{
  "user_id": "annotator_10", "session_idx": 0, "date": "2025/November/05",
  "duration_minutes": 6.7, "word_count": 86,
  "summary": "...", "daily_summary": "...",
  "user_feedback": {"rating": 6, "comment": "..."},            // 1–7 per-session satisfaction
  "turns": [{"role": "user", "content": "...", "importance": 0.07},
            {"role": "assistant", "content": "...", "latency_seconds_total": 1.59, "input_tokens": 257, "output_tokens": 41, "cost_usd": 0.000168}],
  "usage": {...}, "grand_totals": {...}                          // deployment telemetry, not a target
}
```
`session_idx` is chronological within a user and joins to `trigger_session_idx`. 30 sessions have empty `turns` (app opened, nothing sent). `talking_index` is a raw within-day counter from the study; ignore it.
</details>

<details>
<summary><b>users</b> row</summary>

```jsonc
{"user_id": "annotator_10", "num_sessions": 57, "date_range": {...}, "total_turns": 678,
 "mean_rating": 5.95, "num_rated_sessions": 57, "monthly_summaries": {"2025": {"November": "..."}}}
```
</details>

---

## How it was built

1. **Deployment.** 40 proficient English speakers talked daily with "Luke" (GPT-4.1-mini) for ~4 months under 7 randomized memory conditions and rated every session 1–7.
2. **Detection.** GPT-5.4 scanned all 1,872 sessions for explicit memory moments (probes, re-provisions, proactive recalls, reactions); every candidate was **verified by human annotators** (95.5 % precision).
3. **Benchmark.** The 72 user-cued (reactive) moments were paired with their referenced prior context and decomposed into 316 fact questions. Judges were validated against humans (Natural Integration: human–human κ = 0.57, judge matched the human positive rate; fact questions: Fleiss κ = 0.65; judge stable across models, κ = 0.89 vs GPT-5.5).
4. **Negatives.** Matched turns with no memory cue (see above), added for the public release to measure false positives.

Memory moments are rare — ~1.4 % of user turns — which is exactly why they are worth testing on: existing benchmarks pair externally authored questions with 15–24 % of turns.

---

## Provenance, privacy & license

- Real diary conversations from an IRB-approved longitudinal study; participants consented to public release of their conversations and ratings for research use, self-curated what they shared, and had a withdrawal window (one person withdrew; their data is excluded).
- Conversations were written in English (a few residual Japanese strings were translated).
- **Pseudonymized:** participants are `annotator_NN`; all personal names of participants and third parties (replaced with English names), pet names, and residence-level places / small local organizations were replaced with **consistent substitutes** (same name → same substitute everywhere, including gold facts and questions), after a GPT-5.4 screen of every turn and human review. Public figures, brands, large cities and travel destinations are unchanged. Paper numbers were computed before pseudonymization; the substitution does not change what any instance tests.
- **License:** data CC BY-NC 4.0 (non-commercial research only; no commercial redistribution of the conversational content); evaluation code MIT.

## Citation

```bibtex
@inproceedings{sumida2026memuse,
  title     = {{MemUse}: Moving Memory Evaluation from Direct {QA} to Natural Integration in Long-Term Human-{AI} Conversation},
  author    = {Sumida, Ryuichi and Inoue, Koji and Kawahara, Tatsuya},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
