# Baselines

All runners write the release response format (`{"natural": {...}, "direct_qa": {...}}`)
and are scored with `../eval/run_judge.py` (positives) or `../eval/run_judge_negatives.py`
(negatives). Generator is GPT-4.1-mini (temperature 0) with the deployed persona prompt.

| script | memory | notes |
|---|---|---|
| `run_fullcontext.py --backend full` | all prior turns in context (LC-100 %-style) | `--backend none` = current session only |
| `run_mem0.py` | Mem0 OSS (`mem0ai` 2.0.11): incremental per-user ingest, retrieve top-10 once on the trigger, same memory context for the natural reply and every Direct-QA answer | local Qdrant store, no Mem0 API key needed |
| `run_letta.py` | Letta / MemGPT (`letta` 0.16.8): fresh `memgpt_agent` per instance, prior sessions loaded into archival memory, tools restricted to `archival_memory_search` + `send_message` | needs a running Letta server (`letta server`, default :8283) |

Letta generates through its own agent loop, so its generation is not perfectly matched to the
deployed pipeline; Mem0 is the cleaner like-for-like comparison.

`results/` holds the per-instance outputs and judge reports behind the numbers in the README
(replies were pseudonymized with the same map as the data).
