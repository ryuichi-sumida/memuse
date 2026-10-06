# MemUse — Natural Integration human-annotation pilot (Appendix B.4)

The 56-item recalibration pilot used in Appendix B.4 / Table "multi_judge" of
*MemUse: Moving Memory Evaluation from Direct QA to Natural Integration in Long-Term
Human-AI Conversation*, released for judge-agreement studies.

## File

`ni_pilot.jsonl` — one item per line:

| field | description |
|---|---|
| `item_id` | 0–55 |
| `memuse_instance_id` | matching instance in the public benchmark (`RuiSumida/memuse`, `instances.test.jsonl`); `null` for item 54, whose instance (annotator_40) is excluded from the public benchmark |
| `user_id`, `event_type` | as in the public benchmark |
| `condition` | memory condition that produced the response (7 conditions × 8 items, stratified) |
| `trigger` | the user's memory cue (`{trigger_quote}` in the judge prompt). This is the extracted quote span; the public benchmark's `trigger_quote` is the full user turn containing it |
| `expected` | expected-memory description, exactly as shown to raters and the judge (`{ground_truth_facts}`; all ≤ 300 chars, so the judge's 300-char truncation has no effect). Worded differently from `target.ground_truth` in the public instances |
| `response` | GPT-5.4 natural response (`{response}`) |
| `label_human1`, `label_human2` | independent human labels (`yes`/`no`) |
| `label_gpt54nano` | GPT-5.4-nano judge label (temperature 0) |

## Judge prompt (identical for humans and the LLM)

System: `Answer ONLY 'yes' or 'no'.`

```
User said: {trigger}
Expected: {expected}
Assistant responded: {response}
Does the response demonstrate memory of the referenced topic?
```

Human annotators were calibrated beforehand on 3 worked examples (not included).

## Reference numbers (reproducible from this file)

Positive rate: human1 55.4%, human2 51.8%, GPT-5.4-nano 51.8%.
Cohen's κ: human1–human2 0.57, GPT-5.4-nano–human1 0.43, GPT-5.4-nano–human2 0.21.
Fleiss' κ (3 raters): 0.40.

## Notes

- Text is pseudonymized with the same mapping as the public release (personal/pet
  names, residence-level places, small local organizations), so names match the
  public benchmark. The paper's numbers were computed before pseudonymization; labels
  are unchanged.
- The 316-question Direct QA annotations from the original validation round are not
  included.
- License: CC BY-NC 4.0, same as the MemUse data.
