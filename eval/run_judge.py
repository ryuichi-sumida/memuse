"""
Reference scorer for MemUse — runs the three metrics against a responses file.

Reads:
  ../data/instances.test.jsonl       # the benchmark
  --responses  (path)                # your system's outputs

Writes:
  --output   (path, default report.json)

A responses file is one JSON object with two top-level fields:

  {
    "natural": {
      "<instance_id>": "<system-under-test's reply to input.trigger_quote>"
    },
    "direct_qa": {                             # optional; required only for the
      "<instance_id>": [                       # Direct QA auxiliary metric
        "<reply to sub_questions[0].question>",
        "<reply to sub_questions[1].question>",
        ...
      ]
    }
  }

If "direct_qa" is omitted, only Natural Integration is scored (the primary
metric) and Reference is scored from the "natural" replies. This is enough to
report the headline number.

The judge is GPT-5.4-nano @ temp 0 (paper default), but swap_judge() shows how
to use any callable; that lets you drop in your own SDK / model without
editing the rest of the file.

  uv run --with openai python run_judge.py --responses my_responses.json
"""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
from typing import Awaitable, Callable, Optional

HERE = Path(__file__).resolve().parent
METADATA = json.load(open(HERE / "metadata.json"))
PROMPTS = {
    name: (HERE / spec["prompt_file"]).read_text()
    for name, spec in METADATA["metrics"].items()
}
NATURAL_SYS = METADATA["metrics"]["natural_integration"]["system_prompt"]
DIRECT_QA_SYS = METADATA["metrics"]["direct_qa"]["system_prompt"]
JUDGE_CFG = METADATA["judge"]

JudgeFn = Callable[[str, str], Awaitable[bool]]  # (system_prompt, user_prompt) -> yes/no


# --- default judge: OpenAI GPT-5.4-nano, matches the paper exactly ----------
def make_openai_judge() -> JudgeFn:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    async def judge(system: str, user: str) -> bool:
        r = await client.chat.completions.create(
            model=JUDGE_CFG["model"],
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=JUDGE_CFG["temperature"],
            max_completion_tokens=JUDGE_CFG["max_completion_tokens"],
        )
        return (r.choices[0].message.content or "").strip().lower().startswith("yes")

    return judge


# Hook for using a different SDK / model: pass your own JudgeFn to score().
def swap_judge(fn: JudgeFn) -> JudgeFn:
    return fn


# --- prompt construction ----------------------------------------------------
def natural_prompt(instance: dict, response: str) -> str:
    return PROMPTS["natural_integration"].format(
        trigger_quote=instance["input"]["trigger_quote"],
        ground_truth=instance["target"]["ground_truth"][:300],   # paper truncation
        response=response,
    )

def qa_prompt(sub_q: dict, response: str) -> str:
    return PROMPTS["direct_qa"].format(
        response=response,
        question=sub_q["question"],
        expected_answer=sub_q["answer"],
    )


# --- scoring ----------------------------------------------------------------
async def score(
    instances: list[dict],
    natural_responses: dict[str, str],
    direct_qa_responses: Optional[dict[str, list[str]]],
    judge: JudgeFn,
    concurrency: int = 20,
) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def with_sem(coro):
        async with sem:
            return await coro

    # Natural Integration (primary): one judge call per instance.
    nat_pairs = [(i, natural_responses[i["instance_id"]]) for i in instances if i["instance_id"] in natural_responses]
    nat_tasks = [with_sem(judge(NATURAL_SYS, natural_prompt(i, r))) for i, r in nat_pairs]
    nat_yes = await asyncio.gather(*nat_tasks)
    per_instance_nat = {i["instance_id"]: bool(y) for (i, _), y in zip(nat_pairs, nat_yes)}

    # Reference: same DirectQA judge prompt against the SAME natural reply.
    ref_tasks, ref_keys = [], []
    for inst, resp in nat_pairs:
        for k, sq in enumerate(inst["target"]["sub_questions"]):
            ref_keys.append((inst["instance_id"], k))
            ref_tasks.append(with_sem(judge(DIRECT_QA_SYS, qa_prompt(sq, resp))))
    ref_yes = await asyncio.gather(*ref_tasks)
    per_q_ref = dict(zip(ref_keys, [bool(y) for y in ref_yes]))

    # Direct QA: only if the user supplied per-sub-question replies.
    per_q_qa: dict[tuple[str, int], bool] = {}
    if direct_qa_responses:
        qa_tasks, qa_keys = [], []
        for inst in instances:
            replies = direct_qa_responses.get(inst["instance_id"])
            if not replies:
                continue
            for k, sq in enumerate(inst["target"]["sub_questions"]):
                if k >= len(replies):
                    break
                qa_keys.append((inst["instance_id"], k))
                qa_tasks.append(with_sem(judge(DIRECT_QA_SYS, qa_prompt(sq, replies[k]))))
        qa_yes = await asyncio.gather(*qa_tasks)
        per_q_qa = dict(zip(qa_keys, [bool(y) for y in qa_yes]))

    def rate(d):
        return round(sum(d.values()) / len(d), 4) if d else None

    return {
        "summary": {
            "natural_integration": rate(per_instance_nat),
            "reference":           rate(per_q_ref),
            "direct_qa":           rate(per_q_qa) if direct_qa_responses else None,
            "n_instances_scored":  len(per_instance_nat),
            "n_sub_questions":     len(per_q_ref),
            "judge_model":         JUDGE_CFG["model"],
        },
        "per_instance":   {iid: {"natural_integration": v} for iid, v in per_instance_nat.items()},
        "per_sub_question": [
            {"instance_id": iid, "sub_q_idx": k,
             "reference": per_q_ref.get((iid, k)),
             "direct_qa": per_q_qa.get((iid, k))}
            for (iid, k) in ref_keys
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(HERE.parent / "data" / "instances.test.jsonl"))
    ap.add_argument("--responses", required=True, help="JSON with {natural: {...}, direct_qa: {...}}")
    ap.add_argument("--output",    default="report.json")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    instances = [json.loads(l) for l in open(args.benchmark)]
    resp_obj = json.load(open(args.responses))
    nat   = resp_obj.get("natural") or {}
    qa    = resp_obj.get("direct_qa")
    if not nat:
        raise SystemExit('responses file must include a "natural" map of {instance_id: reply}')

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Either export it, or import run_judge and pass your own JudgeFn via swap_judge().")

    judge = make_openai_judge()
    report = asyncio.run(score(instances, nat, qa, judge, args.concurrency))

    json.dump(report, open(args.output, "w"), indent=2)
    s = report["summary"]
    print(f"Natural Integration : {s['natural_integration']}   ({s['n_instances_scored']}/72 instances)")
    print(f"Reference           : {s['reference']}             ({s['n_sub_questions']} sub-Qs)")
    if s["direct_qa"] is not None:
        print(f"Direct QA           : {s['direct_qa']}")
    else:
        print(f"Direct QA           : not scored (omit \"direct_qa\" key in responses file to skip)")
    print(f"\nFull report -> {args.output}")


if __name__ == "__main__":
    main()
