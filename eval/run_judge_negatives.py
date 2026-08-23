"""
False-positive scorer for the MemUse negative-control split.

The negatives (data/negatives.test.jsonl) are user turns where NO prior-session
memory is needed (memory-event detector found nothing in the session AND a
GPT-5.4 prior-context-dependence check scored 0). A system that over-triggers
memory — "as you mentioned…", "how did your trip go?" — gets caught here.

Two metrics, both from the system's natural reply to input.trigger_quote:
  unprompted_recall_rate : share of negatives where the reply asserts memory of a
                           specific prior conversation (yes/no judge; same judge
                           model/config as the positives).
  fabricated_recall_rate : share of negatives where that asserted memory is NOT
                           grounded in the user's real prior sessions (grounding
                           judge 0/1/2 against target.prior_session_summaries;
                           0 = fabricated). This is the headline false-positive
                           number.
Also reported: grounded_recall_rate (asserted & fully grounded), partial rate.

Responses file: {"natural": {"neg_001": "<reply>", ...}}   (same shape as positives)

  uv run --with openai python run_judge_negatives.py --responses my_neg_responses.json
"""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
METADATA = json.load(open(HERE / "metadata.json"))
NEG = METADATA["negatives"]
P_RECALL = (HERE / NEG["metrics"]["unprompted_recall"]["prompt_file"]).read_text()
P_GROUND = (HERE / NEG["metrics"]["fabricated_recall"]["prompt_file"]).read_text()
RECALL_SYS = NEG["metrics"]["unprompted_recall"]["system_prompt"]
GROUND_SYS = NEG["metrics"]["fabricated_recall"]["system_prompt"]
JUDGE_CFG = METADATA["judge"]


def make_openai_client():
    from openai import AsyncOpenAI
    return AsyncOpenAI()


async def ask(client, system: str, user: str, max_tokens: int) -> str:
    r = await client.chat.completions.create(
        model=JUDGE_CFG["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=JUDGE_CFG["temperature"], max_completion_tokens=max_tokens)
    return (r.choices[0].message.content or "").strip()


def fmt_summaries(inst) -> str:
    return "\n".join(f"[session {s['session_idx']} {s['date']}] {s['summary']}" for s in inst["target"]["prior_session_summaries"]) or "(none)"


async def score(instances, responses, client, concurrency=20) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(inst):
        iid = inst["instance_id"]
        resp = responses.get(iid)
        if resp is None:
            return iid, None
        async with sem:
            recall = (await ask(client, RECALL_SYS, P_RECALL.format(trigger_quote=inst["input"]["trigger_quote"], response=resp), 5)).lower().startswith("yes")
            grounding = None
            if recall:
                g = await ask(client, GROUND_SYS, P_GROUND.format(response=resp, prior_summaries=fmt_summaries(inst)), 5)
                grounding = next((int(c) for c in g if c in "012"), 0)
        return iid, {"unprompted_recall": recall, "grounding": grounding}

    res = dict(await asyncio.gather(*[one(i) for i in instances]))
    scored = {k: v for k, v in res.items() if v is not None}
    n = len(scored)
    rec = [v for v in scored.values() if v["unprompted_recall"]]
    def rate(x): return round(x / n, 4) if n else None
    return {
        "summary": {
            "n_instances_scored": n,
            "unprompted_recall_rate": rate(len(rec)),
            "fabricated_recall_rate": rate(sum(1 for v in rec if v["grounding"] == 0)),
            "partially_grounded_recall_rate": rate(sum(1 for v in rec if v["grounding"] == 1)),
            "grounded_recall_rate": rate(sum(1 for v in rec if v["grounding"] == 2)),
            "judge_model": JUDGE_CFG["model"],
        },
        "per_instance": scored,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(HERE.parent / "data" / "negatives.test.jsonl"))
    ap.add_argument("--responses", required=True)
    ap.add_argument("--output", default="report_negatives.json")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")
    instances = [json.loads(l) for l in open(args.benchmark)]
    responses = json.load(open(args.responses)).get("natural") or {}
    report = asyncio.run(score(instances, responses, make_openai_client(), args.concurrency))
    json.dump(report, open(args.output, "w"), indent=2)
    s = report["summary"]
    print(f"Unprompted recall rate : {s['unprompted_recall_rate']}  ({s['n_instances_scored']} negatives)")
    print(f"Fabricated recall rate : {s['fabricated_recall_rate']}   <- headline false-positive rate")
    print(f"  partially grounded   : {s['partially_grounded_recall_rate']}")
    print(f"  fully grounded       : {s['grounded_recall_rate']}")
    print(f"\nFull report -> {args.output}")


if __name__ == "__main__":
    main()
