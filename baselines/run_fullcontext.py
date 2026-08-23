"""
Simple reference baselines for MemUse (positives or negatives), release format.

Backends (GPT-4.1-mini, temp 0, the deployed persona prompt):
  full  : ALL prior-session turns are prepended as earlier messages (LC-100%-style),
          then the current session up to the trigger.
  none  : current session only (no long-term memory at all) — a floor that should
          never fabricate prior-session memory.

Natural replies only (enough for Natural Integration / Reference on positives and
for both false-positive metrics on negatives). Direct QA is not produced here.

  uv run --directory .../code/Equ python .../benchmark/run_fullcontext.py \
      --benchmark ../data/negatives.test.jsonl --backend full --output results/negatives/full/responses.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv()  # optional .env in cwd
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY not found")
from openai import OpenAI

GEN_MODEL = "gpt-4.1-mini"
PERSONA_PROMPT = """You are Luke, a friendly, kind, and understanding diary AI chatbot.
Your goal is to make conversations feel like chatting with a friend. You are also a reflective companion who helps the user think about their day.
- Ask questions only a third of the time.
- Adapt your tone based on the user's mood.
- Aim for one or two sentences, like a real conversation.
- You are also here to act as a personal diary."""
client = OpenAI()


def msgs(turns):
    return [{"role": t["role"], "content": t["content"]} for t in turns if t.get("content")]


def build(inst, backend):
    cur = inst["input"]["current_session"]
    n_prior = len(inst["input"]["sessions"])
    system = PERSONA_PROMPT + f"\n\nThis is conversation #{inst['trigger_session_idx'] + 1} with this user."
    history = []
    if backend == "full":
        for s in inst["input"]["sessions"]:
            history.append({"role": "user", "content": f"[Previous conversation on {s['date']}]"})
            history += msgs(s["turns"])
        if history:
            history.append({"role": "user", "content": f"[Current conversation on {cur['date']}]"})
            history.append({"role": "assistant", "content": "Okay."})
    current = msgs(cur["turns"])
    if not current:
        current = [{"role": "user", "content": inst["input"]["trigger_quote"]}]
    return system, history + current


def gen(system, messages):
    for attempt in range(5):
        try:
            r = client.chat.completions.create(model=GEN_MODEL, temperature=0, max_tokens=500,
                                               messages=[{"role": "system", "content": system}] + messages)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            time.sleep(3 * (attempt + 1)); last = e
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--backend", choices=["full", "none"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    insts = [json.loads(l) for l in open(args.benchmark)]
    out = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(gen, *build(i, args.backend)): i["instance_id"] for i in insts}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            print(f"{futs[f]} done ({len(out)}/{len(insts)})", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"natural": out}, open(args.output, "w"), indent=1, ensure_ascii=False)
    print("saved", args.output)


if __name__ == "__main__":
    main()
