"""
MemUse baseline runner: Mem0 (OSS, self-hosted) as a modern memory system.

Fair comparison to the paper's conditions: the GENERATOR (GPT-4.1-mini + the same
persona prompt) and JUDGE (GPT-5.4-nano, via release/eval/run_judge.py) are held
constant; only the memory mechanism changes (Mem0's extract/store/retrieve replaces
LC/RAG). For each instance we retrieve ONCE on the trigger and reuse that same memory
context for both the natural reply and every Direct-QA answer ("identical reconstructed
context", matching the paper).

Outputs release-format responses.json: {"natural": {id: reply}, "direct_qa": {id: [..]}}

Usage:
  python run_mem0.py \
    [--limit N] [--workers 6] [--top-k 10]
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

# Disable Mem0 telemetry BEFORE importing mem0: its telemetry "migrations" Qdrant
# store lives at a single fixed per-process path (~/.mem0/migrations_qdrant) that all
# concurrent Memory instances would fight over ("Storage folder already accessed").
os.environ.setdefault("MEM0_TELEMETRY", "False")

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "data" / "instances.test.jsonl"
OUT_DIR = HERE / "results" / "mem0"

# --- load OpenAI key (no MEM0_API_KEY needed for OSS Memory) ---
load_dotenv()  # optional .env in cwd
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY not set")

from openai import OpenAI
from mem0 import Memory

GEN_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"
TEMPERATURE = 0

# Verbatim persona prompt from generate.py (the deployed pipeline), for fairness.
PERSONA_PROMPT = """You are Luke, a friendly, kind, and understanding diary AI chatbot.
Your goal is to make conversations feel like chatting with a friend. You are also a reflective companion who helps the user think about their day.
- Ask questions only a third of the time.
- Adapt your tone based on the user's mood.
- Aim for one or two sentences, like a real conversation.
- You are also here to act as a personal diary."""

_oai = OpenAI()
_print_lock = threading.Lock()


# Per-user isolated Qdrant storage dirs (local Qdrant clients cannot share a path
# across concurrent processes/threads — otherwise "Storage folder already accessed").
QDRANT_ROOT = Path(os.environ.get("MEM0_QDRANT_ROOT", str(HERE / "results" / "qdrant_stores")))


def mem0_config(collection: str, path: str) -> dict:
    return {
        "llm": {"provider": "openai",
                "config": {"model": GEN_MODEL, "temperature": 0}},
        "embedder": {"provider": "openai",
                     "config": {"model": EMBED_MODEL}},
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": collection, "path": path,
                                    "embedding_model_dims": 1536}},
    }


def turns_to_messages(turns: list[dict]) -> list[dict]:
    out = []
    for t in turns:
        role = t.get("role")
        content = (t.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def current_up_to_trigger(current_session: dict, trigger_quote: str) -> list[dict]:
    """Current-session turns up to and including the trigger user turn."""
    turns = turns_to_messages(current_session.get("turns", []))
    key = (trigger_quote or "").strip()[:80].lower()
    result = []
    for t in turns:
        result.append(t)
        if t["role"] == "user" and key and key in t["content"].lower():
            return result
    # trigger not found -> first user turn only
    result = []
    for t in turns:
        result.append(t)
        if t["role"] == "user":
            break
    return result


def format_memories(hits) -> str:
    results = hits.get("results") if isinstance(hits, dict) else hits
    mems = [m.get("memory", "") for m in (results or []) if m.get("memory")]
    if not mems:
        return "No relevant memories found."
    return "\n".join(f"- {m}" for m in mems)


def gen(system: str, messages: list[dict], max_tokens: int) -> str:
    for attempt in range(3):
        try:
            r = _oai.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=TEMPERATURE,
                max_completion_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return f"[ERROR: {e}]"


def process_user(user_id: str, insts: list[dict], top_k: int) -> dict:
    """Incrementally ingest one user's history; produce natural + direct_qa replies."""
    insts = sorted(insts, key=lambda r: int(r["trigger_session_idx"]))
    # Union of sessions seen for this user, keyed by session_idx (content is identical
    # across instances; the longest input.sessions is a superset of the shorter ones).
    sess_by_idx: dict[int, dict] = {}
    for r in insts:
        for s in r["input"]["sessions"]:
            sess_by_idx[int(s["session_idx"])] = s

    store_path = str(QDRANT_ROOT / f"u_{user_id}")
    mem = Memory.from_config(mem0_config(f"u_{user_id}", store_path))
    ingested = 0
    natural, direct_qa = {}, {}

    for r in insts:
        iid = r["instance_id"]
        tsi = int(r["trigger_session_idx"])
        # ingest sessions [ingested .. tsi-1]
        for idx in range(ingested, tsi):
            s = sess_by_idx.get(idx)
            if not s:
                continue
            msgs = turns_to_messages(s.get("turns", []))
            if msgs:
                try:
                    mem.add(msgs, user_id=user_id)
                except Exception as e:
                    with _print_lock:
                        print(f"  [warn] add failed u={user_id} sess={idx}: {e}")
        ingested = max(ingested, tsi)

        trigger = r["input"]["trigger_quote"]
        # Retrieve ONCE on the trigger; reuse for natural + all sub-questions.
        try:
            hits = mem.search(trigger, filters={"user_id": user_id}, top_k=top_k)
        except Exception as e:
            hits = {"results": []}
            with _print_lock:
                print(f"  [warn] search failed {iid}: {e}")
        mem_text = format_memories(hits)

        # --- Natural reply (conversational, to the trigger) ---
        nat_sys = (
            PERSONA_PROMPT
            + f"\n\nThis is conversation #{tsi + 1} with this user.\n"
            + "Relevant memories retrieved from previous conversations:\n"
            + mem_text
            + "\n\nRespond naturally to the user, using these memories if relevant."
        )
        nat_msgs = current_up_to_trigger(r["input"]["current_session"], trigger)
        if not nat_msgs:
            nat_msgs = [{"role": "user", "content": trigger}]
        natural[iid] = gen(nat_sys, nat_msgs, max_tokens=500)

        # --- Direct QA (literal fact probes, SAME memory context) ---
        qa_sys = (
            "You are answering questions using your memory of past conversations with this user.\n"
            "Relevant memories:\n" + mem_text + "\n\nAnswer the question concisely and directly."
        )
        answers = []
        for sq in r["target"].get("sub_questions", []):
            answers.append(gen(qa_sys, [{"role": "user", "content": sq["question"]}], max_tokens=150))
        direct_qa[iid] = answers

        with _print_lock:
            print(f"  done {iid} (u={user_id}, sess<={tsi}, mems={mem_text.count(chr(10)) + 1})")

    return {"natural": natural, "direct_qa": direct_qa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit #instances (debug)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output", default=str(OUT_DIR / "responses.json"))
    ap.add_argument("--benchmark", default=str(BENCHMARK))
    args = ap.parse_args()

    instances = [json.loads(l) for l in open(args.benchmark)]
    if args.limit:
        instances = instances[: args.limit]
    by_user = defaultdict(list)
    for r in instances:
        by_user[r["user_id"]].append(r)
    print(f"Loaded {len(instances)} instances across {len(by_user)} users. "
          f"Generator={GEN_MODEL}, top_k={args.top_k}, workers={args.workers}")

    natural, direct_qa = {}, {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_user, u, insts, args.top_k): u for u, insts in by_user.items()}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                part = fut.result()
                natural.update(part["natural"])
                direct_qa.update(part["direct_qa"])
                with _print_lock:
                    print(f"[user {u} complete] {len(part['natural'])} instances "
                          f"({len(natural)}/{len(instances)} total, {time.time()-start:.0f}s)")
            except Exception as e:
                with _print_lock:
                    print(f"[ERROR user {u}]: {e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"natural": natural, "direct_qa": direct_qa},
              open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(natural)} natural + {len(direct_qa)} direct_qa -> {out}")
    print(f"Total time: {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
