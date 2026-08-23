"""
MemUse baseline runner: MemGPT / Letta (end-to-end memory agent) — SECONDARY, CAVEATED.

Unlike Mem0 (which swaps only the retrieval step while the paper's GPT-4.1-mini generator is
held fixed), Letta is a full agent that does its OWN generation with its own reasoning/tool
loop. We pin its model to openai/gpt-4.1-mini and embedding to text-embedding-3-small to reduce
the confound, but its system prompt + agentic loop still differ from the deployed pipeline. So
its numbers answer the reviewer's literal "end-to-end memory system" ask; they are NOT an
apples-to-apples controlled comparison like Mem0.

Per instance: fresh agent, load input.sessions into archival memory, ask the trigger (natural
reply), then ask each sub-question (direct_qa) to the same agent. Emits release-format
responses.json: {"natural": {...}, "direct_qa": {...}}.

Usage:
  uv run --directory .../Equ python .../benchmark/run_letta.py [--limit N] [--workers 4]
"""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from letta_client import Letta

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent / "data" / "instances.test.jsonl"
OUT_DIR = HERE / "results" / "letta"

BASE_URL = "http://localhost:8283"
MODEL = "openai/gpt-4.1-mini"
EMBED = "openai/text-embedding-3-small"

PERSONA = ("You are Luke, a friendly, kind, and understanding diary AI chatbot. Make "
           "conversations feel like chatting with a friend; be a reflective companion who "
           "helps the user think about their day. Ask questions only a third of the time, "
           "adapt to the user's mood, and keep replies to one or two sentences. You keep a "
           "long-term archival memory of the user's past diary entries: whenever the user "
           "references or relates to anything from the past, call archival_memory_search to "
           "recall the relevant details before you respond.")

_print_lock = threading.Lock()
_client = Letta(base_url=BASE_URL)

# Restrict to archival search + send_message so the MemGPT agent actually consults its
# archival memory (the default base toolset prefers conversation_search, which cannot see
# loaded passages). This is the fair parallel to Mem0: ensure memory is retrievable, then
# measure whether it is integrated into the natural reply vs. surfaced on a direct probe.
_tool_ids = {getattr(t, "name", None): t.id for t in _client.tools.list(limit=200)}
_AGENT_TOOLS = [_tool_ids[n] for n in ("archival_memory_search", "send_message") if n in _tool_ids]


def session_text(s: dict) -> str:
    lines = [f"[Prior session {s.get('session_idx')}, {s.get('date','')}]"]
    for t in s.get("turns", []):
        c = (t.get("content") or "").strip()
        if c:
            lines.append(f"{t.get('role')}: {c}")
    return "\n".join(lines)


def assistant_text(resp) -> str:
    """Extract the last assistant_message content from a Letta message response."""
    out = ""
    for m in resp.messages:
        if getattr(m, "message_type", None) == "assistant_message":
            c = getattr(m, "content", None)
            if isinstance(c, list):
                c = " ".join(getattr(x, "text", str(x)) for x in c)
            if c:
                out = c
    return out.strip()


def ask(agent_id: str, text: str) -> str:
    for attempt in range(3):
        try:
            resp = _client.agents.messages.create(
                agent_id=agent_id, messages=[{"role": "user", "content": text}])
            return assistant_text(resp)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                return f"[ERROR: {e}]"


def process_instance(r: dict) -> tuple[str, str, list[str]]:
    iid = r["instance_id"]
    agent = _client.agents.create(
        agent_type="memgpt_agent",
        name=f"memuse_{iid}_{int(time.time()*1000)%100000}",
        memory_blocks=[{"label": "persona", "value": PERSONA},
                       {"label": "human", "value": "The user writes daily diary entries."}],
        model=MODEL, embedding=EMBED,
        include_base_tools=False, tool_ids=_AGENT_TOOLS,
    )
    try:
        # Load prior history into archival memory (MemGPT's long-term store). Batch several
        # sessions per passage.create to cut round-trips (~1 call/session -> a few/instance);
        # Letta chunks each passage internally by embedding_chunk_size, so retrieval
        # granularity is preserved.
        batch, blen = [], 0
        def flush(chunks):
            if not chunks:
                return
            try:
                _client.agents.passages.create(agent_id=agent.id, text="\n\n".join(chunks))
            except Exception as e:
                with _print_lock:
                    print(f"  [warn] passage {iid}: {e}")
        for s in r["input"]["sessions"]:
            txt = session_text(s)
            if not txt.strip():
                continue
            batch.append(txt); blen += len(txt)
            if blen >= 6000:          # ~a handful of sessions per insert
                flush(batch); batch, blen = [], 0
        flush(batch)
        # Natural reply to the bare trigger (tests spontaneous integration).
        natural = ask(agent.id, r["input"]["trigger_quote"])
        # Direct QA: explicit probe that nudges archival search (mirrors the paper's
        # Direct-QA condition, where the question is asked with the context available).
        answers = [
            ask(agent.id,
                "Based on your memory of our past conversations with this user, answer "
                f"this question. If you are unsure, search your memory first.\nQuestion: {sq['question']}")
            for sq in r["target"]["sub_questions"]
        ]
        return iid, natural, answers
    finally:
        try:
            _client.agents.delete(agent.id)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default=str(OUT_DIR / "responses.json"))
    args = ap.parse_args()

    instances = [json.loads(l) for l in open(BENCHMARK)]
    if args.limit:
        instances = instances[: args.limit]
    print(f"Loaded {len(instances)} instances. Model={MODEL}, workers={args.workers}")

    natural, direct_qa = {}, {}
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_instance, r): r["instance_id"] for r in instances}
        for fut in as_completed(futs):
            iid = futs[fut]
            try:
                _iid, nat, ans = fut.result()
                natural[_iid] = nat
                direct_qa[_iid] = ans
                with _print_lock:
                    print(f"  done {_iid} ({len(natural)}/{len(instances)}, {time.time()-start:.0f}s)")
            except Exception as e:
                with _print_lock:
                    print(f"  [ERROR {iid}]: {e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"natural": natural, "direct_qa": direct_qa}, open(out, "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved {len(natural)} natural + {len(direct_qa)} direct_qa -> {out}")
    print(f"Total time: {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
