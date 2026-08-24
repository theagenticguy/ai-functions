"""Evolutionary search with a stateful variation thread and a stall supervisor.

`examples/evolution_regex_golf.py` runs each variation step as a one-shot
call: the operator sees the scored lineage but forgets its own reasoning
between steps. This example upgrades the operator to a live AI Thread
(`thread_vary`), so every proposal is a cycle of the SAME conversation —
proposal N remembers why proposals 1..N-1 were shaped the way they were,
which is what makes late-stage refinement smart in the AVO paper
(arXiv:2603.24517).

It also attaches the paper's conditional supervisor (`notify_on_stall`):
after `stall_after` consecutive proposals fail to commit, a redirection is
injected into the operator's own context via the thread's side-channel
`notify`, and the loop keeps going — the operator changes strategy without
losing its history.

The task is the same regex golf as the one-shot example, so the two files
diff cleanly against each other.
"""

import asyncio
import re

from _utils import display

from ai_functions import ai_function
from ai_functions.experimental.evolution import Score, evolve, notify_on_stall, thread_vary

MATCH = ["afoot", "catfoot", "dogfoot", "fanfoot", "foody", "foolery"]
REJECT = ["Atlas", "Aymoro", "Iberic", "Mahran", "Ormazd", "Silipan"]


def score_pattern(pattern: str) -> Score:
    """Correctness gate (match all / leak none) plus fitness (brevity)."""
    compiled = re.compile(pattern.strip())
    missed = [s for s in MATCH if not compiled.search(s)]
    leaked = [s for s in REJECT if compiled.search(s)]
    if missed or leaked:
        return Score(correct=False, notes=f"missed={missed} leaked={leaked}")
    pattern = pattern.strip()
    return Score(correct=True, value=-len(pattern), metrics={"length": len(pattern)}, notes=pattern)


# The proposing function. Spawned into a thread below, so its conversation
# accumulates: every prior proposal, score, and supervisor note stays in
# context across steps (summarization compacts it if the search runs long).
@ai_function
def propose_pattern(lineage_context: str) -> str:
    """
    You are optimizing a regex-golf pattern. It must match EVERY one of
    afoot, catfoot, dogfoot, fanfoot, foody, foolery and NONE of
    Atlas, Aymoro, Iberic, Mahran, Ormazd, Silipan. Shorter is better.

    {lineage_context}

    Propose ONE new pattern that is correct and shorter than the current
    best. Return only the pattern, no delimiters.
    """


async def main():
    handle = await propose_pattern.spawn()
    try:
        lineage, report = await evolve(
            thread_vary(handle),  # stateful: all steps on one thread
            score_pattern,
            steps=12,
            stall_after=4,
            on_stall=notify_on_stall(handle),  # supervisor redirection via notify
        )
        for attempt in report.attempts:
            outcome = f"committed v{attempt.committed_version}" if attempt.committed_version else "rejected"
            display(f"step {attempt.step}: {outcome} — {attempt.score.notes}")
        if (best := lineage.best) is not None:
            display(f"\nbest after {report.committed} commits: {best.candidate.strip()!r} (len {-best.score.value:g})")
    finally:
        await handle.terminate()


if __name__ == "__main__":
    asyncio.run(main())
