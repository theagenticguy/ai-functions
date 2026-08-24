"""Agentic evolutionary search — an AVO-style variation loop over a scored lineage.

The pattern from "AVO: Agentic Variation Operators for Autonomous Evolutionary
Search" (arXiv:2603.24517): instead of confining the model to one-shot
generation inside a fixed pipeline, the whole variation step is an agent that
consults the lineage of previous solutions and their scores, then proposes
the next candidate. A caller-owned score function gates and ranks candidates;
only improvements are committed.

The task here is regex golf — match every string in one list, none in the
other, with the shortest pattern — chosen because the score function is
deterministic and instant: correctness = matches/rejects everything it
should; fitness = brevity. The domain lives entirely in the prompt and the
score function; the loop and the agent are generic (the paper's central
property).
"""

import asyncio
import re

from _utils import display

from ai_functions import ai_function
from ai_functions.experimental.evolution import Lineage, Score, evolve

MATCH = ["afoot", "catfoot", "dogfoot", "fanfoot", "foody", "foolery"]
REJECT = ["Atlas", "Aymoro", "Iberic", "Mahran", "Ormazd", "Silipan"]


# The score function f: a correctness gate plus a fitness value.
# Raising is fine — evolve() turns an exception into a failing Score whose
# message feeds back into the next variation step.
def score_pattern(pattern: str) -> Score:
    compiled = re.compile(pattern)
    missed = [s for s in MATCH if not compiled.search(s)]
    leaked = [s for s in REJECT if compiled.search(s)]
    if missed or leaked:
        return Score(correct=False, notes=f"missed={missed} leaked={leaked}")
    # Shorter is fitter; invert length so higher is better.
    return Score(correct=True, value=-len(pattern), metrics={"length": len(pattern)}, notes=pattern)


# The variation operator: an AI Function that sees the scored lineage and the
# current champion, and proposes the next candidate.
@ai_function
def propose_pattern(lineage_context: str, champion: str, match: list[str], reject: list[str]) -> str:
    """
    You are optimizing a regex-golf pattern. It must match EVERY string in
    {match} and NONE in {reject}. Shorter patterns score higher.

    {lineage_context}

    Current champion pattern: `{champion}` — propose ONE new pattern that is
    correct and shorter. Return only the pattern, no delimiters.
    """


async def vary(lineage: Lineage) -> str:
    best = lineage.best
    return await propose_pattern(
        lineage_context=lineage.as_context(k=5),
        champion=best.candidate if best else "(none yet)",
        match=MATCH,
        reject=REJECT,
    )


async def main():
    lineage, report = await evolve(
        vary,
        score_pattern,
        steps=8,
        stall_after=4,  # no on_stall: four straight rejections end the run
    )
    for attempt in report.attempts:
        outcome = f"committed v{attempt.committed_version}" if attempt.committed_version else "rejected"
        display(f"step {attempt.step}: {outcome} — {attempt.score.notes}")
    best = lineage.best
    if best is not None:
        display(f"\nbest pattern after {report.committed} commits: {best.candidate!r} (len {-best.score.value:g})")


if __name__ == "__main__":
    asyncio.run(main())
