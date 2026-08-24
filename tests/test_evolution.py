"""Tests for the evolution module: scores, the lineage commit gate, and the variation loop."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from ai_functions.experimental.evolution import (
    EvolutionReport,
    Lineage,
    RejectedCommit,
    Score,
    evolve,
)

# ══════════════════════════════════════════════════════════════════
# Score: the correctness gate dominates fitness (V1)
# ══════════════════════════════════════════════════════════════════


class TestScore:
    def test_incorrect_forces_zero_value(self):
        s = Score(correct=False, value=99.0)
        assert s.value == 0.0

    def test_correct_keeps_value_and_metrics(self):
        s = Score(correct=True, value=3.5, metrics={"seq4k": 3.2, "seq32k": 3.8})
        assert s.value == 3.5
        assert s.metrics["seq32k"] == 3.8

    def test_frozen(self):
        s = Score(correct=True, value=1.0)
        with pytest.raises(ValidationError):
            s.value = 2.0  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════
# Lineage: append-only, score-gated
# ══════════════════════════════════════════════════════════════════


class TestLineageGate:
    def test_first_commit_seeds_the_lineage(self):
        lineage = Lineage()
        v = lineage.commit("seed", Score(correct=True, value=1.0))
        assert v.version == 1
        assert v.parent_version is None
        assert lineage.best is v

    def test_improvement_commits_with_parent(self):
        lineage = Lineage()
        _ = lineage.commit("seed", Score(correct=True, value=1.0))
        v2 = lineage.commit("better", Score(correct=True, value=2.0))
        assert v2.version == 2
        assert v2.parent_version == 1
        assert lineage.best is v2

    def test_tie_is_admitted_by_default(self):
        lineage = Lineage()
        _ = lineage.commit("seed", Score(correct=True, value=1.0))
        v2 = lineage.commit("same", Score(correct=True, value=1.0))
        assert lineage.best is v2  # latest wins ties

    def test_minimum_improvement_rejects_ties(self):
        lineage = Lineage(minimum_improvement=0.1)
        _ = lineage.commit("seed", Score(correct=True, value=1.0))
        with pytest.raises(RejectedCommit):
            _ = lineage.commit("same", Score(correct=True, value=1.05))

    def test_regression_is_rejected(self):
        lineage = Lineage()
        _ = lineage.commit("seed", Score(correct=True, value=2.0))
        with pytest.raises(RejectedCommit):
            _ = lineage.commit("worse", Score(correct=True, value=1.0))
        assert len(lineage) == 1

    def test_incorrect_is_rejected_regardless_of_value(self):
        lineage = Lineage()
        with pytest.raises(RejectedCommit):
            _ = lineage.commit("broken", Score(correct=False, value=100.0))

    def test_admits_mirrors_the_gate(self):
        lineage = Lineage()
        assert lineage.admits(Score(correct=True, value=0.0))
        assert not lineage.admits(Score(correct=False))
        _ = lineage.commit("seed", Score(correct=True, value=1.0))
        assert not lineage.admits(Score(correct=True, value=0.5))
        assert lineage.admits(Score(correct=True, value=1.0))


class TestLineagePersistence:
    def test_jsonl_roundtrip(self, tmp_path):
        path = tmp_path / "lineage.jsonl"
        lineage = Lineage(path=path)
        _ = lineage.commit({"code": "v1"}, Score(correct=True, value=1.0, notes="seed"))
        _ = lineage.commit({"code": "v2"}, Score(correct=True, value=2.0, metrics={"cfg": 2.0}))

        reloaded = Lineage(path=path)
        assert len(reloaded) == 2
        assert reloaded.best is not None
        assert reloaded.best.candidate == {"code": "v2"}
        assert reloaded.best.score.metrics == {"cfg": 2.0}
        # The gate keeps working against reloaded history.
        with pytest.raises(RejectedCommit):
            _ = reloaded.commit({"code": "v3"}, Score(correct=True, value=1.5))


class TestLineageContext:
    def test_empty_lineage_asks_for_a_seed(self):
        assert "seed" in Lineage().as_context()

    def test_renders_versions_scores_and_notes(self):
        lineage = Lineage()
        _ = lineage.commit("a", Score(correct=True, value=1.0, notes="baseline"))
        _ = lineage.commit("b", Score(correct=True, value=2.5, metrics={"seq4k": 2.5}))
        text = lineage.as_context()
        assert "v1: score=1" in text and "baseline" in text
        assert "v2: score=2.5" in text and "seq4k" in text

    def test_k_limits_the_window(self):
        lineage = Lineage()
        for i in range(5):
            _ = lineage.commit(i, Score(correct=True, value=float(i)))
        text = lineage.as_context(k=2)
        assert "v4" in text and "v5" in text and "v1" not in text


# ══════════════════════════════════════════════════════════════════
# evolve: the variation loop
# ══════════════════════════════════════════════════════════════════


class TestEvolve:
    def test_commits_improvements_and_skips_regressions(self):
        candidates = iter([1.0, 0.5, 2.0, 2.0, 1.5])

        async def vary(lineage: Lineage) -> float:
            return next(candidates)

        def score_fn(x: float) -> Score:
            return Score(correct=True, value=x)

        lineage, report = asyncio.run(evolve(vary, score_fn, steps=5))
        assert [v.candidate for v in lineage.versions] == [1.0, 2.0, 2.0]
        assert report.committed == 3
        assert len(report.attempts) == 5

    def test_async_score_fn(self):
        async def score_fn(x: int) -> Score:
            return Score(correct=True, value=float(x))

        async def vary(lineage: Lineage) -> int:
            return len(lineage) + 1

        lineage, report = asyncio.run(evolve(vary, score_fn, steps=3))
        assert report.committed == 3

    def test_raising_score_fn_becomes_a_failed_score(self):
        async def vary(lineage: Lineage) -> str:
            return "candidate"

        def score_fn(x: str) -> Score:
            raise ValueError("kernel produced NaN at seq_len=32768")

        lineage, report = asyncio.run(evolve(vary, score_fn, steps=2))
        assert len(lineage) == 0
        assert all(not a.score.correct for a in report.attempts)
        assert "NaN" in report.attempts[0].score.notes

    def test_vary_sees_the_growing_lineage(self):
        seen: list[int] = []

        async def vary(lineage: Lineage) -> int:
            seen.append(len(lineage))
            return len(lineage) + 1

        def score_fn(x: int) -> Score:
            return Score(correct=True, value=float(x))

        _ = asyncio.run(evolve(vary, score_fn, steps=3))
        assert seen == [0, 1, 2]

    def test_stall_without_intervention_stops_early(self):
        async def vary(lineage: Lineage) -> float:
            return 0.0  # never improves after the seed

        def score_fn(x: float) -> Score:
            return Score(correct=False, notes="always fails")

        _, report = asyncio.run(evolve(vary, score_fn, steps=100, stall_after=3))
        assert len(report.attempts) == 3

    def test_stall_intervention_is_called_and_loop_continues(self):
        interventions: list[int] = []
        values = iter([1.0, 0.0, 0.0, 0.0, 2.0, 3.0])

        async def vary(lineage: Lineage) -> float:
            return next(values)

        def score_fn(x: float) -> Score:
            return Score(correct=x > 0, value=x)

        async def on_stall(lineage: Lineage, report: EvolutionReport) -> None:
            interventions.append(len(report.attempts))

        lineage, report = asyncio.run(evolve(vary, score_fn, steps=6, stall_after=3, on_stall=on_stall))
        assert interventions == [4]  # fired once, after the 3 consecutive rejections
        assert len(report.attempts) == 6  # the loop kept going after the intervention
        assert lineage.best is not None and lineage.best.score.value == 3.0

    def test_report_consecutive_rejections(self):
        report = EvolutionReport()
        assert report.consecutive_rejections == 0
