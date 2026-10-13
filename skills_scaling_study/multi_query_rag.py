#!/usr/bin/env python3
"""Shared utilities for multi-query RAG candidate aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class AggregatedCandidate:
    """A candidate selected from one or more query rankings."""

    item: Any
    score: float
    best_rank: int
    query_hits: int


class MultiQueryCandidateAggregator:
    """Aggregate per-query scored skill rankings into one candidate list.

    Input format is a list of rankings, one per query.  Each ranking must be
    sorted from best to worst and contain ``(item, score)`` pairs.  ``item`` is
    expected to expose a stable ``name`` attribute; dicts with ``name`` are also
    supported.
    """

    STRATEGIES = {"max", "mean", "quota"}

    @classmethod
    def aggregate(
        cls,
        query_rankings: Sequence[Sequence[Tuple[Any, float]]],
        budget: int,
        strategy: str = "max",
        per_query_quota: int | None = None,
    ) -> List[AggregatedCandidate]:
        strategy = strategy.lower().strip()
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Unknown aggregation strategy: {strategy}. Choose from {sorted(cls.STRATEGIES)}")
        if budget <= 0 or not query_rankings:
            return []
        if strategy == "max":
            return cls._aggregate_score(query_rankings, budget, mode="max")
        if strategy == "mean":
            return cls._aggregate_score(query_rankings, budget, mode="mean")
        return cls._aggregate_quota(query_rankings, budget, per_query_quota=per_query_quota)

    @staticmethod
    def _name(item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("name", ""))
        return str(getattr(item, "name", ""))

    @classmethod
    def _aggregate_score(
        cls,
        query_rankings: Sequence[Sequence[Tuple[Any, float]]],
        budget: int,
        mode: str,
    ) -> List[AggregatedCandidate]:
        items: Dict[str, Any] = {}
        scores: Dict[str, List[float]] = {}
        ranks: Dict[str, List[int]] = {}

        for ranking in query_rankings:
            for rank, (item, score) in enumerate(ranking, 1):
                name = cls._name(item)
                if not name:
                    continue
                items.setdefault(name, item)
                scores.setdefault(name, []).append(float(score))
                ranks.setdefault(name, []).append(rank)

        candidates: List[AggregatedCandidate] = []
        for name, item in items.items():
            vals = scores[name]
            score = max(vals) if mode == "max" else sum(vals) / len(vals)
            candidates.append(
                AggregatedCandidate(
                    item=item,
                    score=score,
                    best_rank=min(ranks[name]),
                    query_hits=len(vals),
                )
            )

        candidates.sort(key=lambda x: (-x.score, x.best_rank, -x.query_hits, cls._name(x.item)))
        return candidates[:budget]

    @classmethod
    def _aggregate_quota(
        cls,
        query_rankings: Sequence[Sequence[Tuple[Any, float]]],
        budget: int,
        per_query_quota: int | None = None,
    ) -> List[AggregatedCandidate]:
        n_queries = len(query_rankings)
        quota = per_query_quota or max(1, math.ceil(budget / n_queries))
        selected: Dict[str, AggregatedCandidate] = {}

        for ranking in query_rankings:
            taken = 0
            for rank, (item, score) in enumerate(ranking, 1):
                name = cls._name(item)
                if not name or name in selected:
                    continue
                selected[name] = AggregatedCandidate(
                    item=item,
                    score=float(score),
                    best_rank=rank,
                    query_hits=1,
                )
                taken += 1
                if taken >= quota or len(selected) >= budget:
                    break
            if len(selected) >= budget:
                break

        if len(selected) < budget:
            # Fill any duplicate-induced holes by global max-score order.
            for cand in cls._aggregate_score(query_rankings, budget=len(query_rankings) * max(len(r) for r in query_rankings), mode="max"):
                name = cls._name(cand.item)
                if name not in selected:
                    selected[name] = cand
                if len(selected) >= budget:
                    break

        out = list(selected.values())
        out.sort(key=lambda x: (-x.score, x.best_rank, cls._name(x.item)))
        return out[:budget]


__all__ = ["AggregatedCandidate", "MultiQueryCandidateAggregator"]
