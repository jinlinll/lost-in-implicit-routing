#!/usr/bin/env python3
"""Preselection-only simulation for Multi-query RAG-LLM selector prompts.

This script does NOT run the agent. It reuses/generated multi-query RAG
candidate pools, calls only the final LLM skill-selection step, and logs full
LLM inputs/outputs for prompt variants:
  1. no_limit: ask for skills that will be used; no explicit max count.
  2. max10: ask for at most 10 skills.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from multi_query_rag import MultiQueryCandidateAggregator
from simulate_multi_query_rag import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_EMBEDDING_API_BASE,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_MODEL,
    build_skills_list,
    cosine,
    discover_golden,
    discover_tasks,
    embed_texts,
    load_embedding_cache,
    load_instruction,
    parse_json_array,
    text_hash,
)

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))


def avg(values) -> float:
    """Mean of an iterable; returns 0 if empty."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
try:
    from analyze_study_metrics import ALWAYS_EXCLUDED
except Exception:
    ALWAYS_EXCLUDED = set()

PROMPTS = {
    "no_limit": '''You are a skill-routing assistant. Given a task description and candidate skills retrieved by semantic search, select the skills that will be used to complete the task.

Return your answer as a JSON array of skill names. Select every candidate skill that is concretely useful for the task, but do not include irrelevant skills. If none of the candidates is relevant, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no explanation, no markdown code fences.

## Candidate Skills Retrieved by RAG

{skills_list}

## Task Description

{task_instruction}

## Your Selection (JSON array of skill names)
''',
    "max10": '''You are a skill-routing assistant. Given a task description and candidate skills retrieved by semantic search, select the skills that will be used to complete the task.

Return your answer as a JSON array of skill names. Return at most 10 skills. Use fewer when fewer skills are needed, but include all clearly useful skills up to the budget. If none of the candidates is relevant, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no explanation, no markdown code fences.

## Candidate Skills Retrieved by RAG

{skills_list}

## Task Description

{task_instruction}

## Your Selection (JSON array of skill names)
''',
}


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("simulate_multi_query_selection_variants")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def load_query_cache(path: Path, tasks: List[str]) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"query cache not found: {path}; run simulate_multi_query_rag.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [t for t in tasks if t not in data]
    if missing:
        raise RuntimeError(f"query cache missing {len(missing)} tasks, examples={missing[:5]}")
    return data


def build_candidates(task: str, sc: int, args, cache: Dict[str, Any], queries: List[str]):
    skills = build_skills_list(task, sc, args.tasks_dir, args.skills_pool_dir, args.seed)
    skill_vectors = []
    for s in skills:
        v = cache.get(text_hash(f"{s.name.strip()}: {s.description.strip()}"))
        if v is not None:
            skill_vectors.append((s, v))
    query_rankings = []
    for q in queries:
        qv = cache.get(text_hash(q))
        if qv is None:
            continue
        scored = [(s, cosine(qv, sv)) for s, sv in skill_vectors]
        scored.sort(key=lambda x: x[1], reverse=True)
        query_rankings.append(scored)
    aggregated = MultiQueryCandidateAggregator.aggregate(query_rankings, args.candidate_budget, strategy=args.aggregation_strategy)
    return [c.item for c in aggregated], aggregated, len(skills), len(skill_vectors)


def call_selector(task: str, sc: int, model: str, variant: str, instruction: str, candidates, args, output_dir: Path) -> dict:
    skills_list = "\n".join(f"- **{s.name}**: {s.description}" for s in candidates)
    prompt = PROMPTS[variant].format(skills_list=skills_list, task_instruction=instruction.strip())
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    log_dir = output_dir / "llm_io" / variant / model / safe_name(task)
    log_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = args.session.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(30 * attempt)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
            selected_raw = [x.strip() for x in parse_json_array(content) if x.strip()]
            record = {
                "task": task,
                "model": model,
                "variant": variant,
                "attempt": attempt,
                "payload": payload,
                "response_text": content,
                "usage": data.get("usage", {}),
                "selected_raw": selected_raw,
            }
            (log_dir / f"skills_{sc}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return record
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(30)
            else:
                error_record = {
                    "task": task,
                    "model": model,
                    "variant": variant,
                    "payload": payload,
                    "error": str(e),
                }
                (log_dir / f"skills_{sc}.error.json").write_text(
                    json.dumps(error_record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                raise
    raise last_error or RuntimeError("selector failed")


def map_selected(selected_raw: List[str], candidates, max_output: int | None) -> List[str]:
    name_to_skill = {s.name: s for s in candidates}
    selected = []
    seen = set()
    for name in selected_raw:
        skill = name_to_skill.get(name)
        if skill is None:
            for cand_name, cand in name_to_skill.items():
                if cand_name.lower() == name.lower():
                    skill = cand
                    break
        if skill is None or skill.name in seen:
            continue
        selected.append(skill.name)
        seen.add(skill.name)
        if max_output is not None and len(selected) >= max_output:
            break
    return selected


def evaluate(selected: List[str], golden: Set[str]) -> Dict[str, Any]:
    sel = set(selected)
    hits = sel & golden
    return {
        "selected_count": len(selected),
        "hits": ";".join(sorted(hits)),
        "misses": ";".join(sorted(golden - hits)),
        "recall": len(hits) / len(golden) * 100 if golden else 0.0,
        "precision": len(hits) / len(sel) * 100 if sel else 100.0,
        "any_hit": 100.0 if hits else 0.0,
        "exact": 100.0 if sel == golden else 0.0,
    }


def main() -> None:
    import requests

    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", type=Path, default=Path("../skillsbench/tasks"))
    p.add_argument("--skills-pool-dir", type=Path, default=Path(".agents/skills"))
    p.add_argument("--rag-dir", type=Path, default=Path("rag_precomputed"))
    p.add_argument("--query-cache", type=Path, default=Path("multi_query_rag_sim/queries.json"))
    p.add_argument("--output-dir", type=Path, default=Path("multi_query_selection_variants"))
    p.add_argument("--skills-counts", type=int, nargs="+", default=[1000, 10000])
    p.add_argument("--models", nargs="+", default=["glm-5", "kimi-k2.5", "minimax-m2.5"])
    p.add_argument("--variants", nargs="+", choices=sorted(PROMPTS), default=["no_limit", "max10"])
    p.add_argument("--candidate-budget", type=int, default=100)
    p.add_argument("--aggregation-strategy", default="quota")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--api-base", default=os.environ.get("API_BASE", DEFAULT_API_BASE))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", DEFAULT_API_KEY))
    p.add_argument("--embedding-api-base", default=os.environ.get("EMBEDDING_API_BASE", os.environ.get("DASHSCOPE_API_BASE", DEFAULT_EMBEDDING_API_BASE)))
    p.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY", os.environ.get("DASHSCOPE_API_KEY", DEFAULT_EMBEDDING_API_KEY)))
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = p.parse_args()
    if not args.api_key:
        raise RuntimeError("API_KEY environment variable or --api-key is required")
    args.session = requests.Session()

    logger = setup_logger(args.output_dir)
    tasks = [t for t in discover_tasks(args.tasks_dir) if t not in ALWAYS_EXCLUDED]
    if args.task_limit:
        tasks = tasks[: args.task_limit]
    golden = {k: v for k, v in discover_golden(args.tasks_dir).items() if k in tasks}
    query_cache = load_query_cache(args.query_cache, tasks)
    cache = load_embedding_cache(args.rag_dir)
    logger.info(f"tasks={len(tasks)} models={args.models} skills_counts={args.skills_counts} variants={args.variants}")

    # Embed any queries that are missing from the precomputed cache
    all_queries_to_embed = []
    for task in tasks:
        for q in query_cache[task]:
            h = text_hash(q)
            if h not in cache:
                all_queries_to_embed.append(q)
    # Deduplicate
    seen_hashes = set()
    unique_queries = []
    for q in all_queries_to_embed:
        h = text_hash(q)
        if h not in seen_hashes:
            unique_queries.append(q)
            seen_hashes.add(h)
    if unique_queries:
        logger.info(f"Embedding {len(unique_queries)} missing query vectors online...")
        vecs = embed_texts(unique_queries, args, logger)
        for q, v in zip(unique_queries, vecs):
            cache[text_hash(q)] = v
        logger.info(f"Embedded {len(unique_queries)} queries; cache now has {len(cache)} entries")

    rows = []
    row_path = args.output_dir / "per_selection.csv"
    tasks_work = []
    for task in tasks:
        if task not in golden:
            continue
        instruction = load_instruction(args.tasks_dir, task)
        for sc in args.skills_counts:
            candidates, aggregated, num_skills, num_vec = build_candidates(task, sc, args, cache, query_cache[task])
            candidate_names = [c.name for c in candidates]
            cand_info = {
                "task": task,
                "skills_count": sc,
                "num_skills": num_skills,
                "num_skill_vectors": num_vec,
                "num_candidates": len(candidates),
                "candidate_names": candidate_names,
                "aggregated": [(a.item.name, float(a.score), a.best_rank, a.query_hits) for a in aggregated],
                "queries": query_cache[task],
                "golden": sorted(golden[task]),
            }
            cand_dir = args.output_dir / "candidates" / safe_name(task)
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / f"skills_{sc}.json").write_text(json.dumps(cand_info, indent=2, ensure_ascii=False), encoding="utf-8")
            for model in args.models:
                for variant in args.variants:
                    tasks_work.append((task, instruction, sc, candidates, model, variant))

    def run_one(item):
        task, instruction, sc, candidates, model, variant = item
        rec = call_selector(task, sc, model, variant, instruction, candidates, args, args.output_dir)
        max_output = 10 if variant == "max10" else None
        selected = map_selected(rec["selected_raw"], candidates, max_output=max_output)
        ev = evaluate(selected, golden[task])
        usage = rec.get("usage", {})
        return {
            "task": task,
            "skills_count": sc,
            "model": model,
            "variant": variant,
            "num_golden": len(golden[task]),
            "num_candidates": len(candidates),
            "selected": ";".join(selected),
            "raw_selected": ";".join(rec.get("selected_raw", [])),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            **ev,
        }

    with ThreadPoolExecutor(max_workers=min(4, max(1, args.workers))) as ex:
        futs = [ex.submit(run_one, item) for item in tasks_work]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                rows.append(row)
                logger.info(
                    f"{i}/{len(futs)} {row['task']} s={row['skills_count']} {row['model']} {row['variant']} "
                    f"selected={row['selected_count']} recall={row['recall']:.1f}"
                )
            except Exception as e:
                logger.error(f"selection failed: {e}")
            if rows:
                with row_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)

    summary = {}
    for variant in args.variants:
        summary[variant] = {}
        for sc in args.skills_counts:
            sub = [r for r in rows if r["variant"] == variant and int(r["skills_count"]) == sc]
            summary[variant][str(sc)] = {
                "n": len(sub),
                "avg_selected": avg(float(r["selected_count"]) for r in sub),
                "recall": avg(float(r["recall"]) for r in sub),
                "precision": avg(float(r["precision"]) for r in sub),
                "any_hit": avg(float(r["any_hit"]) for r in sub),
                "exact": avg(float(r["exact"]) for r in sub),
                "prompt_tokens": avg(float(r["prompt_tokens"] or 0) for r in sub),
                "completion_tokens": avg(float(r["completion_tokens"] or 0) for r in sub),
            }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
