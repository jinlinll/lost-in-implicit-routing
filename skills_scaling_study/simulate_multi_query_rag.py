#!/usr/bin/env python3
"""Simulate multi-query RAG candidate aggregation strategies.

For each task, an LLM generates multiple retrieval queries.  Each query is
embedded and used to rank candidate skills.  We compare aggregation strategies
(max, mean, quota) by measuring how well the final top-N candidate pool covers
known golden skills.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from pool_order import ordered_pool_paths
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

from multi_query_rag import MultiQueryCandidateAggregator
from tokenplan_client import get_default_tokenplan_api_base, get_default_tokenplan_api_key

DEFAULT_API_BASE = get_default_tokenplan_api_base()
DEFAULT_API_KEY = get_default_tokenplan_api_key()
DEFAULT_EMBEDDING_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_API_KEY = os.environ.get("QWEN_EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"

QUERY_PROMPT = '''You are a retrieval query planner for a skill-selection system.
Given a task description, write {num_queries} short search queries that cover distinct capabilities likely needed to solve the task.

Guidelines:
- Each query should be concise, concrete, and skill-oriented.
- Prefer capability/library/workflow phrases over restating the whole task.
- Return ONLY a JSON array of strings, no explanation, no markdown.

## Task Description

{task_instruction}

## Search Queries (JSON array)
'''


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    location: str
    source: str


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("simulate_multi_query_rag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "run.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def parse_frontmatter(skill_md: Path) -> Optional[Dict[str, str]]:
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = content.strip().split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end == -1:
        return None
    result: Dict[str, str] = {}
    current_key = None
    for line in lines[1:end]:
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and value != "|":
                result[key] = value
                current_key = key
            else:
                result[key] = ""
                current_key = key
        elif current_key and (line.startswith("  ") or line.startswith("\t")):
            result[current_key] += " " + line.strip()
    if "name" not in result or "description" not in result:
        return None
    return {"name": result["name"].strip(), "description": result["description"].strip()}


def discover_tasks(tasks_dir: Path) -> List[str]:
    return sorted(d.name for d in tasks_dir.iterdir() if d.is_dir() and (d / "instruction.md").exists())


def load_instruction(tasks_dir: Path, task: str) -> str:
    return (tasks_dir / task / "instruction.md").read_text(encoding="utf-8", errors="replace")


def discover_golden(tasks_dir: Path) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for task in discover_tasks(tasks_dir):
        skills_dir = tasks_dir / task / "environment" / "skills"
        if not skills_dir.exists():
            continue
        names = set()
        for d in skills_dir.iterdir():
            if not d.is_dir():
                continue
            skill_md = d / "SKILL.md"
            if not skill_md.exists():
                continue
            fm = parse_frontmatter(skill_md)
            names.add(fm["name"] if fm else d.name)
        if names:
            out[task] = names
    return out


def build_skills_list(task: str, extra_count: int, tasks_dir: Path, skills_pool_dir: Path, seed: int) -> List[SkillEntry]:
    task_skills_dir = tasks_dir / task / "environment" / "skills"
    skills: List[SkillEntry] = []
    task_skill_names: List[str] = []
    if task_skills_dir.exists():
        task_skill_names = [p.name for p in task_skills_dir.iterdir() if p.is_dir()]
        for item in sorted(task_skills_dir.iterdir()):
            if item.is_dir() and (item / "SKILL.md").exists():
                fm = parse_frontmatter(item / "SKILL.md")
                if fm:
                    skills.append(SkillEntry(fm["name"], fm["description"], f"/root/.terminus/skills/{item.name}/SKILL.md", "task"))
    if extra_count > 0:
        available = [p for p in ordered_pool_paths(skills_pool_dir) if p.is_dir() and (p / "SKILL.md").exists()]
        pool = [p for p in available if p.name not in task_skill_names]
        random.seed(seed)
        sampled = random.sample(pool, min(extra_count, len(pool)))
        for p in sampled:
            fm = parse_frontmatter(p / "SKILL.md")
            if fm:
                skills.append(SkillEntry(fm["name"], fm["description"], f"/root/.terminus/skills/{p.name}/SKILL.md", "extra"))
    return skills


def load_embedding_cache(rag_dir: Path) -> Dict[str, Any]:
    index = json.loads((rag_dir / "cache" / "embedding_index.json").read_text())
    data = np.load(rag_dir / "cache" / "embedding_cache.npz")
    return {h: data[k] for h, k in index.items() if k in data}


def parse_json_array(content: str) -> List[str]:
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(line for line in content.splitlines() if not line.strip().startswith("```"))
    try:
        x = json.loads(content)
        if isinstance(x, list):
            return [str(v) for v in x]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", content, re.S)
    if m:
        try:
            x = json.loads(m.group(0))
            if isinstance(x, list):
                return [str(v) for v in x]
        except json.JSONDecodeError:
            pass
    return re.findall(r'"([^"]+)"', content)


def call_llm_queries(task: str, instruction: str, args, logger: logging.Logger) -> List[str]:
    url = f"{args.api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    payload = {
        "model": args.query_model,
        "messages": [{"role": "user", "content": QUERY_PROMPT.format(num_queries=args.num_queries, task_instruction=instruction.strip())}],
        "temperature": 0.0,
    }
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 30 * attempt
                logger.warning(f"[{task}] query LLM HTTP {resp.status_code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
            queries = [q.strip() for q in parse_json_array(content) if q.strip()]
            return queries[: args.num_queries] or [instruction[:500]]
        except Exception as e:
            if attempt == 3:
                raise
            logger.warning(f"[{task}] query LLM error attempt {attempt}: {e}")
            time.sleep(30)
    return [instruction[:500]]


def embed_texts(texts: List[str], args, logger: logging.Logger) -> List[np.ndarray]:
    url = f"{args.embedding_api_base.rstrip('/')}/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.embedding_api_key}"}
    out: List[np.ndarray] = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i+10]
        payload = {"model": args.embedding_model, "input": batch}
        for attempt in range(1, 4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 10 * attempt
                    logger.warning(f"embedding HTTP {resp.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                items = sorted(resp.json()["data"], key=lambda x: x.get("index", 0))
                out.extend(np.array(x["embedding"], dtype=np.float32) for x in items)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(10)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", type=Path, default=Path("../skillsbench/tasks"))
    p.add_argument("--skills-pool-dir", type=Path, default=Path(".agents/skills"))
    p.add_argument("--rag-dir", type=Path, default=Path("rag_precomputed"))
    p.add_argument("--output-dir", type=Path, default=Path("multi_query_rag_sim"))
    p.add_argument("--skills-counts", type=int, nargs="+", default=[0, 10, 100, 1000, 10000])
    p.add_argument("--candidate-budget", type=int, default=100)
    p.add_argument("--num-queries", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-model", default="glm-5")
    p.add_argument("--api-base", default=os.environ.get("API_BASE", DEFAULT_API_BASE))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", DEFAULT_API_KEY))
    p.add_argument("--embedding-api-base", default=os.environ.get("EMBEDDING_API_BASE", os.environ.get("DASHSCOPE_API_BASE", DEFAULT_EMBEDDING_API_BASE)))
    p.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY", os.environ.get("DASHSCOPE_API_KEY", DEFAULT_EMBEDDING_API_KEY)))
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--query-workers", type=int, default=4)
    args = p.parse_args()

    logger = setup_logger(args.output_dir)
    logger.info(f"Starting simulation: output={args.output_dir}, budget={args.candidate_budget}, queries={args.num_queries}")
    tasks = discover_tasks(args.tasks_dir)
    if args.task_limit:
        tasks = tasks[: args.task_limit]
    golden = discover_golden(args.tasks_dir)
    cache = load_embedding_cache(args.rag_dir)
    logger.info(f"Loaded {len(cache)} cached skill embeddings; tasks={len(tasks)}")

    queries_path = args.output_dir / "queries.json"
    query_cache = json.loads(queries_path.read_text()) if queries_path.exists() else {}
    missing_query_tasks = [t for t in tasks if t not in query_cache]
    if missing_query_tasks:
        logger.info(f"Generating queries for {len(missing_query_tasks)} tasks with {args.query_workers} workers")
        with ThreadPoolExecutor(max_workers=min(4, max(1, args.query_workers))) as ex:
            futures = {
                ex.submit(call_llm_queries, task, load_instruction(args.tasks_dir, task), args, logger): task
                for task in missing_query_tasks
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                task = futures[fut]
                try:
                    query_cache[task] = fut.result()
                except Exception as e:
                    logger.error(f"[{task}] query generation failed: {e}; using task instruction fallback")
                    query_cache[task] = [load_instruction(args.tasks_dir, task)[:500]]
                logger.info(f"Generated queries {idx}/{len(missing_query_tasks)}: {task} -> {query_cache[task]}")
                queries_path.write_text(json.dumps(query_cache, indent=2, ensure_ascii=False), encoding="utf-8")

    all_queries = []
    seen_query_hashes = set()
    for task in tasks:
        for q in query_cache[task]:
            h = text_hash(q)
            if h not in cache and h not in seen_query_hashes:
                all_queries.append(q)
                seen_query_hashes.add(h)
    if all_queries:
        logger.info(f"Embedding {len(all_queries)} generated queries")
        vecs = embed_texts(all_queries, args, logger)
        for q, v in zip(all_queries, vecs):
            cache[text_hash(q)] = v

    row_path = args.output_dir / "per_combo_metrics.csv"
    rows = []
    strategies = ["max", "mean", "quota"]
    for task in tasks:
        if task not in golden:
            continue
        for sc in args.skills_counts:
            skills = build_skills_list(task, sc, args.tasks_dir, args.skills_pool_dir, args.seed)
            skill_vectors = []
            missing_vectors = 0
            for s in skills:
                v = cache.get(text_hash(f"{s.name.strip()}: {s.description.strip()}"))
                if v is None:
                    missing_vectors += 1
                    continue
                skill_vectors.append((s, v))
            if missing_vectors:
                logger.warning(f"{task}/s={sc}: missing skill vectors={missing_vectors}")
            query_rankings = []
            for q in query_cache[task]:
                qv = cache.get(text_hash(q))
                if qv is None:
                    continue
                scored = [(s, cosine(qv, sv)) for s, sv in skill_vectors]
                scored.sort(key=lambda x: x[1], reverse=True)
                query_rankings.append(scored)
            for strategy in strategies:
                cands = MultiQueryCandidateAggregator.aggregate(query_rankings, args.candidate_budget, strategy=strategy)
                selected = {c.item.name for c in cands}
                g = golden[task]
                hits = selected & g
                indiv = len(hits) / len(g) * 100 if g else 0.0
                all_cov = 100.0 if g and hits == g else 0.0
                any_cov = 100.0 if hits else 0.0
                rows.append({
                    "task": task,
                    "skills_count": sc,
                    "strategy": strategy,
                    "num_golden": len(g),
                    "num_skills": len(skills),
                    "candidate_budget": args.candidate_budget,
                    "num_candidates": len(cands),
                    "individual_recall": indiv,
                    "all_coverage": all_cov,
                    "any_coverage": any_cov,
                    "hits": ";".join(sorted(hits)),
                    "misses": ";".join(sorted(g - hits)),
                    "queries": " || ".join(query_cache[task]),
                })

    with row_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = {}
    for sc in args.skills_counts:
        summary[str(sc)] = {}
        for strategy in strategies:
            sub = [r for r in rows if r["skills_count"] == sc and r["strategy"] == strategy]
            summary[str(sc)][strategy] = {
                "n": len(sub),
                "individual_recall": sum(r["individual_recall"] for r in sub) / len(sub) if sub else 0,
                "all_coverage": sum(r["all_coverage"] for r in sub) / len(sub) if sub else 0,
                "any_coverage": sum(r["any_coverage"] for r in sub) / len(sub) if sub else 0,
            }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Summary:")
    for sc in args.skills_counts:
        for strategy in strategies:
            s = summary[str(sc)][strategy]
            logger.info(f"s={sc:<5} {strategy:<5} indiv={s['individual_recall']:.1f}% all={s['all_coverage']:.1f}% any={s['any_coverage']:.1f}%")
    best = max(
        ((strategy, sum(summary[str(sc)][strategy]["individual_recall"] for sc in args.skills_counts) / len(args.skills_counts)) for strategy in strategies),
        key=lambda x: x[1],
    )
    logger.info(f"Best average individual recall strategy: {best[0]} ({best[1]:.2f}%)")


if __name__ == "__main__":
    main()
