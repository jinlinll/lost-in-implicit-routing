#!/usr/bin/env python3
"""
Pre-compute RAG (embedding-based) skill matching results.

Since we use a fixed random seed (42), for each (task_name, extra_skills_count)
combination the list of candidate skills is deterministic and identical across
all models and passes.  Therefore we only need to compute:

    87 tasks × 5 skills configs (0, 10, 100, 1000, 10000) = 435 RAG queries

For each combination we:
  1. Build the full skills list (task-own skills + sampled extra skills)
  2. Compute embeddings for the task instruction + all skill descriptions
  3. Rank by cosine similarity and record the top-k results

Embeddings are cached per unique text so that identical skills appearing in
multiple combinations are only embedded once.

Incremental / resumable usage
-----------------------------
The script is **safe to run incrementally**:

  - Per-combination results are written to ``{output}/{task}/skills_{N}.json``.
    If such a file already exists and ``--force`` is not given, it is SKIPPED.
  - ``{output}/index.json`` and ``{output}/task_embeddings.npz`` are MERGED
    with any existing contents, so running a new skills_count later does not
    wipe out earlier results.

Typical workflows::

    # First-time: compute the baseline skills_counts
    python precompute_rag.py --skills-counts 0 10 100 1000

    # Later, add a larger-scale combination without redoing the earlier ones
    python precompute_rag.py --skills-counts 10000

    # Force recompute a specific combination (e.g., after changing prompts)
    python precompute_rag.py --skills-counts 1000 --task-filter 3d-scan-calc --force

Usage:
    python precompute_rag.py [--top-k 5] [--seed 42] [--output-dir rag_precomputed]
                             [--skills-counts 0 10 100 1000 10000]
                             [--force] [--task-filter TASK ...]
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from pool_order import ordered_pool_paths
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_API_KEY = os.environ.get("QWEN_EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TOP_K = 5
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "rag_precomputed"
DEFAULT_SKILLS_COUNTS = [0, 10, 100, 1000, 10000]
EMBEDDING_DIMENSION = 1024  # text-embedding-v4 default dimension
BATCH_SIZE = 10  # Max texts per embedding API call (dashscope limit is 10)


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("precompute_rag")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# =============================================================================
# Skill Frontmatter Parser (replicates standalone_agent.SkillLoader logic)
# =============================================================================

def parse_skill_frontmatter(skill_md_path: Path) -> Optional[Dict[str, str]]:
    """Parse YAML frontmatter from SKILL.md, return {name, description} or None."""
    try:
        content = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    lines = content.strip().split("\n")
    if not lines or lines[0].strip() != "---":
        return None

    # Find closing ---
    end_idx = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return None

    result = {}
    current_key = None
    for line in lines[1:end_idx]:
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and value != "|":
                result[key] = value
                current_key = key
            else:
                current_key = key
                result[key] = ""
        elif current_key and (line.startswith("  ") or line.startswith("\t")):
            # Continuation of multi-line value
            result[current_key] += " " + line.strip()

    if "name" not in result or "description" not in result:
        return None

    return {"name": result["name"].strip(), "description": result["description"].strip()}


# =============================================================================
# Embedding Cache
# =============================================================================

class EmbeddingCache:
    """Persistent embedding cache backed by .npz files.
    
    Each unique text is hashed (SHA-256) to create a cache key.
    Embeddings are stored in a single .npz file for efficiency.
    """

    def __init__(self, cache_dir: Path, dimension: int = EMBEDDING_DIMENSION):
        self.cache_dir = cache_dir
        self.dimension = dimension
        self.cache_file = cache_dir / "embedding_cache.npz"
        self.index_file = cache_dir / "embedding_index.json"
        self._vectors: Dict[str, np.ndarray] = {}  # hash -> vector
        self._load()

    def _load(self):
        """Load existing cache from disk."""
        if self.index_file.exists() and self.cache_file.exists():
            try:
                with open(self.index_file) as f:
                    index = json.load(f)  # {hash: array_key}
                data = np.load(self.cache_file)
                for text_hash, array_key in index.items():
                    if array_key in data:
                        self._vectors[text_hash] = data[array_key]
                logger.info(f"[Cache] Loaded {len(self._vectors)} cached embeddings")
            except Exception as e:
                logger.warning(f"[Cache] Failed to load cache: {e}, starting fresh")
                self._vectors = {}

    def save(self):
        """Persist cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self._vectors:
            return
        # Build numpy arrays and index
        index = {}
        arrays = {}
        for i, (text_hash, vec) in enumerate(self._vectors.items()):
            key = f"v_{i}"
            arrays[key] = vec
            index[text_hash] = key

        np.savez_compressed(self.cache_file, **arrays)
        with open(self.index_file, "w") as f:
            json.dump(index, f)
        logger.info(f"[Cache] Saved {len(self._vectors)} embeddings to disk")

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> Optional[np.ndarray]:
        h = self.text_hash(text)
        return self._vectors.get(h)

    def put(self, text: str, vector: np.ndarray):
        h = self.text_hash(text)
        self._vectors[h] = vector

    def has(self, text: str) -> bool:
        return self.text_hash(text) in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)


# =============================================================================
# Embedding Client
# =============================================================================

class EmbeddingClient:
    """Calls Qwen text-embedding-v4 via OpenAI-compatible API with caching."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        cache: EmbeddingCache,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.cache = cache
        self.total_api_calls = 0
        self.total_texts_embedded = 0

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for texts, using cache where possible.
        
        Texts already in cache are returned directly; only uncached texts
        are sent to the API.
        """
        results = [None] * len(texts)
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            logger.debug(f"  All {len(texts)} texts found in cache")
            return results

        logger.info(f"  Embedding {len(uncached_indices)} new texts "
                     f"({len(texts) - len(uncached_indices)} cached)...")

        # Batch API calls for uncached texts
        uncached_texts = [texts[i] for i in uncached_indices]
        uncached_vectors = self._batch_embed(uncached_texts)

        # Store results and update cache
        for idx, vec in zip(uncached_indices, uncached_vectors):
            results[idx] = vec
            self.cache.put(texts[idx], vec)

        return results

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Call the embedding API in batches."""
        all_vectors = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start:batch_start + BATCH_SIZE]
            vectors = self._call_api(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def _call_api(
        self,
        texts: List[str],
        max_retries: int = 3,
    ) -> List[np.ndarray]:
        """Call the embedding API with retries."""
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                self.total_api_calls += 1
                self.total_texts_embedded += len(texts)

                # Sort by index (API may return out of order)
                items = sorted(response.data, key=lambda x: x.index)
                return [np.array(item.embedding, dtype=np.float32) for item in items]

            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * attempt
                    logger.warning(f"  API error (attempt {attempt}/{max_retries}): {e}, "
                                   f"retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Embedding API failed after {max_retries} attempts: {e}")

        return []  # unreachable


# =============================================================================
# Core Logic
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass
class SkillEntry:
    """A skill with its metadata."""
    name: str
    description: str
    location: str  # Container path
    source: str  # "task" or "extra"


def build_skills_list(
    task_name: str,
    extra_skills_count: int,
    tasks_dir: Path,
    skills_pool_dir: Path,
    seed: int,
    include_task_skills: bool = True,
) -> List[SkillEntry]:
    """
    Build the exact same skills list that parallel_runner would produce.
    
    This replicates SkillsManager.sample_skills + copy_skills_to_dir logic.
    """
    task_dir = tasks_dir / task_name
    task_skills_dir = task_dir / "environment" / "skills"

    skills: List[SkillEntry] = []
    task_skill_names: List[str] = []

    # 1. Task-own skills
    # NOTE: task_skill_names must match parallel_runner.py which only checks is_dir()
    if task_skills_dir.exists():
        task_skill_names = [p.name for p in task_skills_dir.iterdir() if p.is_dir()]

    if include_task_skills and task_skills_dir.exists():
        for item in sorted(task_skills_dir.iterdir()):
            if item.is_dir() and (item / "SKILL.md").exists():
                fm = parse_skill_frontmatter(item / "SKILL.md")
                if fm:
                    skills.append(SkillEntry(
                        name=fm["name"],
                        description=fm["description"],
                        location=f"/root/.terminus/skills/{item.name}/SKILL.md",
                        source="task",
                    ))

    # 2. Extra skills (sampled from pool)
    # NOTE: Must NOT sort here — parallel_runner.py uses unsorted iterdir() order.
    # Using sorted() would produce a different random.sample() result for the same seed.
    if extra_skills_count > 0 and skills_pool_dir.exists():
        available = [
            p for p in ordered_pool_paths(skills_pool_dir)
            if p.is_dir() and (p / "SKILL.md").exists()
        ]
        # Exclude task's own skills (using task_skill_names from is_dir() check above)
        pool = [p for p in available if p.name not in task_skill_names]

        if len(pool) < extra_skills_count:
            logger.warning(f"  Only {len(pool)} skills available, requested {extra_skills_count}")
            extra_skills_count = len(pool)

        random.seed(seed)
        sampled = random.sample(pool, extra_skills_count)

        for skill_path in sampled:
            fm = parse_skill_frontmatter(skill_path / "SKILL.md")
            if fm:
                skills.append(SkillEntry(
                    name=fm["name"],
                    description=fm["description"],
                    location=f"/root/.terminus/skills/{skill_path.name}/SKILL.md",
                    source="extra",
                ))

    return skills


def precompute_rag_for_combination(
    task_name: str,
    task_instruction: str,
    skills: List[SkillEntry],
    top_k: int,
    client: EmbeddingClient,
) -> Dict:
    """
    Compute RAG matching for a single (task, skills_config) combination.
    
    Returns a dict with the matching results.
    """
    if not skills:
        return {
            "task_name": task_name,
            "num_skills": 0,
            "top_k": top_k,
            "selected": [],
            "all_similarities": [],
        }

    if len(skills) <= top_k:
        # All skills fit, no need to rank
        return {
            "task_name": task_name,
            "num_skills": len(skills),
            "top_k": top_k,
            "selected": [
                {"name": s.name, "description": s.description, "location": s.location,
                 "source": s.source, "similarity": 1.0, "rank": i + 1}
                for i, s in enumerate(skills)
            ],
            "all_similarities": [
                {"name": s.name, "similarity": 1.0, "rank": i + 1}
                for i, s in enumerate(skills)
            ],
        }

    # Build texts for embedding
    skill_texts = [f"{s.name}: {s.description}" for s in skills]
    all_texts = [task_instruction] + skill_texts

    # Get embeddings (with caching)
    embeddings = client.get_embeddings(all_texts)

    task_emb = embeddings[0]
    skill_embs = embeddings[1:]

    # Compute similarities
    similarities = []
    for i, skill_emb in enumerate(skill_embs):
        sim = cosine_similarity(task_emb, skill_emb)
        similarities.append((sim, i))

    # Sort descending
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Build results
    all_sims = [
        {"name": skills[idx].name, "similarity": round(sim, 6), "rank": rank + 1}
        for rank, (sim, idx) in enumerate(similarities)
    ]

    selected = [
        {
            "name": skills[idx].name,
            "description": skills[idx].description,
            "location": skills[idx].location,
            "source": skills[idx].source,
            "similarity": round(sim, 6),
            "rank": rank + 1,
        }
        for rank, (sim, idx) in enumerate(similarities[:top_k])
    ]

    return {
        "task_name": task_name,
        "num_skills": len(skills),
        "top_k": top_k,
        "selected": selected,
        "all_similarities": all_sims,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute RAG skill matching for all task × skills_count combinations"
    )
    parser.add_argument("--tasks-dir", type=Path, default=Path("../skillsbench/tasks"),
                        help="Path to tasks directory")
    parser.add_argument("--skills-pool-dir", type=Path, default=Path(".agents/skills"),
                        help="Path to skills pool directory")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                        help="Output directory for RAG results")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of top skills to select")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for skills sampling")
    parser.add_argument("--skills-counts", type=int, nargs="+",
                        default=DEFAULT_SKILLS_COUNTS,
                        help="Extra skills counts to compute")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="DashScope API key")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE,
                        help="API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model name")
    parser.add_argument("--task-filter", type=str, nargs="*", default=None,
                        help="Only process these task names (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be computed without calling the API")
    parser.add_argument("--force", action="store_true",
                        help="Recompute combinations even if {task}/skills_{N}.json already exists. "
                             "Default behavior is to skip existing per-combination files, which makes "
                             "this script safe to re-run incrementally (e.g., add a new skills_count "
                             "later without redoing the previous ones).")

    args = parser.parse_args()

    # Discover tasks
    tasks_dir = args.tasks_dir
    if not tasks_dir.exists():
        logger.error(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    all_task_names = sorted([
        d.name for d in tasks_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    ])

    if args.task_filter:
        all_task_names = [t for t in all_task_names if t in args.task_filter]

    logger.info(f"Tasks: {len(all_task_names)}")
    logger.info(f"Skills counts: {args.skills_counts}")
    logger.info(f"Top-k: {args.top_k}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Total combinations: {len(all_task_names) * len(args.skills_counts)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model: {args.model}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would compute:")
        for task_name in all_task_names:
            for sc in args.skills_counts:
                skills = build_skills_list(
                    task_name, sc, tasks_dir, args.skills_pool_dir, args.seed
                )
                logger.info(f"  {task_name} × skills_{sc}: {len(skills)} skills")
        return

    # Setup output
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup embedding cache
    cache = EmbeddingCache(output_dir / "cache")

    # Setup embedding client
    client = EmbeddingClient(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        cache=cache,
    )

    # Pre-read all task instructions (cache in memory)
    task_instructions = {}
    for task_name in all_task_names:
        inst_file = tasks_dir / task_name / "instruction.md"
        task_instructions[task_name] = inst_file.read_text()

    # Compute RAG for all combinations
    total = len(all_task_names) * len(args.skills_counts)
    done = 0
    skipped = 0
    start_time = time.time()

    # ── Master index: pre-load any existing entries so we MERGE rather than OVERWRITE ──
    # This lets the script be invoked incrementally, e.g.:
    #   1st run: --skills-counts 0 10 100 1000
    #   2nd run: --skills-counts 10000        (adds to the same index.json)
    index_path = output_dir / "index.json"
    master_index: Dict[str, Dict] = {}
    prior_metadata: Dict = {}
    if index_path.exists():
        try:
            with open(index_path) as f:
                prior = json.load(f)
            master_index = dict(prior.get("combinations", {}))
            prior_metadata = prior.get("metadata", {}) or {}
            logger.info(
                f"[Index] Loaded {len(master_index)} existing combinations from "
                f"{index_path} (will merge with new results)"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[Index] Could not read existing {index_path}: {e}, starting fresh")

    for task_name in all_task_names:
        instruction = task_instructions[task_name]

        for skills_count in args.skills_counts:
            done += 1
            result_dir = output_dir / task_name
            result_file = result_dir / f"skills_{skills_count}.json"

            # Skip-existing: if the per-combination JSON already exists and the
            # user didn't pass --force, we skip it. The master_index entry for
            # this combination is already preserved from the prior-loaded index.
            if result_file.exists() and not args.force:
                skipped += 1
                logger.info(f"[{done}/{total}] {task_name} × skills_{skills_count}  "
                            f"{'(skipped — file exists, use --force to recompute)'}")
                # Ensure master_index has an entry even if the prior index file
                # was missing/corrupted but the per-combination JSON exists.
                idx_key = f"{task_name}/skills_{skills_count}"
                if idx_key not in master_index:
                    try:
                        with open(result_file) as rf:
                            existing = json.load(rf)
                        master_index[idx_key] = {
                            "file": str(result_file.relative_to(output_dir)),
                            "num_skills": existing.get("num_skills", 0),
                            "num_selected": len(existing.get("selected", [])),
                            "selected_names": [s["name"] for s in existing.get("selected", [])],
                        }
                    except (json.JSONDecodeError, OSError):
                        pass
                continue

            logger.info(f"[{done}/{total}] {task_name} × skills_{skills_count}")

            # Build skills list
            skills = build_skills_list(
                task_name=task_name,
                extra_skills_count=skills_count,
                tasks_dir=tasks_dir,
                skills_pool_dir=args.skills_pool_dir,
                seed=args.seed,
            )

            # Compute RAG matching
            result = precompute_rag_for_combination(
                task_name=task_name,
                task_instruction=instruction,
                skills=skills,
                top_k=args.top_k,
                client=client,
            )

            # Add metadata
            result["extra_skills_count"] = skills_count
            result["seed"] = args.seed
            result["embedding_model"] = args.model

            # Save individual result
            result_dir.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update master index (merge semantics: overwrite this key only)
            master_index[f"{task_name}/skills_{skills_count}"] = {
                "file": str(result_file.relative_to(output_dir)),
                "num_skills": result["num_skills"],
                "num_selected": len(result["selected"]),
                "selected_names": [s["name"] for s in result["selected"]],
            }

            # Periodic cache save (every 20 combinations)
            if done % 20 == 0:
                cache.save()
                elapsed = time.time() - start_time
                computed = done - skipped
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(f"  Progress: {done}/{total} ({done/total*100:.1f}%), "
                            f"computed={computed}, skipped={skipped}, "
                            f"ETA: {eta:.0f}s, "
                            f"API calls: {client.total_api_calls}, "
                            f"cached: {len(cache)}")

    # Final save
    cache.save()

    # ── Save master index (MERGED) ──
    # skills_counts in metadata reflects ALL counts present in master_index,
    # not just the ones passed on this invocation.
    all_skills_counts = sorted({
        int(k.split("/skills_", 1)[1]) for k in master_index if "/skills_" in k
    })
    all_tasks_in_index = sorted({
        k.split("/skills_", 1)[0] for k in master_index if "/skills_" in k
    })
    # Track cumulative API usage across invocations via the prior metadata.
    prior_api_calls = int(prior_metadata.get("api_calls_cumulative",
                                             prior_metadata.get("api_calls", 0)) or 0)
    prior_texts = int(prior_metadata.get("texts_embedded_cumulative",
                                         prior_metadata.get("texts_embedded", 0)) or 0)

    metadata = {
        "embedding_model": args.model,
        "top_k": args.top_k,
        "seed": args.seed,
        "skills_counts": all_skills_counts,
        "num_tasks": len(all_tasks_in_index),
        "total_combinations": len(master_index),
        # This run only
        "api_calls_last_run": client.total_api_calls,
        "texts_embedded_last_run": client.total_texts_embedded,
        "skills_counts_last_run": sorted(set(args.skills_counts)),
        "combinations_computed_last_run": done - skipped,
        "combinations_skipped_last_run": skipped,
        # Cumulative across all invocations
        "api_calls_cumulative": prior_api_calls + client.total_api_calls,
        "texts_embedded_cumulative": prior_texts + client.total_texts_embedded,
        "cache_size": len(cache),
    }
    with open(index_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "combinations": master_index,
        }, f, indent=2, ensure_ascii=False)

    # ── Save task embedding vectors (MERGED) ──
    # Load any existing per-task embeddings and overwrite/extend with fresh ones.
    task_emb_file = output_dir / "task_embeddings.npz"
    task_embeddings: Dict[str, np.ndarray] = {}
    if task_emb_file.exists():
        try:
            loaded = np.load(task_emb_file)
            for name in loaded.files:
                task_embeddings[name] = loaded[name]
            logger.info(f"[TaskEmb] Loaded {len(task_embeddings)} existing task embeddings")
        except Exception as e:
            logger.warning(f"[TaskEmb] Could not read existing {task_emb_file}: {e}")

    # Overwrite with fresh embeddings for tasks we processed this run
    for task_name in all_task_names:
        instruction = task_instructions[task_name]
        emb = cache.get(instruction)
        if emb is not None:
            task_embeddings[task_name] = emb
    if task_embeddings:
        np.savez_compressed(
            task_emb_file,
            **{name: vec for name, vec in task_embeddings.items()}
        )
        logger.info(f"Saved {len(task_embeddings)} task embeddings (total)")

    # Summary
    elapsed = time.time() - start_time
    computed = done - skipped
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f}s")
    logger.info(f"  This run: computed={computed}, skipped={skipped}")
    logger.info(f"  Index now covers: {len(master_index)} combinations "
                f"across skills_counts={all_skills_counts}")
    logger.info(f"  API calls this run: {client.total_api_calls}")
    logger.info(f"  Texts embedded this run: {client.total_texts_embedded}")
    logger.info(f"  Cache entries: {len(cache)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Master index: {index_path}")


if __name__ == "__main__":
    main()
