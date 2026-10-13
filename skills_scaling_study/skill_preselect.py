#!/usr/bin/env python3
"""
Skill Pre-selection Module for SkillsBench experiments.

Provides strategies to pre-select a subset of skills before injecting them
into the agent's system prompt. Dynamic RAG selects no skills up front and
lets the agent retrieve relevant skills during execution. This reduces prompt
size and potential noise.

Strategies:
1. naive          – Include ALL skills (the default / baseline behavior).
2. llm            – Ask an LLM to pick the most relevant skills for the task.
3. rag            – Use embedding-based semantic search to find relevant skills.
4. hybrid-rag-llm – Retrieve top candidates with RAG, then rerank with an LLM.

Usage:
    from skill_preselect import create_preselector

    preselector = create_preselector(
        strategy="llm",
        max_skills=5,
        model_name="gpt-4o",
        api_base="https://...",
        api_key="sk-...",
    )

    selected = preselector.select(task_instruction, all_skills_metadata)
    # selected is a list of SkillMetadata
"""

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import requests

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional speedup for RAG scoring
    np = None

from multi_query_rag import MultiQueryCandidateAggregator

DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_EMBEDDING_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_EMBEDDING_API_KEY = os.environ.get("QWEN_EMBEDDING_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")


def _normalize_api_base(url: str) -> str:
    """Strip trailing /chat/completions (and optional /) so callers can
    always build the endpoint as ``f"{base}/chat/completions"``."""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    return url


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class SkillInfo:
    """Lightweight skill descriptor for pre-selection.
    
    Compatible with standalone_agent.SkillMetadata but kept independent
    so that this module can be used standalone.
    """
    name: str
    description: str
    location: str  # path to SKILL.md inside the container


# =============================================================================
# Base Class
# =============================================================================

class SkillPreselector(ABC):
    """Abstract base for skill pre-selection strategies."""

    def __init__(
        self,
        max_skills: int = 5,
        logger: Optional[logging.Logger] = None,
        preselect_log_dir: Optional[Path] = None,
    ):
        self.max_skills = max_skills
        self.logger = logger or logging.getLogger("experiment.preselect")
        self.preselect_log_dir = Path(preselect_log_dir) if preselect_log_dir else None

    def _write_llm_io_log(
        self,
        stage: str,
        payload: Dict[str, Any],
        response_text: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist full preselection LLM input/output for debugging."""
        if not self.preselect_log_dir:
            return
        try:
            self.preselect_log_dir.mkdir(parents=True, exist_ok=True)
            safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "llm"
            path = self.preselect_log_dir / f"{int(time.time() * 1000)}_{safe_stage}.json"
            record = {
                "stage": stage,
                "model": payload.get("model"),
                "messages": payload.get("messages"),
                "temperature": payload.get("temperature"),
                "usage": usage or {},
                "response_text": response_text,
                "error": error,
            }
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"[Preselect] Failed to write LLM I/O log: {e}")

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Short identifier used in log paths, e.g. 'naive', 'llm', 'rag'."""
        ...

    @abstractmethod
    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        """Return the selected subset of skills.

        Args:
            task_instruction: The task description / instruction text.
            all_skills: All available skills.

        Returns:
            A list of selected SkillInfo (at most self.max_skills items).
        """
        ...


# =============================================================================
# Strategy 1: Naive (include all)
# =============================================================================

class NaivePreselector(SkillPreselector):
    """Include ALL available skills. This is the baseline / no-filtering strategy."""

    @property
    def strategy_name(self) -> str:
        return "naive"

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        self.logger.info(f"[Preselect/naive] Returning all {len(all_skills)} skills (no filtering)")
        return list(all_skills)


class EmptyPreselector(SkillPreselector):
    """Select no skills up front; skills may be discovered dynamically later."""

    @property
    def strategy_name(self) -> str:
        return "dynamic-rag"

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        self.logger.info(
            f"[Preselect/dynamic-rag] Selecting 0 upfront skills from {len(all_skills)} candidates"
        )
        return []


# =============================================================================
# Strategy 2: LLM-based pre-selection
# =============================================================================

LLM_PRESELECT_PROMPT = '''You are a skill-routing assistant. Given a task description and a list of available skills (each with a name and description), select the skills that are most likely to be useful for completing the task.

Return your answer as a JSON array of skill names. Return at most {max_skills} skills.
If no skill is relevant, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no explanation, no markdown code fences.

## Available Skills

{skills_list}

## Task Description

{task_instruction}

## Your Selection (JSON array of skill names)
'''


class LLMPreselector(SkillPreselector):
    """Use an LLM to select the most relevant skills for a task."""

    def __init__(
        self,
        max_skills: int = 5,
        model_name: str = "gpt-4o",
        api_base: str = "",
        api_key: str = "",
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        preselect_log_dir: Optional[Path] = None,
    ):
        super().__init__(max_skills=max_skills, logger=logger, preselect_log_dir=preselect_log_dir)
        self.model_name = model_name
        self.api_base = _normalize_api_base(api_base)
        self.api_key = api_key
        self.temperature = temperature

        # Track token usage across all pre-selection calls
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def strategy_name(self) -> str:
        return "llm"

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        if not all_skills:
            return []

        if len(all_skills) <= self.max_skills:
            self.logger.info(
                f"[Preselect/llm] Only {len(all_skills)} skills available "
                f"(<= max_skills={self.max_skills}), returning all"
            )
            return list(all_skills)

        # Build the skills list for the prompt
        skills_text = "\n".join(
            f"- **{s.name}**: {s.description}" for s in all_skills
        )

        prompt = LLM_PRESELECT_PROMPT.format(
            max_skills=self.max_skills,
            skills_list=skills_text,
            task_instruction=task_instruction.strip(),
        )

        self.logger.info(
            f"[Preselect/llm] Asking {self.model_name} to select up to "
            f"{self.max_skills} from {len(all_skills)} skills..."
        )

        # Call the LLM
        selected_names = self._call_llm(prompt)

        # Map names back to SkillInfo objects (preserve order from LLM)
        name_to_skill = {s.name: s for s in all_skills}
        selected = []
        for name in selected_names:
            name = name.strip()
            if name in name_to_skill:
                selected.append(name_to_skill[name])
            else:
                # Try fuzzy match (case-insensitive)
                for skill_name, skill in name_to_skill.items():
                    if skill_name.lower() == name.lower():
                        selected.append(skill)
                        break
                else:
                    self.logger.warning(
                        f"[Preselect/llm] LLM returned unknown skill name: '{name}'"
                    )

        # Enforce max_skills limit
        selected = selected[: self.max_skills]

        self.logger.info(
            f"[Preselect/llm] Selected {len(selected)} skills: "
            f"{[s.name for s in selected]}"
        )
        return selected

    def _call_llm(self, prompt: str, max_retries: int = 3) -> List[str]:
        """Call the LLM and parse the response into a list of skill names."""
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 30 * attempt
                    self.logger.warning(
                        f"[Preselect/llm] HTTP {resp.status_code} (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s..."
                    )
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                data = resp.json()

                # Track tokens
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)

                content = data["choices"][0]["message"]["content"]
                # Handle list-type content (some models return list of text chunks)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = "".join(text_parts)

                self._write_llm_io_log("llm-preselect", payload, content, usage)
                return self._parse_skill_names(content)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[Preselect/llm] Connection error (attempt {attempt}/{max_retries}): {e}"
                )
                last_error = e
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                raise

        if last_error:
            raise last_error
        return []

    @staticmethod
    def _parse_skill_names(content: str) -> List[str]:
        """Parse a JSON array of skill names from LLM output."""
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (code fence markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # Try to find a JSON array in the content
        # First attempt: direct parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

        # Second attempt: find [...] in the text
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return [str(item) for item in result]
            except json.JSONDecodeError:
                pass

        # Fallback: extract quoted strings
        names = re.findall(r'"([^"]+)"', content)
        return names


# =============================================================================
# Strategy 3: RAG / Embedding-based pre-selection
# =============================================================================

class RAGPreselector(SkillPreselector):
    """Use embedding similarity to select the most relevant skills.
    
    Uses an embedding API (OpenAI-compatible /embeddings endpoint) to compute
    cosine similarity between the task instruction and each skill's description.
    Returns the top-k most similar skills.
    """

    def __init__(
        self,
        max_skills: int = 5,
        embedding_model: str = "text-embedding-v4",
        api_base: str = "",
        api_key: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(max_skills=max_skills, logger=logger)
        self.embedding_model = embedding_model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

        # Cache: description_text -> embedding vector
        self._embedding_cache: Dict[str, List[float]] = {}

    @property
    def strategy_name(self) -> str:
        return "rag"

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        if not all_skills:
            return []

        if len(all_skills) <= self.max_skills:
            self.logger.info(
                f"[Preselect/rag] Only {len(all_skills)} skills available "
                f"(<= max_skills={self.max_skills}), returning all"
            )
            return list(all_skills)

        self.logger.info(
            f"[Preselect/rag] Computing embeddings for {len(all_skills)} skills + task..."
        )

        # Build texts to embed
        # For each skill, combine name + description for richer semantics
        skill_texts = [f"{s.name}: {s.description}" for s in all_skills]

        # Embed everything in one batch call (task instruction + all skill texts)
        all_texts = [task_instruction] + skill_texts
        embeddings = self._get_embeddings(all_texts)

        if embeddings is None or len(embeddings) != len(all_texts):
            self.logger.warning(
                "[Preselect/rag] Embedding call failed, falling back to returning all skills"
            )
            return list(all_skills)

        task_embedding = embeddings[0]
        skill_embeddings = embeddings[1:]

        # Compute cosine similarities
        similarities = []
        for i, skill_emb in enumerate(skill_embeddings):
            sim = self._cosine_similarity(task_embedding, skill_emb)
            similarities.append((sim, i))

        # Sort by similarity (descending) and pick top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for _, idx in similarities[: self.max_skills]]

        selected = [all_skills[i] for i in selected_indices]

        self.logger.info(
            f"[Preselect/rag] Selected {len(selected)} skills by embedding similarity: "
            f"{[(all_skills[idx].name, f'{sim:.3f}') for sim, idx in similarities[:self.max_skills]]}"
        )
        return selected

    def _get_embeddings(
        self,
        texts: List[str],
        max_retries: int = 3,
    ) -> Optional[List[List[float]]]:
        """Get embeddings for a list of texts via the /embeddings API."""
        url = f"{self.api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.embedding_model,
            "input": texts,
        }

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)

                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 30 * attempt
                    self.logger.warning(
                        f"[Preselect/rag] HTTP {resp.status_code} (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    self.logger.error(
                        f"[Preselect/rag] Embedding API error {resp.status_code}: {resp.text[:500]}"
                    )
                    return None

                data = resp.json()
                # Sort by index (API may not return them in order)
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[Preselect/rag] Connection error (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                return None

        return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# Strategy 4: Pre-computed RAG (load from disk)
# =============================================================================

class PrecomputedRAGPreselector(SkillPreselector):
    """Load pre-computed RAG results from disk instead of calling the embedding API.

    This strategy reads results produced by ``precompute_rag.py`` which stores
    per-(task, skills_count) top-k matching results as JSON files.

    The preselector needs to know the current *task_name* and *extra_skills_count*
    to locate the correct pre-computed file. These are set by the caller via
    :meth:`set_context` before calling :meth:`select`.

    Directory layout expected::

        {rag_precomputed_dir}/
            {task_name}/
                skills_{extra_skills_count}.json
    """

    def __init__(
        self,
        max_skills: int = 5,
        rag_precomputed_dir: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(max_skills=max_skills, logger=logger)
        self.rag_precomputed_dir = rag_precomputed_dir
        # Caller must set these before calling select()
        self._task_name: Optional[str] = None
        self._extra_skills_count: Optional[int] = None

    @property
    def strategy_name(self) -> str:
        return "rag"

    def set_context(self, task_name: str, extra_skills_count: int) -> None:
        """Set the current task context so select() can find the right file."""
        self._task_name = task_name
        self._extra_skills_count = extra_skills_count

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        if not all_skills:
            return []

        if len(all_skills) <= self.max_skills:
            self.logger.info(
                f"[Preselect/rag-precomputed] Only {len(all_skills)} skills available "
                f"(<= max_skills={self.max_skills}), returning all"
            )
            return list(all_skills)

        if not self._task_name or self._extra_skills_count is None:
            self.logger.warning(
                "[Preselect/rag-precomputed] No context set (task_name/extra_skills_count), "
                "falling back to returning all skills"
            )
            return list(all_skills)

        # Locate pre-computed file
        precomputed_file = (
            Path(self.rag_precomputed_dir)
            / self._task_name
            / f"skills_{self._extra_skills_count}.json"
        )

        if not precomputed_file.exists():
            self.logger.warning(
                f"[Preselect/rag-precomputed] File not found: {precomputed_file}, "
                f"falling back to returning all skills"
            )
            return list(all_skills)

        try:
            with open(precomputed_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(
                f"[Preselect/rag-precomputed] Failed to read {precomputed_file}: {e}, "
                f"falling back to returning all skills"
            )
            return list(all_skills)

        # Extract selected skill names from pre-computed results
        selected_names = [entry["name"] for entry in data.get("selected", [])]
        selected_names = selected_names[: self.max_skills]

        # Map back to SkillInfo objects
        name_to_skill = {s.name: s for s in all_skills}
        selected = []
        for name in selected_names:
            if name in name_to_skill:
                selected.append(name_to_skill[name])
            else:
                self.logger.warning(
                    f"[Preselect/rag-precomputed] Pre-computed skill '{name}' "
                    f"not found in current skill list, skipping"
                )

        self.logger.info(
            f"[Preselect/rag-precomputed] Loaded {len(selected)} skills from "
            f"{precomputed_file.name}: {[s.name for s in selected]}"
        )
        return selected


# =============================================================================
# Strategy 5: Hybrid RAG -> LLM reranking
# =============================================================================

MULTI_QUERY_GENERATION_PROMPT = '''You are a retrieval query planner for a skill-selection system.
Given a task description, write {num_queries} short search queries that cover distinct capabilities likely needed to solve the task.

Guidelines:
- Each query should be concise, concrete, and skill-oriented.
- Prefer capability/library/workflow phrases over restating the whole task.
- Return ONLY a JSON array of strings, no explanation, no markdown.

## Task Description

{task_instruction}

## Search Queries (JSON array)
'''


HYBRID_RAG_LLM_PROMPT = '''You are a budget-aware skill-routing assistant. Given a task description and a short list of candidate skills retrieved by semantic search, select only the skills that are truly useful for completing the task.

Return your answer as a JSON array of skill names. Return at most {max_skills} skills, and use fewer when only one or two skills are needed. If none of the candidates is relevant, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no explanation, no markdown code fences.

## Candidate Skills Retrieved by RAG

{skills_list}

## Task Description

{task_instruction}

## Your Selection (JSON array of skill names)
'''


class HybridRAGLLMPreselector(SkillPreselector):
    """Coarse-to-fine selector: precomputed RAG recall followed by LLM reranking.

    The first stage reads ``rag_precomputed/{task}/skills_{N}.json`` and keeps
    the top ``rag_retrieval_k`` candidates.  The second stage asks an LLM to
    choose at most ``max_skills`` from those candidates.  This keeps the prompt
    small enough for large skill pools while retaining LLM budget awareness.
    """

    def __init__(
        self,
        max_skills: int = 5,
        model_name: str = "gpt-4o",
        api_base: str = "",
        api_key: str = "",
        rag_precomputed_dir: str = "",
        rag_retrieval_k: int = 50,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        preselect_log_dir: Optional[Path] = None,
    ):
        super().__init__(max_skills=max_skills, logger=logger, preselect_log_dir=preselect_log_dir)
        self.model_name = model_name
        self.api_base = _normalize_api_base(api_base)
        self.api_key = api_key
        self.rag_precomputed_dir = rag_precomputed_dir
        self.rag_retrieval_k = rag_retrieval_k
        self.temperature = temperature
        self.final_selection_prompt = HYBRID_RAG_LLM_PROMPT
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._task_name: Optional[str] = None
        self._extra_skills_count: Optional[int] = None

    @property
    def strategy_name(self) -> str:
        return "hybrid-rag-llm"

    def set_context(self, task_name: str, extra_skills_count: int) -> None:
        self._task_name = task_name
        self._extra_skills_count = extra_skills_count

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        if not all_skills:
            return []

        candidates = self._load_rag_candidates(all_skills)
        if not candidates:
            self.logger.warning("[Preselect/hybrid-rag-llm] No RAG candidates available; selecting no skills")
            return []

        self.logger.info(
            f"[Preselect/hybrid-rag-llm] RAG recalled {len(candidates)} candidates "
            f"from {len(all_skills)} skills; asking {self.model_name} to select up to {self.max_skills}"
        )

        selected_names = self._call_llm(task_instruction, candidates)
        name_to_skill = {s.name: s for s in candidates}
        selected: List[SkillInfo] = []
        seen = set()
        for name in selected_names:
            name = name.strip()
            if name in seen:
                continue
            skill = name_to_skill.get(name)
            if skill is None:
                # Case-insensitive fallback.
                for candidate_name, candidate in name_to_skill.items():
                    if candidate_name.lower() == name.lower():
                        skill = candidate
                        break
            if skill is None:
                self.logger.warning(f"[Preselect/hybrid-rag-llm] LLM returned unknown skill name: '{name}'")
                continue
            selected.append(skill)
            seen.add(skill.name)
            if len(selected) >= self.max_skills:
                break

        self.logger.info(
            f"[Preselect/hybrid-rag-llm] Selected {len(selected)} skills: {[s.name for s in selected]}"
        )
        return selected

    def _load_rag_candidates(self, all_skills: List[SkillInfo]) -> List[SkillInfo]:
        if not self._task_name or self._extra_skills_count is None:
            self.logger.warning("[Preselect/hybrid-rag-llm] No context set; cannot locate RAG candidates")
            return list(all_skills[: self.rag_retrieval_k]) if len(all_skills) <= self.rag_retrieval_k else []

        precomputed_file = (
            Path(self.rag_precomputed_dir)
            / self._task_name
            / f"skills_{self._extra_skills_count}.json"
        )
        if not precomputed_file.exists():
            self.logger.warning(f"[Preselect/hybrid-rag-llm] RAG file not found: {precomputed_file}")
            return list(all_skills[: self.rag_retrieval_k]) if len(all_skills) <= self.rag_retrieval_k else []

        try:
            with open(precomputed_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"[Preselect/hybrid-rag-llm] Failed to read {precomputed_file}: {e}")
            return []

        ranked_entries = data.get("all_similarities") or data.get("selected") or []
        candidate_names = [entry.get("name") for entry in ranked_entries if entry.get("name")]
        candidate_names = candidate_names[: self.rag_retrieval_k]

        name_to_skill = {s.name: s for s in all_skills}
        candidates: List[SkillInfo] = []
        missing = []
        for name in candidate_names:
            skill = name_to_skill.get(name)
            if skill is not None:
                candidates.append(skill)
            else:
                missing.append(name)

        if missing:
            self.logger.warning(
                f"[Preselect/hybrid-rag-llm] {len(missing)} RAG candidates not found in current skill list; "
                f"examples={missing[:5]}"
            )
        return candidates

    def _call_llm(self, task_instruction: str, candidates: List[SkillInfo], max_retries: int = 3) -> List[str]:
        skills_text = "\n".join(f"- **{s.name}**: {s.description}" for s in candidates)
        prompt = self.final_selection_prompt.format(
            max_skills=self.max_skills,
            skills_list=skills_text,
            task_instruction=task_instruction.strip(),
        )
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 30 * attempt
                    self.logger.warning(
                        f"[Preselect/hybrid-rag-llm] HTTP {resp.status_code} "
                        f"(attempt {attempt}/{max_retries}), retrying in {wait}s..."
                    )
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                data = resp.json()
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)

                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                self._write_llm_io_log("hybrid-rag-llm-selection", payload, content, usage)
                return LLMPreselector._parse_skill_names(content)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[Preselect/hybrid-rag-llm] Connection error "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                last_error = e
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                raise

        if last_error:
            raise last_error
        return []


class MultiQueryRAGLLMPreselector(HybridRAGLLMPreselector):
    """LLM query planning + multi-query RAG + LLM reranking.

    Compared with ``hybrid-rag-llm``, this method first asks an LLM to produce
    several retrieval queries, runs RAG independently for each query, aggregates
    the resulting scored skill lists, and then asks an LLM to select final skills.
    """

    def __init__(
        self,
        max_skills: int = 5,
        model_name: str = "gpt-4o",
        api_base: str = "",
        api_key: str = "",
        rag_precomputed_dir: str = "",
        rag_retrieval_k: int = 100,
        multi_query_count: int = 5,
        aggregation_strategy: str = "quota",
        embedding_model: str = "text-embedding-v4",
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        preselect_log_dir: Optional[Path] = None,
    ):
        super().__init__(
            max_skills=max_skills,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            rag_precomputed_dir=rag_precomputed_dir,
            rag_retrieval_k=rag_retrieval_k,
            temperature=temperature,
            logger=logger,
            preselect_log_dir=preselect_log_dir,
        )
        self.final_selection_prompt = LLM_PRESELECT_PROMPT
        self.multi_query_count = multi_query_count
        self.aggregation_strategy = aggregation_strategy
        self.embedding_model = embedding_model or DEFAULT_QWEN_EMBEDDING_MODEL
        self.embedding_api_base = (
            os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("DASHSCOPE_API_BASE")
            or os.environ.get("QWEN_BASE_URL")
            or DEFAULT_QWEN_EMBEDDING_API_BASE
        ).rstrip("/")
        self.embedding_api_key = (
            os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or DEFAULT_QWEN_EMBEDDING_API_KEY
        )
        self._cache_vectors: Dict[str, Any] = {}
        self._load_embedding_cache()

    @property
    def strategy_name(self) -> str:
        return "multi-query-rag-llm"

    def select(
        self,
        task_instruction: str,
        all_skills: List[SkillInfo],
    ) -> List[SkillInfo]:
        if not all_skills:
            return []
        if len(all_skills) <= self.max_skills:
            self.logger.info(
                f"[Preselect/multi-query-rag-llm] Only {len(all_skills)} skills available "
                f"(<= max_skills={self.max_skills}), returning all"
            )
            return list(all_skills)
        skill_vectors = self._build_skill_vectors(all_skills)
        if not skill_vectors:
            self.logger.warning("[Preselect/multi-query-rag-llm] No cached skill vectors available; selecting no skills")
            return []

        queries = self._generate_queries(task_instruction)
        if not queries:
            queries = [task_instruction]
        query_rankings = []
        for query in queries:
            ranking = self._search_query(query, skill_vectors)
            if ranking:
                query_rankings.append(ranking)
        if not query_rankings:
            self.logger.warning("[Preselect/multi-query-rag-llm] No RAG query rankings available; selecting no skills")
            return []

        aggregated = MultiQueryCandidateAggregator.aggregate(
            query_rankings,
            budget=self.rag_retrieval_k,
            strategy=self.aggregation_strategy,
        )
        candidates = [c.item for c in aggregated]
        self.logger.info(f"[Preselect/multi-query-rag-llm] Generated queries: {queries}")
        self.logger.info(
            f"[Preselect/multi-query-rag-llm] Aggregated {len(candidates)} candidates by "
            f"{self.aggregation_strategy}: "
            f"{[(c.item.name, round(float(c.score), 4), c.best_rank, c.query_hits) for c in aggregated]}"
        )
        self.logger.info(
            f"[Preselect/multi-query-rag-llm] Asking {self.model_name} to select up to {self.max_skills}"
        )
        selected_names = self._call_llm(task_instruction, candidates)
        return self._map_selected_names(selected_names, candidates)

    def _map_selected_names(self, selected_names: List[str], candidates: List[SkillInfo]) -> List[SkillInfo]:
        name_to_skill = {s.name: s for s in candidates}
        selected: List[SkillInfo] = []
        seen = set()
        for name in selected_names:
            name = name.strip()
            if name in seen:
                continue
            skill = name_to_skill.get(name)
            if skill is None:
                for candidate_name, candidate in name_to_skill.items():
                    if candidate_name.lower() == name.lower():
                        skill = candidate
                        break
            if skill is None:
                self.logger.warning(f"[Preselect/multi-query-rag-llm] LLM returned unknown skill name: '{name}'")
                continue
            selected.append(skill)
            seen.add(skill.name)
            if len(selected) >= self.max_skills:
                break
        self.logger.info(
            f"[Preselect/multi-query-rag-llm] Selected {len(selected)} skills: {[s.name for s in selected]}"
        )
        return selected

    def _generate_queries(self, task_instruction: str) -> List[str]:
        prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            num_queries=self.multi_query_count,
            task_instruction=task_instruction.strip(),
        )
        names = self._call_json_array_prompt(prompt, log_prefix="query-generation")
        queries = [q.strip() for q in names if q.strip()]
        return queries[: self.multi_query_count]

    def _call_json_array_prompt(self, prompt: str, log_prefix: str, max_retries: int = 3) -> List[str]:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 30 * attempt
                    self.logger.warning(
                        f"[Preselect/multi-query-rag-llm:{log_prefix}] HTTP {resp.status_code} "
                        f"(attempt {attempt}/{max_retries}), retrying in {wait}s..."
                    )
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                self._write_llm_io_log(f"multi-query-rag-llm-{log_prefix}", payload, content, usage)
                return LLMPreselector._parse_skill_names(content)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[Preselect/multi-query-rag-llm:{log_prefix}] Connection error "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                last_error = e
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                raise
        if last_error:
            raise last_error
        return []

    def _load_embedding_cache(self) -> None:
        cache_dir = Path(self.rag_precomputed_dir) / "cache"
        index_file = cache_dir / "embedding_index.json"
        cache_file = cache_dir / "embedding_cache.npz"
        if np is None:
            self.logger.warning("[Preselect/multi-query-rag-llm] numpy unavailable; cannot load embedding cache")
            return
        if not index_file.exists() or not cache_file.exists():
            self.logger.warning(f"[Preselect/multi-query-rag-llm] Embedding cache missing under {cache_dir}")
            return
        try:
            with open(index_file) as f:
                index = json.load(f)
            data = np.load(cache_file)
            self._cache_vectors = {h: data[k] for h, k in index.items() if k in data}
            self.logger.info(f"[Preselect/multi-query-rag-llm] Loaded {len(self._cache_vectors)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"[Preselect/multi-query-rag-llm] Failed to load embedding cache: {e}")
            self._cache_vectors = {}

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _build_skill_vectors(self, skills: List[SkillInfo]) -> List[Tuple[SkillInfo, Any]]:
        out = []
        missing = 0
        for skill in skills:
            text = f"{skill.name.strip()}: {skill.description.strip()}"
            vec = self._cache_vectors.get(self._text_hash(text))
            if vec is None:
                missing += 1
                continue
            out.append((skill, vec))
        self.logger.info(
            f"[Preselect/multi-query-rag-llm] Indexed {len(out)}/{len(skills)} skill vectors (missing={missing})"
        )
        return out

    def _embed_query(self, query: str) -> Optional[Any]:
        cached = self._cache_vectors.get(self._text_hash(query))
        if cached is not None:
            return cached
        url = f"{self.embedding_api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.embedding_api_key}",
        }
        payload = {"model": self.embedding_model, "input": [query]}
        for attempt in range(1, 4):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 10 * attempt
                    self.logger.warning(
                        f"[Preselect/multi-query-rag-llm] Embedding HTTP {resp.status_code} "
                        f"(attempt {attempt}/3), retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    self.logger.error(
                        f"[Preselect/multi-query-rag-llm] Embedding API error {resp.status_code}: {resp.text[:500]}"
                    )
                    return None
                data = resp.json()
                item = sorted(data["data"], key=lambda x: x.get("index", 0))[0]
                return np.array(item["embedding"], dtype=np.float32) if np is not None else item["embedding"]
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.logger.warning(
                    f"[Preselect/multi-query-rag-llm] Embedding connection error (attempt {attempt}/3): {e}"
                )
                if attempt < 3:
                    time.sleep(10)
        return None

    def _search_query(self, query: str, skill_vectors: List[Tuple[SkillInfo, Any]]) -> List[Tuple[SkillInfo, float]]:
        query_vec = self._embed_query(query)
        if query_vec is None:
            return []
        scored = []
        for skill, vec in skill_vectors:
            scored.append((skill, self._cosine_similarity(query_vec, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        if np is not None:
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# Factory
# =============================================================================

def create_preselector(
    strategy: str = "naive",
    max_skills: int = 5,
    model_name: str = "",
    api_base: str = "",
    api_key: str = "",
    embedding_model: str = "text-embedding-v4",
    temperature: float = 0.0,
    logger: Optional[logging.Logger] = None,
    rag_precomputed_dir: str = "",
    hybrid_rag_retrieval_k: int = 100,
    preselect_log_dir: Optional[Path] = None,
) -> SkillPreselector:
    """Factory function to create a skill pre-selector.

    Args:
        strategy: One of "naive", "llm", "rag", "hybrid-rag-llm", "dynamic-rag".
        max_skills: Maximum number of skills to select (ignored for "naive").
        model_name: LLM model name (for "llm" strategy).
        api_base: API base URL (for "llm" and "rag" strategies).
        api_key: API key (for "llm" and "rag" strategies).
        embedding_model: Embedding model name (for "rag" strategy).
        temperature: LLM temperature (for "llm" strategy).
        logger: Optional logger.
        rag_precomputed_dir: Path to pre-computed RAG results directory.
            When provided and strategy is "rag", uses PrecomputedRAGPreselector
            instead of the online RAGPreselector.

    Returns:
        A SkillPreselector instance.
    """
    strategy = strategy.lower().strip()

    if strategy == "naive":
        return NaivePreselector(max_skills=max_skills, logger=logger)

    elif strategy in ("dynamic-rag", "dynamic_rag", "rag-tool", "rag_tool"):
        return EmptyPreselector(max_skills=max_skills, logger=logger)

    elif strategy in ("llm", "llm-preselect", "llm_preselect"):
        if not api_base or not api_key:
            raise ValueError("LLM pre-selection requires api_base and api_key")
        return LLMPreselector(
            max_skills=max_skills,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            logger=logger,
        )

    elif strategy in ("multi-query-rag-llm", "multi_query_rag_llm", "multi-rag-llm", "multi_rag_llm"):
        if not api_base or not api_key:
            raise ValueError("Multi-query RAG-LLM pre-selection requires api_base and api_key")
        if not rag_precomputed_dir or not Path(rag_precomputed_dir).exists():
            raise ValueError("Multi-query RAG-LLM pre-selection requires rag_precomputed_dir")
        return MultiQueryRAGLLMPreselector(
            max_skills=max_skills,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            rag_precomputed_dir=rag_precomputed_dir,
            rag_retrieval_k=hybrid_rag_retrieval_k,
            multi_query_count=5,
            aggregation_strategy="quota",
            embedding_model=embedding_model,
            temperature=temperature,
            logger=logger,
        )

    elif strategy in ("hybrid-rag-llm", "hybrid_rag_llm", "rag-llm", "rag_llm", "retrieve-rerank", "retrieve_rerank"):
        if not api_base or not api_key:
            raise ValueError("Hybrid RAG-LLM pre-selection requires api_base and api_key")
        if not rag_precomputed_dir or not Path(rag_precomputed_dir).exists():
            raise ValueError("Hybrid RAG-LLM pre-selection requires rag_precomputed_dir")
        return HybridRAGLLMPreselector(
            max_skills=max_skills,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            rag_precomputed_dir=rag_precomputed_dir,
            rag_retrieval_k=hybrid_rag_retrieval_k,
            temperature=temperature,
            logger=logger,
            preselect_log_dir=preselect_log_dir,
        )

    elif strategy in ("rag", "embedding", "semantic"):
        # If pre-computed directory is provided, use disk-based preselector
        if rag_precomputed_dir and Path(rag_precomputed_dir).exists():
            return PrecomputedRAGPreselector(
                max_skills=max_skills,
                rag_precomputed_dir=rag_precomputed_dir,
                logger=logger,
            )
        # Otherwise fall back to online API
        if not api_base or not api_key:
            raise ValueError("RAG pre-selection requires api_base and api_key")
        return RAGPreselector(
            max_skills=max_skills,
            embedding_model=embedding_model,
            api_base=api_base,
            api_key=api_key,
            logger=logger,
        )

    else:
        raise ValueError(
            f"Unknown pre-selection strategy: '{strategy}'. "
            f"Choose from: 'naive', 'llm', 'rag', 'hybrid-rag-llm', 'multi-query-rag-llm', 'dynamic-rag'"
        )
