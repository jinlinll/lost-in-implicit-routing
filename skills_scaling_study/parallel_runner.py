#!/usr/bin/env python3
"""
Parallel Batch Runner for Skills Scaling Study.

Features:
- Parallel task execution with configurable concurrency
- Multi-model support: test multiple models across the same task suite
- Docker base image caching: same task shares one base image across all
  skills counts, models, and passes; skills are injected at container runtime
- Automatic cleanup/LRU eviction of cached base images
"""

import asyncio
import datetime
import json
import logging
import math
import os
import random
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from pool_order import ordered_pool_paths
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from contextlib import asynccontextmanager

from standalone_agent import (
    StandaloneAgent, 
    AgentConfig, 
    AgentResult,
    DockerEnvironment,
    DockerEnvironmentBase,
    get_docker_resource_args,  # Docker 资源限制
)
from tokenplan_client import get_default_tokenplan_api_base, get_default_tokenplan_api_key


# Special sentinel value: when extra_skills_count == NO_SKILLS, the runner
# will NOT copy any skills at all (neither task-specific nor extra random
# skills) into the Docker container.  The agent will see "No skills
# available." and must solve the task without any skill guidance.
NO_SKILLS = -1


def skills_dir_label(extra_skills_count: int) -> str:
    """Return the directory-name component for a given extra_skills_count.

    -1 (NO_SKILLS) → "skills_none"   (completely empty skill directory)
     0              → "skills_0"      (only task-specific skills, no extras)
     N              → "skills_N"
    """
    if extra_skills_count == NO_SKILLS:
        return "skills_none"
    return f"skills_{extra_skills_count}"


def setup_experiment_logging(output_dir: Path) -> logging.Logger:
    """
    Set up experiment-wide logging.
    
    Creates a logger that writes to both:
    - Console (stderr) with concise format
    - File (output_dir/experiment.log) with detailed format including timestamps
    
    Returns the configured logger.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers (in case of re-init)
    logger.handlers.clear()
    
    # File handler: detailed, append mode
    log_file = output_dir / "experiment.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    
    # Console handler: concise
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    
    return logger


def setup_task_logging(task_output_dir: Path, task_id: str) -> logging.Logger:
    """
    Set up per-task logging to a dedicated file.
    
    Each task gets its own log file at: {task_output_dir}/run.log
    Also propagates to the experiment-wide logger.
    """
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"experiment.task.{task_id}")
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Per-task file handler
    log_file = task_output_dir / "run.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    
    # Propagate to parent (experiment logger) so messages appear in
    # experiment.log and console as well
    logger.propagate = True
    
    return logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ParallelExperimentConfig:
    """Configuration for parallel skills scaling experiment."""
    
    # API settings - supports multiple models
    model_names: List[str]  # List of model names to test
    api_base: str
    api_key: str
    
    # Task settings
    task_names: List[str]
    tasks_dir: Path
    
    # Experiment settings
    extra_skills_counts: List[int] = field(default_factory=lambda: [0, 10, 100])
    num_passes: int = 1
    random_seed: int = 42
    
    # Agent settings
    max_episodes: int = 50
    temperature: Optional[float] = None
    
    # Skill pre-selection settings
    preselect_strategy: str = "naive"  # "naive", "llm", "rag", "hybrid-rag-llm", or "dynamic-rag"
    preselect_max_skills: int = 5  # Max skills to select (ignored for "naive")
    preselect_model_name: Optional[str] = None  # LLM model for "llm" strategy
    preselect_embedding_model: str = "text-embedding-v4"  # Qwen/DashScope embedding model for "rag"
    rag_precomputed_dir: str = ""  # Path to pre-computed RAG results directory
    hybrid_rag_retrieval_k: int = 50  # RAG candidates passed to LLM reranker for hybrid-rag-llm
    
    # Parallel execution settings
    max_parallel_tasks: int = 4
    
    # Docker base image cache settings
    max_cached_images: int = 20  # LRU cache limit for Docker base images
    
    # Output settings
    output_dir: Path = Path("experiment_logs")
    
    # Skills pool
    skills_pool_dir: Optional[Path] = None
    include_task_skills: bool = True
    
    # Backward compatibility: single model name property
    @property
    def model_name(self) -> str:
        """Return first model name for backward compatibility."""
        return self.model_names[0] if self.model_names else ""
    
    def get_output_dir_name(self) -> str:
        """Generate output directory name with preselect strategy suffix.
        
        Examples:
            - experiment_logs (naive, the default)
            - experiment_logs-llm-preselect
            - experiment_logs-rag
            - experiment_logs-dynamic-rag
        """
        base_name = str(self.output_dir)
        if self.preselect_strategy == "naive":
            return base_name
        return f"{base_name}-{self.preselect_strategy}"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_name: str
    model_name: str
    extra_skills_count: int
    pass_index: int
    seed: int
    success: bool
    episodes: int
    input_tokens: int
    output_tokens: int
    error: Optional[str] = None
    execution_time_sec: float = 0.0
    loaded_skills: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pre-selection info
    preselect_strategy: str = "naive"
    preselected_skills: List[str] = field(default_factory=list)
    preselect_input_tokens: int = 0
    preselect_output_tokens: int = 0


@dataclass
class AggregatedResult:
    """Aggregated results for a model + skill count combination."""
    model_name: str
    extra_skills_count: int
    task_results: Dict[str, List[bool]] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time_sec: float = 0.0
    
    @property
    def tasks_run(self) -> int:
        return len(self.task_results)
    
    @property
    def tasks_passed(self) -> int:
        return sum(1 for results in self.task_results.values() if any(results))
    
    def pass_at_k(self, k: int) -> float:
        """Calculate pass@k using the unbiased estimator."""
        if not self.task_results or k <= 0:
            return 0.0
        
        total = 0.0
        valid_tasks = 0
        
        for task, results in self.task_results.items():
            n = len(results)
            if n == 0 or k > n:
                continue
            
            c = sum(results)
            valid_tasks += 1
            
            if c == 0:
                total += 0.0
            else:
                if n - c < k:
                    total += 1.0
                else:
                    log_num = sum(math.log(n - c - i) for i in range(k))
                    log_den = sum(math.log(n - i) for i in range(k))
                    total += 1.0 - math.exp(log_num - log_den)
        
        return total / valid_tasks if valid_tasks > 0 else 0.0


# =============================================================================
# Skills Manager
# =============================================================================

class SkillsManager:
    """Manages skill selection and injection for experiments."""
    
    def __init__(self, skills_pool_dir: Optional[Path] = None):
        self.skills_pool_dir = skills_pool_dir
        self._available_skills: List[Path] = []
        
        if skills_pool_dir and skills_pool_dir.exists():
            self._available_skills = [
                p for p in ordered_pool_paths(skills_pool_dir) 
                if p.is_dir() and (p / "SKILL.md").exists()
            ]
    
    def get_available_skills(self) -> List[str]:
        """Get list of available skill names."""
        return [p.name for p in self._available_skills]
    
    def sample_skills(self, count: int, exclude: List[str], seed: int) -> List[Path]:
        """Sample random skills from the pool, excluding certain names."""
        if count <= 0:
            return []
        
        available = [p for p in self._available_skills if p.name not in exclude]
        
        if len(available) < count:
            print(f"Warning: Only {len(available)} skills available, requested {count}")
            count = len(available)
        
        random.seed(seed)
        return random.sample(available, count)
    
    def copy_skills_to_dir(
        self, 
        skills: List[Path], 
        dest_dir: Path,
        task_skills_dir: Optional[Path] = None,
    ) -> None:
        """Copy skills to a destination directory.
        
        Copies:
        1. All contents from task_skills_dir (subdirectories AND loose files
           like reference.md, LICENSE, etc.)
        2. Extra sampled skill directories
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if task_skills_dir and task_skills_dir.exists():
            for item in task_skills_dir.iterdir():
                dest = dest_dir / item.name
                if dest.exists():
                    continue
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        
        for skill_path in skills:
            dest = dest_dir / skill_path.name
            if not dest.exists():
                shutil.copytree(skill_path, dest)


# =============================================================================
# Docker Image Cache Manager
# =============================================================================

class BaseImageBusy(Exception):
    """Raised when base image is currently being built by another worker.
    
    When a worker encounters this exception, it should put the task back
    into the queue and process other tasks first.
    """
    pass


class BaseImageBuildFailed(Exception):
    """Raised when a task base image build has failed and should not be retried."""
    pass


class DockerImageCache:
    """
    Manages Docker image caching for experiments with LRU eviction.
    
    Base-image-only caching architecture:
    
    Task base image (shared across ALL skills_count values):
        Cache key: task_name -> base_image_tag
        Contains: Dockerfile build result (OS packages, downloads, source code, etc.)
        This is the expensive part (network downloads, compilation) and is
        built exactly ONCE per task, then shared by all skills_count/seed combos.

    Runtime injection:
        Environment files and selected skills are copied into each task container
        after it starts. We no longer commit per-(task, skills_count, seed) images.
    
    Network acceleration:
        All Dockerfiles are automatically patched to use Tencent mirrors for:
        - apt (Ubuntu/Debian)
        - pip (PyPI)
        - npm (Node.js)
        - Maven (Java)
        - Go modules
    
    LRU eviction: Only the most recently used `max_cached_images` base images
    are kept in Docker. Evicting a base image removes its tag; running containers
    based on that image continue to work until they exit.
    
    Non-blocking base image builds:
        When a base image is being built, other workers requesting the same
        base image will receive a BaseImageBusy exception instead of blocking.
        This allows workers to skip the task and process other tasks first.
    """
    
    DEFAULT_MAX_CACHED_IMAGES = 20
    
    def __init__(self, max_cached_images: int = DEFAULT_MAX_CACHED_IMAGES, strategy: str = "naive", max_parallel: int = 4):
        # Base cache: task_name -> base_image_tag (shared across skills_count/seed/model/pass)
        self._task_base_cache: Dict[str, str] = {}
        self._base_images: Set[str] = set()  # track _base images actually created
        # One asyncio.Lock per task: ensures only one coroutine builds the base image.
        self._task_base_locks: Dict[str, asyncio.Lock] = {}
        # Track which tasks are currently building their base image (for non-blocking check)
        self._task_base_building: Set[str] = set()
        # Task-level build failures are memoized so all later configs fail fast.
        self._failed_base_builds: Dict[str, str] = {}
        self._meta_lock = asyncio.Lock()  # protects base locks/build/failure/LRU state
        
        # Base-image LRU tracking: task names ordered by last access time (most recent at end)
        self._base_access_order: List[str] = []
        # Active runtime containers using each task base; active bases are not evicted.
        self._base_ref_counts: Dict[str, int] = {}
        self._max_cached_images = max_cached_images
        self._strategy = strategy  # preselect strategy, used to namespace image names
        self._max_parallel = max_parallel  # 用于计算 Docker 资源限制
    
    def _touch_base(self, task_name: str) -> None:
        """Move a task's base image to the end of the LRU order."""
        if task_name in self._base_access_order:
            self._base_access_order.remove(task_name)
        self._base_access_order.append(task_name)
    
    def _make_base_image_name(self, task_name: str) -> str:
        """Generate a base image name for a task (shared across skills_count).
        
        Includes preselect strategy in the name so concurrent runs with
        different strategies don't collide.
        """
        safe_task = task_name.lower().replace("-", "_").replace(".", "_")
        if self._strategy and self._strategy != "naive":
            return f"skillscache_{self._strategy}_{safe_task}_base"
        return f"skillscache_{safe_task}_base"
    
    async def _record_base_build_failure(self, task_name: str, error_msg: str) -> None:
        """Memoize a task-level base image build failure."""
        async with self._meta_lock:
            self._failed_base_builds[task_name] = error_msg
    
    async def _cleanup_failed_build_artifacts(self) -> None:
        """Best-effort cleanup for Docker legacy-builder leftovers after build failure."""
        # Never prune unrelated Docker resources by default.
        if os.environ.get("SKILLS_ALLOW_GLOBAL_DOCKER_PRUNE") != "1":
            return
        for cmd in (["docker", "container", "prune", "-f"], ["docker", "image", "prune", "-f"]):
            try:
                await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
            except Exception as e:
                print(f"[ImageCache] Warning: failed to run {' '.join(cmd)}: {e}")
    
    async def _evict_base_images_if_needed(self) -> None:
        """Evict least-recently-used inactive base images if cache exceeds max size."""
        while len(self._base_access_order) > self._max_cached_images:
            evict_task = None
            for candidate in list(self._base_access_order):
                if self._base_ref_counts.get(candidate, 0) > 0:
                    continue
                if candidate in self._task_base_building:
                    continue
                evict_task = candidate
                break
            if evict_task is None:
                # All cached bases are currently active; defer cleanup until a task finishes.
                return

            self._base_access_order.remove(evict_task)
            base_image = self._task_base_cache.pop(evict_task, None)
            if not base_image or base_image == "ubuntu:22.04":
                continue
            
            self._base_images.discard(base_image)
            print(
                f"[ImageCache/LRU] Evicting inactive base image: {base_image} "
                f"(task={evict_task}, limit={self._max_cached_images})"
            )
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rmi", "-f", base_image],
                    capture_output=True,
                )
            except Exception as e:
                print(f"[ImageCache/LRU] Warning: Failed to remove base image {base_image}: {e}")
    
    # ── Network acceleration patches ──
    
    @staticmethod
    def _apt_mirror_snippet() -> str:
        """Return a RUN snippet that switches apt sources to nearby mirrors.
        
        NOTE: Some base images (e.g. python:slim) run as non-root user.
        We use '|| true' to ignore permission errors gracefully, since
        apt mirror switching is optional optimization.
        """
        return (
            "# --- Auto-injected: switch to nearby apt mirrors (best-effort) ---\n"
            "RUN "
            # Ubuntu (legacy sources.list) — also handles Debian legacy
            "(if [ -f /etc/apt/sources.list ]; then "
            "sed -i 's|http://archive.ubuntu.com|http://mirrors.tencent.com|g; "
            "s|http://security.ubuntu.com|http://mirrors.tencent.com|g; "
            "s|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g; "
            "s|http://deb.debian.org|http://mirrors.tencent.com|g; "
            "s|http://security.debian.org|http://mirrors.tencent.com|g' "
            "/etc/apt/sources.list 2>/dev/null; "
            "fi; "
            # Ubuntu 24.04+ (DEB822 format)
            "if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then "
            "sed -i 's|http://archive.ubuntu.com|http://mirrors.tencent.com|g; "
            "s|http://security.ubuntu.com|http://mirrors.tencent.com|g; "
            "s|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' "
            "/etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; "
            "fi; "
            # Debian (DEB822 format, e.g. python:slim images)
            "if [ -f /etc/apt/sources.list.d/debian.sources ]; then "
            "sed -i 's|http://deb.debian.org|http://mirrors.tencent.com|g; "
            "s|http://security.debian.org|http://mirrors.tencent.com|g' "
            "/etc/apt/sources.list.d/debian.sources 2>/dev/null; "
            "fi) || true\n"
            "# --- End auto-injected apt mirror patch ---\n"
        )
    
    @staticmethod
    def _network_accel_env_snippet() -> str:
        """Return ENV lines for pip/npm/Maven/Go mirrors (inserted once after first FROM)."""
        return (
            "# --- Auto-injected: network acceleration for pip/npm/maven/go ---\n"
            "# pip: use the smoke-tested Tsinghua PyPI mirror\n"
            "ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple "
            "PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn\n"
            "# npm: use Tencent npm mirror\n"
            "ENV NPM_CONFIG_REGISTRY=https://mirrors.tencent.com/npm/\n"
            "# Go: use Tencent Go module proxy\n"
            "ENV GOPROXY=https://mirrors.tencent.com/go/,direct\n"
            "# --- End network acceleration env ---\n"
        )
    
    @staticmethod
    def _maven_mirror_snippet() -> str:
        """Return a RUN snippet that injects Tencent Maven mirror into settings.xml.
        
        Maven doesn't support ENV-based mirror config, so we write settings.xml.
        This is injected once after the first FROM line.
        
        NOTE: Some images run as non-root. We write to $HOME/.m2 which works
        for both root and non-root users. Use '|| true' to ignore failures.
        """
        # Use printf to write multi-line content in a single RUN command
        # Write to $HOME/.m2 instead of /root/.m2 to support non-root users
        settings_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\\n'
            "<settings>\\n"
            "  <mirrors>\\n"
            "    <mirror>\\n"
            "      <id>tencent</id>\\n"
            "      <mirrorOf>*</mirrorOf>\\n"
            "      <url>https://mirrors.tencent.com/nexus/repository/maven-public/</url>\\n"
            "    </mirror>\\n"
            "  </mirrors>\\n"
            "</settings>\\n"
        )
        return (
            "# --- Auto-injected: Maven Tencent mirror (best-effort) ---\n"
            f'RUN (mkdir -p $HOME/.m2 && printf \'{settings_xml}\' > $HOME/.m2/settings.xml) || true\n'
            "# --- End Maven mirror ---\n"
        )
    
    def _patch_dockerfile(self, original_content: str) -> str:
        """Patch a Dockerfile with network acceleration.
        
        Injects after each FROM line:
        1. apt mirror switch (after every FROM, for multi-stage builds)
        2. pip/npm/go ENV vars (after first FROM only)
        3. Maven settings.xml (after first FROM only)
        """
        apt_snippet = self._apt_mirror_snippet()
        env_snippet = self._network_accel_env_snippet()
        maven_snippet = self._maven_mirror_snippet()
        
        patched_lines = []
        first_from_seen = False
        
        for line in original_content.splitlines(keepends=True):
            patched_lines.append(line)
            if line.strip().upper().startswith("FROM"):
                # apt mirror after every FROM
                patched_lines.append(apt_snippet)
                if not first_from_seen:
                    # pip/npm/go/maven only after first FROM
                    patched_lines.append(env_snippet)
                    patched_lines.append(maven_snippet)
                    first_from_seen = True
        
        return "".join(patched_lines)
    
    # ── Tier 1: Task base image (shared across skills_count) ──
    
    async def _get_or_build_task_base(self, task_name: str, task_dir: Path, blocking: bool = True) -> str:
        """
        Build or retrieve the task base image (Tier 1).
        
        This image contains the full Dockerfile build result and is SHARED
        across all skills_count values for the same task. It is the expensive
        part (network downloads, compilation).
        
        Args:
            task_name: Name of the task
            task_dir: Path to the task directory
            blocking: If False and base image is being built by another worker,
                      raise BaseImageBusy instead of waiting.
        
        Returns:
            The base image name/tag.
        
        Raises:
            BaseImageBusy: If blocking=False and another worker is building the base image.
            RuntimeError: If the Docker build fails.
        
        Thread-safe: if multiple coroutines request the same task base
        concurrently, only one will build it; others will wait (if blocking=True)
        or receive BaseImageBusy (if blocking=False).
        """
        # Fast path: already cached or known failed
        if task_name in self._task_base_cache:
            return self._task_base_cache[task_name]
        if task_name in self._failed_base_builds:
            raise BaseImageBuildFailed(self._failed_base_builds[task_name])
        
        # Get or create the lock, and check if building is in progress
        async with self._meta_lock:
            if task_name in self._failed_base_builds:
                raise BaseImageBuildFailed(self._failed_base_builds[task_name])
            # Check if another worker is building this base image
            if task_name in self._task_base_building:
                if not blocking:
                    raise BaseImageBusy(f"Base image for '{task_name}' is being built by another worker")
                # If blocking=True, we'll wait for the lock below
            
            if task_name not in self._task_base_locks:
                self._task_base_locks[task_name] = asyncio.Lock()
            task_lock = self._task_base_locks[task_name]
        
        # Acquire the lock (blocking or non-blocking)
        async with task_lock:
            # Double-check cache/failure after acquiring lock (another worker may have finished)
            if task_name in self._task_base_cache:
                return self._task_base_cache[task_name]
            if task_name in self._failed_base_builds:
                raise BaseImageBuildFailed(self._failed_base_builds[task_name])
            
            # Actually build the base image
            return await self._build_task_base_impl(task_name, task_dir)
    
    async def _build_task_base_impl(self, task_name: str, task_dir: Path) -> str:
        """Internal implementation of base image building."""
        # Double-check after acquiring lock
        if task_name in self._task_base_cache:
            return self._task_base_cache[task_name]
        if task_name in self._failed_base_builds:
            raise BaseImageBuildFailed(self._failed_base_builds[task_name])
        
        # Mark this task as building
        async with self._meta_lock:
            self._task_base_building.add(task_name)
        
        try:
            dockerfile = task_dir / "environment" / "Dockerfile"
            
            if not dockerfile.exists():
                base_image = "ubuntu:22.04"
                self._task_base_cache[task_name] = base_image
                return base_image
            
            base_image = self._make_base_image_name(task_name)
            build_context = str(task_dir / "environment")
            
            # Check if the image already exists in Docker (from a previous run)
            check_result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "image", "inspect", base_image],
                capture_output=True,
            )
            if check_result.returncode == 0:
                print(f"[ImageCache] Reusing existing base image: {base_image}")
                self._task_base_cache[task_name] = base_image
                self._base_images.add(base_image)
                return base_image
            
            print(f"[ImageCache] Building SHARED base image for task '{task_name}'...")
            print(f"[ImageCache]   Dockerfile: {dockerfile}")
            print(f"[ImageCache]   Build context: {build_context}")
            print(f"[ImageCache]   Target tag: {base_image}")
            print(f"[ImageCache]   (This base is shared across ALL skills_count values)")
            
            # Patch the Dockerfile with network acceleration
            original_content = dockerfile.read_text()
            patched_content = self._patch_dockerfile(original_content)
            
            # Write patched Dockerfile to a temp file in the build context.
            # Multiple ablation runners may build the same task concurrently.
            # A process-unique path prevents one runner's cleanup from deleting
            # another runner's Dockerfile while its `docker build` is active.
            patched_dockerfile = (
                Path(build_context)
                / f".Dockerfile.patched.{base_image}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            )
            try:
                patched_dockerfile.write_text(patched_content)
                
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "build",
                     "-f", str(patched_dockerfile),
                     "-t", base_image, build_context],
                    capture_output=True,
                    text=True,
                )
            finally:
                patched_dockerfile.unlink(missing_ok=True)
            
            if result.returncode != 0:
                print(f"[ImageCache] *** DOCKER BUILD FAILED for task '{task_name}' ***")
                print(f"[ImageCache]   Return code: {result.returncode}")
                print(f"[ImageCache]   Dockerfile: {dockerfile}")
                if result.stdout:
                    stdout_lines = result.stdout.strip().splitlines()
                    tail = stdout_lines[-50:] if len(stdout_lines) > 50 else stdout_lines
                    print(f"[ImageCache]   Build stdout (last {len(tail)} lines):")
                    for line in tail:
                        print(f"[ImageCache]     | {line}")
                if result.stderr:
                    stderr_lines = result.stderr.strip().splitlines()
                    tail = stderr_lines[-50:] if len(stderr_lines) > 50 else stderr_lines
                    print(f"[ImageCache]   Build stderr (last {len(tail)} lines):")
                    for line in tail:
                        print(f"[ImageCache]     | {line}")
                error_msg = (
                    f"Docker build failed for task '{task_name}', "
                    f"rc={result.returncode}.\n"
                    f"Dockerfile: {dockerfile}\n"
                    f"STDERR:\n{result.stderr}\n"
                    f"STDOUT (last 30 lines):\n{''.join(result.stdout.strip().splitlines(keepends=True)[-30:])}"
                )
                await self._record_base_build_failure(task_name, error_msg)
                await self._cleanup_failed_build_artifacts()
                raise BaseImageBuildFailed(error_msg)
            
            self._task_base_cache[task_name] = base_image
            self._base_images.add(base_image)
            print(f"[ImageCache] Base image ready: {base_image}")
            return base_image
        
        except BaseImageBuildFailed:
            raise
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            await self._record_base_build_failure(task_name, error_msg)
            await self._cleanup_failed_build_artifacts()
            raise BaseImageBuildFailed(error_msg) from e
        finally:
            # Remove from building set
            async with self._meta_lock:
                self._task_base_building.discard(task_name)
    
    # ── Base image retrieval (skills are injected into runtime containers) ──
    
    async def get_or_build_image(
        self,
        task_name: str,
        skills_count: int,
        seed: int,
        task_dir: Path,
        skills_dir: Path,
        blocking: bool = True,
    ) -> str:
        """
        Get or build the shared task base image.
        
        Skills/env files are no longer committed into per-skills images. The
        returned base image is used to start a fresh runtime container, and the
        caller copies the selected skills into that container before running.
        
        Args:
            task_name: Name of the task.
            skills_count: Kept for call-site compatibility; no longer part of the image cache key.
            seed: Kept for call-site compatibility; no longer part of the image cache key.
            task_dir: Path to the task directory.
            skills_dir: Kept for call-site compatibility; copied at runtime by the caller.
            blocking: If False and base image is being built by another worker,
                      raise BaseImageBusy instead of waiting.
        
        Returns:
            The base image name to use for creating containers.
        """
        _ = (skills_count, seed, skills_dir)  # compatibility-only arguments
        base_image = await self._get_or_build_task_base(task_name, task_dir, blocking=blocking)
        async with self._meta_lock:
            self._touch_base(task_name)
            self._base_ref_counts[task_name] = self._base_ref_counts.get(task_name, 0) + 1
        await self._evict_base_images_if_needed()
        return base_image
    
    async def release_base_image(self, task_name: str) -> None:
        """Release a runtime reference to a task base image and trigger LRU cleanup."""
        async with self._meta_lock:
            count = self._base_ref_counts.get(task_name, 0)
            if count <= 1:
                self._base_ref_counts.pop(task_name, None)
            else:
                self._base_ref_counts[task_name] = count - 1
        await self._evict_base_images_if_needed()
    
    def get_all_cached_images(self) -> List[str]:
        """Get list of cached base image names (only actually created ones)."""
        return list(self._base_images)
    
    async def cleanup_all(self) -> None:
        """Remove all cached base images."""
        images = self.get_all_cached_images()
        
        if not images:
            return
        
        print(f"[ImageCache] Cleaning up {len(images)} cached base images...")
        
        for image_name in images:
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rmi", "-f", image_name],
                    capture_output=True,
                )
            except Exception as e:
                print(f"[ImageCache] Warning: Failed to remove {image_name}: {e}")
        
        self._base_images.clear()
        self._task_base_cache.clear()
        self._base_access_order.clear()
        self._base_ref_counts.clear()
        self._failed_base_builds.clear()
        
        print("[ImageCache] Cleanup complete")


# =============================================================================
# Parallel Task Runner
# =============================================================================

class ParallelTaskRunner:
    """Runs tasks in parallel with Docker base-image caching and runtime skill injection."""
    
    def __init__(self, config: ParallelExperimentConfig, image_cache: DockerImageCache, logger: logging.Logger):
        self.config = config
        self.image_cache = image_cache
        self.skills_manager = SkillsManager(config.skills_pool_dir)
        self.logger = logger
    
    async def run_task(
        self,
        task_name: str,
        model_name: str,
        extra_skills_count: int,
        pass_index: int,
        skills_seed: int,
    ) -> TaskResult:
        """Run a single task execution with non-blocking base image build.
        
        May raise BaseImageBusy if another worker is building the base image.
        """
        return await self._run_task_impl(
            task_name, model_name, extra_skills_count, pass_index, skills_seed,
            blocking=False
        )
    
    async def run_task_blocking(
        self,
        task_name: str,
        model_name: str,
        extra_skills_count: int,
        pass_index: int,
        skills_seed: int,
    ) -> TaskResult:
        """Run a single task execution, waiting for base image if needed.
        
        This version will block and wait if the base image is being built.
        """
        return await self._run_task_impl(
            task_name, model_name, extra_skills_count, pass_index, skills_seed,
            blocking=True
        )
    
    async def _run_task_impl(
        self,
        task_name: str,
        model_name: str,
        extra_skills_count: int,
        pass_index: int,
        skills_seed: int,
        blocking: bool = False,
    ) -> TaskResult:
        """Internal implementation of task execution."""
        
        start_time = time.time()
        task_id = f"{task_name}[m={model_name[:20]},s={extra_skills_count},p={pass_index}]"
        
        task_dir = self.config.tasks_dir / task_name
        if not task_dir.exists():
            return TaskResult(
                task_name=task_name,
                model_name=model_name,
                extra_skills_count=extra_skills_count,
                pass_index=pass_index,
                seed=skills_seed,
                success=False,
                episodes=0,
                input_tokens=0,
                output_tokens=0,
                error=f"Task directory not found: {task_dir}",
                loaded_skills=[],
            )
        
        # Create output directory
        output_dir = (
            self.config.output_dir 
            / task_name 
            / model_name 
            / skills_dir_label(extra_skills_count)
            / f"seed_{skills_seed}"
            / f"pass_{pass_index}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up stale result.json from previous interrupted runs
        # (e.g., file exists but is empty or corrupt)
        stale_result_path = output_dir / "result.json"
        if stale_result_path.exists():
            try:
                with open(stale_result_path) as f:
                    json.load(f)  # Try to parse
            except (json.JSONDecodeError, ValueError):
                # File is empty or corrupt — remove it
                stale_result_path.unlink()
                print(f"[{task_id}] Removed stale/corrupt result.json from previous run")
        
        # Read instruction
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            return TaskResult(
                task_name=task_name,
                model_name=model_name,
                extra_skills_count=extra_skills_count,
                pass_index=pass_index,
                seed=skills_seed,
                success=False,
                episodes=0,
                input_tokens=0,
                output_tokens=0,
                error="No instruction.md found",
                loaded_skills=[],
            )
        
        instruction = instruction_file.read_text()
        
        # Build output path helper for infrastructure failures that happen
        # before the agent/container lifecycle starts (e.g. base image build failure).
        def write_infrastructure_failure_result(error_msg: str, failure_type: str) -> TaskResult:
            execution_time = time.time() - start_time
            result_log = {
                "task_name": task_name,
                "model_name": model_name,
                "extra_skills_count": extra_skills_count,
                "pass_index": pass_index,
                "skills_seed": skills_seed,
                "cached_image": None,
                "base_image": None,
                "runtime_skills_dir": "",
                "preselect_strategy": self.config.preselect_strategy,
                "agent_result": {
                    "success": False,
                    "episodes": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": f"{failure_type}: {error_msg}",
                    "loaded_skills": [],
                    "preselected_skills": [],
                    "preselect_input_tokens": 0,
                    "preselect_output_tokens": 0,
                },
                "verification_result": {
                    "ran": False,
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                },
                "final_success": False,
                "reward": 0.0,
                "execution_time_sec": execution_time,
                "exception": f"{failure_type}: {error_msg}",
                "infrastructure_failure": True,
                "failure_type": failure_type,
            }
            with open(output_dir / "result.json", "w") as f:
                json.dump(result_log, f, indent=2)
            return TaskResult(
                task_name=task_name,
                model_name=model_name,
                extra_skills_count=extra_skills_count,
                pass_index=pass_index,
                seed=skills_seed,
                success=False,
                episodes=0,
                input_tokens=0,
                output_tokens=0,
                error=f"{failure_type}: {error_msg}",
                execution_time_sec=execution_time,
                loaded_skills=[],
                preselect_strategy=self.config.preselect_strategy,
                preselected_skills=[],
                preselect_input_tokens=0,
                preselect_output_tokens=0,
            )
        
        # Prepare skills directory
        # NO_SKILLS (-1): completely empty skills dir — no task skills, no extras.
        # 0: task-specific skills only (no extra random skills).
        # N>0: task-specific skills + N randomly sampled extra skills.
        skills_cache_dir = (
            self.config.output_dir
            / "_image_cache"
            / task_name
            / skills_dir_label(extra_skills_count)
            / f"seed_{skills_seed}"
            / "skills"
        )

        if extra_skills_count == NO_SKILLS:
            # Create an empty skills directory — agent will see "No skills available."
            skills_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            task_skills_dir = task_dir / "environment" / "skills"
            task_skill_names = []
            if task_skills_dir.exists():
                task_skill_names = [p.name for p in task_skills_dir.iterdir() if p.is_dir()]

            extra_skills = self.skills_manager.sample_skills(
                count=extra_skills_count,
                exclude=task_skill_names,
                seed=skills_seed,
            )

            self.skills_manager.copy_skills_to_dir(
                skills=extra_skills,
                dest_dir=skills_cache_dir,
                task_skills_dir=task_skills_dir if self.config.include_task_skills else None,
            )
        
        # Get or build cached base image (shared across skills/model/pass)
        # blocking=False: if base image is being built by another worker,
        # raise BaseImageBusy so this task can be deferred
        try:
            base_image = await self.image_cache.get_or_build_image(
                task_name=task_name,
                skills_count=extra_skills_count,
                seed=skills_seed,
                task_dir=task_dir,
                skills_dir=skills_cache_dir,
                blocking=blocking,
            )
        except BaseImageBuildFailed as e:
            return write_infrastructure_failure_result(str(e), "base_image_build_failed")
        
        # Create agent config
        agent_config = AgentConfig(
            model_name=model_name,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_episodes=self.config.max_episodes,
            # Skill pre-selection settings
            preselect_strategy=self.config.preselect_strategy,
            preselect_max_skills=self.config.preselect_max_skills,
            preselect_model_name=self.config.preselect_model_name,
            preselect_embedding_model=self.config.preselect_embedding_model,
            rag_precomputed_dir=self.config.rag_precomputed_dir,
            hybrid_rag_retrieval_k=self.config.hybrid_rag_retrieval_k,
        )
        
        # Create unique container for this run
        # Each (model, pass) gets its own container from the shared base image.
        container_name = f"agent_{task_name}_{pass_index}_{uuid.uuid4().hex[:8]}"
        
        env = DockerEnvironment(
            image_name=base_image,
            container_name=container_name,
            work_dir=None,  # use Dockerfile's default WORKDIR from the base image
            logs_dir=output_dir,
            max_parallel=self.config.max_parallel_tasks,  # 用于计算 Docker 资源限制
        )
        
        # Run the entire agent lifecycle in a worker thread so that
        # asyncio.gather can truly run multiple tasks in parallel.
        # All operations inside (env.start, LLM calls via requests.post,
        # docker exec, env.stop) are synchronous / blocking.
        def _run_in_thread() -> TaskResult:
            # Per-task logger: writes to {output_dir}/run.log AND propagates
            # to the global experiment.log + console.
            task_logger = setup_task_logging(output_dir, task_id)
            
            container_started = False
            agent = None  # Track agent ref so we can extract loaded_skills on exception
            try:
                task_logger.info(f"[{task_id}] Starting (using base image: {base_image})...")
                env.start()
                container_started = True
                task_logger.info(f"[{task_id}] Container started: {env.container_name}")
                
                # Runtime injection: only selected skills are copied into the
                # container. Task environment files are already placed by the
                # task Dockerfile in the shared base image.
                self._inject_runtime_files_sync(task_dir, skills_cache_dir, env, task_logger)
                
                # Create and run agent (synchronous: LLM + docker exec)
                agent = StandaloneAgent(agent_config, output_dir, logger=task_logger)
                result = agent.run_task_sync(
                    instruction=instruction,
                    environment=env,
                    skill_dirs=["/root/.terminus/skills"],
                    task_name=task_name,
                    extra_skills_count=extra_skills_count,
                )
                
                execution_time = time.time() - start_time
                task_logger.info(f"[{task_id}] Agent finished: success={result.success}, "
                                 f"episodes={result.episodes}, time={execution_time:.1f}s")
                
                # Run verification if agent reports success
                agent_success = result.success
                verification_success = False
                verification_stdout = ""
                verification_stderr = ""
                
                if agent_success:
                    task_logger.info(f"[{task_id}] Running verification...")
                    verification_success, verification_stdout, verification_stderr = \
                        self._run_verification_sync(task_dir, env, output_dir)
                    status = "PASSED" if verification_success else "FAILED"
                    task_logger.info(f"[{task_id}] Verification: {status}")
                else:
                    task_logger.info(f"[{task_id}] Agent failed, skipping verification. "
                                     f"Error: {result.error}")
                
                oom_killed = env.was_oom_killed()
                terminal_error = result.error
                if oom_killed:
                    terminal_error = "container_oom_killed"
                    task_logger.error(
                        f"[{task_id}] Docker recorded an OOM kill during the run; "
                        "marking result retryable"
                    )

                final_success = agent_success and verification_success and not oom_killed
                reward = 1.0 if final_success else 0.0
                
                # Save result log
                result_log = {
                    "task_name": task_name,
                    "model_name": model_name,
                    "extra_skills_count": extra_skills_count,
                    "pass_index": pass_index,
                    "skills_seed": skills_seed,
                    "cached_image": base_image,
                    "base_image": base_image,
                    "runtime_skills_dir": str(skills_cache_dir),
                    "preselect_strategy": result.preselect_strategy,
                    "agent_result": {
                        "success": agent_success,
                        "episodes": result.episodes,
                        "input_tokens": result.total_input_tokens,
                        "output_tokens": result.total_output_tokens,
                        "error": terminal_error,
                        "loaded_skills": result.loaded_skills,
                        "preselected_skills": result.preselected_skills,
                        "preselect_input_tokens": result.preselect_input_tokens,
                        "preselect_output_tokens": result.preselect_output_tokens,
                    },
                    "verification_result": {
                        "ran": agent_success,
                        "success": verification_success,
                        "stdout": verification_stdout[:5000] if verification_stdout else "",
                        "stderr": verification_stderr[:2000] if verification_stderr else "",
                    },
                    "final_success": final_success,
                    "reward": reward,
                    "execution_time_sec": execution_time,
                    "container_oom_killed": oom_killed,
                }
                if oom_killed:
                    result_log.update({
                        "infrastructure_failure": True,
                        "failure_type": "container_oom_killed",
                    })
                
                result_log_path = output_dir / "result.json"
                with open(result_log_path, "w") as f:
                    json.dump(result_log, f, indent=2)
                
                task_logger.info(f"[{task_id}] Result saved: final_success={final_success}")
                
                return TaskResult(
                    task_name=task_name,
                    model_name=model_name,
                    extra_skills_count=extra_skills_count,
                    pass_index=pass_index,
                    seed=skills_seed,
                    success=final_success,
                    episodes=result.episodes,
                    input_tokens=result.total_input_tokens,
                    output_tokens=result.total_output_tokens,
                    error=terminal_error,
                    execution_time_sec=execution_time,
                    loaded_skills=result.loaded_skills,
                    preselect_strategy=result.preselect_strategy,
                    preselected_skills=result.preselected_skills,
                    preselect_input_tokens=result.preselect_input_tokens,
                    preselect_output_tokens=result.preselect_output_tokens,
                )
            except Exception as e:
                # ── CRITICAL: Always report errors clearly ──
                execution_time = time.time() - start_time
                error_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                
                # Try to recover loaded_skills from the agent if it was created
                # (the agent may have loaded some skills before the exception)
                partial_loaded_skills: List[Dict[str, Any]] = []
                partial_episodes = 0
                partial_input_tokens = 0
                partial_output_tokens = 0
                partial_preselected_skills: List[str] = []
                partial_preselect_input_tokens = 0
                partial_preselect_output_tokens = 0
                if agent is not None:
                    try:
                        partial_loaded_skills = getattr(agent, '_loaded_skills_log', [])
                        # The LLM client tracks cumulative token usage
                        partial_input_tokens = agent.llm.total_input_tokens
                        partial_output_tokens = agent.llm.total_output_tokens
                        # Recover pre-selection info
                        partial_preselected_skills = getattr(agent, '_preselected_skills', [])
                        partial_preselect_input_tokens = getattr(agent, '_preselect_input_tokens', 0)
                        partial_preselect_output_tokens = getattr(agent, '_preselect_output_tokens', 0)
                    except Exception:
                        pass  # Best-effort extraction
                
                # Log the full error — this is the key: errors must be visible
                task_logger.error(f"[{task_id}] *** TASK FAILED WITH EXCEPTION ***")
                task_logger.error(f"[{task_id}] {error_msg}")
                task_logger.error(f"[{task_id}] Traceback:\n{tb}")
                
                # Save error to result.json so it's never silently lost
                error_result_log = {
                    "task_name": task_name,
                    "model_name": model_name,
                    "extra_skills_count": extra_skills_count,
                    "pass_index": pass_index,
                    "skills_seed": skills_seed,
                    "cached_image": base_image,
                    "base_image": base_image,
                    "runtime_skills_dir": str(skills_cache_dir),
                    "preselect_strategy": self.config.preselect_strategy,
                    "agent_result": {
                        "success": False,
                        "episodes": partial_episodes,
                        "input_tokens": partial_input_tokens,
                        "output_tokens": partial_output_tokens,
                        "error": error_msg,
                        "loaded_skills": partial_loaded_skills,
                        "preselected_skills": partial_preselected_skills,
                        "preselect_input_tokens": partial_preselect_input_tokens,
                        "preselect_output_tokens": partial_preselect_output_tokens,
                    },
                    "verification_result": {
                        "ran": False,
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                    },
                    "final_success": False,
                    "reward": 0.0,
                    "execution_time_sec": execution_time,
                    "exception": error_msg,
                    "traceback": tb,
                }
                
                error_log_path = output_dir / "result.json"
                with open(error_log_path, "w") as f:
                    json.dump(error_result_log, f, indent=2)
                
                # Return a failed TaskResult — program continues, but error is recorded
                return TaskResult(
                    task_name=task_name,
                    model_name=model_name,
                    extra_skills_count=extra_skills_count,
                    pass_index=pass_index,
                    seed=skills_seed,
                    success=False,
                    episodes=partial_episodes,
                    input_tokens=partial_input_tokens,
                    output_tokens=partial_output_tokens,
                    error=error_msg,
                    execution_time_sec=execution_time,
                    loaded_skills=partial_loaded_skills,
                    preselect_strategy=self.config.preselect_strategy,
                    preselected_skills=partial_preselected_skills,
                    preselect_input_tokens=partial_preselect_input_tokens,
                    preselect_output_tokens=partial_preselect_output_tokens,
                )
            finally:
                if container_started:
                    task_logger.info(f"[{task_id}] Cleaning up container...")
                    try:
                        env.stop()
                    except Exception as stop_err:
                        task_logger.error(f"[{task_id}] Error during container cleanup: {stop_err}")
                    task_logger.info(f"[{task_id}] Done.")
        
        try:
            return await asyncio.to_thread(_run_in_thread)
        finally:
            await self.image_cache.release_base_image(task_name)
    
    def _inject_runtime_files_sync(
        self,
        task_dir: Path,
        skills_dir: Path,
        env: DockerEnvironmentBase,
        logger: logging.Logger,
    ) -> None:
        """Copy selected skills into a running container.

        Task environment files are intentionally NOT copied here. They are part
        of the task base image and should remain wherever the Dockerfile placed
        them. This keeps the runtime container consistent with the direct
        Dockerfile build while avoiding per-skills image commits.
        """
        _ = task_dir  # kept for call-site clarity and future use
        env.exec("mkdir -p /root/.terminus/skills", timeout_sec=10)
        if skills_dir.exists() and any(skills_dir.iterdir()):
            logger.info(f"[RuntimeInject] Copying skills from {skills_dir}")
            env.copy_to_container(skills_dir, "/root/.terminus/skills/")
        else:
            logger.info("[RuntimeInject] No skills to copy; using empty skills directory")
    
    def _run_verification_sync(
        self, 
        task_dir: Path, 
        env: DockerEnvironmentBase, 
        output_dir: Path
    ) -> Tuple[bool, str, str]:
        """Run task verification tests (synchronous, called from worker thread)."""
        tests_dir = task_dir / "tests"
        if not tests_dir.exists():
            return True, "No tests directory found, assuming success", ""
        
        test_script = tests_dir / "test.sh"
        if not test_script.exists():
            return False, "", f"test.sh not found in {tests_dir}"
        
        # Copy tests to container
        env.copy_to_container(tests_dir, "/tests/")
        env.exec("mkdir -p /logs/verifier", timeout_sec=10)

        # Many SkillsBench verifier scripts install uv from a GitHub release on
        # every run. On a throttled connection that download can consume the
        # entire verifier timeout and turn a valid agent run into an
        # infrastructure-only failure. Keep the pinned uv version and verifier
        # semantics, but use the configured pip mirror when uv is absent. The
        # replacement touches only the copied runtime script.
        uv_bootstrap_patch = (
            "sed -i -e "
            "'s#^curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh$#"
            "command -v uvx >/dev/null 2>\\&1 || "
            "python3 -m pip install --break-system-packages "
            "--disable-pip-version-check -q "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple "
            "uv==0.9.7#' -e "
            "'/^source .*HOME.*\\.local\\/bin\\/env/ "
            "s#^#[ ! -f \"$HOME/.local/bin/env\" ] || #' "
            "/tests/test.sh"
        )
        patch_exit, _, patch_stderr = env.exec(uv_bootstrap_patch, timeout_sec=10)
        if patch_exit != 0:
            self.logger.warning(
                "[Verification] Could not patch the runtime uv bootstrap; "
                f"using the task's original installer: {patch_stderr[:300]}"
            )
        
        # Run tests
        env.exec("chmod +x /tests/test.sh", timeout_sec=10)
        # Some verifiers create a fresh uvx environment with pandas/scipy.
        # On Docker Desktop ARM64, downloading and unpacking those wheels can
        # remain actively progressing beyond ten minutes. Give the complete
        # verifier fifteen minutes; env.exec still kills the full process
        # group on expiry, so this does not permit stale installers.
        exit_code, stdout, stderr = env.exec("/tests/test.sh", timeout_sec=900)
        
        # Read reward
        _, reward_stdout, _ = env.exec(
            "cat /logs/verifier/reward.txt 2>/dev/null || echo -1",
            timeout_sec=10,
        )
        
        try:
            reward = float(reward_stdout.strip())
            success = reward > 0
        except (ValueError, TypeError):
            success = exit_code == 0
        
        # Save verification log
        verification_log = output_dir / "verification.txt"
        with open(verification_log, "w") as f:
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"Success: {success}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)

        # A process-group timeout is an infrastructure outcome, not evidence
        # that the agent's answer failed the task tests. Persist a resume
        # marker beside result.json so the normal resume classifier archives
        # and reruns this pass even though verification itself returned False.
        if exit_code == 124:
            retry_marker = output_dir / "retry_required.json"
            retry_marker.write_text(json.dumps({
                "reason": "verifier_timeout_exit_124",
                "retry_required": True,
                "detail": (
                    "The verifier exceeded its process-group deadline; "
                    "this is an infrastructure timeout, not a task verdict."
                ),
            }, indent=2) + "\n")
        
        return success, stdout, stderr


# =============================================================================
# Parallel Experiment Runner - Main API
# =============================================================================

class ParallelExperimentRunner:
    """
    Main API for running parallel skills scaling experiments.
    
    Supports multiple models: Docker images are cached by (task, skills_count, seed)
    and shared across all models and passes.
    
    Usage:
        config = ParallelExperimentConfig(
            model_names=["claude-sonnet-4-6", "gpt-4o", "gemini-2.5-pro"],
            api_base="https://api.example.com",
            api_key="your-key",
            task_names=["3d-scan-calc", "another-task"],
            tasks_dir=Path("../skillsbench/tasks"),
            extra_skills_counts=[0, 10, 100],
            num_passes=1,
            max_parallel_tasks=4,
            skills_pool_dir=Path(".agents/skills"),
        )
        
        runner = ParallelExperimentRunner(config)
        results = await runner.run_all()
        runner.print_summary(results)
    """
    
    def __init__(self, config: ParallelExperimentConfig):
        self.config = config
        self.logger = setup_experiment_logging(config.output_dir)
        self.image_cache = DockerImageCache(
            max_cached_images=config.max_cached_images,
            strategy=config.preselect_strategy,
            max_parallel=config.max_parallel_tasks,  # 用于计算 Docker 资源限制
        )
        self.task_runner = ParallelTaskRunner(config, self.image_cache, self.logger)
        self.results: List[TaskResult] = []
    
    async def run_all(self, cleanup_images: bool = True, resume: bool = False) -> Dict[Tuple[str, int], AggregatedResult]:
        """
        Run all experiments in parallel and return aggregated results.
        
        Docker images are cached by (task, skills_count, seed) and shared across
        all models and all passes. This means:
        - Different models running the same task+skills+seed use the same base image
        - Different passes of the same task+skills+seed use the same base image
        
        Args:
            cleanup_images: Whether to clean up cached Docker images after completion
            resume: If True, skip terminal results already present in the output
                    directory. Corrupt, infrastructure-failed, or exception-
                    interrupted runs are archived and re-queued automatically.
            
        Returns:
            Dictionary mapping (model_name, skills_count) to AggregatedResult
        """
        # Generate all task configurations: task × skills × pass × model
        # 
        # Ordering strategy: round-robin interleave by (task, skills, pass, model).
        # The outer loop is task × skills_count, so all runs that share the
        # same Docker image (same task + skills_count + seed) are grouped
        # together. Within each group, passes and models are interleaved.
        # This maximises Docker cache hits (the image stays warm while all
        # models/passes that need it run) while spreading API load across
        # models evenly (models rotate in the inner loop).
        #
        # skills_seed is fixed (determines which extra skills are sampled).
        # Different passes are independent re-runs of the same environment,
        # so they all share the same Docker image.
        skills_seed = self.config.random_seed
        
        all_tasks = []  # (model_name, task_name, skills_count, pass_idx, skills_seed)
        for task_name in self.config.task_names:
            for skills_count in self.config.extra_skills_counts:
                for pass_idx in range(self.config.num_passes):
                    for model_name in self.config.model_names:
                        all_tasks.append((model_name, task_name, skills_count, pass_idx, skills_seed))
        
        # ── Resume: scan for already-completed tasks ──
        completed_results: List[TaskResult] = []
        tasks_to_run = []
        
        if resume:
            for task_tuple in all_tasks:
                model_name, task_name, skills_count, pass_idx, s_seed = task_tuple
                result_path = (
                    self.config.output_dir / task_name / model_name
                    / skills_dir_label(skills_count) / f"seed_{s_seed}" / f"pass_{pass_idx}"
                    / "result.json"
                )
                if result_path.exists():
                    # Load only terminal results. Retryable/interrupted results
                    # return None and are archived before the task is re-queued,
                    # preserving every failed run log without letting a stale
                    # result.json block resume.
                    loaded = self._load_completed_result(result_path, task_tuple)
                    if loaded is not None:
                        completed_results.append(loaded)
                        continue
                    self._archive_retryable_result(result_path, task_tuple)
                # Not completed — needs to run
                tasks_to_run.append(task_tuple)
            
            n_skipped = len(completed_results)
            if n_skipped > 0:
                self.logger.info(f"[Resume] Found {n_skipped} completed tasks, skipping them.")
                self.logger.info(f"[Resume] {len(tasks_to_run)} tasks remaining to run.")
        else:
            tasks_to_run = all_tasks
        
        # NOTE: tasks_to_run is already ordered by (task, skills, pass, model)
        # which gives an optimal round-robin interleave:
        #   - Runs sharing the same task base image are grouped together,
        #     maximizing base-image cache hits while skills are injected at runtime.
        #   - Models rotate in the inner loop, so no single model's API
        #     receives a burst of requests.
        # No shuffle is needed.
        
        total_tasks = len(all_tasks)
        tasks_remaining = len(tasks_to_run)
        
        # Count unique base images needed for remaining tasks
        unique_base_images = set()
        for _, task_name, _, _, _ in tasks_to_run:
            unique_base_images.add(task_name)
        
        n_models = len(self.config.model_names)
        n_passes = self.config.num_passes
        
        self.logger.info(f"Total configurations: {total_tasks} "
                         f"({len(completed_results)} completed, {tasks_remaining} to run)")
        self.logger.info(f"Running {tasks_remaining} tasks with max {self.config.max_parallel_tasks} parallel...")
        self.logger.info(f"Models: {self.config.model_names}")
        self.logger.info(f"Tasks: {len(self.config.task_names)} tasks")
        self.logger.info(f"Skills counts: {self.config.extra_skills_counts}")
        self.logger.info(f"Passes per config: {self.config.num_passes}")
        self.logger.info(f"Skills seed: {skills_seed} (fixed across all passes)")
        if unique_base_images:
            self.logger.info(f"Unique Docker base images to build/reuse: {len(unique_base_images)}")
        self.logger.info(f"Docker base image cache limit: {self.config.max_cached_images} (LRU eviction)")
        self.logger.info(f"Task ordering: round-robin (task × skills × pass × model)")
        self.logger.info("=" * 60)
        
        try:
            # Run remaining tasks (may be empty if all resumed)
            if tasks_to_run:
                # ── Worker-pool approach ──
                # Use an asyncio.Queue + N workers so that whenever a worker
                # finishes one task it immediately picks up the next one from
                # the queue.  This is *truly* replenishing — unlike the old
                # asyncio.gather+Semaphore approach which assigned semaphore
                # slots in strict FIFO list-order, meaning all tasks of the
                # first Docker-image group had to finish before ANY task from
                # the next group could start.
                #
                # Non-blocking base image builds:
                # When a worker encounters a task whose base image is being built
                # by another worker, it receives a BaseImageBusy exception and
                # puts the task back at the end of the queue, then processes
                # other tasks first. This prevents all workers from blocking on
                # a single slow base image build.
                task_queue: asyncio.Queue = asyncio.Queue()
                # results_by_index preserves original order for determinism
                results_by_index: Dict[int, TaskResult] = {}
                # Track deferred tasks to avoid infinite loops
                deferred_counts: Dict[int, int] = {}
                MAX_DEFER_COUNT = 100  # Safety limit to prevent infinite deferral
                
                for idx, task_tuple in enumerate(tasks_to_run):
                    await task_queue.put((idx, task_tuple))
                    deferred_counts[idx] = 0
                
                async def worker(worker_id: int):
                    while True:
                        try:
                            idx, task_tuple = task_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        model_name, task_name, skills_count, pass_idx, s_seed = task_tuple
                        try:
                            result = await self.task_runner.run_task(
                                task_name=task_name,
                                model_name=model_name,
                                extra_skills_count=skills_count,
                                pass_index=pass_idx,
                                skills_seed=s_seed,
                            )
                            results_by_index[idx] = result
                        except BaseImageBusy as busy_exc:
                            # Base image is being built by another worker.
                            # Put this task back at the end of the queue and process others.
                            deferred_counts[idx] += 1
                            if deferred_counts[idx] <= MAX_DEFER_COUNT:
                                self.logger.debug(
                                    f"Worker {worker_id}: Task {task_name} deferred "
                                    f"(base image busy, defer #{deferred_counts[idx]})"
                                )
                                # Put task back at the end of the queue
                                await task_queue.put((idx, task_tuple))
                                # Progressive delay: start short, increase over time
                                delay = min(2.0, 0.5 + deferred_counts[idx] * 0.1)
                                await asyncio.sleep(delay)
                            else:
                                # Too many deferrals — run with blocking=True to wait for the base image
                                self.logger.info(
                                    f"Worker {worker_id}: Task {task_name} exceeded {MAX_DEFER_COUNT} deferrals, "
                                    f"switching to blocking mode"
                                )
                                try:
                                    # Call run_task_blocking which will wait for the base image
                                    result = await self.task_runner.run_task_blocking(
                                        task_name=task_name,
                                        model_name=model_name,
                                        extra_skills_count=skills_count,
                                        pass_index=pass_idx,
                                        skills_seed=s_seed,
                                    )
                                    results_by_index[idx] = result
                                except BaseException as exc2:
                                    error_msg = f"{type(exc2).__name__}: {exc2}"
                                    self.logger.error(
                                        f"Worker {worker_id}: Task {task_name} failed even in blocking mode: {error_msg}"
                                    )
                                    results_by_index[idx] = TaskResult(
                                        task_name=task_name,
                                        model_name=model_name,
                                        extra_skills_count=skills_count,
                                        pass_index=pass_idx,
                                        seed=s_seed,
                                        success=False,
                                        episodes=0,
                                        input_tokens=0,
                                        output_tokens=0,
                                        error=error_msg,
                                        loaded_skills=[],
                                    )
                        except BaseException as exc:
                            error_msg = f"{type(exc).__name__}: {exc}"
                            self.logger.error(
                                f"*** Unhandled exception in task {task_name} "
                                f"[m={model_name},s={skills_count},p={pass_idx}]: {error_msg}"
                            )
                            self.logger.error(
                                f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                            )
                            results_by_index[idx] = TaskResult(
                                task_name=task_name,
                                model_name=model_name,
                                extra_skills_count=skills_count,
                                pass_index=pass_idx,
                                seed=s_seed,
                                success=False,
                                episodes=0,
                                input_tokens=0,
                                output_tokens=0,
                                error=error_msg,
                                loaded_skills=[],
                            )
                        finally:
                            task_queue.task_done()
                
                # Launch N workers (= max_parallel_tasks)
                workers = [
                    asyncio.create_task(worker(i))
                    for i in range(self.config.max_parallel_tasks)
                ]
                await asyncio.gather(*workers)
                
                # Collect results in original order
                new_results = [results_by_index[i] for i in range(len(tasks_to_run))]
            else:
                new_results = []
                self.logger.info("All tasks already completed — nothing to run.")
            
            # Merge resumed results + new results
            self.results = completed_results + new_results
            
            # Print summary of newly completed tasks
            self.logger.info("\n" + "=" * 60)
            if completed_results:
                n_resumed_ok = sum(1 for r in completed_results if r.success)
                self.logger.info(f"[Resume] {len(completed_results)} previously completed "
                                 f"({n_resumed_ok} passed, {len(completed_results) - n_resumed_ok} failed)")
            for result in new_results:
                status = "✓" if result.success else "✗"
                error_hint = f"  ERROR: {result.error}" if result.error and not result.success else ""
                skills_hint = ""
                if result.loaded_skills:
                    skill_names = [s["skill"] for s in result.loaded_skills]
                    skills_hint = f"  loaded: {skill_names}"
                self.logger.info(f"[{status}] {result.model_name[:25]:25s} | {result.task_name} "
                                 f"(skills={result.extra_skills_count}, pass={result.pass_index})"
                                 f"{skills_hint}{error_hint}")
            
            aggregated = self._aggregate_results()
            
            # Save structured experiment summary for downstream analysis
            self._save_experiment_summary(aggregated, total_tasks)
            
            return aggregated
            
        finally:
            if cleanup_images:
                await self.image_cache.cleanup_all()
                # Also clean up _image_cache directory
                image_cache_dir = self.config.output_dir / "_image_cache"
                if image_cache_dir.exists():
                    shutil.rmtree(image_cache_dir, ignore_errors=True)
                    self.logger.info("[Cleanup] Removed skills cache directory")
    
    def _load_completed_result(
        self,
        result_path: Path,
        task_tuple: Tuple[str, str, int, int, int],
    ) -> Optional[TaskResult]:
        """
        Load a previously completed TaskResult from result.json.
        
        Returns None when the file is corrupt/unreadable or represents a run
        interrupted by infrastructure/provider/runtime exceptions. A normal
        task-level failure (including verifier failure or max episodes reached)
        is terminal and remains resumable.
        """
        model_name, task_name, skills_count, pass_idx, s_seed = task_tuple
        try:
            with open(result_path) as f:
                data = json.load(f)

            retry_reason = None
            retry_marker = result_path.parent / "retry_required.json"
            if retry_marker.exists():
                try:
                    marker_data = json.loads(retry_marker.read_text())
                    retry_reason = marker_data.get("reason") or "external_retry_marker"
                except Exception:
                    retry_reason = "external_retry_marker"
            elif data.get("infrastructure_failure"):
                retry_reason = data.get("failure_type") or "infrastructure_failure"
            elif data.get("exception"):
                retry_reason = str(data["exception"])
            elif not isinstance(data.get("agent_result"), dict):
                retry_reason = "missing_or_invalid_agent_result"

            if retry_reason:
                self.logger.warning(
                    f"[Resume] Retryable result at {result_path}: {retry_reason}"
                )
                return None
            
            agent_result = data.get("agent_result", {})
            return TaskResult(
                task_name=data.get("task_name", task_name),
                model_name=data.get("model_name", model_name),
                extra_skills_count=data.get("extra_skills_count", skills_count),
                pass_index=data.get("pass_index", pass_idx),
                seed=data.get("skills_seed", s_seed),
                success=data.get("final_success", False),
                episodes=agent_result.get("episodes", 0),
                input_tokens=agent_result.get("input_tokens", 0),
                output_tokens=agent_result.get("output_tokens", 0),
                error=agent_result.get("error"),
                execution_time_sec=data.get("execution_time_sec", 0.0),
                loaded_skills=agent_result.get("loaded_skills", []),
            )
        except Exception as e:
            self.logger.warning(f"[Resume] Failed to load {result_path}: {e}, will re-run")
            return None

    def _archive_retryable_result(
        self,
        result_path: Path,
        task_tuple: Tuple[str, str, int, int, int],
    ) -> Optional[Path]:
        """Move an interrupted pass directory aside before resume overwrites it."""
        model_name, task_name, skills_count, pass_idx, s_seed = task_tuple
        pass_dir = result_path.parent
        if not pass_dir.exists():
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        archive_dir = (
            self.config.output_dir / "_retry_archive" / task_name / model_name
            / skills_dir_label(skills_count) / f"seed_{s_seed}"
            / f"pass_{pass_idx}_{timestamp}_{uuid.uuid4().hex[:8]}"
        )
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(pass_dir), str(archive_dir))
            self.logger.warning(
                f"[Resume] Archived interrupted run to {archive_dir}; will re-run"
            )
            return archive_dir
        except Exception as exc:
            # Never overwrite the original failed-run logs if archival fails.
            raise RuntimeError(
                f"Cannot archive retryable result {pass_dir}: {exc}"
            ) from exc
    
    def _aggregate_results(self) -> Dict[Tuple[str, int], AggregatedResult]:
        """Aggregate results by model and skill count."""
        aggregated: Dict[Tuple[str, int], AggregatedResult] = {}
        
        for result in self.results:
            key = (result.model_name, result.extra_skills_count)
            
            if key not in aggregated:
                aggregated[key] = AggregatedResult(
                    model_name=result.model_name,
                    extra_skills_count=result.extra_skills_count,
                )
            
            agg = aggregated[key]
            
            if result.task_name not in agg.task_results:
                agg.task_results[result.task_name] = []
            agg.task_results[result.task_name].append(result.success)
            
            agg.total_input_tokens += result.input_tokens
            agg.total_output_tokens += result.output_tokens
            agg.total_time_sec += result.execution_time_sec
        
        return aggregated
    
    def _save_experiment_summary(
        self,
        aggregated: Dict[Tuple[str, int], AggregatedResult],
        total_tasks: int,
    ) -> Path:
        """
        Save a structured experiment summary JSON for downstream analysis.
        
        The output is designed for easy consumption when writing papers:
        - Per-task, per-model, per-skills_count pass/fail counts
        - Aggregated pass@k metrics
        - Token usage and timing
        
        Returns the path to the saved file.
        """
        # ── Build per-task breakdown ──
        # Structure: { task_name: { model_name: { skills_count: { ... } } } }
        task_breakdown: Dict[str, Dict[str, Dict[int, Any]]] = {}
        
        for result in self.results:
            task = result.task_name
            model = result.model_name
            sc = result.extra_skills_count
            
            task_breakdown.setdefault(task, {})
            task_breakdown[task].setdefault(model, {})
            
            if sc not in task_breakdown[task][model]:
                task_breakdown[task][model][sc] = {
                    "passes": [],          # list of bool per pass
                    "pass_count": 0,
                    "fail_count": 0,
                    "total_episodes": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_time_sec": 0.0,
                    "errors": [],
                }
            
            entry = task_breakdown[task][model][sc]
            entry["passes"].append(result.success)
            if result.success:
                entry["pass_count"] += 1
            else:
                entry["fail_count"] += 1
            entry["total_episodes"] += result.episodes
            entry["total_input_tokens"] += result.input_tokens
            entry["total_output_tokens"] += result.output_tokens
            entry["total_time_sec"] += result.execution_time_sec
            if result.error:
                entry["errors"].append(result.error)
        
        # ── Build model-level aggregation ──
        model_summary = {}
        for (model, sc), agg in sorted(aggregated.items()):
            model_summary.setdefault(model, {})
            model_summary[model][sc] = {
                "tasks_run": agg.tasks_run,
                "tasks_passed": agg.tasks_passed,
                "pass_at_1": round(agg.pass_at_k(1), 4),
                "pass_at_3": round(agg.pass_at_k(3), 4) if self.config.num_passes >= 3 else None,
                "pass_at_5": round(agg.pass_at_k(5), 4) if self.config.num_passes >= 5 else None,
                "total_input_tokens": agg.total_input_tokens,
                "total_output_tokens": agg.total_output_tokens,
                "total_time_sec": round(agg.total_time_sec, 2),
            }
        
        # ── Assemble final JSON ──
        summary = {
            "experiment_time": datetime.datetime.now().isoformat(),
            "config": {
                "models": self.config.model_names,
                "tasks": self.config.task_names,
                "extra_skills_counts": self.config.extra_skills_counts,
                "num_passes": self.config.num_passes,
                "random_seed": self.config.random_seed,
                "max_parallel_tasks": self.config.max_parallel_tasks,
                "max_episodes": self.config.max_episodes,
                "temperature": self.config.temperature if self.config.temperature is not None else "model_default",
                "top_p": "model_default",
            },
            "overview": {
                "total_runs": total_tasks,
                "total_passed": sum(1 for r in self.results if r.success),
                "total_failed": sum(1 for r in self.results if not r.success),
                "total_input_tokens": sum(r.input_tokens for r in self.results),
                "total_output_tokens": sum(r.output_tokens for r in self.results),
                "total_time_sec": round(sum(r.execution_time_sec for r in self.results), 2),
            },
            "model_summary": model_summary,
            "task_breakdown": task_breakdown,
            "raw_results": [
                {
                    "task": r.task_name,
                    "model": r.model_name,
                    "extra_skills": r.extra_skills_count,
                    "pass_index": r.pass_index,
                    "seed": r.seed,
                    "success": r.success,
                    "episodes": r.episodes,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "time_sec": round(r.execution_time_sec, 2),
                    "error": r.error,
                    "loaded_skills": r.loaded_skills,
                }
                for r in self.results
            ],
        }
        
        # Save
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.config.output_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n📊 Experiment summary saved to: {summary_path}")
        return summary_path
    
    def print_summary(self, aggregated: Dict[Tuple[str, int], AggregatedResult]) -> None:
        """Print experiment summary, grouped by model."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 70)
        
        # Group by model for cleaner output
        models = sorted(set(model for model, _ in aggregated.keys()))
        
        for model in models:
            self.logger.info(f"\n{'─' * 70}")
            self.logger.info(f"Model: {model}")
            self.logger.info(f"{'─' * 70}")
            
            for skills_count in sorted(self.config.extra_skills_counts):
                key = (model, skills_count)
                if key not in aggregated:
                    continue
                agg = aggregated[key]
                
                self.logger.info(f"\n  + {skills_count} extra skills:")
                self.logger.info(f"    Tasks: {agg.tasks_passed}/{agg.tasks_run} passed")
                self.logger.info(f"    Pass@1: {agg.pass_at_k(1):.2%}")
                if self.config.num_passes >= 3:
                    self.logger.info(f"    Pass@3: {agg.pass_at_k(3):.2%}")
                if self.config.num_passes >= 5:
                    self.logger.info(f"    Pass@5: {agg.pass_at_k(5):.2%}")
                self.logger.info(f"    Tokens: {agg.total_input_tokens:,} in / {agg.total_output_tokens:,} out")
                self.logger.info(f"    Time: {agg.total_time_sec:.1f}s")
        
        # Print cross-model comparison table
        if len(models) > 1:
            self._print_comparison_table(aggregated, models)
    
    def _print_comparison_table(
        self, 
        aggregated: Dict[Tuple[str, int], AggregatedResult],
        models: List[str],
    ) -> None:
        """Print a cross-model comparison table."""
        # Use the highest available pass@k for the comparison table
        if self.config.num_passes >= 5:
            display_k = 5
        elif self.config.num_passes >= 3:
            display_k = 3
        else:
            display_k = 1
        
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"CROSS-MODEL COMPARISON (Pass@{display_k})")
        self.logger.info(f"{'=' * 70}")
        
        # Header
        skills_counts = sorted(self.config.extra_skills_counts)
        header = f"{'Model':<30s}"
        for sc in skills_counts:
            header += f" | skills={sc:>4d}"
        self.logger.info(header)
        self.logger.info("-" * len(header))
        
        # Rows
        for model in models:
            row = f"{model:<30s}"
            for sc in skills_counts:
                key = (model, sc)
                if key in aggregated:
                    pass_at_k = aggregated[key].pass_at_k(display_k)
                    row += f" | {pass_at_k:>10.1%}"
                else:
                    row += f" | {'N/A':>10s}"
            self.logger.info(row)
    
    def save_results(self, aggregated: Dict[Tuple[str, int], AggregatedResult]) -> Path:
        """Save results to JSON file and return the path."""
        output = {
            "config": {
                "models": self.config.model_names,
                "tasks": self.config.task_names,
                "skills_counts": self.config.extra_skills_counts,
                "num_passes": self.config.num_passes,
                "seed": self.config.random_seed,
                "max_parallel_tasks": self.config.max_parallel_tasks,
            },
            "results": [],
        }
        
        for (model, skills_count), agg in sorted(aggregated.items()):
            output["results"].append({
                "model": model,
                "extra_skills": skills_count,
                "tasks_run": agg.tasks_run,
                "tasks_passed": agg.tasks_passed,
                "pass_at_1": agg.pass_at_k(1),
                "pass_at_3": agg.pass_at_k(3) if self.config.num_passes >= 3 else None,
                "pass_at_5": agg.pass_at_k(5) if self.config.num_passes >= 5 else None,
                "total_input_tokens": agg.total_input_tokens,
                "total_output_tokens": agg.total_output_tokens,
                "total_time_sec": agg.total_time_sec,
                "task_details": {
                    task: {
                        "passes": results,
                        "success_rate": sum(results) / len(results) if results else 0,
                    }
                    for task, results in agg.task_results.items()
                },
            })
        
        results_file = self.config.output_dir / "experiment_results.json"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(output, indent=2))
        self.logger.info(f"\nResults saved to: {results_file}")
        return results_file


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_experiment(
    model_names: Union[str, List[str]],  # Single model or list of models
    api_base: str,
    api_key: str,
    task_names: List[str],
    tasks_dir: Path,
    skills_counts: List[int] = [0, 10, 100],
    num_passes: int = 1,
    max_parallel: int = 4,
    skills_pool_dir: Optional[Path] = None,
    output_dir: Path = Path("experiment_logs"),
    seed: int = 42,
    cleanup_images: bool = True,
    resume: bool = False,
    max_cached_images: int = DockerImageCache.DEFAULT_MAX_CACHED_IMAGES,
    # Skill pre-selection settings
    preselect_strategy: str = "naive",
    preselect_max_skills: int = 5,
    preselect_model_name: Optional[str] = None,
    preselect_embedding_model: str = "text-embedding-v4",
    rag_precomputed_dir: str = "",
    hybrid_rag_retrieval_k: int = 100,
) -> Dict[Tuple[str, int], AggregatedResult]:
    """
    Convenience function to run a parallel experiment across models.
    
    Docker base images are cached by task_name and shared across all
    skills_counts, models, and passes. Skills/env files are injected into each
    runtime container, so no per-(task, skills_count, seed) image is committed.
    For example, with 3 models × 3 skills_counts × 5 tasks, only 5 base images
    are built, not 15 final images or 45 per-run images.
    
    LRU eviction: Only the most recently used `max_cached_images` base images
    are kept in Docker at any time. When the limit is exceeded, the least
    recently used base image tag is removed. If needed later, it is rebuilt.
    
    Task ordering: Tasks are ordered as (task × skills × pass × model) in a
    round-robin fashion instead of random shuffle. This ensures:
    - Docker base images stay warm while all models/passes that need them run
    - API load is spread evenly across models (models rotate in inner loop)
    
    No-Skill Baseline:
        Pass -1 (or NO_SKILLS) in skills_counts to run an experiment where
        **no skills at all** are loaded — neither task-specific skills nor
        extra random skills. The container's skills directory will be empty,
        so the agent prompt will contain "No skills available." and the
        agent must solve the task purely from its own knowledge.
        Output goes to a ``skills_none/`` subdirectory.
    
    Args:
        model_names: Single model name string or list of model names.
        api_base: API base URL.
        api_key: API key.
        task_names: List of task names to run.
        tasks_dir: Path to tasks directory.
        skills_counts: List of extra skills counts to test. Use -1 (NO_SKILLS) for
            a no-skill baseline where even task-specific skills are excluded.
        num_passes: Number of passes per configuration.
        max_parallel: Maximum parallel task executions.
        skills_pool_dir: Path to skills pool directory.
        output_dir: Path to output directory (suffix will be added for non-naive strategies).
        seed: Random seed.
        cleanup_images: Whether to clean up cached Docker base images after completion.
        resume: If True, skip tasks with existing result.json and resume from
                where the last run left off.
        max_cached_images: Maximum number of Docker base images to keep in cache (LRU).
        preselect_strategy: Skill pre-selection strategy: "naive" (all), "llm", "rag", "hybrid-rag-llm", "multi-query-rag-llm", or "dynamic-rag".
        preselect_max_skills: Maximum skills to select (for llm/rag strategies).
        preselect_model_name: LLM model for "llm" strategy (defaults to first model).
        preselect_embedding_model: Embedding model for "rag" strategy.
    
    Returns:
        Dictionary mapping (model_name, skills_count) to AggregatedResult.
    
    Example:
        # Single model with default (naive) pre-selection
        results = await run_experiment(
            model_names="claude-sonnet-4-6",
            api_base="https://api.example.com",
            api_key="your-key",
            task_names=["3d-scan-calc"],
            tasks_dir=Path("../skillsbench/tasks"),
        )
        
        # LLM pre-selection: logs go to experiment_logs-llm
        results = await run_experiment(
            model_names=["claude-sonnet-4-6", "gpt-4o"],
            api_base="https://api.example.com",
            api_key="your-key",
            task_names=["3d-scan-calc", "data-analysis"],
            tasks_dir=Path("../skillsbench/tasks"),
            preselect_strategy="llm",
            preselect_max_skills=5,
        )
        
        # RAG pre-selection: logs go to experiment_logs-rag
        results = await run_experiment(
            model_names=["gpt-4o"],
            api_base="https://api.example.com",
            api_key="your-key",
            task_names=["3d-scan-calc"],
            tasks_dir=Path("../skillsbench/tasks"),
            preselect_strategy="rag",
            preselect_max_skills=5,
        )

        # Dynamic RAG: no skills are listed initially; the agent can call
        # {"search_skills": "..."} at runtime to retrieve top matching skills.
        results = await run_experiment(
            model_names=["gpt-4o"],
            api_base="https://api.example.com",
            api_key="your-key",
            task_names=["3d-scan-calc"],
            tasks_dir=Path("../skillsbench/tasks"),
            preselect_strategy="dynamic-rag",
            preselect_max_skills=5,
            rag_precomputed_dir="rag_precomputed",
        )
    """
    # Normalize model_names to a list
    if isinstance(model_names, str):
        model_names = [model_names]
    
    # Compute actual output directory with strategy suffix
    # e.g., experiment_logs -> experiment_logs-llm for LLM pre-selection
    actual_output_dir = output_dir
    if preselect_strategy != "naive":
        actual_output_dir = Path(f"{output_dir}-{preselect_strategy}")
    
    config = ParallelExperimentConfig(
        model_names=model_names,
        api_base=api_base,
        api_key=api_key,
        task_names=task_names,
        tasks_dir=tasks_dir,
        extra_skills_counts=skills_counts,
        num_passes=num_passes,
        max_parallel_tasks=max_parallel,
        skills_pool_dir=skills_pool_dir,
        output_dir=actual_output_dir,
        random_seed=seed,
        max_cached_images=max_cached_images,
        # Skill pre-selection settings
        preselect_strategy=preselect_strategy,
        preselect_max_skills=preselect_max_skills,
        preselect_model_name=preselect_model_name,
        preselect_embedding_model=preselect_embedding_model,
        rag_precomputed_dir=rag_precomputed_dir,
        hybrid_rag_retrieval_k=hybrid_rag_retrieval_k,
    )
    
    runner = ParallelExperimentRunner(config)
    results = await runner.run_all(cleanup_images=cleanup_images, resume=resume)
    runner.print_summary(results)
    runner.save_results(results)
    
    return results


# =============================================================================
# Example Usage (not CLI, just for testing)
# =============================================================================

if __name__ == "__main__":
    raise SystemExit("Use python run_study.py --help; automatic example sweeps are disabled.")
