#!/usr/bin/env python3
"""Run distractor metadata masking and plan-first prompting experiments.

Masking replaces selected metadata fields with approximately length-matched
placeholders while preserving golden metadata. Plan-first prompting keeps the
full library visible and asks the agent to plan skill use before acting.
Both use the shared Docker executor, logging, and task verification.

Usage:
  # Mask all distractor metadata
  python3 run_masking_and_planning.py --experiment masked --passes 3 --run --max-parallel 4

  # Prompt the agent to plan skill use
  python3 run_masking_and_planning.py --experiment plan-first --passes 3 --run --max-parallel 4

  # Run full masking and plan-first prompting sequentially
  python3 run_masking_and_planning.py --experiment both --run --max-parallel 4

  # Field-level masking
  python3 run_masking_and_planning.py --experiment field-masks --run --passes 1 --max-parallel 2

  # Mask distractor locations
  python3 run_masking_and_planning.py --experiment mask-location --run --passes 1 --max-parallel 4

  # Mask distractor names and locations
  python3 run_masking_and_planning.py --experiment mask-name-location --run --passes 1 --max-parallel 4

  # Quick test: 1 model, 1 pass, 1 task
  python3 run_masking_and_planning.py --experiment both --run --models glm-5 --passes 1 --tasks adaptive-cruise-control --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parallel_runner import ParallelExperimentRunner, ParallelExperimentConfig
from standalone_agent import StandaloneAgent, SkillMetadata, JSON_PROMPT_TEMPLATE
from tokenplan_client import get_default_tokenplan_api_base, get_default_tokenplan_api_key

# ── Constants ──
ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT.parent / "skillsbench" / "tasks"
SKILLS_POOL_DIR = ROOT / ".agents" / "skills"

SUBSET_TASKS = [
    "adaptive-cruise-control",
    "data-to-d3",
    "earthquake-phase-association",
    "econ-detrending-correlation",
    "enterprise-information-search",
    "manufacturing-equipment-maintenance",
    "react-performance-debugging",
    "reserves-at-risk-calc",
    "simpo-code-reproduction",
    "weighted-gdp-calc",
]

EXTRA_SKILLS_COUNT = 1000
SKILLS_SEED = 42
MODELS = ["glm-5", "kimi-k2.5", "minimax-m2.5"]
PASSES = 1

MASK_TOKEN = "<mask>"
CHARS_PER_MASK = 12  # empirically, 1 <mask> ≈ 3 tokens in GLM-5; 12 chars English ≈ 3 tokens

NUDGE_SENTENCE = (
    "IMPORTANT: Before executing any commands, you MUST first explicitly think about "
    "which of the available skills (if any) are relevant to this task. State your "
    "reasoning about skill relevance, then load the skills you need. Only after that "
    "should you begin solving the task."
)

# Modified prompt template for nudge experiment
PROMPT_TEMPLATE_NUDGE = JSON_PROMPT_TEMPLATE.replace(
    "---\n\nTask Description:",
    f"---\n\n{NUDGE_SENTENCE}\n\nTask Description:",
)


# ══════════════════════════════════════════════════════════════════════════════
# Monkey-patch helpers
# ══════════════════════════════════════════════════════════════════════════════

def _length_matched_placeholder(text: str) -> str:
    """Approximate the original character length with <mask> tokens; not exact token matching."""
    n = max(1, round(len(text) / CHARS_PER_MASK))
    return " ".join([MASK_TOKEN] * n)


def _make_masked_build_skills_block(original_method, masked_fields=None, preview_label="masked"):
    """Create a patched skill block masking selected distractor fields.

    ``masked_fields`` defaults to all metadata fields to preserve the original
    coarse masked intervention. Field-level interventions pass one or more metadata
    fields, such as ``{"name"}`` or ``{"name", "location"}``.
    """
    masked_fields = set(masked_fields or {"name", "description", "location"})

    def patched_build_skills_block(self, skill_loader, selected_only=False):
        """Masking patch: replace distractor metadata with <mask> tokens."""
        if selected_only:
            skills_metadata = self._skills_metadata
        else:
            skills_metadata = skill_loader.get_metadata()

        if not skills_metadata:
            return "No skills available."

        # Identify golden skills (those from the task's own environment/skills dir)
        # Golden skills have locations under /root/.terminus/skills/ matching
        # the task's environment/skills subdirectories.
        # We detect them by checking which skills were in the task's own skills dir.
        golden_names = set(getattr(self, '_ablation_golden_names', set()))

        skills_list = []
        for skill in skills_metadata:
            if skill.name in golden_names:
                # Keep golden skill as-is
                skills_list.append({
                    "name": skill.name,
                    "description": skill.description,
                    "location": skill.location,
                })
            else:
                # Replace only the requested fields with length-matched masks.
                skills_list.append({
                    "name": _length_matched_placeholder(skill.name)
                    if "name" in masked_fields else skill.name,
                    "description": _length_matched_placeholder(skill.description)
                    if "description" in masked_fields else skill.description,
                    "location": _length_matched_placeholder(skill.location)
                    if "location" in masked_fields else skill.location,
                })

        skills_section = "available_skills:\n" + json.dumps(skills_list, indent=2, ensure_ascii=False)
        active_skills = self._format_loaded_skills()

        block = f"""{skills_section}

To load a skill, use: cat <location> (where location is from the skill's <location> tag above)

LOADED SKILLS:
{active_skills}
"""
        # Persist the exact first skills block sent to the LLM.  Subsequent
        # episodes include loaded-skill state, so keep the initial block only.
        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir:
            preview_path = Path(logs_dir) / "skills_context_preview.txt"
            if not preview_path.exists():
                header = (
                    f"experiment={preview_label}\n"
                    f"masked_fields={','.join(sorted(masked_fields))}\n"
                    f"golden_skills={','.join(sorted(golden_names))}\n\n"
                )
                preview_path.write_text(header + block, encoding="utf-8")
        return block

    return patched_build_skills_block


def _get_golden_skill_names(task_name: str) -> set:
    """Get the set of golden skill names for a task."""
    skills_dir = TASKS_DIR / task_name / "environment" / "skills"
    names = set()
    if not skills_dir.exists():
        return names
    for d in sorted(skills_dir.iterdir()):
        skill_md = d / "SKILL.md"
        if not d.is_dir() or not skill_md.exists():
            continue
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^name:\s*(.+)", content, re.MULTILINE)
        names.add(m.group(1).strip().strip('"').strip("'") if m else d.name)
    return names


# ══════════════════════════════════════════════════════════════════════════════
# Patched runner setup
# ══════════════════════════════════════════════════════════════════════════════

def _create_patched_runner(config: ParallelExperimentConfig, experiment: str):
    """Create a ParallelExperimentRunner that patches agents for the given experiment."""
    runner = ParallelExperimentRunner(config)

    # Monkey-patch StandaloneAgent.run_task_sync to apply experiment changes
    original_run_task_sync = StandaloneAgent.run_task_sync

    def patched_run_task_sync(self, instruction, environment, skill_dirs,
                              task_name=None, extra_skills_count=None):
        # Apply experiment patches before the agent runs
        masked_fields_by_experiment = {
            "masked": {"name", "description", "location"},
            "mask-name": {"name"},
            "mask-description": {"description"},
            "mask-location": {"location"},
            "mask-name-location": {"name", "location"},
        }
        if experiment in masked_fields_by_experiment and task_name:
            import types
            self._ablation_golden_names = _get_golden_skill_names(task_name)
            patched = _make_masked_build_skills_block(
                self._build_skills_block,
                masked_fields=masked_fields_by_experiment[experiment],
                preview_label=experiment,
            )
            self._build_skills_block = types.MethodType(patched, self)
        elif experiment == "nudge":
            self.prompt_template = PROMPT_TEMPLATE_NUDGE

        return original_run_task_sync(self, instruction, environment, skill_dirs,
                                      task_name=task_name,
                                      extra_skills_count=extra_skills_count)

    # Apply the monkey patch
    StandaloneAgent.run_task_sync = patched_run_task_sync

    return runner, original_run_task_sync


def run_experiment(experiment: str, models: List[str], passes: int,
                   tasks: List[str], max_parallel: int, max_episodes: int,
                   output_dir: Optional[Path] = None):
    """Run one ablation experiment using ParallelExperimentRunner infrastructure."""
    output_dir = output_dir or ROOT / f"experiment_logs-ablation-{experiment}"

    api_key = os.environ.get("API_KEY") or get_default_tokenplan_api_key()
    if not api_key:
        raise SystemExit(
            "Tencent TokenPlan key is missing. Set TENCENT_TOKENPLAN_KEY "
            "in .env or the environment before using --run."
        )

    config = ParallelExperimentConfig(
        model_names=models,
        api_base=os.environ.get("API_BASE") or get_default_tokenplan_api_base(),
        api_key=api_key,
        task_names=tasks,
        tasks_dir=TASKS_DIR,
        extra_skills_counts=[EXTRA_SKILLS_COUNT],
        num_passes=passes,
        random_seed=SKILLS_SEED,
        max_parallel_tasks=max_parallel,
        max_episodes=max_episodes,
        temperature=None,
        preselect_strategy="naive",  # No preselection — all skills in context
        output_dir=output_dir,
        skills_pool_dir=SKILLS_POOL_DIR,
        include_task_skills=True,
    )

    print(f"\n{'='*70}")
    print(f"ABLATION EXPERIMENT: {experiment.upper()}")
    print(f"{'='*70}")
    print(f"  Models: {models}")
    print(f"  Tasks: {len(tasks)} tasks")
    print(f"  Passes: {passes}")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Extra skills: {EXTRA_SKILLS_COUNT}")
    print(f"  Output: {output_dir}")
    print(f"  Total runs: {len(tasks) * len(models) * passes}")
    print(f"  Provider: TokenPlan ({config.api_base})")
    print(f"{'='*70}\n")

    # Create runner with monkey-patched agent
    runner, original_run_task_sync = _create_patched_runner(config, experiment)

    try:
        import asyncio
        asyncio.run(runner.run_all(resume=True, cleanup_images=False))
    finally:
        # Restore original method
        StandaloneAgent.run_task_sync = original_run_task_sync


def main():
    parser = argparse.ArgumentParser(
        description="Mask distractor metadata or prompt the agent to plan skill use"
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "masked", "plan-first", "nudge", "both", "mask-name", "mask-description",
            "mask-location", "mask-name-location", "field-masks",
        ],
                        required=True, help="Which experiment to run")
    parser.add_argument("--run", action="store_true", help="Run the experiment")
    parser.add_argument("--max-parallel", type=int, default=2, help="Max parallel runs (1-4)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to use")
    parser.add_argument("--passes", type=int, default=PASSES, help="Number of passes")
    parser.add_argument(
        "--max-episodes", type=int, default=50,
        help="Maximum agent episodes per run; use a small value for context smoke tests",
    )
    parser.add_argument("--tasks", nargs="+", default=SUBSET_TASKS, help="Tasks to run")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Optional output directory, useful for isolated context smoke tests",
    )
    args = parser.parse_args()

    if not 1 <= args.max_parallel <= 4:
        parser.error("--max-parallel must be between 1 and 4")
    if args.max_episodes < 1:
        parser.error("--max-episodes must be at least 1")

    if not args.run:
        parser.error("Specify --run to execute")

    if args.experiment == "plan-first":
        # Preserve the existing log directory so previous results can be resumed.
        experiments = ["nudge"]
    elif args.experiment == "both":
        experiments = ["masked", "nudge"]
    elif args.experiment == "field-masks":
        experiments = ["mask-name", "mask-description"]
    else:
        experiments = [args.experiment]

    for exp in experiments:
        run_experiment(
            exp, args.models, args.passes, args.tasks, args.max_parallel, args.max_episodes,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
