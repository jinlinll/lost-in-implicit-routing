#!/usr/bin/env python3
"""Analyze skill loading and semantic similarity of non-golden substitutions.

This is intentionally read-only with respect to experiment runs. It reads all
available pass_0/pass_1/pass_2 logs and results, checks loads against the recorded
skill context, and uses the existing precomputed embedding cache to compare loaded
distractors with golden skills.
"""

from __future__ import annotations

import argparse
import ast
import csv
from functools import lru_cache
import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from pool_order import ordered_pool_paths
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from parse_skills_loaded import extract_cat_skills_from_run_log


ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT.parent / "skillsbench" / "tasks"
SKILLSBENCH_ROOT = ROOT.parent / "skillsbench"
POOL_DIR = ROOT / ".agents" / "skills"
LOGS_DIR = ROOT / "experiment_logs"
RAG_DIR = ROOT / "rag_precomputed"
OUTPUT_DIR = ROOT / "analysis" / "skill_substitution"
TEST_DIR = ROOT / "analysis" / "skill_substitution" / "smoke_test"

PRIMARY_MODELS = ["glm-5", "kimi-k2.5", "minimax-m2.5"]
PROPRIETARY_MODELS = ["claude-opus-4-6", "gemini-3.1-pro", "gpt-5.4"]
MODELS = PRIMARY_MODELS + PROPRIETARY_MODELS
TASKS = [
    "adaptive-cruise-control", "data-to-d3", "earthquake-phase-association",
    "econ-detrending-correlation", "enterprise-information-search",
    "manufacturing-equipment-maintenance", "react-performance-debugging",
    "reserves-at-risk-calc", "simpo-code-reproduction", "weighted-gdp-calc",
]  # Diagnostic subset; substitution analysis discovers all logged tasks.
COUNTS = [0, 10, 100, 1000]
SEED = 42


def all_logged_tasks(models: Iterable[str]) -> List[str]:
    """All tasks with at least one full-context pass_0 result for these models."""
    wanted = set(models)
    tasks = []
    for task_dir in sorted(LOGS_DIR.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        if any(
            (task_dir / model / f"skills_{count}" / f"seed_{SEED}" / "pass_0" / "result.json").exists()
            for model in wanted for count in COUNTS
        ):
            tasks.append(task_dir.name)
    return tasks


@dataclass(frozen=True)
class Skill:
    folder: str
    name: str
    description: str
    source: str

    @property
    def aliases(self) -> Set[str]:
        base = {self.folder, self.name}
        return base | {value.replace("-", "_") for value in base} | {value.replace("_", "-") for value in base}


def parse_frontmatter_text(text: str) -> Optional[Tuple[str, str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), -1)
    if end < 0:
        return None
    values: Dict[str, str] = {}
    current: Optional[str] = None
    for line in lines[1:end]:
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            current = key.strip()
            value = value.strip().strip('"').strip("'")
            values[current] = "" if value == "|" else value
        elif current and line.startswith(("  ", "\t")):
            values[current] = (values[current] + " " + line.strip()).strip()
    name, description = values.get("name"), values.get("description")
    return (name, description) if name and description is not None else None


def parse_frontmatter(path: Path) -> Optional[Tuple[str, str]]:
    """Parse the name/description fields exactly as the precompute path does."""
    try:
        return parse_frontmatter_text(path.read_text(errors="replace"))
    except OSError:
        return None


@lru_cache(maxsize=None)
def historical_frontmatter(relative_path: Path) -> Optional[Tuple[str, str]]:
    """Recover metadata that existed when historical trajectories were run."""
    repo_path = relative_path.as_posix()
    try:
        commits = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--", repo_path],
            cwd=SKILLSBENCH_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for commit in commits:
            shown = subprocess.run(
                ["git", "show", f"{commit}:{repo_path}"],
                cwd=SKILLSBENCH_ROOT,
                capture_output=True,
                text=True,
            )
            if shown.returncode == 0:
                parsed = parse_frontmatter_text(shown.stdout)
                if parsed:
                    return parsed
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


@lru_cache(maxsize=None)
def golden_skills(task: str) -> List[Skill]:
    base = TASKS_DIR / task / "environment" / "skills"
    result: List[Skill] = []
    if not base.exists():
        return result
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_frontmatter(folder / "SKILL.md")
        if not parsed:
            parsed = historical_frontmatter(
                Path("tasks") / task / "environment" / "skills" / folder.name / "SKILL.md"
            )
        if parsed:
            result.append(Skill(folder.name, parsed[0], parsed[1], "golden"))
    return result


def exact_library(task: str, count: int) -> List[Skill]:
    """Mirror SkillsManager.sample_skills, including unsorted iterdir order."""
    gold = golden_skills(task)
    excluded = {
        p.name
        for p in (TASKS_DIR / task / "environment" / "skills").iterdir()
        if p.is_dir()
    }
    available = [
        p for p in ordered_pool_paths(POOL_DIR)
        if p.is_dir() and (p / "SKILL.md").exists() and p.name not in excluded
    ]
    rng = random.Random(SEED)
    sampled = rng.sample(available, min(count, len(available))) if count else []
    extra: List[Skill] = []
    for folder in sampled:
        parsed = parse_frontmatter(folder / "SKILL.md")
        if parsed:
            extra.append(Skill(folder.name, parsed[0], parsed[1], "distractor"))
    return gold + extra


def loaded_names(pass_dir: Path) -> Set[str]:
    strict_path = pass_dir / "skills_loaded.strict-v3.json"
    if strict_path.exists():
        try:
            strict = json.loads(strict_path.read_text())
            if strict.get("parser_version") == "strict-v3-context-checked":
                return {name for name in strict.get("skill_names", []) if isinstance(name, str) and name}
        except (OSError, json.JSONDecodeError):
            pass
    loaded: Set[str] = set()
    result_path = pass_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            for entry in data.get("agent_result", {}).get("loaded_skills", []):
                value = entry.get("skill", "") if isinstance(entry, dict) else entry
                if isinstance(value, str) and value:
                    loaded.add(value)
        except (json.JSONDecodeError, OSError):
            pass
    run_log = pass_dir / "run.log"
    if run_log.exists():
        # Count direct `cat .../SKILL.md` reads only when the corresponding
        # command really returned skill content.  The historical helper counted
        # attempted reads even when the file did not exist (exit=1 or output=0).
        pending: Dict[str, List[str]] = {}
        for line in run_log.read_text(errors="replace").splitlines():
            running = re.search(r"\[Exec (\d+/\d+)\] Running: (.*)", line)
            if running:
                pending[running.group(1)] = [
                    match.group(2)
                    for match in re.finditer(
                        r"cat\s+['\"]?((?:[~\w/.:-]+)?/skills/([^/'\"\s]+)/SKILL\.md)['\"]?",
                        running.group(2),
                    )
                ]
                continue
            completed = re.search(
                r"\[Exec (\d+/\d+)\] Completed .*\(exit=(\d+), output=(\d+) chars\)", line
            )
            if completed:
                names = pending.pop(completed.group(1), [])
                if int(completed.group(2)) == 0 and int(completed.group(3)) > 0:
                    loaded.update(names)
    if "*" in loaded:
        loaded.remove("*")
        loaded.update(recorded_skill_names(pass_dir))
    return loaded


def recorded_skill_names(pass_dir: Path) -> List[str]:
    """Read the authoritative library names recorded by the original agent run."""
    run_log = pass_dir / "run.log"
    if not run_log.exists():
        return []
    match = re.search(r"\[Skills\] Found \d+ skills(?: in container)?: (\[.*\])", run_log.read_text(errors="replace"))
    if not match:
        return []
    try:
        value = ast.literal_eval(match.group(1))
        return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []
    except (SyntaxError, ValueError):
        return []


def pool_catalog() -> Dict[str, Skill]:
    catalog: Dict[str, Skill] = {}
    for folder in ordered_pool_paths(POOL_DIR):
        if not folder.is_dir():
            continue
        parsed = parse_frontmatter(folder / "SKILL.md")
        if parsed:
            skill = Skill(folder.name, parsed[0], parsed[1], "distractor")
            for alias in skill.aliases:
                catalog.setdefault(alias, skill)
    return catalog


class Embeddings:
    def __init__(self) -> None:
        index = json.loads((RAG_DIR / "cache" / "embedding_index.json").read_text())
        data = np.load(RAG_DIR / "cache" / "embedding_cache.npz")
        self.vectors = {key: data[array_key] for key, array_key in index.items() if array_key in data}

    @staticmethod
    def key(skill: Skill) -> str:
        text = f"{skill.name.strip()}: {skill.description.strip()}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, skill: Skill) -> Optional[np.ndarray]:
        return self.vectors.get(self.key(skill))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def analyze_one(
    task: str,
    model: str,
    count: int,
    pass_index: int,
    embeddings: Embeddings,
    global_catalog: Optional[Dict[str, Skill]] = None,
) -> Optional[dict]:
    pass_dir = LOGS_DIR / task / model / f"skills_{count}" / f"seed_{SEED}" / f"pass_{pass_index}"
    result_path = pass_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if ((pass_dir / "retry_required.json").exists() or result.get("infrastructure_failure")
            or result.get("exception") or not isinstance(result.get("agent_result"), dict)):
        return None

    all_task_gold = golden_skills(task)
    recorded = recorded_skill_names(pass_dir)
    recorded_set = set(recorded)
    gold = [
        skill for skill in all_task_gold
        if not recorded or any(alias in recorded_set for alias in skill.aliases)
    ]
    alias_map: Dict[str, Skill] = {}
    for alias, skill in (global_catalog or pool_catalog()).items():
        alias_map.setdefault(alias, skill)
    for golden in gold:
        for alias in golden.aliases:
            alias_map[alias] = golden
    raw_loaded = sorted(loaded_names(pass_dir))
    raw_resolved = {name: alias_map.get(name) for name in raw_loaded}
    loaded = [
        name for name in raw_loaded
        if not recorded or name in recorded_set or (
            raw_resolved[name] is not None
            and any(alias in recorded_set for alias in raw_resolved[name].aliases)
        )
    ]
    out_of_context = sorted(set(raw_loaded) - set(loaded))
    resolved = {name: raw_resolved[name] for name in loaded}
    golden_hits = {skill.folder for skill in resolved.values() if skill and skill.source == "golden"}
    missed_gold = [skill for skill in gold if skill.folder not in golden_hits]
    non_golden_details = []
    pairwise_details = []
    for loaded_name, skill in resolved.items():
        if skill and skill.source == "golden":
            continue
        loaded_vector = embeddings.get(skill) if skill else None
        comparisons = []
        for golden in missed_gold:
            golden_vector = embeddings.get(golden)
            similarity = (
                cosine(loaded_vector, golden_vector)
                if loaded_vector is not None and golden_vector is not None
                else None
            )
            pair = {
                "loaded_non_golden": loaded_name,
                "loaded_canonical_name": skill.name if skill else None,
                "loaded_description": skill.description if skill else None,
                "missed_golden_folder": golden.folder,
                "missed_golden_name": golden.name,
                "missed_golden_description": golden.description,
                "cosine_similarity": similarity,
                "embedding_text_definition": "name.strip() + ': ' + description.strip()",
            }
            pairwise_details.append(pair)
            if similarity is not None:
                comparisons.append((similarity, golden))
        best = max(comparisons, key=lambda item: item[0]) if comparisons else None
        non_golden_details.append({
            "loaded": loaded_name,
            "canonical_name": skill.name if skill else None,
            "description": skill.description if skill else None,
            "resolved_metadata": skill is not None,
            "n_missed_golden_compared": len(comparisons),
            "closest_missed_golden": best[1].name if best else None,
            "max_cosine_similarity_to_missed_golden": best[0] if best else None,
        })

    n_loaded = len(loaded)
    n_gold = len(gold)
    return {
        "task": task,
        "model": model,
        "skills_count": count,
        "pass": pass_index,
        "success": bool(result.get("final_success", False)),
        "golden_skills": [{"folder": x.folder, "name": x.name, "description": x.description} for x in gold],
        "task_source_golden_skills_excluded_because_absent_from_recorded_context": [
            skill.name for skill in all_task_gold if skill not in gold
        ],
        "recorded_available_skill_names": recorded,
        "library_size": len(recorded) if recorded else count + len(gold),
        "library_source": "run.log [Skills] Found" if recorded else "expected count fallback",
        "loaded_skills": loaded,
        "out_of_context_loaded_references": out_of_context,
        "loaded_names_present_in_recorded_library": {
            name: name in set(recorded) for name in loaded
        } if recorded else None,
        "unresolved_loaded_skills": sorted(name for name, skill in resolved.items() if skill is None),
        "golden_hits": sorted(golden_hits),
        "missed_golden_skills": [skill.name for skill in missed_gold],
        "n_loaded": n_loaded,
        "zero_load": n_loaded == 0,
        "golden_recall": len(golden_hits) / n_gold if n_gold else 0.0,
        "golden_precision": len(golden_hits) / n_loaded if n_loaded else None,
        "n_loaded_non_golden": len(non_golden_details),
        "loaded_non_golden_similarity": non_golden_details,
        "n_pairwise_loaded_non_golden_x_missed_golden": len(pairwise_details),
        "pairwise_similarity_details": pairwise_details,
    }


def aggregate(rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, int], List[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["skills_count"]), []).append(row)
        scope = "PRIMARY_ALL" if row["model"] in PRIMARY_MODELS else "PROPRIETARY_ALL"
        groups.setdefault((scope, row["skills_count"]), []).append(row)
        groups.setdefault(("ALL_EXISTING", row["skills_count"]), []).append(row)
    output = []
    for (model, count), items in sorted(groups.items()):
        loaded_event_sims = [
            detail["max_cosine_similarity_to_missed_golden"]
            for item in items for detail in item["loaded_non_golden_similarity"]
            if detail["max_cosine_similarity_to_missed_golden"] is not None
        ]
        pairwise_sims = [
            pair["cosine_similarity"]
            for item in items for pair in item["pairwise_similarity_details"]
            if pair["cosine_similarity"] is not None
        ]
        precisions = [item["golden_precision"] for item in items if item["golden_precision"] is not None]
        task_triplets: Dict[Tuple[str, str, int], List[dict]] = {}
        for item in items:
            task_triplets.setdefault((item["task"], item["model"], item["skills_count"]), []).append(item)
        complete_triplets = [group for group in task_triplets.values() if {x["pass"] for x in group} == {0, 1, 2}]
        high_sim_missing_golden_runs_ge_0_9 = sum(
            any(
                detail["max_cosine_similarity_to_missed_golden"] is not None
                and detail["max_cosine_similarity_to_missed_golden"] >= 0.9
                for detail in item["loaded_non_golden_similarity"]
            )
            for item in items
        )
        high_sim_missing_golden_runs_gt_0_8 = sum(
            any(
                detail["max_cosine_similarity_to_missed_golden"] is not None
                and detail["max_cosine_similarity_to_missed_golden"] > 0.8
                for detail in item["loaded_non_golden_similarity"]
            )
            for item in items
        )
        runs_loading_non_golden = sum(item["n_loaded_non_golden"] > 0 for item in items)
        output.append({
            "model": model,
            "skills_count": count,
            "n_runs": len(items),
            "success_rate": sum(item["success"] for item in items) / len(items),
            "n_complete_triplets": len(complete_triplets),
            "pass_at_3": sum(any(x["success"] for x in group) for group in complete_triplets) / len(complete_triplets) if complete_triplets else None,
            "avg_skills_loaded": sum(item["n_loaded"] for item in items) / len(items),
            "zero_load_rate": sum(item["zero_load"] for item in items) / len(items),
            "avg_golden_recall": sum(item["golden_recall"] for item in items) / len(items),
            "avg_golden_precision_nonzero": sum(precisions) / len(precisions) if precisions else None,
            "n_loaded_non_golden_events": sum(item["n_loaded_non_golden"] for item in items),
            "n_runs_loading_non_golden": runs_loading_non_golden,
            "fraction_runs_loading_non_golden": runs_loading_non_golden / len(items),
            "n_loaded_non_golden_events_with_comparison": len(loaded_event_sims),
            "n_pairwise_comparisons": sum(item["n_pairwise_loaded_non_golden_x_missed_golden"] for item in items),
            "n_pairwise_comparisons_with_embeddings": len(pairwise_sims),
            "avg_non_golden_max_similarity_to_missed_golden": sum(loaded_event_sims) / len(loaded_event_sims) if loaded_event_sims else None,
            "high_similarity_rate_ge_0_9": sum(value >= 0.9 for value in loaded_event_sims) / len(loaded_event_sims) if loaded_event_sims else None,
            "sensitivity_rate_ge_0_8": sum(value >= 0.8 for value in loaded_event_sims) / len(loaded_event_sims) if loaded_event_sims else None,
            "n_loaded_non_golden_events_gt_0_8": sum(value > 0.8 for value in loaded_event_sims),
            "n_pairwise_comparisons_gt_0_8": sum(value > 0.8 for value in pairwise_sims),
            "n_runs_with_high_sim_gt_0_8": high_sim_missing_golden_runs_gt_0_8,
            "fraction_runs_with_high_sim_gt_0_8": high_sim_missing_golden_runs_gt_0_8 / len(items),
            "runs_with_high_sim_alternative_but_missing_golden": high_sim_missing_golden_runs_ge_0_9,
            "unresolved_loaded_non_golden_events": sum(len(item["unresolved_loaded_skills"]) for item in items),
        })
    return output


def markdown_table(summary: List[dict]) -> str:
    lines = [
        "# Skill loading and non-golden substitution analysis",
        "",
        "Precision is averaged over non-zero-load runs. Similarity uses the existing RAG vector for full `name: description` text and compares every loaded non-golden skill only with golden skills missed in that same run.",
        "",
        "## Skill-loading behavior",
        "",
        "| model/scope | s | runs | success/run | Pass@3 | avg loaded | zero load | recall | precision* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pct = lambda value: "—" if value is None else f"{100 * value:.1f}%"
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['skills_count']} | {row['n_runs']} | {pct(row['success_rate'])} | {pct(row['pass_at_3'])} | "
            f"{row['avg_skills_loaded']:.2f} | {pct(row['zero_load_rate'])} | {pct(row['avg_golden_recall'])} | "
            f"{pct(row['avg_golden_precision_nonzero'])} |"
        )
    lines.extend([
        "",
        "## Loaded non-golden vs missed-golden similarity",
        "",
        "| model/scope | s | loaded non-golden | runs loading non-golden | fraction runs | comparable events | pairwise comparisons | sim >= .9 | sim >= .8 | events > .8 | pairs > .8 | runs > .8 | run fraction > .8 | runs >= .9 | unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['skills_count']} | {row['n_loaded_non_golden_events']} | "
            f"{row['n_runs_loading_non_golden']} | {pct(row['fraction_runs_loading_non_golden'])} | "
            f"{row['n_loaded_non_golden_events_with_comparison']} | {row['n_pairwise_comparisons_with_embeddings']} | "
            f"{pct(row['high_similarity_rate_ge_0_9'])} | {pct(row['sensitivity_rate_ge_0_8'])} | "
            f"{row['n_loaded_non_golden_events_gt_0_8']} | {row['n_pairwise_comparisons_gt_0_8']} | "
            f"{row['n_runs_with_high_sim_gt_0_8']} | {pct(row['fraction_runs_with_high_sim_gt_0_8'])} | "
            f"{row['runs_with_high_sim_alternative_but_missing_golden']} | {row['unresolved_loaded_non_golden_events']} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(task: str, model: str, count: int) -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    row = analyze_one(task, model, count, 0, Embeddings(), pool_catalog())
    if row is None:
        raise SystemExit("Smoke input run is missing")
    (TEST_DIR / "smoke_exact_inputs_and_result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
    report = [
        "# Skill substitution smoke test",
        "",
        f"- Source run: `{task}/{model}/skills_{count}/seed_42/pass_0`",
        f"- Authoritative recorded library: {row['library_size']} skills ({row['library_source']})",
        f"- Golden skills: {', '.join(x['name'] for x in row['golden_skills'])}",
        f"- Actually loaded: {', '.join(row['loaded_skills']) or '(none)' }",
        f"- Missed golden skills: {', '.join(row['missed_golden_skills']) or '(none)'}",
        f"- Unresolved loaded names: {', '.join(row['unresolved_loaded_skills']) or '(none)'}",
        "- Similarity definition: for every loaded non-golden × missed-golden pair, cosine of the existing full `name: description` RAG vectors",
        "",
        "## Loaded non-golden checks",
        "",
    ]
    for detail in row["loaded_non_golden_similarity"]:
        similarity = detail["max_cosine_similarity_to_missed_golden"]
        report.append(
            f"- `{detail['loaded']}` -> closest missed golden `{detail['closest_missed_golden']}`, "
            f"cosine={similarity:.6f}" if similarity is not None else
            f"- `{detail['loaded']}` -> no comparable cached vector"
        )
    if not row["loaded_non_golden_similarity"]:
        report.append("- No non-golden skill was loaded in this smoke source run.")
    (TEST_DIR / "smoke_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"PASS: exact smoke artifact written to {TEST_DIR}")


def run_full() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = Embeddings()
    catalog = pool_catalog()
    tasks = all_logged_tasks(MODELS)
    rows = [
        row for task in tasks for model in MODELS for count in COUNTS for pass_index in range(3)
        if (row := analyze_one(task, model, count, pass_index, embeddings, catalog)) is not None
    ]
    summary = aggregate(rows)
    pairwise = [
        {
            "task": row["task"],
            "model": row["model"],
            "skills_count": row["skills_count"],
            "pass": row["pass"],
            **pair,
        }
        for row in rows for pair in row["pairwise_similarity_details"]
    ]
    comparable_pairwise = [pair for pair in pairwise if pair["cosine_similarity"] is not None]
    missing_primary = [
        {"task": task, "model": model, "skills_count": count, "pass": pass_index}
        for task in tasks for model in PRIMARY_MODELS for count in COUNTS for pass_index in range(3)
        if not (LOGS_DIR / task / model / f"skills_{count}" / f"seed_{SEED}" / f"pass_{pass_index}" / "result.json").exists()
    ]
    (OUTPUT_DIR / "per_run.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (OUTPUT_DIR / "pairwise_similarity.json").write_text(json.dumps(pairwise, ensure_ascii=False, indent=2))
    write_csv(OUTPUT_DIR / "pairwise_similarity.csv", pairwise)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUTPUT_DIR / "summary.md").write_text(markdown_table(summary))
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    (OUTPUT_DIR / "coverage.json").write_text(json.dumps({
        "tasks": tasks,
        "n_tasks": len(tasks),
        "passes": [0, 1, 2],
        "primary_expected_grid_runs": len(tasks) * len(PRIMARY_MODELS) * len(COUNTS) * 3,
        "primary_observed_runs": sum(row["model"] in PRIMARY_MODELS for row in rows),
        "primary_missing_runs": missing_primary,
        "all_existing_runs": len(rows),
        "loaded_non_golden_x_missed_golden_pairs": len(pairwise),
        "pairs_with_cached_full_name_description_vectors": len(comparable_pairwise),
    }, ensure_ascii=False, indent=2))
    print(markdown_table(summary))
    primary_rows = sum(row["model"] in PRIMARY_MODELS for row in rows)
    proprietary_rows = len(rows) - primary_rows
    print(
        f"Analyzed {len(rows)} existing full-context trajectories across pass_0/pass_1/pass_2 and {len(tasks)} tasks "
        f"(primary={primary_rows}, proprietary validation={proprietary_rows}); artifacts: {OUTPUT_DIR}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--task", default="data-to-d3")
    parser.add_argument("--model", default="minimax-m2.5")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    if not args.smoke and not args.run:
        parser.error("choose --smoke or --run")
    if args.smoke:
        run_smoke(args.task, args.model, args.count)
    if args.run:
        run_full()


if __name__ == "__main__":
    main()
