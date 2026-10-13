#!/usr/bin/env python3
"""
Comprehensive Experiment Analysis
==================================
All-in-one analysis script for the Skills Scaling Study.

Covers:
  A. End-to-end pass rate (strict Pass@1, Pass@3)
  B. Token consumption
  C. Revised reward analysis (partial scores)
  D. Skills hit rate (golden skills recall & precision)
  E. Domain & difficulty breakdown
  F. Failure cause classification
  G. Network-failure list (for rerun)

Data filtering:
  - For claude-opus-4-6, gemini-3.1-pro, gpt-5.4: exclude tasks from
    civ6-adjacency-optimizer to dialogue-parser (inclusive, alphabetical).
  - For minimax-m2.5, kimi-k2.5, glm-5: use all tasks except always-excluded tasks.
  - Always exclude: mhc-layer-impl (requires GPU), fix-visual-stability (verifier timeout),
    python-scala-translation, speaker-diarization-subtitles, and multilingual-video-dubbing
    (Docker base-image build failures in multi-query).

Supports multiple log directories for comparing pre-selection strategies:
  experiment_logs/      → naive (default, all skills in prompt)
  experiment_logs-llm/  → LLM pre-selection
  experiment_logs-rag/  → RAG pre-selection

Usage:
    python analyze_study_metrics.py
    python analyze_study_metrics.py --no-color > report.txt
    python analyze_study_metrics.py --csv-prefix output/analysis

    # Multi-strategy analysis:
    python analyze_study_metrics.py --all-strategies
    python analyze_study_metrics.py --logs-dir experiment_logs experiment_logs-llm
    python analyze_study_metrics.py --strategy llm rag
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import log-based skill parser for on-the-fly fallback
try:
    from parse_skills_loaded import parse_pass_directory as _parse_pass_dir
except ImportError:
    _parse_pass_dir = None

# ─── ANSI Colors ──────────────────────────────────────────────────────
USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, t: str) -> str:
    return f"\033[{code}m{t}\033[0m" if USE_COLOR else t

def green(t): return _c("32", t)
def red(t): return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t): return _c("36", t)
def magenta(t): return _c("35", t)
def bold(t): return _c("1", t)
def dim(t): return _c("2", t)

def rate_color(rate, fmt="{:5.1f}%"):
    s = fmt.format(rate)
    if rate >= 60: return green(s)
    elif rate > 0: return yellow(s)
    else: return red(s)

def hline(ch="═", n=80):
    print(ch * n)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

PROPRIETARY_MODELS = {"claude-opus-4-6", "gemini-3.1-pro", "gpt-5.4"}
OPEN_MODELS = {"minimax-m2.5", "kimi-k2.5", "glm-5"}

# Tasks always excluded from evaluation.
# The last three tasks are excluded because their multi-query runs are dominated
# by Docker base-image build failures, so they are infrastructure failures rather
# than method failures.
ALWAYS_EXCLUDED = {
    "mhc-layer-impl",
    "fix-visual-stability",
    "python-scala-translation",
    "speaker-diarization-subtitles",
    "multilingual-video-dubbing",
}

# Tasks excluded for proprietary models (claude network issue + cost cut)
# Alphabetical range: civ6-adjacency-optimizer .. dialogue-parser
PROPRIETARY_EXCLUDED_RANGE = {
    "civ6-adjacency-optimizer",
    "court-form-filling",
    "crystallographic-wyckoff-position-analysis",
    "dapt-intrusion-detection",
    "data-to-d3",
    "dialogue-parser",
}

# Domain mapping from paper Table 9 (normalizes task.toml category → paper domain)
DOMAIN_MAP = {
    # Cybersecurity
    "azure-bgp-oscillation-route-leak": "Cybersecurity",
    "dapt-intrusion-detection": "Cybersecurity",
    "fix-druid-loophole-cve": "Cybersecurity",
    "fix-erlang-ssh-cve": "Cybersecurity",
    "setup-fuzzing-py": "Cybersecurity",
    "software-dependency-audit": "Cybersecurity",
    "suricata-custom-exfil": "Cybersecurity",
    "syzkaller-ppdev-syzlang": "Cybersecurity",
    # Energy
    "energy-ac-optimal-power-flow": "Energy",
    "energy-market-pricing": "Energy",
    "grid-dispatch-operator": "Energy",
    # Finance
    "econ-detrending-correlation": "Finance",
    "financial-modeling-qa": "Finance",
    "invoice-fraud-detection": "Finance",
    "reserves-at-risk-calc": "Finance",
    "sec-financial-report": "Finance",
    "shock-analysis-demand": "Finance",
    "shock-analysis-supply": "Finance",
    "weighted-gdp-calc": "Finance",
    # Healthcare
    "lab-unit-harmonization": "Healthcare",
    "protein-expression-analysis": "Healthcare",
    # Manufacturing
    "manufacturing-codebook-normalization": "Manufacturing",
    "manufacturing-equipment-maintenance": "Manufacturing",
    "manufacturing-fjsp-optimization": "Manufacturing",
    # Mathematics
    "civ6-adjacency-optimizer": "Mathematics",
    "lean4-proof": "Mathematics",
    # Media & Content
    "3d-scan-calc": "Media & Content",
    "dynamic-object-aware-egomotion": "Media & Content",
    "mario-coin-counting": "Media & Content",
    "multilingual-video-dubbing": "Media & Content",
    "pedestrian-traffic-counting": "Media & Content",
    "pg-essay-to-audiobook": "Media & Content",
    "speaker-diarization-subtitles": "Media & Content",
    "threejs-structure-parser": "Media & Content",
    "threejs-to-obj": "Media & Content",
    "video-filler-word-remover": "Media & Content",
    "video-tutorial-indexer": "Media & Content",
    # Natural Science
    "crystallographic-wyckoff-position-analysis": "Natural Science",
    "earthquake-phase-association": "Natural Science",
    "earthquake-plate-calculation": "Natural Science",
    "exoplanet-detection-period": "Natural Science",
    "find-topk-similiar-chemicals": "Natural Science",
    "flood-risk-analysis": "Natural Science",
    "glm-lake-mendota": "Natural Science",
    "gravitational-wave-detection": "Natural Science",
    "lake-warming-attribution": "Natural Science",
    "mars-clouds-clustering": "Natural Science",
    "quantum-numerical-simulation": "Natural Science",
    "seismic-phase-picking": "Natural Science",
    # Office & White Collar
    "court-form-filling": "Office",
    "exceltable-in-ppt": "Office",
    "jpg-ocr-stat": "Office",
    "latex-formula-extraction": "Office",
    "offer-letter-generator": "Office",
    "organize-messy-files": "Office",
    "paper-anonymizer": "Office",
    "pdf-excel-diff": "Office",
    "powerlifting-coef-calc": "Office",
    "pptx-reference-formatting": "Office",
    "sales-pivot-analysis": "Office",
    "scheduling-email-assistant": "Office",
    "travel-planning": "Office",
    "xlsx-recover-data": "Office",
    # Robotics
    "adaptive-cruise-control": "Robotics",
    "hvac-control": "Robotics",
    "pddl-tpp-planning": "Robotics",
    "r2r-mpc-control": "Robotics",
    "virtualhome-agent-planning": "Robotics",
    # Software Engineering
    "citation-check": "Software Eng.",
    "data-to-d3": "Software Eng.",
    "dialogue-parser": "Software Eng.",
    "enterprise-information-search": "Software Eng.",
    "fix-build-agentops": "Software Eng.",
    "fix-build-google-auto": "Software Eng.",
    "flink-query": "Software Eng.",
    "gh-repo-analytics": "Software Eng.",
    "jax-computing-basics": "Software Eng.",
    "parallel-tfidf-search": "Software Eng.",
    "python-scala-translation": "Software Eng.",
    "react-performance-debugging": "Software Eng.",
    "simpo-code-reproduction": "Software Eng.",
    "spring-boot-jakarta-migration": "Software Eng.",
    "taxonomy-tree-merge": "Software Eng.",
    "trend-anomaly-causal-inference": "Software Eng.",
    # Extras not in paper but present
    "video-silence-remover": "Media & Content",
}

# Difficulty mapping from paper Table 9
DIFFICULTY_MAP = {
    "azure-bgp-oscillation-route-leak": "medium", "dapt-intrusion-detection": "hard",
    "fix-druid-loophole-cve": "hard", "fix-erlang-ssh-cve": "hard",
    "setup-fuzzing-py": "medium", "software-dependency-audit": "medium",
    "suricata-custom-exfil": "medium", "syzkaller-ppdev-syzlang": "medium",
    "energy-ac-optimal-power-flow": "medium", "energy-market-pricing": "hard",
    "grid-dispatch-operator": "medium",
    "econ-detrending-correlation": "medium", "financial-modeling-qa": "hard",
    "invoice-fraud-detection": "hard", "reserves-at-risk-calc": "medium",
    "sec-financial-report": "hard", "shock-analysis-demand": "medium",
    "shock-analysis-supply": "hard", "weighted-gdp-calc": "medium",
    "lab-unit-harmonization": "medium", "protein-expression-analysis": "medium",
    "manufacturing-codebook-normalization": "medium", "manufacturing-equipment-maintenance": "medium",
    "manufacturing-fjsp-optimization": "medium",
    "civ6-adjacency-optimizer": "hard", "lean4-proof": "medium",
    "3d-scan-calc": "hard", "dynamic-object-aware-egomotion": "medium",
    "mario-coin-counting": "medium", "multilingual-video-dubbing": "medium",
    "pedestrian-traffic-counting": "hard", "pg-essay-to-audiobook": "medium",
    "speaker-diarization-subtitles": "hard", "threejs-structure-parser": "medium",
    "threejs-to-obj": "medium", "video-filler-word-remover": "medium",
    "video-tutorial-indexer": "hard",
    "crystallographic-wyckoff-position-analysis": "medium",
    "earthquake-phase-association": "hard", "earthquake-plate-calculation": "medium",
    "exoplanet-detection-period": "medium", "find-topk-similiar-chemicals": "medium",
    "flood-risk-analysis": "medium", "glm-lake-mendota": "hard",
    "gravitational-wave-detection": "medium", "lake-warming-attribution": "medium",
    "mars-clouds-clustering": "hard", "quantum-numerical-simulation": "medium",
    "seismic-phase-picking": "hard",
    "court-form-filling": "easy", "exceltable-in-ppt": "medium",
    "jpg-ocr-stat": "hard", "latex-formula-extraction": "medium",
    "offer-letter-generator": "easy", "organize-messy-files": "medium",
    "paper-anonymizer": "medium", "pdf-excel-diff": "medium",
    "powerlifting-coef-calc": "easy", "pptx-reference-formatting": "medium",
    "sales-pivot-analysis": "medium", "scheduling-email-assistant": "medium",
    "travel-planning": "medium", "xlsx-recover-data": "medium",
    "adaptive-cruise-control": "medium", "hvac-control": "medium",
    "pddl-tpp-planning": "medium", "r2r-mpc-control": "medium",
    "virtualhome-agent-planning": "medium",
    "citation-check": "medium", "data-to-d3": "medium",
    "dialogue-parser": "easy", "enterprise-information-search": "hard",
    "fix-build-agentops": "easy", "fix-build-google-auto": "easy",
    "flink-query": "hard", "gh-repo-analytics": "medium",
    "jax-computing-basics": "medium", "parallel-tfidf-search": "medium",
    "python-scala-translation": "medium", "react-performance-debugging": "hard",
    "simpo-code-reproduction": "hard", "spring-boot-jakarta-migration": "hard",
    "taxonomy-tree-merge": "hard", "trend-anomaly-causal-inference": "hard",
    "video-silence-remover": "hard",
}


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_results(logs_dir: Path) -> List[dict]:
    """Load all result.json files, tagging each with preselect_strategy.

    Also loads ``skills_loaded.json`` (merged cat + JSON skill loads) when
    available, or falls back to on-the-fly parsing of episode logs via
    ``parse_skills_loaded.parse_pass_directory``.
    """
    # Infer strategy from directory name
    dir_name = logs_dir.name
    if dir_name.startswith("experiment_logs-"):
        dir_strategy = dir_name.split("experiment_logs-", 1)[1]
    elif dir_name == "experiment_logs":
        dir_strategy = "naive"
    else:
        dir_strategy = "naive"

    results = []
    for rf in sorted(logs_dir.rglob("result.json")):
        if "_image_cache" in str(rf):
            continue
        try:
            d = json.loads(rf.read_text())
            d["_run_dir"] = str(rf.parent)
            # Tag preselect strategy
            if "preselect_strategy" not in d:
                d["preselect_strategy"] = dir_strategy
            # Also load revised_reward if available
            rr = rf.parent / "revised_reward.json"
            if rr.exists():
                try:
                    d["_revised"] = json.loads(rr.read_text())
                except:
                    pass

            # ── Load merged skills (cat + JSON) ──
            pass_dir = rf.parent
            skills_loaded_file = pass_dir / "skills_loaded.json"
            if skills_loaded_file.exists():
                try:
                    sl = json.loads(skills_loaded_file.read_text())
                    d["_skills_loaded_override"] = sl.get("skill_names", [])
                except (json.JSONDecodeError, OSError):
                    pass
            elif _parse_pass_dir is not None:
                try:
                    parsed = _parse_pass_dir(pass_dir)
                    if parsed:
                        d["_skills_loaded_override"] = parsed.get("skill_names", [])
                except Exception:
                    pass

            results.append(d)
        except:
            pass
    return results


def discover_log_dirs(base_dir: Path) -> List[tuple]:
    """Auto-discover experiment log directories for all strategies."""
    found = []
    main_dir = base_dir / "experiment_logs"
    if main_dir.exists():
        found.append(("naive", main_dir))
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("experiment_logs-"):
            strategy = d.name.split("experiment_logs-", 1)[1]
            found.append((strategy, d))
    return found


def should_include(r: dict) -> bool:
    """Check if a result should be included in analysis."""
    task = r.get("task_name", "")
    model = r.get("model_name", "")
    if task in ALWAYS_EXCLUDED:
        return False
    if model in PROPRIETARY_MODELS and task in PROPRIETARY_EXCLUDED_RANGE:
        return False
    return True


def get_revised_reward(r: dict) -> float:
    """Get revised reward if available, else original."""
    rev = r.get("_revised", {})
    if rev and "revised_reward" in rev:
        return rev["revised_reward"]
    return r.get("reward", 0.0)


def get_domain(task: str) -> str:
    return DOMAIN_MAP.get(task, "Other")


def get_difficulty(task: str) -> str:
    return DIFFICULTY_MAP.get(task, "unknown")


# ─── Golden Skills (with SKILL.md name) ──────────────────────────────

def discover_golden_skills(tasks_dir: Path) -> Dict[str, Set[str]]:
    """Golden skills keyed by SKILL.md 'name' field (not folder name)."""
    golden = {}
    if not tasks_dir.exists():
        return golden
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.exists():
            continue
        skill_names = set()
        for d in skills_dir.iterdir():
            if not d.is_dir():
                continue
            skill_md = d / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text()
            m = re.search(r'^name:\s*(.+)', content, re.MULTILINE)
            if m:
                name = m.group(1).strip().strip('"').strip("'")
                skill_names.add(name)
            else:
                skill_names.add(d.name)  # fallback to folder name
        if skill_names:
            golden[task_dir.name] = skill_names
    return golden


def get_loaded_skills(r: dict) -> Set[str]:
    """Get unique skill names loaded during a run.

    Prefers merged ``_skills_loaded_override`` (from ``skills_loaded.json``
    or on-the-fly episode parsing) which captures both ``cat``-loaded and
    ``load_skill`` JSON-loaded skills.
    """
    override = r.get("_skills_loaded_override")
    if override is not None:
        return set(override)
    ls = r.get("agent_result", {}).get("loaded_skills", [])
    if not ls:
        return set()
    return {e["skill"] for e in ls if isinstance(e, dict) and "skill" in e}


# ─── Failure Classification ──────────────────────────────────────────

def classify_failure(r: dict) -> str:
    """Classify failure cause for a non-successful result."""
    err = str(r.get("exception") or r.get("agent_result", {}).get("error") or "")

    if "Access to Anthropic" in err:
        return "network:anthropic_access_denied"
    if "ReadTimeout" in err or "ConnectTimeout" in err:
        return "network:read_timeout"
    if "Rate limit" in err or "429" in err:
        return "network:rate_limit"
    if "Connection" in err and "Error" in err:
        return "network:connection_error"
    if "MALFORMED_FUNCTION_CALL" in err:
        return "api:malformed_function_call"
    if "context length" in err or "too many tokens" in err.lower() or "maximum context" in err.lower():
        return "agent:context_overflow"
    if "Max episodes" in err or "max_episodes" in err:
        return "agent:max_episodes"
    if "no space left" in err.lower():
        return "env:disk_full"
    if "未知错误" in err:
        return "api:unknown_api_error"
    if "LLM response missing" in err:
        return "api:empty_llm_response"
    if "Server dis" in err or "API error 500" in err:
        return "api:server_error_500"
    if err and err != "None":
        return "other:" + err[:80]

    # No explicit error → check verification
    v = r.get("verification_result", {})
    if v.get("ran") and not v.get("success"):
        return "task:verification_failed"
    if not v.get("ran"):
        return "agent:no_output"
    return "unknown"


def is_network_failure(cause: str) -> bool:
    return cause.startswith("network:")


# ═══════════════════════════════════════════════════════════════════════
# PASS RATE CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════

def calc_pass_at_1(passes: List[bool]) -> float:
    """Strict Pass@1: all 3 passes must succeed."""
    # Pass@1 = fraction of tasks where at least 1 pass succeeded
    # Actually: Pass@1 with k=3 passes = 1 - C(n-c, 1)/C(n, 1)
    # Simplified: for each (task, model, skills), pass@1 = 1 - (1 - p)
    # where p = num_success / num_passes
    # Standard: pass@1 = E[at least 1 success in 1 sample] = num_success / num_passes
    if not passes:
        return 0.0
    return sum(passes) / len(passes)


def calc_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n=total, c=correct, k=k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ═══════════════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ═══════════════════════════════════════════════════════════════════════

def section_a_pass_rate(results: List[dict]):
    """A. End-to-end pass rate (Pass@1, Pass@3)."""
    print()
    hline()
    print(bold("  A. END-TO-END PASS RATE"))
    hline()

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    # Group by (model, skills, task) → list of passes
    groups = defaultdict(list)
    for r in results:
        key = (r["model_name"], r["extra_skills_count"], r["task_name"])
        groups[key].append(r.get("final_success", False))

    # ── Pass@1 Table ──
    print()
    print(bold("  Pass@1 (unbiased estimator from 3 passes): Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        all_p1 = []
        for sc in skills_counts:
            # Collect all (task) groups for this model+sc
            task_p1s = []
            tasks_for = set(r["task_name"] for r in results
                           if r["model_name"] == model and r["extra_skills_count"] == sc)
            for task in sorted(tasks_for):
                passes = groups[(model, sc, task)]
                n = len(passes)
                c = sum(passes)
                p1 = calc_pass_at_k(n, c, 1)
                task_p1s.append(p1)
            if task_p1s:
                avg = sum(task_p1s) / len(task_p1s) * 100
                row += f"  {rate_color(avg)}"
                all_p1.extend(task_p1s)
            else:
                row += dim("    —   ")
        if all_p1:
            overall = sum(all_p1) / len(all_p1) * 100
            row += f"  {rate_color(overall)}"
        print(row)

    # ALL row
    row = f"  {bold('ALL'):<32s}"
    grand = []
    for sc in skills_counts:
        sc_p1 = []
        for model in models:
            tasks_for = set(r["task_name"] for r in results
                           if r["model_name"] == model and r["extra_skills_count"] == sc)
            for task in sorted(tasks_for):
                passes = groups[(model, sc, task)]
                p1 = calc_pass_at_k(len(passes), sum(passes), 1)
                sc_p1.append(p1)
        if sc_p1:
            avg = sum(sc_p1) / len(sc_p1) * 100
            row += f"  {rate_color(avg)}"
            grand.extend(sc_p1)
        else:
            row += dim("    —   ")
    if grand:
        overall = sum(grand) / len(grand) * 100
        row += f"  {rate_color(overall)}"
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))
    print(row)

    # ── Pass@3 Table ──
    print()
    print(bold("  Pass@3 (at least 1 of 3 passes succeeds): Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        all_p3 = []
        for sc in skills_counts:
            tasks_for = set(r["task_name"] for r in results
                           if r["model_name"] == model and r["extra_skills_count"] == sc)
            task_p3s = []
            for task in sorted(tasks_for):
                passes = groups[(model, sc, task)]
                n = len(passes)
                c = sum(passes)
                p3 = calc_pass_at_k(n, c, min(3, n))
                task_p3s.append(p3)
            if task_p3s:
                avg = sum(task_p3s) / len(task_p3s) * 100
                row += f"  {rate_color(avg)}"
                all_p3.extend(task_p3s)
            else:
                row += dim("    —   ")
        if all_p3:
            overall = sum(all_p3) / len(all_p3) * 100
            row += f"  {rate_color(overall)}"
        print(row)

    print()


def section_b_tokens(results: List[dict]):
    """B. Token consumption analysis."""
    print()
    hline()
    print(bold("  B. TOKEN CONSUMPTION"))
    hline()

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    # Per model × skills: avg input/output tokens
    print()
    print(bold("  Average Tokens per Run (input / output): Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  {'s=' + str(sc):^18s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 19))

    for model in models:
        row = f"  {cyan(model):<32s}"
        for sc in skills_counts:
            group = [r for r in results
                     if r["model_name"] == model and r["extra_skills_count"] == sc]
            if not group:
                row += dim("        —         ")
                continue
            inp = [r.get("agent_result", {}).get("input_tokens", 0) for r in group]
            out = [r.get("agent_result", {}).get("output_tokens", 0) for r in group]
            avg_i = sum(inp) / len(inp)
            avg_o = sum(out) / len(out)
            row += f"  {avg_i/1000:6.0f}k/{avg_o/1000:5.0f}k   "
        print(row)

    # Total tokens per model
    print()
    print(bold("  Total Tokens per Model:"))
    for model in models:
        group = [r for r in results if r["model_name"] == model]
        total_i = sum(r.get("agent_result", {}).get("input_tokens", 0) for r in group)
        total_o = sum(r.get("agent_result", {}).get("output_tokens", 0) for r in group)
        n = len(group)
        print(f"    {cyan(model):<30s}  runs={n:>4d}  "
              f"input={total_i/1e6:>7.1f}M  output={total_o/1e6:>5.1f}M  "
              f"total={bold(f'{(total_i+total_o)/1e6:.1f}M')}")
    print()


def section_c_reward(results: List[dict]):
    """C. Revised reward analysis."""
    print()
    hline()
    print(bold("  C. REVISED REWARD ANALYSIS"))
    hline()
    print(f"  {dim('(Uses revised_reward.json when available, else original reward)')}")

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    print()
    print(bold("  Average Reward: Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        model_rewards = []
        for sc in skills_counts:
            group = [r for r in results
                     if r["model_name"] == model and r["extra_skills_count"] == sc]
            if not group:
                row += dim("    —   ")
                continue
            rewards = [get_revised_reward(r) for r in group]
            avg = sum(rewards) / len(rewards)
            model_rewards.extend(rewards)
            row += f"  {avg:5.3f} "
        if model_rewards:
            overall = sum(model_rewards) / len(model_rewards)
            row += f"  {overall:5.3f}"
        print(row)

    # Distribution of rewards
    print()
    print(bold("  Reward Distribution:"))
    all_rewards = [get_revised_reward(r) for r in results]
    bins = [(0, 0, "= 0.0 (total fail)"),
            (0.001, 0.499, "0.0 < r < 0.5"),
            (0.5, 0.999, "0.5 ≤ r < 1.0"),
            (1.0, 1.0, "= 1.0 (full pass)")]
    for lo, hi, label in bins:
        if lo == hi:
            cnt = sum(1 for r in all_rewards if abs(r - lo) < 1e-6)
        else:
            cnt = sum(1 for r in all_rewards if lo <= r <= hi)
        pct = cnt / len(all_rewards) * 100 if all_rewards else 0
        bar = "█" * int(pct / 2)
        print(f"    {cnt:>5d} ({pct:5.1f}%)  {bar}  {label}")
    print()


def section_d_skills(results: List[dict], golden_skills: Dict[str, Set[str]]):
    """D. Skills hit rate analysis."""
    print()
    hline()
    print(bold("  D. SKILLS HIT RATE"))
    hline()
    print(f"  {dim('(golden skill names from SKILL.md name field)')}")

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))
    tasks = sorted(set(r["task_name"] for r in results))
    tasks_with_golden = [t for t in tasks if t in golden_skills and golden_skills[t]]

    # Group
    groups = defaultdict(list)
    for r in results:
        if r["task_name"] in golden_skills:
            groups[(r["model_name"], r["extra_skills_count"], r["task_name"])].append(r)

    # ── Hit Rate (Recall) Table ──
    print()
    print(bold("  Hit Rate (Recall) = |loaded ∩ golden| / |golden|: Model × Skills"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        model_rates = []
        for sc in skills_counts:
            task_rates = []
            for task in tasks_with_golden:
                group = groups[(model, sc, task)]
                if not group:
                    continue
                golden = golden_skills[task]
                pass_rates = []
                for r in group:
                    loaded = get_loaded_skills(r)
                    hits = loaded & golden
                    rate = len(hits) / len(golden) * 100 if golden else 0
                    pass_rates.append(rate)
                avg = sum(pass_rates) / len(pass_rates)
                task_rates.append(avg)
            if task_rates:
                avg = sum(task_rates) / len(task_rates)
                row += f"  {rate_color(avg)}"
                model_rates.extend(task_rates)
            else:
                row += dim("    —   ")
        if model_rates:
            overall = sum(model_rates) / len(model_rates)
            row += f"  {rate_color(overall)}"
        print(row)
    print()

    # ── Precision Table ──
    print(bold("  Precision = |loaded ∩ golden| / |loaded|: Model × Skills"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        model_precs = []
        for sc in skills_counts:
            task_precs = []
            for task in tasks_with_golden:
                group = groups[(model, sc, task)]
                if not group:
                    continue
                golden = golden_skills[task]
                pass_precs = []
                for r in group:
                    loaded = get_loaded_skills(r)
                    hits = loaded & golden
                    prec = len(hits) / len(loaded) * 100 if loaded else 100.0
                    pass_precs.append(prec)
                avg = sum(pass_precs) / len(pass_precs)
                task_precs.append(avg)
            if task_precs:
                avg = sum(task_precs) / len(task_precs)
                row += f"  {rate_color(avg)}"
                model_precs.extend(task_precs)
            else:
                row += dim("    —   ")
        if model_precs:
            overall = sum(model_precs) / len(model_precs)
            row += f"  {rate_color(overall)}"
        print(row)
    print()


def get_skill_load_count(r: dict) -> int:
    """Get the number of times the agent loaded skills during a run.

    Prefers ``_skills_loaded_override`` when available.
    """
    override = r.get("_skills_loaded_override")
    if override is not None:
        return len(override)
    ls = r.get("agent_result", {}).get("loaded_skills", [])
    if not ls:
        return 0
    return len(ls)


def section_d2_skill_usage_count(results: List[dict]):
    """D2. Skills usage count — how many times agents load skills per run."""
    print()
    hline()
    print(bold("  D2. SKILLS USAGE COUNT"))
    hline()
    print(f"  {dim('(Number of times the agent invoked load_skill during a run, averaged per task then per model)')}")

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    # Group by (model, skills, task) → list of results
    groups = defaultdict(list)
    for r in results:
        key = (r["model_name"], r["extra_skills_count"], r["task_name"])
        groups[key].append(r)

    # ── Average Skill Loads per Run: Model × Skills Count ──
    print()
    print(bold("  Average Skill Loads per Run: Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        model_avgs = []
        for sc in skills_counts:
            # Per-task averages
            task_avgs = []
            tasks_for = set(r["task_name"] for r in results
                           if r["model_name"] == model and r["extra_skills_count"] == sc)
            for task in sorted(tasks_for):
                runs = groups[(model, sc, task)]
                counts = [get_skill_load_count(r) for r in runs]
                task_avgs.append(sum(counts) / len(counts))
            if task_avgs:
                avg = sum(task_avgs) / len(task_avgs)
                row += f"  {avg:5.2f} "
                model_avgs.extend(task_avgs)
            else:
                row += dim("    —   ")
        if model_avgs:
            overall = sum(model_avgs) / len(model_avgs)
            row += f"  {overall:5.2f}"
        print(row)

    # ALL row
    row = f"  {bold('ALL'):<32s}"
    grand = []
    for sc in skills_counts:
        sc_avgs = []
        for model in models:
            tasks_for = set(r["task_name"] for r in results
                           if r["model_name"] == model and r["extra_skills_count"] == sc)
            for task in sorted(tasks_for):
                runs = groups[(model, sc, task)]
                counts = [get_skill_load_count(r) for r in runs]
                sc_avgs.append(sum(counts) / len(counts))
        if sc_avgs:
            avg = sum(sc_avgs) / len(sc_avgs)
            row += f"  {avg:5.2f} "
            grand.extend(sc_avgs)
        else:
            row += dim("    —   ")
    if grand:
        overall = sum(grand) / len(grand)
        row += f"  {overall:5.2f}"
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))
    print(row)

    # ── Zero-Load Rate (runs that loaded no skills) ──
    print()
    print(bold("  Zero-Load Rate (% of runs that loaded no skills at all): Model × Skills Count"))
    print()
    header = f"  {'Model':<22s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    for model in models:
        row = f"  {cyan(model):<32s}"
        model_total = 0
        model_zero = 0
        for sc in skills_counts:
            group = [r for r in results
                     if r["model_name"] == model and r["extra_skills_count"] == sc]
            total = len(group)
            zero = sum(1 for r in group if get_skill_load_count(r) == 0)
            model_total += total
            model_zero += zero
            if total > 0:
                rate = zero / total * 100.0
                row += f"  {rate_color(rate, '{:5.1f}%')}"
            else:
                row += dim("    —   ")
        if model_total > 0:
            rate = model_zero / model_total * 100.0
            row += f"  {rate_color(rate, '{:5.1f}%')}"
        print(row)
    print()

    # ── Distribution of skill load counts ──
    print(bold("  Distribution of Skill Load Counts (all runs):"))
    all_counts = [get_skill_load_count(r) for r in results]
    max_count = max(all_counts) if all_counts else 0
    bins = [(0, "0 (no skills loaded)"),
            (1, "1"),
            (2, "2"),
            (3, "3+")]
    total_runs = len(all_counts)
    for threshold, label in bins:
        if threshold < 3:
            cnt = sum(1 for c in all_counts if c == threshold)
        else:
            cnt = sum(1 for c in all_counts if c >= threshold)
        pct = cnt / total_runs * 100 if total_runs else 0
        bar = "█" * int(pct / 2)
        print(f"    {cnt:>5d} ({pct:5.1f}%)  {bar}  {label}")
    if max_count >= 3:
        print(f"    {dim(f'Max skill loads in a single run: {max_count}')}")
    print()


def section_e_domain_difficulty(results: List[dict]):
    """E. Breakdown by domain and difficulty."""
    print()
    hline()
    print(bold("  E. DOMAIN & DIFFICULTY BREAKDOWN"))
    hline()

    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    # ── By Domain ──
    print()
    print(bold("  Pass@1 by Domain (averaged across tasks, models, skills):"))
    print()

    domain_tasks = defaultdict(set)
    for r in results:
        domain_tasks[get_domain(r["task_name"])].add(r["task_name"])

    # Group for pass@1
    groups = defaultdict(list)
    for r in results:
        groups[(r["model_name"], r["extra_skills_count"], r["task_name"])].append(
            r.get("final_success", False))

    header = f"  {'Domain':<22s}  {'#Tasks':>6s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (32 + len(skills_counts) * 9 + 10))

    for domain in sorted(domain_tasks.keys()):
        dtasks = sorted(domain_tasks[domain])
        row = f"  {domain:<22s}  {len(dtasks):>6d}"
        domain_all = []
        for sc in skills_counts:
            sc_p1 = []
            for model in models:
                for task in dtasks:
                    passes = groups.get((model, sc, task))
                    if not passes:
                        continue
                    p1 = calc_pass_at_k(len(passes), sum(passes), 1)
                    sc_p1.append(p1)
            if sc_p1:
                avg = sum(sc_p1) / len(sc_p1) * 100
                row += f"  {rate_color(avg)}"
                domain_all.extend(sc_p1)
            else:
                row += dim("    —   ")
        if domain_all:
            overall = sum(domain_all) / len(domain_all) * 100
            row += f"  {rate_color(overall)}"
        print(row)

    # ── By Difficulty ──
    print()
    print(bold("  Pass@1 by Difficulty:"))
    print()
    header = f"  {'Difficulty':<12s}  {'#Tasks':>6s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (22 + len(skills_counts) * 9 + 10))

    diff_tasks = defaultdict(set)
    for r in results:
        diff_tasks[get_difficulty(r["task_name"])].add(r["task_name"])

    for diff in ["easy", "medium", "hard"]:
        if diff not in diff_tasks:
            continue
        dtasks = sorted(diff_tasks[diff])
        row = f"  {diff:<12s}  {len(dtasks):>6d}"
        diff_all = []
        for sc in skills_counts:
            sc_p1 = []
            for model in models:
                for task in dtasks:
                    passes = groups.get((model, sc, task))
                    if not passes:
                        continue
                    p1 = calc_pass_at_k(len(passes), sum(passes), 1)
                    sc_p1.append(p1)
            if sc_p1:
                avg = sum(sc_p1) / len(sc_p1) * 100
                row += f"  {rate_color(avg)}"
                diff_all.extend(sc_p1)
            else:
                row += dim("    —   ")
        if diff_all:
            overall = sum(diff_all) / len(diff_all) * 100
            row += f"  {rate_color(overall)}"
        print(row)

    # ── Skills hit rate by difficulty ──
    print()
    print(bold("  Average Reward by Difficulty × Skills:"))
    print()
    header = f"  {'Difficulty':<12s}"
    for sc in skills_counts:
        header += f"  s={sc:<6d}"
    header += f"    {'ALL':>5s}"
    print(bold(header))
    print("  " + "─" * (12 + len(skills_counts) * 9 + 10))

    for diff in ["easy", "medium", "hard"]:
        if diff not in diff_tasks:
            continue
        dtasks = sorted(diff_tasks[diff])
        row = f"  {diff:<12s}"
        diff_rewards = []
        for sc in skills_counts:
            rewards = [get_revised_reward(r) for r in results
                       if get_difficulty(r["task_name"]) == diff
                       and r["extra_skills_count"] == sc]
            if rewards:
                avg = sum(rewards) / len(rewards)
                row += f"  {avg:5.3f} "
                diff_rewards.extend(rewards)
            else:
                row += dim("    —   ")
        if diff_rewards:
            overall = sum(diff_rewards) / len(diff_rewards)
            row += f"  {overall:5.3f}"
        print(row)
    print()


def section_f_failures(results: List[dict]):
    """F. Failure cause classification."""
    print()
    hline()
    print(bold("  F. FAILURE CAUSE ANALYSIS"))
    hline()

    failed = [r for r in results if not r.get("final_success")]
    if not failed:
        print(f"  {green('No failures!')}")
        return

    causes = defaultdict(int)
    for r in failed:
        causes[classify_failure(r)] += 1

    print()
    print(f"  Total failures: {red(str(len(failed)))} / {len(results)}")
    print()
    print(bold("  Failure Breakdown:"))
    for cause, cnt in sorted(causes.items(), key=lambda x: -x[1]):
        pct = cnt / len(failed) * 100
        bar_len = min(int(cnt / max(causes.values()) * 40), 40)
        is_net = is_network_failure(cause)
        color_fn = magenta if is_net else red
        tag = " [NETWORK - rerunnable]" if is_net else ""
        print(f"    {cnt:>5d} ({pct:4.1f}%)  {color_fn('█' * bar_len)}  {cause}{tag}")

    # Network failures summary
    net_count = sum(v for k, v in causes.items() if is_network_failure(k))
    print()
    print(f"  {magenta('Network-related failures')}: {net_count} "
          f"({net_count/len(failed)*100:.1f}% of failures, "
          f"{net_count/len(results)*100:.1f}% of all runs)")
    print()


def section_g_network_rerun(results: List[dict]):
    """G. List of network-failed runs for rerun."""
    print()
    hline()
    print(bold("  G. NETWORK FAILURES (candidates for rerun)"))
    hline()

    net_fails = []
    for r in results:
        if r.get("final_success"):
            continue
        cause = classify_failure(r)
        if is_network_failure(cause):
            net_fails.append((r, cause))

    if not net_fails:
        print(f"  {green('No network failures!')}")
        return

    print(f"\n  Total: {magenta(str(len(net_fails)))} runs\n")

    by_model = defaultdict(list)
    for r, cause in net_fails:
        by_model[r["model_name"]].append((r, cause))

    for model in sorted(by_model):
        items = by_model[model]
        print(f"  {cyan(model)}: {len(items)} runs")
        for r, cause in sorted(items, key=lambda x: (x[0]["task_name"], x[0]["extra_skills_count"])):
            print(f"    {r['task_name']:<45s} s={r['extra_skills_count']:<5d} "
                  f"p={r.get('pass_index','?')}  [{cause}]")
    print()


# ═══════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_csvs(results: List[dict], golden_skills: Dict[str, Set[str]], prefix: str):
    """Export comprehensive CSV files."""
    os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)

    # 1. Per-run detail
    path = f"{prefix}_per_run.csv"
    strategies = sorted(set(r.get("preselect_strategy", "naive") for r in results))
    has_strategies = len(strategies) > 1 or strategies != ["naive"]
    
    fields = ["task_name", "model_name", "extra_skills_count", "pass_index",
              "domain", "difficulty", "final_success", "original_reward",
              "revised_reward", "failure_cause",
              "input_tokens", "output_tokens", "episodes", "execution_time_sec",
              "skill_load_count", "n_loaded_skills", "loaded_skills", "n_golden_hits", "golden_skills"]
    if has_strategies:
        fields.insert(4, "preselect_strategy")
        fields.extend(["preselected_skills", "preselect_input_tokens", "preselect_output_tokens"])
    
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            task = r.get("task_name", "")
            agent = r.get("agent_result", {})
            loaded = get_loaded_skills(r)
            golden = golden_skills.get(task, set())
            hits = loaded & golden
            cause = "" if r.get("final_success") else classify_failure(r)
            row = {
                "task_name": task,
                "model_name": r.get("model_name"),
                "extra_skills_count": r.get("extra_skills_count"),
                "pass_index": r.get("pass_index"),
                "domain": get_domain(task),
                "difficulty": get_difficulty(task),
                "final_success": r.get("final_success"),
                "original_reward": r.get("reward"),
                "revised_reward": get_revised_reward(r),
                "failure_cause": cause,
                "input_tokens": agent.get("input_tokens", 0),
                "output_tokens": agent.get("output_tokens", 0),
                "episodes": agent.get("episodes", 0),
                "execution_time_sec": r.get("execution_time_sec", 0),
                "skill_load_count": get_skill_load_count(r),
                "n_loaded_skills": len(loaded),
                "loaded_skills": ";".join(sorted(loaded)),
                "n_golden_hits": len(hits),
                "golden_skills": ";".join(sorted(golden)),
            }
            if has_strategies:
                row["preselect_strategy"] = r.get("preselect_strategy", "naive")
                ps = r.get("preselected_skills") or agent.get("preselected_skills")
                row["preselected_skills"] = ";".join(sorted(ps)) if ps and isinstance(ps, list) else ""
                row["preselect_input_tokens"] = r.get("preselect_input_tokens", agent.get("preselect_input_tokens", 0))
                row["preselect_output_tokens"] = r.get("preselect_output_tokens", agent.get("preselect_output_tokens", 0))
            w.writerow(row)
    print(f"  ✅ {path} ({len(results)} rows)")

    # 2. Aggregated pass rate
    path2 = f"{prefix}_pass_rate.csv"
    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))
    groups = defaultdict(list)
    for r in results:
        strat = r.get("preselect_strategy", "naive")
        groups[(r["model_name"], r["extra_skills_count"], r["task_name"], strat)].append(
            r.get("final_success", False))
    
    csv_fields = ["model", "skills_count", "task", "domain", "difficulty",
                  "n_passes", "n_success", "pass_at_1", "pass_at_3", "avg_reward"]
    if has_strategies:
        csv_fields.insert(5, "preselect_strategy")
    
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_fields)
        for (model, sc, task, strat), passes in sorted(groups.items()):
            rewards = [get_revised_reward(r) for r in results
                       if r["model_name"] == model and r["extra_skills_count"] == sc
                       and r["task_name"] == task
                       and r.get("preselect_strategy", "naive") == strat]
            n = len(passes)
            c = sum(passes)
            row = [model, sc, task, get_domain(task), get_difficulty(task)]
            if has_strategies:
                row.append(strat)
            row.extend([n, c, f"{calc_pass_at_k(n,c,1):.4f}",
                        f"{calc_pass_at_k(n,c,min(3,n)):.4f}",
                        f"{sum(rewards)/len(rewards):.4f}" if rewards else ""])
            w.writerow(row)
    print(f"  ✅ {path2}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive experiment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        nargs="+",
        default=[str(Path(__file__).parent / "experiment_logs")],
        help="Path(s) to experiment_logs directory(ies)",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Auto-discover all experiment_logs* directories (naive, llm, rag, etc.)",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        help="Filter by preselect strategy (naive, llm, rag)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        help="Filter by model name(s)",
    )
    parser.add_argument("--tasks-dir", default=str(Path(__file__).parent.parent / "skillsbench" / "tasks"))
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--csv-prefix", type=str, help="Export CSV files with this prefix")
    args = parser.parse_args()

    global USE_COLOR
    if args.no_color:
        USE_COLOR = False

    tasks_dir = Path(args.tasks_dir)

    # Determine which directories to load
    if args.all_strategies:
        base_dir = Path(__file__).parent
        strategy_dirs = discover_log_dirs(base_dir)
        if not strategy_dirs:
            print("❌ No experiment_logs* directories found.", file=sys.stderr)
            sys.exit(1)
        print(f"📂 Auto-discovered {len(strategy_dirs)} log directories:")
        for strategy, path in strategy_dirs:
            print(f"   {cyan(strategy):>15s} → {path}")
        print()
        logs_dirs = [path for _, path in strategy_dirs]
    else:
        logs_dirs = [Path(d) for d in args.logs_dir]

    # Load all data
    print(dim("  Loading results..."), end="", flush=True)
    all_results = []
    for logs_dir in logs_dirs:
        if not logs_dir.exists():
            print(f"\n⚠ Directory not found: {logs_dir}", file=sys.stderr)
            continue
        dir_results = load_all_results(logs_dir)
        all_results.extend(dir_results)
    print(dim(f" {len(all_results)} total"))

    if not all_results:
        print("❌ No result.json files found.", file=sys.stderr)
        sys.exit(1)

    # Filter by strategy if requested
    if args.strategy:
        all_results = [r for r in all_results if r.get("preselect_strategy") in args.strategy]
        if not all_results:
            print("❌ No results match the given strategy filter.", file=sys.stderr)
            sys.exit(1)

    # Filter by model if requested
    if args.model:
        all_results = [r for r in all_results if r.get("model_name") in args.model]
        if not all_results:
            print("❌ No results match the given model filter.", file=sys.stderr)
            sys.exit(1)

    # Filter by inclusion rules
    results = [r for r in all_results if should_include(r)]
    excluded_count = len(all_results) - len(results)

    # Discover golden skills
    golden_skills = discover_golden_skills(tasks_dir)

    # Summary
    print()
    hline()
    print(bold("  COMPREHENSIVE SKILLS SCALING STUDY ANALYSIS"))
    hline()
    models = sorted(set(r["model_name"] for r in results))
    tasks = sorted(set(r["task_name"] for r in results))
    strategies = sorted(set(r.get("preselect_strategy", "naive") for r in results))
    print(f"  Results loaded     : {len(all_results)}")
    print(f"  After filtering    : {bold(str(len(results)))} (excluded {excluded_count})")
    print(f"  Models             : {', '.join(cyan(m) for m in models)}")
    print(f"  Tasks              : {len(tasks)}")
    print(f"  Skills counts      : {sorted(set(r['extra_skills_count'] for r in results))}")
    if len(strategies) > 1 or strategies != ["naive"]:
        print(f"  Strategies         : {', '.join(magenta(s) for s in strategies)}")
    print(f"  Always excluded    : {', '.join(dim(t) for t in sorted(ALWAYS_EXCLUDED))}")
    print(f"  Proprietary-only   : {', '.join(dim(t) for t in sorted(PROPRIETARY_EXCLUDED_RANGE))}")
    print(f"    (excluded for {', '.join(sorted(PROPRIETARY_MODELS))})")
    print(f"  Tasks w/ golden    : {green(str(len([t for t in tasks if t in golden_skills])))}")

    # Count tasks per model
    for model in models:
        m_tasks = set(r["task_name"] for r in results if r["model_name"] == model)
        print(f"    {cyan(model):<30s}: {len(m_tasks)} tasks, "
              f"{sum(1 for r in results if r['model_name']==model)} runs")

    # Run all sections
    section_a_pass_rate(results)
    section_b_tokens(results)
    section_c_reward(results)
    section_d_skills(results, golden_skills)
    section_d2_skill_usage_count(results)
    section_e_domain_difficulty(results)
    section_f_failures(results)
    section_g_network_rerun(results)

    if args.csv_prefix:
        print()
        hline()
        print(bold("  CSV EXPORT"))
        hline()
        export_csvs(results, golden_skills, args.csv_prefix)

    print(dim("  Done."))


if __name__ == "__main__":
    main()
