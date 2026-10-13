#!/usr/bin/env python3
"""
Experiment Results Analyzer
===========================
Reads all result.json files from experiment_logs/ and displays
pass rates broken down by model, skills count, and task.

Supports multiple log directories for comparing pre-selection strategies:
  experiment_logs/      → naive (default, all skills in prompt)
  experiment_logs-llm/  → LLM pre-selection
  experiment_logs-rag/  → RAG pre-selection

Usage:
    python analyze_results.py                    # Full report (experiment_logs/ only)
    python analyze_results.py --model glm-5      # Filter by model
    python analyze_results.py --task 3d-scan-calc # Filter by task
    python analyze_results.py --skills 0 100      # Filter by skills count
    python analyze_results.py --csv results.csv   # Export to CSV
    python analyze_results.py --errors            # Show error breakdown
    python analyze_results.py --crashes           # Show mid-run crashes

    # Multi-strategy analysis:
    python analyze_results.py --all-strategies    # Auto-discover naive/llm/rag logs
    python analyze_results.py --logs-dir experiment_logs experiment_logs-llm
    python analyze_results.py --strategy llm      # Filter by strategy
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── ANSI Colors ──────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Check if terminal supports color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

def _c(code: str, text: str) -> str:
    """Wrap text with ANSI color code."""
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(text: str) -> str:
    return _c("32", text)

def red(text: str) -> str:
    return _c("31", text)

def yellow(text: str) -> str:
    return _c("33", text)

def cyan(text: str) -> str:
    return _c("36", text)

def bold(text: str) -> str:
    return _c("1", text)

def dim(text: str) -> str:
    return _c("2", text)

def bg_green(text: str) -> str:
    return _c("42;30", text)

def bg_red(text: str) -> str:
    return _c("41;37", text)

def bg_yellow(text: str) -> str:
    return _c("43;30", text)


# ─── Result classification ───────────────────────────────────────────

def classify_result(r: dict) -> str:
    """
    Classify a result into one of:
      'pass'   — final_success is True
      'fail'   — ran verification, but failed
      'crash'  — never reached verification (mid-run error)
    """
    if r.get("final_success"):
        return "pass"
    
    verif = r.get("verification_result", {})
    has_exception = bool(r.get("exception") or r.get("traceback"))
    agent = r.get("agent_result", {})
    agent_error = agent.get("error", "")
    
    # "crash" = never got to verification.  Indicators:
    #   - verification didn't run AND there's an exception/traceback
    #   - OR agent episodes == 0 (never completed a single episode)
    #   - OR explicit exception field present
    if has_exception:
        return "crash"
    if not verif.get("ran") and agent.get("episodes", 0) == 0:
        return "crash"
    
    return "fail"


def classify_counts(results: List[dict]) -> Tuple[int, int, int]:
    """Return (pass, fail, crash) counts."""
    p = f = c = 0
    for r in results:
        cls = classify_result(r)
        if cls == "pass":
            p += 1
        elif cls == "fail":
            f += 1
        else:
            c += 1
    return p, f, c


# ─── Helpers ──────────────────────────────────────────────────────────

def load_results(logs_dir: Path) -> List[dict]:
    """Load all result.json files from the experiment logs directory.
    
    Automatically tags each result with 'preselect_strategy' based on:
    1. The result.json's own 'preselect_strategy' field (if present)
    2. The directory name suffix (e.g., experiment_logs-llm → 'llm')
    3. Defaults to 'naive' if neither is available
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
    for result_file in sorted(logs_dir.rglob("result.json")):
        if "_image_cache" in str(result_file):
            continue
        try:
            data = json.loads(result_file.read_text())
            data["_path"] = str(result_file)
            # Tag with preselect_strategy
            if "preselect_strategy" not in data:
                data["preselect_strategy"] = dir_strategy
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠ Skipping {result_file}: {e}", file=sys.stderr)
    return results


def discover_log_dirs(base_dir: Path) -> List[Tuple[str, Path]]:
    """Auto-discover experiment log directories for all strategies.
    
    Looks for:
      base_dir/experiment_logs       → naive
      base_dir/experiment_logs-llm   → llm
      base_dir/experiment_logs-rag   → rag
      (and any other experiment_logs-* directories)
    
    Returns: [(strategy_name, path), ...]
    """
    found = []
    # Main experiment_logs
    main_dir = base_dir / "experiment_logs"
    if main_dir.exists():
        found.append(("naive", main_dir))
    
    # Discover experiment_logs-* variants
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("experiment_logs-"):
            strategy = d.name.split("experiment_logs-", 1)[1]
            found.append((strategy, d))
    
    return found


def filter_results(
    results: List[dict],
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    skills: Optional[List[int]] = None,
    strategies: Optional[List[str]] = None,
) -> List[dict]:
    """Filter results by model, task, skills count, or preselect strategy."""
    filtered = results
    if models:
        filtered = [r for r in filtered if r.get("model_name") in models]
    if tasks:
        filtered = [r for r in filtered if r.get("task_name") in tasks]
    if skills is not None:
        filtered = [r for r in filtered if r.get("extra_skills_count") in skills]
    if strategies:
        filtered = [r for r in filtered if r.get("preselect_strategy") in strategies]
    return filtered


def calc_pass_rate(results: List[dict]) -> Tuple[int, int, float]:
    """Calculate pass rate: (passed, total, rate)."""
    total = len(results)
    passed = sum(1 for r in results if r.get("final_success"))
    rate = passed / total * 100 if total > 0 else 0.0
    return passed, total, rate


def _rate_color(rate: float) -> str:
    """Color a rate string based on value."""
    text = f"{rate:4.0f}%"
    if rate >= 60:
        return green(text)
    elif rate > 0:
        return yellow(text)
    else:
        return red(text)


def _status_emoji(cls: str) -> str:
    """Get colored emoji for a classification."""
    if cls == "pass":
        return green("✅")
    elif cls == "crash":
        return red("💥")
    else:
        return yellow("❌")


def _cell_str(group: List[dict]) -> str:
    """Format a cell showing pass/total with color."""
    if not group:
        return dim("  —         ")
    p, t, rate = calc_pass_rate(group)
    ratio = f"{p}/{t:>2d}"
    rate_s = _rate_color(rate)
    return f"  {ratio} ({rate_s})"


# ─── Report generators ───────────────────────────────────────────────


def print_overview(results: List[dict]):
    """Print overall summary."""
    passed, total, rate = calc_pass_rate(results)
    p, f, c = classify_counts(results)
    models = sorted(set(r["model_name"] for r in results))
    tasks = sorted(set(r["task_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))
    strategies = sorted(set(r.get("preselect_strategy", "naive") for r in results))

    print(bold("=" * 70))
    print(bold("  📊 EXPERIMENT RESULTS OVERVIEW"))
    print(bold("=" * 70))
    print(f"  Total results : {bold(str(total))}")
    print(f"  ✅ Passed      : {green(str(p))}")
    print(f"  ❌ Failed      : {yellow(str(f))}  (ran verification but didn't pass)")
    print(f"  💥 Crashed     : {red(str(c))}  (mid-run error, never reached verification)")
    print(f"  Pass rate     : {_rate_color(rate)}")
    print(f"  Models        : {', '.join(cyan(m) for m in models)}")
    print(f"  Tasks         : {len(tasks)}")
    print(f"  Skills counts : {skills_counts}")
    if len(strategies) > 1 or strategies != ["naive"]:
        print(f"  Strategies    : {', '.join(cyan(s) for s in strategies)}")
    print(bold("=" * 70))
    print()


def print_model_x_skills(results: List[dict]):
    """Table: pass rate per (model × skills_count)."""
    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    lookup: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in results:
        lookup[(r["model_name"], r["extra_skills_count"])].append(r)

    print(bold("┌─ 📈 Pass Rate: Model × Skills Count ─────────────────────────────┐"))
    print()

    header = f"  {bold('Model'):<35s}"
    for sc in skills_counts:
        header += f"  skills={sc:<5d}"
    header += f"    {bold('ALL')}"
    print(header)
    print("  " + "─" * (25 + len(skills_counts) * 13 + 10))

    for model in models:
        row = f"  {cyan(model):<35s}"
        for sc in skills_counts:
            row += _cell_str(lookup[(model, sc)])
        model_results = [r for r in results if r["model_name"] == model]
        p, t, rate = calc_pass_rate(model_results)
        row += f"  {p}/{t:>3d} ({_rate_color(rate)})"
        print(row)

    row = f"  {bold('ALL'):<35s}"
    for sc in skills_counts:
        group = [r for r in results if r["extra_skills_count"] == sc]
        row += _cell_str(group)
    p, t, rate = calc_pass_rate(results)
    row += f"  {p}/{t:>3d} ({_rate_color(rate)})"
    print("  " + "─" * (25 + len(skills_counts) * 13 + 10))
    print(row)
    print()


def print_task_details(results: List[dict]):
    """Table: per-task pass rate across models and skills counts."""
    tasks = sorted(set(r["task_name"] for r in results))
    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    lookup: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for r in results:
        lookup[(r["task_name"], r["model_name"], r["extra_skills_count"])].append(r)

    print(bold("┌─ 📋 Per-Task Results ────────────────────────────────────────────┐"))
    print()

    for task in tasks:
        task_results = [r for r in results if r["task_name"] == task]
        p, t, rate = calc_pass_rate(task_results)
        tp, tf, tc = classify_counts(task_results)

        # Task header with crash indicator
        crash_hint = f"  {red(f'💥{tc} crashed')}" if tc > 0 else ""
        print(f"  📋 {bold(task)}  [{p}/{t} = {_rate_color(rate)}]{crash_hint}")

        for model in models:
            model_task = [r for r in task_results if r["model_name"] == model]
            if not model_task:
                continue
            parts = []
            for sc in skills_counts:
                group = lookup[(task, model, sc)]
                if not group:
                    continue
                p_g, t_g, rate_g = calc_pass_rate(group)
                # Classify each run in the group
                crashes_in_group = sum(1 for r in group if classify_result(r) == "crash")

                if p_g == t_g and t_g > 0:
                    mark = green("✅")
                elif p_g > 0:
                    mark = yellow("⚠️")
                elif crashes_in_group > 0:
                    mark = red("💥")
                else:
                    mark = red("❌")
                parts.append(f"s{sc}={p_g}/{t_g}{mark}")

            p_m, t_m, rate_m = calc_pass_rate(model_task)
            crashes_model = sum(1 for r in model_task if classify_result(r) == "crash")
            crash_tag = f" {red(f'💥{crashes_model}')}" if crashes_model > 0 else ""
            print(f"     {cyan(model):<32s}  {' | '.join(parts):55s}  total: {p_m}/{t_m} ({_rate_color(rate_m)}){crash_tag}")
        print()


def print_error_breakdown(results: List[dict]):
    """Show breakdown of error types for failed tasks."""
    failed = [r for r in results if not r.get("final_success")]
    if not failed:
        print(f"  🎉 {green('No failures found!')}")
        return

    error_counts: Dict[str, int] = defaultdict(int)
    for r in failed:
        error = r.get("exception") or r.get("agent_result", {}).get("error", "unknown")
        # Normalize
        err_str = str(error)
        if "KeyError" in err_str:
            error = "KeyError: " + err_str.split("KeyError:")[-1].strip().strip("'\"")
        elif "Max episodes" in err_str:
            error = "Max episodes reached"
        elif "Rate limit" in err_str or "429" in err_str:
            error = "Rate limit (429)"
        elif "Connection" in err_str or "Timeout" in err_str:
            error = "Connection/Timeout error"
        elif "context length" in err_str or "token" in err_str.lower():
            error = "Context length exceeded"
        elif "no space left" in err_str:
            error = "Disk full (no space left on device)"
        elif "MALFORMED_FUNCTION_CALL" in err_str:
            error = "Malformed function call"
        elif "temperature" in err_str and "top_p" in err_str:
            error = "temperature+top_p conflict"
        elif "UnicodeDecodeError" in err_str:
            error = "UnicodeDecodeError"
        error_counts[error] = error_counts.get(error, 0) + 1

    print(bold("┌─ 🔍 Error Breakdown (Failed Tasks) ─────────────────────────────┐"))
    print()
    print(f"  Total failures: {red(str(len(failed)))}")
    print()
    for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        bar = red("█" * min(count, 40))
        print(f"  {count:>4d}  {bar}  {error}")
    print()


def print_crash_report(results: List[dict]):
    """Show tasks that crashed mid-run (never reached verification)."""
    crashes = [r for r in results if classify_result(r) == "crash"]
    if not crashes:
        print(f"  🎉 {green('No mid-run crashes!')}")
        return

    print(bold("┌─ 💥 Mid-Run Crashes (never reached verification) ────────────────┐"))
    print()
    print(f"  Total crashes: {red(str(len(crashes)))}")
    print()

    # Group by error type
    by_error: Dict[str, List[dict]] = defaultdict(list)
    for r in crashes:
        error = r.get("exception") or r.get("agent_result", {}).get("error", "unknown")
        err_str = str(error)
        # Normalize
        if "KeyError" in err_str:
            key = "KeyError: " + err_str.split("KeyError:")[-1].strip().strip("'\"")
        elif "Rate limit" in err_str or "429" in err_str:
            key = "Rate limit (429)"
        elif "Connection" in err_str or "Timeout" in err_str:
            key = "Connection/Timeout error"
        elif "context length" in err_str or "token" in err_str.lower():
            key = "Context length exceeded"
        elif "no space left" in err_str:
            key = "Disk full"
        elif "UnicodeDecodeError" in err_str:
            key = "UnicodeDecodeError"
        else:
            key = err_str[:80]
        by_error[key].append(r)

    for error, group in sorted(by_error.items(), key=lambda x: -len(x[1])):
        print(f"  {red('💥')} {bold(error)} ({len(group)} tasks)")
        for r in group:
            task = r.get("task_name", "?")
            model = r.get("model_name", "?")
            sc = r.get("extra_skills_count", "?")
            pi = r.get("pass_index", "?")
            ep = r.get("agent_result", {}).get("episodes", 0)
            t = r.get("execution_time_sec", 0)
            print(f"       {cyan(model):<20s}  {task:<40s}  s={sc:<5} p={pi}  ep={ep:>2d}  {t:.0f}s")
        print()


def print_progress_matrix(results: List[dict]):
    """Compact matrix showing completion status per (task × model × skills)."""
    tasks = sorted(set(r["task_name"] for r in results))
    models = sorted(set(r["model_name"] for r in results))
    skills_counts = sorted(set(r["extra_skills_count"] for r in results))

    columns = [(m, sc) for m in models for sc in skills_counts]

    print(bold(f"┌─ 🗂️  Completion Matrix  (✅=pass ⚠️=partial ❌=fail 💥=crash ·=no data) ─┐"))
    print()

    # Model names row
    header1 = f"  {'':40s}"
    prev_model = None
    for m, sc in columns:
        if m != prev_model:
            header1 += f" {cyan(m[:8]):^{len(skills_counts)*4}s}"
            prev_model = m
    print(header1)

    # Skills row
    header2 = f"  {bold('Task'):<50s}"
    for m, sc in columns:
        header2 += f" {sc:>3d}"
    print(header2)
    print("  " + "─" * (40 + len(columns) * 4))

    lookup: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for r in results:
        lookup[(r["task_name"], r["model_name"], r["extra_skills_count"])].append(r)

    for task in tasks:
        row = f"  {task[:40]:<40s}"
        for m, sc in columns:
            group = lookup[(task, m, sc)]
            if not group:
                row += dim("   ·")
            else:
                p, t, _ = calc_pass_rate(group)
                crashes = sum(1 for r in group if classify_result(r) == "crash")
                if p == t and t > 0:
                    row += green("  ✅")
                elif p > 0:
                    row += yellow("  ⚠️")
                elif crashes > 0:
                    row += red("  💥")
                else:
                    row += red("  ❌")
        print(row)
    print()


def export_csv(results: List[dict], output_path: str):
    """Export results to CSV."""
    import csv

    fields = [
        "task_name", "model_name", "extra_skills_count", "pass_index",
        "skills_seed", "preselect_strategy", "final_success", "classification",
        "reward", "execution_time_sec", "agent_episodes", "agent_input_tokens",
        "agent_output_tokens", "agent_error", "verification_ran",
        "verification_success", "exception",
        "preselected_skills", "preselect_input_tokens", "preselect_output_tokens",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            agent = r.get("agent_result", {})
            verif = r.get("verification_result", {})
            preselected = agent.get("preselected_skills", [])
            writer.writerow({
                "task_name": r.get("task_name"),
                "model_name": r.get("model_name"),
                "extra_skills_count": r.get("extra_skills_count"),
                "pass_index": r.get("pass_index"),
                "skills_seed": r.get("skills_seed"),
                "preselect_strategy": r.get("preselect_strategy", "naive"),
                "final_success": r.get("final_success"),
                "classification": classify_result(r),
                "reward": r.get("reward"),
                "execution_time_sec": r.get("execution_time_sec"),
                "agent_episodes": agent.get("episodes"),
                "agent_input_tokens": agent.get("input_tokens"),
                "agent_output_tokens": agent.get("output_tokens"),
                "agent_error": agent.get("error"),
                "verification_ran": verif.get("ran"),
                "verification_success": verif.get("success"),
                "exception": r.get("exception"),
                "preselected_skills": ";".join(preselected) if preselected else "",
                "preselect_input_tokens": agent.get("preselect_input_tokens", 0),
                "preselect_output_tokens": agent.get("preselect_output_tokens", 0),
            })
    print(f"✅ Exported {len(results)} results to {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results from experiment_logs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        nargs="+",
        default=[str(Path(__file__).parent / "experiment_logs")],
        help="Path(s) to experiment_logs directory(ies). "
             "Can specify multiple: --logs-dir experiment_logs experiment_logs-llm",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Auto-discover all experiment_logs* directories (naive, llm, rag, etc.)",
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        help="Filter by preselect strategy (naive, llm, rag, hybrid-rag-llm, multi-query-rag-llm, dynamic-rag)",
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Filter by model name(s)",
    )
    parser.add_argument(
        "--task", "-t",
        nargs="+",
        help="Filter by task name(s)",
    )
    parser.add_argument(
        "--skills", "-s",
        nargs="+",
        type=int,
        help="Filter by skills count(s), e.g. --skills 0 100 1000",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--errors", "-e",
        action="store_true",
        help="Show error breakdown for failed tasks",
    )
    parser.add_argument(
        "--crashes",
        action="store_true",
        help="Show detailed crash report (tasks that died before verification)",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show compact completion matrix",
    )
    parser.add_argument(
        "--no-tasks",
        action="store_true",
        help="Skip per-task detail (faster for quick overview)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    global USE_COLOR
    if args.no_color:
        USE_COLOR = False

    # Determine which directories to load
    if args.all_strategies:
        # Auto-discover all experiment_logs* dirs
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

    # Load results from all directories
    results = []
    for logs_dir in logs_dirs:
        if not logs_dir.exists():
            print(f"⚠ Directory not found: {logs_dir}", file=sys.stderr)
            continue
        dir_results = load_results(logs_dir)
        print(f"📂 Loaded {len(dir_results)} results from {logs_dir}", file=sys.stderr)
        results.extend(dir_results)

    if not results:
        print("❌ No result.json files found.", file=sys.stderr)
        sys.exit(1)

    results = filter_results(
        results,
        models=args.model,
        tasks=args.task,
        skills=args.skills,
        strategies=args.strategy,
    )
    if not results:
        print("❌ No results match the given filters.", file=sys.stderr)
        sys.exit(1)

    # Reports
    print_overview(results)
    print_model_x_skills(results)

    if not args.no_tasks:
        print_task_details(results)

    if args.errors:
        print_error_breakdown(results)

    if args.crashes:
        print_crash_report(results)

    if args.matrix:
        print_progress_matrix(results)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
