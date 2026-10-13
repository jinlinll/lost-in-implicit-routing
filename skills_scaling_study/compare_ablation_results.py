#!/usr/bin/env python3
"""Compare ablation experiments (masked, nudge) vs naive baseline on the 10-task subset."""

import json, re, math, collections
from pathlib import Path

from parse_skills_loaded import extract_cat_skills_from_run_log

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT.parent / "skillsbench" / "tasks"

SUBSET_TASKS = [
    "adaptive-cruise-control", "data-to-d3", "earthquake-phase-association",
    "econ-detrending-correlation", "enterprise-information-search",
    "manufacturing-equipment-maintenance", "react-performance-debugging",
    "reserves-at-risk-calc", "simpo-code-reproduction", "weighted-gdp-calc",
]
MODELS = ["glm-5", "kimi-k2.5", "minimax-m2.5"]

EXPERIMENTS = {
    "naive":  ROOT / "experiment_logs",
    "masked": ROOT / "experiment_logs-ablation-masked",
    "nudge":  ROOT / "experiment_logs-ablation-nudge",
}

def get_golden_skill_names(task_name):
    skills_dir = TASKS_DIR / task_name / "environment" / "skills"
    names = set()
    if not skills_dir.exists():
        return names
    for d in sorted(skills_dir.iterdir()):
        skill_md = d / "SKILL.md"
        if not d.is_dir() or not skill_md.exists():
            continue
        content = skill_md.read_text(errors="replace")
        m = re.search(r"^name:\s*(.+)", content, re.MULTILINE)
        name = m.group(1).strip().strip('"').strip("'") if m else d.name
        names.add(name)
        names.add(d.name)  # folder name as alias
    return names


def get_all_loaded_skills(pass_dir):
    loaded = set()
    result_path = pass_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
        except:
            return loaded
        for entry in data.get("agent_result", {}).get("loaded_skills", []):
            if isinstance(entry, dict):
                loaded.add(entry.get("skill", ""))
            elif isinstance(entry, str):
                loaded.add(entry)

    run_log = pass_dir / "run.log"
    if run_log.exists():
        log_text = run_log.read_text(errors="replace")
        for episode, full_path, folder_name in extract_cat_skills_from_run_log(log_text):
            loaded.add(folder_name)

    loaded.discard("")
    return loaded


def pass_at_k(n, c, k):
    if n <= 0: return 0.0
    k = min(k, n)
    if n - c < k: return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def collect_all():
    all_data = []
    for exp_name, exp_dir in EXPERIMENTS.items():
        if not exp_dir.exists():
            print(f"WARNING: {exp_dir} not found, skipping {exp_name}")
            continue
        for task in SUBSET_TASKS:
            golden = get_golden_skill_names(task)
            for model in MODELS:
                base = exp_dir / task / model / "skills_1000" / "seed_42"
                for p in range(3):
                    pass_dir = base / f"pass_{p}"
                    result_path = pass_dir / "result.json"
                    if not result_path.exists():
                        continue
                    try:
                        data = json.loads(result_path.read_text())
                    except:
                        continue

                    err = data.get("agent_result", {}).get("error", "") or ""
                    if any(x in err.lower() for x in ["quota", "exhausted", "base_image_build", "readtimeout", "rate_limit"]):
                        continue

                    final_success = data.get("final_success", False)
                    loaded = get_all_loaded_skills(pass_dir)
                    golden_hits = loaded & golden
                    n_golden = len(golden)

                    all_data.append({
                        "experiment": exp_name,
                        "task": task,
                        "model": model,
                        "pass": p,
                        "success": final_success,
                        "n_loaded": len(loaded),
                        "n_golden": n_golden,
                        "golden_hits": len(golden_hits),
                        "individual_recall": len(golden_hits) / n_golden * 100 if n_golden > 0 else 0,
                        "all_gold": golden_hits == golden and n_golden > 0,
                        "zero_load": len(loaded) == 0,
                    })
    return all_data


def analyze(all_data):
    print(f"\nTotal valid runs: {len(all_data)}")
    for exp in ["naive", "masked", "nudge"]:
        runs = [d for d in all_data if d["experiment"] == exp]
        print(f"  {exp}: {len(runs)} runs")

    # ═══════════════════════════════════════════════════════════════════
    # 1. Overall comparison (averaged over all models and tasks)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"OVERALL COMPARISON (10 tasks × 3 models × 3 passes, at s=1000)")
    print(f"{'='*90}")

    header = f"{'Metric':<30s}"
    for exp in ["naive", "masked", "nudge"]:
        header += f"{'  ' + exp:>15s}"
    header += "   masked-naive  nudge-naive"
    print(header)
    print("-" * 100)

    for metric_name, metric_key, is_pct in [
        ("End-to-end success rate", "success", True),
        ("Loaded-skill recall", "individual_recall", False),
        ("Zero-load rate", "zero_load", True),
        ("All-gold coverage", "all_gold", True),
        ("Avg skills loaded", "n_loaded", False),
    ]:
        vals = {}
        for exp in ["naive", "masked", "nudge"]:
            runs = [d for d in all_data if d["experiment"] == exp]
            if metric_key == "individual_recall":
                vals[exp] = sum(d[metric_key] for d in runs) / len(runs) if runs else 0
            elif is_pct:
                vals[exp] = sum(d[metric_key] for d in runs) / len(runs) * 100 if runs else 0
            else:
                vals[exp] = sum(d[metric_key] for d in runs) / len(runs) if runs else 0

        row = f"{metric_name:<30s}"
        for exp in ["naive", "masked", "nudge"]:
            if metric_key == "n_loaded":
                row += f"{vals[exp]:>14.1f}"
            else:
                row += f"{vals[exp]:>13.1f}%"
        # Deltas
        d_m = vals["masked"] - vals["naive"]
        d_n = vals["nudge"] - vals["naive"]
        if metric_key == "n_loaded":
            row += f"    {d_m:>+8.1f}     {d_n:>+8.1f}"
        else:
            row += f"    {d_m:>+7.1f}pp    {d_n:>+7.1f}pp"
        print(row)

    # ═══════════════════════════════════════════════════════════════════
    # 2. Pass@3 (task-level)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"PASS@3 (task-level, averaged over models)")
    print(f"{'='*90}")

    for exp in ["naive", "masked", "nudge"]:
        runs = [d for d in all_data if d["experiment"] == exp]
        groups = collections.defaultdict(list)
        for d in runs:
            groups[(d["task"], d["model"])].append(d["success"])
        p3_vals = [pass_at_k(len(v), sum(v), 3) * 100 for v in groups.values()]
        avg_p3 = sum(p3_vals) / len(p3_vals) if p3_vals else 0
        print(f"  {exp:>8s}: Pass@3 = {avg_p3:.1f}%  ({len(groups)} task×model groups)")

    # ═══════════════════════════════════════════════════════════════════
    # 3. By model
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"BY MODEL")
    print(f"{'='*90}")

    for metric_name, metric_key, is_pct in [
        ("Success rate", "success", True),
        ("Skill recall", "individual_recall", False),
        ("Zero-load rate", "zero_load", True),
    ]:
        print(f"\n  {metric_name}:")
        header = f"    {'Model':<20s}"
        for exp in ["naive", "masked", "nudge"]:
            header += f"{'  ' + exp:>12s}"
        print(header)

        for model in MODELS:
            row = f"    {model:<20s}"
            for exp in ["naive", "masked", "nudge"]:
                runs = [d for d in all_data if d["experiment"] == exp and d["model"] == model]
                if metric_key == "individual_recall":
                    v = sum(d[metric_key] for d in runs) / len(runs) if runs else 0
                elif is_pct:
                    v = sum(d[metric_key] for d in runs) / len(runs) * 100 if runs else 0
                else:
                    v = sum(d[metric_key] for d in runs) / len(runs) if runs else 0
                row += f"{v:>11.1f}%"
            print(row)

    # ═══════════════════════════════════════════════════════════════════
    # 4. By task
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"BY TASK (skill recall, averaged over models)")
    print(f"{'='*90}")

    header = f"    {'Task':<40s}"
    for exp in ["naive", "masked", "nudge"]:
        header += f"{'  ' + exp:>12s}"
    header += "  golden"
    print(header)

    for task in SUBSET_TASKS:
        golden = get_golden_skill_names(task)
        n_golden = len([n for n in golden if not any(c == '/' for c in n)]) // 2  # approx (name + folder dedupe)
        row = f"    {task:<40s}"
        for exp in ["naive", "masked", "nudge"]:
            runs = [d for d in all_data if d["experiment"] == exp and d["task"] == task]
            v = sum(d["individual_recall"] for d in runs) / len(runs) if runs else 0
            row += f"{v:>11.1f}%"
        row += f"  {len(get_golden_skill_names(task))//2}"  # approx
        print(row)


if __name__ == "__main__":
    data = collect_all()
    analyze(data)
