#!/usr/bin/env python3
"""Analyze all field-mask ablations against the s=1000 naive baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from analyze_skill_substitution import PRIMARY_MODELS, TASKS, golden_skills, loaded_names

MODELS = PRIMARY_MODELS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "field_masking"
EXPERIMENTS = {
    "naive": ROOT / "experiment_logs",
    "mask-name": ROOT / "experiment_logs-ablation-mask-name",
    "mask-description": ROOT / "experiment_logs-ablation-mask-description",
    "mask-location": ROOT / "experiment_logs-ablation-mask-location",
    "mask-name-location": ROOT / "experiment_logs-ablation-mask-name-location",
}


def collect_one(experiment: str, task: str, model: str, pass_index: int = 0) -> Optional[dict]:
    pass_dir = EXPERIMENTS[experiment] / task / model / "skills_1000" / "seed_42" / f"pass_{pass_index}"
    result_path = pass_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if ((pass_dir / "retry_required.json").exists() or result.get("infrastructure_failure")
            or result.get("exception") or not isinstance(result.get("agent_result"), dict)):
        return None

    gold = golden_skills(task)
    aliases = {alias: skill.folder for skill in gold for alias in skill.aliases}
    loaded = sorted(loaded_names(pass_dir))
    loaded_golden = [name for name in loaded if name in aliases]
    hits = {aliases[name] for name in loaded if name in aliases}
    preview = pass_dir / "skills_context_preview.txt"
    return {
        "experiment": experiment,
        "task": task,
        "model": model,
        "success": bool(result.get("final_success", False)),
        "n_loaded": len(loaded),
        "n_golden": len(gold),
        "n_golden_hits": len(hits),
        "n_loaded_golden": len(loaded_golden),
        "n_loaded_non_golden": len(loaded) - len(loaded_golden),
        "zero_load": not loaded,
        "golden_recall": len(hits) / len(gold) if gold else 0.0,
        "golden_precision": len(loaded_golden) / len(loaded) if loaded else None,
        "all_gold": len(hits) == len(gold) and bool(gold),
        "loaded_skills": loaded,
        "golden_hits": sorted(hits),
        "context_preview_present": preview.exists() if experiment != "naive" else None,
        "context_preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest() if preview.exists() else None,
        "error": result.get("agent_result", {}).get("error", "") or "",
    }


def summarize(rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in rows:
        groups.setdefault((row["experiment"], row["model"]), []).append(row)
        groups.setdefault((row["experiment"], "ALL"), []).append(row)
    result = []
    order = {name: index for index, name in enumerate(EXPERIMENTS)}
    for (experiment, model), items in sorted(groups.items(), key=lambda x: (order[x[0][0]], x[0][1])):
        precision = [row["golden_precision"] for row in items if row["golden_precision"] is not None]
        total_loaded = sum(row["n_loaded"] for row in items)
        total_loaded_golden = sum(row["n_loaded_golden"] for row in items)
        total_gold = sum(row["n_golden"] for row in items)
        total_golden_hits = sum(row["n_golden_hits"] for row in items)
        result.append({
            "experiment": experiment,
            "model": model,
            "n_runs": len(items),
            "success_rate": sum(row["success"] for row in items) / len(items),
            "avg_skills_loaded": sum(row["n_loaded"] for row in items) / len(items),
            "avg_non_golden_loaded": sum(row["n_loaded_non_golden"] for row in items) / len(items),
            "non_golden_load_rate": sum(row["n_loaded_non_golden"] > 0 for row in items) / len(items),
            "zero_load_rate": sum(row["zero_load"] for row in items) / len(items),
            "avg_golden_recall": sum(row["golden_recall"] for row in items) / len(items),
            "avg_golden_precision_nonzero": sum(precision) / len(precision) if precision else None,
            "micro_golden_recall": total_golden_hits / total_gold if total_gold else None,
            "micro_golden_precision": total_loaded_golden / total_loaded if total_loaded else None,
            "all_gold_rate": sum(row["all_gold"] for row in items) / len(items),
            "context_previews_present": sum(row["context_preview_present"] is True for row in items),
        })
    return result


def markdown(summary: List[dict]) -> str:
    pct = lambda value: "—" if value is None else f"{100 * value:.1f}%"
    lines = [
        "# Field-level masking results",
        "",
        "| Setting | Avg. skills loaded | ZeroLoad | Loaded-skill recall |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "naive": "Original metadata",
        "mask-name": "Mask distractor names only",
        "mask-description": "Mask distractor descriptions only",
        "mask-location": "Mask distractor locations only",
        "mask-name-location": "Mask distractor names + locations",
    }
    for row in (item for item in summary if item["model"] == "ALL"):
        lines.append(
            f"| {labels[row['experiment']]} | {row['avg_skills_loaded']:.2f} | "
            f"{pct(row['zero_load_rate'])} | {pct(row['avg_golden_recall'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = [
        row for experiment in EXPERIMENTS for task in TASKS for model in MODELS
        if (row := collect_one(experiment, task, model)) is not None
    ]
    counts = {experiment: sum(row["experiment"] == experiment for row in rows) for experiment in EXPERIMENTS}
    missing = {experiment: 30 - count for experiment, count in counts.items() if count != 30}
    if missing and not args.allow_incomplete:
        raise SystemExit(f"Incomplete inputs: {missing}; rerun with --allow-incomplete for a progress snapshot")

    summary = summarize(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / "summary.md").write_text(markdown(summary))
    if summary:
        with (OUT / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"input counts: {counts}")
    print(markdown(summary))
    print(f"Artifacts: {OUT}")


if __name__ == "__main__":
    main()
