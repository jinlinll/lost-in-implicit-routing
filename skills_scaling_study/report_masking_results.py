"""Recompute all six field-masking rows using the strict loader audit."""
import json
from analyze_field_masking import ROOT, OUT, EXPERIMENTS, MODELS, TASKS, collect_one, summarize

def main():
    EXPERIMENTS["masked"] = ROOT / "experiment_logs-ablation-masked"
    rows = []
    for exp in EXPERIMENTS:
        passes = range(3) if exp == "masked" else range(1)
        for task in TASKS:
            for model in MODELS:
                for pass_index in passes:
                    path = EXPERIMENTS[exp]/task/model/"skills_1000"/"seed_42"/f"pass_{pass_index}"
                    if not (path/"skills_loaded.strict-v3.json").is_file():
                        raise SystemExit(f"Missing strict-v3 labels: {path}. Run audit_skill_loading.py first.")
                    row = collect_one(exp,task,model,pass_index)
                    if row is None: raise SystemExit(f"Missing, invalid or retryable result: {path}")
                    rows.append(row)
    summary = [r for r in summarize(rows) if r["model"] == "ALL"]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"six_condition_summary.json").write_text(json.dumps(summary,indent=2))
    labels = ["Original metadata", "Mask names", "Mask descriptions", "Mask locations",
              "Mask names + locations", "Mask all distractor (reference)"]
    print("Setting | Avg. skills loaded | ZeroLoad | Loaded-skill recall")
    for label, row in zip(labels,summary):
        print(f"{label} | {row['avg_skills_loaded']:.2f} | {row['zero_load_rate']:.1%} | {row['avg_golden_recall']:.1%}")

if __name__ == "__main__": main()
