"""Safe command-line entry point; preview by default, explicit --run for API/Docker work."""
import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from parallel_runner import ParallelExperimentConfig, ParallelExperimentRunner
from tokenplan_client import get_default_tokenplan_api_base, get_default_tokenplan_api_key

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["glm-5", "kimi-k2.5", "minimax-m2.5"])
    p.add_argument("--tasks", nargs="+", help="Default: the 82 main-study tasks in the manifest")
    p.add_argument("--tasks-dir", type=Path, default=ROOT.parent/"skillsbench/tasks")
    p.add_argument("--skills-pool-dir", type=Path, default=ROOT/".agents/skills")
    p.add_argument("--counts", type=int, nargs="+", default=[0,10,100,1000])
    p.add_argument("--strategy", choices=["naive","llm","rag","multi-query-rag-llm"], default="naive")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--max-parallel", type=int, choices=range(1,5), default=1)
    p.add_argument("--max-episodes", type=int, default=50)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--rag-dir", type=Path, default=ROOT/"rag_precomputed")
    p.add_argument("--run", action="store_true", help="Permit actual paid API calls and Docker execution")
    a = p.parse_args()
    if a.passes < 1 or a.max_episodes < 1 or any(c < -1 for c in a.counts):
        p.error("Pass/episode counts must be positive and extra-skill counts >= -1")
    tasks = a.tasks or json.loads((ROOT.parent/"manifests/benchmark.json").read_text())["main_tasks"]
    output_names = {"naive":"experiment_logs", "llm":"experiment_logs-llm", "rag":"experiment_logs-rag",
                    "multi-query-rag-llm":"experiment_logs-mq-rag-llm-fixed"}
    output = a.output_dir or ROOT/output_names[a.strategy]
    print(json.dumps({"models":a.models,"tasks":tasks,"counts":a.counts,"strategy":a.strategy,
                      "passes":a.passes,"max_parallel":a.max_parallel,"resume":True,
                      "output":str(output),"execute":a.run}, indent=2))
    if not a.run:
        print("Preview only. Add --run after preparing the benchmark, skills pool and API credentials.")
        return
    if not a.tasks_dir.is_dir(): p.error("Benchmark tasks are missing; see README.md")
    for name in tasks:
        if not (a.tasks_dir/name/"instruction.md").is_file(): p.error(f"Missing task: {name}")
    if any(c > 0 for c in a.counts):
        from pool_order import ordered_pool_paths
        ordered_pool_paths(a.skills_pool_dir)
    key = os.environ.get("API_KEY") or get_default_tokenplan_api_key()
    if not key: p.error("Set TENCENT_TOKENPLAN_API_KEY or API_KEY")
    subprocess.run(["docker","info"], stdout=subprocess.DEVNULL, check=True)
    config = ParallelExperimentConfig(
        model_names=a.models, api_base=os.environ.get("API_BASE") or get_default_tokenplan_api_base(),
        api_key=key, task_names=tasks, tasks_dir=a.tasks_dir, extra_skills_counts=a.counts,
        num_passes=a.passes, random_seed=42, max_episodes=a.max_episodes,
        preselect_strategy=a.strategy, preselect_max_skills=5, hybrid_rag_retrieval_k=100,
        max_parallel_tasks=a.max_parallel, output_dir=output,
        skills_pool_dir=a.skills_pool_dir if any(c > 0 for c in a.counts) else None,
        rag_precomputed_dir=str(a.rag_dir))
    runner = ParallelExperimentRunner(config)
    results = asyncio.run(runner.run_all(resume=True, cleanup_images=False))
    runner.print_summary(results)
    runner.save_results(results)

if __name__ == "__main__":
    main()
