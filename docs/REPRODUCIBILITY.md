# Experiment Guide

This repository provides the agent executor, skill preselection methods, masking and planning experiments, and result analysis tools.

## Setup

Use Python 3.10 or newer, Docker Engine with a Linux backend, and an OpenAI-compatible chat API. The experiments use Tencent TokenPlan for chat and Qwen `text-embedding-v4` for retrieval embeddings.

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r skills_scaling_study/requirements.txt
cp skills_scaling_study/.env.example skills_scaling_study/.env
```

Fill in your chat credentials in `.env`. For embedding computation, set `QWEN_EMBEDDING_API_KEY`. Keep credentials out of Git.

Set resource budgets to fit your Docker host or VM, for example:

```sh
export SKILLS_DOCKER_TOTAL_CPUS=8
export SKILLS_DOCKER_TOTAL_MEMORY_GB=6
```

The launcher defaults to one worker and supports up to four. Use a dedicated evaluation environment for third-party task containers.

## Benchmark and Skills

Prepare SkillsBench from the repository root:

```sh
git clone https://github.com/benchflow-ai/skillsbench.git skillsbench
git -C skillsbench checkout 17dec325f0f0eb7431009373387bd66de8d3eb1a
git -C skillsbench apply ../manifests/benchmark_environment.patch
```

[The benchmark manifest](../manifests/benchmark.json) records the version, 82 main-study tasks, and five exclusions. The patch contains the environment and verifier changes used in our experiments.

Due to licensing restrictions on some skills, we provide [source links](SKILL_SOURCES.md) rather than skill contents or download scripts. Readers can obtain skills independently under their original licenses. External repositories and files may have changed since our experiments.

Place the downloaded inputs in this layout:

```text
lost-in-implicit-routing/
  skillsbench/tasks/
  manifests/
  skills_scaling_study/
    .agents/skills/<skill-directory>/SKILL.md
    rag_precomputed/
    experiment_logs*/
```

[The skill pool manifest](../manifests/skill_pool.json) records directory identifiers, definition hashes, and sampling order. Include each skill's accompanying resources. Check the prepared inputs with:

```sh
python verify_data.py
```

Sampling uses seed 42 and the recorded pool order. The launcher requires the complete pool.

## Running Experiments

Run the following commands from `skills_scaling_study/`.

```sh
python -B -m unittest -v test_routing
python run_study.py --strategy naive --models glm-5 \
  --tasks adaptive-cruise-control --counts 0 --passes 1
```

The second command previews the configuration. Add `--run` to start API calls and Docker execution. Begin with a short test in a separate output directory:

```sh
python run_study.py --strategy naive --models glm-5 \
  --tasks adaptive-cruise-control --counts 0 --passes 1 \
  --max-parallel 1 --max-episodes 2 --output-dir smoke_baseline --run
```

For the main study:

```sh
python run_study.py --strategy naive --counts 0 10 100 1000 \
  --passes 3 --max-parallel 4 --run
```

Choose `naive`, `llm`, `rag`, or `multi-query-rag-llm` (hybrid). Preselection retains at most five skills. RAG and hybrid also support 10,000 distractors. Selection coverage and skills loaded during execution are recorded separately.

Before RAG or hybrid experiments, compute the embedding cache:

```sh
python precompute_rag.py --skills-counts 0 10 100 1000 10000 \
  --output-dir rag_precomputed
```

## Masking and Plan-First Prompting

Full masking and plan-first prompting use three repetitions:

```sh
python run_masking_and_planning.py --experiment masked --passes 3 --max-parallel 4 --run
python run_masking_and_planning.py --experiment plan-first --passes 3 --max-parallel 4 --run
```

Field-level conditions use one repetition:

```sh
python run_masking_and_planning.py --experiment mask-name --passes 1 --max-parallel 4 --run
python run_masking_and_planning.py --experiment mask-description --passes 1 --max-parallel 4 --run
python run_masking_and_planning.py --experiment mask-location --passes 1 --max-parallel 4 --run
python run_masking_and_planning.py --experiment mask-name-location --passes 1 --max-parallel 4 --run
```

Execute these commands sequentially to keep total concurrency within four. `plan-first` retains the existing `experiment_logs-ablation-nudge/` output directory for resume compatibility; `nudge` remains an accepted alias.

For an initial context check, add `--models glm-5 --tasks adaptive-cruise-control --max-parallel 1 --max-episodes 2 --output-dir smoke_mask`, then inspect the saved preview:

```sh
python validate_skills_context.py \
  smoke_mask/adaptive-cruise-control/glm-5/skills_1000/seed_42/pass_0/skills_context_preview.txt \
  --task adaptive-cruise-control --masked-fields name location
```

Use the matching fields for each condition. Masking preserves golden metadata and installed skill files; placeholders approximately preserve character length.

## Logs, Resume, and Analysis

Outputs follow `<log-root>/<task>/<model>/skills_<count>/seed_42/pass_<index>/`. The launcher resumes automatically. Results marked for retry, infrastructure failures, and incomplete results are retried; normal task failures remain valid outcomes. Previous attempts are stored in `_retry_archive`.

Generate context-checked skill-loading labels for each log directory:

```sh
python audit_skill_loading.py --logs-dir experiment_logs --write-per-pass
```

The analysis tools use these `strict-v3` labels:

| Analysis | Command | Output |
|---|---|---|
| Non-golden substitutions | `python analyze_skill_substitution.py --run` | `analysis/skill_substitution/` |
| Field-level masking | `python analyze_field_masking.py` | `analysis/field_masking/` |
| Masking comparison table | `python report_masking_results.py` | Console; `analysis/field_masking/` |
| Preselection latency | `python analyze_routing_latency.py` | `analysis/routing_latency/` |
| Task and skill metrics | `python analyze_study_metrics.py --all-strategies` | Console; optional CSV |

The substitution analysis compares loaded non-golden skills with missed golden skills using full `name: description` embeddings and cosine similarity above 0.8. Use `--smoke` to inspect one example. The paper's substitution analysis covers an 86-task archive across all three repetitions; the main study uses 82 tasks.

The masking table requires complete results and audited labels for the baseline and five masking conditions. The latency analysis reads `experiment_logs-llm`, `experiment_logs-rag`, and `experiment_logs-mq-rag-llm-fixed`, measuring routing-stage delay rather than end-to-end execution time.

`analyze_planned_skill_loading.py` compares planned and loaded skills using LLM annotation; set `API_BASE` and `API_KEY` before using it. Candidate aggregation experiments are provided in `simulate_multi_query_rag.py` and `simulate_multi_query_selection_variants.py`.

## Data Availability

We provide experiment code, configurations, benchmark modifications, and skill source lists. Third-party skill contents and resources, benchmark task payloads, raw trajectories, precomputed embeddings, API credentials, and provider model snapshots are not included. See [Data and Licenses](DATA_AND_LICENSES.md).
