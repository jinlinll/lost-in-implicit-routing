#!/usr/bin/env python3
"""Extract planned skills from initial responses and compare with loaded skills.

For each valid run in the 10-task nudge subset, the script gives an LLM the
available skill names and the first model response.  The LLM returns which
available skills the agent appears to consider or plan to use.  We compare that
planned set with golden skills and with the merged executed-load record
(explicit load_skill events plus direct SKILL document reads from run logs).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from compare_ablation_results import MODELS, SUBSET_TASKS, get_golden_skill_names
from parse_skills_loaded import parse_pass_directory

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
NUDGE_ROOT = ROOT / "experiment_logs-ablation-nudge"
OUT_PATH = ROOT / "analysis" / "planned_skill_loading.json"

PROMPT_TEMPLATE = """You extract skill planning decisions from an agent's first response.

Return ONLY valid JSON with this exact schema:
{{"planned_skills": ["exact-skill-name-1", "exact-skill-name-2"]}}

Rules:
- Choose only names from AVAILABLE_SKILLS.
- Include a skill if the first response says or strongly implies that the agent plans to load, use, or consider that skill.
- If the response lists relevant skills before doing work, include all listed relevant skills even if only one is actually loaded later in the same response.
- Do not infer skills from the task alone. If the agent does not name or clearly describe a skill in its plan, return an empty list.
- Preserve the exact name from AVAILABLE_SKILLS.

AVAILABLE_SKILLS:
{available_skills}

Conservative pre-scan candidates from the response:
{candidates}

AGENT_FIRST_RESPONSE:
<<<RESPONSE
{response}
RESPONSE>>>
"""


def parse_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def endpoint_from_env(env: dict[str, str]) -> str:
    url = env.get("Tencent_Cloud_Base_Url") or env.get("API_BASE") or os.environ.get("API_BASE") or ""
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url:
        return f"{url}/chat/completions"
    raise RuntimeError("Missing Tencent_Cloud_Base_Url/API_BASE")


def api_key_from_env(env: dict[str, str]) -> str:
    key = env.get("Tencent_Cloud_Key") or env.get("API_KEY") or os.environ.get("API_KEY") or ""
    if not key:
        raise RuntimeError("Missing Tencent_Cloud_Key/API_KEY")
    return key


def is_valid_run(data: dict[str, Any]) -> bool:
    err = (data.get("agent_result", {}).get("error") or "").lower()
    return not any(x in err for x in ["quota", "exhausted", "base_image_build", "readtimeout", "read timeout", "rate_limit"])


def loaded_skill_record(pass_dir: Path) -> tuple[list[str], dict[str, int]]:
    parsed = parse_pass_directory(pass_dir)
    if parsed:
        return sorted(parsed.get("skill_names", [])), dict(parsed.get("sources", {}))
    loaded: set[str] = set()
    result_path = pass_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
            for entry in data.get("agent_result", {}).get("loaded_skills", []):
                if isinstance(entry, dict) and entry.get("skill"):
                    loaded.add(entry["skill"])
                elif isinstance(entry, str) and entry:
                    loaded.add(entry)
        except Exception:
            pass
    return sorted(loaded), {"fallback_result_json": len(loaded)}


def available_skill_names(data: dict[str, Any], pass_dir: Path) -> list[str]:
    names = data.get("agent_result", {}).get("preselected_skills") or []
    out = [str(x) for x in names if isinstance(x, str) and x]
    if out:
        return sorted(dict.fromkeys(out))

    prompt_path = pass_dir / "episode-0" / "prompt.txt"
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"available_skills:\s*\n(\[.*?\])", text, re.DOTALL)
        if m:
            try:
                skills = json.loads(m.group(1))
                return sorted({str(s.get("name")) for s in skills if isinstance(s, dict) and s.get("name")})
            except Exception:
                pass
    return []


def collect_runs() -> list[dict[str, Any]]:
    runs = []
    for task in SUBSET_TASKS:
        golden = get_golden_skill_names(task)
        for model in MODELS:
            base = NUDGE_ROOT / task / model / "skills_1000" / "seed_42"
            for pass_idx in range(3):
                pass_dir = base / f"pass_{pass_idx}"
                result_path = pass_dir / "result.json"
                response_path = pass_dir / "episode-0" / "response.txt"
                if not result_path.exists() or not response_path.exists():
                    continue
                try:
                    data = json.loads(result_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
                if not is_valid_run(data):
                    continue
                loaded, loaded_sources = loaded_skill_record(pass_dir)
                runs.append({
                    "run_id": f"{task}/{model}/pass_{pass_idx}",
                    "task": task,
                    "model": model,
                    "pass": pass_idx,
                    "success": bool(data.get("final_success")),
                    "golden_skills": sorted(golden),
                    "loaded_skills": loaded,
                    "loaded_sources": loaded_sources,
                    "available_skills": available_skill_names(data, pass_dir),
                    "episode0_response": response_path.read_text(encoding="utf-8", errors="replace"),
                })
    return runs


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"planned_skills": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"planned_skills": []}


def normalize_skill_list(value: Any, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    lower_to_name = {x.lower(): x for x in allowed or set()}
    for item in value:
        if not isinstance(item, str):
            continue
        name = item.strip().strip("` '").strip()
        if allowed is not None:
            if name not in allowed and name.lower() in lower_to_name:
                name = lower_to_name[name.lower()]
            if name not in allowed:
                continue
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def prescan_planned_skills(response: str, allowed: set[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    lower_to_name = {x.lower(): x for x in allowed}

    def add(name: str) -> None:
        name = name.strip().strip("`'\" ").strip()
        if name not in allowed and name.lower() in lower_to_name:
            name = lower_to_name[name.lower()]
        if name in allowed and name not in seen:
            candidates.append(name)
            seen.add(name)

    for m in re.finditer(r'"load_skill"\s*:\s*"([^"]+)"', response):
        add(m.group(1))

    lines = response.splitlines()
    in_load_block = False
    for raw in lines:
        line = raw.strip()
        low = line.lower()
        if any(phrase in low for phrase in ["load the relevant skills", "load these relevant skills", "relevant skills are", "most relevant skills", "available skills"]):
            in_load_block = True
        if in_load_block:
            for m in re.finditer(r"`([^`]+)`", line):
                add(m.group(1))
            for name in allowed:
                if re.search(rf"\b{re.escape(name)}\b", line, re.IGNORECASE):
                    add(name)
        if in_load_block and any(marker in low for marker in ["let me first", "examine", "explore", "loading the first skill"]):
            in_load_block = False
    return candidates


def http_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", {}) if isinstance(data, dict) else {}
        message = err.get("message") or resp.text[:300]
        code = err.get("code")
        return f"HTTP {resp.status_code}: {message}" + (f" ({code})" if code else "")
    except Exception:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"


def call_llm(run: dict[str, Any], endpoint: str, api_key: str, model: str, retries: int) -> dict[str, Any]:
    allowed = set(run.get("available_skills", []))
    prescan = prescan_planned_skills(run["episode0_response"], allowed)
    prompt = PROMPT_TEMPLATE.format(
        available_skills=json.dumps(run.get("available_skills", []), ensure_ascii=False),
        response=run["episode0_response"][:20000],
        candidates=json.dumps(prescan, ensure_ascii=False),
    )
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_tokens": 512,
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = http_error(resp)
                time.sleep(min(60, 10 * attempt))
                continue
            if resp.status_code != 200:
                return {**run, "planned_skills": prescan, "extract_error": http_error(resp)}
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
            parsed = extract_json_object(str(content))
            llm_planned = normalize_skill_list(parsed.get("planned_skills"), allowed)
            planned = sorted(dict.fromkeys(llm_planned + prescan))
            return {
                **run,
                "planned_skills": planned,
                "llm_planned_skills": llm_planned,
                "prescan_planned_skills": prescan,
                "llm_raw_response": str(content),
                "extract_error": "",
            }
        except Exception as exc:
            last_error = type(exc).__name__
            if attempt < retries:
                time.sleep(min(60, 10 * attempt))
    return {
        **run,
        "planned_skills": prescan,
        "llm_planned_skills": [],
        "prescan_planned_skills": prescan,
        "llm_raw_response": "",
        "extract_error": last_error or "unknown",
    }


def annotate(record: dict[str, Any]) -> dict[str, Any]:
    planned = set(record.get("planned_skills", []))
    golden = set(record.get("golden_skills", []))
    loaded = set(record.get("loaded_skills", []))
    planned_golden = planned & golden
    return {
        k: v for k, v in record.items() if k != "episode0_response"
    } | {
        "planned_golden_skills": sorted(planned_golden),
        "planned_not_loaded": sorted(planned - loaded),
        "planned_golden_not_loaded": sorted(planned_golden - loaded),
        "has_planned_golden_gap": bool(planned_golden - loaded),
        "planned_recall": len(planned_golden) / len(golden) * 100 if golden else 0.0,
        "planned_all_gold": bool(golden) and planned_golden == golden,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("extract_error")]
    with_planned_golden = [r for r in ok if r.get("planned_golden_skills")]
    gap = [r for r in ok if r.get("has_planned_golden_gap")]
    planned_golden_total = sum(len(r.get("planned_golden_skills", [])) for r in ok)
    planned_golden_missed = sum(len(r.get("planned_golden_not_loaded", [])) for r in ok)
    error_counts = Counter(r.get("extract_error") for r in records if r.get("extract_error"))
    return {
        "valid_runs": len(records),
        "extracted_runs": len(ok),
        "failed_extractions": len(records) - len(ok),
        "extract_error_counts": dict(error_counts.most_common()),
        "runs_with_planned_golden": len(with_planned_golden),
        "runs_with_planned_golden_gap": len(gap),
        "gap_rate_all_extracted": len(gap) / len(ok) * 100 if ok else 0.0,
        "gap_rate_among_planned_golden": len(gap) / len(with_planned_golden) * 100 if with_planned_golden else 0.0,
        "planned_golden_total": planned_golden_total,
        "planned_golden_not_loaded_total": planned_golden_missed,
        "planned_golden_load_recall": (planned_golden_total - planned_golden_missed) / planned_golden_total * 100 if planned_golden_total else 0.0,
        "planned_skill_recall": sum(r.get("planned_recall", 0.0) for r in ok) / len(ok) if ok else 0.0,
        "planned_all_gold": sum(float(r.get("planned_all_gold", False)) for r in ok) / len(ok) * 100 if ok else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N runs for API debugging")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    workers = min(4, max(1, args.workers))

    runs = collect_runs()
    if args.limit > 0:
        runs = runs[:args.limit]
    existing: dict[str, dict[str, Any]] = {}
    if args.out.exists() and not args.force:
        try:
            old = json.loads(args.out.read_text(encoding="utf-8"))
            existing = {r["run_id"]: r for r in old.get("runs", []) if r.get("run_id")}
        except Exception:
            existing = {}

    env = parse_env(ENV_PATH)
    endpoint = endpoint_from_env(env)
    api_key = api_key_from_env(env)

    results: list[dict[str, Any]] = []
    todo = []
    for run in runs:
        cached = existing.get(run["run_id"])
        if cached and not cached.get("extract_error") and "planned_skill_recall" in cached:
            results.append(cached)
        else:
            todo.append(run)

    if todo:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(call_llm, run, endpoint, api_key, args.model, args.retries) for run in todo]
            for fut in futures.as_completed(futs):
                results.append(annotate(fut.result()))

    results.sort(key=lambda r: (r["task"], r["model"], r["pass"]))
    summary = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_extractions"]:
        print("Top extraction errors:")
        for error, count in summary.get("extract_error_counts", {}).items():
            print(f"  {count}x {error}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
