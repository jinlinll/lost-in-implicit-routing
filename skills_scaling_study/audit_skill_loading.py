#!/usr/bin/env python3
"""Re-label historical skill loads with a conservative, auditable v3 parser.

Never overwrites the legacy skills_loaded.json.  Each completed trajectory gets
skills_loaded.strict-v3.json so downstream analyses can consume one fixed rule.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from analyze_skill_substitution import (
    LOGS_DIR,
    Skill,
    golden_skills,
    pool_catalog,
    recorded_skill_names,
)


CAT_RE = re.compile(
    r"cat\s+['\"]?((?:[~\w/.*?:-]+)?/skills/([^/'\"\s]+)/SKILL\.md)['\"]?"
)


def make_alias_map(task: str, catalog: Dict[str, Skill]) -> Dict[str, Skill]:
    aliases = dict(catalog)
    for skill in golden_skills(task):
        for alias in skill.aliases:
            aliases[alias] = skill
    return aliases


def context_name(name: str, available: set[str], aliases: Dict[str, Skill]) -> Optional[str]:
    """Return the exact available metadata name represented by a request."""
    if name in available:
        return name
    normalized = {name.replace("_", "-"), name.replace("-", "_")}
    for value in normalized:
        if value in available:
            return value
    skill = aliases.get(name)
    if skill:
        for alias in skill.aliases:
            if alias in available:
                return skill.name if skill.name in available else alias
    return None


def successful_cat_events(run_log: Path) -> List[dict]:
    events: List[dict] = []
    if not run_log.exists():
        return events
    episode = -1
    pending: Dict[str, dict] = {}
    for line in run_log.read_text(errors="replace").splitlines():
        episode_match = re.search(r"\[Episode (\d+)\] Starting", line)
        if episode_match:
            episode = int(episode_match.group(1))
        running = re.search(r"\[Exec (\d+/\d+)\] Running: (.*)", line)
        if running:
            matches = [
                {"path": match.group(1), "requested_name": match.group(2)}
                for match in CAT_RE.finditer(running.group(2))
            ]
            pending[running.group(1)] = {"episode": episode, "matches": matches}
            continue
        completed = re.search(
            r"\[Exec (\d+/\d+)\] Completed .*\(exit=(\d+), output=(\d+) chars\)", line
        )
        if completed:
            item = pending.pop(completed.group(1), {"episode": episode, "matches": []})
            for match in item["matches"]:
                events.append({
                    "episode": item["episode"],
                    "method": "cat_skill_md",
                    **match,
                    "exit_code": int(completed.group(2)),
                    "output_chars": int(completed.group(3)),
                })
    return events


def reparse_pass(pass_dir: Path, catalog: Dict[str, Skill]) -> Optional[dict]:
    result_path = pass_dir / "result.json"
    if not result_path.exists():
        return None
    task = pass_dir.parents[3].name
    available_list = recorded_skill_names(pass_dir)
    available = set(available_list)
    aliases = make_alias_map(task, catalog)
    accepted: List[dict] = []
    rejected: List[dict] = []

    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for entry in result.get("agent_result", {}).get("loaded_skills", []):
        requested = entry.get("skill", "") if isinstance(entry, dict) else entry
        episode = entry.get("episode", -1) if isinstance(entry, dict) else -1
        if not isinstance(requested, str) or not requested:
            continue
        canonical = context_name(requested, available, aliases)
        event = {
            "episode": episode,
            "method": "load_skill_json",
            "requested_name": requested,
            "canonical_available_name": canonical,
            "source": "result.json",
        }
        if canonical:
            accepted.append(event)
        else:
            event["rejection_reason"] = "not_in_recorded_available_skills"
            rejected.append(event)

    for event in successful_cat_events(pass_dir / "run.log"):
        requested = event["requested_name"]
        if event["exit_code"] != 0 or event["output_chars"] <= 0:
            event["rejection_reason"] = "cat_failed_or_empty"
            rejected.append(event)
            continue
        if requested == "*":
            for canonical in available_list:
                accepted.append({
                    **event,
                    "canonical_available_name": canonical,
                    "wildcard_expansion": True,
                })
            continue
        canonical = context_name(requested, available, aliases)
        event["canonical_available_name"] = canonical
        if canonical:
            accepted.append(event)
        else:
            event["rejection_reason"] = "not_in_recorded_available_skills"
            rejected.append(event)

    deduped: List[dict] = []
    seen = set()
    for event in sorted(accepted, key=lambda x: (x.get("episode", -1), x["method"])):
        name = event["canonical_available_name"]
        if name not in seen:
            seen.add(name)
            deduped.append(event)

    return {
        "parser_version": "strict-v3-context-checked",
        "trust_rules": {
            "load_skill_json": "accepted only if it resolves to a skill in this run's recorded available-skills list",
            "cat_skill_md": "accepted only if command executed, exit=0, output>0, and path resolves to a recorded available skill",
            "wildcards": "a successful */SKILL.md read expands to the recorded available-skills list",
            "legacy_warning": "strict-v2 accepted failed/empty cat attempts; runtime result.json can log Skill not found because the handler returns a truthy message",
        },
        "available_skills_source": "run.log [Skills] Found",
        "available_skill_count": len(available_list),
        "accepted_events": deduped,
        "skill_names": sorted(seen),
        "count": len(seen),
        "rejected_events": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--write-per-pass", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    catalog = pool_catalog()
    pass_dirs = sorted({p.parent for p in args.logs_dir.glob("*/*/skills_*/seed_*/pass_*/result.json")})
    summary = {
        "parser_version": "strict-v3-context-checked",
        "logs_dir": str(args.logs_dir.resolve()),
        "completed_passes_seen": 0,
        "accepted_unique_skill_loads": 0,
        "rejected_events": 0,
        "passes_with_rejections": 0,
        "rejection_reasons": {},
    }
    for pass_dir in pass_dirs:
        parsed = reparse_pass(pass_dir, catalog)
        if parsed is None:
            continue
        summary["completed_passes_seen"] += 1
        summary["accepted_unique_skill_loads"] += parsed["count"]
        summary["rejected_events"] += len(parsed["rejected_events"])
        summary["passes_with_rejections"] += bool(parsed["rejected_events"])
        for event in parsed["rejected_events"]:
            reason = event["rejection_reason"]
            summary["rejection_reasons"][reason] = summary["rejection_reasons"].get(reason, 0) + 1
        if args.write_per_pass:
            (pass_dir / "skills_loaded.strict-v3.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2)
            )

    summary_path = args.summary or args.logs_dir / "skills_loaded.strict-v3.audit-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
