#!/usr/bin/env python3
"""Validate and summarize an exact skills-context preview saved by a test run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT.parent / "skillsbench" / "tasks"


def golden_names(task_name: str) -> set[str]:
    result: set[str] = set()
    skills_dir = TASKS_DIR / task_name / "environment" / "skills"
    for directory in sorted(skills_dir.iterdir()):
        skill_file = directory / "SKILL.md"
        if not directory.is_dir() or not skill_file.exists():
            continue
        content = skill_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^name:\s*(.+)", content, re.MULTILINE)
        result.add(match.group(1).strip().strip('"').strip("'") if match else directory.name)
    return result


def shorten(value: str, limit: int = 180) -> str:
    value = value.replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preview", type=Path)
    parser.add_argument("--task", required=True)
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument(
        "--masked-field",
        choices=["name", "description", "location"],
        help="Validate a single masked distractor field",
    )
    mask_group.add_argument(
        "--masked-fields",
        nargs="+",
        choices=["name", "description", "location"],
        help="Validate multiple masked distractor fields",
    )
    args = parser.parse_args()

    text = args.preview.read_text(encoding="utf-8", errors="replace")
    marker = "available_skills:\n"
    start = text.index(marker) + len(marker)
    entries, _ = json.JSONDecoder().raw_decode(text[start:])

    golden = golden_names(args.task)
    golden_entries = [item for item in entries if item["name"] in golden]
    distractor_entries = [item for item in entries if item["name"] not in golden]
    targets = set(args.masked_fields or [args.masked_field])
    other_fields = {"name", "description", "location"} - targets

    errors: list[str] = []
    if len(golden_entries) != len(golden):
        errors.append(f"golden entry count {len(golden_entries)} != expected {len(golden)}")
    for item in golden_entries:
        if any("<mask>" in str(item[field]) for field in ("name", "description", "location")):
            errors.append(f"golden skill unexpectedly masked: {item['name']}")
    for item in distractor_entries:
        for target in targets:
            if "<mask>" not in str(item[target]):
                errors.append(
                    f"distractor target field {target} is readable: "
                    f"{shorten(str(item[target]), 60)}"
                )
                break
        for field in other_fields:
            if "<mask>" in str(item[field]):
                errors.append(f"distractor non-target field {field} was masked")
                break
        if errors:
            break

    print(f"preview={args.preview}")
    print(f"task={args.task}")
    print(f"total_skills={len(entries)} golden={len(golden_entries)} distractors={len(distractor_entries)}")
    print(f"masked_fields={','.join(sorted(targets))}")
    print(f"validation={'PASS' if not errors else 'FAIL'}")
    for error in errors:
        print(f"error={error}")

    if golden_entries:
        print("golden_sample=" + json.dumps(golden_entries[0], ensure_ascii=False))
    if distractor_entries:
        sample = {key: shorten(str(value)) for key, value in distractor_entries[0].items()}
        print("distractor_sample=" + json.dumps(sample, ensure_ascii=False))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
