#!/usr/bin/env python3
"""
Parse skills loaded via `cat` commands from episode logs.

The agent can load skills in two ways:
  1. JSON: {"load_skill": "skill-name"}         — tracked in result.json
  2. cat:  cat /root/.terminus/skills/xxx/SKILL.md  — NOT tracked in result.json

This script scans every episode's response.txt for `cat ... SKILL.md` commands,
maps the path back to a skill name (from episode-0's prompt.txt available_skills),
merges with result.json's loaded_skills, and writes skills_loaded.json.

Usage:
    # Single pass directory
    python parse_skills_loaded.py experiment_logs-llm/task/model/skills_0/seed_42/pass_0

    # Batch: process an entire experiment_logs tree
    python parse_skills_loaded.py --batch experiment_logs-llm

    # Batch: process all experiment_logs* directories
    python parse_skills_loaded.py --batch-all

    # Dry-run (show what would be written without writing)
    python parse_skills_loaded.py --batch experiment_logs-llm --dry-run
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ─── Regex patterns ───────────────────────────────────────────────────────────

# Matches `cat <path>/skills/<skill-name>/SKILL.md` in keystrokes / commands.
# Handles diverse path prefixes found in the wild, including:
#   /root/.terminus/skills/threejs/SKILL.md
#   /root/.claude/skills/google-calendar-skill/SKILL.md
#   /root/.gemini/skills/video-frame-extraction/SKILL.md
#   /root/.agents/skills/modal-gpu/SKILL.md
#   /root/.codex/skills/gh-cli/SKILL.md
#   /root/.factory/skills/xxx/SKILL.md
#   ~/.terminus/skills/xxx/SKILL.md
#   /root/skills/xxx/SKILL.md
#   /etc/claude-code/.claude/skills/xxx/SKILL.md
# Group 1: full path, Group 2: skill directory name (used as fallback name)
CAT_SKILL_RE = re.compile(
    r"""cat\s+['"]?((?:[~\w/.:-]+)?/skills/([^/'"\s]+)/SKILL\.md)['"]?""",
)

# Matches {"load_skill": "name"} patterns in response text
LOAD_SKILL_JSON_RE = re.compile(
    r'\{\s*"load_skill[^"]*"\s*:\s*"([^"]+)"\s*\}',
)


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def parse_available_skills_from_prompt(prompt_text: str) -> Dict[str, str]:
    """
    Parse available_skills from episode-0 prompt.txt.

    Returns a dict mapping skill location path -> skill name.
    e.g. {"/root/.terminus/skills/threejs/SKILL.md": "threejs"}
    """
    path_to_name: Dict[str, str] = {}
    # Try parsing the JSON array after "available_skills:"
    match = re.search(r"available_skills:\s*\n(\[.*?\])", prompt_text, re.DOTALL)
    if match:
        try:
            skills_list = json.loads(match.group(1))
            for skill in skills_list:
                name = skill.get("name", "")
                location = skill.get("location", "")
                if name and location:
                    path_to_name[location] = name
        except json.JSONDecodeError:
            pass

    return path_to_name


def extract_cat_skills_from_response(response_text: str) -> List[Tuple[str, str]]:
    """
    Extract cat-loaded skills from executed command keystrokes in a response.

    This intentionally does not scan raw response text.  Raw text may contain
    plans, examples, or copied history, none of which proves that the command was
    executed in the current episode.
    """
    results = []
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return results
    if not isinstance(data, dict):
        return results
    commands = data.get("commands", [])
    if not isinstance(commands, list):
        return results
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        ks = cmd.get("keystrokes", "")
        if not isinstance(ks, str):
            continue
        for match in CAT_SKILL_RE.finditer(ks):
            pair = (match.group(1), match.group(2))
            if pair not in results:
                results.append(pair)
    return results


def extract_cat_skills_from_run_log(log_text: str) -> List[Tuple[int, str, str]]:
    """Extract skill reads from commands that the runtime actually executed."""
    results: List[Tuple[int, str, str]] = []
    current_episode: Optional[int] = None
    for line in log_text.splitlines():
        ep_match = re.search(r"\[Episode (\d+)\] Starting", line)
        if ep_match:
            current_episode = int(ep_match.group(1))
            continue
        if current_episode is None or "[Exec " not in line or " Running: " not in line:
            continue
        command = line.split(" Running: ", 1)[1]
        for match in CAT_SKILL_RE.finditer(command):
            item = (current_episode, match.group(1), match.group(2))
            if item not in results:
                results.append(item)
    return results


def extract_json_load_skills_from_response(response_text: str) -> List[str]:
    """Deprecated fallback kept for compatibility; result.json is authoritative."""
    return []


def parse_pass_directory(pass_dir: Path) -> Optional[dict]:
    """
    Parse a single pass directory to find all loaded skills.

    Returns a dict suitable for skills_loaded.json, or None if no episodes found
    or if result.json does not exist (indicating the task is still running).
    """
    pass_dir = Path(pass_dir)

    # ── 0. Only process completed runs (result.json must exist) ──
    if not (pass_dir / "result.json").exists():
        return None

    # ── 1. Read available_skills mapping from episode-0 prompt ──
    ep0_prompt = pass_dir / "episode-0" / "prompt.txt"
    path_to_name: Dict[str, str] = {}
    if ep0_prompt.exists():
        path_to_name = parse_available_skills_from_prompt(
            ep0_prompt.read_text(encoding="utf-8", errors="replace")
        )

    # ── 2. Enumerate episodes in order ──
    episode_dirs = sorted(
        [d for d in pass_dir.iterdir() if d.is_dir() and d.name.startswith("episode-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not episode_dirs:
        return None

    # ── 3. Runtime-tracked JSON load_skill events ──
    result_json_path = pass_dir / "result.json"
    result_loaded: List[Dict] = []
    if result_json_path.exists():
        try:
            result_data = json.loads(
                result_json_path.read_text(encoding="utf-8", errors="replace")
            )
            agent_result = result_data.get("agent_result", {})
            for entry in agent_result.get("loaded_skills", []):
                if not isinstance(entry, dict):
                    continue
                skill_name = entry.get("skill", "")
                if skill_name:
                    result_loaded.append({
                        "episode": entry.get("episode", -1),
                        "skill": skill_name,
                        "method": "load_skill_json",
                        "source": "result.json",
                    })
        except (json.JSONDecodeError, KeyError):
            pass

    # ── 4. Direct SKILL.md reads. Prefer run.log because it records executed commands. ──
    cat_loaded: List[Dict] = []
    cat_from_run_log = 0
    cat_from_response = 0
    run_log = pass_dir / "run.log"
    if run_log.exists():
        log_text = run_log.read_text(encoding="utf-8", errors="replace")
        for episode_num, full_path, dir_name in extract_cat_skills_from_run_log(log_text):
            skill_name = path_to_name.get(full_path, dir_name)
            cat_loaded.append({
                "episode": episode_num,
                "skill": skill_name,
                "method": "cat",
                "path": full_path,
                "source": "run.log",
            })
        cat_from_run_log = len(cat_loaded)
    else:
        for ep_dir in episode_dirs:
            episode_num = int(ep_dir.name.split("-")[1])
            response_file = ep_dir / "response.txt"
            if not response_file.exists():
                continue
            response_text = response_file.read_text(encoding="utf-8", errors="replace")
            for full_path, dir_name in extract_cat_skills_from_response(response_text):
                skill_name = path_to_name.get(full_path, dir_name)
                cat_loaded.append({
                    "episode": episode_num,
                    "skill": skill_name,
                    "method": "cat",
                    "path": full_path,
                    "source": "response.commands",
                })
        cat_from_response = len(cat_loaded)

    # ── 5. Combine all sources, sorted by episode and action type ──
    method_order = {"load_skill_json": 0, "cat": 1}
    all_loaded = result_loaded + cat_loaded
    all_loaded.sort(key=lambda x: (x.get("episode", 0), method_order.get(x.get("method", ""), 9)))

    # Deduplicate (keep first occurrence by skill name)
    final: List[Dict] = []
    final_names: Set[str] = set()
    for entry in all_loaded:
        if entry["skill"] not in final_names:
            final_names.add(entry["skill"])
            final.append(entry)

    return {
        "parser_version": "strict-executed-commands-v2",
        "skills_loaded": final,
        "skill_names": sorted(final_names),
        "count": len(final),
        "sources": {
            "cat_commands": len(cat_loaded),
            "cat_commands_from_run_log": cat_from_run_log,
            "cat_commands_from_response": cat_from_response,
            "load_skill_json_from_logs": 0,
            "load_skill_json_from_result": len(result_loaded),
        },
    }


def process_pass_dir(pass_dir: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Process a single pass directory.  Returns True if skills_loaded.json was
    written (or would be written in dry-run mode).
    """
    result = parse_pass_directory(pass_dir)
    if result is None:
        return False

    out_path = pass_dir / "skills_loaded.json"

    if dry_run:
        if verbose or result["count"] > 0:
            print(f"  [dry-run] {pass_dir.relative_to(pass_dir.parent.parent.parent.parent.parent)}: "
                  f"{result['count']} skill(s) "
                  f"(cat={result['sources']['cat_commands']}, "
                  f"json_log={result['sources']['load_skill_json_from_logs']}, "
                  f"json_result={result['sources']['load_skill_json_from_result']})")
        return result["count"] > 0

    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(f"  wrote {out_path.relative_to(pass_dir.parent.parent.parent.parent.parent)}: "
              f"{result['count']} skill(s)")
    return True


def find_pass_dirs(root: Path) -> List[Path]:
    """
    Recursively find pass directories under a log root.

    A pass directory is identified by having episode-0/ and result.json.
    Pattern: .../task/model/skills_N/seed_S/pass_P/
    """
    pass_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        if "result.json" in filenames and (dp / "episode-0").is_dir():
            pass_dirs.append(dp)
    return sorted(pass_dirs)


def discover_log_dirs(base: Path) -> List[Path]:
    """Find experiment_logs and experiment_logs-* directories."""
    results = []
    parent = base if base.is_dir() else base.parent
    for d in sorted(parent.iterdir()):
        if d.is_dir() and (d.name == "experiment_logs" or d.name.startswith("experiment_logs-")):
            results.append(d)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse skills loaded via cat commands from episode logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pass_dir", nargs="?", type=Path,
        help="Single pass directory to process",
    )
    parser.add_argument(
        "--batch", type=Path, metavar="LOGS_DIR",
        help="Process all pass directories under the given logs root",
    )
    parser.add_argument(
        "--batch-all", action="store_true",
        help="Process all experiment_logs* directories in the current folder",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without writing files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show details for each pass directory",
    )

    args = parser.parse_args()

    if args.batch_all:
        log_dirs = discover_log_dirs(Path("."))
        if not log_dirs:
            print("No experiment_logs* directories found in current folder.")
            return
        total_written = 0
        total_dirs = 0
        for log_dir in log_dirs:
            print(f"\n{'='*60}")
            print(f"Processing: {log_dir}")
            print(f"{'='*60}")
            pass_dirs = find_pass_dirs(log_dir)
            total_dirs += len(pass_dirs)
            for pd in pass_dirs:
                if process_pass_dir(pd, dry_run=args.dry_run, verbose=args.verbose):
                    total_written += 1
        action = "would write" if args.dry_run else "wrote"
        print(f"\nDone. {action} skills_loaded.json for {total_written}/{total_dirs} pass directories.")

    elif args.batch:
        print(f"Scanning {args.batch} ...")
        pass_dirs = find_pass_dirs(args.batch)
        print(f"Found {len(pass_dirs)} pass directories.")
        written = 0
        for pd in pass_dirs:
            if process_pass_dir(pd, dry_run=args.dry_run, verbose=args.verbose):
                written += 1
        action = "would write" if args.dry_run else "wrote"
        print(f"\nDone. {action} skills_loaded.json for {written}/{len(pass_dirs)} pass directories.")

    elif args.pass_dir:
        result = parse_pass_directory(args.pass_dir)
        if result is None:
            print(f"No episodes found in {args.pass_dir}")
            return

        if args.dry_run:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            out_path = args.pass_dir / "skills_loaded.json"
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {out_path}")
            print(f"  Total skills loaded: {result['count']}")
            print(f"  Skills: {', '.join(result['skill_names'])}")
            print(f"  Sources: cat={result['sources']['cat_commands']}, "
                  f"json_log={result['sources']['load_skill_json_from_logs']}, "
                  f"json_result={result['sources']['load_skill_json_from_result']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
