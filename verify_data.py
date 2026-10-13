"""Offline verification of separately obtained benchmark and distractor files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool",type=Path,default=ROOT/"skills_scaling_study/.agents/skills")
    p.add_argument("--tasks",type=Path,default=ROOT/"skillsbench/tasks")
    a=p.parse_args()
    missing=[]; changed=[]
    for row in json.loads((ROOT/"manifests/skill_pool.json").read_text()):
        f=a.pool/row["directory"]/"SKILL.md"
        if not f.is_file(): missing.append(row["directory"])
        elif hashlib.sha256(f.read_bytes()).hexdigest()!=row["skill_md_sha256"]: changed.append(row["directory"])
    tasks=json.loads((ROOT/"manifests/benchmark.json").read_text())["main_tasks"]
    absent=[t for t in tasks if not (a.tasks/t/"instruction.md").is_file()]
    print(json.dumps({"missing_skill_files":len(missing),"changed_skill_files":len(changed),"missing_main_tasks":absent,
                      "first_missing":missing[:10],"first_changed":changed[:10]},indent=2))
    if missing or changed or absent: raise SystemExit(1)
if __name__=="__main__": main()
