"""Make seed-based sampling portable without changing the recorded pool order."""
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "manifests/skill_pool.json"

def ordered_pool_paths(pool_dir):
    pool_dir = Path(pool_dir)
    entries = json.loads(MANIFEST.read_text())
    missing = [e["directory"] for e in entries if not (pool_dir/e["directory"]/"SKILL.md").is_file()]
    if missing:
        raise ValueError(f"The recorded skill pool is incomplete: {len(missing)} missing directories. "
                         "See README.md; do not silently substitute a different distractor pool.")
    return [pool_dir/e["directory"] for e in entries]
