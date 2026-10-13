"""Find recorded skill sources locally; never download or verify a URL."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query', help='Skill identifier or repository substring')
    parser.add_argument('--exact', action='store_true', help='Match the complete skill identifier')
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    matches = []
    total = 0
    with (ROOT/'manifests/skill_sources.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            match = row['skill_id'] == args.query if args.exact else args.query.lower() in (row['skill_id']+' '+(row['recorded_repository'] or '')).lower()
            if match:
                total += 1
                if len(matches) < args.limit:
                    matches.append(row)
    print(json.dumps({'total_matches':total, 'shown':len(matches), 'notice':'Use the linked sources under their original licenses. External files may have changed.', 'entries':matches}, indent=2))

if __name__ == '__main__':
    main()
