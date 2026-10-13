"""Build provenance-only indexes offline; never download or execute skills."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = frozenset({
    'spreadsheet-processor', 'docx-official', 'xlsx-official', 'xlsx', 'pdf',
    'pptx', 'pdf-official', 'portable-document-handler', 'pptx-official',
    'word-document-processor', 'docx',
})

def derive(pool):
    rows = []
    for i, item in enumerate(pool):
        source = item.get('source')
        if source and (item.get('source_type') != 'github' or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', source)):
            raise ValueError(f'Unexpected source format at index {i}')
        rows.append({
            'pool_index': i,
            'skill_id': item['directory'],
            'recorded_repository': source,
            'repository_url': f'https://github.com/{source}' if source else None,
            'upstream_skill_path': None,
            'git_commit': None,
            'skill_md_sha256': item['skill_md_sha256'],
            'source_content_hash': item.get('source_content_hash'),
            'source_status': 'recorded_repository' if source else 'unresolved',
            'url_check_status': 'not_checked' if source else 'not_applicable',
            'license_review_status': 'restricted_local_notice' if item['directory'] in RESTRICTED else 'unreviewed',
        })
    return rows

def render_sources(rows):
    counts = Counter(row['recorded_repository'] for row in rows if row['recorded_repository'])
    text = [
        '# Recorded skill source repositories', '',
        'We provide source links for the skills used in our experiments. Due to licensing restrictions on some skills, we do not redistribute their contents or provide download scripts. Readers can obtain skills from the linked repositories under their original licenses. External repositories and files may have changed since the experiments.', '',
        f'{len(rows):,} skill entries; {sum(counts.values()):,} with sources; {len(counts):,} distinct repositories; {len(rows)-sum(counts.values()):,} unresolved entries.', '',
        '[Per-skill JSONL index](../manifests/skill_sources.jsonl) · [Unresolved entries](../manifests/unresolved_sources.json) · [Licensing boundaries](DATA_AND_LICENSES.md)', '',
        'Counts show how many skills in our experiment pool map to each recorded source.', '',
        '| Recorded repository | Skill entries |', '|---|---:|',
    ]
    for source, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        text.append(f'| [{source}](https://github.com/{source}) | {count} |')
    return '\n'.join(text) + '\n'

def main():
    pool = json.loads((ROOT/'manifests/skill_pool.json').read_text())
    rows = derive(pool)
    (ROOT/'manifests/skill_sources.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in rows))
    (ROOT/'manifests/unresolved_sources.json').write_text(json.dumps([row for row in rows if row['source_status']=='unresolved'], indent=2) + '\n')
    (ROOT/'docs/SKILL_SOURCES.md').write_text(render_sources(rows))
    print(f'Indexed {len(rows)} skills offline. No URLs requested; no assets downloaded.')

if __name__ == '__main__':
    main()
