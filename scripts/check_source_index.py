"""Validate generated indexes and provenance without network access."""
import json
from build_source_index import ROOT, RESTRICTED, derive, render_sources

def main():
    pool = json.loads((ROOT/'manifests/skill_pool.json').read_text())
    rows = [json.loads(line) for line in (ROOT/'manifests/skill_sources.jsonl').read_text().splitlines()]
    assert rows == derive(pool), 'Per-skill index differs from the recorded pool'
    assert len(rows) == 21262
    assert len({r['skill_id'] for r in rows}) == len(rows)
    assert sum(r['repository_url'] is not None for r in rows) == 20788
    assert len({r['repository_url'] for r in rows if r['repository_url']}) == 670
    unresolved = [r for r in rows if r['source_status']=='unresolved']
    assert len(unresolved) == 474
    assert json.loads((ROOT/'manifests/unresolved_sources.json').read_text()) == unresolved
    assert {r['skill_id'] for r in rows if r['license_review_status']=='restricted_local_notice'} == RESTRICTED
    assert (ROOT/'docs/SKILL_SOURCES.md').read_text() == render_sources(rows)
    assert all(r['git_commit'] is None and r['upstream_skill_path'] is None for r in rows)
    assert all(r['url_check_status'] in {'not_checked','not_applicable'} for r in rows)
    print('PASS: 21,262 ordered entries, 20,788 recorded sources, 670 repositories, 474 unresolved entries, 11 restrictive-notice flags. No live availability or legal clearance asserted.')

if __name__ == '__main__':
    main()
