"""Measure preselection latency from execution logs."""
import json
import re
from pathlib import Path
from datetime import datetime
from statistics import mean, median
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT
MODELS = {'glm-5', 'kimi-k2.5', 'minimax-m2.5'}
EXCLUDED = {'fix-visual-stability', 'mhc-layer-impl', 'multilingual-video-dubbing',
            'python-scala-translation', 'speaker-diarization-subtitles'}
LOGS = {'LLM': 'experiment_logs-llm', 'RAG': 'experiment_logs-rag',
        'Hybrid': 'experiment_logs-mq-rag-llm-fixed'}

def read_timing(path, method):
    start = end = None
    retries = cached = False
    boundary = None
    with path.open(errors='replace') as handle:
        for line in handle:
            if '[Episode 0]' in line or '[Episode 1]' in line:
                break
            if '[Preselect/' not in line:
                continue
            stamp = re.match(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', line)
            if not stamp:
                continue
            t = datetime.strptime(stamp.group(), '%Y-%m-%d %H:%M:%S')
            if method == 'LLM' and '[Preselect/llm] Asking ' in line and start is None:
                start, boundary = t, line.strip()
            if method == 'Hybrid' and '[Preselect/multi-query-rag-llm] Indexed ' in line and start is None:
                start, boundary = t, line.strip()
            if method == 'RAG' and '[Preselect/rag-precomputed] Loaded ' in line:
                cached = True
            if 'WARNING' in line or 'ERROR' in line:
                retries = True
            if ' skills for prompt:' in line:
                end = t
                break
    if start is not None and end is not None and end >= start:
        return {'seconds': (end-start).total_seconds(), 'warning': retries,
                'start': start.isoformat(), 'end': end.isoformat(), 'boundary': boundary}
    return {'seconds': None, 'warning': retries, 'cached_rag': cached}

def main():
    rows, excluded = [], Counter()
    for method, folder in LOGS.items():
        for rp in sorted((DATA / folder).glob('*/*/skills_*/seed_42/pass_*/result.json')):
            rel = rp.relative_to(DATA / folder).parts
            if rel[0] in EXCLUDED or rel[1] not in MODELS or rel[2] not in {'skills_10','skills_100','skills_1000'} or rel[4] not in {'pass_0','pass_1','pass_2'}:
                continue
            try:
                result = json.loads(rp.read_text())
                agent = result.get('agent_result')
                if not isinstance(agent, dict):
                    excluded['invalid_agent'] += 1
                    continue
                error = agent.get('error') or ''
                if result.get('exception') or result.get('infrastructure_failure') or (rp.parent/'retry_required.json').exists() or error not in ('', 'Max episodes reached'):
                    excluded['nonterminal_or_infrastructure'] += 1
                    continue
                timing = read_timing(rp.parent/'run.log', method)
                rows.append({'method': method, 'task': rel[0], 'model': rel[1],
                             'extra_skills': int(rel[2][7:]), 'pass': int(rel[4][5:]),
                             'source': str(rp.parent), **timing})
            except (OSError, ValueError):
                excluded['unreadable'] += 1
    summary = []
    for method in LOGS:
        for scale in (10,100,1000):
            group = [r for r in rows if r['method']==method and r['extra_skills']==scale]
            values = sorted(r['seconds'] for r in group if r['seconds'] is not None)
            clean = [r['seconds'] for r in group if r['seconds'] is not None and not r['warning']]
            summary.append({'method':method, 'extra_skills':scale, 'eligible':len(group),
                            'timed':len(values), 'mean_sec':mean(values) if values else None,
                            'median_sec':median(values) if values else None,
                            'p90_sec':values[min(len(values)-1, int(.9*len(values)))] if values else None,
                            'clean_median_sec':median(clean) if clean else None,
                            'warning_count':sum(r['warning'] for r in group),
                            'cached_rag':sum(r.get('cached_rag',False) for r in group)})
    out=ROOT/'analysis'/'routing_latency'
    out.mkdir(parents=True, exist_ok=True)
    (out/'latency_per_evaluation.json').write_text(json.dumps(rows, indent=2))
    (out/'latency_summary.json').write_text(json.dumps({'summary':summary,'excluded':dict(excluded)},indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
