"""Summarize completed probe CSVs; percentile uses nearest rank."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
summary = {}
for path in sorted(root.glob('*.csv')):
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith('connect_ms='):
        continue
    groups = defaultdict(list)
    for row in csv.DictReader(lines[1:]):
        groups[(int(row['upload_bytes']), int(row['download_bytes']))].append(float(row['roundtrip_ms']))
    values = []
    for (upload, download), samples in sorted(groups.items()):
        samples.sort()
        values.append(dict(upload_bytes=upload, download_bytes=download,
                           samples=len(samples), median_ms=statistics.median(samples),
                           p95_ms=samples[math.ceil(.95 * len(samples))-1],
                           min_ms=min(samples), max_ms=max(samples)))
    summary[path.stem] = dict(connect_ms=float(lines[0].split('=')[1]), results=values)
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
for name, trial in summary.items():
    print(name)
    for row in trial['results']:
        print(f"  {row['upload_bytes']:7d} each way: n={row['samples']} median={row['median_ms']:.1f} ms p95={row['p95_ms']:.1f} ms max={row['max_ms']:.1f} ms")
