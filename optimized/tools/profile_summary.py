#!/usr/bin/env python3
"""Per-op machine minstret summaries; deliberately NOT an instruction-mix report."""
import csv,json,re,sys
from pathlib import Path
p=Path(sys.argv[1]);text=p.read_text();m=json.loads(Path('generated/manifest.json').read_text())
if 'PASS full_model ' not in text:raise SystemExit('Cannot summarize an incomplete/failed run')
rx=re.compile(r'^PROFILE op=(\d+) kind=(\S+) instret=(\d+) dense_macs=(\d+) valid_macs=(\d+)$',re.M)
rows=[]
for op,kind,inst,dense,valid in rx.findall(text):
    rows.append(dict(op=int(op),kind=kind,instret=int(inst),dense_macs=int(dense),valid_macs=int(valid)))
if len(rows)!=len(m['operators']) or [r['op'] for r in rows]!=list(range(len(rows))):
    raise SystemExit('Missing/duplicate operator counters; use make profile, not make spike')
total=sum(r['instret'] for r in rows)
if not total or any(r['instret']<=0 for r in rows):raise SystemExit('Counter did not advance')
for r in rows:
    r['instruction_share']=r['instret']/total
    r['dense_macs_per_instret']=r['dense_macs']/r['instret']
    r['valid_macs_per_instret']=r['valid_macs']/r['instret']
    r['instret_per_output']=r['instret']/m['operators'][r['op']]['output_elements']
out=Path('results/operator_profile.csv')
with out.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
by_kind={}
for r in rows:by_kind[r['kind']]=by_kind.get(r['kind'],0)+r['instret']
summary={'model_sha256':m['model_sha256'],'dispatch_regions_instret':total,'by_kind_instret':by_kind,
         'note':'Each region includes nn_execute and called helpers plus small counter-call boundary overhead. Startup, console, golden checks and checksum are outside regions. This is NOT entire-executable retired count, NOT instruction mix, NOT CV-Wally cycles.',
         'remaining_metrics':['instruction mix','dynamic loads/stores','full executable instruction total if reported','custom-instruction use after implementation']}
Path('results/profile_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Wrote',out,'and results/profile_summary.json')
for r in sorted(rows,key=lambda r:r['instret'],reverse=True)[:5]:
    print(f"op={r['op']:2d} {r['kind']:20s} instret={r['instret']} share={r['instruction_share']:.2%}")
