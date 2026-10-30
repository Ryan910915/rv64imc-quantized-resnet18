#!/usr/bin/env python3
"""Summarize measured ROI instruction encodings; never reconstruct from averages."""
from __future__ import annotations
import collections,csv,io,json,re
from pathlib import Path
from common import ELF,CATEGORIES,classify,sha256

def read_counts(path: Path,elf: ELF):
    text=path.read_text();lines=text.splitlines()
    if not lines or not lines[0].startswith('# RESNET_MIX_V1 '):raise ValueError('Missing observer header')
    end=re.findall(r'^# END regions=(\d+) expected=(\d+) active=(\d+)$',text,re.M)
    if len(end)!=1 or end[0][0]!=end[0][1] or end[0][2]!='0':
        raise ValueError('Incomplete/missing ROI pairs; no valid measurement summary')
    expected=int(end[0][0]);rows=[];seen=set()
    totals=collections.Counter();categories={i:collections.Counter() for i in range(expected)}
    closed={int(a):int(b) for a,b in re.findall(r'^# CLOSE op=(\d+) total=(\d+)$',text,re.M)}
    for raw in csv.DictReader(io.StringIO('\n'.join(x for x in lines if not x.startswith('#')))):
        r={k:int(v,0) for k,v in raw.items()};i=r['op'];pc=r['pc'];key=(i,pc)
        if i not in categories or key in seen or r['count']<=0:raise ValueError('Duplicate/invalid histogram entry')
        seen.add(key)
        if elf.instruction(pc,r['length'])!=r['bits']:raise ValueError('Histogram instruction differs from the measured ELF')
        category,name,width=classify(r['bits'],r['length'])
        if category=='unknown':raise ValueError(f'Unclassified encoding at {pc:#x}: {r["bits"]:#x}. Keep raw evidence; extend decoder before reporting.')
        r.update(category=category,mnemonic_family=name,memory_width_bytes=width,function=elf.function(pc))
        rows.append(r);totals[i]+=r['count'];categories[i][category]+=r['count']
    if len(closed)!=expected or any(totals[i]!=closed.get(i) for i in range(expected)):
        raise ValueError('ROI footer/count disagreement')
    for i in range(expected):
        gates=[r for r in rows if r['op']==i and r['pc']==elf.counter_pc()]
        if len(gates)!=1 or gates[0]['count']!=1:raise ValueError('Expected exactly one retired opening counter instruction per ROI')
    return rows,totals,categories

def write_csv(path:Path,rows:list[dict]):
    if not rows:raise ValueError('Refusing to emit empty CSV')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(counts:Path,log:Path,elf_path:Path,manifest:Path,out:Path,old_csv:Path|None=None):
    elf=ELF(elf_path);rows,totals,cats=read_counts(counts,elf)
    text=log.read_text(errors='replace');m=json.loads(manifest.read_text())
    if 'FAIL' in text or 'TRAP cause=' in text or 'PASS full_model ' not in text:
        raise ValueError('Full-model functionality did not pass')
    rx=r'^PROFILE op=(\d+) kind=(\S+) instret=(\d+) dense_macs=(\d+) valid_macs=(\d+)$'
    records=[dict(op=int(i),kind=k,instret=int(n),dense_macs=int(d),valid_macs=int(v))
             for i,k,n,d,v in re.findall(rx,text,re.M)]
    if len(records)!=len(m['operators']) or [r['op'] for r in records]!=list(range(len(records))):
        raise ValueError('Missing/duplicate PROFILE rows')
    if len(totals)!=len(records):raise ValueError('Histogram ROI count differs from graph')
    for r in records:
        if totals[r['op']]!=r['instret']:
            raise ValueError(f'op={r["op"]}: histogram sum={totals[r["op"]]} != minstret delta={r["instret"]}; do not combine mismatched scopes')
        spec=m['operators'][r['op']]['compiled']
        if any(int(spec[k])!=r[k] for k in ('dense_macs','valid_macs')):
            raise ValueError('MAC definition differs between manifest and current log')
    total=sum(totals.values());printed=re.findall(r'^PROFILE_SUM dispatch_regions_instret=(\d+)$',text,re.M)
    if printed!=[str(total)]:raise ValueError('PROFILE_SUM mismatch')
    old_check='not supplied';old_hash=None
    if old_csv is not None and old_csv.is_file():
        old_hash=sha256(old_csv)
        old=list(csv.DictReader(old_csv.open()))
        if len(old)!=len(records):raise ValueError('Old B0 CSV has a different number of operators')
        for a,b in zip(old,records):
            if a['kind']!=b['kind'] or any(int(a[k])!=b[k] for k in ('op','instret','dense_macs','valid_macs')):
                raise ValueError(f'New run does not reproduce saved B0 at op={b["op"]}; keep results separate and check ELF/input/compiler')
        old_check='all per-operator instret and MAC counts matched saved B0'
    grand=collections.Counter()
    for c in cats.values():grand.update(c)
    if sum(grand.values())!=total:raise ValueError('Category sum invariant failed')
    dense=sum(r['dense_macs'] for r in records);valid=sum(r['valid_macs'] for r in records)
    out.mkdir(parents=True,exist_ok=True)
    mix=[dict(category=c,count=grand[c],fraction=grand[c]/total,percent=100*grand[c]/total) for c in CATEGORIES]
    write_csv(out/'instruction_mix.csv',mix)
    op_rows=[]
    for r in records:
        i=r['op'];c=cats[i]
        rr={**r,**{k:c[k] for k in CATEGORIES},'memory_instructions':c['load']+c['store'],
            'instruction_share':r['instret']/total,'dense_macs_per_instret':r['dense_macs']/r['instret'],
            'valid_macs_per_instret':r['valid_macs']/r['instret']}
        op_rows.append(rr)
    write_csv(out/'operator_instruction_mix.csv',op_rows)
    memory=[]
    for r in op_rows:
        ins=r['memory_instructions'];i=r['op']
        memory.append(dict(op=i,kind=r['kind'],loads=r['load'],stores=r['store'],memory_instructions=ins,
            load_bytes=sum(x['count']*x['memory_width_bytes'] for x in rows if x['op']==i and x['category']=='load'),
            store_bytes=sum(x['count']*x['memory_width_bytes'] for x in rows if x['op']==i and x['category']=='store'),
            load_store_per_dense_mac=ins/r['dense_macs'] if r['dense_macs'] else '',
            load_store_per_valid_mac=ins/r['valid_macs'] if r['valid_macs'] else ''))
    write_csv(out/'memory_instruction_profile.csv',memory)
    detailed=[{**r,'pc':hex(r['pc']),'bits':hex(r['bits'])} for r in rows]
    write_csv(out/'pc_instruction_counts.csv',detailed)
    functions=collections.Counter()
    for r in rows:functions[(r['op'],r['function'])]+=r['count']
    write_csv(out/'function_instruction_profile.csv',[dict(op=i,function=f,instret=n,share_within_operator=n/totals[i])
         for (i,f),n in sorted(functions.items())])
    checksum=re.findall(r'PASS full_model operators=\d+ outputs=\d+ mismatches=0 checksum=(\d+)',text)
    summary=dict(status='PASS',measurement_scope='[opening platform_instret CSR, closing CSR), per operator; closing read excluded',
        model_sha256=m.get('model_sha256'),elf_sha256=sha256(elf_path),manifest_sha256=sha256(manifest),
        old_csv_sha256=old_hash,old_b0_check=old_check,operators=len(records),final_checksum=checksum[-1] if checksum else None,
        dispatch_regions_instret=total,instruction_mix=dict(grand),
        loads=grand['load'],stores=grand['store'],memory_instructions=grand['load']+grand['store'],
        dense_macs=dense,valid_macs=valid,dense_macs_per_instret=dense/total,valid_macs_per_instret=valid/total,
        load_store_per_dense_mac=(grand['load']+grand['store'])/dense if dense else None,
        load_store_per_valid_mac=(grand['load']+grand['store'])/valid if valid else None,
        custom_opcode_count=grand['custom'],custom_opcode_fraction=grand['custom']/total,
        custom_note='B0 custom opcode count is normally zero. This is not a validation of custom ISA semantics or lane utilization.',
        checks=['full model PASS','all gate pairs closed','every observed encoding matches ELF',
                'every per-op category sum equals target minstret delta','no unknown encoding','saved B0 checked when supplied'],
        cautions=['Instruction share is not time share; no CV-Wally cycle claim.',
                  'Load/store bytes are architectural access widths, including stack/metadata, not DRAM traffic or cache misses.',
                  'Multiply includes address arithmetic as well as MAC arithmetic.',
                  'Function attribution uses actual symbols; inlined requantization cannot be separated semantically.',
                  'No whole-ELF startup/validation total is claimed.'])
    (out/'mix_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS: instruction categories sum to target minstret for every operator.')
    print('Total:',total,'  loads:',grand['load'],'  stores:',grand['store'])
    print('Saved B0 check:',old_check)
    print('Reports:',out)
    return summary
