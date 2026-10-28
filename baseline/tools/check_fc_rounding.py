#!/usr/bin/env python3
"""Diagnose FC rounding from an existing export without running the whole model.
Reads model/reference bytes and never changes them. Uses the preceding saved
reference tensor ONLY for this standalone diagnostic, not for target inference.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_fc_rounding import build_library, encode

def sha(data): return hashlib.sha256(data).hexdigest()

def q31(x,m,shift):
    product=int(x)*(1<<max(shift,0))*int(m)
    nudge=(1<<30) if product>=0 else 1-(1<<30)
    v=product+nudge
    high=(abs(v)//(1<<31))*(-1 if v<0 else 1)
    right=max(-shift,0)
    if not right: return high
    magnitude=(abs(high)+(1<<(right-1)))>>right
    return -magnitude if high<0 else magnitude

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',type=Path,default=Path('generated'))
    parser.add_argument('--report',type=Path,default=Path('results/fc_reference_diagnosis.json'))
    parser.add_argument('--verify-reference-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.generated/'manifest.json').read_text())
    raw=(args.generated/'model.bin').read_bytes()
    gold=(args.generated/'reference.bin').read_bytes()
    if sha(raw)!=manifest['blob_sha256'] or sha(gold)!=manifest['reference_blob_sha256']:
        raise ValueError('Export file hash mismatch; do not continue with mixed generated files.')
    if args.verify_reference_only:
        previous=json.loads(args.report.read_text())
        if previous['reference_sha256']!=sha(gold):
            raise ValueError('Reference changed after re-export. STOP and inspect input/runtime versions.')
        if previous.get('model_sha256')!=manifest.get('model_sha256'):
            raise ValueError('Model changed after re-export.')
        if previous.get('input_sha256')!=manifest.get('input_sha256'):
            raise ValueError('Model input changed after re-export.')
        fcs=[n for n in manifest['operators'] if n['kind']==9]
        if not fcs or any(n['compiled'].get('fc_reference')!=1 for n in fcs):
            raise ValueError('FC fix is not in the regenerated workload.')
        print('PASS: model and input unchanged; reference.bin SHA256 unchanged.')
        print('PASS: regenerated FC operators use integer-only reference-compatible scaling.')
        return 0
    meta={t['index']:t for t in manifest['tensors']}
    producers={n['output']:n for n in manifest['operators']}
    def tensor(i):
        t=meta[i];dtype=np.dtype(t['dtype']).newbyteorder('<')
        count=math.prod(t['shape'])
        if t['storage']=='model_blob':
            source=raw;offset=t['offset']
        else:
            source=gold;offset=producers[i]['compiled']['ref_offset']
        return np.frombuffer(source,dtype=dtype,count=count,offset=offset).reshape(t['shape']).copy()
    reports=[];all_match=True
    with tempfile.TemporaryDirectory(prefix='fc-diagnostic-') as td:
        lib=build_library(Path(td))
        for n in manifest['operators']:
            if n['kind']!=9: continue
            p=n['compiled'];xi,wi,bi=p['in'];yi=p['out']
            x=tensor(xi);w=tensor(wi);co,ci=w.shape
            bias=tensor(bi).astype(np.int64) if bi>=0 else np.zeros(co,dtype=np.int64)
            acc=(x.astype(np.int64).reshape(-1,ci)-meta[xi]['zero_points'][0])@w.astype(np.int64).T+bias
            if np.any(acc<-(1<<31)) or np.any(acc>(1<<31)-1):
                raise ValueError('FC accumulator is not INT32; this patch is not applicable.')
            sx=meta[xi]['scales'][0];sy=meta[yi]['scales'][0]
            ws=meta[wi]['scales'];ws=ws if len(ws)>1 else ws*co
            scales=[float(sx)*float(s)/float(sy) for s in ws]
            qm=np.frombuffer(raw,dtype='<i4',count=2*p['q_count'],offset=p['q_offset'])
            expected=tensor(yi).reshape(-1)
            old=[];native=[];new=[];zp=meta[yi]['zero_points'][0]
            def clamp(v):return min(p['act_max'],max(p['act_min'],int(v)+zp))
            for row in acc:
                for oc,a in enumerate(row):
                    old.append(clamp(q31(int(a),int(qm[oc]),int(qm[p['q_count']+oc]))))
                    native.append(clamp(lib.native_fc_oracle(int(a),scales[oc])))
                    m,e=encode(scales[oc])
                    new.append(clamp(lib.nn_requantize_fc_reference(int(a),m,e)))
            old=np.asarray(old,dtype=np.int16);native=np.asarray(native,dtype=np.int16);new=np.asarray(new,dtype=np.int16)
            bad=np.flatnonzero(old!=expected)
            r={'operator':n['index'],'outputs':int(expected.size),
               'q31_mismatches':int(len(bad)),
               'native_binary64_mismatches':int(np.count_nonzero(native!=expected)),
               'integer_fc_mismatches':int(np.count_nonzero(new!=expected)),
               'first_q31_differences':[{'index':int(i),'q31':int(old[i]),
                                        'expected':int(expected[i]),'integer_fc':int(new[i])}
                                       for i in bad[:16]]}
            reports.append(r)
            print(f"FC op={r['operator']} outputs={r['outputs']}")
            print(f"  original Q31 mismatches: {r['q31_mismatches']}")
            print(f"  native binary64 reference mismatches: {r['native_binary64_mismatches']}")
            print(f"  patched integer FC mismatches: {r['integer_fc_mismatches']}")
            for item in r['first_q31_differences'][:4]:print(' ',item)
            all_match &= r['native_binary64_mismatches']==0 and r['integer_fc_mismatches']==0
    if not reports:raise ValueError('No FC operator found.')
    result={'diagnostic_only':True,'reference_unchanged':True,
            'model_sha256':manifest.get('model_sha256'),'input_sha256':manifest.get('input_sha256'),
            'reference_sha256':sha(gold),'old_model_blob_sha256':sha(raw),
            'operators':reports,'safe_to_reexport_with_this_fix':bool(all_match),
            'full_model_verified':False,'spike_verified':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    if not all_match:
        print('STOP: this correction does not match every saved FC reference. Keep the original golden; share this report.',file=sys.stderr)
        return 2
    print('PASS: corrected integer FC matches the unmodified saved reference.')
    print('Next: make export, verify-reference-only, then rerun the COMPLETE host model.')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError,KeyError) as exc:
        print('FC DIAGNOSIS FAILED:',exc,file=sys.stderr);sys.exit(2)
