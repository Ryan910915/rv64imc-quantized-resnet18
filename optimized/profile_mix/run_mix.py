#!/usr/bin/env python3
"""Run preflight then the SAME PROFILE=1 ELF using the measurement-only Spike.
No target/source/model changes and no overwritten old results. One full rerun
is required because the previous aggregate counters contain no instruction mix.
"""
from __future__ import annotations
import argparse,collections,datetime,json,os,shutil,subprocess,sys,time,threading
from pathlib import Path
from common import ELF,sha256
from summarize import read_counts,summarize

HERE=Path(__file__).resolve().parent

def env_for(elf_path:Path,count:int,out:Path):
    elf=ELF(elf_path);env=os.environ.copy()
    env.update(RESNET_MIX_COUNTER_PC=hex(elf.counter_pc()),RESNET_MIX_TEXT_BASE=hex(elf.base),
               RESNET_MIX_TEXT_LIMIT=hex(elf.limit),RESNET_MIX_REGIONS=str(count),RESNET_MIX_OUTPUT=str(out.resolve()))
    return env

def run_spike(spike:Path,isa:str,memory:int,elf:Path,count:int,out:Path,time_limit:float|None=None):
    out.mkdir(parents=True,exist_ok=True)
    raw=out/'roi_counts.csv';log=out/'spike.log'
    if raw.exists() or log.exists():raise ValueError('Evidence already exists: '+str(out))
    cmd=[str(spike),'-p1','-g','--isa='+isa,'-m'+str(memory),str(elf.resolve())]
    print('+',' '.join(cmd),flush=True)
    t0=time.monotonic();rc=None
    with log.open('x') as f:
        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            env=env_for(elf,count,raw),text=True,errors='replace',bufsize=1)
        timer=None
        if time_limit is not None:
            timer=threading.Timer(time_limit,child.kill);timer.daemon=True;timer.start()
        try:
            for line in child.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            rc=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        finally:
            if timer:timer.cancel()
            record=dict(command=cmd,returncode=rc,elf_sha256=sha256(elf),spike_sha256=sha256(spike),
                host_wall_seconds=time.monotonic()-t0,
                warning='Host elapsed time is simulator runtime, NOT target performance or CV-Wally cycles.')
            (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    if rc!=0:raise ValueError(f'Spike returned {rc}; see {log}')
    if not raw.is_file():raise ValueError('No observer output. This is likely an uninstrumented Spike; use make mix-setup.')
    return raw,log

def smoke(spike:Path,cross:str,out:Path):
    out.mkdir(parents=True,exist_ok=True);elf=out/'smoke.elf'
    compiler=shutil.which(cross+'gcc')
    if not compiler:raise ValueError('Compiler not found: '+cross+'gcc')
    cmd=[compiler,'-march=rv64imc','-mabi=lp64','-mcmodel=medany','-nostdlib','-nostartfiles','-static',
         '-Wl,--no-relax','-T',str(HERE/'smoke.ld'),str(HERE/'smoke.S'),'-o',str(elf)]
    print('+',' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    raw,log=run_spike(spike,'rv64imc_zicsr',16,elf,2,out,time_limit=30.0)
    rows,totals,cats=read_counts(raw,ELF(elf))
    expected=[dict(csr=1,jump_call_return=2,integer_alu=9,branch=3,load=3,store=3,multiply=3),
              dict(csr=1,jump_call_return=2,integer_alu=5,load=1,store=1,divide_remainder=2,fence=1)]
    if dict(totals)!={0:24,1:13} or any(dict(cats[i])!=expected[i] for i in range(2)):
        raise ValueError('Smoke observer counts/categories failed; full model was NOT started: '+repr(cats))
    print('PASS measurement preflight: 24 and 13 retired instructions; all categories verified.')
    (out/'validation.json').write_text(json.dumps(dict(status='PASS',totals=dict(totals),categories={i:dict(c) for i,c in cats.items()}),indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spike',type=Path,required=True);ap.add_argument('--cross',default='riscv64-unknown-elf-')
    ap.add_argument('--isa',default='rv64imc_zicsr');ap.add_argument('--memory',type=int,default=64)
    ap.add_argument('--elf',type=Path,default=Path('build/profile/resnet.elf'))
    ap.add_argument('--manifest',type=Path,default=Path('generated/manifest.json'))
    ap.add_argument('--old-csv',type=Path,default=Path('results/operator_profile.csv'))
    ap.add_argument('--out',type=Path,default=Path('results/profile_mix'));ap.add_argument('--smoke-only',action='store_true')
    a=ap.parse_args();spike=a.spike.expanduser().resolve()
    if not spike.is_file():raise ValueError('Instrumented Spike not found. First run make mix-setup.')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=a.out.resolve()/stamp;out.mkdir(parents=True)
    smoke(spike,a.cross,out/'preflight')
    if not a.smoke_only:
        if a.memory<64:raise ValueError('The complete-model linker requires at least 64 MiB')
        m=json.loads(a.manifest.read_text());elf=a.elf.resolve();before=sha256(elf)
        if 'rv64imc' not in a.isa:raise ValueError('Specify the project RV64IMC ISA explicitly')
        # Snapshot metadata so future reports cannot accidentally use changed files.
        shutil.copy2(a.manifest,out/'manifest.json')
        old=None
        if a.old_csv.is_file():
            old=out/'previous_operator_profile.csv';shutil.copy2(a.old_csv,old)
        (out/'elf_identity.json').write_text(json.dumps(dict(path=str(elf),sha256=before),indent=2)+'\n')
        print('Preflight passed. Starting ONE full-model rerun; no per-instruction text trace.',flush=True)
        raw,log=run_spike(spike,a.isa,a.memory,elf,len(m['operators']),out)
        if sha256(elf)!=before:raise ValueError('ELF changed during execution')
        summarize(raw,log,elf,out/'manifest.json',out,old)
    # Update this pointer ONLY after all validations pass; old runs remain.
    pointer='latest_preflight.txt' if a.smoke_only else 'latest_run.txt'
    (a.out/pointer).write_text(str(out)+'\n')
    print('Validated output:',out)
    print('Previous B0 CSV/JSON/logs and program source were not changed.')
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('Interrupted: not a completed measurement.',file=sys.stderr);sys.exit(130)
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as e:
        print('PROFILE MIX STOPPED:',e,file=sys.stderr);sys.exit(2)
