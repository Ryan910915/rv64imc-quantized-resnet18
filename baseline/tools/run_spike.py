#!/usr/bin/env python3
"""Stream Spike progress, preserve failures, and record a run. No enormous trace."""
import argparse
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--spike',default='spike');p.add_argument('--isa',default='rv64imc_zicsr')
p.add_argument('--memory',type=int,default=64);p.add_argument('--elf',type=Path,required=True);p.add_argument('--log',type=Path,required=True)
a=p.parse_args();binary=shutil.which(a.spike)
if not binary:raise SystemExit('spike not found in PATH; restore your existing Spike PATH setting')
if a.memory<64:raise SystemExit('This linker uses 64 MiB. Do not reuse the old -m16 smoke-test command.')
a.log.parent.mkdir(parents=True,exist_ok=True)
cmd=[binary,'--isa='+a.isa,'-m'+str(a.memory),str(a.elf)]
print('Running:', ' '.join(cmd),flush=True)
print('Full scalar ResNet can be slow. BEGIN/CHECK indicate progress; no -l trace is enabled.',flush=True)
t0=time.monotonic();passed=False
with a.log.open('w') as log:
    child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
    try:
        for line in child.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
            if line.startswith('PASS full_model '):passed=True
        rc=child.wait()
    except KeyboardInterrupt:
        child.terminate()
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        print('\nStopped by user; this is NOT a completed test.',file=sys.stderr)
        raise SystemExit(130)
report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':cmd,'returncode':rc,
        'pass_marker':passed,'elf_sha256':hashlib.sha256(a.elf.read_bytes()).hexdigest(),
        'spike_binary_sha256':hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'host_wall_seconds':time.monotonic()-t0,
        'warning':'host_wall_seconds is simulator elapsed time, NOT CV-Wally cycle count or latency'}
a.log.with_suffix('.run.json').write_text(json.dumps(report,indent=2)+'\n')
if rc!=0 or not passed:raise SystemExit(f'Spike test FAILED or incomplete (status={rc}, pass_marker={passed}); see {a.log}')
print('Spike full-model test exited successfully (all exported operator outputs matched).')
