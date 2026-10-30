#!/usr/bin/env python3
"""Record the actual binary, generated workload, flags, and compiler identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--elf',type=Path,required=True)
p.add_argument('--compiler',required=True);p.add_argument('--flags',required=True)
p.add_argument('--profile',type=int,required=True);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads(Path('generated/manifest.json').read_text())
r={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'elf':str(a.elf),'elf_sha256':sha(a.elf),
   'compiler':subprocess.check_output([a.compiler,'--version'],text=True).splitlines()[0],
   'flags':a.flags,'profile':a.profile,'model_sha256':m['model_sha256'],
   'input_sha256':m.get('input_sha256'),'blob_sha256':m['blob_sha256'],
   'source_sha256':{str(f):sha(f) for f in sorted(Path('src').glob('*')) if f.is_file()}}
(a.elf.parent/'build_manifest.json').write_text(json.dumps(r,indent=2)+'\n')
