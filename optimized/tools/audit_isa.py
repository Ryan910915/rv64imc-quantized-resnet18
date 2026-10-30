#!/usr/bin/env python3
"""Flag floating-point/vector instructions; not a complete ISA formal proof."""
import re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text();bad=[];instructions=0
for line in text.splitlines():
    m=re.match(r'\s*[0-9a-f]+:\s+[0-9a-f]+\s+([a-zA-Z0-9_.]+)',line)
    if not m:continue
    instructions+=1;op=m.group(1)
    if op.startswith('c.'):op=op[2:]
    if (op.startswith('f') and not op.startswith('fence')) or op.startswith('v'):
        bad.append(line)
if not instructions:raise SystemExit('ISA audit failed: no disassembly instructions parsed')
if bad:raise SystemExit('Unexpected floating/vector instructions:\n'+'\n'.join(bad[:30]))
print(f'ISA audit: {instructions} static instruction lines scanned; no FP/vector mnemonics. Not dynamic counts.')
