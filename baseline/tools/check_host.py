#!/usr/bin/env python3
"""Verify final host output again and create a machine-readable test record."""
import hashlib
import json
from pathlib import Path
import numpy as np

manifest=json.loads(Path('generated/manifest.json').read_text())
expected=np.load('generated/expected_output.npy',allow_pickle=False)
actual=np.fromfile('results/host_output.bin',dtype=np.int8)
if actual.size!=expected.size:
    raise SystemExit('FAIL: host output length differs from expected output')
actual=actual.reshape(expected.shape)
diff=actual.astype(np.int32)-expected.astype(np.int32)
report={'model_sha256':manifest['model_sha256'], 'input_sha256':manifest.get('input_sha256'),
        'model_blob_sha256':manifest['blob_sha256'], 'output_shape':list(actual.shape),
        'elements':int(actual.size),'mismatches':int(np.count_nonzero(diff)),
        'max_abs_diff':int(np.abs(diff).max(initial=0)),
        'host_output_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
        'intermediates':'main.c checks every operator before continuing; no reference injection',
        'test_scope':'one exported graph input; not ImageNet accuracy'}
Path('results/host_validation.json').write_text(json.dumps(report,indent=2)+'\n')
np.save('results/host_output.npy',actual,allow_pickle=False)
if report['mismatches']:raise SystemExit('FAIL: host/reference mismatch')
print('Host full-graph validation saved: results/host_validation.json')
