# ResNet-18 W8A8: complete-graph RV64IMC baseline

This package belongs INSIDE your existing `rv64imc_starter/`, next to
`conv_op14/`. It does not replace the parent Makefile, kernels, or cases.

## Quick start

Run from the parent project, after placing the downloaded archive there:

```bash
source .venv/bin/activate
unzip -n resnet_full_starter.zip
cd resnet_full
make export
make host-test
make elf && make disasm && make spike
```

Stop at the first error. The host test is not optional evidence: it is the
fastest way to catch a numerical/export problem before a long Spike run.

Defaults:

- Model: `../models/resnet18_int8_channelwise.tflite`.
- Input: `../results/conv_case/model_input.npy`, when present.
- Otherwise: synthetic INT8 input with seed 20261006.
- Reference: host LiteRT `BUILTIN_REF`, one thread, preserved intermediates.
- Target: RV64IMC, LP64, integer-only scalar C, `-O2`.
- Spike: `--isa=rv64imc_zicsr -m64`, bare metal, no `pk`.
- Generated inference parameters: integer multiplier/shift and zero points;
  floating-point scale processing occurs ONLY in the host exporter.

To select a different already-preprocessed INT8 model input:

```bash
make export INPUT=/absolute/path/input.npy
make host-test
make elf && make disasm && make spike
```

To generate another synthetic input:

```bash
make export INPUT= SEED=123
make host-test
make elf && make disasm && make spike
```

Always explicitly run `make export` after changing MODEL/INPUT/SEED. Multiple
inputs are regression tests, not ImageNet accuracy unless you also provide the
proper dataset, preprocessing and labels. Do not use a JPG directly as INPUT.

## What actually runs

```
model input, immutable weights and bias
        -> TRANSPOSE/PAD/Conv/pooling/.../residual ADD/.../FC
        -> final model output
```

Every dynamic tensor gets a distinct statically allocated arena slice. Real
outputs from earlier operators feed later operators, including both branches
of a residual block. NO intermediate reference is injected into the computation.

The program checks every operator's output against a separate read-only golden
blob, stopping at the first mismatch. These are automatic checks within one
complete-model run, not isolated layer workloads. Checks do not replace data
flow. Once all operators match, the final output is reported as:

```
PASS full_model operators=<actual count> outputs=<actual size> mismatches=0 checksum=<value>
```

The previously inspected model is expected to have 37 operators and 1000 final
outputs, but the exporter reads the actual graph rather than hard-coding them.
`generated/operators.txt` and `generated/manifest.json` are authoritative for
the exported workload. A successful export is NOT a successful C or Spike test.

## Files

```
resnet_full/
  Makefile
  link.ld                    # 64 MiB Spike test RAM, 256 KiB stack
  src/
    nn.h, nn.c               # integer operators and quantization
    main.c                   # graph schedule, comparison, result reporting
    platform.h, platform.c   # host stdio / bare-metal HTIF console
    start.S                  # boot, stack, BSS, trap reporting, HTIF exit
  tools/
    export_model.py          # host model compiler and reference exporter
    flatbuffer_min.py        # bounded subset reader of the TFLite schema
    check_host.py            # final output and validation JSON
    run_spike.py             # stream logs and propagate simulator failure
    profile_summary.py       # per-operator minstret summary (not instruction mix)
    audit_isa.py              # check disassembly for FP/vector mnemonics
    record_build.py          # source, binary, model and compiler fingerprints
  tests/test_suite.py        # local semantic/synthetic integration tests
  generated/                 # created on YOUR computer by make export
    model_data.h             # tensor/graph sizes and declarations
    model_data.c             # static tensor descriptors, operator schedule, arena
    model_blob.S             # embeds raw blobs into ELF read-only data at build time
    model.bin                # weights, biases, one model input, integer scales
    reference.bin            # verification-only golden outputs
    manifest.json            # shapes, quantization, source hashes, memory/MACs
    operators.txt
    input.npy, expected_output.npy
  build/baseline/
    resnet.elf, resnet.dis, resnet.map, attributes.txt, size.txt, build_manifest.json
  results/
    host.log, host_output.bin, host_output.npy, host_validation.json
    spike.log, spike.run.json
```

### Why `.bin` and `.S`, rather than millions of numbers in a header?

A 12 MB weight tensor expressed as decimal C initializers becomes a much larger
text file. This package uses GNU assembler `.incbin` so weights remain raw
bytes, while C receives normal typed pointers via the static tensor descriptors.
The linker embeds all blobs in the ELF. No `.bin`, `.npy`, `.json`, or `.tflite`
file is opened by the RISC-V program. `.h` and `.c` are still compile-time
artifacts, not formats interpreted by the processor. This is static model data,
not a runtime loader. Constants are little-endian and 16-byte aligned.

## Output and failures

During host/Spike execution:

```
BEGIN op=0 kind=TRANSPOSE
CHECK op=0 elements=... mismatches=0 max_abs_diff=0
BEGIN op=1 kind=PAD
...
```

A mismatch prints the operator index, number of mismatches, largest absolute
difference and first expected/actual pair. Nonzero status propagates through
`make`; a `tee` pipeline cannot hide the error. Share the FIRST failure and
`generated/manifest.json`, not just the last Make error.

A trap prints `cause`, `pc`, and `tval`, then fails. A silent stopped operator is
not automatically a trap: the scalar 3x3 kernels do substantial computation.
There is no promised runtime. Full-model Spike execution can be much slower
than the earlier 6.4 million-MAC op14 test. Do not enable `spike -l` for the whole
model as a first attempt; instruction-by-instruction logs can be enormous.
Ctrl-C is an incomplete test, never a PASS.

`make spike` defaults to 64 MiB. Do NOT reuse `-m16` from the earlier smoke test.
This memory layout is for Spike and is NOT verified against CV-Wally RTL.
The exporter and linker reject a model that exceeds the static budget.
References and non-reused activations intentionally trade memory for easier
correctness debugging; they are not an optimized deployment memory plan.

## Supported / rejected semantics

Supported: dense static INT8 Conv2D (1x1, 3x3, 7x7 and other checked sizes),
stride/dilation and SAME/VALID, symmetric per-tensor or output-channel-wise
weights, INT32 bias/accumulation, FC, equal-shape residual ADD with independent
input scales, max/average pooling, PAD/PADV2, TRANSPOSE, RESHAPE, and clamp-based
ReLU/ReLU6/ReLU[-1,1]. Two-stage Q31 rounding matches the uploaded op14 case.

Rejected instead of guessed: unknown/custom operators; unsupported versions;
multiple subgraphs/control flow; sparse/external/variable weights; grouped
Conv; floating/hybrid tensors; broadcast ADD; incompatible quantization for
copy/pooling; unsafe accumulator bounds; nonlinear activations beyond the
supported clamps. Dynamic signatures are specialized to the actual allocated
positive dimensions; runtime resizing is not supported. The tool is purpose-
built for this INT8 ResNet graph, NOT a universal TFLite compiler.

Changing rounding to make a failing test pass without understanding the build
is not allowed. LiteRT builds can differ; check the first failing operator.

## Initial ISA-level profiling (after a real PASS)

```bash
make profile
```

This builds a separate `build/profile/resnet.elf` and executes the complete
model again. It reads the standard machine `minstret` CSR before and after each
`nn_execute()` call, then writes:

```
results/spike_profile.log
results/operator_profile.csv
results/profile_summary.json
```

The measured region includes dispatch and all called kernel/requantization
helpers, plus a small counter-call boundary overhead. It excludes console
output, golden checking, checksum, startup, and inter-operator scheduling.
`dispatch_regions_instret` is the SUM of these regions, not the total retired
instructions of the entire ELF. Do not relabel it as such. Both conventional
(dense) MACs and actually in-bounds MACs are reported because the direct Conv
skips padding positions. State the convention in the report.

Counters use standard system CSR instructions, NOT the required custom NN ISA.
No `mcycle` value is presented as CV-Wally latency. Instruction mix and dynamic
load/store counts still need separate instrumentation/histogram analysis. This
is an initial operator contribution report, not all of spec section 2.B.
Golden checking between operators also changes cache state, so this functional
runner must not be used unmodified for hardware cache/timing conclusions.

## Project milestone

A host+Spike PASS completes a full-model functional baseline FOR THIS EXPORTED
INPUT on this simulator. It is not accuracy validation, a proof for all inputs,
or the midterm custom-instruction implementation. The project specification
(section 3) additionally requires packed integer MAC AND requantization
extensions, simulator support, kernel use, correctness tests and instruction
reduction evidence. Preserve B0, establish instruction profiles, then implement
and compare B1/X1/X2 incrementally. See `docs/NEXT_AFTER_BASELINE.md`.

## Local test status

See `VALIDATION.md`. The package was checked on synthetic complete schedules
and the uploaded real op14 data. The FULL `.tflite` and a runnable Spike were
not available in the authoring environment; no full-ResNet/Spike PASS is claimed
here. The commands above produce that evidence on your existing environment.
