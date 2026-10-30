# Quantized ResNet-18 Acceleration on RV64IMC with Custom ISA Extensions

## Overview

This project implements and evaluates a quantized ResNet-18 inference workload on a bare-metal RV64IMC platform using Spike.

The selected model is:

```text
resnet18_int8_channelwise.tflite
```

The model uses static INT8 quantization with:

- INT8 activations
- INT8 channel-wise weights
- INT32 accumulation
- Integer-only inference kernels
- No floating-point hardware dependency

The project evaluates three configurations:

| Configuration | Description |
|---|---|
| **B0** | RV64IMC bare-metal baseline |
| **X1** | B0 + `XQMAC8` packed INT8 multiply-accumulate instruction |
| **X2** | X1 + `XREQUANT` fused requantization instruction |

The midterm evaluation focuses on ISA-level acceleration using Spike. PPA and cycle-accurate RTL evaluation are outside the scope of the midterm.

---

# Repository Structure

```text
.
├── baseline/          # Original RV64IMC bare-metal baseline
├── optimized/         # Final X2 implementation: XQMAC8 + XREQUANT
├── custom_isa/
│   ├── spike/         # Spike custom-ISA patch
│   ├── xqmac8_test/   # XQMAC8 directed/random verification
│   └── xrequant_test/ # XREQUANT directed/random verification
├── models/            # Quantized ResNet-18 model
├── results/
│   ├── baseline/      # B0 profiling results
│   ├── x1_mac/        # X1 profiling results
│   └── x2/            # Final X2 profiling results
├── scripts/           # Reproducible workload scripts
├── slides/            # Midterm presentation
└── README.md
```

---

# 1. Bare-Metal Baseline

The baseline implements the complete quantized ResNet-18 graph using integer-only bare-metal C code.

Implemented operators include:

- `CONV_2D`
- `FULLY_CONNECTED`
- requantization
- residual `ADD`
- `MAX_POOL_2D`
- `AVERAGE_POOL_2D`
- `PAD`
- `TRANSPOSE`
- required activation/clamping operations

Memory is statically allocated and no dynamic allocation or floating-point hardware is required.

Full-model correctness is verified against exported reference tensors.

Expected result:

```text
PASS full_model operators=37 outputs=1000 mismatches=0
```

---

# 2. Quantized Model

Model:

```text
models/resnet18_int8_channelwise.tflite
```

The exported model contains:

```text
20 × CONV_2D
8  × ADD
5  × PAD
1  × TRANSPOSE
1  × MAX_POOL_2D
1  × AVERAGE_POOL_2D
1  × FULLY_CONNECTED
```

Python model-export and reference-generation utilities are provided under:

```text
baseline/tools/
optimized/tools/
```

---

# 3. Custom ISA Extensions

## XQMAC8

`XQMAC8` is a packed signed INT8 multiply-accumulate instruction.

Encoding:

```text
Opcode : CUSTOM_0 (0x0B)
funct3 : 0
funct7 : 0
```

Conceptual semantics:

```text
rd =
    rd_old
  + A0 × W0
  + A1 × W1
  + A2 × W2
  + A3 × W3
  + A4 × W4
  + A5 × W5
  + A6 × W6
  + A7 × W7
```

`rs1` and `rs2` each contain eight packed signed INT8 values in one 64-bit register.

One `XQMAC8` therefore performs eight signed INT8 multiply-accumulate operations.

The instruction is integrated into the dominant ResNet-18 convolution inner loops.

Dynamic XQMAC8 execution count:

```text
196,640,768
```

This corresponds to:

```text
1,573,126,144 packed INT8 MAC operations
```

or approximately 92.99% of the valid MAC operations in the full model.

---

## XREQUANT

`XREQUANT` performs fused integer requantization.

Encoding:

```text
Opcode : CUSTOM_0 (0x0B)
funct3 : 1
funct7 : 0
```

Conceptually, one instruction performs:

```text
INT32 accumulator
        ↓
fixed-point multiply
        ↓
rounding
        ↓
shift
        ↓
output zero-point addition
        ↓
INT8 saturation / activation clamp
```

Input accumulator width:

```text
signed INT32
```

Internal fixed-point arithmetic uses:

```text
signed INT64
```

Final output:

```text
signed INT8
```

The second source operand contains packed requantization parameters, including the multiplier, shift, output zero-point, activation minimum, and activation maximum.

Dynamic XREQUANT execution count:

```text
3,236,352
```

This is exactly one XREQUANT execution for every Conv and residual ADD output element accelerated by this instruction.

---

# 4. Spike Custom ISA Implementation

The Spike modifications are stored as:

```text
custom_isa/spike/x2_custom_isa.patch
custom_isa/spike/base_spike_commit.txt
```

The patch contains the implementation of both:

```text
XQMAC8
XREQUANT
```

including:

- instruction encoding
- instruction declaration
- Spike execution semantics
- build integration

## Reconstructing the Custom Spike

Clone Spike:

```bash
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
```

Checkout the base revision recorded by this project:

```bash
git checkout "$(cat /path/to/project/custom_isa/spike/base_spike_commit.txt)"
```

Apply the custom ISA patch:

```bash
git apply /path/to/project/custom_isa/spike/x2_custom_isa.patch
```

Configure and build:

```bash
mkdir build
cd build

../configure --prefix="$HOME/tools/spike-x2"

make -j2
make install
```

The resulting simulator is expected at:

```text
$HOME/tools/spike-x2/bin/spike
```

---

# 5. Custom Instruction Verification

## XQMAC8

```bash
cd custom_isa/xqmac8_test

make clean
make elf
make disasm
make spike
```

Expected result:

```text
PASS: XQMAC8 directed + 1000 randomized tests.
```

## XREQUANT

```bash
cd custom_isa/xrequant_test

make clean
make elf
make disasm
make spike
```

The test compares the custom instruction against the software requantization reference using directed and randomized cases.

---

# 6. Full-Model Execution

## Baseline

Run:

```bash
./scripts/run_baseline.sh
```

or manually:

```bash
cd baseline

make host-test
make elf
make disasm
make spike
```

Expected result:

```text
PASS full_model operators=37 outputs=1000 mismatches=0
```

---

## Optimized X2

Run:

```bash
./scripts/run_optimized.sh
```

or manually:

```bash
cd optimized

make host-test
make elf
make disasm

make spike \
  SPIKE="$HOME/tools/spike-x2/bin/spike"
```

Expected result:

```text
PASS full_model operators=37 outputs=1000 mismatches=0
```

---

# 7. ISA Evaluation Results

All three configurations produce the same full-model output checksum and pass all exported operator-output comparisons.

## Overall Results

| Metric | B0 RV64IMC | X1 + XQMAC8 | X2 + XQMAC8 + XREQUANT |
|---|---:|---:|---:|
| Full-model correctness | PASS | PASS | PASS |
| Dynamic retired instructions | 20,239,232,360 | 4,648,177,001 | 4,696,357,152 |
| Change vs B0 | — | −77.03% | −76.80% |
| Change vs previous stage | — | −77.03% | +1.04% |
| Dense MAC / instruction | 0.08963 | 0.39028 | 0.38627 |
| Memory instructions | 8.848B | 1.266B | 1.427B |
| Custom instruction count | 0 | 196,640,768 | 199,877,120 |
| Custom instruction frequency | 0% | 4.23% | 4.26% |
| `.text` size | 4,976 B | 5,250 B | 5,468 B |

---

## XQMAC8 Effectiveness

The packed MAC extension provides the dominant end-to-end improvement.

Total retired instructions decrease from:

```text
20.239B → 4.648B
```

corresponding to:

```text
77.03% fewer retired instructions
```

Dense MAC efficiency improves from:

```text
0.08963 → 0.39028 MAC/instruction
```

or approximately:

```text
4.35× higher MAC efficiency
```

---

## XREQUANT Incremental Effectiveness

XREQUANT executes:

```text
3,236,352 times
```

corresponding to one execution for every accelerated Conv/ADD output element.

For residual ADD operators:

```text
X1: 103.04 instructions/output
X2:  65.87 instructions/output
```

This corresponds to approximately:

```text
36.08% fewer instructions per ADD output
```

XREQUANT also reduces:

- integer-ALU instructions
- multiply instructions
- branch instructions
- store instructions

However, additional load overhead in the first RGB convolution offsets these local savings.

As a result:

```text
X1 total = 4.648B instructions
X2 total = 4.696B instructions
```

X2 therefore requires approximately 1.04% more retired instructions than X1, while still reducing total instructions by approximately 76.80% relative to the original RV64IMC baseline.

This result is reported as measured rather than assuming that every custom instruction necessarily improves the complete workload.

---

# 8. Instruction-Mix and Detailed Results

Detailed measurement files are stored under:

```text
results/baseline/
results/x1_mac/
results/x2/
```

Available reports include:

```text
instruction_mix.csv
operator_instruction_mix.csv
memory_instruction_profile.csv
function_instruction_profile.csv
pc_instruction_counts.csv
mix_summary.json
operator_profile.csv
profile_summary.json
size.txt
```

These files provide the evidence used for the reported instruction counts, operator contributions, memory-instruction counts, and custom-instruction utilization.

---

# 9. Reproducing Profiling

Baseline:

```bash
./scripts/profile_baseline.sh
```

Optimized:

```bash
./scripts/profile_optimized.sh
```

The profiling flow measures:

- total dynamic retired instruction count
- load instructions
- store instructions
- integer ALU instructions
- multiply instructions
- branch instructions
- dynamic memory instructions
- MACs per retired instruction
- per-operator instruction contribution
- custom instruction utilization
- code size

---

# 10. Notes on Cycle Measurements

Spike is used in this midterm as a functional ISA simulator.

The reported performance results therefore use:

```text
retired instruction counts
instructions/output element
custom-instruction utilization
```

as the ISA-level metrics.

Host wall-clock runtime is not interpreted as target hardware cycle count.

Cycle/output-element measurements and hardware end-to-end speedup require a cycle-aware implementation or timing model and are deferred to the later CV-Wally RTL evaluation.

No PPA claims are made for the midterm evaluation.

---

# 11. Midterm Submission Mapping

The required submission items are provided as follows:

| Required item | Repository location |
|---|---|
| C/C++ bare-metal workload | `baseline/`, `optimized/` |
| Quantized model / conversion utilities | `models/`, `baseline/tools/` |
| Spike/custom ISA implementation | `custom_isa/spike/` |
| Verification tests | `custom_isa/xqmac8_test/`, `custom_isa/xrequant_test/` |
| Reproducible benchmark scripts | `scripts/` |
| Presentation | `slides/midterm.pdf` |
| Run instructions and expected measurements | This README |

---

# Summary

The project demonstrates full-model quantized ResNet-18 inference on bare-metal RV64IMC and evaluates two custom ISA extensions incrementally.

The major result is the XQMAC8 packed INT8 MAC instruction, which reduces full-model retired instructions by 77.03%.

XREQUANT successfully fuses fixed-point requantization, rounding, shifting, zero-point addition, and saturation into one custom instruction and substantially reduces instructions per output for residual ADD operations. Its local arithmetic savings are partially offset by additional load overhead in the first RGB convolution.

All B0, X1, and X2 configurations preserve full-model correctness.