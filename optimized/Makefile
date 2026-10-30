SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
PYTHON ?= python3
CROSS ?= riscv64-unknown-elf-
HOST_CC ?= cc
SPIKE ?= spike
MODEL ?= ../models/resnet18_int8_channelwise.tflite
# Reuse the same graph input as the successful op14 case when it exists.
INPUT ?= $(firstword $(wildcard ../results/conv_case/model_input.npy))
SEED ?= 20261006
PROFILE ?= 0
OPT ?= -O2
SPIKE_ISA ?= rv64imc_zicsr
SPIKE_MEM ?= 64
BUILD := build/$(if $(filter 1,$(PROFILE)),profile,baseline)
LOG := results/$(if $(filter 1,$(PROFILE)),spike_profile,spike).log
INCLUDES := -Isrc -Igenerated
WARN := -Wall -Wextra -Werror
COMMON := $(OPT) -g -std=c11 $(WARN) $(INCLUDES) -DPROFILE=$(PROFILE)
ARCH := -march=rv64imc -mabi=lp64 -mcmodel=medany -msmall-data-limit=0
BARE := -ffreestanding -fno-builtin -fno-stack-protector -fno-pic -fno-pie -fno-lto -fno-asynchronous-unwind-tables -fno-unwind-tables
CFLAGS := $(COMMON) $(ARCH) $(BARE) $(EXTRA_CFLAGS)
CSOURCES := src/main.c src/nn.c src/platform.c generated/model_data.c
OBJECTS := $(BUILD)/main.o $(BUILD)/nn.o $(BUILD)/platform.o $(BUILD)/model_data.o $(BUILD)/model_blob.o $(BUILD)/start.o
GENERATED := generated/model_data.c generated/model_data.h generated/model_blob.S generated/model.bin generated/reference.bin generated/manifest.json
EXPORT_CMD = $(PYTHON) tools/export_model.py --model "$(MODEL)" $(if $(INPUT),--input-npy "$(INPUT)",--seed $(SEED)) --out generated

.PHONY: help export host-test elf disasm spike profile size check-env clean distclean test FORCE
help:
	@echo 'make export && make host-test && make elf && make disasm && make spike'
	@echo 'make profile   # additional full run with per-operator minstret counters'
	@echo 'make test      # synthetic kernel/compiler tests; not full ResNet verification'
	@echo 'MODEL=... INPUT=... may be passed to make export. Existing conv_op14 is untouched.'
export:
	$(EXPORT_CMD)
generated/.stamp: tools/export_model.py tools/flatbuffer_min.py $(MODEL) $(INPUT)
	$(EXPORT_CMD)
$(GENERATED): generated/.stamp
	@test -f "$@" || { echo 'Missing export file; run make export.'; exit 2; }
$(BUILD) build/host results:
	mkdir -p "$@"
FORCE:
$(BUILD)/.config: FORCE | $(BUILD)
	@{ printf '%s\n' '$(CFLAGS)' '$(CROSS)gcc'; $(CROSS)gcc --version | head -n 1; } > "$@.tmp"
	@cmp -s "$@.tmp" "$@" || cp "$@.tmp" "$@"
	@rm -f "$@.tmp"
build/host/.config: FORCE | build/host
	@{ printf '%s\n' '$(COMMON)' '$(HOST_CC)'; $(HOST_CC) --version | head -n 1; } > "$@.tmp"
	@cmp -s "$@.tmp" "$@" || cp "$@.tmp" "$@"
	@rm -f "$@.tmp"
build/host/full_test: build/host/.config $(CSOURCES) src/nn.h src/platform.h $(GENERATED) | build/host
	$(HOST_CC) $(COMMON) -DHOST_TEST $(CSOURCES) generated/model_blob.S -o "$@"
host-test: build/host/full_test | results
	./build/host/full_test | tee results/host.log
	$(PYTHON) tools/check_host.py
$(BUILD)/%.o: src/%.c $(BUILD)/.config src/nn.h src/platform.h generated/model_data.h | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -c "$<" -o "$@"
$(BUILD)/model_data.o: generated/model_data.c $(BUILD)/.config generated/model_data.h src/nn.h | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -c "$<" -o "$@"
$(BUILD)/model_blob.o: generated/model_blob.S $(BUILD)/.config generated/model.bin generated/reference.bin | $(BUILD)
	$(CROSS)gcc $(ARCH) -c "$<" -o "$@"
$(BUILD)/start.o: src/start.S $(BUILD)/.config | $(BUILD)
	$(CROSS)gcc $(ARCH) -DPROFILE=$(PROFILE) -c "$<" -o "$@"
$(BUILD)/resnet.elf: $(OBJECTS) link.ld
	$(CROSS)gcc $(ARCH) $(BARE) -nostdlib -nostartfiles -static -T link.ld -Wl,-Map,$(BUILD)/resnet.map $(OBJECTS) -o "$@"
elf: $(BUILD)/resnet.elf
	$(CROSS)size -A $(BUILD)/resnet.elf > $(BUILD)/size.txt
	$(CROSS)readelf -A $(BUILD)/resnet.elf > $(BUILD)/attributes.txt
	$(PYTHON) tools/record_build.py --elf $(BUILD)/resnet.elf --compiler $(CROSS)gcc --flags '$(CFLAGS)' --profile $(PROFILE)
disasm: elf
	$(CROSS)objdump -d -M no-aliases $(BUILD)/resnet.elf > $(BUILD)/resnet.dis
	$(PYTHON) tools/audit_isa.py $(BUILD)/resnet.dis
size: elf
	cat $(BUILD)/size.txt
spike: elf | results
	$(PYTHON) tools/run_spike.py --spike "$(SPIKE)" --isa "$(SPIKE_ISA)" --memory $(SPIKE_MEM) --elf $(BUILD)/resnet.elf --log $(LOG)
profile:
	$(MAKE) PROFILE=1 spike
	$(PYTHON) tools/profile_summary.py results/spike_profile.log
check-env:
	@command -v $(HOST_CC)
	@command -v $(CROSS)gcc
	@command -v $(SPIKE)
	@$(PYTHON) -c 'import numpy; from ai_edge_litert.interpreter import Interpreter; print("Host dependencies OK")'
test:
	$(PYTHON) tests/test_suite.py
clean:
	rm -rf build
# Explicitly removes ONLY this package's exports. Does not touch parent results/models.
distclean: clean
	rm -rf generated

# Optional measurement-only profiling add-on
include profile_mix/mix.mk
