# Add-on only: does not redefine profile, spike, CFLAGS, sources, or exports.
SPIKE_SRC ?= $(HOME)/tools/src/riscv-isa-sim
MIX_JOBS ?= 2
MIX_WORK ?= build/mix_spike
MIX_SPIKE ?= $(abspath $(MIX_WORK))/install/bin/spike
MIX_OUT ?= results/profile_mix

.PHONY: mix-setup mix-check mix-selftest profile-mix
mix-setup:
	$(PYTHON) profile_mix/setup_spike.py --source "$(SPIKE_SRC)" --work "$(MIX_WORK)" --jobs $(MIX_JOBS)
mix-selftest:
	$(PYTHON) profile_mix/tests/test_tools.py
mix-check:
	$(PYTHON) profile_mix/run_mix.py --smoke-only --spike "$(MIX_SPIKE)" --cross "$(CROSS)" --out "$(MIX_OUT)"
profile-mix:
	$(MAKE) PROFILE=1 elf
	$(PYTHON) profile_mix/run_mix.py --spike "$(MIX_SPIKE)" --cross "$(CROSS)" --isa "$(SPIKE_ISA)" --memory $(SPIKE_MEM) --elf build/profile/resnet.elf --manifest generated/manifest.json --old-csv results/operator_profile.csv --out "$(MIX_OUT)"
