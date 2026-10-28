CC      := riscv64-unknown-elf-gcc
OBJDUMP := riscv64-unknown-elf-objdump

SPIKE_XQMAC := $(HOME)/tools/spike-xqmac/bin/spike

BASELINE := ../../baseline

BUILD := build
ELF   := $(BUILD)/xqmac8_test.elf
DIS   := $(BUILD)/xqmac8_test.dis

CFLAGS := \
	-O2 -g -std=c11 \
	-Wall -Wextra -Werror \
	-march=rv64imc \
	-mabi=lp64 \
	-mcmodel=medany \
	-msmall-data-limit=0 \
	-ffreestanding \
	-fno-builtin \
	-fno-stack-protector \
	-fno-pic \
	-fno-pie \
	-fno-lto \
	-fno-asynchronous-unwind-tables \
	-fno-unwind-tables

LDFLAGS := \
	-nostdlib \
	-nostartfiles \
	-static \
	-Wl,-T,$(BASELINE)/link.ld \
	-Wl,-Map,$(BUILD)/xqmac8_test.map

.PHONY: all elf disasm spike clean

all: spike

$(BUILD):
	mkdir -p $(BUILD)

elf: $(ELF)

$(ELF): main.c xqmac8.h \
	$(BASELINE)/src/start.S \
	$(BASELINE)/src/platform.c \
	$(BASELINE)/src/platform.h \
	$(BASELINE)/link.ld | $(BUILD)
	$(CC) $(CFLAGS) \
		-I$(BASELINE)/src \
		$(BASELINE)/src/start.S \
		$(BASELINE)/src/platform.c \
		main.c \
		$(LDFLAGS) \
		-o $(ELF)

disasm: $(ELF)
	$(OBJDUMP) -d -M no-aliases $(ELF) > $(DIS)
	@echo "Wrote $(DIS)"

spike: $(ELF)
	$(SPIKE_XQMAC) \
		--isa=rv64imc_zicsr \
		-m64 \
		$(ELF)
	@echo "PASS: XQMAC8 directed + 1000 randomized tests."

clean:
	rm -rf $(BUILD)