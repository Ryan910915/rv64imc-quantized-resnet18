#!/bin/bash
set -e

cd "$(dirname "$0")/../optimized"

SPIKE_X2="${SPIKE_X2:-$HOME/tools/spike-x2/bin/spike}"

make host-test
make elf
make disasm
make spike SPIKE="$SPIKE_X2"
