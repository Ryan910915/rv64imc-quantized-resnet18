#!/bin/bash
set -e

cd "$(dirname "$0")/../baseline"

make host-test
make elf
make disasm
make spike SPIKE="${SPIKE_BASELINE:-spike}"
