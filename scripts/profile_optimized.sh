#!/bin/bash
set -e

cd "$(dirname "$0")/../optimized"

SPIKE_X2="${SPIKE_X2:-$HOME/tools/spike-x2/bin/spike}"

make profile SPIKE="$SPIKE_X2"
make size
