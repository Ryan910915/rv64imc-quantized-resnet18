#!/bin/bash
set -e

cd "$(dirname "$0")/../baseline"

make profile SPIKE="${SPIKE_BASELINE:-spike}"
make size
