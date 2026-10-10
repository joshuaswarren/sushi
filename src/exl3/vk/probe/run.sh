#!/bin/sh
# Probe run: compile, validate, build, execute. On a device host set
# EXL3_VK_DEVICE=<deviceName substring> (same selector as the oracle harness).
# Exit 3 = the driver contracts acc + he*x into fma (separate-op column
# returned the correctly-rounded fma bits). Exit 0 = no contraction seen.
set -eux
cd "$(dirname "$0")"
PATH="$HOME/.local/bin:$PATH"
glslc --target-env=vulkan1.1 -I. -o probe.spv probe.comp
spirv-val probe.spv
gcc -O2 -std=c11 -Wall -Wextra -o probe probe.c -lvulkan -lm
exec ./probe probe.spv ../test/oracle.bin
