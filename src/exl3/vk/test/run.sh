#!/bin/sh
# M1 oracle run: compile shaders, validate SPIR-V, generate CPU-reference
# vectors, build the harness, run on lavapipe, compare. Exit code is the pass bit.
# Note: glslc here is a shim over glslangValidator; the .comp extension sets
# the stage, and 64-bit ints are avoided so no Int64 feature is needed.
set -eux
cd "$(dirname "$0")"
ZIG=${ZIG:-$HOME/src/sushi/.zig-toolchain/zig}
PATH="$HOME/.local/bin:$PATH"

glslc --target-env=vulkan1.1 -o exl3_decode.spv ../exl3_decode.comp
glslc --target-env=vulkan1.1 -o exl3_mcg_test.spv ../exl3_mcg_test.comp
spirv-val exl3_decode.spv
spirv-val exl3_mcg_test.spv

"$ZIG" build-exe --dep exl3 -Mroot=gen_oracle.zig -Mexl3=../../expert_exl3.zig -O ReleaseSafe --name gen_oracle
./gen_oracle oracle.bin

gcc -O2 -std=c11 -Wall -Wextra -o harness harness.c -lvulkan -lm
exec ./harness oracle.bin exl3_decode.spv exl3_mcg_test.spv
