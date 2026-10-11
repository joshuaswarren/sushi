#!/bin/sh
# M3d fused-GEMV oracle run: compile the fused shader, generate the fused
# oracle (reference chain R1 from expert_exl3.zig), build the harness, run on
# lavapipe. Exit code is the pass bit. Same conventions as run.sh; the glslc
# here is a shim over glslangValidator.
set -eux
cd "$(dirname "$0")"
ZIG=${ZIG:-$HOME/src/sushi/.zig-toolchain/zig}
PATH="$HOME/.local/bin:$PATH"

glslc --target-env=vulkan1.1 -o exl3_gemv_fused.spv ../exl3_gemv_fused.comp
spirv-val exl3_gemv_fused.spv

"$ZIG" build-exe --dep exl3 -Mroot=gen_moe_fused.zig -Mexl3=../../expert_exl3.zig -O ReleaseSafe --name gen_moe_fused
./gen_moe_fused oracle_fused.bin

gcc -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -o harness_fused harness_fused.c -lvulkan -lm
exec ./harness_fused oracle_fused.bin exl3_gemv_fused.spv
