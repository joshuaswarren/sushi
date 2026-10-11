# Engine: EXL3 experts on a Linux Vulkan GPU

How the non-Metal expert path runs an EXL3 pack on the Vulkan backend of omarchy-mlx, what has been checked on a device, and what is still open. Read this before touching the non-Metal branch of `moeSwigluClamped`, `patches/mlxc-exl3-moe.patch` or the `test-exl3` build step.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md), [engine-expert-streaming](engine-expert-streaming.md).

## The path

| Piece | Where |
|---|---|
| Route | `moeSwigluClamped`, non-Metal branch in `src/exl3/expert_exl3_kernels.zig`. It calls `mlx_fast_exl3_moe`. If the op raises, it logs the MLX error once and runs the host loop (`plainMoeFused`). `SUSHI_EXL3_HOST=1` forces the host loop. |
| C entry points | `mlx_exl3_moe`, `mlx_exl3_decode` in `patches/mlxc-exl3-moe.patch`, applied to the pinned mlx-c. Zig bindings in `src/mlx.zig`. |
| Op | `fast::exl3_moe` in omarchy-mlx (3d1ba9cc or later): GLSL decode of the trellis banks, then the SwiGLU/clamp/combine ops. Per-expert `[E, size]` scale planes are accepted. |
| Log | `[exl3-vk] rows=R slots=S N ms` per call. The ms are enqueue time unless `SUSHI_EXL3_LAYER_UBENCH=1` drains the stream first. |
| Tests | `zig build test-exl3`. The full unit-test artifact does not compile on Linux at `src/expert_io.zig:86`, a problem of the base branch. |

The host loop is the reference for numerics only. It took 9.75 s per MoE layer per forward on the test machine (133 samples, 9.7 to 10.0 s), which is about 7.8 minutes for 48 layers.

## Checked on a device

Machine: M1 Max, 64 GB, Linux, Honeykrisp driver, omarchy-mlx b0a5551da.

- Gate 1, CPU stream against an f64 chain: 34 of 34 tests pass. Two are new: shape and dtype with shared scales (max abs error 4.2e-5) and per-expert scale planes (2.7e-5). The decode check shows 0 bit diffs against the Zig reference.
- Gate 2, GPU stream at pack shapes (hidden 2560, inter 640, top-10, rows 1 and 2, limit 0 and 5, bf16 out): 4 of 4 cases are within 0.02 + 0.01*|ref| of the f64 oracle, the route engaged without a host fallback, and dmesg shows no new GPU fault or timeout lines.
- One serve forward: with the real 65 GB pack, the route made 48 `[exl3-vk]` calls and enqueued a whole 48-layer forward in about 8.5 s.

The gate 2 oracle is an f64 host chain. The Metal project chain (`clampedSortedReference`) is built from MSL custom kernels and cannot run on a Vulkan stream (`as_type<f16vec4>` is unsupported there).

## What fails today

1. Resident serve. After the forward above, `vkAllocateMemory` returns `VK_ERROR_OUT_OF_DEVICE_MEMORY` (request 6.8 MB) and the request ends as HTTP 500. The resident experts (about 29 GB), the 5.25 GB trunk and the page cache of the mmapped files do not fit the device memory budget of a 64 GB machine.
2. Streamed serve (`--ssd-budget-gb 40`). The plan resolves (expert cache 30.67 GB, 512 slots per layer) and the weights load, then `ExpertSlabImportCopied` stops the load. Streaming reads experts into a page-aligned slab and expects the array to alias it. `importSlab` (`src/expert_io.zig`) reports `aliased` only when the array data pointer equals the slab pointer. The Vulkan allocator copies instead, so the check fails.

Two fixes are in progress: a host-pointer import in omarchy-mlx (`VK_EXT_external_memory_host`), and a copy-and-refresh mode for the slab on non-Metal streams.

## Not measured yet

- Whole-model tok/s and KLD against the Metal fixture on Linux.
- Per-layer GPU time through the route. Only the standalone kernel has a number: about 0.49 s per row-1 layer, against 9.75 s on the host.
- Prefill at large row counts.
- Expert id range checking. The op does not reject out-of-range ids, and the host loop does.

## Running it

Build libmlx and libmlxc with `scripts/build-mlx-linux.sh`, then `zig build -Doptimize=ReleaseFast`. On a machine where `cblas.h` lives under `/usr/include/openblas`, pass `BLAS_INCLUDE_DIRS` and `LAPACK_INCLUDE_DIRS` to the libmlx configure step. The test binaries need `LD_LIBRARY_PATH` to point at `lib/mlx/lib`, because the relative rpath does not resolve from `.zig-cache`.
