#!/usr/bin/env bash
# Build MLX with the omarchy Vulkan backend + mlx-c into lib/mlx/{lib,include}
# for Linux builds of sushi (the counterpart of scripts/build-mlx.sh, which is
# macOS/Metal only).
#
# Source layout: omarchy-mlx (github.com/joshuaswarren/omarchy-mlx) pins
# upstream mlx 0.32.3 and adds mlx/backend/omarchy (Honeykrisp Vulkan) with its
# overlay + patch series. Sushi's vendored mlx pin (lib/mlx-src) is the same
# 0.32.3 release, so the mlx-c wrapper + sushi patch set apply unchanged.
#
# Prerequisites (Omarchy/Arch): cmake, ninja, clang (C++20), vulkan-headers,
# vulkan-icd-loader, shaderc or glslang (shader compiler), libwebp, zig 0.17
# (scripts/fetch-zig.sh). The Honeykrisp ICD comes from the installed driver
# (mesa-honeykrisp-omarchy); mlx resolves it in-process at first GPU use.
#
# Usage: MLX_OMARCHY_DIR=/path/to/omarchy-mlx ./scripts/build-mlx-linux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_SRC="$REPO_ROOT/lib/mlx-src"
MLXC_SRC="$REPO_ROOT/lib/mlxc-src"
STAGE="$REPO_ROOT/lib/mlx"
BUILD_ROOT="$REPO_ROOT/lib/.mlx-build"
OMARCHY_DIR="${MLX_OMARCHY_DIR:-$REPO_ROOT/../mlx-omarchy}"
STAMP="$STAGE/.version"

die() { echo "[build-mlx-linux] ERROR: $*" >&2; exit 1; }

[ -f "$MLX_SRC/CMakeLists.txt" ] && [ -f "$MLXC_SRC/CMakeLists.txt" ] \
  || die "submodules missing — run: git submodule update --init lib/mlx-src lib/mlxc-src"
[ -d "$OMARCHY_DIR/overlay/mlx/backend/omarchy" ] \
  || die "omarchy-mlx checkout not found at $OMARCHY_DIR (set MLX_OMARCHY_DIR); it supplies the Vulkan backend overlay + patches"
[ -f "$OMARCHY_DIR/scripts/prepare-mlx.sh" ] \
  || die "$OMARCHY_DIR does not look like omarchy-mlx (no scripts/prepare-mlx.sh)"

# Version pins recorded in the stamp: the omarchy mlx source (authoritative —
# it is what we build: version AND overlay commit), the vendored mlx pin
# (provenance parity check), mlx-c, and the sushi mlxc patch hash.
OMARCHY_MLX_VER="$(grep -oP '^MLX_VERSION=\K.*' "$OMARCHY_DIR/mlx.lock" 2>/dev/null || echo unknown)"
OMARCHY_SHA="$(git -C "$OMARCHY_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
VENDORED_MLX_SHA="$(git -C "$MLX_SRC" rev-parse --short=12 HEAD)"
MLXC_SHA="$(git -C "$MLXC_SRC" rev-parse --short=12 HEAD)"
MLXC_PATCH="$REPO_ROOT/patches/mlxc-gather-qmm-global-scale.patch"
PATCH_SHA="$(sha256sum "$MLXC_PATCH" | cut -c1-12)"
WANT="mlx=$VENDORED_MLX_SHA mlxc=$MLXC_SHA patch=$PATCH_SHA target=omarchy-$OMARCHY_MLX_VER@$OMARCHY_SHA"

if [ -f "$STAMP" ] && [ -f "$STAGE/lib/libmlx.so" ] && [ -f "$STAGE/lib/libmlxc.so" ]; then
  if [ "$(cat "$STAMP")" = "$WANT" ]; then
    echo "[build-mlx-linux] lib/mlx already at ($WANT) — nothing to do"
    exit 0
  fi
  echo "[build-mlx-linux] staged '$(cat "$STAMP")' != '$WANT' — rebuilding"
fi

# ── Prepare the omarchy mlx source (upstream tarball + overlay + patches) ────
( cd "$OMARCHY_DIR" && ./scripts/prepare-mlx.sh )
OMARCHY_MLX_SRC="$OMARCHY_DIR/.work/mlx"
[ -f "$OMARCHY_MLX_SRC/CMakeLists.txt" ] || die "prepare-mlx.sh did not produce $OMARCHY_MLX_SRC"

NCPU="$(nproc)"

# ── mlx (C++ core, omarchy Vulkan backend, CPU fallback) ─────────────────────
cmake -S "$OMARCHY_MLX_SRC" -B "$BUILD_ROOT/mlx" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_BUILD_OMARCHY=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlx" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlx" >/dev/null

# ── mlx-c against the staged mlx (same pairing as the macOS script) ──────────
for p in "$REPO_ROOT"/patches/mlxc-*.patch; do
  git -C "$MLXC_SRC" apply -p1 "$p" 2>/dev/null \
    || echo "[build-mlx-linux] $(basename "$p") already applied"
done
cmake -S "$MLXC_SRC" -B "$BUILD_ROOT/mlxc" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_USE_SYSTEM_MLX=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$STAGE" \
  -DCMAKE_INSTALL_PREFIX="$STAGE"
cmake --build "$BUILD_ROOT/mlxc" -j "$NCPU"
cmake --install "$BUILD_ROOT/mlxc" >/dev/null

[ -f "$STAGE/lib/libmlxc.so" ] || die "libmlxc.so missing from stage"
[ -f "$STAGE/lib/libmlx.so" ] || die "libmlx.so missing from stage"

echo "$WANT" > "$STAMP"
echo "[build-mlx-linux] staged lib/mlx OK: $WANT"
