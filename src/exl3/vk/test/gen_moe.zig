//! M2 oracle generator: MoE-layer test vectors for exl3_gemv.comp +
//! exl3_moe.comp on lavapipe.
//!
//! Two expected-output references per case:
//!   R1 — the in-tree CPU reference chain (expert_exl3.zig):
//!        prepareInput -> innerGemv -> finishOutput per projection, SwiGLU in
//!        f32 on the f16-valued gate/up, score applied to the finished per-slot
//!        output, f32 accumulate over slots, one final f16/bf16 round. The
//!        shader must match this bit-exactly outside exp() ULP noise.
//!   R2 — a mirror of the production host path plainMoeFused
//!        (expert_exl3_kernels.zig:3045-3300): dense f16 weights from
//!        reconstructPublic, raw x, dense GEMV accumulation, one final round.
//!        R1 vs R2 differ only in rounding points; the harness reports the
//!        error table (scope 5(i): max-abs <= 1 ulp bf16).
//!
//! Case families:
//!   block128  — six 128x128 experts composed from the 18 M1 oracle blocks
//!               (blocks 3e/3e+1/3e+2 = gate/up/down of expert e), rows 3,
//!               topk 2. Includes the edge-scale and zero-trellis blocks, so
//!               R2 compare is disabled for this case (flags bit 0 clear):
//!               f16 overflow produces legitimately different inf/NaN patterns
//!               in dense vs fused rounding.
//!   pack-rN   — Qwen3.8 pack shapes: hidden 2560, inter 640, topk 10,
//!               window 15 mcg K=2, 32 random experts, rows in
//!               {1,2,4,8,17,289}, one rows=2 case with limit=5.
//!
//! Random trellis needs no magnitude control: with mask 0x7FFF the mcg pair
//! halves carry exponent bits {6,7,14,15} (0x3B60 xored under 0x8FFF), so
//! every decoded weight has |w| <= 4 — asserted here, not assumed.
//!
//! File layout (all little-endian):
//!   u32 magic "XL3M" (0x4D334C58), u32 version 1, u32 caseCount
//!   per case: char name[16]; u32 rows, hidden, inter, experts, topk, n, mask,
//!             flags; f32 limit; then arrays in this order:
//!     x rows*hidden f32; slots rows*topk u32; scores rows*topk f32;
//!     tg,tu,td  E*tstride u16 (tstride = (hidden/16)*(inter/16)*n);
//!     suhg,suhu E*hidden u16; suhd E*inter u16;
//!     svhg,svhu E*inter u16; svhd E*hidden u16;
//!     ag,au,h16 rows*topk*inter u16; xtd,slotout rows*topk*hidden u16;
//!     r1out32 rows*hidden f32; r1out16,r1outbf rows*hidden u16;
//!     r2out32 rows*hidden f32; r2out16,r2outbf rows*hidden u16
const std = @import("std");
const exl3 = @import("exl3");

const HW_PER_TILE = 32; // K=2
const RATE: exl3.Rate = exl3.kFromPackedDim(HW_PER_TILE).?;
const DEC: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };

fn bf16RNE(b: u32) u16 {
    const lsb: u32 = (b >> 16) & 1;
    return @truncate((b +% 0x7FFF +% lsb) >> 16);
}

const Case = struct {
    name: []const u8,
    rows: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    topk: usize,
    limit: f32,
    gate_r2: bool,

    x: []f32 = &.{},
    slots: []u32 = &.{},
    scores: []f32 = &.{},
    tg: []u16 = &.{},
    tu: []u16 = &.{},
    td: []u16 = &.{},
    suhg: []u16 = &.{},
    suhu: []u16 = &.{},
    suhd: []u16 = &.{},
    svhg: []u16 = &.{},
    svhu: []u16 = &.{},
    svhd: []u16 = &.{},
    ag: []u16 = &.{},
    au: []u16 = &.{},
    h16: []u16 = &.{},
    xtd: []u16 = &.{},
    slotout: []u16 = &.{},
    r1o32: []f32 = &.{},
    r1o16: []u16 = &.{},
    r1obf: []u16 = &.{},
    r2o32: []f32 = &.{},
    r2o16: []u16 = &.{},
    r2obf: []u16 = &.{},

    fn tstride(c: *const Case) usize {
        return (c.hidden / 16) * (c.inter / 16) * HW_PER_TILE;
    }
};

// --- random helpers ---------------------------------------------------------

/// Random finite f16 with |v| in [2^(lo-15), 2^(hi-15+1)): exp in lo..hi.
fn randF16(rnd: std.Random, lo: u6, hi: u6) u16 {
    const e = lo + rnd.uintLessThan(u6, hi - lo + 1);
    return (@as(u16, rnd.int(u1)) << 15) | (@as(u16, e) << 10) | rnd.int(u10);
}

fn randX(rnd: std.Random) f32 {
    return exl3.f16BitsToF32(randF16(rnd, 13, 14)); // |x| in [0.25, 1]
}

fn randScale(rnd: std.Random) u16 {
    return randF16(rnd, 13, 14); // |s| in [0.25, 1]
}

// --- reference R1: expert_exl3.zig project chain ---------------------------

/// Decoded-tile cache: for every (projection, expert, tile) the 256 decoded
/// f16 weights in tensorCorePerm placement — byte-for-byte what exl3.innerGemv
/// derives from decodeTile. decodeTile is a pure function of the tile bits, so
/// serving it from the cache is bit-identical to decoding per call and removes
/// the dominant generator cost (the tile decode, measured 2.9 s per MoE row).
/// innerGemvCached below keeps innerGemv's exact loop and accumulation order.
const PackCache = struct {
    tiles_g: []u16,
    tiles_u: []u16,
    tiles_d: []u16,
    gw: []u16 = &.{},
    uw: []u16 = &.{},
    dw: []u16 = &.{},
};

fn buildTiles(gpa: std.mem.Allocator, c: *const Case, trellis: []const u16) ![]u16 {
    const ts = c.tstride();
    var out = try gpa.alloc(u16, c.experts * ts / HW_PER_TILE * 256);
    var tile_bits: [HW_PER_TILE]u16 = undefined;
    var tile_w: [256]u16 = undefined;
    for (0..c.experts * ts / HW_PER_TILE) |ti| {
        @memcpy(&tile_bits, trellis[ti * HW_PER_TILE ..][0..HW_PER_TILE]);
        exl3.decodeTile(&tile_bits, RATE, DEC, &tile_w);
        @memcpy(out[ti * 256 ..][0..256], &tile_w);
    }
    return out;
}

/// exl3.innerGemv with the tile decode replaced by the cache. Identical
/// accumulation order (in-tile major, then the tile's 16 rows) and rounding.
fn innerGemvCached(
    tiles: []const u16,
    transformed: []const f32,
    in_features: usize,
    out_features: usize,
    out: []f32,
) void {
    const in_tiles = in_features / 16;
    const out_tiles = out_features / 16;
    @memset(out, 0);
    for (0..in_tiles) |tk| {
        for (0..out_tiles) |tn| {
            const tile_w = tiles[(tk * out_tiles + tn) * 256 ..][0..256];
            const xbase = tk * 16;
            const ybase = tn * 16;
            for (0..16) |r| {
                const xv = transformed[xbase + r];
                for (0..16) |c| {
                    out[ybase + c] += xv * exl3.f16BitsToF32(tile_w[r * 16 + c]);
                }
            }
        }
    }
    for (out) |*v| v.* = exl3.f16BitsToF32(exl3.f32ToF16Bits(v.*));
}

fn moeProjectRef(
    c: *const Case,
    cache: *const PackCache,
    scratch: anytype, // struct of scratch slices
    acc: []f32,
) void {
    const ts = c.tstride();
    const tstride_cache = ts / HW_PER_TILE * 256;
    for (0..c.rows) |r| {
        const xr = c.x[r * c.hidden ..][0..c.hidden];
        const accr = acc[r * c.hidden ..][0..c.hidden];
        @memset(accr, 0);
        for (0..c.topk) |j| {
            const slot = r * c.topk + j;
            const e: usize = c.slots[slot];
            const sc = c.scores[slot];

            // gate + up: prepareInput -> innerGemv -> finishOutput
            exl3.prepareInput(xr, c.suhg[e * c.hidden ..][0..c.hidden], scratch.xt_g);
            innerGemvCached(cache.tiles_g[e * tstride_cache ..][0..tstride_cache], scratch.xt_g, c.hidden, c.inter, scratch.agf);
            exl3.finishOutput(scratch.agf, c.svhg[e * c.inter ..][0..c.inter], scratch.ag2);
            exl3.prepareInput(xr, c.suhu[e * c.hidden ..][0..c.hidden], scratch.xt_u);
            innerGemvCached(cache.tiles_u[e * tstride_cache ..][0..tstride_cache], scratch.xt_u, c.hidden, c.inter, scratch.auf);
            exl3.finishOutput(scratch.auf, c.svhu[e * c.inter ..][0..c.inter], scratch.auf2);

            // SwiGLU (plainMoeFused semantics) + f16 intermediates
            for (0..c.inter) |o| {
                var a = scratch.ag2[o];
                var u = scratch.auf2[o];
                if (c.limit > 0) {
                    a = @min(a, c.limit);
                    const am = @min(@abs(u), c.limit);
                    u = if (u < 0) -am else am;
                }
                const sig = @as(f32, 1.0) / (1.0 + @exp(-a));
                const h = (a * sig) * u;
                scratch.h[o] = h;
                c.ag[slot * c.inter + o] = exl3.f32ToF16Bits(scratch.ag2[o]);
                c.au[slot * c.inter + o] = exl3.f32ToF16Bits(scratch.auf2[o]);
                c.h16[slot * c.inter + o] = exl3.f32ToF16Bits(h);
            }

            // down projection on the SwiGLU output
            exl3.prepareInput(scratch.h, c.suhd[e * c.inter ..][0..c.inter], scratch.xtdf);
            for (0..c.inter) |o| c.xtd[slot * c.inter + o] = exl3.f32ToF16Bits(scratch.xtdf[o]);
            innerGemvCached(cache.tiles_d[e * tstride_cache ..][0..tstride_cache], scratch.xtdf, c.inter, c.hidden, scratch.y);
            exl3.finishOutput(scratch.y, c.svhd[e * c.hidden ..][0..c.hidden], scratch.y2);
            for (0..c.hidden) |o| {
                c.slotout[slot * c.hidden + o] = exl3.f32ToF16Bits(scratch.y2[o]);
                accr[o] += sc * scratch.y2[o];
            }
        }
    }
}

// --- reference R2: plainMoeFused mirror ------------------------------------

fn moePlainRef(
    c: *const Case,
    gw: []const u16,
    uw: []const u16,
    dw: []const u16,
    scratch: anytype,
    acc: []f32,
) void {
    const hw = c.hidden * c.inter;
    for (0..c.rows) |r| {
        const xr = c.x[r * c.hidden ..][0..c.hidden];
        const accr = acc[r * c.hidden ..][0..c.hidden];
        @memset(accr, 0);
        for (0..c.topk) |j| {
            const slot = r * c.topk + j;
            const e: usize = c.slots[slot];
            const sc = c.scores[slot];
            const gwe = gw[e * hw ..][0..hw];
            const uwe = uw[e * hw ..][0..hw];
            const dwe = dw[e * hw ..][0..hw];
            @memset(scratch.agf, 0);
            @memset(scratch.auf, 0);
            for (0..c.hidden) |k| {
                const xv = xr[k];
                for (0..c.inter) |o| {
                    scratch.agf[o] += xv * exl3.f16BitsToF32(gwe[k * c.inter + o]);
                    scratch.auf[o] += xv * exl3.f16BitsToF32(uwe[k * c.inter + o]);
                }
            }
            for (0..c.inter) |o| {
                var a = scratch.agf[o];
                var u = scratch.auf[o];
                if (c.limit > 0) {
                    a = @min(a, c.limit);
                    const am = @min(@abs(u), c.limit);
                    u = if (u < 0) -am else am;
                }
                const sig = @as(f32, 1.0) / (1.0 + @exp(-a));
                scratch.h[o] = (a * sig) * u * sc;
            }
            for (0..c.inter) |k| {
                const gy = scratch.h[k];
                for (0..c.hidden) |o| {
                    accr[o] += gy * exl3.f16BitsToF32(dwe[k * c.hidden + o]);
                }
            }
        }
    }
}

// --- case construction -----------------------------------------------------

fn allocCase(gpa: std.mem.Allocator, c: *Case) !void {
    const rts = c.rows * c.topk;
    c.x = try gpa.alloc(f32, c.rows * c.hidden);
    c.slots = try gpa.alloc(u32, rts);
    c.scores = try gpa.alloc(f32, rts);
    const ts = c.tstride();
    c.tg = try gpa.alloc(u16, c.experts * ts);
    c.tu = try gpa.alloc(u16, c.experts * ts);
    c.td = try gpa.alloc(u16, c.experts * ts);
    c.suhg = try gpa.alloc(u16, c.experts * c.hidden);
    c.suhu = try gpa.alloc(u16, c.experts * c.hidden);
    c.suhd = try gpa.alloc(u16, c.experts * c.inter);
    c.svhg = try gpa.alloc(u16, c.experts * c.inter);
    c.svhu = try gpa.alloc(u16, c.experts * c.inter);
    c.svhd = try gpa.alloc(u16, c.experts * c.hidden);
    c.ag = try gpa.alloc(u16, rts * c.inter);
    c.au = try gpa.alloc(u16, rts * c.inter);
    c.h16 = try gpa.alloc(u16, rts * c.inter);
    c.xtd = try gpa.alloc(u16, rts * c.inter);
    c.slotout = try gpa.alloc(u16, rts * c.hidden);
    c.r1o32 = try gpa.alloc(f32, c.rows * c.hidden);
    c.r1o16 = try gpa.alloc(u16, c.rows * c.hidden);
    c.r1obf = try gpa.alloc(u16, c.rows * c.hidden);
    c.r2o32 = try gpa.alloc(f32, c.rows * c.hidden);
    c.r2o16 = try gpa.alloc(u16, c.rows * c.hidden);
    c.r2obf = try gpa.alloc(u16, c.rows * c.hidden);
}

const Scratch = struct {
    xt_g: []f32,
    xt_u: []f32,
    agf: []f32,
    auf: []f32,
    h: []f32,
    xtdf: []f32,
    y: []f32,
    ag2: []f32,
    auf2: []f32,
    y2: []f32,
};

fn genPackCase(gpa: std.mem.Allocator, rnd: std.Random, c: *Case) !void {
    try allocCase(gpa, c);
    const ts = c.tstride();
    var tile: [256]u16 = undefined;
    var max_w: f32 = 0;
    for (0..c.experts) |e| {
        for (0..3) |proj| {
            const t = switch (proj) {
                0 => &c.tg,
                1 => &c.tu,
                else => &c.td,
            };
            for (0..ts / HW_PER_TILE) |ti| {
                var tile_bits: [HW_PER_TILE]u16 = undefined;
                for (&tile_bits) |*x| x.* = rnd.int(u16);
                exl3.decodeTile(&tile_bits, RATE, DEC, &tile);
                for (tile) |bits| {
                    const v = @abs(exl3.f16BitsToF32(bits));
                    if (v > max_w) max_w = v;
                }
                @memcpy(t.*[e * ts + ti * HW_PER_TILE ..][0..HW_PER_TILE], &tile_bits);
            }
        }
        for (0..c.hidden) |i| {
            c.suhg[e * c.hidden + i] = randScale(rnd);
            c.suhu[e * c.hidden + i] = randScale(rnd);
            c.svhd[e * c.hidden + i] = randScale(rnd);
        }
        for (0..c.inter) |i| {
            c.suhd[e * c.inter + i] = randScale(rnd);
            c.svhg[e * c.inter + i] = randScale(rnd);
            c.svhu[e * c.inter + i] = randScale(rnd);
        }
    }
    if (max_w > 4.0) return error.TrellisWeightOutOfRange;

    for (0..c.rows) |r| {
        for (0..c.hidden) |i| c.x[r * c.hidden + i] = randX(rnd);
        // topk distinct experts per row
        var perm: [512]u32 = undefined;
        for (&perm, 0..) |*p, i| p.* = @intCast(i);
        var n: usize = c.experts;
        for (0..c.topk) |j| {
            const k = rnd.uintLessThan(usize, n);
            c.slots[r * c.topk + j] = perm[k];
            perm[k] = perm[n - 1];
            n -= 1;
        }
        var sum: f32 = 0;
        for (0..c.topk) |j| {
            const v = exl3.f16BitsToF32(randF16(rnd, 13, 15));
            c.scores[r * c.topk + j] = v;
            sum += v;
        }
        for (0..c.topk) |j| c.scores[r * c.topk + j] /= sum;
    }
}

/// block128: six experts assembled from the 18 M1 oracle blocks
/// (expert e: gate = block 3e, up = 3e+1, down = 3e+2; 128x128 each).
fn genBlockCase(gpa: std.mem.Allocator, rnd: std.Random, c: *Case, oracle: []const u8) !void {
    const blk = struct {
        fn trellis(o: []const u8, b: usize, out: []u16) void {
            const base = 20 + 2 * 65536 + b * 233984;
            for (0..2048) |i| {
                out[i] = std.mem.readInt(u16, o[base + 2 * i ..][0..2], .little);
            }
        }
        fn scales(o: []const u8, b: usize, comptime suh: bool, out: []u16) void {
            const off: usize = if (suh) 4096 else 4352;
            const base = 20 + 2 * 65536 + b * 233984 + off;
            for (0..128) |i| out[i] = std.mem.readInt(u16, o[base + 2 * i ..][0..2], .little);
        }
    };
    try allocCase(gpa, c);
    const ts = c.tstride();
    std.debug.assert(ts == 2048); // 128x128 block = 64 tiles of 32 hw
    for (0..c.experts) |e| {
        blk.trellis(oracle, 3 * e, c.tg[e * ts ..][0..ts]);
        blk.trellis(oracle, 3 * e + 1, c.tu[e * ts ..][0..ts]);
        blk.trellis(oracle, 3 * e + 2, c.td[e * ts ..][0..ts]);
        if (e < 4) {
            // The M1 blocks carry full-domain f16 scales; fed through
            // prepareInput they overflow f16 and degenerate the whole case to
            // inf/NaN. Experts 0..3 get tame scales so the gate comparisons
            // are meaningful; experts 4 (edge scales) and 5 (zero trellis)
            // keep the M1 data for the overflow-path coverage.
            for (0..c.hidden) |i| {
                c.suhg[e * c.hidden + i] = randScale(rnd);
                c.suhu[e * c.hidden + i] = randScale(rnd);
                c.svhd[e * c.hidden + i] = randScale(rnd);
            }
            for (0..c.inter) |i| {
                c.suhd[e * c.inter + i] = randScale(rnd);
                c.svhg[e * c.inter + i] = randScale(rnd);
                c.svhu[e * c.inter + i] = randScale(rnd);
            }
        } else {
            blk.scales(oracle, 3 * e, true, c.suhg[e * c.hidden ..][0..c.hidden]);
            blk.scales(oracle, 3 * e + 1, true, c.suhu[e * c.hidden ..][0..c.hidden]);
            blk.scales(oracle, 3 * e + 2, true, c.suhd[e * c.inter ..][0..c.inter]);
            blk.scales(oracle, 3 * e, false, c.svhg[e * c.inter ..][0..c.inter]);
            blk.scales(oracle, 3 * e + 1, false, c.svhu[e * c.inter ..][0..c.inter]);
            blk.scales(oracle, 3 * e + 2, false, c.svhd[e * c.hidden ..][0..c.hidden]);
        }
    }
    for (0..c.rows) |r| {
        for (0..c.hidden) |i| c.x[r * c.hidden + i] = randX(rnd);
        var perm: [8]u32 = undefined;
        for (&perm, 0..) |*p, i| p.* = @intCast(i);
        var n: usize = c.experts;
        for (0..c.topk) |j| {
            const k = rnd.uintLessThan(usize, n);
            c.slots[r * c.topk + j] = perm[k];
            perm[k] = perm[n - 1];
            n -= 1;
        }
        var sum: f32 = 0;
        for (0..c.topk) |j| {
            const v = exl3.f16BitsToF32(randF16(rnd, 13, 15));
            c.scores[r * c.topk + j] = v;
            sum += v;
        }
        for (0..c.topk) |j| c.scores[r * c.topk + j] /= sum;
    }
}

// --- output -----------------------------------------------------------------

fn put16(w: anytype, v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try w.writeAll(&b);
}
fn putF32(w: anytype, v: f32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, @bitCast(v), .little);
    try w.writeAll(&b);
}
fn put32(w: anytype, v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try w.writeAll(&b);
}

fn putName(w: anytype, name: []const u8) !void {
    var buf: [16]u8 = @splat(0);
    @memcpy(buf[0..name.len], name);
    try w.writeAll(&buf);
}

fn computeR1(gpa: std.mem.Allocator, c: *Case, cache: *const PackCache) !void {
    const scratch = Scratch{
        .xt_g = try gpa.alloc(f32, c.hidden),
        .xt_u = try gpa.alloc(f32, c.hidden),
        .agf = try gpa.alloc(f32, c.inter),
        .auf = try gpa.alloc(f32, c.inter),
        .h = try gpa.alloc(f32, c.inter),
        .xtdf = try gpa.alloc(f32, c.inter),
        .y = try gpa.alloc(f32, c.hidden),
        .ag2 = try gpa.alloc(f32, c.inter),
        .auf2 = try gpa.alloc(f32, c.inter),
        .y2 = try gpa.alloc(f32, c.hidden),
    };
    const acc = try gpa.alloc(f32, c.rows * c.hidden);
    moeProjectRef(c, cache, scratch, acc);
    for (acc, 0..) |v, i| {
        c.r1o32[i] = v;
        c.r1o16[i] = exl3.f32ToF16Bits(v);
        c.r1obf[i] = bf16RNE(@bitCast(v));
    }
}

/// Dense plainMoeFused weights: 96 reconstructPublic calls cost ~4.5 min, so
/// they are cached on disk next to the oracle (deterministic seed -> safe).
const DenseHdr = struct { magic: u32, hw: u32, experts: u32 };

fn denseFromScratch(gpa: std.mem.Allocator, c: *Case, cache: *PackCache, io: std.Io, cache_path: []const u8) !void {
    const hw = c.hidden * c.inter;
    const total = 3 * c.experts * hw;
    if (std.Io.Dir.cwd().openFile(io, cache_path, .{})) |f| {
        defer f.close(io);
        var head: [12]u8 = undefined;
        {
            var fr = f.reader(io, &.{});
            try fr.interface.readSliceAll(&head);
        }
        const magic = std.mem.readInt(u32, head[0..4], .little);
        const hwv = std.mem.readInt(u32, head[4..8], .little);
        const ev = std.mem.readInt(u32, head[8..12], .little);
        if (magic == 0x444E5345 and hwv == hw and ev == c.experts and
            try f.length(io) == 12 + 2 * total)
        {
            const gwb = try gpa.alloc(u16, total);
            {
                var fr = f.reader(io, &.{});
                try fr.interface.readSliceAll(std.mem.sliceAsBytes(gwb));
            }
            for (gwb) |*v| v.* = std.mem.readInt(u16, std.mem.asBytes(v), .little);
            cache.gw = gwb[0 .. c.experts * hw];
            cache.uw = gwb[c.experts * hw ..][0 .. c.experts * hw];
            cache.dw = gwb[2 * c.experts * hw ..][0 .. c.experts * hw];
            std.debug.print("dense cache: loaded {s}\n", .{cache_path});
            return;
        }
    } else |_| {}
    cache.gw = try gpa.alloc(u16, c.experts * hw);
    cache.uw = try gpa.alloc(u16, c.experts * hw);
    cache.dw = try gpa.alloc(u16, c.experts * hw);
    const ts = c.tstride();
    for (0..c.experts) |e| {
        try exl3.reconstructPublic(gpa, c.tg[e * ts ..][0..ts], c.suhg[e * c.hidden ..][0..c.hidden], c.svhg[e * c.inter ..][0..c.inter], c.hidden, c.inter, RATE, DEC, cache.gw[e * hw ..][0..hw]);
        try exl3.reconstructPublic(gpa, c.tu[e * ts ..][0..ts], c.suhu[e * c.hidden ..][0..c.hidden], c.svhu[e * c.inter ..][0..c.inter], c.hidden, c.inter, RATE, DEC, cache.uw[e * hw ..][0..hw]);
        try exl3.reconstructPublic(gpa, c.td[e * ts ..][0..ts], c.suhd[e * c.inter ..][0..c.inter], c.svhd[e * c.hidden ..][0..c.hidden], c.inter, c.hidden, RATE, DEC, cache.dw[e * hw ..][0..hw]);
    }
    // Write-through for the next run.
    var file = try std.Io.Dir.cwd().createFile(io, cache_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    var head: [12]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], 0x444E5345, .little);
    std.mem.writeInt(u32, head[4..8], @intCast(hw), .little);
    std.mem.writeInt(u32, head[8..12], @intCast(c.experts), .little);
    try w.writeAll(&head);
    for (cache.gw) |v| try put16(w, v);
    for (cache.uw) |v| try put16(w, v);
    for (cache.dw) |v| try put16(w, v);
    try w.flush();
    std.debug.print("dense cache: wrote {s}\n", .{cache_path});
}

fn computeR2(gpa: std.mem.Allocator, c: *Case, cache: *const PackCache) !void {
    const scratch = Scratch{
        .xt_g = &.{},
        .xt_u = &.{},
        .agf = try gpa.alloc(f32, c.inter),
        .auf = try gpa.alloc(f32, c.inter),
        .h = try gpa.alloc(f32, c.inter),
        .xtdf = &.{},
        .y = &.{},
        .ag2 = &.{},
        .auf2 = &.{},
        .y2 = &.{},
    };
    const acc = try gpa.alloc(f32, c.rows * c.hidden);
    moePlainRef(c, cache.gw, cache.uw, cache.dw, scratch, acc);
    for (acc, 0..) |v, i| {
        c.r2o32[i] = v;
        c.r2o16[i] = exl3.f32ToF16Bits(v);
        c.r2obf[i] = bf16RNE(@bitCast(v));
    }
}

fn writeCase(w: anytype, c: *const Case) !void {
    try putName(w, c.name);
    try put32(w, @intCast(c.rows));
    try put32(w, @intCast(c.hidden));
    try put32(w, @intCast(c.inter));
    try put32(w, @intCast(c.experts));
    try put32(w, @intCast(c.topk));
    try put32(w, RATE.n);
    try put32(w, DEC.window.mask());
    try put32(w, if (c.gate_r2) 1 else 0);
    try putF32(w, c.limit);
    for (c.x) |v| try putF32(w, v);
    for (c.slots) |v| try put32(w, v);
    for (c.scores) |v| try putF32(w, v);
    for (c.tg) |v| try put16(w, v);
    for (c.tu) |v| try put16(w, v);
    for (c.td) |v| try put16(w, v);
    for (c.suhg) |v| try put16(w, v);
    for (c.suhu) |v| try put16(w, v);
    for (c.suhd) |v| try put16(w, v);
    for (c.svhg) |v| try put16(w, v);
    for (c.svhu) |v| try put16(w, v);
    for (c.svhd) |v| try put16(w, v);
    for (c.ag) |v| try put16(w, v);
    for (c.au) |v| try put16(w, v);
    for (c.h16) |v| try put16(w, v);
    for (c.xtd) |v| try put16(w, v);
    for (c.slotout) |v| try put16(w, v);
    for (c.r1o32) |v| try putF32(w, v);
    for (c.r1o16) |v| try put16(w, v);
    for (c.r1obf) |v| try put16(w, v);
    for (c.r2o32) |v| try putF32(w, v);
    for (c.r2o16) |v| try put16(w, v);
    for (c.r2obf) |v| try put16(w, v);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const oracle_path = args.next() orelse return error.Usage;
    const out_path = args.next() orelse return error.Usage;
    // "quick" drops the 289-row timing case so the shader path can be tested
    // without waiting for the full reference compute. "tiny" also keeps only the
    // 1- and 2-row pack cases: the first lab-GPU run keeps every dispatch short.
    const mode = args.next();
    const tiny = if (mode) |a| std.mem.eql(u8, a, "tiny") else false;
    const quick = tiny or (if (mode) |a| std.mem.eql(u8, a, "quick") else false);

    // Everything below is transient generator state; one arena free keeps the
    // safe allocator quiet without bookkeeping per case.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    // Read the M1 oracle for the block128 case.
    var of = try std.Io.Dir.cwd().openFile(io, oracle_path, .{});
    defer of.close(io);
    const olen = try of.length(io);
    const oracle = try aa.alloc(u8, @intCast(olen));
    {
        var fr = of.reader(io, &.{});
        try fr.interface.readSliceAll(oracle);
    }

    var prng = std.Random.DefaultPrng.init(0x5EEDC0DE);
    const rnd = prng.random();

    var cases: std.ArrayList(Case) = .empty;
    // block128: 6 experts from the 18 M1 blocks, rows 3, topk 2.
    try cases.append(aa, .{ .name = "block128", .rows = 3, .hidden = 128, .inter = 128, .experts = 6, .topk = 2, .limit = 0, .gate_r2 = false });
    // Qwen3.8 pack shapes: hidden 2560, inter 640, topk 10, 32 random experts.
    const row_sets = [_]struct { name: []const u8, rows: usize, limit: f32 }{
        .{ .name = "pack-r1", .rows = 1, .limit = 0 },
        .{ .name = "pack-r2-limit5", .rows = 2, .limit = 5.0 },
        .{ .name = "pack-r2", .rows = 2, .limit = 0 },
        .{ .name = "pack-r4", .rows = 4, .limit = 0 },
        .{ .name = "pack-r8", .rows = 8, .limit = 0 },
        .{ .name = "pack-r17", .rows = 17, .limit = 0 },
        .{ .name = "pack-r289", .rows = 289, .limit = 0 },
    };
    for (row_sets) |rs| {
        if (quick and std.mem.eql(u8, rs.name, "pack-r289")) continue;
        if (tiny and rs.rows > 2) continue;
        try cases.append(aa, .{ .name = rs.name, .rows = rs.rows, .hidden = 2560, .inter = 640, .experts = 32, .topk = 10, .limit = rs.limit, .gate_r2 = true });
    }

    // Fresh trellis per pack case would cost nothing, but sharing one random
    // pool across row variants keeps the timing comparison apples-to-apples:
    // generate pack-r1 first, then clone its tensors for the other row counts
    // and regenerate only x/slots/scores.
    const pack_first: usize = 1;
    for (cases.items, 0..) |*c, ci| {
        if (ci == 0) {
            try genBlockCase(aa, rnd, c, oracle);
        } else if (ci == pack_first) {
            try genPackCase(aa, rnd, c);
        } else {
            const src = &cases.items[pack_first];
            try allocCase(aa, c);
            @memcpy(c.tg, src.tg);
            @memcpy(c.tu, src.tu);
            @memcpy(c.td, src.td);
            @memcpy(c.suhg, src.suhg);
            @memcpy(c.suhu, src.suhu);
            @memcpy(c.suhd, src.suhd);
            @memcpy(c.svhg, src.svhg);
            @memcpy(c.svhu, src.svhu);
            @memcpy(c.svhd, src.svhd);
            for (0..c.rows) |r| {
                for (0..c.hidden) |i| c.x[r * c.hidden + i] = randX(rnd);
                var perm: [512]u32 = undefined;
                for (&perm, 0..) |*p, i| p.* = @intCast(i);
                var n: usize = c.experts;
                for (0..c.topk) |j| {
                    const k = rnd.uintLessThan(usize, n);
                    c.slots[r * c.topk + j] = perm[k];
                    perm[k] = perm[n - 1];
                    n -= 1;
                }
                var sum: f32 = 0;
                for (0..c.topk) |j| {
                    const v = exl3.f16BitsToF32(randF16(rnd, 13, 15));
                    c.scores[r * c.topk + j] = v;
                    sum += v;
                }
                for (0..c.topk) |j| c.scores[r * c.topk + j] /= sum;
            }
        }
    }

    // References + timings. Tile and dense caches are built once per family:
    // the row-variant pack cases share identical tensors (clone + fresh
    // x/slots/scores), so one decode pass and one reconstruct pass serve all.
    var block_cache: PackCache = undefined;
    var pack_cache: PackCache = undefined;
    var have_block = false;
    var have_pack = false;
    const dense_cache_path = try std.fmt.allocPrint(aa, "{s}.dense", .{out_path});
    defer aa.free(dense_cache_path);
    for (cases.items, 0..) |*c, ci| {
        const is_pack = ci != 0;
        if (!is_pack and !have_block) {
            block_cache = .{
                .tiles_g = try buildTiles(aa, c, c.tg),
                .tiles_u = try buildTiles(aa, c, c.tu),
                .tiles_d = try buildTiles(aa, c, c.td),
            };
            have_block = true;
        }
        if (is_pack and !have_pack) {
            pack_cache = .{
                .tiles_g = try buildTiles(aa, c, c.tg),
                .tiles_u = try buildTiles(aa, c, c.tu),
                .tiles_d = try buildTiles(aa, c, c.td),
            };
            if (c.gate_r2) try denseFromScratch(aa, c, &pack_cache, io, dense_cache_path);
            // Self-check: the tile cache must reproduce exl3.innerGemv
            // bit-exactly (both f32 accumulators), or R1 is not the reference.
            const ts = c.tstride();
            const tstride_cache = ts / HW_PER_TILE * 256;
            const e0: usize = c.slots[0];
            const xt0 = try aa.alloc(f32, c.hidden);
            const a1 = try aa.alloc(f32, c.inter);
            const a2 = try aa.alloc(f32, c.inter);
            exl3.prepareInput(c.x[0..c.hidden], c.suhg[e0 * c.hidden ..][0..c.hidden], xt0);
            exl3.innerGemv(c.tg[e0 * ts ..][0..ts], xt0, c.hidden, c.inter, RATE, DEC, a1);
            innerGemvCached(pack_cache.tiles_g[e0 * tstride_cache ..][0..tstride_cache], xt0, c.hidden, c.inter, a2);
            for (a1, a2) |x, y| {
                if (exl3.f32ToF16Bits(x) != exl3.f32ToF16Bits(y)) return error.CacheMismatch;
            }
            have_pack = true;
        }
        const cache: *const PackCache = if (is_pack) &pack_cache else &block_cache;
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        try computeR1(aa, c, cache);
        const t1 = std.Io.Clock.awake.now(io).nanoseconds;
        if (c.gate_r2) {
            try computeR2(aa, c, cache);
        }
        const t2 = std.Io.Clock.awake.now(io).nanoseconds;
        std.debug.print("gen {s}: r1 {d:.1} ms, r2 {d:.1} ms\n", .{
            c.name,
            @as(f64, @floatFromInt(t1 - t0)) / 1e6,
            @as(f64, @floatFromInt(t2 - t1)) / 1e6,
        });
    }

    // Write.
    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    try put32(w, 0x4D334C58); // "XL3M"
    try put32(w, 1);
    try put32(w, @intCast(cases.items.len));
    for (cases.items) |*c| try writeCase(w, c);
    try w.flush();
    std.debug.print("wrote {s}: {d} cases\n", .{ out_path, cases.items.len });
}
