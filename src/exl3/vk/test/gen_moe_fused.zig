//! M3d oracle generator: fused-GEMV test vectors for exl3_gemv_fused.comp.
//!
//! Expected outputs are the project reference chain R1 (expert_exl3.zig):
//! prepareInput -> innerGemv -> finishOutput per projection, SwiGLU in f32 on
//! the finished f16 gate/up, score applied to the f32 finished per-slot down
//! output, f32 accumulate over slots in slot order, one final f16/bf16 round.
//! R2 (the dense plainMoeFused mirror) is intentionally not emitted: the fused
//! kernel implements the R1 chain exactly, and the M2 harness already printed
//! the R1/R2 informational table.
//!
//! Case families (hidden 2560, inter 640, topk 10, 32 experts unless stated):
//!   mini-512        — 512/256, topk 3, 8 experts, n=32: multi-block shapes
//!                     away from the pack dims, cheap indexing coverage.
//!   pack-r{1,2,4,17} at n=32 (2b), r2 also with limit 5 — pack shapes.
//!   pack3/4/6/8-r1  — n = 48/64/96/128 (3/4/6/8 bpw), r1; pack4-r2 at n=64.
//! Window is w15 mcg throughout, matching the existing harness.
//!
//! File layout (all little-endian):
//!   u32 magic "XL3F" (0x46334C58), u32 version 1, u32 caseCount
//!   per case: char name[16]; u32 rows, hidden, inter, experts, topk, n, mask;
//!             f32 limit; then arrays:
//!     x rows*hidden f32; slots rows*topk u32; scores rows*topk f32;
//!     tg,tu,td  E*tstride u16 (tstride = (hidden/16)*(inter/16)*n);
//!     suhg,suhu E*hidden u16; suhd E*inter u16;
//!     svhg,svhu E*inter u16; svhd E*hidden u16;
//!     ag,au rows*topk*inter u16;
//!     r1out32 rows*hidden f32; r1out16,r1outbf rows*hidden u16
const std = @import("std");
const exl3 = @import("exl3");

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
    rate: exl3.Rate,
    limit: f32,

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
    r1o32: []f32 = &.{},
    r1o16: []u16 = &.{},
    r1obf: []u16 = &.{},

    fn tstride(c: *const Case) usize {
        return (c.hidden / 16) * (c.inter / 16) * c.rate.n;
    }
};

// --- random helpers (identical distributions to gen_moe.zig) ----------------

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

// --- reference R1 chain -----------------------------------------------------

/// Decoded-tile cache per (projection, expert, tile), tensorCorePerm placement
/// — byte-for-byte what exl3.innerGemv derives from decodeTile.
const PackCache = struct {
    tiles_g: []u16,
    tiles_u: []u16,
    tiles_d: []u16,
};

fn buildTiles(gpa: std.mem.Allocator, c: *const Case, trellis: []const u16) ![]u16 {
    const rate = c.rate;
    const ts = c.tstride();
    var out = try gpa.alloc(u16, c.experts * ts / rate.n * 256);
    const tile_bits: []u16 = try gpa.alloc(u16, rate.n);
    defer gpa.free(tile_bits);
    var tile_w: [256]u16 = undefined;
    for (0..c.experts * ts / rate.n) |ti| {
        @memcpy(tile_bits, trellis[ti * rate.n ..][0..rate.n]);
        exl3.decodeTile(tile_bits, rate, DEC, &tile_w);
        @memcpy(out[ti * 256 ..][0..256], &tile_w);
    }
    return out;
}

/// exl3.innerGemv with the tile decode replaced by the cache. Identical
/// accumulation order and rounding.
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

fn moeProjectRef(c: *const Case, cache: *const PackCache, scratch: Scratch, acc: []f32) void {
    const ts = c.tstride();
    const tsc = ts / c.rate.n * 256;
    for (0..c.rows) |r| {
        const xr = c.x[r * c.hidden ..][0..c.hidden];
        const accr = acc[r * c.hidden ..][0..c.hidden];
        @memset(accr, 0);
        for (0..c.topk) |j| {
            const slot = r * c.topk + j;
            const e: usize = c.slots[slot];
            const sc = c.scores[slot];

            exl3.prepareInput(xr, c.suhg[e * c.hidden ..][0..c.hidden], scratch.xt_g);
            innerGemvCached(cache.tiles_g[e * tsc ..][0..tsc], scratch.xt_g, c.hidden, c.inter, scratch.agf);
            exl3.finishOutput(scratch.agf, c.svhg[e * c.inter ..][0..c.inter], scratch.ag2);
            exl3.prepareInput(xr, c.suhu[e * c.hidden ..][0..c.hidden], scratch.xt_u);
            innerGemvCached(cache.tiles_u[e * tsc ..][0..tsc], scratch.xt_u, c.hidden, c.inter, scratch.auf);
            exl3.finishOutput(scratch.auf, c.svhu[e * c.inter ..][0..c.inter], scratch.auf2);

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
            }

            exl3.prepareInput(scratch.h, c.suhd[e * c.inter ..][0..c.inter], scratch.xtdf);
            innerGemvCached(cache.tiles_d[e * tsc ..][0..tsc], scratch.xtdf, c.inter, c.hidden, scratch.y);
            exl3.finishOutput(scratch.y, c.svhd[e * c.hidden ..][0..c.hidden], scratch.y2);
            for (0..c.hidden) |o| {
                accr[o] += sc * scratch.y2[o];
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
    c.r1o32 = try gpa.alloc(f32, c.rows * c.hidden);
    c.r1o16 = try gpa.alloc(u16, c.rows * c.hidden);
    c.r1obf = try gpa.alloc(u16, c.rows * c.hidden);
}

fn genFullCase(gpa: std.mem.Allocator, rnd: std.Random, c: *Case) !void {
    try allocCase(gpa, c);
    const ts = c.tstride();
    var tile: [256]u16 = undefined;
    var max_w: f32 = 0;
    const tile_bits = try gpa.alloc(u16, c.rate.n);
    defer gpa.free(tile_bits);
    for (0..c.experts) |e| {
        for (0..3) |proj| {
            const t = switch (proj) {
                0 => &c.tg,
                1 => &c.tu,
                else => &c.td,
            };
            for (0..ts / c.rate.n) |ti| {
                for (tile_bits) |*x| x.* = rnd.int(u16);
                exl3.decodeTile(tile_bits, c.rate, DEC, &tile);
                for (tile) |bits| {
                    const v = @abs(exl3.f16BitsToF32(bits));
                    if (v > max_w) max_w = v;
                }
                @memcpy(t.*[e * ts + ti * c.rate.n ..][0..c.rate.n], tile_bits);
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
    fillRouting(c, rnd);
}

fn fillRouting(c: *Case, rnd: std.Random) void {
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

/// Row-variant clone: shares the weight tensors of `src`, regenerates only
/// x/slots/scores so the timing comparison across row counts stays
/// apples-to-apples.
fn cloneCase(gpa: std.mem.Allocator, rnd: std.Random, c: *Case, src: *const Case) !void {
    try allocCase(gpa, c);
    @memcpy(c.tg, src.tg);
    @memcpy(c.tu, src.tu);
    @memcpy(c.td, src.td);
    @memcpy(c.suhg, src.suhg);
    @memcpy(c.suhu, src.suhu);
    @memcpy(c.suhd, src.suhd);
    @memcpy(c.svhg, src.svhg);
    @memcpy(c.svhu, src.svhu);
    @memcpy(c.svhd, src.svhd);
    fillRouting(c, rnd);
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

fn writeCase(w: anytype, c: *const Case) !void {
    try putName(w, c.name);
    try put32(w, @intCast(c.rows));
    try put32(w, @intCast(c.hidden));
    try put32(w, @intCast(c.inter));
    try put32(w, @intCast(c.experts));
    try put32(w, @intCast(c.topk));
    try put32(w, c.rate.n);
    try put32(w, DEC.window.mask());
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
    for (c.r1o32) |v| try putF32(w, v);
    for (c.r1o16) |v| try put16(w, v);
    for (c.r1obf) |v| try put16(w, v);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const out_path = args.next() orelse return error.Usage;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    var prng = std.Random.DefaultPrng.init(0x5EEDF00D);
    const rnd = prng.random();

    // One weight pool per rate; row variants clone it (gen_moe.zig convention).
    const specs = [_]struct { name: []const u8, rows: usize, hid: usize, inter: usize, E: usize, k: usize, hw: usize, limit: f32, pool: usize }{
        .{ .name = "mini-512", .rows = 2, .hid = 512, .inter = 256, .E = 8, .k = 3, .hw = 32, .limit = 0, .pool = 0 },
        .{ .name = "pack-r1", .rows = 1, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 32, .limit = 0, .pool = 1 },
        .{ .name = "pack-r2-limit5", .rows = 2, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 32, .limit = 5.0, .pool = 1 },
        .{ .name = "pack-r2", .rows = 2, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 32, .limit = 0, .pool = 1 },
        .{ .name = "pack-r4", .rows = 4, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 32, .limit = 0, .pool = 1 },
        .{ .name = "pack-r17", .rows = 17, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 32, .limit = 0, .pool = 1 },
        .{ .name = "pack3-r1", .rows = 1, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 48, .limit = 0, .pool = 2 },
        .{ .name = "pack4-r1", .rows = 1, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 64, .limit = 0, .pool = 3 },
        .{ .name = "pack4-r2", .rows = 2, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 64, .limit = 0, .pool = 3 },
        .{ .name = "pack6-r1", .rows = 1, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 96, .limit = 0, .pool = 4 },
        .{ .name = "pack8-r1", .rows = 1, .hid = 2560, .inter = 640, .E = 32, .k = 10, .hw = 128, .limit = 0, .pool = 5 },
    };

    var cases: std.ArrayList(Case) = .empty;
    var pools: [6]?Case = .{ null, null, null, null, null, null };
    var pool_cache: [6]?PackCache = .{ null, null, null, null, null, null };
    for (specs) |sp| {
        var c: Case = .{
            .name = sp.name,
            .rows = sp.rows,
            .hidden = sp.hid,
            .inter = sp.inter,
            .experts = sp.E,
            .topk = sp.k,
            .rate = exl3.kFromPackedDim(sp.hw).?,
            .limit = sp.limit,
        };
        if (pools[sp.pool]) |*src| {
            try cloneCase(aa, rnd, &c, src);
        } else {
            try genFullCase(aa, rnd, &c);
            pools[sp.pool] = c;
            pool_cache[sp.pool] = .{
                .tiles_g = try buildTiles(aa, &c, c.tg),
                .tiles_u = try buildTiles(aa, &c, c.tu),
                .tiles_d = try buildTiles(aa, &c, c.td),
            };
            // Self-check: the tile cache must reproduce exl3.innerGemv
            // bit-exactly (both f32 accumulators), or R1 is not the reference.
            {
                const pc0 = &pool_cache[sp.pool].?;
                const ts = c.tstride();
                const tsc = ts / c.rate.n * 256;
                const e0: usize = c.slots[0];
                const xt0 = try aa.alloc(f32, c.hidden);
                const a1 = try aa.alloc(f32, c.inter);
                const a2 = try aa.alloc(f32, c.inter);
                exl3.prepareInput(c.x[0..c.hidden], c.suhg[e0 * c.hidden ..][0..c.hidden], xt0);
                exl3.innerGemv(c.tg[e0 * ts ..][0..ts], xt0, c.hidden, c.inter, c.rate, DEC, a1);
                innerGemvCached(pc0.tiles_g[e0 * tsc ..][0..tsc], xt0, c.hidden, c.inter, a2);
                for (a1, a2) |x, y| {
                    if (exl3.f32ToF16Bits(x) != exl3.f32ToF16Bits(y)) return error.CacheMismatch;
                }
            }
        }
        try cases.append(aa, c);
    }

    for (cases.items, 0..) |*c, ci| {
        const cache = &pool_cache[specs[ci].pool].?;
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        try computeR1(aa, c, cache);
        const t1 = std.Io.Clock.awake.now(io).nanoseconds;
        std.debug.print("gen {s}: r1 {d:.1} ms\n", .{ c.name, @as(f64, @floatFromInt(t1 - t0)) / 1e6 });
    }

    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    try put32(w, 0x46334C58); // "XL3F"
    try put32(w, 1);
    try put32(w, @intCast(cases.items.len));
    for (cases.items) |*c| try writeCase(w, c);
    try w.flush();
    std.debug.print("wrote {s}: {d} cases\n", .{ out_path, cases.items.len });
}
