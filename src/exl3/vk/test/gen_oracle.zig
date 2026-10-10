//! M1 oracle generator: dumps binary test vectors for exl3_decode.comp.
//! Expected outputs come from the in-tree CPU reference (expert_exl3.zig).
//! Layout of oracle.bin: header (magic "EXL3", version, n, mask, blocks),
//! the 65536-entry decodeMcg table, then per block: trellis (64 tiles of
//! 32 halfwords), suh[128], svh[128] (f16 bits), expected dec16 (f16 bits),
//! tmp f32 (after row H128 + suh), out f32 (after col H128 + svh), final
//! f16 bits, final bf16 bits.
const std = @import("std");
const exl3 = @import("exl3");

const BLOCKS = 18; // 16 random, 1 edge scales, 1 zero trellis + edge scales
const DIM = 128;
const EL = DIM * DIM;
const TILES_PER_BLOCK = 64;
const HW_PER_TILE = 32; // K=2

const RATE: exl3.Rate = exl3.kFromPackedDim(HW_PER_TILE).?;
const DEC: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };

fn bf16RNE(b: u32) u16 {
    const lsb: u32 = (b >> 16) & 1;
    return @truncate((b +% 0x7FFF +% lsb) >> 16);
}

const Mirror = struct {
    trellis: [TILES_PER_BLOCK * HW_PER_TILE]u16,
    suh: [DIM]u16,
    svh: [DIM]u16,
    dec16: [EL]u16,
    tmp: [EL]f32,
    out: [EL]f32,
    w16: [EL]u16,
    wbf: [EL]u16,
};

/// Same math and op order as reconstructPublic (:287-315), exposing the f32
/// intermediates the shader is compared against.
fn mirrorBlock(m: *Mirror) void {
    var inner: [EL]u16 = undefined;
    exl3.reconstructInner(&m.trellis, DIM, DIM, RATE, DEC, &inner);
    m.dec16 = inner;

    var w: [EL]f32 = undefined;
    for (inner, 0..) |bits, i| w[i] = exl3.f16BitsToF32(bits);

    var vec: [DIM]f32 = undefined;
    for (0..DIM) |col| {
        for (0..DIM) |r| vec[r] = w[r * DIM + col];
        exl3.hadamard128(&vec);
        for (0..DIM) |r| w[r * DIM + col] = vec[r];
    }
    for (0..DIM) |r| {
        const s = exl3.f16BitsToF32(m.suh[r]);
        for (0..DIM) |c| w[r * DIM + c] *= s;
    }
    m.tmp = w;

    for (0..DIM) |r| {
        for (0..DIM) |c| vec[c] = w[r * DIM + c];
        exl3.hadamard128(&vec);
        for (0..DIM) |c| w[r * DIM + c] = vec[c];
    }
    for (0..DIM) |c| {
        const s = exl3.f16BitsToF32(m.svh[c]);
        for (0..DIM) |r| w[r * DIM + c] *= s;
    }
    for (w, 0..) |v, i| {
        m.out[i] = v;
        m.w16[i] = exl3.f32ToF16Bits(v);
        m.wbf[i] = bf16RNE(@bitCast(v));
    }
}

/// Random f16 bit patterns with exp in 1..30: finite normals and subnormals,
/// no inf/NaN (f16 inf/NaN scales are outside the pack domain).
fn randFiniteF16(rnd: std.Random) u16 {
    while (true) {
        const b = rnd.int(u16);
        if ((b & 0x7C00) != 0x7C00) return b;
    }
}

const EDGE_SCALES = [_]u16{
    0x0000, 0x8000, // +/-0
    0x0001, 0x8001, // min subnormal
    0x03FF, 0x83FF, // max subnormal
    0x0400, 0x8400, // min normal
    0x3C00, 0xBC00, // +/-1
    0x4000, 0xC000, // +/-2
    0x7BFF, 0xFBFF, // max finite
    0x3555, 0xB555, // arbitrary normal
};
fn put16(w: *std.Io.Writer, v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try w.writeAll(&b);
}

fn putF32(w: *std.Io.Writer, v: f32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, @bitCast(v), .little);
    try w.writeAll(&b);
}

fn put32(w: *std.Io.Writer, v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try w.writeAll(&b);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const out_path = args.next() orelse return error.Usage;

    var prng = std.Random.DefaultPrng.init(0xE0C4FEED);
    const rnd = prng.random();

    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;

    try put32(w, 0x45584C33); // "EXL3"
    try put32(w, 1); // version
    try put32(w, RATE.n);
    try put32(w, DEC.window.mask());
    try put32(w, BLOCKS);

    // decodeMcg over all 65536 codewords.
    for (0..65536) |cw| try put16(w, exl3.decodeMcg(@intCast(cw)));

    var m: Mirror = undefined;
    for (0..BLOCKS) |b| {
        const edge = (b == BLOCKS - 2 or b == BLOCKS - 1);
        for (&m.trellis) |*x| x.* = if (b == BLOCKS - 1) 0 else rnd.int(u16);
        if (edge) {
            for (&m.suh, 0..) |_, i| m.suh[i] = EDGE_SCALES[i % EDGE_SCALES.len];
            for (&m.svh, 0..) |_, i| m.svh[i] = EDGE_SCALES[(i + 5) % EDGE_SCALES.len];
        } else {
            for (&m.suh) |*x| x.* = randFiniteF16(rnd);
            for (&m.svh) |*x| x.* = randFiniteF16(rnd);
        }
        mirrorBlock(&m);

        // Self-check against the public reference entry point.
        var pub_out: [EL]u16 = undefined;
        try exl3.reconstructPublic(gpa, &m.trellis, &m.suh, &m.svh, DIM, DIM, RATE, DEC, &pub_out);
        for (m.w16, pub_out) |a, e| {
            if (a != e) return error.MirrorMismatch;
        }

        for (m.trellis) |x| try put16(w, x);
        for (m.suh) |x| try put16(w, x);
        for (m.svh) |x| try put16(w, x);
        for (m.dec16) |x| try put16(w, x);
        for (m.tmp) |x| try putF32(w, x);
        for (m.out) |x| try putF32(w, x);
        for (m.w16) |x| try put16(w, x);
        for (m.wbf) |x| try put16(w, x);
    }
    try w.flush();
    std.debug.print("wrote {s}: {d} blocks, n={d}, mask=0x{X:0>4}\n", .{ out_path, BLOCKS, RATE.n, DEC.window.mask() });
}
