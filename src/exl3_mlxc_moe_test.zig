//! M3b acceptance: mlx_exl3_moe (patches/mlxc-exl3-moe.patch over
//! lib/mlxc-src) returns the right shape and dtype on the CPU stream, and
//! the values agree with the in-tree exl3 reference decode driven through
//! a double-precision host MoE loop.
//!
//! Run (after MLX_OMARCHY_DIR=<omarchy-mlx worktree>
//! ./scripts/build-mlx-linux.sh):
//!   zig test src/exl3_mlxc_moe_test.zig \
//!       -L lib/mlx/lib -lmlxc -rpath lib/mlx/lib
//!
//! Shapes are tiny but real EXL3 geometry: hidden 128, inter 256,
//! 4 experts, rate n = 32 (2 bpw), window 15, gate clamp limit 7.

const std = @import("std");
const mlx = @import("mlx.zig");
const exl3 = @import("exl3/expert_exl3.zig");

extern "c" fn mlx_array_data_float32(arr: mlx.mlx_array) ?[*]f32;
extern "c" fn mlx_array_dim(arr: mlx.mlx_array, dim: c_int) c_int;
extern "c" fn mlx_array_free(arr: mlx.mlx_array) c_int;

const hidden: usize = 128;
const inter: usize = 256;
const experts: usize = 4;
const rows: usize = 3;
const topk: usize = 2;
const rate_n: u32 = 32; // bits * 16
const window: c_int = 15;
const limit: f32 = 7.0;

fn f16bits(v: u16) f32 {
    return @floatCast(@as(f16, @bitCast(v)));
}

fn arrayNewData(data: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype) mlx.mlx_array {
    return mlx.mlx_array_new_data(
        @ptrCast(data.ptr),
        @ptrCast(shape.ptr),
        @intCast(shape.len),
        dtype,
    );
}

test "mlx_exl3_moe returns the right shape and dtype" {
    const alloc = std.testing.allocator;
    const tiles_gu: usize = (hidden / 128) * (inter / 128);
    const tiles_d: usize = (inter / 128) * (hidden / 128);
    const packed_n: usize = rate_n;

    var prng = std.Random.DefaultPrng.init(0xE8301C);
    const rand = prng.random();

    // Trellis banks [E, tiles * packed] uint16 (the sushi bank layout),
    // full-range codewords; scales as raw f16 bit patterns (normals
    // 0.5..~2, positive).
    const gbuf = try alloc.alloc(u16, experts * tiles_gu * packed_n);
    defer alloc.free(gbuf);
    const ubuf = try alloc.alloc(u16, experts * tiles_gu * packed_n);
    defer alloc.free(ubuf);
    const dbuf = try alloc.alloc(u16, experts * tiles_d * packed_n);
    defer alloc.free(dbuf);
    for (gbuf) |*v| v.* = rand.int(u16);
    for (ubuf) |*v| v.* = rand.int(u16);
    for (dbuf) |*v| v.* = rand.int(u16);

    var suh_g: [hidden]u16 = undefined;
    var svh_g: [inter]u16 = undefined;
    var suh_u: [hidden]u16 = undefined;
    var svh_u: [inter]u16 = undefined;
    var suh_d: [inter]u16 = undefined;
    var svh_d: [hidden]u16 = undefined;
    inline for (.{ &suh_g, &suh_u, &svh_d }) |s| {
        for (s) |*v| v.* = 0x3800 | (rand.int(u16) & 0x3FF);
    }
    inline for (.{ &svh_g, &svh_u, &suh_d }) |s| {
        for (s) |*v| v.* = 0x3800 | (rand.int(u16) & 0x3FF);
    }

    var x: [rows * hidden]f32 = undefined;
    for (&x) |*v| v.* = @floatCast(rand.float(f64) * 4.0 - 2.0);
    var slots: [rows * topk]u32 = undefined;
    var scores: [rows * topk]f32 = undefined;
    for (0..rows) |r| {
        var used: [experts]bool = @splat(false);
        for (0..topk) |k| {
            var e: u32 = rand.uintLessThan(u32, experts);
            while (used[e]) e = (e + 1) % experts;
            used[e] = true;
            slots[r * topk + k] = e;
            scores[r * topk + k] = 0.5;
        }
    }

    const x_bytes = std.mem.sliceAsBytes(&x);
    const g_bytes = std.mem.sliceAsBytes(gbuf);
    const u_bytes = std.mem.sliceAsBytes(ubuf);
    const d_bytes = std.mem.sliceAsBytes(dbuf);
    const suhg_bytes = std.mem.sliceAsBytes(&suh_g);
    const svhg_bytes = std.mem.sliceAsBytes(&svh_g);
    const suhu_bytes = std.mem.sliceAsBytes(&suh_u);
    const svhu_bytes = std.mem.sliceAsBytes(&svh_u);
    const suhd_bytes = std.mem.sliceAsBytes(&suh_d);
    const svhd_bytes = std.mem.sliceAsBytes(&svh_d);
    const slots_bytes = std.mem.sliceAsBytes(&slots);
    const scores_bytes = std.mem.sliceAsBytes(&scores);

    const shape_x = [_]c_int{ @intCast(rows), @intCast(hidden) };
    const shape_bank_gu = [_]c_int{ @intCast(experts), @intCast(tiles_gu * packed_n) };
    const shape_bank_d = [_]c_int{ @intCast(experts), @intCast(tiles_d * packed_n) };
    const shape_hidden = [_]c_int{@intCast(hidden)};
    const shape_inter = [_]c_int{@intCast(inter)};
    const shape_slots = [_]c_int{ @intCast(rows), @intCast(topk) };

    const xa = arrayNewData(x_bytes, &shape_x, .float32);
    defer _ = mlx_array_free(xa);
    const gta = arrayNewData(g_bytes, &shape_bank_gu, .uint16);
    defer _ = mlx_array_free(gta);
    const uta = arrayNewData(u_bytes, &shape_bank_gu, .uint16);
    defer _ = mlx_array_free(uta);
    const dta = arrayNewData(d_bytes, &shape_bank_d, .uint16);
    defer _ = mlx_array_free(dta);
    const gsuh = arrayNewData(suhg_bytes, &shape_hidden, .float16);
    defer _ = mlx_array_free(gsuh);
    const gsvh = arrayNewData(svhg_bytes, &shape_inter, .float16);
    defer _ = mlx_array_free(gsvh);
    const usuh = arrayNewData(suhu_bytes, &shape_hidden, .float16);
    defer _ = mlx_array_free(usuh);
    const usvh = arrayNewData(svhu_bytes, &shape_inter, .float16);
    defer _ = mlx_array_free(usvh);
    const dsuh = arrayNewData(suhd_bytes, &shape_inter, .float16);
    defer _ = mlx_array_free(dsuh);
    const dsvh = arrayNewData(svhd_bytes, &shape_hidden, .float16);
    defer _ = mlx_array_free(dsvh);
    const slots_a = arrayNewData(slots_bytes, &shape_slots, .uint32);
    defer _ = mlx_array_free(slots_a);
    const scores_a = arrayNewData(scores_bytes, &shape_slots, .float32);
    defer _ = mlx_array_free(scores_a);

    // Explicit CPU stream: M3b pins the C ABI composition, not a GPU path.
    const dev = mlx.mlx_device_new_type(.cpu, 0);
    const stream = mlx.mlx_stream_new_device(dev);

    var res = mlx.mlx_array{};
    const rc = mlx.mlx_exl3_moe(
        &res,
        xa,
        gta,
        gsuh,
        gsvh,
        uta,
        usuh,
        usvh,
        dta,
        dsuh,
        dsvh,
        slots_a,
        scores_a,
        @intCast(topk),
        window,
        limit,
        .float32,
        stream,
    );
    try std.testing.expectEqual(@as(c_int, 0), rc);
    defer _ = mlx_array_free(res);

    // Shape/dtype contract.
    try std.testing.expectEqual(@as(usize, 2), mlx.mlx_array_ndim(res));
    try std.testing.expectEqual(@as(c_int, @intCast(rows)), mlx_array_dim(res, 0));
    try std.testing.expectEqual(@as(c_int, @intCast(hidden)), mlx_array_dim(res, 1));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(res));

    _ = mlx.mlx_array_eval(res);
    const out = mlx_array_data_float32(res).?;

    // Double-precision host reference: decode every expert with the
    // in-tree reference decoder (reconstructPublic writes f16 bits) and
    // run the same clamped routed MoE in f64. The kernel-level gates live
    // in the omarchy-mlx doctest; this pins the C ABI composition end to
    // end.
    const dec = exl3.Decode{ .codebook = .mcg, .window = .w15 };
    const rate = exl3.Rate{ .n = rate_n };
    const w = try alloc.alloc(u16, hidden * inter);
    defer alloc.free(w);
    const wu = try alloc.alloc(u16, hidden * inter);
    defer alloc.free(wu);
    const wd = try alloc.alloc(u16, inter * hidden);
    defer alloc.free(wd);
    var acc: [rows * hidden]f64 = @splat(0);
    for (0..experts) |e| {
        try exl3.reconstructPublic(
            alloc,
            gbuf[e * tiles_gu * packed_n ..][0 .. tiles_gu * packed_n],
            &suh_g,
            &svh_g,
            hidden,
            inter,
            rate,
            dec,
            w,
        );
        try exl3.reconstructPublic(
            alloc,
            ubuf[e * tiles_gu * packed_n ..][0 .. tiles_gu * packed_n],
            &suh_u,
            &svh_u,
            hidden,
            inter,
            rate,
            dec,
            wu,
        );
        try exl3.reconstructPublic(
            alloc,
            dbuf[e * tiles_d * packed_n ..][0 .. tiles_d * packed_n],
            &suh_d,
            &svh_d,
            inter,
            hidden,
            rate,
            dec,
            wd,
        );
        for (0..rows) |r| {
            for (0..topk) |k| {
                if (slots[r * topk + k] != e) continue;
                var act: [inter]f64 = undefined;
                for (0..inter) |j| {
                    var h: f64 = 0;
                    var u: f64 = 0;
                    for (0..hidden) |t| {
                        const xv: f64 = x[r * hidden + t];
                        h += xv * @as(f64, f16bits(w[t * inter + j]));
                        u += xv * @as(f64, f16bits(wu[t * inter + j]));
                    }
                    if (h > limit) h = limit; // gate: upper clamp only
                    if (u > limit) u = limit; // up: symmetric clamp
                    if (u < -@as(f64, limit)) u = -@as(f64, limit);
                    const sig = 1.0 / (1.0 + @exp(-h));
                    act[j] = sig * u;
                }
                for (0..hidden) |o| {
                    var d: f64 = 0;
                    for (0..inter) |t| {
                        d += act[t] * @as(f64, f16bits(wd[t * hidden + o]));
                    }
                    acc[r * hidden + o] += @as(f64, scores[r * topk + k]) * d;
                }
            }
        }
    }

    var max_abs: f64 = 0;
    for (0..rows * hidden) |i| {
        const diff = @abs(@as(f64, out[i]) - acc[i]);
        if (diff > max_abs) max_abs = diff;
    }
    std.debug.print("mlx_exl3_moe vs f64 reference: max_abs = {d:.6}\n", .{max_abs});
    // The composed path evaluates in f32 (mlx sigmoid, f32 matmuls); the
    // reference is f64 with the same clamp order. Values here are O(10^2),
    // so the bound is loose but real: a wrong route, a misread scale dtype
    // or the wrong clamp side blows far past it.
    try std.testing.expect(max_abs < 1e-2);
}
