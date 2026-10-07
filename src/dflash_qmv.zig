//! Reuse each A4 g64 FFN weight across the eight-row DFlash noise block.
const std = @import("std");
const mlx = @import("mlx.zig");
// Same subchunk and shuffle reduction as Apple's MLX qmv_wide_impl (MIT).
const SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint kl = lane % 8, r = threadgroup_position_in_grid.y * 8 + sg * 4 + lane / 8;
    \\uint v0 = threadgroup_position_in_grid.x * NV;
    \\uint row = min(r, uint(N - 1));
    \\device const uchar* wr = reinterpret_cast<device const uchar*>(w) + size_t(row) * (K / 2);
    \\float result[NV] = {0};
    \\for (uint g = kl; g < K / 64; g += 8) {
    \\    float scale = float(sc[size_t(row) * (K / 64) + g]);
    \\    float bias = float(bi[size_t(row) * (K / 64) + g]);
    \\    #pragma unroll
    \\    for (uint chunk = 0; chunk < 8; ++chunk) {
    \\        uint k0 = g * 64 + chunk * 8;
    \\        float dq[8];
    \\        #pragma unroll
    \\        for (uint i = 0; i < 4; ++i) {
    \\            uint packed = wr[(k0 / 2) + i];
    \\            dq[2*i] = scale * float(packed & 15) + bias;
    \\            dq[2*i+1] = (scale / 16.0f) * float(packed & 240) + bias;
    \\        }
    \\        #pragma unroll
    \\        for (uint v = 0; v < NV; ++v) {
    \\            uint input_row = min(v0 + v, uint(M - 1));
    \\            float acc = 0;
    \\            #pragma unroll
    \\            for (uint i = 0; i < 8; ++i) acc += float(x[size_t(input_row) * K + k0 + i]) * dq[i];
    \\            result[v] += acc;
    \\        }
    \\    }
    \\}
    \\for (uint v = 0; v < NV; ++v) {
    \\    result[v] += simd_shuffle_down(result[v], 4);
    \\    result[v] += simd_shuffle_down(result[v], 2);
    \\    result[v] += simd_shuffle_down(result[v], 1);
    \\}
    \\if (kl == 0 && r < N) {
    \\    for (uint v = 0; v < NV; ++v)
    \\        if (v0 + v < M) y[size_t(v0 + v) * N + r] = T(result[v]);
    \\}
;
var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
// Only the two admitted FFN geometries need configurations.
var configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);

pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsMetal(s) or !@import("transformer.zig").verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] != 8 or ws.len != 2 or ss.len != 2) return null;
    const m = xs[1];
    const k = xs[2];
    const n = ws[0];
    if (!((k == 4096 and n == 12288) or (k == 12288 and n == 4096)) or ws[1] != @divExact(k, 8) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (kernel_cache) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("sushi_dflash_a4_wide", iv, ov, SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernel_cache = value;
        break :blk value;
    };
    const slot: usize = if (k == 4096) 0 else 1;
    const config = blk: {
        if (configs[slot]) |hit| break :blk hit;
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, m, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @intCast(@divExact(n, 8) * 2), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        inline for (.{ "M", "N", "K", "NV" }, .{ m, n, k, 8 }) |name, number| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, name, number));
        configs[slot] = value;
        break :blk value;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ x, w, sc, bi }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

test "DFlash A4 eight-row FFN projection matches MLX exactly" {
    if (mlx.noGpuBackend() or !mlx.streamIsMetal(mlx.gpuStream()) or !@import("transformer.zig").verifySharedHardware()) return error.SkipZigTest;
    const Ops = @import("glm5_model.zig").Ops;
    const fixture = @import("dflash.zig").TinyFix;
    for ([_][2]c_int{ .{ 4096, 12288 }, .{ 12288, 4096 } }) |nk| {
        const k = nk[0];
        const n = nk[1];
        var inputs = Ops{ .s = mlx.gpuStream() };
        defer inputs.deinit();
        const x = try inputs.own(try fixture.bf16ArrShaped(&.{ 1, 8, k }, 781, inputs.s));
        const dense = try inputs.own(try fixture.bf16ArrShaped(&.{ n, k }, 783, inputs.s));
        const w = try inputs.slot();
        const sc = try inputs.slot();
        const bi = try inputs.slot();
        var quant = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(quant);
        try mlx.check(mlx.mlx_quantize(&quant, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, inputs.s));
        try mlx.check(mlx.mlx_vector_array_get(w, quant, 0));
        try mlx.check(mlx.mlx_vector_array_get(sc, quant, 1));
        try mlx.check(mlx.mlx_vector_array_get(bi, quant, 2));
        for ([_]mlx.mlx_array{ x, w.*, sc.*, bi.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const want = try inputs.slot();
        try mlx.check(mlx.mlx_quantized_matmul(want, x, w.*, sc.*, bi.*, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", inputs.s));
        try mlx.check(mlx.mlx_array_eval(want.*));
        const got = try inputs.own((try matmul(inputs.s, x, w.*, sc.*, bi.*)).?);
        try mlx.check(mlx.mlx_array_eval(got));
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want.*).?[0..@intCast(8 * n)], mlx.mlx_array_data_bfloat16(got).?[0..@intCast(8 * n)]);
        try std.testing.expect((try matmul(inputs.s, try inputs.slice(x, 1, 0, 3), w.*, sc.*, bi.*)) == null);
        try std.testing.expect((try matmul(inputs.s, try inputs.cast(x, .float32), w.*, sc.*, bi.*)) == null);
    }
}
