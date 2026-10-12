const std = @import("std");
const mlx = @import("mlx.zig");
const gdn = @import("gdn_decode.zig");

fn array(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

fn expectValues(a: mlx.mlx_array, expected: []const f32, s: mlx.mlx_stream) !void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    try std.testing.expectEqualSlices(f32, expected, p[0..expected.len]);
}

fn checkGroupedHeads(chunked: bool) !void {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const q = array(&.{ 1, 2 }, &.{ 1, 1, 2, 1 });
    const k = array(&.{ 1, 2 }, &.{ 1, 1, 2, 1 });
    const v = array(&.{ 1, 1, 1, 1 }, &.{ 1, 1, 4, 1 });
    const gate = array(&.{ 1, 1, 1, 1 }, &.{ 1, 1, 4 });
    const beta = array(&.{ 0.25, 0.25, 0.25, 0.25 }, &.{ 1, 1, 4 });
    const state = array(&.{ 0, 0, 0, 0 }, &.{ 1, 4, 1, 1 });
    defer inline for (.{ q, k, v, gate, beta, state }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const in = gdn.PlainIn{ .q = q, .k = k, .v = v, .g = gate, .beta = beta, .state = state };
    var out = if (chunked) try gdn.chunkedRecur(in, s) else try gdn.serialRecur(in, false, s);
    defer out.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ 1, 1, 4, 1 }, mlx.getShape(out.y));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 4, 1, 1 }, mlx.getShape(out.state));
    try expectValues(out.state, &.{ 0.25, 0.25, 0.5, 0.5 }, s);
    try expectValues(out.y, &.{ 0.25, 0.25, 1, 1 }, s);
}

test "GDN serial CPU uses contiguous grouped heads" {
    try checkGroupedHeads(false);
}

test "GDN chunk CPU preserves solve shape and padded tail" {
    try checkGroupedHeads(true);
}
