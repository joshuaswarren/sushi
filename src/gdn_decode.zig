//! Fused GatedDeltaNet decode and verify step (B=1, per-head gate, bf16): the
//! prework (conv, silu, q/k norm, gate, beta) and the recurrence in ONE dispatch,
//! then the norm-gate epilogue, or folds it and rollback history into verify.
//! Each head runs over SPLIT threadgroups in the unfolded kernel; each
//! recomputes its head's prework (cheaper than a barrier between kernels) before
//! its slice of the recurrence rows. Outputs are the bf16 y, state and capture
//! buffers of `gdnPreworkFused -> gated_delta_step[_seq]`, bit for bit.
//! Port of mlx-serve `src/gdn_decode.zig` (K1 by David Dalcu; the verify-width
//! kernel and host plumbing from mlx-serve #517 by Samuel Reed); its Hadamard
//! norm-gate-rotate kernel is not ported. The verify fold adapts mlx-serve #558.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const HEADER = @import("transformer.zig").GDN_KERNEL_HEADER;

const K1_SOURCE =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;       // dv rows per threadgroup
    \\constexpr int R = RB / NSG;          // dv rows per simdgroup
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[DK], ks[DK], vs[DV];
    \\threadgroup float gb[2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  T act[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) {
    \\    uint ch = cb + lane * 4 + i;
    \\    float acc = 0.0f;
    \\    for (int tap = 0; tap < 3; ++tap) acc += float(conv_state[tap * C + ch]) * float(conv_w[ch * 4 + tap]);
    \\    acc += float(qkv[ch]) * float(conv_w[ch * 4 + 3]);
    \\    const T conv = T(acc);
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\    act[i] = conv * sig;
    \\    float v = float(act[i]);
    \\    sumsq += v * v;
    \\  }
    \\  if (sg < 2) {
    \\    sumsq = simd_sum(sumsq);
    \\    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\    const T scale = sg == 0 ? q_scale : k_scale;
    \\    threadgroup float* dst = sg == 0 ? qs : ks;
    \\    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\  } else {
    \\    for (int i = 0; i < 4; ++i) vs[lane * 4 + i] = float(act[i]);
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      conv_out[ch] = conv_state[C + ch];
    \\      conv_out[C + ch] = conv_state[2 * C + ch];
    \\      conv_out[2 * C + ch] = qkv[ch];
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  const T bv = b_in[hv];
    \\  T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\  gb[1] = float(bsig);
    \\  const T apd = T(float(a_in[hv]) + float(dt_bias[hv]));
    \\  float sp = sushi_log1p(metal::precise::exp(float(apd)));
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  gb[0] = float(T(metal::precise::exp(-(ea * sp))));
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float kk[4], qq[4];
    \\for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    \\const float g = gb[0], beta = gb[1];
    \\for (int j = 0; j < R; ++j) {
    \\  uint dv = row0 + j;
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\  kv_mem = simd_sum(kv_mem);
    \\  float delta = (vs[dv] - kv_mem) * beta;
    \\  float out = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\  out = simd_sum(out);
    \\  uint base = (hv * DV + dv) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\  if (lane == 0) y[hv * DV + dv] = static_cast<T>(out);
    \\}
;

// K1 over TL tokens (verify widths). Each captured step stores the state and
// carries the stored value on, as gated_delta_step_seq does, so state_seq[t]
// is the state serial decode holds after token t. state_seq[TL-1] is never
// written (the capture-tail trim).
const K1S_HEAD =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;
    \\constexpr int R = RB / NSG;
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[TL][DK], ks[TL][DK], vs[TL][DV];
    \\threadgroup float gb[TL][2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  for (int t = 0; t < TL; ++t) {
    \\    T act[4];
    \\    float sumsq = 0.0f;
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      float acc = 0.0f;
    \\      for (int tap = 0; tap < 4; ++tap) {
    \\        int w = t + tap;
    \\        const T xv = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\        acc += float(xv) * float(conv_w[ch * 4 + tap]);
    \\      }
    \\      const T conv = T(acc);
    \\      T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\      act[i] = conv * sig;
    \\      float v = float(act[i]);
    \\      sumsq += v * v;
    \\    }
    \\    if (sg < 2) {
    \\      sumsq = simd_sum(sumsq);
    \\      float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\      const T scale = sg == 0 ? q_scale : k_scale;
    \\      threadgroup float* dst = sg == 0 ? qs[t] : ks[t];
    \\      for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\    } else {
    \\      for (int i = 0; i < 4; ++i) vs[t][lane * 4 + i] = float(act[i]);
    \\    }
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      for (int j = 0; j < 3; ++j) {
    \\        int w = TL + j;
    \\        conv_out[j * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\      }
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  for (int t = 0; t < TL; ++t) {
    \\    const T bv = b_in[t * HV + hv];
    \\    T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\    gb[t][1] = float(bsig);
    \\    const T apd = T(float(a_in[t * HV + hv]) + float(dt_bias[hv]));
    \\    float sp = sushi_log1p(metal::precise::exp(float(apd)));
    \\    gb[t][0] = float(T(metal::precise::exp(-(ea * sp))));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int t = 0; t < TL; ++t) {
    \\  float kk[4], qq[4];
    \\  for (int i = 0; i < 4; ++i) { kk[i] = ks[t][lane * 4 + i]; qq[i] = qs[t][lane * 4 + i]; }
    \\  const float g = gb[t][0], beta = gb[t][1];
    \\  for (int j = 0; j < R; ++j) {
    \\    uint dv = row0 + j;
    \\    float kv_mem = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\    kv_mem = simd_sum(kv_mem);
    \\    float delta = (vs[t][dv] - kv_mem) * beta;
    \\    float out = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\    out = simd_sum(out);
;
const K1S_TAIL =
    \\    if (t + 1 < TL) {
    \\      uint sbase = t * (HV * DV * DK) + (hv * DV + dv) * DK + lane * 4;
    \\      for (int i = 0; i < 4; ++i) {
    \\        state_seq[sbase + i] = static_cast<StT>(st[j][i]);
    \\        st[j][i] = static_cast<float>(state_seq[sbase + i]);
    \\      }
    \\    }
    \\  }
    \\}
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\}
;

const K1S_SOURCE = K1S_HEAD ++ "\n" ++
    \\    if (lane == 0) y[(t * HV + hv) * DV + dv] = static_cast<T>(out);
++ "\n" ++ K1S_TAIL;

// One threadgroup owns a head. Its stored-precision y stays in shared memory
// for the norm-gate reduction; rollback also receives the convolution inputs.
const K1S_FOLD_SOURCE = "threadgroup float ys[TL][DV];\n" ++ K1S_HEAD ++ "\n" ++
    \\    if (lane == 0) ys[t][dv] = float(static_cast<T>(out));
++ "\n" ++ K1S_TAIL ++ "\n" ++
    \\if (sg < 3 && (sg == 2 || hv % GRP == 0)) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  for (int i = 0; i < 4; ++i) {
    \\    uint ch = cb + lane * 4 + i;
    \\    for (int w = 0; w < 3 + TL; ++w) conv_in[w * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg < TL) {
    \\  float xs[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { xs[i] = ys[sg][lane * 4 + i]; sumsq += xs[i] * xs[i]; }
    \\  sumsq = simd_sum(sumsq);
    \\  float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    \\  uint base = (sg * HV + hv) * DV + lane * 4;
    \\  for (int i = 0; i < 4; ++i) {
    \\    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    \\    const T zv = z[base + i];
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); T sig = zv < T(0) ? sy : T(1) - sy;
    \\    gated[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
    \\  }
    \\}
;

const SPLIT: c_int = 4;
const NT: c_int = 256; // 4 dv rows per simdgroup
pub const MAX_SEQ: c_int = 8;

pub const Geometry = struct { hk: c_int, hv: c_int, dk: c_int, dv: c_int };

/// The unprojected GDN inputs the chain's prework reads, for S tokens.
pub const Inputs = struct {
    qkv: mlx.mlx_array, // [1,S,C]
    a: mlx.mlx_array, // [1,S,Hv]
    b: mlx.mlx_array, // [1,S,Hv]
    conv_state: mlx.mlx_array, // [1,3,C]
    ssm_state: mlx.mlx_array, // [1,Hv,Dv,Dk]
    conv_w: mlx.mlx_array, // [C,4,1]
    A_log: mlx.mlx_array, // [Hv]
    dt_bias: mlx.mlx_array, // [Hv]
    q_scale: mlx.mlx_array, // 0-dim bf16
    k_scale: mlx.mlx_array, // 0-dim bf16
};

/// y [1,S,Hv,Dv], the next conv state [1,3,C] and the final state [1,Hv,Dv,Dk];
/// `state_seq` [S,1,Hv,Dv,Dk] (row S-1 unwritten) at verify widths only.
pub const Recur = struct {
    y: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    state_seq: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: Recur) void {
        inline for (.{ self.y, self.conv_state, self.ssm_state, self.state_seq }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }
};

var k1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var k1s_kernel: ?mlx.mlx_fast_metal_kernel = null;
var cfg_key: ?Geometry = null;
/// Index = token count; 1 is the decode kernel's, 2..MAX_SEQ the verify kernel's.
var cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);

fn makeKernel(name: [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn buildConfig(g: Geometry, t_len: c_int) !mlx.mlx_fast_metal_kernel_config {
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t_len, g.hv, g.dv }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, .bfloat16));
    if (t_len > 1) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, 5, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * SPLIT * NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", .bfloat16));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", NT }, .{ "SPLIT", SPLIT } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    if (t_len > 1) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TL", t_len));
    return cfg;
}

fn shapeIs(a: mlx.mlx_array, want: []const c_int) bool {
    return std.mem.eql(c_int, mlx.getShape(a), want);
}

fn inputsFit(g: Geometry, t_len: c_int, in: Inputs) bool {
    if (t_len < 1 or t_len > MAX_SEQ) return false;
    if (g.dk != 128 or g.dv != 128 or g.hk < 1 or @rem(g.hv, g.hk) != 0) return false;
    for ([_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.dt_bias, in.q_scale, in.k_scale }) |arr|
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return false;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    if (!shapeIs(in.qkv, &.{ 1, t_len, c }) or !shapeIs(in.a, &.{ 1, t_len, g.hv }) or !shapeIs(in.b, &.{ 1, t_len, g.hv }) or
        !shapeIs(in.conv_state, &.{ 1, 3, c }) or !shapeIs(in.ssm_state, &.{ 1, g.hv, g.dv, g.dk }) or
        mlx.mlx_array_size(in.conv_w) != @as(usize, @intCast(c * 4)) or mlx.getShape(in.conv_w)[0] != c) return false;

    return mlx.mlx_array_size(in.A_log) == @as(usize, @intCast(g.hv)) and
        mlx.mlx_array_size(in.dt_bias) == @as(usize, @intCast(g.hv)) and
        mlx.mlx_array_size(in.q_scale) == 1 and mlx.mlx_array_size(in.k_scale) == 1;
}

/// Prework + recurrence over `t_len` tokens (1 = decode, 2..MAX_SEQ = verify
/// with per-step state capture). Null outside the kernels' geometry and outside
/// bf16, the one width the chain's prework, recurrence and norm-gate all serve.
pub fn step(g: Geometry, t_len: c_int, in: Inputs, s: mlx.mlx_stream) !?Recur {
    if (!mlx.streamIsMetal(s) or !inputsFit(g, t_len, in)) return null;

    const seq = t_len > 1;
    const kernel = if (seq)
        k1s_kernel orelse blk: {
            k1s_kernel = try makeKernel("sushi_gdn_decode_recur_seq", &.{ "y", "conv_out", "state_out", "state_seq" }, K1S_SOURCE);
            break :blk k1s_kernel.?;
        }
    else
        k1_kernel orelse blk: {
            k1_kernel = try makeKernel("sushi_gdn_decode_recur", &.{ "y", "conv_out", "state_out" }, K1_SOURCE);
            break :blk k1_kernel.?;
        };
    if (cfg_key == null or !std.meta.eql(cfg_key.?, g)) {
        for (&cfgs) |*slot| if (slot.*) |cf| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cf);
            slot.* = null;
        };
        cfg_key = g;
    }
    const idx: usize = @intCast(t_len);
    if (cfgs[idx] == null) cfgs[idx] = try buildConfig(g, t_len);

    const ins = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfgs[idx].?, s));
    var out = Recur{ .y = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new() };
    if (seq) out.state_seq = mlx.mlx_array_new();
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.y, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, o, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.ssm_state, o, 2));
    if (seq) try mlx.check(mlx.mlx_vector_array_get(&out.state_seq, o, 3));
    return out;
}


pub const Fold = struct {
    gated: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    state_seq: mlx.mlx_array,
    conv_input: mlx.mlx_array,

    pub fn deinit(self: Fold) void {
        inline for (.{ self.gated, self.conv_state, self.ssm_state, self.state_seq, self.conv_input }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }
};


const FOLD_NT: c_int = 1024;
pub var fold_nt_override: ?c_int = null;
var fold_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fold_cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var fold_ok: [MAX_SEQ + 1]?bool = @splat(null);
const FoldKey = struct { g: Geometry, swish: bool, nt: c_int };
var fold_key: ?FoldKey = null;

pub fn foldDeclined(t_len: c_int) bool {
    return t_len >= 2 and t_len <= MAX_SEQ and fold_ok[@intCast(t_len)] == false;
}

fn buildFoldConfig(key: FoldKey, t_len: c_int) !mlx.mlx_fast_metal_kernel_config {
    const g = key.g;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    inline for (.{ &[_]c_int{ 1, t_len, g.hv * g.dv }, &[_]c_int{ 1, 3, c }, &[_]c_int{ 1, g.hv, g.dv, g.dk }, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, &[_]c_int{ 1, 3 + t_len, c } }) |shape|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, shape.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * key.nt, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, key.nt, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", .bfloat16));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", key.nt }, .{ "SPLIT", @as(c_int, 1) }, .{ "TL", t_len }, .{ "SWISH", @as(c_int, @intFromBool(key.swish)) } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    return cfg;
}

/// Verify recurrence, norm-gate and rollback convolution history in one dispatch.
/// The recurrence retains the stored bf16 state between tokens, as serial decode does.
pub fn stepFold(g: Geometry, t_len: c_int, in: Inputs, z: mlx.mlx_array, norm_w: mlx.mlx_array, eps: mlx.mlx_array, swish: bool, s: mlx.mlx_stream) !?Fold {
    if (!mlx.streamIsMetal(s) or t_len < 2 or !inputsFit(g, t_len, in)) return null;
    if (mlx.mlx_array_dtype(z) != .bfloat16 or mlx.mlx_array_dtype(norm_w) != .bfloat16 or mlx.mlx_array_dtype(eps) != .float32) return null;
    if (!shapeIs(z, &.{ 1, t_len, g.hv * g.dv }) or !shapeIs(norm_w, &.{g.dv}) or mlx.mlx_array_size(eps) != 1) return null;
    if (fold_kernel == null) {
        const ins = [_][*:0]const u8{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale", "z", "norm_w", "eps" };
        const outs = [_][*:0]const u8{ "gated", "conv_out", "state_out", "state_seq", "conv_in" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_gdn_verify_fold", iv, ov, K1S_FOLD_SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        fold_kernel = k;
    }
    const key = FoldKey{ .g = g, .swish = swish, .nt = fold_nt_override orelse FOLD_NT };
    if (fold_key == null or !std.meta.eql(fold_key.?, key)) {
        for (&fold_cfgs) |*slot| if (slot.*) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            slot.* = null;
        };
        fold_key = key;
        fold_ok = @splat(null);
    }
    const idx: usize = @intCast(t_len);
    if (fold_ok[idx] == false) return null;
    if (fold_cfgs[idx] == null) fold_cfgs[idx] = try buildFoldConfig(key, t_len);
    const ins = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale, z, norm_w, eps };
    // Probe with independent zeros: evaluating real inputs here could read a PLE
    // leaf before the caller fills it, or synchronize a still-lazy draft chain.
    if (fold_ok[idx] == null) {
        if (mlx.errorPending()) return error.MlxError;
        var dummy: [ins.len]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (dummy) |a| { if (a.ctx != null) _ = mlx.mlx_array_free(a); };
        for (ins, 0..) |a, i| {
            dummy[i] = mlx.mlx_array_new();
            const shape = mlx.getShape(a);
            try mlx.check(mlx.mlx_zeros(&dummy[i], shape.ptr, shape.len, mlx.mlx_array_dtype(a), s));
        }
        _ = mlx.mlx_array_free(dummy[12]);
        dummy[12] = mlx.mlx_array_new_float(1e-6);
        const pv = mlx.mlx_vector_array_new_data(&dummy, dummy.len);
        defer _ = mlx.mlx_vector_array_free(pv);
        var po = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(po);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&po, fold_kernel.?, pv, fold_cfgs[idx].?, s));
        const rc = mlx.mlx_eval(po);
        if (mlx.takeErrorIf("maximum allowed threads per threadgroup")) {
            fold_ok[idx] = false;
            log.info("[gdn-fold] declined at S={d}: pipeline threadgroup limit below {d}\n", .{ t_len, key.nt });
            return null;
        }
        try mlx.check(rc);
        if (mlx.errorPending()) return error.MlxError;
        fold_ok[idx] = true;
    }
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, fold_kernel.?, iv, fold_cfgs[idx].?, s));
    var out = Fold{ .gated = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .state_seq = mlx.mlx_array_new(), .conv_input = mlx.mlx_array_new() };
    errdefer out.deinit();
    inline for (.{ &out.gated, &out.conv_state, &out.ssm_state, &out.state_seq, &out.conv_input }, 0..) |a, i|
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    return out;
}

// ── Plain-MLX recurrence (no Metal) ──────────────────────────────────────
//
// The MSL kernels above are the fast lane; these composed-op ports keep GDN
// correct on streams that cannot run MSL (omarchy Vulkan). They take the
// post-prework tensors exactly as the fused chain hands them to the kernel:
// q/k/v/g/beta plus the recurrent state. Slow is fine — correctness port.

/// Post-prework recurrence inputs, all living on `s`.
pub const PlainIn = struct {
    q: mlx.mlx_array, // [B,S,Hk,Dk] normed + scaled
    k: mlx.mlx_array, // [B,S,Hk,Dk] normed + scaled
    v: mlx.mlx_array, // [B,S,Hv,Dv] activated
    g: mlx.mlx_array, // [B,S,Hv] decay factors in (0,1]
    beta: mlx.mlx_array, // [B,S,Hv] sigmoid(b)
    state: mlx.mlx_array, // [B,Hv,Dv,Dk]
};

/// y [B,S,Hv,Dv] bf16, the next state [B,Hv,Dv,Dk] bf16 and — at verify
/// widths — state_seq [S,B,Hv,Dv,Dk] bf16, row t = the state after token t.
pub const PlainOut = struct {
    y: mlx.mlx_array,
    state: mlx.mlx_array,
    state_seq: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: PlainOut) void {
        inline for (.{ self.y, self.state, self.state_seq }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }
};

fn release(a: *mlx.mlx_array) void {
    if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
        a.* = .{ .ctx = null };
    }
}

fn astypeF32(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_astype(&out, x, .float32, s));
    return out;
}

fn astypeBf16(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_astype(&out, x, .bfloat16, s));
    return out;
}

fn mul(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_multiply(&out, a, b, s));
    return out;
}

fn add(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

fn sub(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_subtract(&out, a, b, s));
    return out;
}

fn matmul(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_matmul(&out, a, b, s));
    return out;
}

fn expm(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_exp(&out, x, s));
    return out;
}

fn logm(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_log(&out, x, s));
    return out;
}

fn concatParts(parts: mlx.mlx_vector_array, axis: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, axis, s));
    return out;
}

fn transposeAxes(x: mlx.mlx_array, axes: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

fn reshapeTo(x: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, @intCast(shape.len), s));
    return out;
}

/// `x[..., from:to, ...]` on one axis (≤ rank 4).
fn sliceAxis(x: mlx.mlx_array, axis: c_int, from: c_int, to: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(0);
    const strides: [4]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[@intCast(axis)] = from;
    stop[@intCast(axis)] = to;
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_slice(&out, x, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    return out;
}

fn setCopy(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_array_set(&out, x));
    return out;
}

fn maximumScalar(x: mlx.mlx_array, v: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_maximum(&out, x, c, s));
    return out;
}

fn cumsumAxis2(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_cumsum(&out, x, 2, false, true, s));
    return out;
}

/// Cast to f32 and move time under heads: [B,S,H,D] → [B,H,S,D].
fn headsMajor4(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var f = try astypeF32(x, s);
    defer release(&f);
    return transposeAxes(f, &.{ 0, 2, 1, 3 }, s);
}

/// [B,S,H] → [B,H,S] f32.
fn headsMajor3(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var f = try astypeF32(x, s);
    defer release(&f);
    return transposeAxes(f, &.{ 0, 2, 1 }, s);
}

/// GQA broadcast: [B,Hk,T,D] → [B,Hv,T,D] when Hv > Hk.
fn tileHeads(x: mlx.mlx_array, grp: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (grp == 1) return setCopy(x, s);
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    try mlx.check(mlx.mlx_tile(&out, x, &.{ 1, grp, 1, 1 }, 4, s));
    return out;
}

fn rowsLike(x: mlx.mlx_array, rows: c_int, comptime ones: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var shape: [4]c_int = @splat(1);
    for (sh, 0..) |d, i| shape[i] = d;
    shape[1] = rows;
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    if (ones) {
        try mlx.check(mlx.mlx_ones(&out, &shape, sh.len, mlx.mlx_array_dtype(x), s));
    } else {
        try mlx.check(mlx.mlx_zeros(&out, &shape, sh.len, mlx.mlx_array_dtype(x), s));
    }
    return out;
}

/// Concatenate `rows` filler rows onto axis 1 of a [B,S,...] array.
fn padTime(x: mlx.mlx_array, rows: c_int, comptime ones: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    var filler = try rowsLike(x, rows, ones, s);
    defer release(&filler);
    var out = mlx.mlx_array_new();
    errdefer release(&out);
    const arrs = [_]mlx.mlx_array{ x, filler };
    const vec = mlx.mlx_vector_array_new_data(&arrs, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

/// Causal masks over one CHUNK-sized tile: exponent-domain pushdowns (0
/// keep / MASK_BIG discard) and the identity for the triangular inverse.
const CHUNK: usize = 64;
const MASK_BIG: f32 = 1e4;
const Masks = struct { strict_big: mlx.mlx_array, incl_big: mlx.mlx_array, ident: mlx.mlx_array };
var masks: ?Masks = null;

fn getMasks() Masks {
    if (masks) |m| return m;
    var strict: [CHUNK * CHUNK]f32 = undefined;
    var incl: [CHUNK * CHUNK]f32 = undefined;
    var ident: [CHUNK * CHUNK]f32 = undefined;
    for (0..CHUNK) |i| {
        for (0..CHUNK) |j| {
            strict[i * CHUNK + j] = if (j < i) 0 else MASK_BIG;
            incl[i * CHUNK + j] = if (j > i) MASK_BIG else 0;
            ident[i * CHUNK + j] = if (i == j) 1 else 0;
        }
    }
    const shape = [_]c_int{ @intCast(CHUNK), @intCast(CHUNK) };
    masks = .{
        .strict_big = mlx.mlx_array_new_data(&strict, &shape, 2, .float32),
        .incl_big = mlx.mlx_array_new_data(&incl, &shape, 2, .float32),
        .ident = mlx.mlx_array_new_data(&ident, &shape, 2, .float32),
    };
    return masks.?;
}

/// Per-token plain recurrence for decode and verify widths (S ≤ MAX_SEQ):
/// f32 math with the state stored to bf16 (and reloaded) after every token,
/// so `state_seq[t]` is exactly the state serial decode holds after token t.
/// Any batch/head geometry.
pub fn serialRecur(in: PlainIn, want_seq: bool, s: mlx.mlx_stream) !PlainOut {
    const qsh = mlx.getShape(in.q);
    if (qsh.len != 4 or qsh[1] < 1 or qsh[1] > MAX_SEQ) return error.GdnPlainWidth;
    const b = qsh[0];
    const t = qsh[1];
    const hk = qsh[2];
    const dk = qsh[3];
    const vsh = mlx.getShape(in.v);
    if (vsh.len != 4 or vsh[0] != b or vsh[1] != t) return error.GdnPlainShape;
    const hv = vsh[2];
    const dv = vsh[3];
    const gsh = mlx.getShape(in.g);
    if (gsh.len != 3 or gsh[0] != b or gsh[1] != t or gsh[2] != hv) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.k), &.{ b, t, hk, dk })) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.beta), &.{ b, t, hv })) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.state), &.{ b, hv, dv, dk })) return error.GdnPlainShape;
    const grp = @divExact(hv, hk);

    var qh = try headsMajor4(in.q, s); defer release(&qh); // [B,Hk,T,Dk]
    var kh = try headsMajor4(in.k, s); defer release(&kh);
    var vh = try headsMajor4(in.v, s); defer release(&vh); // [B,Hv,T,Dv]
    var gh = try headsMajor3(in.g, s); defer release(&gh); // [B,Hv,T]
    var bh = try headsMajor3(in.beta, s); defer release(&bh);
    var qg = try tileHeads(qh, grp, s); defer release(&qg); // [B,Hv,T,Dk]
    var kg = try tileHeads(kh, grp, s); defer release(&kg);

    var st = try astypeF32(in.state, s); // [B,Hv,Dv,Dk] f32, loop-carried
    errdefer release(&st);
    const y_parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(y_parts);
    const seq_parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(seq_parts);

    for (0..@intCast(t)) |ti| {
        const tix: c_int = @intCast(ti);
        var q_t = try sliceAxis(qg, 2, tix, tix + 1, s); defer release(&q_t); // [B,Hv,1,Dk]
        var k_t = try sliceAxis(kg, 2, tix, tix + 1, s); defer release(&k_t);
        var v_t = try sliceAxis(vh, 2, tix, tix + 1, s); defer release(&v_t); // [B,Hv,1,Dv]
        var g_t = try sliceAxis(gh, 2, tix, tix + 1, s); defer release(&g_t); // [B,Hv,1]
        var b_t = try sliceAxis(bh, 2, tix, tix + 1, s); defer release(&b_t);
        var g4 = try reshapeTo(g_t, &.{ b, hv, 1, 1 }, s); defer release(&g4);
        var b4 = try reshapeTo(b_t, &.{ b, hv, 1, 1 }, s); defer release(&b4);
        var v4 = try transposeAxes(v_t, &.{ 0, 1, 3, 2 }, s); defer release(&v4); // [B,Hv,Dv,1]

        // st ← g·st; kv = st·k; delta = beta·(v − kv); st ← st + k⊗delta; y = st·q
        var decayed = try mul(st, g4, s); defer release(&decayed);
        var k_t3 = try transposeAxes(k_t, &.{ 0, 1, 3, 2 }, s); defer release(&k_t3); // [B,Hv,Dk,1]
        var kv = try matmul(decayed, k_t3, s); defer release(&kv); // [B,Hv,Dv,1]
        var err = try sub(v4, kv, s); defer release(&err);
        var delta = try mul(err, b4, s); defer release(&delta);
        var outer = try mul(k_t, delta, s); defer release(&outer); // [B,Hv,Dv,Dk]
        const next = try add(decayed, outer, s);
        var q_t3 = try transposeAxes(q_t, &.{ 0, 1, 3, 2 }, s); defer release(&q_t3);
        var y4 = try matmul(next, q_t3, s); defer release(&y4); // [B,Hv,Dv,1]
        var yrow = try reshapeTo(y4, &.{ b, 1, hv, dv }, s); defer release(&yrow);

        // Store bf16, reload — the kernel's per-token rounding contract.
        var st_b = try astypeBf16(next, s); defer release(&st_b);
        const reloaded = try astypeF32(st_b, s);
        release(&st);
        st = reloaded;
        try mlx.check(mlx.mlx_vector_array_append_value(y_parts, yrow));
        if (want_seq) try mlx.check(mlx.mlx_vector_array_append_value(seq_parts, st_b));
    }

    const y = blk: {
        var y_cat = try concatParts(y_parts, 1, s); // [B,T,Hv,Dv]
        defer release(&y_cat);
        break :blk try astypeBf16(y_cat, s);
    };
    var out = PlainOut{ .y = y, .state = .{ .ctx = null } };
    errdefer out.deinit();
    out.state = try astypeBf16(st, s);
    release(&st);
    if (want_seq) {
        var seq_cat = try concatParts(seq_parts, 0, s); // [T,B,Hv,Dv,Dk]
        defer release(&seq_cat);
        out.state_seq = try astypeBf16(seq_cat, s);
    }
    return out;
}

/// Chunked plain recurrence for prefill widths (S > MAX_SEQ): the FLA
/// chunked gated-delta-rule formulation. Per chunk of 64 tokens, the
/// within-chunk delta terms solve as one lower-triangular system
/// (I − A)·U = β·v − β·e^L·(k·S_enter) with
/// A[i,j] = β_i·e^(L_i−L_j)·(k_i·k_j), L the in-chunk cumulative log decay;
/// y = e^L·(q·S_enter) + Σ_j≤i e^(L_i−L_j)(q_i·k_j)·U_j; the chunk exit
/// contribution e^(L_end−L_i)·k_i⊗U_i decays the carried state. All decay
/// exponents are ≤ 0, so nothing overflows and full forgetting underflows
/// to exact zeros. The chunk scan is sequential (one eval per chunk bounds
/// the lazy graph); a doubling scan or a Vulkan GDN kernel is the later
/// speed lane.
pub fn chunkedRecur(in: PlainIn, s: mlx.mlx_stream) !PlainOut {
    const qsh = mlx.getShape(in.q);
    if (qsh.len != 4 or qsh[1] < 1) return error.GdnPlainShape;
    const b = qsh[0];
    const t = qsh[1];
    const hk = qsh[2];
    const dk = qsh[3];
    const vsh = mlx.getShape(in.v);
    if (vsh.len != 4 or vsh[0] != b or vsh[1] != t) return error.GdnPlainShape;
    const hv = vsh[2];
    const dv = vsh[3];
    const gsh = mlx.getShape(in.g);
    if (gsh.len != 3 or gsh[0] != b or gsh[1] != t or gsh[2] != hv) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.k), &.{ b, t, hk, dk })) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.beta), &.{ b, t, hv })) return error.GdnPlainShape;
    if (!std.mem.eql(c_int, mlx.getShape(in.state), &.{ b, hv, dv, dk })) return error.GdnPlainShape;
    const grp = @divExact(hv, hk);
    const cchunk: c_int = @intCast(CHUNK);
    const nc = @divTrunc(t + cchunk - 1, cchunk);
    const tp = nc * cchunk;
    const msk = getMasks();

    // Pad the tail chunk: zero k/q/v/beta rows, decay-one g rows.
    const pad = tp - t;
    var qp = if (pad == 0) try setCopy(in.q, s) else try padTime(in.q, pad, false, s);
    defer release(&qp);
    var kp = if (pad == 0) try setCopy(in.k, s) else try padTime(in.k, pad, false, s);
    defer release(&kp);
    var vp = if (pad == 0) try setCopy(in.v, s) else try padTime(in.v, pad, false, s);
    defer release(&vp);
    var gp = if (pad == 0) try setCopy(in.g, s) else try padTime(in.g, pad, true, s);
    defer release(&gp);
    var bp = if (pad == 0) try setCopy(in.beta, s) else try padTime(in.beta, pad, false, s);
    defer release(&bp);

    var qh = try headsMajor4(qp, s); defer release(&qh); // [B,Hk,tp,dk]
    var kh = try headsMajor4(kp, s); defer release(&kh);
    var vh = try headsMajor4(vp, s); defer release(&vh); // [B,Hv,tp,dv]
    var gh = try headsMajor3(gp, s); defer release(&gh); // [B,Hv,tp]
    var bh = try headsMajor3(bp, s); defer release(&bh);

    // L = inclusive cumsum of log g — the log-space cumulative decay. Clamped
    // so a bf16-underflowed gate decays to (a very large finite) zero.
    var gclamp = try maximumScalar(gh, 1e-30, s); defer release(&gclamp);
    var logg = try logm(gclamp, s); defer release(&logg);
    var lfull = try cumsumAxis2(logg, s); defer release(&lfull); // [B,Hv,tp]
    var lr = try reshapeTo(lfull, &.{ b, hv, nc, cchunk }, s); defer release(&lr);
    var lend = try sliceAxis(lr, 3, cchunk - 1, cchunk, s); defer release(&lend); // [B,Hv,nc,1]

    var s_enter = try astypeF32(in.state, s); // [B,Hv,Dv,Dk], loop-carried
    errdefer release(&s_enter);
    const y_parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(y_parts);

    for (0..@intCast(nc)) |ci| {
        const c0: c_int = @intCast(ci * CHUNK);
        const c1 = c0 + cchunk;
        var kc0 = try sliceAxis(kh, 2, c0, c1, s); defer release(&kc0); // [B,Hk,C,dk]
        var kgt = try tileHeads(kc0, grp, s); defer release(&kgt); // [B,Hv,C,dk]
        var qc0 = try sliceAxis(qh, 2, c0, c1, s); defer release(&qc0);
        var qct = try tileHeads(qc0, grp, s); defer release(&qct);
        var vc = try sliceAxis(vh, 2, c0, c1, s); defer release(&vc); // [B,Hv,C,dv]
        var lc1 = try sliceAxis(lr, 2, @intCast(ci), @as(c_int, @intCast(ci)) + 1, s); defer release(&lc1); // [B,Hv,1,C]
        var lc = try reshapeTo(lc1, &.{ b, hv, cchunk }, s); defer release(&lc);
        var lc4 = try reshapeTo(lc, &.{ b, hv, cchunk, 1 }, s); defer release(&lc4); // [B,Hv,C,1]
        var bc1 = try sliceAxis(bh, 2, c0, c1, s); defer release(&bc1);
        var bc4 = try reshapeTo(bc1, &.{ b, hv, cchunk, 1 }, s); defer release(&bc4);

        // Decay weights in the exponent domain: the differences are ≤ 0 and
        // masked entries are pushed underflow-negative. fst is strictly
        // lower, fin covers the diagonal too (the y read).
        var lc4t = try transposeAxes(lc4, &.{ 0, 1, 3, 2 }, s); defer release(&lc4t); // [B,Hv,1,C]
        var diff = try sub(lc4, lc4t, s); defer release(&diff); // [B,Hv,C,C]
        var fst_e = try sub(diff, msk.strict_big, s); defer release(&fst_e);
        var fst = try expm(fst_e, s); defer release(&fst);
        var fin_e = try sub(diff, msk.incl_big, s); defer release(&fin_e);
        var fin = try expm(fin_e, s); defer release(&fin);

        var kt = try transposeAxes(kgt, &.{ 0, 1, 3, 2 }, s); defer release(&kt); // [B,Hv,dk,C]
        var kk = try matmul(kgt, kt, s); defer release(&kk);
        var aw = try mul(kk, fst, s); defer release(&aw);
        var amat = try mul(aw, bc4, s); defer release(&amat); // A[i,j] = β_i·e^(L_i−L_j)·(k_i·k_j)

        // (I − A)^{-1} = Π_{k=0..5} (I + A^{2^k}): powers of one matrix
        // commute, A is nilpotent at A^64. Five squarings, five products.
        var apow = try setCopy(amat, s); defer release(&apow);
        var series = try add(amat, msk.ident, s); defer release(&series);
        var pp: c_int = 1;
        while (pp < 32) : (pp *= 2) {
            const ap2 = try matmul(apow, apow, s);
            release(&apow);
            apow = ap2;
            var eye = try add(apow, msk.ident, s);
            const s2 = try matmul(series, eye, s);
            release(&series);
            release(&eye);
            series = s2;
        }

        var st = try transposeAxes(s_enter, &.{ 0, 1, 3, 2 }, s); defer release(&st); // [B,Hv,dk,dv]
        var s0t = try matmul(kgt, st, s); defer release(&s0t); // k·S_enter [B,Hv,C,dv]
        var exp_lc = try expm(lc4, s); defer release(&exp_lc); // ≤ 1
        var scarry = try mul(s0t, bc4, s); defer release(&scarry);
        var scarry2 = try mul(scarry, exp_lc, s); defer release(&scarry2); // β_i·e^(L_i)·(k_i·S_enter)
        var vbeta = try mul(vc, bc4, s); defer release(&vbeta); // β_i·v_i
        var rhs = try sub(vbeta, scarry2, s); defer release(&rhs);
        var u = try matmul(series, rhs, s); defer release(&u); // the delta updates [B,Hv,C,dv]

        var qq = try matmul(qct, kt, s); defer release(&qq);
        var w = try mul(qq, fin, s); defer release(&w);
        var yin = try matmul(w, u, s); defer release(&yin);
        var ys0 = try matmul(qct, st, s); defer release(&ys0); // q·S_enter
        var ys0s = try mul(ys0, exp_lc, s); defer release(&ys0s);
        var yc = try add(yin, ys0s, s); defer release(&yc);
        try mlx.check(mlx.mlx_vector_array_append_value(y_parts, yc));

        // Chunk exit contribution; the carried state decays by e^(L_end).
        var lend_c = try sliceAxis(lend, 2, @intCast(ci), @as(c_int, @intCast(ci)) + 1, s); defer release(&lend_c); // [B,Hv,1,1]
        var end_diff = try sub(lend_c, lc4, s); defer release(&end_diff);
        var endw = try expm(end_diff, s); defer release(&endw);
        var uw = try mul(u, endw, s); defer release(&uw);
        var uwt = try transposeAxes(uw, &.{ 0, 1, 3, 2 }, s); defer release(&uwt); // [B,Hv,dv,C]
        var mc = try matmul(uwt, kgt, s); defer release(&mc); // [B,Hv,dv,dk]
        var exp_lend = try expm(lend_c, s); defer release(&exp_lend);
        var decayed = try mul(s_enter, exp_lend, s); defer release(&decayed);
        const snext = try add(decayed, mc, s);
        release(&s_enter);
        s_enter = snext;
        // ponytail: one eval per chunk bounds the lazy graph (chunk temporaries
        // would otherwise accumulate across the scan); a doubling scan or a
        // Vulkan GDN kernel removes the per-chunk sync later.
        try mlx.check(mlx.mlx_array_eval(s_enter));
    }

    const y = blk: {
        var y_cat = try concatParts(y_parts, 2, s); // [B,Hv,tp,dv]
        defer release(&y_cat);
        var trim = try sliceAxis(y_cat, 2, 0, t, s);
        defer release(&trim);
        var yt = try transposeAxes(trim, &.{ 0, 2, 1, 3 }, s);
        defer release(&yt);
        break :blk try astypeBf16(yt, s);
    };
    const state = try astypeBf16(s_enter, s);
    release(&s_enter);
    return .{ .y = y, .state = state };
}
