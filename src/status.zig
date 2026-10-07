const std = @import("std");
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// ── macOS Mach/IOKit/CoreFoundation ABI ──
// Declared only on macOS: none of these symbols exist in glibc, and every
// Linux caller below takes a /proc branch instead. A stray reference from a
// non-macOS path is a compile error, not a silent link failure.

const mac = if (is_macos) struct {
    extern "c" var mach_task_self_: u32;
    extern "c" fn mach_host_self() u32;
    extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
    extern "c" fn host_statistics(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
    extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
    extern "c" fn host_page_size(host: u32, out: *usize) i32;
    extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

    extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
    extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
    extern "c" fn IOIteratorNext(iter: u32) u32;
    extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
    extern "c" fn IOObjectRelease(obj: u32) i32;
    extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
    extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
    extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
    extern "c" fn CFRelease(cf: ?*const anyopaque) void;
} else struct {};

// ── Linux /proc readers. Same contract as the Mach queries: 0 = unknown,
// never "tiny machine". Raw POSIX (std.c), like round_cost's fingerprinting:
// these run on paths without an io handle. ──

fn openProc(path: []const u8) c_int {
    var pbuf: [4096]u8 = undefined;
    if (path.len >= pbuf.len) return -1;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    return std.c.open(pbuf[0..path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
}

fn readOnce(fd: c_int, buf: []u8) usize {
    var got: usize = 0;
    while (got < buf.len) {
        const n = std.c.read(fd, buf.ptr + got, buf.len - got);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return got;
}

/// First `Field:  <n> kB` line in a /proc file, in bytes. 0 when absent.
fn procFieldBytes(path: []const u8, field: []const u8) u64 {
    const fd = openProc(path);
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [16384]u8 = undefined;
    var lines = std.mem.splitScalar(u8, buf[0..readOnce(fd, &buf)], '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, field)) continue;
        const rest = std.mem.trim(u8, line[field.len..], " \t:");
        const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
        const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return 0;
        return kb * 1024;
    }
    return 0;
}

/// Resident set in MB from /proc/self/status (the footprint we have — Linux
/// has no phys_footprint).
fn linuxRssMb() u32 {
    return @intCast(procFieldBytes("/proc/self/status", "VmRSS") / (1024 * 1024));
}

var prev_cpu_total: u64 = 0;
var prev_cpu_idle: u64 = 0;

/// /proc/stat's aggregate cpu line; same first-sample-returns-0 shape as the
/// Mach tick math below (idle counts iowait, the standard Linux convention).
fn linuxCpuPct() u32 {
    const fd = openProc("/proc/stat");
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const raw = buf[0..readOnce(fd, &buf)];
    const line_end = std.mem.indexOfScalar(u8, raw, '\n') orelse raw.len;
    var toks = std.mem.tokenizeScalar(u8, raw[0..line_end], ' ');
    if (toks.next()) |head| {
        if (!std.mem.eql(u8, head, "cpu")) return 0;
    } else return 0;
    var total: u64 = 0;
    var idle: u64 = 0;
    var i: usize = 0;
    while (toks.next()) |tok| : (i += 1) {
        const v = std.fmt.parseInt(u64, tok, 10) catch 0;
        total += v;
        if (i == 3 or i == 4) idle += v; // idle + iowait
    }
    const d_total = total -| prev_cpu_total;
    const d_idle = idle -| prev_cpu_idle;
    prev_cpu_total = total;
    prev_cpu_idle = idle;
    if (d_total == 0) return 0;
    return @intCast((d_total - d_idle) * 100 / d_total);
}

// ── Mach struct layouts (extern = C ABI) ──

const TaskBasicInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_sec: i32,
    user_time_usec: i32,
    sys_time_sec: i32,
    sys_time_usec: i32,
    policy: i32,
    suspend_count: i32,
};

/// task_vm_info truncated through phys_footprint (rev1). Field order matches
/// <mach/task_info.h> exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 38 ==
/// TASK_VM_INFO_REV1_COUNT, so the kernel fills through phys_footprint without
/// overrunning the buffer.
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
};

const CpuLoadInfo = extern struct {
    ticks: [4]u32, // user, system, idle, nice
};

const VmStats64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

// ── CPU delta tracking (module-level state) ──
var prev_ticks: [4]u64 = @splat(0);

// ── Public metric helpers ──

pub fn getAppRssMb() u32 {
    if (comptime !is_macos) return linuxRssMb();
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (mac.task_info(mac.mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

/// Process physical memory footprint in MB (TASK_VM_INFO flavor 22). Unlike
/// resident_size, this includes MLX's Metal/IOKit + compressed memory — the
/// only figure that reflects a loaded model's true footprint on Apple Silicon.
pub fn getAppMemFootprintMb() u32 {
    if (comptime !is_macos) return linuxRssMb();
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32); // 38 = TASK_VM_INFO_REV1_COUNT
    if (mac.task_info(mac.mach_task_self_, 22, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.phys_footprint / (1024 * 1024));
}

/// Bytes of physical memory available for new allocation without heavy
/// Pure: bytes available for a new large allocation given the live page counts.
///
/// Subtracts the genuinely non-reclaimable set: `wired` (pinned), `compressor`
/// (already-compressed app data), and `internal` (anonymous app pages — crucially
/// INCLUDING a resident MLX model). File-backed cache (`external`) plus
/// free/speculative/purgeable pages are NOT subtracted: macOS evicts them the
/// instant a big allocation lands, so they don't block a load. That keeps a 12B
/// (~7.7 GB) loading on a 16 GB Mac that shows only ~7.8 GB *instantaneous* free
/// (the rest is reclaimable file cache). It also fixes the #45 OOM: a prior model
/// still resident lives in the anonymous (`internal`) set, NOT necessarily in
/// `wired` (verified live: a 5 GB resident model with only ~2.8 GB total wired) —
/// so counting `internal` makes a second large load correctly fail the guard.
/// `purgeable` anon pages (caches an app explicitly marked discardable — e.g.
/// image/tile caches) are a SUBSET of `internal` that macOS drops the instant a
/// big allocation lands, so they must NOT count as used; subtracting them back
/// out of the internal set is the accuracy fix. Still slightly conservative
/// (wired-anonymous pages can appear in both `wired` and `internal`), which is
/// the safe direction for an OOM guard (`--skip-mem-preflight` overrides).
/// Returns 0 when total is 0 or used ≥ total (a failed query must never block).
fn computeAvailableBytes(total_mem: u64, wire_pages: u64, compressor_pages: u64, internal_pages: u64, purgeable_pages: u64, page: u64) u64 {
    // Purgeable is reclaimable, so exclude it from the resident anon set. Saturate
    // (never underflow) in case the counters momentarily disagree.
    const resident_anon: u64 = internal_pages -| purgeable_pages;
    const used: u64 = (wire_pages + compressor_pages + resident_anon) * page;
    if (total_mem == 0 or used >= total_mem) return 0;
    return total_mem - used;
}

extern "c" fn os_proc_available_memory() usize;

/// Per-process memory headroom before jetsam (iOS only — the entitlement-
/// aware figure that actually governs whether a big allocation survives).
/// Returns 0 on macOS, where the concept doesn't apply; the symbol is only
/// referenced on iOS builds so macOS links are unaffected.
pub fn getProcAvailableMemBytes() u64 {
    if (comptime builtin.os.tag != .ios) return 0;
    return @intCast(os_proc_available_memory());
}

/// Total physical RAM (hw.memsize on macOS/iOS; /proc/meminfo MemTotal on
/// Linux). 0 on failure.
pub fn getTotalMemBytes() u64 {
    if (comptime !is_macos) return procFieldBytes("/proc/meminfo", "MemTotal");
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (mac.sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

/// Bytes the kernel holds wired right now (Metal's resident buffers among
/// them). 0 on failure. Always 0 on Linux — there is no wired set; callers
/// treat 0 as unknown (the unwire wait skips itself).
pub fn getWiredMemBytes() u64 {
    if (comptime !is_macos) return 0;
    var page: usize = 0;
    if (mac.host_page_size(mac.mach_host_self(), &page) != 0) return 0;
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (mac.host_statistics64(mac.mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;
    return @as(u64, vm.wire_count) * page;
}

pub fn getAvailableMemBytes() u64 {
    if (comptime !is_macos) return procFieldBytes("/proc/meminfo", "MemAvailable");
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (mac.sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (mac.host_page_size(mac.mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (mac.host_statistics64(mac.mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    return computeAvailableBytes(total_mem, vm.wire_count, vm.compressor_page_count, vm.internal_page_count, vm.purgeable_count, page);
}

test "computeAvailableBytes counts the resident anon set, not file cache or purgeable" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page; // pages per GB

    // 16 GB Mac, light anon load: 3 GB wired, 1 GB compressed, 2 GB anonymous app
    // pages, no purgeable; the remaining ~10 GB is free + reclaimable file cache,
    // which must NOT count against availability. Available = 16 − (3+1+2) = 10 GB.
    // (The old `active`-subtracting formula counted file cache and wrongly refused
    // loads that fit; later dropping `active` entirely wrongly ignored resident
    // models.)
    try std.testing.expectEqual(@as(u64, 10 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 2 * ppg, 0, page));

    // #45 OOM guard: a prior 7 GB model is resident. It lives in the anonymous
    // (`internal`) set — here 9 GB = 2 GB apps + 7 GB model — NOT in `wired`. So
    // available = 16 − (3+1+9) = 3 GB, and a second 7 GB load is correctly refused.
    try std.testing.expectEqual(@as(u64, 3 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 9 * ppg, 0, page));

    // Purgeable is reclaimable: 2 GB of the 9 GB internal set is a discardable
    // cache, so it should NOT count as used. Available = 16 − (3+1+(9−2)) = 5 GB,
    // up from the 3 GB the old formula reported — the accuracy fix.
    try std.testing.expectEqual(@as(u64, 5 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 9 * ppg, 2 * ppg, page));

    // Purgeable never underflows the anon set even if the counters disagree.
    try std.testing.expectEqual(@as(u64, 12 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 1 * ppg, 5 * ppg, page));

    // Degenerate guards: failed query (total 0) and used ≥ total → 0, never block.
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(0, 1, 1, 1, 0, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(8 * GB, 4 * ppg, 0, 5 * ppg, 0, page));
}

pub fn getSysMemPct() u32 {
    if (comptime !is_macos) {
        const total = procFieldBytes("/proc/meminfo", "MemTotal");
        const avail = procFieldBytes("/proc/meminfo", "MemAvailable");
        if (total == 0) return 0;
        return @intCast((total -| avail) * 100 / total);
    }
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (mac.sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (mac.host_page_size(mac.mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (mac.host_statistics64(mac.mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    const used: u64 = (@as(u64, vm.active_count) + vm.wire_count + vm.compressor_page_count) * page;
    if (total_mem == 0) return 0;
    return @intCast(used * 100 / total_mem);
}

pub fn getCpuPct() u32 {
    if (comptime !is_macos) return linuxCpuPct();
    var info = std.mem.zeroes(CpuLoadInfo);
    var count: u32 = 4;
    if (mac.host_statistics(mac.mach_host_self(), 3, @ptrCast(&info), &count) != 0) return 0;

    var total: u64 = 0;
    var idle: u64 = 0;
    for (0..4) |i| {
        const cur: u64 = info.ticks[i];
        const delta = cur -| prev_ticks[i];
        total += delta;
        if (i == 2) idle = delta;
        prev_ticks[i] = cur;
    }
    if (total == 0) return 0;
    return @intCast((total - idle) * 100 / total);
}

pub fn getGpuPct() u32 {
    // IOKit's IOServiceMatching/AGXAccelerator path is macOS-only. On every
    // other target we report 0 — the value is only a log-line stat.
    if (comptime !is_macos) return 0;
    const matching = mac.IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (mac.IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = mac.IOObjectRelease(iter);

    const entry = mac.IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = mac.IOObjectRelease(entry);

    var props: ?*anyopaque = null;
    if (mac.IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| mac.CFRelease(p);

    const perf = cfDictGet(props, "PerformanceStatistics") orelse return 0;
    const util = cfDictGet(perf, "Device Utilization %") orelse return 0;

    var value: i64 = 0;
    _ = mac.CFNumberGetValue(util, 4, @ptrCast(&value));
    return if (value >= 0 and value <= 100) @intCast(value) else 0;
}

fn cfDictGet(dict: ?*const anyopaque, key_name: [*:0]const u8) ?*const anyopaque {
    const key = mac.CFStringCreateWithCString(null, key_name, 0x08000100) orelse return null;
    defer mac.CFRelease(key);
    return mac.CFDictionaryGetValue(dict, key);
}

test "getAppMemFootprintMb returns a plausible nonzero footprint" {
    const fp = getAppMemFootprintMb();
    // The test process itself footprints several MB; a wrong flavor/offset
    // would yield 0 or absurd garbage.
    try std.testing.expect(fp > 0);
    try std.testing.expect(fp < 1024 * 1024); // < 1 TB sanity bound
}
