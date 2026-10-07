//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.

const std = @import("std");

/// Open-fd stat across macOS and Linux. std.c.fstat is darwin-only in Zig
/// 0.17 and Linux lost the plain-Stat layout, so the Linux arm goes through
/// statx on /proc/self/fd.
pub const FdStat = struct { size: u64, mode: u32, nlink: u64 };

pub fn fdStat(fd: std.c.fd_t) !FdStat {
    if (comptime @import("builtin").os.tag == .macos) {
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.FdStatFailed;
        return .{ .size = @intCast(st.size), .mode = @intCast(st.mode), .nlink = st.nlink };
    }
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const proc_path = std.fmt.bufPrintSentinel(&buf, "/proc/self/fd/{d}", .{fd}, 0) catch
        return error.FdStatFailed;
    var stx: std.os.linux.Statx = undefined;
    const linux = std.os.linux;
    const rc = linux.statx(-100, proc_path.ptr, 0, linux.STATX.BASIC_STATS, &stx);
    if (rc != 0) return error.FdStatFailed;
    return .{ .size = stx.size, .mode = stx.mode, .nlink = stx.nlink };
}

pub fn fdSize(fd: std.c.fd_t) !u64 {
    return (try fdStat(fd)).size;
}

/// Seconds since the Unix epoch (wall-clock).
pub fn nowSecs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toSeconds();
}

/// Milliseconds since the Unix epoch (wall-clock).
pub fn nowMs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toMilliseconds();
}

/// Milliseconds on the MONOTONIC `boot` clock (arbitrary epoch — only
/// differences are meaningful). Use for deadlines/intervals, never for
/// timestamps a client sees: an NTP step or a manual clock change must not be
/// able to stall (or spam) a timer. `.boot` counts across pmset sleep, so a
/// deadline that expired while the lid was shut fires immediately on wake.
pub fn nowMsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMilliseconds();
}

/// Drop-in replacement for `std.time.Timer` — uses the `boot` clock, which
/// counts wall-time across pmset sleep (vs `.awake`, which stops during
/// suspend). Important for long-running stopwatches that span a lid close.
/// Resolution is set by the `Io` implementation.
pub const Stopwatch = struct {
    io: std.Io,
    started_at: std.Io.Timestamp,

    pub fn init(io: std.Io) Stopwatch {
        return .{ .io = io, .started_at = std.Io.Timestamp.now(io, .boot) };
    }

    /// Nanoseconds since `init` (or last `reset`).
    pub fn read(s: Stopwatch) u64 {
        return @intCast(s.started_at.untilNow(s.io, .boot).nanoseconds);
    }

    /// Reset the start point to "now".
    pub fn reset(s: *Stopwatch) void {
        s.started_at = std.Io.Timestamp.now(s.io, .boot);
    }
};
