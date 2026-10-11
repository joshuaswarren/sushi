const std = @import("std");
const builtin = @import("builtin");

comptime {
    // 0.16.0's bundled libc++ fails to compile against the macOS 27 SDK
    // (`use of undeclared identifier 'INFINITY'` in its vendored <random>).
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 17) {
        @compileError(std.fmt.comptimePrint(
            "sushi requires Zig 0.17 (have {d}.{d}.{d}). Run ./scripts/fetch-zig.sh, or grab 0.17.0 from https://ziglang.org/download/.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
    }
}

pub fn build(b: *std.Build) void {
    // Match libmlx's macOS 26.2 deployment target, required by its NAX kernels.
    // Only meaningful for macOS builds; Linux targets build native.
    const target = b.standardTargetOptions(.{
        .default_target = if (builtin.os.tag == .macos) .{
            .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
        } else .{},
    });
    const optimize = b.standardOptimizeOption(.{});
    const is_macos = target.result.os.tag == .macos;

    // Setting any non-default target field disables Zig's native macOS SDK detection,
    // so we resolve the SDK path ourselves and surface its frameworks dir.
    const macos_sdk_frameworks: ?[]const u8 = blk: {
        if (target.result.os.tag != .macos) break :blk null;
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
            &code,
            .inherit,
        ) catch break :blk null;
        const sdk = std.mem.trim(u8, stdout, " \n\r\t");
        if (sdk.len == 0) break :blk null;
        break :blk b.fmt("{s}/System/Library/Frameworks", .{sdk});
    };

    if (target.result.os.tag == .macos) {
        verifyBrewDeps(b);
        verifyMlxStage(b);
    } else {
        verifyMlxStageLinux(b);
    }

    // Version: SemVer, build.zig.zon's `.version` unless the release workflow
    // passes the tag's version (release.sh checks the two agree).
    const version = b.option([]const u8, "version", "SemVer version string (default: build.zig.zon)") orelse @import("build.zig.zon").version;
    _ = std.SemanticVersion.parse(version) catch {
        std.debug.print("[sushi] -Dversion={s} is not a SemVer version (MAJOR.MINOR.PATCH[-pre])\n", .{version});
        std.process.exit(1);
    };

    // MLX reports its version at runtime; mlx-c and the guest manifest need build-time pins.
    const mlx_c_version = b.option([]const u8, "mlx-c-version", "Pinned mlx-c version") orelse readMlxPin(b, "mlxc=") orelse "unknown";
    const mlx_sha = readMlxPin(b, "mlx=") orelse "";

    const webp_header: []const u8 = if (is_macos) "/opt/homebrew/include/webp/decode.h" else "/usr/include/webp/decode.h";
    const webp_include: []const u8 = if (is_macos) "/opt/homebrew/include" else "/usr/include";

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", version);
    build_options.addOption([]const u8, "mlx_c_version", mlx_c_version);
    build_options.addOption([]const u8, "mlx_sha", mlx_sha);
    const git_sha = b.option([]const u8, "git-sha", "Engine build id for the round-cost table: a release sha stands for the executable bytes, which are then not hashed; the MLX dylib and metallib fingerprints are always mixed in") orelse "";
    build_options.addOption([]const u8, "git_sha", git_sha);

    const mod = b.createModule(.{
        .root_source_file = b.path(if (b.option(bool, "glm-ffn-replay", "Build offline GLM frozen-prefix extractor") orelse false) "src/glm5_ffn_replay.zig" else "src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = webp_header }, .{ .cwd_relative = webp_include }, target, optimize) },
        },
    });

    // Jinja2 template engine (wangzhaode/jinja.cpp + nlohmann/json; see NOTICE).
    // macOS uses the prebuilt libjinja.a (system clang++, C++17 libc++); Linux
    // compiles the same 7 sources into the module (rebuild instructions in CLAUDE.md).
    if (is_macos) {
        mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    } else {
        addJinjaSources(b, mod);
    }
    mod.addIncludePath(b.path("lib/jinja_cpp"));

    // stb_image for JPEG/PNG decoding in the vision pipeline
    mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    mod.addCSourceFile(.{ .file = b.path("lib/dflash_cache_space.c"), .flags = &.{"-O2"} });
    mod.addIncludePath(b.path("lib"));

    addAneSources(b, mod, is_macos);

    // The staged MLX library path must precede Homebrew's.
    addMlxLib(b, mod, is_macos);
    _ = addExl3Module(b, mod, target, optimize);
    // webp include/lib paths (homebrew on macOS, system on Linux). The include
    // path is native-only: a cross build must not pull host glibc headers into
    // target C compilation; the webp translate module keeps its own header path.
    if (target.query.isNative()) {
        mod.addIncludePath(.{ .cwd_relative = webp_include });
    }
    if (is_macos) {
        mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    }
    mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    if (is_macos) {
        mod.linkFramework("IOKit", .{});
        mod.linkFramework("CoreFoundation", .{});
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("IOSurface", .{});
    }

    const exe = b.addExecutable(.{
        .name = "sushi",
        .root_module = mod,
    });

    // Ensure Mach-O header has room for install_name_tool path changes — the
    // release tarball rewires @rpath/libmlxc.dylib to @executable_path.
    if (is_macos) exe.headerpad_max_install_names = true;

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run sushi");
    run_step.dependOn(&run_cmd.step);

    // Unit tests — reuses the same module config (mlx-c, jinja_cpp, etc.)
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
            .{ .name = "jinja_c", .module = addCHeaderModule(b, b.path("lib/jinja_cpp/jinja_wrapper.h"), b.path("lib/jinja_cpp"), target, optimize) },
            .{ .name = "stb", .module = addCHeaderModule(b, b.path("lib/stb_image.h"), b.path("lib"), target, optimize) },
            .{ .name = "webp", .module = addCHeaderModule(b, .{ .cwd_relative = webp_header }, .{ .cwd_relative = webp_include }, target, optimize) },
        },
    });

    if (is_macos) {
        test_mod.addObjectFile(b.path("lib/jinja_cpp/libjinja.a"));
    } else {
        addJinjaSources(b, test_mod);
    }
    test_mod.addIncludePath(b.path("lib/jinja_cpp"));
    test_mod.addCSourceFile(.{ .file = b.path("lib/stb_image_impl.c"), .flags = &.{"-O2"} });
    test_mod.addCSourceFile(.{ .file = b.path("lib/dflash_cache_space.c"), .flags = &.{"-O2"} });
    test_mod.addIncludePath(b.path("lib"));
    addAneSources(b, test_mod, is_macos);
    test_mod.linkSystemLibrary("c++", .{});
    addMlxLib(b, test_mod, is_macos);
    const exl3_test_mod = addExl3Module(b, test_mod, target, optimize);
    test_mod.addIncludePath(.{ .cwd_relative = webp_include });
    if (is_macos) {
        test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    }
    test_mod.linkSystemLibrary("webp", .{});

    if (macos_sdk_frameworks) |fw_path| {
        test_mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    }
    if (is_macos) {
        test_mod.linkFramework("IOKit", .{});
        test_mod.linkFramework("CoreFoundation", .{});
        test_mod.linkFramework("Foundation", .{});
        test_mod.linkFramework("Metal", .{});
        test_mod.linkFramework("IOSurface", .{});
    }

    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose name contains this substring");
    const qwen_preprocess_fixture = b.option(
        []const u8,
        "qwen-preprocess-fixture",
        "CPU reference fixture for the gated Qwen preprocessing parity test",
    );
    const unit_tests = b.addTest(.{
        .root_module = test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    // Zig collects tests from an artifact's root module only, so src/exl3 is its own.
    const exl3_tests = b.addTest(.{
        .name = "exl3-test",
        .root_module = exl3_test_mod,
        .filters = if (test_filter) |f| &.{f} else &.{},
    });

    const test_build = b.step("test-build", "Compile unit tests without running them");
    test_build.dependOn(&b.addInstallArtifact(unit_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);
    test_build.dependOn(&b.addInstallArtifact(exl3_tests, .{ .dest_dir = .{ .override = .{ .custom = "tests" } } }).step);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    if (qwen_preprocess_fixture) |fixture| {
        run_unit_tests.setEnvironmentVariable("QWEN_PREPROCESS_FIXTURE", fixture);
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/manifest.json", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/source_rgb.bin", .{fixture}) });
        run_unit_tests.addFileInput(.{ .cwd_relative = b.fmt("{s}/pixel_values.bin", .{fixture}) });
    }
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&b.addRunArtifact(exl3_tests).step);

    // exl3 tests only: they link the staged mlx-c and run without the
    // fixtures the full unit-test artifact needs.
    const test_exl3_step = b.step("test-exl3", "Run the exl3 tests");
    test_exl3_step.dependOn(&b.addRunArtifact(exl3_tests).step);
}

fn addCHeaderModule(
    b: *std.Build,
    header_path: std.Build.LazyPath,
    include_dir: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = header_path,
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(include_dir);
    return translate.createModule();
}

/// ARC bridge to AppleNeuralEngine, dlopen'd and checked for availability at
/// runtime. Linux links ane_stub.c instead: every symbol answers unavailable,
/// so the opt-in --ane-prefill path never engages.
fn addAneSources(b: *std.Build, module: *std.Build.Module, is_macos: bool) void {
    module.addIncludePath(b.path("lib/ane"));
    if (!is_macos) {
        module.addCSourceFile(.{ .file = b.path("lib/ane/ane_stub.c"), .flags = &.{"-O2"} });
        return;
    }
    const objc_flags = &[_][]const u8{
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
    };
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_bridge.m"), .flags = objc_flags });
    module.addCSourceFile(.{ .file = b.path("lib/ane/ane_mlp.m"), .flags = objc_flags });
}

/// Linux: compile the jinja.cpp sources into the module instead of the
/// macOS-prebuilt libjinja.a (CLAUDE.md: clang++ -std=c++17 -O2 -DNDEBUG -I .).
fn addJinjaSources(b: *std.Build, module: *std.Build.Module) void {
    const cpp_flags = &[_][]const u8{ "-O2", "-DNDEBUG" };
    const sources = [_][]const u8{
        "caps.cpp",
        "jinja_string.cpp",
        "jinja_wrapper.cpp",
        "lexer.cpp",
        "parser.cpp",
        "runtime.cpp",
        "value.cpp",
    };
    for (sources) |src| {
        module.addCSourceFile(.{
            .file = b.path(b.fmt("lib/jinja_cpp/{s}", .{src})),
            .flags = cpp_flags,
        });
    }
}

fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// src/exl3: the EXL3 expert engine, a module so another MLX host (mlx-serve)
/// can root it too. It reaches mlx, log and io_util through `mlx_host`, so the
/// host root file must expose them as `pub const`.
fn addExl3Module(b: *std.Build, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const exl3 = b.createModule(.{
        .root_source_file = b.path("src/exl3/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "mlx_host", .module = host }},
    });
    host.addImport("sushi_exl3", exl3);
    return exl3;
}

/// Link the libraries staged by scripts/build-mlx.sh (macOS, NAX-enabled) or
/// scripts/build-mlx-linux.sh (Linux, omarchy Vulkan backend).
/// Release packaging rewrites their @rpath install names to @executable_path.
fn addMlxLib(b: *std.Build, module: *std.Build.Module, is_macos: bool) void {
    module.addIncludePath(b.path("lib/mlx/include"));
    module.addLibraryPath(b.path("lib/mlx/lib"));
    // Prevent Homebrew's mlx-c.pc from overriding the staged libraries.
    module.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    // Binary-relative paths cover zig-out/bin and .zig-cache/o/<hash> respectively.
    if (is_macos) {
        module.addRPath(.{ .cwd_relative = "@loader_path/../../lib/mlx/lib" });
        module.addRPath(.{ .cwd_relative = "@loader_path/../../../lib/mlx/lib" });
    } else {
        module.addRPath(.{ .cwd_relative = "$ORIGIN/../../lib/mlx/lib" });
        module.addRPath(.{ .cwd_relative = "$ORIGIN/../../../lib/mlx/lib" });
    }
}

/// Fail before linking if the local MLX stage is missing.
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[sushi] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// Linux stage check: libmlx.so + libmlxc.so from the omarchy Vulkan build.
fn verifyMlxStageLinux(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.so", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlx.so", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[sushi] lib/mlx is not staged for Linux (omarchy mlx + mlx-c). Run:\n" ++
                "  ./scripts/build-mlx-linux.sh   (see the script header for prerequisites)\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// A pinned revision (`key` "mlx=" or "mlxc=") from lib/mlx/.version, written
/// by scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>". Returns
/// null when not staged yet.
fn readMlxPin(b: *std.Build, key: []const u8) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key)) return b.dupe(tok[key.len..]);
    }
    return null;
}

const BrewDep = struct { name: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    .{ .name = "webp", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[sushi] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[sushi] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[sushi] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[sushi] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
