const std = @import("std");
const shared = @import("shared_benchmarks.zig");

pub fn runSingleThreadedBenchmarks(allocator: std.mem.Allocator, config: shared.BenchmarkConfig) !void {
    std.debug.print("Running Single-Threaded Benchmarks\n", .{});
    std.debug.print("================================\n\n", .{});

    for (config.dimensions) |dim| {
        // Insertion benchmark
        const insertion_result = try shared.runInsertionBenchmark(allocator, config.num_points, dim, null);
        std.debug.print("{}\n", .{insertion_result});

        // Search benchmarks
        for (config.k_values) |k| {
            const search_result = try shared.runSearchBenchmark(allocator, config.num_points, dim, config.num_queries, k, null);
            std.debug.print("{}\n", .{search_result});
        }

        std.debug.print("\n", .{});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = shared.BenchmarkConfig{
        .num_points = 100000,
        .dimensions = &[_]usize{ 128, 512, 768, 1024 },
        .num_queries = 10000,
        .k_values = &[_]usize{ 10, 25, 50, 100 },
    };

    try runSingleThreadedBenchmarks(allocator, config);
}
