const std = @import("std");
const shared = @import("shared_benchmarks.zig");

pub fn runMultiThreadedBenchmarks(allocator: std.mem.Allocator, config: shared.BenchmarkConfig) !void {
    std.debug.print("Running Multi-Threaded Benchmarks\n", .{});
    std.debug.print("================================\n\n", .{});

    const thread_counts = [_]usize{ 2, 4, 8 };

    for (config.dimensions) |dim| {
        for (thread_counts) |thread_count| {
            // Insertion benchmark
            const insertion_result = try shared.runInsertionBenchmark(allocator, config.num_points, dim, thread_count);
            std.debug.print("{}\n", .{insertion_result});

            // Search benchmarks
            for (config.k_values) |k| {
                const search_result = try shared.runSearchBenchmark(allocator, config.num_points, dim, config.num_queries, k, thread_count);
                std.debug.print("{}\n", .{search_result});
            }

            std.debug.print("\n", .{});
        }
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

    try runMultiThreadedBenchmarks(allocator, config);
}
