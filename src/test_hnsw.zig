const std = @import("std");
const testing = std.testing;
const HNSW = @import("hnsw.zig").HNSW;

// Helper function to create a random point
fn randomPoint(allocator: std.mem.Allocator, dim: usize) ![]f32 {
    const point = try allocator.alloc(f32, dim);
    for (point) |*v| {
        v.* = std.crypto.random.float(f32);
    }
    return point;
}

// Helper function to calculate Euclidean distance
fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, 0..) |_, i| {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

test "HNSW - Basic Functionality" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    // Insert some points
    try hnsw.insert(&[_]f32{ 1, 2, 3 });
    try hnsw.insert(&[_]f32{ 4, 5, 6 });
    try hnsw.insert(&[_]f32{ 7, 8, 9 });

    // Search for nearest neighbors
    const query = &[_]f32{ 3, 4, 5 };
    const results = try hnsw.search(query, 2);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);
    try testing.expect(euclideanDistance(query, results[0].point) <= euclideanDistance(query, results[1].point));
}

test "HNSW - Empty Index" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const query = &[_]f32{ 1, 2, 3 };
    const results = try hnsw.search(query, 5);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 0), results.len);
}

test "HNSW - Single Point" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const point = &[_]f32{ 1, 2, 3 };
    try hnsw.insert(point);

    const results = try hnsw.search(point, 1);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqualSlices(f32, point, results[0].point);
}

test "HNSW - Large Dataset" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert many points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Search for nearest neighbors
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const k = 10;
    const results = try hnsw.search(query, k);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, k), results.len);

    // Check if results are sorted by distance
    var last_dist: f32 = 0;
    for (results) |result| {
        const dist = euclideanDistance(query, result.point);
        try testing.expect(dist >= last_dist);
        last_dist = dist;
    }
}

test "HNSW - Edge Cases" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    // Insert duplicate points
    const point = &[_]f32{ 1, 2, 3 };
    try hnsw.insert(point);
    try hnsw.insert(point);

    const results = try hnsw.search(point, 2);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);
    try testing.expectEqualSlices(f32, point, results[0].point);
    try testing.expectEqualSlices(f32, point, results[1].point);

    // Search with k larger than number of points
    const large_k_results = try hnsw.search(point, 100);
    defer allocator.free(large_k_results);

    try testing.expectEqual(@as(usize, 2), large_k_results.len);
}

test "HNSW - Memory Leaks" {
    var hnsw: HNSW(f32) = undefined;
    {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        hnsw = HNSW(f32).init(allocator, 16, 200);

        const num_points = 1000;
        const dim = 64;

        for (0..num_points) |_| {
            const point = try randomPoint(allocator, dim);
            try hnsw.insert(point);
            // Intentionally not freeing 'point' to test if HNSW properly manages memory
        }

        const query = try randomPoint(allocator, dim);
        const results = try hnsw.search(query, 10);
        _ = results;
        // Intentionally not freeing 'results' or 'query'
    }
    // The ArenaAllocator will detect any memory leaks when it's deinitialized
}

test "HNSW - Concurrent Access" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const num_threads = 8;
    const points_per_thread = 1000;
    const dim = 128;

    const ThreadContext = struct {
        hnsw: *HNSW(f32),
        allocator: std.mem.Allocator,
    };

    const thread_fn = struct {
        fn func(ctx: *const ThreadContext) !void {
            for (0..points_per_thread) |_| {
                const point = try ctx.allocator.alloc(f32, dim);
                defer ctx.allocator.free(point);
                for (point) |*v| {
                    v.* = std.crypto.random.float(f32);
                }
                try ctx.hnsw.insert(point);
            }
        }
    }.func;

    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]ThreadContext = undefined;

    for (&threads, 0..) |*thread, i| {
        contexts[i] = .{
            .hnsw = &hnsw,
            .allocator = allocator,
        };
        thread.* = try std.Thread.spawn(.{}, thread_fn, .{&contexts[i]});
    }

    for (&threads) |*thread| {
        thread.join();
    }

    // Verify that all points were inserted
    const expected_count = num_threads * points_per_thread;
    const actual_count = hnsw.nodes.count();
    try testing.expectEqual(expected_count, actual_count);

    // Test search after concurrent insertion
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const results = try hnsw.search(query, 10);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 10), results.len);
}

test "HNSW - Stress Test" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const num_points = 100000;
    const dim = 128;
    const num_queries = 100;

    // Insert many points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Perform multiple searches
    for (0..num_queries) |_| {
        const query = try randomPoint(allocator, dim);
        defer allocator.free(query);

        const results = try hnsw.search(query, 10);
        defer allocator.free(results);

        try testing.expectEqual(@as(usize, 10), results.len);
    }
}

test "HNSW - Different Data Types" {
    const allocator = testing.allocator;

    // Test with integer type
    {
        var hnsw_int = HNSW(i32).init(allocator, 16, 200);
        defer hnsw_int.deinit();

        try hnsw_int.insert(&[_]i32{ 1, 2, 3 });
        try hnsw_int.insert(&[_]i32{ 4, 5, 6 });
        try hnsw_int.insert(&[_]i32{ 7, 8, 9 });

        const query_int = &[_]i32{ 3, 4, 5 };
        const results_int = try hnsw_int.search(query_int, 2);
        defer allocator.free(results_int);

        try testing.expectEqual(@as(usize, 2), results_int.len);
    }

    // Test with float64 type
    {
        var hnsw_f64 = HNSW(f64).init(allocator, 16, 200);
        defer hnsw_f64.deinit();

        try hnsw_f64.insert(&[_]f64{ 1.1, 2.2, 3.3 });
        try hnsw_f64.insert(&[_]f64{ 4.4, 5.5, 6.6 });
        try hnsw_f64.insert(&[_]f64{ 7.7, 8.8, 9.9 });

        const query_f64 = &[_]f64{ 3.3, 4.4, 5.5 };
        const results_f64 = try hnsw_f64.search(query_f64, 2);
        defer allocator.free(results_f64);

        try testing.expectEqual(@as(usize, 2), results_f64.len);
    }
}

test "HNSW - Consistency" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Perform multiple searches with the same query
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const num_searches = 10;
    const k = 10;
    var first_result = try allocator.alloc(f32, k * dim);
    defer allocator.free(first_result);

    for (0..num_searches) |i| {
        const results = try hnsw.search(query, k);
        defer allocator.free(results);

        if (i == 0) {
            // Store the first result for comparison
            for (results, 0..) |result, j| {
                @memcpy(first_result[j * dim .. (j + 1) * dim], result.point);
            }
        } else {
            // Compare with the first result
            for (results, 0..) |result, j| {
                const start = j * dim;
                const end = (j + 1) * dim;
                try testing.expectEqualSlices(f32, first_result[start..end], result.point);
            }
        }
    }
}

// TODO: Implement deletion
// test "HNSW - Deletion" {
//     const allocator = testing.allocator;
//     var hnsw = HNSW(f32).init(allocator, 16, 200);
//     defer hnsw.deinit();

//     const num_points = 10;
//     const dim = 3;

//     // Insert points
//     var points = try allocator.alloc([]f32, num_points);
//     defer {
//         for (points) |p| allocator.free(p);
//         allocator.free(points);
//     }

//     for (0..num_points) |i| {
//         points[i] = try randomPoint(allocator, dim);
//         try hnsw.insert(points[i]);
//         std.debug.print("Inserted point {d}\n", .{i});
//     }

//     // Delete every other point
//     for (0..num_points, 0..) |i, id| {
//         if (i % 2 == 0) {
//             try hnsw.delete(id);
//             std.debug.print("Deleted point {d}\n", .{id});
//         }
//     }

//     // Verify that deleted points are not in search results
//     const query = try randomPoint(allocator, dim);
//     defer allocator.free(query);

//     const results = try hnsw.search(query, num_points);
//     defer allocator.free(results);

//     std.debug.print("Search results:\n", .{});
//     for (results) |result| {
//         std.debug.print("Result ID: {d}\n", .{result.id});
//         try testing.expect(result.id % 2 != 0);
//     }

//     // Verify that non-deleted points are still searchable
//     for (1..num_points, 1..) |i, id| {
//         if (i % 2 != 0) {
//             std.debug.print("Searching for point {d}\n", .{id});
//             const point_results = try hnsw.search(points[i], 1);
//             defer allocator.free(point_results);
//             if (point_results.len > 0) {
//                 std.debug.print("Found result: {d}\n", .{point_results[0].id});
//                 try testing.expectEqual(id, point_results[0].id);
//             } else {
//                 std.debug.print("No results found for point {d}\n", .{id});
//             }
//         }
//     }
// }

test "HNSW - Performance Benchmarks" {
    const allocator = testing.allocator;

    // Insertion benchmark
    try benchmarkInsertion(allocator, 100000, 128);

    // Search benchmark
    try benchmarkSearch(allocator, 100000, 128, 10000, 10);
}

pub fn benchmarkInsertion(allocator: std.mem.Allocator, num_points: usize, dim: usize) !void {
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    const end = timer.lap();
    const elapsed_ns = end - start;
    const points_per_second = @as(f64, @floatFromInt(num_points)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    std.debug.print("Insertion Benchmark:\n", .{});
    std.debug.print("  Points: {d}\n", .{num_points});
    std.debug.print("  Dimensions: {d}\n", .{dim});
    std.debug.print("  Total time: {d:.2} seconds\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e9});
    std.debug.print("  Points per second: {d:.2}\n", .{points_per_second});
}

pub fn benchmarkSearch(allocator: std.mem.Allocator, num_points: usize, dim: usize, num_queries: usize, k: usize) !void {
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    // Insert points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    for (0..num_queries) |_| {
        const query = try randomPoint(allocator, dim);
        defer allocator.free(query);
        const results = try hnsw.search(query, k);
        allocator.free(results);
    }

    const end = timer.lap();
    const elapsed_ns = end - start;
    const queries_per_second = @as(f64, @floatFromInt(num_queries)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    std.debug.print("Search Benchmark:\n", .{});
    std.debug.print("  Points in index: {d}\n", .{num_points});
    std.debug.print("  Dimensions: {d}\n", .{dim});
    std.debug.print("  Queries: {d}\n", .{num_queries});
    std.debug.print("  k: {d}\n", .{k});
    std.debug.print("  Total time: {d:.2} seconds\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e9});
    std.debug.print("  Queries per second: {d:.2}\n", .{queries_per_second});
}
