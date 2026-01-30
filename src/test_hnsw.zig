const std = @import("std");
const testing = std.testing;
const zvdb = @import("zvdb.zig");
const HNSW = zvdb.HNSW;

// Test metadata struct for code search use case
const TestMeta = struct {
    path: []const u8,
    line_start: u32,
    line_end: u32,
};

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

// =============================================================================
// Distance Metric Tests
// =============================================================================

test "distance metrics - squared euclidean" {
    const a = &[_]f32{ 1, 0, 0 };
    const b = &[_]f32{ 0, 1, 0 };
    const dist = zvdb.distance.distanceFn(f32, .squared_euclidean);
    try testing.expectApproxEqAbs(@as(f32, 2.0), dist(a, b), 1e-6);
}

test "distance metrics - euclidean" {
    const a = &[_]f32{ 1, 0, 0 };
    const b = &[_]f32{ 0, 1, 0 };
    const dist = zvdb.distance.distanceFn(f32, .euclidean);
    try testing.expectApproxEqAbs(@as(f32, std.math.sqrt(2.0)), dist(a, b), 1e-6);
}

test "distance metrics - cosine orthogonal" {
    const a = &[_]f32{ 1, 0, 0 };
    const b = &[_]f32{ 0, 1, 0 };
    const dist = zvdb.distance.distanceFn(f32, .cosine);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dist(a, b), 1e-6);
}

test "distance metrics - cosine identical" {
    const a = &[_]f32{ 1, 2, 3 };
    const b = &[_]f32{ 1, 2, 3 };
    const dist = zvdb.distance.distanceFn(f32, .cosine);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dist(a, b), 1e-6);
}

test "distance metrics - dot product" {
    const a = &[_]f32{ 1, 2, 3 };
    const b = &[_]f32{ 4, 5, 6 };
    const dist = zvdb.distance.distanceFn(f32, .dot_product);
    // dot = 4 + 10 + 18 = 32, negated = -32
    try testing.expectApproxEqAbs(@as(f32, -32.0), dist(a, b), 1e-6);
}

// =============================================================================
// SIMD Distance Equivalence Tests
// =============================================================================

fn scalarSquaredEuclidean(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            var sum: T = 0;
            for (a, b) |av, bv| {
                const d = av - bv;
                sum += d * d;
            }
            return sum;
        }
    }.f;
}

fn scalarDot(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            var sum: T = 0;
            for (a, b) |av, bv| sum += av * bv;
            return sum;
        }
    }.f;
}

fn scalarCosine(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            var dot: T = 0;
            var na: T = 0;
            var nb: T = 0;
            for (a, b) |av, bv| {
                dot += av * bv;
                na += av * av;
                nb += bv * bv;
            }
            if (na == 0 or nb == 0) return 1;
            return 1 - (dot / (std.math.sqrt(na) * std.math.sqrt(nb)));
        }
    }.f;
}

test "SIMD distance - squared euclidean f32 various dims" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 1, 4, 8, 16, 100, 128, 768 };
    const simd_fn = zvdb.distance.distanceFn(f32, .squared_euclidean);
    const scalar_fn = scalarSquaredEuclidean(f32);

    for (dims) |dim| {
        const a = try allocator.alloc(f32, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, dim);
        defer allocator.free(b);

        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.2 + 0.5;

        const simd_result = simd_fn(a, b);
        const scalar_result = scalar_fn(a, b);
        // Use relative tolerance for larger results
        const relative_error = @abs(simd_result - scalar_result) / @max(scalar_result, 1.0);
        try testing.expect(relative_error < 1e-5);
    }
}

test "SIMD distance - squared euclidean f64 various dims" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 1, 4, 8, 16, 100, 128 };
    const simd_fn = zvdb.distance.distanceFn(f64, .squared_euclidean);
    const scalar_fn = scalarSquaredEuclidean(f64);

    for (dims) |dim| {
        const a = try allocator.alloc(f64, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f64, dim);
        defer allocator.free(b);

        for (a, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * 0.1;
        for (b, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i)) * 0.2 + 0.5;

        const simd_result = simd_fn(a, b);
        const scalar_result = scalar_fn(a, b);
        try testing.expectApproxEqAbs(scalar_result, simd_result, 1e-10);
    }
}

test "SIMD distance - cosine f32 various dims" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 4, 8, 16, 100, 128 };
    const simd_fn = zvdb.distance.distanceFn(f32, .cosine);
    const scalar_fn = scalarCosine(f32);

    for (dims) |dim| {
        const a = try allocator.alloc(f32, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, dim);
        defer allocator.free(b);

        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.2;

        const simd_result = simd_fn(a, b);
        const scalar_result = scalar_fn(a, b);
        try testing.expectApproxEqAbs(scalar_result, simd_result, 1e-5);
    }
}

test "SIMD distance - dot product f32 various dims" {
    const allocator = testing.allocator;
    const dims = [_]usize{ 1, 4, 8, 16, 100, 128 };
    const simd_fn = zvdb.distance.distanceFn(f32, .dot_product);
    const scalar_fn = scalarDot(f32);

    for (dims) |dim| {
        const a = try allocator.alloc(f32, dim);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, dim);
        defer allocator.free(b);

        for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));
        for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 2;

        const simd_result = simd_fn(a, b);
        const scalar_result = -scalar_fn(a, b); // dot_product returns negative
        try testing.expectApproxEqAbs(scalar_result, simd_result, 1e-3);
    }
}

// =============================================================================
// Metadata Tests
// =============================================================================

test "metadata FixedOf generation" {
    const Fixed = zvdb.metadata.FixedOf(TestMeta);
    // Should have: path_off, path_len, line_start, line_end
    try testing.expect(@sizeOf(Fixed) == 16); // 4 * u32
}

test "metadata void type" {
    const Fixed = zvdb.metadata.FixedOf(void);
    try testing.expect(@sizeOf(Fixed) == 0);
}

test "metadata encode and decode" {
    const allocator = testing.allocator;
    var st = zvdb.metadata.StringTable{};
    defer if (st.data.len > 0) allocator.free(st.data);

    const meta = TestMeta{
        .path = "src/main.zig",
        .line_start = 10,
        .line_end = 20,
    };

    const fixed = try zvdb.metadata.encode(TestMeta, meta, &st, allocator);
    try testing.expect(fixed.path_len == 12);
    try testing.expect(fixed.line_start == 10);
    try testing.expect(fixed.line_end == 20);

    // Decode it back
    const decoded = zvdb.metadata.decode(TestMeta, fixed, &st);
    try testing.expectEqualStrings("src/main.zig", decoded.path);
    try testing.expect(decoded.line_start == 10);
    try testing.expect(decoded.line_end == 20);
}

// =============================================================================
// HNSW Core Tests
// =============================================================================

test "HNSW basic insert and search with metadata" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try hnsw.insert(&[_]f32{ 4, 5, 6 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });
    _ = try hnsw.insert(&[_]f32{ 1, 2, 4 }, .{ .path = "c.zig", .line_start = 5, .line_end = 15 });

    const results = try hnsw.searchTopK(&[_]f32{ 1, 2, 3 }, 2, 50);
    defer allocator.free(results);

    try testing.expect(results.len == 2);
    try testing.expect(results[0].id == 0); // Exact match should be first
    try testing.expectApproxEqAbs(@as(f32, 0.0), results[0].distance, 1e-6); // Distance should be 0
}

test "HNSW void metadata" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .cosine, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, {});

    const results = try hnsw.searchTopK(&[_]f32{ 1, 0, 0 }, 1, 50);
    defer allocator.free(results);

    try testing.expect(results.len == 1);
}

test "HNSW - Empty Index" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const query = &[_]f32{ 1, 2, 3 };
    const results = try hnsw.searchTopK(query, 5, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 0), results.len);
}

test "HNSW - Single Point" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const point = &[_]f32{ 1, 2, 3 };
    _ = try hnsw.insert(point, {});

    const results = try hnsw.searchTopK(point, 1, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(usize, 0), results[0].id);
}

test "HNSW - Large Dataset" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 128, 16, 200);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert many points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        _ = try hnsw.insert(point, {});
    }

    // Search for nearest neighbors
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const k = 10;
    const results = try hnsw.searchTopK(query, k, 100);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, k), results.len);

    // Check if results are sorted by distance
    var last_dist: f32 = 0;
    for (results) |result| {
        try testing.expect(result.distance >= last_dist);
        last_dist = result.distance;
    }
}

test "HNSW - Edge Cases" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    // Insert duplicate points
    const point = &[_]f32{ 1, 2, 3 };
    _ = try hnsw.insert(point, {});
    _ = try hnsw.insert(point, {});

    const results = try hnsw.searchTopK(point, 2, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);

    // Search with k larger than number of points
    const large_k_results = try hnsw.searchTopK(point, 100, 50);
    defer allocator.free(large_k_results);

    try testing.expectEqual(@as(usize, 2), large_k_results.len);
}

// =============================================================================
// Batch Insert Tests
// =============================================================================

test "HNSW insertBatch basic" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const points = [_][]const f32{
        &[_]f32{ 1, 2, 3 },
        &[_]f32{ 4, 5, 6 },
        &[_]f32{ 7, 8, 9 },
    };
    const metas = [_]TestMeta{
        .{ .path = "a.zig", .line_start = 1, .line_end = 10 },
        .{ .path = "b.zig", .line_start = 20, .line_end = 30 },
        .{ .path = "c.zig", .line_start = 40, .line_end = 50 },
    };

    const ids = try hnsw.insertBatch(&points, &metas);
    defer allocator.free(ids);

    // Verify IDs are 0, 1, 2
    try testing.expectEqual(@as(usize, 3), ids.len);
    try testing.expectEqual(@as(usize, 0), ids[0]);
    try testing.expectEqual(@as(usize, 1), ids[1]);
    try testing.expectEqual(@as(usize, 2), ids[2]);

    // Verify count
    try testing.expectEqual(@as(usize, 3), hnsw.count());

    // Verify search finds all vectors
    const results = try hnsw.searchTopK(&[_]f32{ 1, 2, 3 }, 3, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 3), results.len);
    try testing.expectEqual(@as(usize, 0), results[0].id); // Exact match first
}

test "HNSW insertBatch empty" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const empty_points: []const []const f32 = &.{};
    const empty_metas: []const void = &.{};

    const ids = try hnsw.insertBatch(empty_points, empty_metas);
    // Empty batch returns empty slice (not allocated)
    try testing.expectEqual(@as(usize, 0), ids.len);
    try testing.expectEqual(@as(usize, 0), hnsw.count());
}

test "HNSW insertBatch large" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 64, 16, 200);
    defer hnsw.deinit();

    const n = 1000;
    var points_list: [n][]f32 = undefined;
    var metas_list: [n]void = undefined;

    // Allocate and initialize points
    for (&points_list, 0..) |*p, i| {
        p.* = try allocator.alloc(f32, 64);
        for (p.*) |*v| v.* = @as(f32, @floatFromInt(i)) * 0.001 + std.crypto.random.float(f32);
        metas_list[i] = {};
    }
    defer for (&points_list) |p| allocator.free(p);

    // Cast to const slice
    var const_points: [n][]const f32 = undefined;
    for (&points_list, 0..) |p, i| const_points[i] = p;

    const ids = try hnsw.insertBatch(&const_points, &metas_list);
    defer allocator.free(ids);

    // Verify count
    try testing.expectEqual(@as(usize, n), ids.len);
    try testing.expectEqual(@as(usize, n), hnsw.count());

    // Search and verify results are sorted
    const results = try hnsw.searchTopK(points_list[0], 10, 100);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 10), results.len);

    // Results should be sorted by distance
    var last_dist: f32 = 0;
    for (results) |r| {
        try testing.expect(r.distance >= last_dist);
        last_dist = r.distance;
    }
}

test "HNSW insertBatch with void metadata" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .cosine, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const points = [_][]const f32{
        &[_]f32{ 1, 0, 0 },
        &[_]f32{ 0, 1, 0 },
        &[_]f32{ 0, 0, 1 },
    };
    const metas = [_]void{ {}, {}, {} };

    const ids = try hnsw.insertBatch(&points, &metas);
    defer allocator.free(ids);

    try testing.expectEqual(@as(usize, 3), ids.len);
    try testing.expectEqual(@as(usize, 3), hnsw.count());
}

test "HNSW insertBatch metadata preserved" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const points = [_][]const f32{
        &[_]f32{ 1, 2, 3 },
        &[_]f32{ 4, 5, 6 },
    };
    const metas = [_]TestMeta{
        .{ .path = "first/path.zig", .line_start = 100, .line_end = 200 },
        .{ .path = "second/path.zig", .line_start = 300, .line_end = 400 },
    };

    const ids = try hnsw.insertBatch(&points, &metas);
    defer allocator.free(ids);

    // Verify metadata is correctly preserved
    const meta0 = hnsw.getMetadata(0).?;
    try testing.expectEqualStrings("first/path.zig", meta0.path);
    try testing.expectEqual(@as(u32, 100), meta0.line_start);
    try testing.expectEqual(@as(u32, 200), meta0.line_end);

    const meta1 = hnsw.getMetadata(1).?;
    try testing.expectEqualStrings("second/path.zig", meta1.path);
    try testing.expectEqual(@as(u32, 300), meta1.line_start);
    try testing.expectEqual(@as(u32, 400), meta1.line_end);
}

// =============================================================================
// Mutation Tests (delete, update, compact)
// =============================================================================

test "HNSW delete and search skips deleted" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    // Insert two nodes
    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, {});

    try testing.expectEqual(@as(usize, 2), hnsw.count());
    try testing.expectEqual(@as(usize, 2), hnsw.liveCount());

    // Delete first node
    try hnsw.delete(0);

    // Count unchanged (tombstoned), but liveCount decreased
    try testing.expectEqual(@as(usize, 2), hnsw.count());
    try testing.expectEqual(@as(usize, 1), hnsw.liveCount());

    // Search should only find second node
    const results = try hnsw.searchTopK(&[_]f32{ 0, 1, 0 }, 2, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(usize, 1), results[0].id);
}

test "HNSW delete already deleted returns error" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    try hnsw.delete(0);

    // Try to delete again - should error
    const result = hnsw.delete(0);
    try testing.expectError(HNSW(f32, .squared_euclidean, void).Error.NodeDeleted, result);
}

test "HNSW delete non-existent returns error" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const result = hnsw.delete(999);
    try testing.expectError(HNSW(f32, .squared_euclidean, void).Error.NodeNotFound, result);
}

test "HNSW updateReplace returns new ID" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const id0 = try hnsw.insert(&[_]f32{ 1, 0, 0 }, .{ .path = "old.zig", .line_start = 1, .line_end = 10 });
    try testing.expectEqual(@as(usize, 0), id0);

    // Update replace - should get new ID
    const id1 = try hnsw.updateReplace(0, &[_]f32{ 0, 1, 0 }, .{ .path = "new.zig", .line_start = 20, .line_end = 30 });
    try testing.expectEqual(@as(usize, 1), id1);

    // Old ID should be deleted
    try testing.expectEqual(@as(usize, 1), hnsw.liveCount());

    // Search should find new node
    const results = try hnsw.searchTopK(&[_]f32{ 0, 1, 0 }, 1, 50);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(usize, 1), results[0].id);

    // New metadata preserved
    const meta = hnsw.getMetadata(1).?;
    try testing.expectEqualStrings("new.zig", meta.path);
}

test "HNSW updateInPlace preserves ID" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, .{ .path = "old.zig", .line_start = 1, .line_end = 10 });

    // Update in place - same ID
    try hnsw.updateInPlace(0, &[_]f32{ 0, 1, 0 }, .{ .path = "updated.zig", .line_start = 100, .line_end = 200 });

    // Count unchanged
    try testing.expectEqual(@as(usize, 1), hnsw.count());
    try testing.expectEqual(@as(usize, 1), hnsw.liveCount());

    // New data in place
    const meta = hnsw.getMetadata(0).?;
    try testing.expectEqualStrings("updated.zig", meta.path);
    try testing.expectEqual(@as(u32, 100), meta.line_start);
}

test "HNSW updateInPlace on deleted returns error" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    try hnsw.delete(0);

    // Try updateInPlace on deleted node
    const result = hnsw.updateInPlace(0, &[_]f32{ 0, 1, 0 }, {});
    try testing.expectError(HNSW(f32, .squared_euclidean, void).Error.NodeDeleted, result);
}

test "HNSW compact remaps IDs" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    // Insert 3 nodes
    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 0, 1 }, {});

    // Delete middle node
    try hnsw.delete(1);

    try testing.expectEqual(@as(usize, 3), hnsw.count());
    try testing.expectEqual(@as(usize, 2), hnsw.liveCount());

    // Compact
    try hnsw.compact();

    // Now should have 2 nodes with IDs 0 and 1
    try testing.expectEqual(@as(usize, 2), hnsw.count());
    try testing.expectEqual(@as(usize, 2), hnsw.liveCount());

    // Search should work correctly
    const results = try hnsw.searchTopK(&[_]f32{ 1, 0, 0 }, 2, 50);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 2), results.len);
}

test "HNSW compact with no deletions is no-op" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, {});

    const count_before = hnsw.count();
    try hnsw.compact();
    const count_after = hnsw.count();

    try testing.expectEqual(count_before, count_after);
}

test "HNSW save after delete compacts automatically" {
    const allocator = testing.allocator;
    const HNSWType = HNSW(f32, .squared_euclidean, TestMeta);

    var hnsw = HNSWType.init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });
    _ = try hnsw.insert(&[_]f32{ 0, 0, 1 }, .{ .path = "c.zig", .line_start = 40, .line_end = 50 });

    // Delete middle node
    try hnsw.delete(1);

    // Save - should compact automatically
    const path = "test_delete_roundtrip.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};
    try hnsw.save(path);

    // Load and verify compacted
    var loaded = try HNSWType.load(allocator, path);
    defer loaded.deinit();

    // Should have 2 nodes (compacted)
    try testing.expectEqual(@as(usize, 2), loaded.count());

    // Search works
    const results = try loaded.searchTopK(&[_]f32{ 1, 0, 0 }, 2, 50);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 2), results.len);
}

test "HNSW filtered search skips deleted" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    // Insert some points with metadata
    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });
    _ = try hnsw.insert(&[_]f32{ 0, 0, 1 }, .{ .path = "c.zig", .line_start = 40, .line_end = 50 });

    // Delete one node
    try hnsw.delete(1);

    const MetaFixed = HNSW(f32, .squared_euclidean, TestMeta).MetaFixed;
    const st = hnsw.getStringTable();

    // Filter predicate: all .zig files
    const FilterCtx = struct {
        st: *const zvdb.metadata.StringTable,
    };
    const pred = struct {
        fn f(ctx: FilterCtx, mf: MetaFixed) bool {
            const path = ctx.st.slice(mf.path_off, mf.path_len);
            return std.mem.endsWith(u8, path, ".zig");
        }
    }.f;

    const results = try hnsw.searchFiltered(
        &[_]f32{ 0, 0, 0 },
        3,
        50,
        FilterCtx{ .st = st },
        pred,
    );
    defer allocator.free(results);

    // Should only return non-deleted nodes (0 and 2)
    try testing.expectEqual(@as(usize, 2), results.len);
    for (results) |r| {
        try testing.expect(r.id != 1); // Deleted node should not appear
    }
}

test "HNSW - Memory Leaks" {
    var hnsw: HNSW(f32, .squared_euclidean, void) = undefined;
    {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 64, 16, 200);

        const num_points = 1000;
        const dim = 64;

        for (0..num_points) |_| {
            const point = try randomPoint(allocator, dim);
            _ = try hnsw.insert(point, {});
            // Intentionally not freeing 'point' to test if HNSW properly manages memory
        }

        const query = try randomPoint(allocator, dim);
        const results = try hnsw.searchTopK(query, 10, 50);
        _ = results;
        // Intentionally not freeing 'results' or 'query'
    }
    // The ArenaAllocator will detect any memory leaks when it's deinitialized
}

test "HNSW - Concurrent Access" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 128, 16, 200);
    defer hnsw.deinit();

    const num_threads = 8;
    const points_per_thread = 1000;
    const dim = 128;

    const ThreadContext = struct {
        hnsw: *HNSW(f32, .squared_euclidean, void),
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
                _ = try ctx.hnsw.insert(point, {});
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
    const actual_count = hnsw.count();
    try testing.expectEqual(expected_count, actual_count);

    // Test search after concurrent insertion
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const results = try hnsw.searchTopK(query, 10, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 10), results.len);
}

test "HNSW - Different Data Types" {
    const allocator = testing.allocator;

    // Test with integer type
    {
        var hnsw_int = HNSW(i32, .squared_euclidean, void).init(allocator, 3, 16, 200);
        defer hnsw_int.deinit();

        _ = try hnsw_int.insert(&[_]i32{ 1, 2, 3 }, {});
        _ = try hnsw_int.insert(&[_]i32{ 4, 5, 6 }, {});
        _ = try hnsw_int.insert(&[_]i32{ 7, 8, 9 }, {});

        const query_int = &[_]i32{ 3, 4, 5 };
        const results_int = try hnsw_int.searchTopK(query_int, 2, 50);
        defer allocator.free(results_int);

        try testing.expectEqual(@as(usize, 2), results_int.len);
    }

    // Test with float64 type
    {
        var hnsw_f64 = HNSW(f64, .squared_euclidean, void).init(allocator, 3, 16, 200);
        defer hnsw_f64.deinit();

        _ = try hnsw_f64.insert(&[_]f64{ 1.1, 2.2, 3.3 }, {});
        _ = try hnsw_f64.insert(&[_]f64{ 4.4, 5.5, 6.6 }, {});
        _ = try hnsw_f64.insert(&[_]f64{ 7.7, 8.8, 9.9 }, {});

        const query_f64 = &[_]f64{ 3.3, 4.4, 5.5 };
        const results_f64 = try hnsw_f64.searchTopK(query_f64, 2, 50);
        defer allocator.free(results_f64);

        try testing.expectEqual(@as(usize, 2), results_f64.len);
    }
}

test "HNSW - Consistency" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, 128, 16, 200);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        _ = try hnsw.insert(point, {});
    }

    // Perform multiple searches with the same query
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const num_searches = 10;
    const k = 10;
    var first_result_ids: [k]usize = undefined;

    for (0..num_searches) |i| {
        const results = try hnsw.searchTopK(query, k, 100);
        defer allocator.free(results);

        if (i == 0) {
            // Store the first result for comparison
            for (results, 0..) |result, j| {
                first_result_ids[j] = result.id;
            }
        } else {
            // Compare with the first result
            for (results, 0..) |result, j| {
                try testing.expectEqual(first_result_ids[j], result.id);
            }
        }
    }
}

// =============================================================================
// Persistence Tests
// =============================================================================

test "persistence header round-trip" {
    const header = zvdb.persistence.FileHeader{
        .metric = @intFromEnum(zvdb.DistanceMetric.squared_euclidean),
        .t_size = 4,
        .dims = 128,
        .m = 16,
        .ef_construction = 200,
        .node_count = 1000,
        .entry_point = 0,
        .max_level = 5,
        .string_blob_len = 4096,
    };

    const path = "test_header.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};

    try zvdb.persistence.save(path, header, &.{}, &.{}, &.{}, &.{});
    const loaded = try zvdb.persistence.loadHeader(path);

    try testing.expect(loaded.dims == 128);
    try testing.expect(loaded.m == 16);
    try testing.expect(loaded.node_count == 1000);
}

// =============================================================================
// Filtered Search Tests
// =============================================================================

test "HNSW - Filtered search" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    // Insert some points with metadata
    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, .{ .path = "b.py", .line_start = 20, .line_end = 30 });
    _ = try hnsw.insert(&[_]f32{ 0, 0, 1 }, .{ .path = "c.zig", .line_start = 5, .line_end = 15 });
    _ = try hnsw.insert(&[_]f32{ 1, 1, 0 }, .{ .path = "d.py", .line_start = 40, .line_end = 50 });
    _ = try hnsw.insert(&[_]f32{ 0, 1, 1 }, .{ .path = "e.zig", .line_start = 60, .line_end = 70 });

    const MetaFixed = HNSW(f32, .squared_euclidean, TestMeta).MetaFixed;
    const st = hnsw.getStringTable();

    // Filter predicate: only .zig files
    const FilterCtx = struct {
        st: *const zvdb.metadata.StringTable,
    };
    const pred = struct {
        fn f(ctx: FilterCtx, mf: MetaFixed) bool {
            const path = ctx.st.slice(mf.path_off, mf.path_len);
            return std.mem.endsWith(u8, path, ".zig");
        }
    }.f;

    const results = try hnsw.searchFiltered(
        &[_]f32{ 1, 0, 0 },
        3,
        50,
        FilterCtx{ .st = st },
        pred,
    );
    defer allocator.free(results);

    // Should only return .zig files (ids 0, 2, 4)
    try testing.expect(results.len == 3);
    for (results) |r| {
        const meta = hnsw.getMetadata(r.id).?;
        try testing.expect(std.mem.endsWith(u8, meta.path, ".zig"));
    }
}

// =============================================================================
// Metadata Retrieval Test
// =============================================================================

test "HNSW - getMetadata" {
    const allocator = testing.allocator;
    var hnsw = HNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const id = try hnsw.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "test/file.zig", .line_start = 100, .line_end = 200 });

    const meta = hnsw.getMetadata(id).?;
    try testing.expectEqualStrings("test/file.zig", meta.path);
    try testing.expect(meta.line_start == 100);
    try testing.expect(meta.line_end == 200);
}

// =============================================================================
// Save/Load Round-Trip Tests
// =============================================================================

test "HNSW full save/load round-trip" {
    const allocator = testing.allocator;
    const HNSWType = HNSW(f32, .squared_euclidean, TestMeta);

    // Create and populate original index
    var original = HNSWType.init(allocator, 3, 16, 200);
    defer original.deinit();

    _ = try original.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try original.insert(&[_]f32{ 4, 5, 6 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });
    _ = try original.insert(&[_]f32{ 7, 8, 9 }, .{ .path = "c.zig", .line_start = 40, .line_end = 50 });

    // Save to file
    const path = "test_roundtrip.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};
    try original.save(path);

    // Load into new instance
    var loaded = try HNSWType.load(allocator, path);
    defer loaded.deinit();

    // Verify structure matches
    try testing.expectEqual(original.count(), loaded.count());
    try testing.expectEqual(original.dims, loaded.dims);
    try testing.expectEqual(original.m, loaded.m);
    try testing.expectEqual(original.ef_construction, loaded.ef_construction);
    try testing.expectEqual(original.max_level, loaded.max_level);
    try testing.expectEqual(original.entry_point, loaded.entry_point);

    // Verify search results match
    const query = &[_]f32{ 1, 2, 3 };
    const orig_results = try original.searchTopK(query, 3, 50);
    defer allocator.free(orig_results);
    const load_results = try loaded.searchTopK(query, 3, 50);
    defer allocator.free(load_results);

    try testing.expectEqual(orig_results.len, load_results.len);
    for (orig_results, load_results) |o, l| {
        try testing.expectEqual(o.id, l.id);
        try testing.expectApproxEqAbs(o.distance, l.distance, 1e-6);
    }

    // Verify metadata preserved
    const orig_meta = original.getMetadata(0).?;
    const load_meta = loaded.getMetadata(0).?;
    try testing.expectEqualStrings(orig_meta.path, load_meta.path);
    try testing.expectEqual(orig_meta.line_start, load_meta.line_start);
    try testing.expectEqual(orig_meta.line_end, load_meta.line_end);
}

test "HNSW save/load with void metadata" {
    const allocator = testing.allocator;
    const HNSWType = HNSW(f32, .cosine, void);

    var original = HNSWType.init(allocator, 3, 16, 200);
    defer original.deinit();

    _ = try original.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try original.insert(&[_]f32{ 0, 1, 0 }, {});

    const path = "test_void_roundtrip.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};
    try original.save(path);

    var loaded = try HNSWType.load(allocator, path);
    defer loaded.deinit();

    try testing.expectEqual(original.count(), loaded.count());

    const query = &[_]f32{ 1, 0, 0 };
    const orig_results = try original.searchTopK(query, 2, 50);
    defer allocator.free(orig_results);
    const load_results = try loaded.searchTopK(query, 2, 50);
    defer allocator.free(load_results);

    try testing.expectEqual(orig_results.len, load_results.len);
    for (orig_results, load_results) |o, l| {
        try testing.expectEqual(o.id, l.id);
    }
}

test "HNSW save/load empty index" {
    const allocator = testing.allocator;
    const HNSWType = HNSW(f32, .squared_euclidean, void);

    var original = HNSWType.init(allocator, 3, 16, 200);
    defer original.deinit();

    const path = "test_empty_roundtrip.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};
    try original.save(path);

    var loaded = try HNSWType.load(allocator, path);
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 0), loaded.count());
    try testing.expectEqual(@as(?usize, null), loaded.entry_point);
}
