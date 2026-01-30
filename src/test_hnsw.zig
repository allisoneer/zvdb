const std = @import("std");
const testing = std.testing;
const zvdb = @import("zvdb.zig");

// Legacy HNSW for backward compatibility tests
const legacy_hnsw = zvdb.legacy.hnsw;
const LegacyHNSW = legacy_hnsw.HNSW;

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
// Extended Metadata FixedOf Tests (Phase 2)
// =============================================================================

test "metadata FixedOf - optionals add presence_words" {
    const Meta = struct { a: ?u32, b: []const u8 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    // presence_words: [1]u64 (8) + a: u32 (4) + b_off: u32 (4) + b_len: u32 (4) = 20
    try testing.expect(@hasField(Fixed, "presence_words"));
    try testing.expect(@hasField(Fixed, "a"));
    try testing.expect(@hasField(Fixed, "b_off"));
    try testing.expect(@hasField(Fixed, "b_len"));
    try testing.expectEqual(@as(usize, 24), @sizeOf(Fixed)); // 8 + 4 + 4 + 4 + 4 (padding to 8)
}

test "metadata FixedOf - no optionals no presence_words" {
    const Meta = struct { x: u32, y: f32 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    // x: u32 (4) + y: f32 (4) = 8
    try testing.expect(!@hasField(Fixed, "presence_words"));
    try testing.expect(@hasField(Fixed, "x"));
    try testing.expect(@hasField(Fixed, "y"));
    try testing.expectEqual(@as(usize, 8), @sizeOf(Fixed));
}

test "metadata FixedOf - primitive array inline" {
    const Meta = struct { arr: [4]f32, tag: u8 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    // arr: [4]f32 (16) + tag: u8 (1) + padding = 20 (extern alignment)
    try testing.expect(@hasField(Fixed, "arr"));
    try testing.expect(@hasField(Fixed, "tag"));
}

test "metadata FixedOf - nested struct flattening" {
    const Inner = struct { x: u32, y: u32 };
    const Meta = struct { loc: Inner, name: []const u8 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    try testing.expect(@hasField(Fixed, "loc__x"));
    try testing.expect(@hasField(Fixed, "loc__y"));
    try testing.expect(@hasField(Fixed, "name_off"));
    try testing.expect(@hasField(Fixed, "name_len"));
}

test "metadata FixedOf - two level nesting" {
    const Deep = struct { val: i16 };
    const Mid = struct { d: Deep, flag: bool };
    const Meta = struct { m: Mid };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    try testing.expect(@hasField(Fixed, "m__d__val"));
    try testing.expect(@hasField(Fixed, "m__flag"));
}

test "metadata FixedOf - optional nested struct" {
    const Inner = struct { a: u32, b: u32 };
    const Meta = struct { data: ?Inner };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    // Should have presence_words since there's an optional
    try testing.expect(@hasField(Fixed, "presence_words"));
    try testing.expect(@hasField(Fixed, "data__a"));
    try testing.expect(@hasField(Fixed, "data__b"));
}

test "metadata FixedOf - optional array" {
    const Meta = struct { arr: ?[3]i16 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    try testing.expect(@hasField(Fixed, "presence_words"));
    try testing.expect(@hasField(Fixed, "arr"));
}

test "metadata FixedOf - enum with explicit tag" {
    const Status = enum(u8) { Active, Inactive, Pending };
    const Meta = struct { status: Status, value: u32 };
    const Fixed = zvdb.metadata.FixedOf(Meta);
    try testing.expect(@hasField(Fixed, "status"));
    try testing.expect(@hasField(Fixed, "value"));
}

// =============================================================================
// Extended Metadata Encode/Decode Tests (Phase 3)
// =============================================================================

test "metadata encode/decode - optional present and null" {
    const Meta = struct { a: ?u32, b: u32 };
    var st = zvdb.metadata.StringTable{};

    // Present
    const m1: Meta = .{ .a = 42, .b = 10 };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expect(r1.a != null);
    try testing.expectEqual(@as(u32, 42), r1.a.?);

    // Null
    const m2: Meta = .{ .a = null, .b = 20 };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.a == null);
    try testing.expectEqual(@as(u32, 20), r2.b);
}

test "metadata encode/decode - optional string" {
    const Meta = struct { name: ?[]const u8, id: u32 };
    var st = zvdb.metadata.StringTable{};
    defer if (st.data.len > 0) testing.allocator.free(st.data);

    const m1: Meta = .{ .name = "hello", .id = 1 };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expect(r1.name != null);
    try testing.expectEqualStrings("hello", r1.name.?);

    const m2: Meta = .{ .name = null, .id = 2 };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.name == null);
}

test "metadata encode/decode - optional array" {
    const Meta = struct { arr: ?[3]i16 };
    var st = zvdb.metadata.StringTable{};

    const m1: Meta = .{ .arr = .{ 1, 2, 3 } };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expect(r1.arr != null);
    try testing.expectEqual(@as(i16, 2), r1.arr.?[1]);

    const m2: Meta = .{ .arr = null };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.arr == null);
}

test "metadata encode/decode - nested struct" {
    const Inner = struct { x: u32, y: u32 };
    const Meta = struct { loc: Inner, tag: u8 };
    var st = zvdb.metadata.StringTable{};

    const m: Meta = .{ .loc = .{ .x = 10, .y = 20 }, .tag = 5 };
    const f = try zvdb.metadata.encode(Meta, m, &st, testing.allocator);
    const r = zvdb.metadata.decode(Meta, f, &st);
    try testing.expectEqual(@as(u32, 10), r.loc.x);
    try testing.expectEqual(@as(u32, 20), r.loc.y);
    try testing.expectEqual(@as(u8, 5), r.tag);
}

test "metadata encode/decode - optional nested struct" {
    const Inner = struct { name: []const u8, val: u16 };
    const Meta = struct { data: ?Inner, score: f32 };
    var st = zvdb.metadata.StringTable{};
    defer if (st.data.len > 0) testing.allocator.free(st.data);

    // Present
    const m1: Meta = .{ .data = .{ .name = "test", .val = 100 }, .score = 1.5 };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expect(r1.data != null);
    try testing.expectEqualStrings("test", r1.data.?.name);

    // Null
    const m2: Meta = .{ .data = null, .score = 2.5 };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.data == null);
}

test "metadata encode/decode - primitive array" {
    const Meta = struct { arr: [4]f32, tag: u8 };
    var st = zvdb.metadata.StringTable{};

    const m: Meta = .{ .arr = .{ 1.0, 2.0, 3.0, 4.0 }, .tag = 42 };
    const f = try zvdb.metadata.encode(Meta, m, &st, testing.allocator);
    const r = zvdb.metadata.decode(Meta, f, &st);
    try testing.expectApproxEqAbs(@as(f32, 1.0), r.arr[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4.0), r.arr[3], 0.001);
    try testing.expectEqual(@as(u8, 42), r.tag);
}

test "metadata encode/decode - multiple optionals" {
    const Meta = struct { a: ?u32, b: ?u32, c: u32 };
    var st = zvdb.metadata.StringTable{};

    // Both present
    const m1: Meta = .{ .a = 1, .b = 2, .c = 3 };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expectEqual(@as(u32, 1), r1.a.?);
    try testing.expectEqual(@as(u32, 2), r1.b.?);

    // First null, second present
    const m2: Meta = .{ .a = null, .b = 99, .c = 3 };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.a == null);
    try testing.expectEqual(@as(u32, 99), r2.b.?);

    // Both null
    const m3: Meta = .{ .a = null, .b = null, .c = 3 };
    const f3 = try zvdb.metadata.encode(Meta, m3, &st, testing.allocator);
    const r3 = zvdb.metadata.decode(Meta, f3, &st);
    try testing.expect(r3.a == null);
    try testing.expect(r3.b == null);
}

test "metadata totalStringBytesForBatch skips null optionals" {
    const Meta = struct { a: ?[]const u8, b: []const u8 };
    const metas = [_]Meta{
        .{ .a = "xx", .b = "bbb" }, // 2 + 3 = 5
        .{ .a = null, .b = "c" }, // 0 + 1 = 1
    };
    const total = zvdb.metadata.totalStringBytesForBatch(Meta, &metas);
    try testing.expectEqual(@as(usize, 6), total);
}

test "metadata encode/decode - nested optional with inner optional" {
    const Inner = struct { val: ?u32 };
    const Meta = struct { data: ?Inner };
    var st = zvdb.metadata.StringTable{};

    // Outer present, inner present
    const m1: Meta = .{ .data = .{ .val = 42 } };
    const f1 = try zvdb.metadata.encode(Meta, m1, &st, testing.allocator);
    const r1 = zvdb.metadata.decode(Meta, f1, &st);
    try testing.expect(r1.data != null);
    try testing.expect(r1.data.?.val != null);
    try testing.expectEqual(@as(u32, 42), r1.data.?.val.?);

    // Outer present, inner null
    const m2: Meta = .{ .data = .{ .val = null } };
    const f2 = try zvdb.metadata.encode(Meta, m2, &st, testing.allocator);
    const r2 = zvdb.metadata.decode(Meta, f2, &st);
    try testing.expect(r2.data != null);
    try testing.expect(r2.data.?.val == null);

    // Outer null
    const m3: Meta = .{ .data = null };
    const f3 = try zvdb.metadata.encode(Meta, m3, &st, testing.allocator);
    const r3 = zvdb.metadata.decode(Meta, f3, &st);
    try testing.expect(r3.data == null);
}

// =============================================================================
// Legacy HNSW Tests (using legacy module for backward compatibility)
// =============================================================================

test "Legacy HNSW basic insert and search" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try hnsw.insert(&[_]f32{ 4, 5, 6 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });
    _ = try hnsw.insert(&[_]f32{ 1, 2, 4 }, .{ .path = "c.zig", .line_start = 5, .line_end = 15 });

    const results = try hnsw.searchTopK(&[_]f32{ 1, 2, 3 }, 2, 50);
    defer allocator.free(results);

    try testing.expect(results.len == 2);
    try testing.expect(results[0].id == 0); // Exact match should be first
}

test "Legacy HNSW void metadata" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .cosine, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    _ = try hnsw.insert(&[_]f32{ 1, 0, 0 }, {});
    _ = try hnsw.insert(&[_]f32{ 0, 1, 0 }, {});

    const results = try hnsw.searchTopK(&[_]f32{ 1, 0, 0 }, 1, 50);
    defer allocator.free(results);

    try testing.expect(results.len == 1);
}

test "Legacy HNSW - Empty Index" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const query = &[_]f32{ 1, 2, 3 };
    const results = try hnsw.searchTopK(query, 5, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 0), results.len);
}

test "Legacy HNSW - Single Point" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const point = &[_]f32{ 1, 2, 3 };
    _ = try hnsw.insert(point, {});

    const results = try hnsw.searchTopK(point, 1, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqual(@as(usize, 0), results[0].id);
}

test "Legacy HNSW delete and search skips deleted" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
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

test "Legacy HNSW compact remaps IDs" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, void).init(allocator, 3, 16, 200);
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

test "Legacy HNSW insertBatch basic" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
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
}

test "Legacy HNSW - getMetadata" {
    const allocator = testing.allocator;
    var hnsw = LegacyHNSW(f32, .squared_euclidean, TestMeta).init(allocator, 3, 16, 200);
    defer hnsw.deinit();

    const id = try hnsw.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "test/file.zig", .line_start = 100, .line_end = 200 });

    const meta = hnsw.getMetadata(id).?;
    try testing.expectEqualStrings("test/file.zig", meta.path);
    try testing.expect(meta.line_start == 100);
    try testing.expect(meta.line_end == 200);
}

test "Legacy HNSW save/load round-trip" {
    const allocator = testing.allocator;
    const HNSWType = LegacyHNSW(f32, .squared_euclidean, TestMeta);

    // Create and populate original index
    var original = HNSWType.init(allocator, 3, 16, 200);
    defer original.deinit();

    _ = try original.insert(&[_]f32{ 1, 2, 3 }, .{ .path = "a.zig", .line_start = 1, .line_end = 10 });
    _ = try original.insert(&[_]f32{ 4, 5, 6 }, .{ .path = "b.zig", .line_start = 20, .line_end = 30 });

    // Save to file
    const path = "test_legacy_roundtrip.zvdb";
    defer std.fs.cwd().deleteFile(path) catch {};
    try original.save(path);

    // Load into new instance
    var loaded = try HNSWType.load(allocator, path);
    defer loaded.deinit();

    // Verify structure matches
    try testing.expectEqual(original.count(), loaded.count());

    // Verify metadata preserved
    const orig_meta = original.getMetadata(0).?;
    const load_meta = loaded.getMetadata(0).?;
    try testing.expectEqualStrings(orig_meta.path, load_meta.path);
    try testing.expectEqual(orig_meta.line_start, load_meta.line_start);
}
