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
// HNSW Integration Test with Extended Metadata (Phase 4)
// =============================================================================

test "HNSW with extended metadata - optionals, arrays, nested structs" {
    const allocator = testing.allocator;

    // Complex metadata with all supported extended types
    const Location = struct { line: u32, col: u32 };
    const ExtendedMeta = struct {
        name: []const u8, // string
        score: f32, // primitive
        tags: [3]u8, // primitive array
        location: Location, // nested struct
        description: ?[]const u8, // optional string
        priority: ?u8, // optional primitive
    };

    var hnsw = HNSW(f32, .squared_euclidean, ExtendedMeta).init(allocator, 4, 16, 200);
    defer hnsw.deinit();

    // Insert with full metadata (all optionals present)
    _ = try hnsw.insert(&[_]f32{ 1.0, 0.0, 0.0, 0.0 }, .{
        .name = "first",
        .score = 0.9,
        .tags = .{ 1, 2, 3 },
        .location = .{ .line = 10, .col = 5 },
        .description = "First item description",
        .priority = 1,
    });

    // Insert with some nulls
    _ = try hnsw.insert(&[_]f32{ 0.0, 1.0, 0.0, 0.0 }, .{
        .name = "second",
        .score = 0.8,
        .tags = .{ 4, 5, 6 },
        .location = .{ .line = 20, .col = 10 },
        .description = null,
        .priority = 2,
    });

    // Insert with all optionals null
    _ = try hnsw.insert(&[_]f32{ 0.0, 0.0, 1.0, 0.0 }, .{
        .name = "third",
        .score = 0.7,
        .tags = .{ 7, 8, 9 },
        .location = .{ .line = 30, .col = 15 },
        .description = null,
        .priority = null,
    });

    // Search and verify metadata retrieval
    const results = try hnsw.searchTopK(&[_]f32{ 1.0, 0.0, 0.0, 0.0 }, 3, 50);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 3), results.len);
    try testing.expectEqual(@as(usize, 0), results[0].id); // First should be closest

    // Retrieve and verify metadata for first result
    const meta0 = hnsw.getMetadata(0).?;
    try testing.expectEqualStrings("first", meta0.name);
    try testing.expectApproxEqAbs(@as(f32, 0.9), meta0.score, 0.001);
    try testing.expectEqual(@as(u8, 1), meta0.tags[0]);
    try testing.expectEqual(@as(u32, 10), meta0.location.line);
    try testing.expect(meta0.description != null);
    try testing.expectEqualStrings("First item description", meta0.description.?);
    try testing.expectEqual(@as(u8, 1), meta0.priority.?);

    // Verify second result (some nulls)
    const meta1 = hnsw.getMetadata(1).?;
    try testing.expectEqualStrings("second", meta1.name);
    try testing.expect(meta1.description == null);
    try testing.expectEqual(@as(u8, 2), meta1.priority.?);

    // Verify third result (all optionals null)
    const meta2 = hnsw.getMetadata(2).?;
    try testing.expectEqualStrings("third", meta2.name);
    try testing.expect(meta2.description == null);
    try testing.expect(meta2.priority == null);
}

test "HNSW batch insert with extended metadata" {
    const allocator = testing.allocator;

    const Meta = struct {
        label: []const u8,
        value: ?i32,
    };

    var hnsw = HNSW(f32, .squared_euclidean, Meta).init(allocator, 2, 16, 200);
    defer hnsw.deinit();

    const points = [_][]const f32{
        &[_]f32{ 1.0, 0.0 },
        &[_]f32{ 0.0, 1.0 },
        &[_]f32{ 1.0, 1.0 },
    };
    const metas = [_]Meta{
        .{ .label = "a", .value = 100 },
        .{ .label = "b", .value = null },
        .{ .label = "c", .value = 300 },
    };

    const ids = try hnsw.insertBatch(&points, &metas);
    defer allocator.free(ids);

    try testing.expectEqual(@as(usize, 3), ids.len);

    // Verify metadata
    const m0 = hnsw.getMetadata(ids[0]).?;
    try testing.expectEqualStrings("a", m0.label);
    try testing.expectEqual(@as(i32, 100), m0.value.?);

    const m1 = hnsw.getMetadata(ids[1]).?;
    try testing.expectEqualStrings("b", m1.label);
    try testing.expect(m1.value == null);
}

test "HNSW save/load with extended metadata" {
    const allocator = testing.allocator;
    const std_fs = @import("std").fs;

    const Meta = struct {
        file: []const u8,
        loc: struct { start: u32, end: u32 },
        opt_rank: ?u16,
        tags: [3]u8,
    };

    const path = "extended_meta_test.zvdb";
    defer std_fs.cwd().deleteFile(path) catch {};

    // Create and populate index
    {
        var g = HNSW(f32, .squared_euclidean, Meta).init(allocator, 3, 8, 50);
        defer g.deinit();

        _ = try g.insert(&[_]f32{ 1, 2, 3 }, .{
            .file = "main.zig",
            .loc = .{ .start = 1, .end = 10 },
            .opt_rank = 7,
            .tags = .{ 1, 2, 3 },
        });
        _ = try g.insert(&[_]f32{ 1, 2, 4 }, .{
            .file = "lib.zig",
            .loc = .{ .start = 20, .end = 30 },
            .opt_rank = null,
            .tags = .{ 4, 5, 6 },
        });

        // Verify before save
        const m0 = g.getMetadata(0).?;
        try testing.expectEqualStrings("main.zig", m0.file);
        try testing.expect(m0.opt_rank != null);
        try testing.expectEqual(@as(u16, 7), m0.opt_rank.?);

        try g.save(path);
    }

    // Load and verify
    var loaded = try HNSW(f32, .squared_euclidean, Meta).load(allocator, path);
    defer loaded.deinit();

    try testing.expectEqual(@as(usize, 2), loaded.count());

    // Verify first record (with optional present)
    const lm0 = loaded.getMetadata(0).?;
    try testing.expectEqualStrings("main.zig", lm0.file);
    try testing.expectEqual(@as(u32, 1), lm0.loc.start);
    try testing.expectEqual(@as(u32, 10), lm0.loc.end);
    try testing.expect(lm0.opt_rank != null);
    try testing.expectEqual(@as(u16, 7), lm0.opt_rank.?);
    try testing.expectEqual(@as(u8, 1), lm0.tags[0]);

    // Verify second record (with optional null)
    const lm1 = loaded.getMetadata(1).?;
    try testing.expectEqualStrings("lib.zig", lm1.file);
    try testing.expectEqual(@as(u32, 20), lm1.loc.start);
    try testing.expect(lm1.opt_rank == null);
    try testing.expectEqual(@as(u8, 5), lm1.tags[1]);

    // Verify search still works after load
    const results = try loaded.searchTopK(&[_]f32{ 1, 2, 3 }, 2, 50);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 2), results.len);
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
