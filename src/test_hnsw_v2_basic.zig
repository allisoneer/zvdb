const std = @import("std");
const testing = std.testing;
const h2 = @import("hnsw_v2.zig");
const dist = @import("distance.zig");

const Meta = struct { label: u32 };

test "HNSWv2 basic insert/search" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, Meta).init(gpa.allocator(), .{ .dims = 4 });
    defer idx.deinit();

    const id0 = try idx.insert(&[_]f32{ 1, 0, 0, 0 }, .{ .label = 0 });
    const id1 = try idx.insert(&[_]f32{ 0, 1, 0, 0 }, .{ .label = 1 });
    const id2 = try idx.insert(&[_]f32{ 0, 0, 1, 0 }, .{ .label = 2 });

    try testing.expectEqual(@as(u32, 0), id0);
    try testing.expectEqual(@as(u32, 1), id1);
    try testing.expectEqual(@as(u32, 2), id2);
    try testing.expectEqual(@as(usize, 3), idx.count());

    // Search for exact match
    const res = try idx.searchTopK(&[_]f32{ 1, 0, 0, 0 }, 1, 64);
    defer gpa.allocator().free(res);

    try testing.expectEqual(@as(usize, 1), res.len);
    try testing.expectEqual(@as(u32, id0), res[0].id);
    try testing.expectEqual(@as(f32, 0), res[0].distance);
}

test "HNSWv2 search finds closest" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 2 });
    defer idx.deinit();

    // Create a grid of points
    _ = try idx.insert(&[_]f32{ 0, 0 }, {});
    _ = try idx.insert(&[_]f32{ 1, 0 }, {});
    _ = try idx.insert(&[_]f32{ 0, 1 }, {});
    _ = try idx.insert(&[_]f32{ 1, 1 }, {});
    _ = try idx.insert(&[_]f32{ 2, 2 }, {});

    // Query near (0.9, 0.9) - closest should be (1,1)
    const res = try idx.searchTopK(&[_]f32{ 0.9, 0.9 }, 1, 64);
    defer gpa.allocator().free(res);

    try testing.expectEqual(@as(usize, 1), res.len);
    try testing.expectEqual(@as(u32, 3), res[0].id); // (1,1) is id 3
}

test "HNSWv2 delete marks as deleted" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 4 });
    defer idx.deinit();

    const id0 = try idx.insert(&[_]f32{ 1, 0, 0, 0 }, {});
    const id1 = try idx.insert(&[_]f32{ 0, 1, 0, 0 }, {});

    try testing.expectEqual(@as(usize, 2), idx.count());
    try testing.expectEqual(@as(usize, 2), idx.liveCount());

    try idx.delete(id0);

    try testing.expectEqual(@as(usize, 2), idx.count()); // Still 2 nodes
    try testing.expectEqual(@as(usize, 1), idx.liveCount()); // But only 1 live

    // Search should not return deleted node
    const res = try idx.searchTopK(&[_]f32{ 1, 0, 0, 0 }, 2, 64);
    defer gpa.allocator().free(res);

    try testing.expectEqual(@as(usize, 1), res.len);
    try testing.expectEqual(@as(u32, id1), res[0].id);
}

test "HNSWv2 delete with slot reuse" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{
        .dims = 3,
        .reuse_deleted_slots = true,
    });
    defer idx.deinit();

    const a = try idx.insert(&[_]f32{ 1, 2, 3 }, {});
    _ = try idx.insert(&[_]f32{ 4, 5, 6 }, {});
    try idx.delete(a);

    // Next insert should reuse slot 'a'
    const c = try idx.insert(&[_]f32{ 7, 8, 9 }, {});
    try testing.expectEqual(a, c);
}

test "HNSWv2 delete tombstone only (no slot reuse)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{
        .dims = 3,
        .reuse_deleted_slots = false,
    });
    defer idx.deinit();

    const a = try idx.insert(&[_]f32{ 1, 2, 3 }, {});
    _ = try idx.insert(&[_]f32{ 4, 5, 6 }, {});
    try idx.delete(a);

    // Next insert should NOT reuse slot 'a', should get new slot
    const c = try idx.insert(&[_]f32{ 7, 8, 9 }, {});
    try testing.expect(c != a);
    try testing.expectEqual(@as(u32, 2), c); // Should be slot 2
}

test "HNSWv2 metadata retrieval" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, Meta).init(gpa.allocator(), .{ .dims = 2 });
    defer idx.deinit();

    const id0 = try idx.insert(&[_]f32{ 0, 0 }, .{ .label = 42 });
    const id1 = try idx.insert(&[_]f32{ 1, 1 }, .{ .label = 99 });

    const m0 = idx.getMetadata(id0).?;
    const m1 = idx.getMetadata(id1).?;

    try testing.expectEqual(@as(u32, 42), m0.label);
    try testing.expectEqual(@as(u32, 99), m1.label);

    // Deleted node returns null
    try idx.delete(id0);
    try testing.expect(idx.getMetadata(id0) == null);
}

test "HNSWv2 empty index search" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 4 });
    defer idx.deinit();

    const res = try idx.search(&[_]f32{ 1, 0, 0, 0 }, 5);
    defer gpa.allocator().free(res);

    try testing.expectEqual(@as(usize, 0), res.len);
}

test "HNSWv2 many insertions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{
        .dims = 8,
        .m = 16,
        .m_base = 32,
        .ef_construction = 100,
    });
    defer idx.deinit();

    // Insert 100 random-ish points
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..100) |_| {
        var vec: [8]f32 = undefined;
        for (&vec) |*v| v.* = random.float(f32);
        _ = try idx.insert(&vec, {});
    }

    try testing.expectEqual(@as(usize, 100), idx.count());

    // Search should return results
    var query: [8]f32 = undefined;
    for (&query) |*v| v.* = random.float(f32);

    const res = try idx.searchTopK(&query, 10, 64);
    defer gpa.allocator().free(res);

    try testing.expect(res.len > 0);
    try testing.expect(res.len <= 10);

    // Results should be sorted by distance
    for (0..res.len - 1) |i| {
        try testing.expect(res[i].distance <= res[i + 1].distance);
    }
}
