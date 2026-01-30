const std = @import("std");
const testing = std.testing;
const h2 = @import("hnsw_v2.zig");

test "HNSWv2 compact removes deleted nodes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{
        .dims = 3,
        .reuse_deleted_slots = false, // Don't reuse so we can track slots
    });
    defer idx.deinit();

    // Insert 5 points
    const points = [_][3]f32{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
        .{ 1, 1, 0 },
        .{ 0, 1, 1 },
    };
    var ids: [5]u32 = undefined;
    for (points, 0..) |p, i| {
        ids[i] = try idx.insert(&p, {});
    }

    try testing.expectEqual(@as(usize, 5), idx.count());
    try testing.expectEqual(@as(usize, 5), idx.liveCount());

    // Delete node 1 (0,1,0)
    try idx.delete(ids[1]);

    try testing.expectEqual(@as(usize, 5), idx.count()); // Still 5 nodes
    try testing.expectEqual(@as(usize, 4), idx.liveCount()); // But only 4 live

    // Compact
    try idx.compact();

    // After compaction: only 4 nodes remain
    try testing.expectEqual(@as(usize, 4), idx.count());
    try testing.expectEqual(@as(usize, 4), idx.liveCount());
}

test "HNSWv2 compact maintains search correctness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 3 });
    defer idx.deinit();

    // Insert points
    const points = [_][3]f32{
        .{ 1, 0, 0 },
        .{ 0, 1, 0 },
        .{ 0, 0, 1 },
        .{ 1, 1, 0 },
        .{ 0, 1, 1 },
    };
    var ids: [5]u32 = undefined;
    for (points, 0..) |p, i| {
        ids[i] = try idx.insert(&p, {});
    }

    // Delete middle node
    try idx.delete(ids[2]);

    // Search before compact - should find closest
    const query = [_]f32{ 1, 0, 0 };
    const res1 = try idx.searchTopK(&query, 1, 64);
    defer gpa.allocator().free(res1);
    try testing.expectEqual(@as(usize, 1), res1.len);
    // Result should be close to query (original id 0)

    // Compact
    try idx.compact();

    // Search after compact - should still work correctly
    const res2 = try idx.searchTopK(&query, 3, 64);
    defer gpa.allocator().free(res2);
    try testing.expect(res2.len >= 1);
    try testing.expect(res2.len <= 4); // At most 4 nodes left

    // Results should be sorted by distance
    for (0..res2.len - 1) |i| {
        try testing.expect(res2[i].distance <= res2[i + 1].distance);
    }
}

test "HNSWv2 compact with metadata preserves metadata" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const Meta = struct { label: u32 };

    var idx = try h2.HNSWv2(f32, .squared_euclidean, Meta).init(gpa.allocator(), .{ .dims = 2 });
    defer idx.deinit();

    const id0 = try idx.insert(&[_]f32{ 0, 0 }, .{ .label = 100 });
    const id1 = try idx.insert(&[_]f32{ 1, 0 }, .{ .label = 200 });
    const id2 = try idx.insert(&[_]f32{ 0, 1 }, .{ .label = 300 });

    // Delete middle one
    try idx.delete(id1);

    // Compact
    try idx.compact();

    try testing.expectEqual(@as(usize, 2), idx.count());

    // Search for original vectors and verify metadata is preserved
    const res0 = try idx.searchTopK(&[_]f32{ 0, 0 }, 1, 64);
    defer gpa.allocator().free(res0);
    try testing.expectEqual(@as(usize, 1), res0.len);
    const meta0 = idx.getMetadata(res0[0].id).?;
    try testing.expectEqual(@as(u32, 100), meta0.label);

    const res2 = try idx.searchTopK(&[_]f32{ 0, 1 }, 1, 64);
    defer gpa.allocator().free(res2);
    try testing.expectEqual(@as(usize, 1), res2.len);
    const meta2 = idx.getMetadata(res2[0].id).?;
    try testing.expectEqual(@as(u32, 300), meta2.label);

    _ = id0;
    _ = id2;
}

test "HNSWv2 compact all deleted becomes empty" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 2 });
    defer idx.deinit();

    const id0 = try idx.insert(&[_]f32{ 0, 0 }, {});
    const id1 = try idx.insert(&[_]f32{ 1, 1 }, {});

    try idx.delete(id0);
    try idx.delete(id1);

    try idx.compact();

    try testing.expectEqual(@as(usize, 0), idx.count());
    try testing.expectEqual(@as(usize, 0), idx.liveCount());
    try testing.expect(idx.entry_point == null);

    // Search on empty index
    const res = try idx.searchTopK(&[_]f32{ 0, 0 }, 1, 64);
    defer gpa.allocator().free(res);
    try testing.expectEqual(@as(usize, 0), res.len);
}

test "HNSWv2 compact no-op when nothing deleted" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 2 });
    defer idx.deinit();

    _ = try idx.insert(&[_]f32{ 0, 0 }, {});
    _ = try idx.insert(&[_]f32{ 1, 1 }, {});
    _ = try idx.insert(&[_]f32{ 2, 2 }, {});

    const count_before = idx.count();

    // Compact without any deletions
    try idx.compact();

    try testing.expectEqual(count_before, idx.count());
}

test "HNSWv2 compact persists and loads correctly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const path = "test_compact_persist.zvdb2";
    defer std.fs.cwd().deleteFile(path) catch {};

    // Create, insert, delete, compact, save
    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 3 });
        defer idx.deinit();

        _ = try idx.insert(&[_]f32{ 1, 0, 0 }, {});
        const id1 = try idx.insert(&[_]f32{ 0, 1, 0 }, {});
        _ = try idx.insert(&[_]f32{ 0, 0, 1 }, {});

        try idx.delete(id1);
        try idx.compact();

        try testing.expectEqual(@as(usize, 2), idx.count());

        try idx.save(path);
    }

    // Load and verify
    {
        var loaded = try h2.HNSWv2(f32, .squared_euclidean, void).load(gpa.allocator(), path);
        defer loaded.deinit();

        try testing.expectEqual(@as(usize, 2), loaded.count());
        try testing.expectEqual(@as(usize, 2), loaded.liveCount());

        // Search should work
        const res = try loaded.searchTopK(&[_]f32{ 1, 0, 0 }, 1, 64);
        defer gpa.allocator().free(res);
        try testing.expectEqual(@as(usize, 1), res.len);
    }
}
