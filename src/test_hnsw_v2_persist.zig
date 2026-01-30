const std = @import("std");
const testing = std.testing;
const h2 = @import("hnsw_v2.zig");

const Meta = struct { label: u32 };

test "HNSWv2 save/load round-trip" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const path = "/tmp/test_hnsw_v2.zvd2";

    // Create and populate index
    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, Meta).init(gpa.allocator(), .{ .dims = 4 });
        defer idx.deinit();

        _ = try idx.insert(&[_]f32{ 1, 0, 0, 0 }, .{ .label = 100 });
        _ = try idx.insert(&[_]f32{ 0, 1, 0, 0 }, .{ .label = 200 });
        _ = try idx.insert(&[_]f32{ 0, 0, 1, 0 }, .{ .label = 300 });

        try idx.save(path);
    }

    // Load and verify
    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, Meta).load(gpa.allocator(), path);
        defer idx.deinit();

        try testing.expectEqual(@as(usize, 3), idx.count());

        // Verify vectors
        const p0 = idx.getPoint(0);
        try testing.expectEqual(@as(f32, 1), p0[0]);
        try testing.expectEqual(@as(f32, 0), p0[1]);

        const p1 = idx.getPoint(1);
        try testing.expectEqual(@as(f32, 0), p1[0]);
        try testing.expectEqual(@as(f32, 1), p1[1]);

        // Verify metadata
        const m0 = idx.getMetadata(0).?;
        try testing.expectEqual(@as(u32, 100), m0.label);
        const m1 = idx.getMetadata(1).?;
        try testing.expectEqual(@as(u32, 200), m1.label);
        const m2 = idx.getMetadata(2).?;
        try testing.expectEqual(@as(u32, 300), m2.label);

        // Verify search still works
        const res = try idx.searchTopK(&[_]f32{ 1, 0, 0, 0 }, 1, 64);
        defer gpa.allocator().free(res);
        try testing.expectEqual(@as(usize, 1), res.len);
        try testing.expectEqual(@as(u32, 0), res[0].id);
    }

    // Cleanup
    std.fs.cwd().deleteFile(path) catch {};
}

test "HNSWv2 save/load with void metadata" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const path = "/tmp/test_hnsw_v2_void.zvd2";

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 2 });
        defer idx.deinit();

        _ = try idx.insert(&[_]f32{ 0, 0 }, {});
        _ = try idx.insert(&[_]f32{ 1, 1 }, {});

        try idx.save(path);
    }

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).load(gpa.allocator(), path);
        defer idx.deinit();

        try testing.expectEqual(@as(usize, 2), idx.count());

        const res = try idx.search(&[_]f32{ 0, 0 }, 1);
        defer gpa.allocator().free(res);
        try testing.expectEqual(@as(u32, 0), res[0].id);
    }

    std.fs.cwd().deleteFile(path) catch {};
}

test "HNSWv2 save/load preserves deleted state" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const path = "/tmp/test_hnsw_v2_deleted.zvd2";

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{
            .dims = 2,
            .reuse_deleted_slots = false, // Don't reuse so we can test deleted state
        });
        defer idx.deinit();

        const id0 = try idx.insert(&[_]f32{ 0, 0 }, {});
        _ = try idx.insert(&[_]f32{ 1, 1 }, {});
        try idx.delete(id0);

        try testing.expectEqual(@as(usize, 2), idx.count());
        try testing.expectEqual(@as(usize, 1), idx.liveCount());

        try idx.save(path);
    }

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).load(gpa.allocator(), path);
        defer idx.deinit();

        try testing.expectEqual(@as(usize, 2), idx.count());
        try testing.expectEqual(@as(usize, 1), idx.liveCount());

        // Search should not return deleted node
        const res = try idx.search(&[_]f32{ 0, 0 }, 2);
        defer gpa.allocator().free(res);
        try testing.expectEqual(@as(usize, 1), res.len);
        try testing.expectEqual(@as(u32, 1), res[0].id);
    }

    std.fs.cwd().deleteFile(path) catch {};
}

test "HNSWv2 save/load empty index" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const path = "/tmp/test_hnsw_v2_empty.zvd2";

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).init(gpa.allocator(), .{ .dims = 4 });
        defer idx.deinit();
        try idx.save(path);
    }

    {
        var idx = try h2.HNSWv2(f32, .squared_euclidean, void).load(gpa.allocator(), path);
        defer idx.deinit();

        try testing.expectEqual(@as(usize, 0), idx.count());
        try testing.expect(idx.entryPoint() == null);
    }

    std.fs.cwd().deleteFile(path) catch {};
}
