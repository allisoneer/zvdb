const std = @import("std");
const testing = std.testing;
const s2 = @import("search2.zig");

const MockGraph = struct {
    pts: [5][2]f32 = .{
        .{ 0, 0 }, // 0: origin
        .{ 1, 0 }, // 1: distance 1 from origin
        .{ 0, 1 }, // 2: distance 1 from origin
        .{ 2, 0 }, // 3: distance 2 from origin
        .{ 0, 2 }, // 4: distance 2 from origin
    },
    // Simple connected graph: 0 connects to 1,2; 1 connects to 0,3; 2 connects to 0,4
    neighbors: [5][]const u32 = .{
        &[_]u32{ 1, 2 },
        &[_]u32{ 0, 3 },
        &[_]u32{ 0, 4 },
        &[_]u32{1},
        &[_]u32{2},
    },
    deleted: [5]bool = .{ false, false, false, false, false },

    pub fn dist(self: *const MockGraph, a: []const f32, b: []const f32) f32 {
        _ = self;
        var sum: f32 = 0;
        for (a, b) |av, bv| sum += (av - bv) * (av - bv);
        return sum;
    }

    pub fn getPoint(self: *const MockGraph, id: u32) []const f32 {
        return &self.pts[id];
    }

    pub fn getNeighborsSlice(self: *const MockGraph, id: u32, level: usize) []const u32 {
        _ = level;
        return self.neighbors[id];
    }

    pub fn isDeleted(self: *const MockGraph, id: u32) bool {
        return self.deleted[id];
    }
};

test "search2 basic graph - find closest" {
    const graph = MockGraph{};
    const q = [_]f32{ 0, 0 };

    const res = try s2.topK2(
        f32,
        *const MockGraph,
        5,
        0,
        0,
        10,
        &q,
        2,
        testing.allocator,
        &graph,
    );
    defer testing.allocator.free(res);

    try testing.expectEqual(@as(usize, 2), res.len);
    try testing.expectEqual(@as(u32, 0), res[0].id); // exact match at origin
    try testing.expectEqual(@as(f32, 0), res[0].distance);
}

test "search2 finds all k results" {
    const graph = MockGraph{};
    const q = [_]f32{ 0, 0 };

    const res = try s2.topK2(
        f32,
        *const MockGraph,
        5,
        0,
        0,
        10,
        &q,
        5,
        testing.allocator,
        &graph,
    );
    defer testing.allocator.free(res);

    try testing.expectEqual(@as(usize, 5), res.len);
    // Results should be sorted by distance
    try testing.expect(res[0].distance <= res[1].distance);
    try testing.expect(res[1].distance <= res[2].distance);
}

test "search2 respects deleted nodes" {
    var graph = MockGraph{};
    graph.deleted[0] = true; // Delete the origin node

    const q = [_]f32{ 0, 0 };

    const res = try s2.topK2(
        f32,
        *const MockGraph,
        5,
        0,
        0,
        10,
        &q,
        3,
        testing.allocator,
        &graph,
    );
    defer testing.allocator.free(res);

    // Origin (id=0) should not be in results
    for (res) |r| {
        try testing.expect(r.id != 0);
    }
}

test "search2 empty entry point" {
    const graph = MockGraph{};
    const q = [_]f32{ 0, 0 };

    const res = try s2.topK2(
        f32,
        *const MockGraph,
        5,
        null, // No entry point
        0,
        10,
        &q,
        2,
        testing.allocator,
        &graph,
    );
    defer testing.allocator.free(res);

    try testing.expectEqual(@as(usize, 0), res.len);
}

test "search2 k=0 returns empty" {
    const graph = MockGraph{};
    const q = [_]f32{ 0, 0 };

    const res = try s2.topK2(
        f32,
        *const MockGraph,
        5,
        0,
        0,
        10,
        &q,
        0, // k=0
        testing.allocator,
        &graph,
    );
    defer testing.allocator.free(res);

    try testing.expectEqual(@as(usize, 0), res.len);
}
