const std = @import("std");

pub fn SearchResult(comptime T: type) type {
    return struct {
        id: u32,
        distance: T,
    };
}

/// Multi-layer HNSW search with DynamicBitSet for O(1) visited tracking.
/// Returns top-k results sorted by distance (ascending).
///
/// The Context type must provide:
/// - dist(ctx, a: []const T, b: []const T) T
/// - getPoint(ctx, id: u32) []const T
/// - getNeighborsSlice(ctx, id: u32, level: usize) []const u32
/// - isDeleted(ctx, id: u32) bool
pub fn topK2(
    comptime T: type,
    comptime Context: type,
    node_count: usize,
    entry_point: ?u32,
    max_level: u16,
    ef_search: usize,
    query: []const T,
    k: usize,
    allocator: std.mem.Allocator,
    ctx: Context,
) ![]SearchResult(T) {
    if (entry_point == null or k == 0 or node_count == 0) return &.{};

    var visited = try std.DynamicBitSet.initEmpty(allocator, node_count);
    defer visited.deinit();

    var ep = entry_point.?;
    var ep_dist = ctx.dist(query, ctx.getPoint(ep));

    // Greedy descent through upper levels
    if (max_level > 0) {
        var level: u16 = max_level;
        while (level > 0) : (level -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const neighbors = ctx.getNeighborsSlice(ep, level);
                for (neighbors) |nb| {
                    const d = ctx.dist(query, ctx.getPoint(nb));
                    if (d < ep_dist) {
                        ep = nb;
                        ep_dist = d;
                        changed = true;
                    }
                }
            }
        }
    }

    const Candidate = SearchResult(T);
    const lessFn = struct {
        fn f(_: void, a: Candidate, b: Candidate) std.math.Order {
            return std.math.order(a.distance, b.distance);
        }
    }.f;
    const greaterFn = struct {
        fn f(_: void, a: Candidate, b: Candidate) std.math.Order {
            return std.math.order(b.distance, a.distance);
        }
    }.f;

    var candidates = std.PriorityQueue(Candidate, void, lessFn).init(allocator, {});
    defer candidates.deinit();
    var results = std.PriorityQueue(Candidate, void, greaterFn).init(allocator, {});
    defer results.deinit();

    try candidates.add(.{ .id = ep, .distance = ep_dist });
    visited.set(ep);
    if (!ctx.isDeleted(ep)) try results.add(.{ .id = ep, .distance = ep_dist });

    while (candidates.count() > 0) {
        const c = candidates.remove();
        if (results.count() >= ef_search and c.distance > results.peek().?.distance) break;

        const neighbors = ctx.getNeighborsSlice(c.id, 0);
        for (neighbors) |nb| {
            if (visited.isSet(nb)) continue;
            visited.set(nb);

            const d = ctx.dist(query, ctx.getPoint(nb));
            try candidates.add(.{ .id = nb, .distance = d });
            if (!ctx.isDeleted(nb)) {
                if (results.count() < ef_search or d < results.peek().?.distance) {
                    try results.add(.{ .id = nb, .distance = d });
                    if (results.count() > ef_search) {
                        _ = results.remove();
                    }
                }
            }
        }
    }

    while (results.count() > k) _ = results.remove();
    const rcount = results.count();
    const out = try allocator.alloc(Candidate, rcount);
    var i: usize = rcount;
    while (i > 0) {
        i -= 1;
        out[i] = results.remove();
    }
    return out;
}
