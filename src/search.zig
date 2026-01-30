const std = @import("std");

pub fn SearchResult(comptime T: type) type {
    return struct {
        id: usize,
        distance: T,
    };
}

/// Multi-layer HNSW search with greedy descent and ef-search at layer 0.
/// Returns top-k results sorted by distance (ascending).
///
/// The `nodes` parameter must be a type supporting `.get(id)` returning an optional
/// node with `.point` and `.connections` fields.
pub fn topK(
    comptime T: type,
    nodes: anytype,
    entry_point: ?usize,
    max_level: usize,
    ef_search: usize,
    comptime dist: fn ([]const T, []const T) T,
    query: []const T,
    k: usize,
    allocator: std.mem.Allocator,
) ![]SearchResult(T) {
    if (entry_point == null or k == 0) return &.{};

    // 1) Greedy descent from top layer down to layer 1
    var ep = entry_point.?;
    const ep_node = nodes.get(ep) orelse return &.{};
    var ep_dist = dist(query, ep_node.point);

    if (max_level > 0) {
        var level: usize = max_level;
        while (level > 0) : (level -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const node = nodes.get(ep) orelse break;
                if (level < node.connections.len) {
                    for (node.connections[level].items) |nb| {
                        const nb_node = nodes.get(nb) orelse continue;
                        const d = dist(query, nb_node.point);
                        if (d < ep_dist) {
                            ep_dist = d;
                            ep = nb;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // 2) ef-search at layer 0
    var visited = std.AutoHashMap(usize, void).init(allocator);
    defer visited.deinit();

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
    try visited.put(ep, {});
    // Only add to results if not deleted
    if (!ep_node.deleted) {
        try results.add(.{ .id = ep, .distance = ep_dist });
    }

    while (candidates.count() > 0) {
        const c = candidates.remove();

        if (results.count() >= ef_search) {
            if (c.distance > results.peek().?.distance) break;
        }

        const node = nodes.get(c.id) orelse continue;
        if (node.connections.len > 0) {
            for (node.connections[0].items) |nb| {
                if (visited.contains(nb)) continue;
                try visited.put(nb, {});

                const nb_node = nodes.get(nb) orelse continue;
                const d = dist(query, nb_node.point);

                // Always add to candidates for traversal (even deleted nodes)
                try candidates.add(.{ .id = nb, .distance = d });

                // Only add to results if not deleted
                if (!nb_node.deleted) {
                    if (results.count() < ef_search or d < results.peek().?.distance) {
                        try results.add(.{ .id = nb, .distance = d });
                        if (results.count() > ef_search) {
                            _ = results.remove();
                        }
                    }
                }
            }
        }
    }

    // Keep only the best k (drop worst until size <= k)
    while (results.count() > k) {
        _ = results.remove(); // remove worst (max distance)
    }

    // Extract sorted ascending (pop worst -> fill from end)
    const rcount = results.count();
    const out = try allocator.alloc(SearchResult(T), rcount);
    var i: usize = rcount;
    while (i > 0) {
        i -= 1;
        out[i] = results.remove();
    }
    return out;
}

/// Filtered search with predicate applied during traversal.
/// Still visits neighbors of filtered-out nodes to preserve graph reachability.
///
/// The `nodes` parameter must be a type supporting `.get(id)` returning an optional
/// node with `.point`, `.connections`, and `.meta_fixed` fields.
pub fn topKFiltered(
    comptime T: type,
    comptime MetaFixed: type,
    comptime Ctx: type,
    nodes: anytype,
    entry_point: ?usize,
    max_level: usize,
    ef_search: usize,
    comptime dist: fn ([]const T, []const T) T,
    query: []const T,
    k: usize,
    ctx: Ctx,
    comptime pred: fn (Ctx, MetaFixed) bool,
    allocator: std.mem.Allocator,
) ![]SearchResult(T) {
    if (entry_point == null or k == 0) return &.{};

    // 1) Greedy descent from top layer down to layer 1
    var ep = entry_point.?;
    const ep_node = nodes.get(ep) orelse return &.{};
    var ep_dist = dist(query, ep_node.point);

    if (max_level > 0) {
        var level: usize = max_level;
        while (level > 0) : (level -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const node = nodes.get(ep) orelse break;
                if (level < node.connections.len) {
                    for (node.connections[level].items) |nb| {
                        const nb_node = nodes.get(nb) orelse continue;
                        const d = dist(query, nb_node.point);
                        if (d < ep_dist) {
                            ep_dist = d;
                            ep = nb;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // 2) ef-search at layer 0 with filtering
    var visited = std.AutoHashMap(usize, void).init(allocator);
    defer visited.deinit();

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
    try visited.put(ep, {});

    // Only add to results if not deleted and passes filter
    if (!ep_node.deleted and pred(ctx, ep_node.meta_fixed)) {
        try results.add(.{ .id = ep, .distance = ep_dist });
    }

    while (candidates.count() > 0) {
        const c = candidates.remove();

        // Early termination check based on filtered results
        if (results.count() >= ef_search) {
            if (c.distance > results.peek().?.distance) break;
        }

        const node = nodes.get(c.id) orelse continue;
        if (node.connections.len > 0) {
            for (node.connections[0].items) |nb| {
                if (visited.contains(nb)) continue;
                try visited.put(nb, {});

                const nb_node = nodes.get(nb) orelse continue;
                const d = dist(query, nb_node.point);

                // Always add to candidates for traversal (even deleted nodes)
                try candidates.add(.{ .id = nb, .distance = d });

                // Only add to results if not deleted and passes filter
                if (!nb_node.deleted and pred(ctx, nb_node.meta_fixed)) {
                    if (results.count() < ef_search or d < results.peek().?.distance) {
                        try results.add(.{ .id = nb, .distance = d });
                        if (results.count() > ef_search) {
                            _ = results.remove();
                        }
                    }
                }
            }
        }
    }

    // Keep only the best k (drop worst until size <= k)
    while (results.count() > k) {
        _ = results.remove(); // remove worst (max distance)
    }

    // Extract sorted ascending (pop worst -> fill from end)
    const rcount = results.count();
    const out = try allocator.alloc(SearchResult(T), rcount);
    var i: usize = rcount;
    while (i > 0) {
        i -= 1;
        out[i] = results.remove();
    }
    return out;
}
