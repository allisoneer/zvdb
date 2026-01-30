const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

pub fn HNSW(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            id: usize,
            point: []const T,
            connections: []ArrayList(usize),

            fn init(allocator: Allocator, id: usize, point: []const T, level: usize) !Node {
                var connections = try allocator.alloc(ArrayList(usize), level + 1);
                for (connections) |*conn| {
                    conn.* = ArrayList(usize).init(allocator);
                }
                return Node{
                    .id = id,
                    .point = point,
                    .connections = connections,
                };
            }

            fn deinit(self: *Node, allocator: Allocator) void {
                for (self.connections) |*conn| {
                    conn.deinit();
                }
                allocator.free(self.connections);
            }
        };

        allocator: Allocator,
        nodes: AutoHashMap(usize, Node),
        entry_point: ?usize,
        max_level: usize,
        m: usize,
        ef_construction: usize,

        pub fn init(allocator: Allocator, m: usize, ef_construction: usize) Self {
            return .{
                .allocator = allocator,
                .nodes = AutoHashMap(usize, Node).init(allocator),
                .entry_point = null,
                .max_level = 0,
                .m = m,
                .ef_construction = ef_construction,
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.nodes.valueIterator();
            while (it.next()) |node| {
                node.deinit(self.allocator);
            }
            self.nodes.deinit();
        }

        pub fn insert(self: *Self, point: []const T) !void {
            const id = self.nodes.count();
            const level = self.randomLevel();
            var node = try Node.init(self.allocator, id, point, level);

            if (level > self.max_level) {
                self.max_level = level;
            }

            if (self.entry_point) |entry| {
                var ep_copy = entry;
                var curr_dist = distance(point, self.nodes.get(ep_copy).?.point);

                for (0..@min(self.max_level, level) + 1) |layer| {
                    var changed = true;
                    while (changed) {
                        changed = false;
                        const curr_node = self.nodes.get(ep_copy).?;
                        for (curr_node.connections[layer].items) |neighbor_id| {
                            const neighbor = self.nodes.get(neighbor_id).?;
                            const dist = distance(point, neighbor.point);
                            if (dist < curr_dist) {
                                ep_copy = neighbor_id;
                                curr_dist = dist;
                                changed = true;
                            }
                        }
                    }

                    if (layer <= level) {
                        try self.connect(id, ep_copy, @intCast(layer));
                    }
                }
            } else {
                self.entry_point = id;
            }

            try self.nodes.put(id, node);
        }

        fn connect(self: *Self, source: usize, target: usize, level: usize) !void {
            try self.nodes.getPtr(source).?.connections[level].append(target);
            try self.nodes.getPtr(target).?.connections[level].append(source);

            try self.shrinkConnections(source, level);
            try self.shrinkConnections(target, level);
        }

        fn shrinkConnections(self: *Self, node_id: usize, level: usize) !void {
            var node = self.nodes.getPtr(node_id).?;
            var connections = &node.connections[level];
            if (connections.items.len <= self.m) return;

            var candidates = try self.allocator.alloc(usize, connections.items.len);
            defer self.allocator.free(candidates);
            @memcpy(candidates, connections.items);

            const Context = struct {
                self: *Self,
                node: *Node,
            };
            const context = Context{ .self = self, .node = node };

            const compareFn = struct {
                fn compare(ctx: Context, a: usize, b: usize) bool {
                    const dist_a = distance(ctx.node.point, ctx.self.nodes.get(a).?.point);
                    const dist_b = distance(ctx.node.point, ctx.self.nodes.get(b).?.point);
                    return dist_a < dist_b;
                }
            }.compare;

            std.sort.insertion(usize, candidates, context, compareFn);

            connections.shrinkRetainingCapacity(self.m);
            @memcpy(connections.items, candidates[0..self.m]);
        }

        fn randomLevel(self: *Self) usize {
            var level: usize = 0;
            while (level < 31 and std.crypto.random.float(f32) < 0.5) {
                level += 1;
            }
            return level;
        }

        fn distance(a: []const T, b: []const T) T {
            var sum: T = 0;
            for (a, 0..) |_, i| {
                const diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum; // Note: We're returning squared distance for efficiency
        }

        pub fn search(self: *Self, query: []const T, k: usize) ![]const Node {
            var result = try ArrayList(Node).initCapacity(self.allocator, k);
            errdefer result.deinit();

            if (self.entry_point) |entry| {
                const Context = struct {
                    pub fn lessThan(_: void, a: CandidateNode, b: CandidateNode) bool {
                        return a.distance > b.distance; // Max-heap for efficient pruning
                    }
                };

                var candidates = std.PriorityQueue(CandidateNode, void, Context.lessThan).init(self.allocator, {});
                defer candidates.deinit();

                var visited = std.AutoHashMap(usize, void).init(self.allocator);
                defer visited.deinit();

                try candidates.add(.{ .id = entry, .distance = distance(query, self.nodes.get(entry).?.point) });
                try visited.put(entry, {});

                while (candidates.count() > 0 and result.items.len < k) {
                    const current = candidates.remove();
                    const current_node = self.nodes.get(current.id).?;
                    try result.append(current_node);

                    for (current_node.connections[0].items) |neighbor_id| {
                        if (!visited.contains(neighbor_id)) {
                            const neighbor = self.nodes.get(neighbor_id).?;
                            const dist = distance(query, neighbor.point);
                            try candidates.add(.{ .id = neighbor_id, .distance = dist });
                            try visited.put(neighbor_id, {});
                        }
                    }
                }
            }

            return result.toOwnedSlice();
        }

        const CandidateNode = struct {
            id: usize,
            distance: T,
        };
    };
}
