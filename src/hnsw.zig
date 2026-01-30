const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const Mutex = std.Thread.Mutex;

pub const distance = @import("distance.zig");
pub const metadata = @import("metadata.zig");
pub const search = @import("search.zig");
pub const persistence = @import("persistence.zig");

pub const DistanceMetric = distance.DistanceMetric;

pub fn HNSW(comptime T: type, comptime metric: DistanceMetric, comptime Metadata: type) type {
    return struct {
        const Self = @This();
        pub const MetaFixed = metadata.FixedOf(Metadata);
        const dist = distance.distanceFn(T, metric);

        pub const Error = error{
            NodeNotFound,
            NodeDeleted,
            DimMismatch,
        };

        pub const Node = struct {
            id: usize,
            point: []T,
            meta_fixed: MetaFixed,
            connections: []ArrayList(usize),
            mutex: Mutex,
            deleted: bool = false,

            pub fn init(allocator: Allocator, id: usize, point: []const T, level: usize, mf: MetaFixed) !Node {
                const conns = try allocator.alloc(ArrayList(usize), level + 1);
                errdefer allocator.free(conns);
                for (conns) |*c| c.* = .{};

                const owned_point = try allocator.alloc(T, point.len);
                errdefer allocator.free(owned_point);
                @memcpy(owned_point, point);

                return .{
                    .id = id,
                    .point = owned_point,
                    .meta_fixed = mf,
                    .connections = conns,
                    .mutex = .{},
                };
            }

            pub fn deinit(self: *Node, allocator: Allocator) void {
                for (self.connections) |*c| c.deinit(allocator);
                allocator.free(self.connections);
                allocator.free(self.point);
            }
        };

        allocator: Allocator,
        nodes: AutoHashMap(usize, Node),
        entry_point: ?usize,
        max_level: usize,
        m: usize,
        ef_construction: usize,
        default_ef_search: usize,
        dims: usize,
        string_table: metadata.StringTable,
        mutex: Mutex,

        pub fn init(allocator: Allocator, dims: usize, m: usize, ef_construction: usize) Self {
            return .{
                .allocator = allocator,
                .nodes = AutoHashMap(usize, Node).init(allocator),
                .entry_point = null,
                .max_level = 0,
                .m = m,
                .ef_construction = ef_construction,
                .default_ef_search = @max(ef_construction, 64),
                .dims = dims,
                .string_table = .{},
                .mutex = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.nodes.iterator();
            while (it.next()) |kv| {
                kv.value_ptr.deinit(self.allocator);
            }
            self.nodes.deinit();
            if (self.string_table.data.len > 0) {
                self.allocator.free(self.string_table.data);
            }
        }

        pub fn insert(self: *Self, point: []const T, meta: Metadata) !usize {
            std.debug.assert(point.len == self.dims);

            self.mutex.lock();
            defer self.mutex.unlock();

            const id = self.nodes.count();
            const level = self.randomLevel();
            const mf = try metadata.encode(Metadata, meta, &self.string_table, self.allocator);

            var node = try Node.init(self.allocator, id, point, level, mf);
            errdefer node.deinit(self.allocator);

            try self.nodes.put(id, node);

            if (self.entry_point) |entry| {
                var ep_copy = entry;
                var curr_dist = dist(node.point, self.nodes.get(ep_copy).?.point);

                for (0..self.max_level + 1) |layer| {
                    var changed = true;
                    while (changed) {
                        changed = false;
                        const curr_node = self.nodes.get(ep_copy).?;
                        if (layer < curr_node.connections.len) {
                            for (curr_node.connections[layer].items) |neighbor_id| {
                                const neighbor = self.nodes.get(neighbor_id).?;
                                const d = dist(node.point, neighbor.point);
                                if (d < curr_dist) {
                                    ep_copy = neighbor_id;
                                    curr_dist = d;
                                    changed = true;
                                }
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

            if (level > self.max_level) {
                self.max_level = level;
            }

            return id;
        }

        /// Batch insert multiple points with their metadata.
        /// Uses single lock and pre-allocated StringTable for efficiency.
        /// Returns slice of inserted IDs (caller owns).
        pub fn insertBatch(self: *Self, points: []const []const T, metas: []const Metadata) ![]usize {
            std.debug.assert(points.len == metas.len);
            if (points.len == 0) return &[_]usize{};

            self.mutex.lock();
            defer self.mutex.unlock();

            // Validate all dimensions
            for (points) |p| std.debug.assert(p.len == self.dims);

            const base_id = self.nodes.count();
            const n = points.len;

            // Pre-compute random levels
            const levels = try self.allocator.alloc(usize, n);
            defer self.allocator.free(levels);
            var max_level_new: usize = self.max_level;
            for (levels) |*lvl| {
                lvl.* = self.randomLevel();
                if (lvl.* > max_level_new) max_level_new = lvl.*;
            }

            // Pre-grow StringTable once for all metadata
            const add_bytes = metadata.totalStringBytesForBatch(Metadata, metas);
            const old_len = self.string_table.data.len;
            try self.string_table.ensureAdditionalCapacity(self.allocator, add_bytes);
            var cursor: usize = old_len;

            // Encode all metadata
            var fixeds = try self.allocator.alloc(MetaFixed, n);
            defer self.allocator.free(fixeds);
            for (metas, 0..) |m, idx| {
                fixeds[idx] = metadata.encodeInto(Metadata, m, &self.string_table, &cursor);
            }

            // Allocate result IDs
            var ids = try self.allocator.alloc(usize, n);
            errdefer self.allocator.free(ids);

            // Insert nodes sequentially, connecting to graph
            for (points, 0..) |p, idx| {
                const id = base_id + idx;
                const level = levels[idx];
                var node = try Node.init(self.allocator, id, p, level, fixeds[idx]);
                errdefer node.deinit(self.allocator);
                try self.nodes.put(id, node);
                ids[idx] = id;

                // Connect using same algorithm as single insert
                if (self.entry_point) |entry| {
                    var ep_copy = entry;
                    var curr_dist = dist(node.point, self.nodes.get(ep_copy).?.point);

                    for (0..self.max_level + 1) |layer| {
                        var changed = true;
                        while (changed) {
                            changed = false;
                            const curr_node = self.nodes.get(ep_copy).?;
                            if (layer < curr_node.connections.len) {
                                for (curr_node.connections[layer].items) |neighbor_id| {
                                    const neighbor = self.nodes.get(neighbor_id).?;
                                    const d = dist(node.point, neighbor.point);
                                    if (d < curr_dist) {
                                        ep_copy = neighbor_id;
                                        curr_dist = d;
                                        changed = true;
                                    }
                                }
                            }
                        }

                        if (layer <= level) {
                            try self.connect(id, ep_copy, @intCast(layer));
                        }
                    }

                    // Update max_level as we go to help subsequent inserts
                    if (level > self.max_level) self.max_level = level;
                } else {
                    self.entry_point = id;
                }
            }

            if (max_level_new > self.max_level) self.max_level = max_level_new;
            return ids;
        }

        /// Soft delete a node by marking it as deleted (tombstone).
        /// Search will still traverse through deleted nodes' neighbors but exclude them from results.
        /// If the entry point is deleted, it will be updated to a non-deleted neighbor if possible.
        pub fn delete(self: *Self, id: usize) Error!void {
            self.mutex.lock();
            defer self.mutex.unlock();
            const node = self.nodes.getPtr(id) orelse return Error.NodeNotFound;
            if (node.deleted) return Error.NodeDeleted;
            node.deleted = true;

            // If we deleted the entry point, try to find a new one
            if (self.entry_point == id) {
                self.entry_point = self.findNewEntryPoint(id);
            }
        }

        /// Find a new entry point after the current one is deleted.
        /// Looks for a non-deleted neighbor, or any non-deleted node.
        fn findNewEntryPoint(self: *Self, deleted_id: usize) ?usize {
            // First try neighbors of the deleted node
            if (self.nodes.get(deleted_id)) |deleted_node| {
                for (deleted_node.connections) |level_conns| {
                    for (level_conns.items) |neighbor_id| {
                        if (self.nodes.get(neighbor_id)) |neighbor| {
                            if (!neighbor.deleted) return neighbor_id;
                        }
                    }
                }
            }

            // If no neighbor found, find any non-deleted node
            var it = self.nodes.iterator();
            while (it.next()) |kv| {
                if (!kv.value_ptr.deleted) return kv.key_ptr.*;
            }

            return null;
        }

        /// Update by replacement: deletes old node and inserts new one with new ID.
        /// Returns the new ID.
        pub fn updateReplace(self: *Self, id: usize, point: []const T, meta: Metadata) !usize {
            try self.delete(id);
            return self.insert(point, meta);
        }

        /// Update in place: preserves ID, updates vector and metadata.
        /// Returns error if node is deleted or not found.
        pub fn updateInPlace(self: *Self, id: usize, point: []const T, meta: Metadata) !void {
            std.debug.assert(point.len == self.dims);
            self.mutex.lock();
            defer self.mutex.unlock();

            const node = self.nodes.getPtr(id) orelse return Error.NodeNotFound;
            if (node.deleted) return Error.NodeDeleted;
            if (node.point.len != point.len) return Error.DimMismatch;

            @memcpy(node.point, point);
            const mf = try metadata.encode(Metadata, meta, &self.string_table, self.allocator);
            node.meta_fixed = mf;
        }

        /// Compact the index by removing deleted nodes and rebuilding contiguous IDs.
        /// Remaps all connections to new IDs. Updates entry point.
        pub fn compact(self: *Self) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.compactLocked();
        }

        fn compactLocked(self: *Self) !void {
            const old_count = self.nodes.count();
            if (old_count == 0) return;

            // Build old->new ID map for non-deleted nodes
            var id_map = AutoHashMap(usize, usize).init(self.allocator);
            defer id_map.deinit();

            var live_count: usize = 0;
            var it = self.nodes.iterator();
            while (it.next()) |kv| {
                if (!kv.value_ptr.deleted) {
                    try id_map.put(kv.key_ptr.*, live_count);
                    live_count += 1;
                }
            }

            if (live_count == old_count) return; // Nothing to compact

            // Rebuild nodes map with new IDs
            var new_nodes = AutoHashMap(usize, Node).init(self.allocator);
            errdefer new_nodes.deinit();

            var it2 = self.nodes.iterator();
            while (it2.next()) |kv| {
                const old_id = kv.key_ptr.*;
                var n = kv.value_ptr.*;

                if (n.deleted) {
                    n.deinit(self.allocator);
                    continue;
                }

                const new_id = id_map.get(old_id).?;

                // Remap connections to new IDs
                for (n.connections) |*level_list| {
                    var write_idx: usize = 0;
                    for (level_list.items) |nb_old| {
                        if (id_map.get(nb_old)) |nb_new| {
                            level_list.items[write_idx] = nb_new;
                            write_idx += 1;
                        }
                    }
                    level_list.shrinkRetainingCapacity(write_idx);
                }

                n.id = new_id;
                n.deleted = false;
                try new_nodes.put(new_id, n);
            }

            // Swap maps
            self.nodes.deinit();
            self.nodes = new_nodes;

            // Update entry point
            if (self.entry_point) |old_ep| {
                self.entry_point = id_map.get(old_ep);
            }

            // Recompute max_level
            var max_lvl: usize = 0;
            var it3 = self.nodes.iterator();
            while (it3.next()) |kv| {
                const lvl = if (kv.value_ptr.connections.len > 0) kv.value_ptr.connections.len - 1 else 0;
                if (lvl > max_lvl) max_lvl = lvl;
            }
            self.max_level = max_lvl;
        }

        /// Returns the count of non-deleted nodes.
        pub fn liveCount(self: *const Self) usize {
            var live: usize = 0;
            var it = self.nodes.iterator();
            while (it.next()) |kv| {
                if (!kv.value_ptr.deleted) live += 1;
            }
            return live;
        }

        fn connect(self: *Self, source: usize, target: usize, level: usize) !void {
            var source_node = self.nodes.getPtr(source) orelse return error.NodeNotFound;
            var target_node = self.nodes.getPtr(target) orelse return error.NodeNotFound;

            source_node.mutex.lock();
            defer source_node.mutex.unlock();
            target_node.mutex.lock();
            defer target_node.mutex.unlock();

            if (level < source_node.connections.len) {
                try source_node.connections[level].append(self.allocator, target);
            }
            if (level < target_node.connections.len) {
                try target_node.connections[level].append(self.allocator, source);
            }

            if (level < source_node.connections.len) {
                try self.shrinkConnections(source, level);
            }
            if (level < target_node.connections.len) {
                try self.shrinkConnections(target, level);
            }
        }

        fn shrinkConnections(self: *Self, node_id: usize, level: usize) !void {
            var node = self.nodes.getPtr(node_id).?;
            var connections = &node.connections[level];
            if (connections.items.len <= self.m) return;

            var candidates = try self.allocator.alloc(usize, connections.items.len);
            defer self.allocator.free(candidates);
            @memcpy(candidates, connections.items);

            const Ctx = struct { self: *Self, node: *Node };
            const ctx = Ctx{ .self = self, .node = node };
            const cmp = struct {
                fn less(c: Ctx, a: usize, b: usize) bool {
                    const da = dist(c.node.point, c.self.nodes.get(a).?.point);
                    const db = dist(c.node.point, c.self.nodes.get(b).?.point);
                    return da < db;
                }
            }.less;

            std.sort.insertion(usize, candidates, ctx, cmp);
            connections.shrinkRetainingCapacity(self.m);
            @memcpy(connections.items, candidates[0..self.m]);
        }

        fn randomLevel(self: *Self) usize {
            _ = self;
            var level: usize = 0;
            const max_level_limit = 31;
            while (level < max_level_limit and std.crypto.random.float(f32) < 0.5) {
                level += 1;
            }
            return level;
        }

        pub fn searchTopK(self: *Self, query: []const T, k: usize, ef_search_param: usize) ![]search.SearchResult(T) {
            std.debug.assert(query.len == self.dims);
            self.mutex.lock();
            defer self.mutex.unlock();

            return try search.topK(
                T,
                self.nodes,
                self.entry_point,
                self.max_level,
                ef_search_param,
                dist,
                query,
                k,
                self.allocator,
            );
        }

        pub fn searchDefault(self: *Self, query: []const T, k: usize) ![]search.SearchResult(T) {
            return self.searchTopK(query, k, self.default_ef_search);
        }

        /// Search with a filter predicate applied during traversal.
        /// Still visits neighbors of filtered-out nodes to preserve graph reachability.
        pub fn searchFiltered(
            self: *Self,
            query: []const T,
            k: usize,
            ef_search_param: usize,
            ctx: anytype,
            comptime pred: fn (@TypeOf(ctx), MetaFixed) bool,
        ) ![]search.SearchResult(T) {
            std.debug.assert(query.len == self.dims);
            self.mutex.lock();
            defer self.mutex.unlock();

            return try search.topKFiltered(
                T,
                MetaFixed,
                @TypeOf(ctx),
                self.nodes,
                self.entry_point,
                self.max_level,
                ef_search_param,
                dist,
                query,
                k,
                ctx,
                pred,
                self.allocator,
            );
        }

        pub fn getStringTable(self: *const Self) *const metadata.StringTable {
            return &self.string_table;
        }

        pub fn getNode(self: *const Self, id: usize) ?*const Node {
            return self.nodes.getPtr(id);
        }

        pub fn getMetadata(self: *const Self, id: usize) ?Metadata {
            const node = self.nodes.getPtr(id) orelse return null;
            return metadata.decode(Metadata, node.meta_fixed, &self.string_table);
        }

        pub fn count(self: *const Self) usize {
            return self.nodes.count();
        }

        /// Save the entire index to a file.
        /// Always compacts first to ensure clean contiguous IDs.
        pub fn save(self: *Self, path: []const u8) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            // Always compact before save to produce clean file
            try self.compactLocked();

            const node_count = self.nodes.count();

            // Build header
            const entry_raw: u64 = if (self.entry_point) |ep| @intCast(ep) else std.math.maxInt(u64);

            const header: persistence.FileHeader = .{
                .metric = @intCast(@intFromEnum(metric)),
                .t_size = @intCast(@sizeOf(T)),
                .dims = @intCast(self.dims),
                .m = @intCast(self.m),
                .ef_construction = @intCast(self.ef_construction),
                .node_count = @intCast(node_count),
                .entry_point = entry_raw,
                .max_level = @intCast(self.max_level),
                .string_blob_len = @intCast(self.string_table.data.len),
            };

            // Serialize sections
            const vectors_bytes = try persistence.serializeVectors(T, self.nodes, node_count, self.dims, self.allocator);
            defer if (vectors_bytes.len > 0) self.allocator.free(vectors_bytes);

            const metas_bytes = try persistence.serializeMetadata(MetaFixed, self.nodes, node_count, self.allocator);
            defer if (metas_bytes.len > 0) self.allocator.free(metas_bytes);

            const graph_bytes = try persistence.serializeGraph(self.nodes, node_count, self.allocator);
            defer if (graph_bytes.len > 0) self.allocator.free(graph_bytes);

            try persistence.save(path, header, vectors_bytes, metas_bytes, graph_bytes, self.string_table.data);
        }

        /// Load an index from a file
        pub fn load(allocator: Allocator, path: []const u8) !Self {
            const header = try persistence.loadHeader(path);

            // Validate compile-time compatibility
            if (header.metric != @as(u8, @intCast(@intFromEnum(metric)))) return error.MetricMismatch;
            if (header.t_size != @as(u8, @intCast(@sizeOf(T)))) return error.TypeSizeMismatch;

            var g = Self.init(allocator, header.dims, header.m, header.ef_construction);
            errdefer g.deinit();
            g.max_level = header.max_level;

            // Open file and read past header
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            var buf: [4096]u8 = undefined;
            var reader = file.reader(&buf);

            // Skip header
            _ = try reader.interface.takeStruct(persistence.FileHeader, .little);

            const node_count: usize = @intCast(header.node_count);
            const dims: usize = @intCast(header.dims);
            const t_size: usize = header.t_size;
            const vec_len: usize = node_count * dims * t_size;
            const mf_size: usize = @sizeOf(MetaFixed);
            const metas_len: usize = node_count * mf_size;
            const string_len: usize = @intCast(header.string_blob_len);

            // Compute graph size from file size
            const stat = try file.stat();
            const header_size: usize = @sizeOf(persistence.FileHeader);
            const total_size: usize = @intCast(stat.size);
            if (total_size < header_size + vec_len + metas_len + string_len) return error.SizeMismatch;
            const graph_len: usize = total_size - header_size - vec_len - metas_len - string_len;

            // Read sections
            const vectors_bytes = try allocator.alloc(u8, vec_len);
            defer if (vectors_bytes.len > 0) allocator.free(vectors_bytes);
            if (vec_len > 0) try reader.interface.readSliceAll(vectors_bytes);

            const metas_bytes = try allocator.alloc(u8, metas_len);
            defer if (metas_bytes.len > 0) allocator.free(metas_bytes);
            if (metas_len > 0) try reader.interface.readSliceAll(metas_bytes);

            const graph_bytes = try allocator.alloc(u8, graph_len);
            defer if (graph_bytes.len > 0) allocator.free(graph_bytes);
            if (graph_len > 0) try reader.interface.readSliceAll(graph_bytes);

            // Read string blob
            if (string_len > 0) {
                g.string_table.data = try allocator.alloc(u8, string_len);
                try reader.interface.readSliceAll(g.string_table.data);
            }

            // Deserialize sections
            const points = try persistence.deserializeVectors(T, vectors_bytes, node_count, dims, allocator);
            defer allocator.free(points); // Outer slice only; inner slices moved to nodes

            const metas = try persistence.deserializeMetadata(MetaFixed, metas_bytes, node_count, allocator);
            defer if (metas.len > 0) allocator.free(metas);

            const conns = try persistence.deserializeGraph(graph_bytes, node_count, allocator);
            defer if (conns.len > 0) allocator.free(conns);

            // Assemble nodes
            for (0..node_count) |id| {
                const node = Node{
                    .id = id,
                    .point = points[id], // Takes ownership
                    .meta_fixed = metas[id],
                    .connections = conns[id], // Takes ownership
                    .mutex = .{},
                };
                try g.nodes.put(id, node);
            }

            // Entry point
            if (node_count == 0) {
                g.entry_point = null;
            } else {
                const ep_raw = header.entry_point;
                if (ep_raw == std.math.maxInt(u64)) {
                    g.entry_point = null;
                } else {
                    if (ep_raw >= header.node_count) return error.InvalidEntryPoint;
                    g.entry_point = @intCast(ep_raw);
                }
            }

            g.default_ef_search = @max(g.ef_construction, 64);
            return g;
        }
    };
}
