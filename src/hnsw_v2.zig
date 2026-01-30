const std = @import("std");
const Allocator = std.mem.Allocator;
const metadata = @import("metadata.zig");
const vs = @import("vector_store.zig");
const tape = @import("graph_tape.zig");
const nref = @import("neighbors_ref.zig");
const s2 = @import("search2.zig");
const dist_mod = @import("distance.zig");

pub fn HNSWv2(comptime T: type, comptime metric: dist_mod.DistanceMetric, comptime Metadata: type) type {
    return struct {
        const Self = @This();
        pub const MetaFixed = metadata.FixedOf(Metadata);
        const distFn = dist_mod.distanceFn(T, metric);
        const has_metadata = Metadata != void;

        pub const Config = struct {
            dims: u32,
            m: u32 = 16,
            m_base: u32 = 32,
            ef_construction: u32 = 200,
            default_ef_search: u32 = 64,
            reuse_deleted_slots: bool = true,
            seed: u64 = 0xBADBEEFCAFED00D,
        };

        const Node = struct {
            graph_offset: u32,
            vector_slot: u32,
            level: u16,
            deleted: bool,
        };

        allocator: Allocator,
        config: Config,
        nodes: std.ArrayListUnmanaged(Node) = .{},
        vectors: vs.VectorStore,
        graph_tape: tape.GraphTape,
        metadata_store: if (has_metadata) std.ArrayListUnmanaged(MetaFixed) else void =
            if (has_metadata) .{} else {},
        string_table: if (has_metadata) metadata.StringTable else void =
            if (has_metadata) .{} else {},
        free_slots: std.ArrayListUnmanaged(u32) = .{},
        free_vector_slots: std.ArrayListUnmanaged(u32) = .{},
        entry_point: ?u32 = null,
        max_level: u16 = 0,
        prng: std.Random.Xoshiro256,

        pub fn init(allocator: Allocator, cfg: Config) !Self {
            const ts: u64 = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
            const seed = cfg.seed ^ ts;
            return .{
                .allocator = allocator,
                .config = cfg,
                .vectors = try vs.VectorStore.init(allocator, 0, cfg.dims, T),
                .graph_tape = tape.GraphTape.init(allocator, .{ .m = cfg.m, .m_base = cfg.m_base }),
                .prng = std.Random.Xoshiro256.init(seed),
            };
        }

        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.allocator);
            self.free_slots.deinit(self.allocator);
            self.free_vector_slots.deinit(self.allocator);
            if (has_metadata) {
                self.metadata_store.deinit(self.allocator);
                self.string_table.deinit(self.allocator);
            }
            self.graph_tape.deinit();
            self.vectors.deinit();
        }

        pub fn count(self: *const Self) usize {
            return self.nodes.items.len;
        }

        pub fn liveCount(self: *const Self) usize {
            var c: usize = 0;
            for (self.nodes.items) |n| {
                if (!n.deleted) c += 1;
            }
            return c;
        }

        /// Get the current entry point (root node for search).
        pub fn entryPoint(self: *const Self) ?u32 {
            return self.entry_point;
        }

        /// Random level generation using ctz trick (fast Xoshiro256).
        fn randomLevel(self: *Self) u16 {
            const max_level_limit: u16 = 31;
            const x = self.prng.random().int(u64);
            if (x == 0) return max_level_limit;
            const lvl: u16 = @intCast(@ctz(x));
            return if (lvl > max_level_limit) max_level_limit else lvl;
        }

        pub fn getPoint(self: *const Self, id: u32) []const T {
            const slot = self.nodes.items[id].vector_slot;
            return self.vectors.getTyped(T, slot, self.config.dims);
        }

        pub fn getNeighborsAt(self: *Self, id: u32, level: usize) nref.NeighborsRef {
            const off = self.nodes.items[id].graph_offset;
            return self.graph_tape.getNeighbors(off, level);
        }

        pub fn getNeighborsSlice(self: *const Self, id: u32, level: usize) []const u32 {
            // Guard: node may not have this level allocated
            if (level > self.nodes.items[id].level) return &[_]u32{};
            const off = self.nodes.items[id].graph_offset;
            const nr = self.graph_tape.getNeighborsConst(off, level);
            return nr.slice();
        }

        fn isDeleted(self: *const Self, id: u32) bool {
            return self.nodes.items[id].deleted;
        }

        fn distance(_: *const Self, a: []const T, b: []const T) T {
            return distFn(a, b);
        }

        /// Prune connections to keep at most max_keep neighbors.
        fn shrinkConnections(self: *Self, id: u32, level: usize, max_keep: u32) !void {
            var neigh = self.getNeighborsAt(id, level);
            const n = neigh.count();
            if (n <= max_keep) return;

            const CWD = struct { id: u32, dist: T };
            var tmp = try self.allocator.alloc(CWD, n);
            defer self.allocator.free(tmp);

            const pv = self.getPoint(id);
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const nb = neigh.get(i);
                tmp[i] = .{ .id = nb, .dist = distFn(pv, self.getPoint(nb)) };
            }
            std.sort.insertion(CWD, tmp, {}, struct {
                fn lessThan(_: void, a: CWD, b: CWD) bool {
                    return a.dist < b.dist;
                }
            }.lessThan);
            neigh.clear();
            const keep = @min(@as(usize, max_keep), tmp.len);
            var k: usize = 0;
            while (k < keep) : (k += 1) neigh.pushBack(tmp[k].id);
        }

        fn connectMutual(self: *Self, a: u32, b: u32, level: usize) !void {
            if (a == b) return;
            const max_n = if (level == 0) self.config.m_base else self.config.m;

            var na = self.getNeighborsAt(a, level);
            if (!na.contains(b) and na.count() < max_n) na.pushBack(b);
            try self.shrinkConnections(a, level, max_n);

            var nb = self.getNeighborsAt(b, level);
            if (!nb.contains(a) and nb.count() < max_n) nb.pushBack(a);
            try self.shrinkConnections(b, level, max_n);
        }

        pub fn insert(self: *Self, point: []const T, meta: Metadata) !u32 {
            std.debug.assert(point.len == self.config.dims);

            // Acquire slot
            const slot_id: u32 = if (self.config.reuse_deleted_slots and self.free_slots.items.len > 0)
                self.free_slots.pop().?
            else
                @intCast(self.nodes.items.len);

            // Acquire vector slot (reuse if available)
            const vslot: u32 = blk: {
                if (self.config.reuse_deleted_slots and self.free_vector_slots.items.len > 0) {
                    const reused_slot = self.free_vector_slots.pop().?;
                    self.vectors.set(reused_slot, std.mem.sliceAsBytes(point));
                    break :blk reused_slot;
                } else {
                    break :blk try self.vectors.add(std.mem.sliceAsBytes(point));
                }
            };

            // Node level and tape allocation
            const level = self.randomLevel();
            const goff = try self.graph_tape.allocateNode(level);

            // Metadata
            const mf = if (has_metadata) try metadata.encode(Metadata, meta, &self.string_table, self.allocator) else {};

            // Append node record
            if (slot_id == self.nodes.items.len) {
                try self.nodes.append(self.allocator, .{
                    .graph_offset = goff,
                    .vector_slot = vslot,
                    .level = level,
                    .deleted = false,
                });
                if (has_metadata) try self.metadata_store.append(self.allocator, mf);
            } else {
                self.nodes.items[slot_id] = .{
                    .graph_offset = goff,
                    .vector_slot = vslot,
                    .level = level,
                    .deleted = false,
                };
                if (has_metadata) self.metadata_store.items[slot_id] = mf;
            }

            // First node case
            if (self.entry_point == null) {
                self.entry_point = slot_id;
                self.max_level = level;
                return slot_id;
            }

            // Greedy descent to insertion level
            var ep = self.entry_point.?;
            var ep_dist = distFn(point, self.getPoint(ep));

            var lvl = self.max_level;
            while (lvl > level) : (lvl -= 1) {
                var changed = true;
                while (changed) {
                    changed = false;
                    const neighbors = self.getNeighborsSlice(ep, lvl);
                    for (neighbors) |nb| {
                        const d = distFn(point, self.getPoint(nb));
                        if (d < ep_dist) {
                            ep = nb;
                            ep_dist = d;
                            changed = true;
                        }
                    }
                }
                if (lvl == 0) break;
            }

            // Connect at each level from min(max_level, level) down to 0
            const insert_level = @min(self.max_level, level);
            var l: isize = @as(isize, @intCast(insert_level));
            while (l >= 0) : (l -= 1) {
                const cur_level: usize = @intCast(l);

                // Find best candidates at this level via greedy search from ep
                var best_ep = ep;
                var best_dist = distFn(point, self.getPoint(ep));

                // Greedy improve at this level
                var improved = true;
                while (improved) {
                    improved = false;
                    const neighbors = self.getNeighborsSlice(best_ep, cur_level);
                    for (neighbors) |nb| {
                        const d = distFn(point, self.getPoint(nb));
                        if (d < best_dist) {
                            best_ep = nb;
                            best_dist = d;
                            improved = true;
                        }
                    }
                }

                // Connect to best_ep and its neighbors
                try self.connectMutual(slot_id, best_ep, cur_level);

                // Also connect to neighbors of best_ep for better connectivity
                const ep_neighbors = self.getNeighborsSlice(best_ep, cur_level);
                const max_connect = if (cur_level == 0) self.config.m_base else self.config.m;
                var connected: u32 = 1; // already connected to best_ep
                for (ep_neighbors) |nb| {
                    if (connected >= max_connect) break;
                    if (nb != slot_id) {
                        try self.connectMutual(slot_id, nb, cur_level);
                        connected += 1;
                    }
                }

                // Update ep for next level
                ep = best_ep;
            }

            if (level > self.max_level) {
                self.max_level = level;
                self.entry_point = slot_id;
            }
            return slot_id;
        }

        /// Search context for search2.topK2
        const SearchContext = struct {
            idx: *const Self,

            pub fn dist(self: SearchContext, a: []const T, b: []const T) T {
                return self.idx.distance(a, b);
            }

            pub fn getPoint(self: SearchContext, id: u32) []const T {
                return self.idx.getPoint(id);
            }

            pub fn getNeighborsSlice(self: SearchContext, id: u32, level: usize) []const u32 {
                return self.idx.getNeighborsSlice(id, level);
            }

            pub fn isDeleted(self: SearchContext, id: u32) bool {
                return self.idx.isDeleted(id);
            }
        };

        pub fn searchTopK(self: *Self, query: []const T, k: usize, ef_search: usize) ![]s2.SearchResult(T) {
            std.debug.assert(query.len == self.config.dims);

            const ctx = SearchContext{ .idx = self };
            return try s2.topK2(
                T,
                SearchContext,
                self.nodes.items.len,
                self.entry_point,
                self.max_level,
                ef_search,
                query,
                k,
                self.allocator,
                ctx,
            );
        }

        pub fn search(self: *Self, query: []const T, k: usize) ![]s2.SearchResult(T) {
            return self.searchTopK(query, k, self.config.default_ef_search);
        }

        pub fn delete(self: *Self, id: u32) !void {
            if (id >= self.nodes.items.len) return error.InvalidId;
            if (self.nodes.items[id].deleted) return error.AlreadyDeleted;
            self.nodes.items[id].deleted = true;
            if (self.config.reuse_deleted_slots) {
                try self.free_slots.append(self.allocator, id);
                try self.free_vector_slots.append(self.allocator, self.nodes.items[id].vector_slot);
            }
            if (self.entry_point == id) self.entry_point = self.findNewEntryPoint(id);
        }

        fn findNewEntryPoint(self: *Self, old: u32) ?u32 {
            _ = old;
            for (self.nodes.items, 0..) |n, i| {
                if (!n.deleted) return @intCast(i);
            }
            return null;
        }

        pub fn getMetadata(self: *const Self, id: u32) ?Metadata {
            if (!has_metadata) return {};
            if (id >= self.metadata_store.items.len) return null;
            if (self.nodes.items[id].deleted) return null;
            return metadata.decode(Metadata, self.metadata_store.items[id], &self.string_table);
        }

        pub fn getMetadataFixed(self: *const Self, id: u32) ?MetaFixed {
            if (!has_metadata) return .{};
            if (id >= self.metadata_store.items.len) return null;
            if (self.nodes.items[id].deleted) return null;
            return self.metadata_store.items[id];
        }

        // =====================================================================
        // Compaction
        // =====================================================================

        /// Find the cluster center for a node via greedy descent from entry point.
        /// Used for cluster-based ordering during compaction.
        fn findCluster(self: *Self, node_id: u32) u32 {
            if (self.entry_point == null) return node_id;
            const node_point = self.getPoint(node_id);

            var current = self.entry_point.?;
            var current_dist = distFn(node_point, self.getPoint(current));

            // Greedy descent through all levels
            var level = self.max_level;
            while (true) {
                var changed = true;
                while (changed) {
                    changed = false;
                    const neighbors = self.getNeighborsSlice(current, level);
                    for (neighbors) |nb| {
                        const d = distFn(node_point, self.getPoint(nb));
                        if (d < current_dist) {
                            current = nb;
                            current_dist = d;
                            changed = true;
                        }
                    }
                }
                if (level == 0) break;
                level -= 1;
            }
            return current;
        }

        /// Cluster-based compaction: removes deleted nodes and reorders remaining
        /// nodes by (level desc, cluster) for better locality.
        pub fn compact(self: *Self) !void {
            // Count live nodes
            var live_count: usize = 0;
            for (self.nodes.items) |n| {
                if (!n.deleted) live_count += 1;
            }
            if (live_count == self.nodes.items.len) return; // Nothing to compact

            if (live_count == 0) {
                // All nodes deleted - reset to empty state
                self.nodes.clearRetainingCapacity();
                self.free_slots.clearRetainingCapacity();
                self.free_vector_slots.clearRetainingCapacity();
                if (has_metadata) self.metadata_store.clearRetainingCapacity();
                self.graph_tape.data.clearRetainingCapacity();
                self.vectors.count = 0;
                self.entry_point = null;
                self.max_level = 0;
                return;
            }

            const SlotInfo = struct { old_slot: u32, level: u16, cluster: u32 };
            var slots = try self.allocator.alloc(SlotInfo, live_count);
            defer self.allocator.free(slots);

            // Gather live nodes with their clusters
            var j: usize = 0;
            for (self.nodes.items, 0..) |n, i| {
                if (n.deleted) continue;
                slots[j] = .{
                    .old_slot = @intCast(i),
                    .level = n.level,
                    .cluster = self.findCluster(@intCast(i)),
                };
                j += 1;
            }

            // Sort by level desc, then cluster
            std.sort.heap(SlotInfo, slots, {}, struct {
                fn lessThan(_: void, a: SlotInfo, b: SlotInfo) bool {
                    if (a.level != b.level) return a.level > b.level;
                    return a.cluster < b.cluster;
                }
            }.lessThan);

            // Build old → new mapping
            var old_to_new = try self.allocator.alloc(u32, self.nodes.items.len);
            defer self.allocator.free(old_to_new);
            for (old_to_new) |*x| x.* = std.math.maxInt(u32);
            for (slots, 0..) |s, new_idx| {
                old_to_new[s.old_slot] = @intCast(new_idx);
            }

            // Create new containers
            var new_nodes = std.ArrayListUnmanaged(Node){};
            try new_nodes.ensureTotalCapacity(self.allocator, live_count);
            errdefer new_nodes.deinit(self.allocator);

            var new_vectors = try vs.VectorStore.init(self.allocator, live_count, self.config.dims, T);
            errdefer new_vectors.deinit();

            var new_tape = tape.GraphTape.init(self.allocator, .{ .m = self.config.m, .m_base = self.config.m_base });
            errdefer new_tape.deinit();

            var new_meta = if (has_metadata) std.ArrayListUnmanaged(MetaFixed){} else {};
            if (has_metadata) {
                try new_meta.ensureTotalCapacity(self.allocator, live_count);
            }
            errdefer if (has_metadata) new_meta.deinit(self.allocator);

            // Copy nodes in new order
            for (slots) |s| {
                const old_i = s.old_slot;
                const old_node = self.nodes.items[old_i];

                // Copy vector
                const old_vec = self.vectors.get(old_node.vector_slot);
                const new_vslot = try new_vectors.add(old_vec);

                // Allocate new graph tape for this node
                const graph_bytes = new_tape.config.nodeGraphBytes(old_node.level);
                const new_goff: u32 = @intCast(new_tape.data.items.len);
                try new_tape.data.appendNTimes(self.allocator, 0, graph_bytes);

                // Copy neighbor data from old tape
                const old_off = old_node.graph_offset;
                @memcpy(
                    new_tape.data.items[new_goff..][0..graph_bytes],
                    self.graph_tape.data.items[old_off..][0..graph_bytes],
                );

                new_nodes.appendAssumeCapacity(.{
                    .graph_offset = new_goff,
                    .vector_slot = new_vslot,
                    .level = old_node.level,
                    .deleted = false,
                });

                if (has_metadata) {
                    new_meta.appendAssumeCapacity(self.metadata_store.items[old_i]);
                }
            }

            // Remap all neighbor references and filter out deleted
            for (new_nodes.items) |nn| {
                for (0..@as(usize, nn.level) + 1) |lvl| {
                    var nr = new_tape.getNeighbors(nn.graph_offset, lvl);
                    const neighbor_count = nr.count();

                    // Filter and remap neighbors
                    var write_idx: usize = 0;
                    var read_idx: usize = 0;
                    while (read_idx < neighbor_count) : (read_idx += 1) {
                        const old_id = nr.get(read_idx);
                        if (old_id < old_to_new.len) {
                            const mapped = old_to_new[old_id];
                            if (mapped != std.math.maxInt(u32)) {
                                nr.set(write_idx, mapped);
                                write_idx += 1;
                            }
                        }
                    }
                    // Update count to reflect only valid neighbors
                    std.mem.writeInt(u32, @as(*[4]u8, @ptrCast(nr.tape)), @intCast(write_idx), .little);
                }
            }

            // Swap in new containers
            self.nodes.deinit(self.allocator);
            self.vectors.deinit();
            self.graph_tape.deinit();
            self.free_slots.clearRetainingCapacity();
            self.free_vector_slots.clearRetainingCapacity();
            if (has_metadata) {
                self.metadata_store.deinit(self.allocator);
                self.metadata_store = new_meta;
            }

            self.nodes = new_nodes;
            self.vectors = new_vectors;
            self.graph_tape = new_tape;

            // Update entry point
            if (self.entry_point) |ep| {
                const mapped = old_to_new[ep];
                self.entry_point = if (mapped == std.math.maxInt(u32)) self.findNewEntryPoint(0) else mapped;
            }

            // Recalculate max_level from surviving nodes
            var new_max_level: u16 = 0;
            for (self.nodes.items) |n| {
                if (n.level > new_max_level) new_max_level = n.level;
            }
            self.max_level = new_max_level;
        }

        // =====================================================================
        // Persistence
        // =====================================================================
        const persist = @import("persistence_v2.zig");

        pub fn save(self: *const Self, path: []const u8) !void {
            const node_count = self.nodes.items.len;

            // Build levels array (handle empty case)
            const levels_bytes: []u8 = if (node_count == 0)
                &[_]u8{}
            else blk: {
                const lb = try self.allocator.alloc(u8, node_count * @sizeOf(persist.NodeLevelRecord));
                const levels: [*]persist.NodeLevelRecord = @ptrCast(@alignCast(lb.ptr));
                for (self.nodes.items, 0..) |n, i| {
                    levels[i] = .{
                        .graph_offset = n.graph_offset,
                        .level = n.level,
                        .deleted = if (n.deleted) 1 else 0,
                    };
                }
                break :blk lb;
            };
            defer if (node_count > 0) self.allocator.free(levels_bytes);

            // Build metadata bytes
            const meta_bytes = if (has_metadata) blk: {
                const mf_size = @sizeOf(MetaFixed);
                const mb = try self.allocator.alloc(u8, node_count * mf_size);
                for (0..node_count) |i| {
                    const rec = std.mem.asBytes(&self.metadata_store.items[i]);
                    @memcpy(mb[i * mf_size ..][0..mf_size], rec);
                }
                break :blk mb;
            } else &[_]u8{};
            defer if (has_metadata and meta_bytes.len > 0) self.allocator.free(meta_bytes);

            const entry_raw: u64 = if (self.entry_point) |ep| @intCast(ep) else std.math.maxInt(u64);
            const string_data = if (has_metadata) self.string_table.data else &[_]u8{};

            const header = persist.FileHeaderV2{
                .metric = @intFromEnum(metric),
                .t_size = @sizeOf(T),
                .dims = self.config.dims,
                .m = self.config.m,
                .m_base = self.config.m_base,
                .ef_construction = self.config.ef_construction,
                .default_ef_search = self.config.default_ef_search,
                .node_count = node_count,
                .entry_point = entry_raw,
                .max_level = self.max_level,
                .vectors_len = self.vectors.count * self.vectors.vector_size_bytes,
                .metadata_len = meta_bytes.len,
                .levels_len = levels_bytes.len,
                .tape_len = self.graph_tape.data.items.len,
                .string_blob_len = string_data.len,
            };

            try persist.saveV2(
                path,
                header,
                self.vectors.data[0 .. self.vectors.count * self.vectors.vector_size_bytes],
                meta_bytes,
                levels_bytes,
                self.graph_tape.data.items,
                string_data,
            );
        }

        pub fn load(allocator: Allocator, path: []const u8) !Self {
            const header = try persist.loadHeaderV2(path);

            // Validate compile-time compatibility
            if (header.metric != @intFromEnum(metric)) return error.MetricMismatch;
            if (header.t_size != @sizeOf(T)) return error.TypeSizeMismatch;

            var sections = try persist.loadSectionsV2(path, header, allocator);
            defer sections.deinit();

            var self = Self{
                .allocator = allocator,
                .config = .{
                    .dims = header.dims,
                    .m = header.m,
                    .m_base = header.m_base,
                    .ef_construction = header.ef_construction,
                    .default_ef_search = header.default_ef_search,
                },
                .vectors = try vs.VectorStore.init(allocator, 0, header.dims, T),
                .graph_tape = tape.GraphTape.init(allocator, .{ .m = header.m, .m_base = header.m_base }),
                .prng = std.Random.Xoshiro256.init(0),
            };
            errdefer self.deinit();

            const node_count: usize = @intCast(header.node_count);

            // Restore vectors
            if (sections.vectors.len > 0) {
                self.vectors.data = try allocator.alignedAlloc(u8, .@"64", sections.vectors.len);
                @memcpy(self.vectors.data, sections.vectors);
                self.vectors.count = node_count;
                self.vectors.capacity = node_count;
            }

            // Restore graph tape
            if (sections.tape.len > 0) {
                try self.graph_tape.data.appendSlice(allocator, sections.tape);
            }

            // Restore nodes from levels array
            if (node_count > 0) {
                const levels: [*]const persist.NodeLevelRecord = @ptrCast(@alignCast(sections.levels.ptr));
                try self.nodes.ensureTotalCapacity(allocator, node_count);
                for (0..node_count) |i| {
                    const rec = levels[i];
                    self.nodes.appendAssumeCapacity(.{
                        .graph_offset = rec.graph_offset,
                        .vector_slot = @intCast(i),
                        .level = rec.level,
                        .deleted = rec.deleted != 0,
                    });
                }
            }

            // Restore metadata
            if (has_metadata and sections.metadata.len > 0) {
                const mf_size = @sizeOf(MetaFixed);
                try self.metadata_store.ensureTotalCapacity(allocator, node_count);
                for (0..node_count) |i| {
                    var mf: MetaFixed = undefined;
                    @memcpy(std.mem.asBytes(&mf), sections.metadata[i * mf_size ..][0..mf_size]);
                    self.metadata_store.appendAssumeCapacity(mf);
                }
            }

            // Restore string table
            if (has_metadata and sections.string_blob.len > 0) {
                self.string_table.data = try allocator.alloc(u8, sections.string_blob.len);
                @memcpy(self.string_table.data, sections.string_blob);
            }

            // Restore entry point and max level
            self.max_level = header.max_level;
            if (header.entry_point == std.math.maxInt(u64)) {
                self.entry_point = null;
            } else {
                self.entry_point = @intCast(header.entry_point);
            }

            return self;
        }
    };
}
