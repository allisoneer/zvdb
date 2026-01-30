const std = @import("std");
const Allocator = std.mem.Allocator;
const nr = @import("neighbors_ref.zig");

/// Manages a contiguous tape of graph neighbor data.
/// Each node's neighbors are stored inline: level 0 first, then levels 1..max.
pub const GraphTape = struct {
    data: std.ArrayListUnmanaged(u8) = .{},
    allocator: Allocator,
    config: nr.GraphConfig,

    pub fn init(allocator: Allocator, config: nr.GraphConfig) GraphTape {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *GraphTape) void {
        self.data.deinit(self.allocator);
    }

    /// Allocate space for a node at the given level.
    /// Returns the offset into the tape where the node's data starts.
    pub fn allocateNode(self: *GraphTape, level: u16) !u32 {
        const node_size = self.config.nodeGraphBytes(level);
        const off: u32 = @intCast(self.data.items.len);
        try self.data.appendNTimes(self.allocator, 0, node_size);
        return off;
    }

    /// Get a NeighborsRef for the given node offset and level.
    pub fn getNeighbors(self: *GraphTape, offset: u32, level: usize) nr.NeighborsRef {
        const base = self.data.items.ptr + offset;
        const lvl_off = if (level == 0)
            0
        else
            self.config.neighborsBaseBytes() + (level - 1) * self.config.neighborsLevelBytes();
        const max_n = if (level == 0) self.config.m_base else self.config.m;
        return .{ .tape = base + lvl_off, .max_neighbors = max_n };
    }

    /// Get const neighbors (read-only view).
    pub fn getNeighborsConst(self: *const GraphTape, offset: u32, level: usize) nr.NeighborsRef {
        const base = self.data.items.ptr + offset;
        const lvl_off = if (level == 0)
            0
        else
            self.config.neighborsBaseBytes() + (level - 1) * self.config.neighborsLevelBytes();
        const max_n = if (level == 0) self.config.m_base else self.config.m;
        return .{ .tape = base + lvl_off, .max_neighbors = max_n };
    }

    /// Get the raw bytes for a node's graph data (for copying during compaction).
    pub fn getNodeBytes(self: *const GraphTape, offset: u32, level: u16) []const u8 {
        const node_size = self.config.nodeGraphBytes(level);
        return self.data.items[offset..][0..node_size];
    }

    /// Current total size of the tape in bytes.
    pub fn size(self: *const GraphTape) usize {
        return self.data.items.len;
    }

    /// Clear all data (for reset/reuse).
    pub fn clear(self: *GraphTape) void {
        self.data.clearRetainingCapacity();
    }
};
