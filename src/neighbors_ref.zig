const std = @import("std");

/// Configuration for graph structure defining neighbor capacity at each level.
pub const GraphConfig = struct {
    m: u32 = 16,
    m_base: u32 = 32,

    /// Bytes needed for level 0 neighbors: [count:u32][neighbors:u32 * m_base]
    pub fn neighborsBaseBytes(self: GraphConfig) usize {
        return @sizeOf(u32) + self.m_base * @sizeOf(u32);
    }

    /// Bytes needed for levels > 0: [count:u32][neighbors:u32 * m]
    pub fn neighborsLevelBytes(self: GraphConfig) usize {
        return @sizeOf(u32) + self.m * @sizeOf(u32);
    }

    /// Total bytes needed for a node at given level (includes all levels 0..level).
    pub fn nodeGraphBytes(self: GraphConfig, level: u16) usize {
        return self.neighborsBaseBytes() + self.neighborsLevelBytes() * @as(usize, level);
    }
};

/// Reference to a neighbor list within the graph tape.
/// Format: [count:u32][neighbor_id:u32]...
pub const NeighborsRef = struct {
    tape: [*]u8,
    max_neighbors: u32,

    /// Get current neighbor count.
    pub fn count(self: NeighborsRef) u32 {
        return std.mem.readInt(u32, self.tape[0..4], .little);
    }

    /// Get neighbor at index i.
    pub fn get(self: NeighborsRef, i: usize) u32 {
        const off = @sizeOf(u32) + i * @sizeOf(u32);
        return std.mem.readInt(u32, @as(*const [4]u8, @ptrCast(self.tape + off)), .little);
    }

    /// Set neighbor at index i.
    pub fn set(self: NeighborsRef, i: usize, val: u32) void {
        const off = @sizeOf(u32) + i * @sizeOf(u32);
        std.mem.writeInt(u32, @as(*[4]u8, @ptrCast(self.tape + off)), val, .little);
    }

    /// Clear all neighbors (set count to 0).
    pub fn clear(self: NeighborsRef) void {
        std.mem.writeInt(u32, @as(*[4]u8, @ptrCast(self.tape)), 0, .little);
    }

    /// Append a neighbor to the list.
    pub fn pushBack(self: NeighborsRef, id: u32) void {
        const n = self.count();
        std.debug.assert(n < self.max_neighbors);
        self.set(n, id);
        std.mem.writeInt(u32, @as(*[4]u8, @ptrCast(self.tape)), n + 1, .little);
    }

    /// Check if a neighbor ID is present.
    pub fn contains(self: NeighborsRef, id: u32) bool {
        const n = self.count();
        for (0..n) |i| {
            if (self.get(i) == id) return true;
        }
        return false;
    }

    /// Iterator for traversing neighbors.
    pub const Iterator = struct {
        ref: NeighborsRef,
        i: u32 = 0,

        pub fn next(self: *Iterator) ?u32 {
            if (self.i >= self.ref.count()) return null;
            const v = self.ref.get(self.i);
            self.i += 1;
            return v;
        }

        pub fn reset(self: *Iterator) void {
            self.i = 0;
        }
    };

    pub fn iterator(self: NeighborsRef) Iterator {
        return .{ .ref = self };
    }

    /// Get slice of all neighbor IDs (for bulk operations).
    pub fn slice(self: NeighborsRef) []const u32 {
        var n = self.count();
        // Clamp to max_neighbors to prevent OOB reads from corrupted count
        if (n > self.max_neighbors) n = self.max_neighbors;
        if (n == 0) return &[_]u32{};
        const ptr: [*]const u32 = @ptrCast(@alignCast(self.tape + @sizeOf(u32)));
        return ptr[0..n];
    }
};
