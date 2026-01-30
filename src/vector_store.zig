const std = @import("std");
const Allocator = std.mem.Allocator;

/// Contiguous 64-byte aligned storage for vectors in Structure-of-Arrays (SoA) format.
/// Provides O(1) access by slot index with cache-friendly sequential layout.
pub const VectorStore = struct {
    data: []align(64) u8,
    capacity: usize,
    count: usize,
    vector_size_bytes: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, capacity: usize, dims: usize, comptime T: type) !VectorStore {
        const vec_size = dims * @sizeOf(T);
        const total = capacity * vec_size;
        const data = if (total == 0)
            @as([]align(64) u8, &[_]u8{})
        else
            try allocator.alignedAlloc(u8, .@"64", total);
        return .{
            .data = data,
            .capacity = capacity,
            .count = 0,
            .vector_size_bytes = vec_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VectorStore) void {
        if (self.data.len > 0) self.allocator.free(self.data);
        self.* = undefined;
    }

    /// Add a vector (as raw bytes) and return its slot index.
    pub fn add(self: *VectorStore, vec_bytes: []const u8) !u32 {
        std.debug.assert(vec_bytes.len == self.vector_size_bytes);
        if (self.count >= self.capacity) try self.grow();
        const slot: u32 = @intCast(self.count);
        const off = self.count * self.vector_size_bytes;
        @memcpy(self.data[off..][0..self.vector_size_bytes], vec_bytes);
        self.count += 1;
        return slot;
    }

    fn grow(self: *VectorStore) !void {
        const new_cap = @max(self.capacity * 2, @as(usize, 1));
        const new_len = new_cap * self.vector_size_bytes;
        const new_data = if (new_len == 0)
            @as([]align(64) u8, &[_]u8{})
        else
            try self.allocator.alignedAlloc(u8, .@"64", new_len);
        if (self.data.len > 0) {
            const used = self.count * self.vector_size_bytes;
            @memcpy(new_data[0..used], self.data[0..used]);
            self.allocator.free(self.data);
        }
        self.data = new_data;
        self.capacity = new_cap;
    }

    /// Get vector at slot as raw bytes.
    pub fn get(self: *const VectorStore, slot: u32) []const u8 {
        const off = @as(usize, slot) * self.vector_size_bytes;
        return self.data[off..][0..self.vector_size_bytes];
    }

    /// Get vector at slot as typed slice.
    pub fn getTyped(self: *const VectorStore, comptime T: type, slot: u32, dims: usize) []const T {
        const off = @as(usize, slot) * self.vector_size_bytes;
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr + off));
        return ptr[0..dims];
    }

    /// Set vector at existing slot (for updates/compaction).
    pub fn set(self: *VectorStore, slot: u32, vec_bytes: []const u8) void {
        std.debug.assert(vec_bytes.len == self.vector_size_bytes);
        std.debug.assert(slot < self.count);
        const off = @as(usize, slot) * self.vector_size_bytes;
        @memcpy(self.data[off..][0..self.vector_size_bytes], vec_bytes);
    }
};
