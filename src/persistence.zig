const std = @import("std");

pub const MAGIC: u32 = 0x5A564442; // 'ZVDB'
pub const VERSION: u16 = 1;

pub const FileHeader = extern struct {
    magic: u32 = MAGIC,
    version: u16 = VERSION,
    metric: u8,
    t_size: u8,
    dims: u32,
    m: u16,
    ef_construction: u16,
    node_count: u64,
    entry_point: u64,
    max_level: u32,
    string_blob_len: u64,
    _reserved: u64 = 0,
};

pub fn save(
    path: []const u8,
    header: FileHeader,
    vectors_bytes: []const u8,
    metas_bytes: []const u8,
    graph_bytes: []const u8,
    string_blob: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var writer = file.writer(&buf);

    try writer.interface.writeStruct(header, .little);
    if (vectors_bytes.len > 0) try writer.interface.writeAll(vectors_bytes);
    if (metas_bytes.len > 0) try writer.interface.writeAll(metas_bytes);
    if (graph_bytes.len > 0) try writer.interface.writeAll(graph_bytes);
    if (string_blob.len > 0) try writer.interface.writeAll(string_blob);
    try writer.interface.flush();
}

pub fn loadHeader(path: []const u8) !FileHeader {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var reader = file.reader(&buf);

    const header = try reader.interface.takeStruct(FileHeader, .little);
    if (header.magic != MAGIC) return error.InvalidMagic;
    if (header.version != VERSION) return error.UnsupportedVersion;
    return header;
}

/// Serialize all node vectors to contiguous bytes
pub fn serializeVectors(
    comptime T: type,
    nodes: anytype,
    node_count: usize,
    dims: usize,
    allocator: std.mem.Allocator,
) ![]u8 {
    const elem_size = @sizeOf(T);
    const total_len: usize = node_count * dims * elem_size;
    if (total_len == 0) return try allocator.alloc(u8, 0);

    const out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out);

    var off: usize = 0;
    for (0..node_count) |id| {
        const node = nodes.get(id) orelse return error.MissingNode;
        if (node.point.len != dims) return error.InvalidDims;
        const bytes = std.mem.sliceAsBytes(node.point);
        @memcpy(out[off..][0..bytes.len], bytes);
        off += bytes.len;
    }
    return out;
}

/// Deserialize vectors from bytes, allocating per-node slices
pub fn deserializeVectors(
    comptime T: type,
    data: []const u8,
    node_count: usize,
    dims: usize,
    allocator: std.mem.Allocator,
) ![][]T {
    const elem_size = @sizeOf(T);
    const expected_len: usize = node_count * dims * elem_size;
    if (data.len != expected_len) return error.SizeMismatch;

    const points = try allocator.alloc([]T, node_count);
    errdefer allocator.free(points);

    var off: usize = 0;
    for (0..node_count) |id| {
        const p = try allocator.alloc(T, dims);
        const src = data[off..][0..(dims * elem_size)];
        @memcpy(std.mem.sliceAsBytes(p), src);
        points[id] = p;
        off += dims * elem_size;
    }
    return points;
}

/// Serialize all node MetaFixed structs to contiguous bytes
pub fn serializeMetadata(
    comptime MetaFixed: type,
    nodes: anytype,
    node_count: usize,
    allocator: std.mem.Allocator,
) ![]u8 {
    const rec_size = @sizeOf(MetaFixed);
    if (rec_size == 0) return try allocator.alloc(u8, 0);

    const total_len: usize = node_count * rec_size;
    const out = try allocator.alloc(u8, total_len);
    errdefer allocator.free(out);

    var off: usize = 0;
    for (0..node_count) |id| {
        const node = nodes.get(id) orelse return error.MissingNode;
        const rec_bytes = std.mem.asBytes(&node.meta_fixed);
        @memcpy(out[off..][0..rec_size], rec_bytes);
        off += rec_size;
    }
    return out;
}

/// Deserialize MetaFixed array from bytes
pub fn deserializeMetadata(
    comptime MetaFixed: type,
    data: []const u8,
    node_count: usize,
    allocator: std.mem.Allocator,
) ![]MetaFixed {
    const rec_size = @sizeOf(MetaFixed);
    if (rec_size == 0) return try allocator.alloc(MetaFixed, 0);
    if (data.len != node_count * rec_size) return error.SizeMismatch;

    const metas = try allocator.alloc(MetaFixed, node_count);
    errdefer allocator.free(metas);

    var off: usize = 0;
    for (0..node_count) |i| {
        const src = data[off..][0..rec_size];
        @memcpy(std.mem.asBytes(&metas[i]), src);
        off += rec_size;
    }
    return metas;
}

/// Serialize graph connections to bytes.
/// Format: For each node in order: [level_count: u8][for each level: [neighbor_count: u32][neighbor_ids: u32[]]]
pub fn serializeGraph(
    nodes: anytype,
    node_count: usize,
    allocator: std.mem.Allocator,
) ![]u8 {
    var buffer: std.ArrayList(u8) = .empty;
    errdefer buffer.deinit(allocator);

    for (0..node_count) |id| {
        if (nodes.get(id)) |node| {
            // Write level count
            const level_count: u8 = @intCast(node.connections.len);
            try buffer.append(allocator, level_count);

            // Write connections for each level
            for (node.connections) |level_conns| {
                const neighbor_count: u32 = @intCast(level_conns.items.len);
                try buffer.appendSlice(allocator, std.mem.asBytes(&neighbor_count));
                for (level_conns.items) |nb| {
                    const nb_id: u32 = @intCast(nb);
                    try buffer.appendSlice(allocator, std.mem.asBytes(&nb_id));
                }
            }
        } else {
            // Empty node
            try buffer.append(allocator, 0);
        }
    }

    return buffer.toOwnedSlice(allocator);
}

/// Deserialize graph connections from bytes.
pub fn deserializeGraph(
    data: []const u8,
    node_count: usize,
    allocator: std.mem.Allocator,
) ![][]std.ArrayList(usize) {
    var connections = try allocator.alloc([]std.ArrayList(usize), node_count);
    errdefer allocator.free(connections);

    var offset: usize = 0;
    for (0..node_count) |id| {
        if (offset >= data.len) {
            connections[id] = &.{};
            continue;
        }

        const level_count = data[offset];
        offset += 1;

        if (level_count == 0) {
            connections[id] = &.{};
            continue;
        }

        const levels = try allocator.alloc(std.ArrayList(usize), level_count);
        errdefer allocator.free(levels);

        for (0..level_count) |level| {
            const neighbor_count = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset += 4;

            levels[level] = std.ArrayList(usize){};
            try levels[level].ensureTotalCapacity(allocator, neighbor_count);

            for (0..neighbor_count) |_| {
                const nb_id = std.mem.readInt(u32, data[offset..][0..4], .little);
                offset += 4;
                try levels[level].append(allocator, @intCast(nb_id));
            }
        }

        connections[id] = levels;
    }

    return connections;
}
