const std = @import("std");

pub const MAGIC_V2: u32 = 0x5A564432; // 'ZVD2'
pub const VERSION_V2: u16 = 2;

/// Header for HNSWv2 persistence format.
/// Layout: header | vectors | metadata | levels | graph_tape | string_blob
pub const FileHeaderV2 = extern struct {
    magic: u32 = MAGIC_V2,
    version: u16 = VERSION_V2,
    metric: u8,
    t_size: u8,
    dims: u32,
    m: u32,
    m_base: u32,
    ef_construction: u32,
    default_ef_search: u32,
    node_count: u64,
    entry_point: u64, // maxInt(u64) means null
    max_level: u16,
    _pad: u16 = 0,
    vectors_len: u64,
    metadata_len: u64,
    levels_len: u64,
    tape_len: u64,
    string_blob_len: u64,
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Per-node level record: graph_offset and level.
pub const NodeLevelRecord = extern struct {
    graph_offset: u32,
    level: u16,
    deleted: u8,
    _pad: u8 = 0,
};

pub fn saveV2(
    path: []const u8,
    header: FileHeaderV2,
    vectors_bytes: []const u8,
    metadata_bytes: []const u8,
    levels_bytes: []const u8,
    tape_bytes: []const u8,
    string_blob: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var writer = file.writer(&buf);

    try writer.interface.writeStruct(header, .little);
    if (vectors_bytes.len > 0) try writer.interface.writeAll(vectors_bytes);
    if (metadata_bytes.len > 0) try writer.interface.writeAll(metadata_bytes);
    if (levels_bytes.len > 0) try writer.interface.writeAll(levels_bytes);
    if (tape_bytes.len > 0) try writer.interface.writeAll(tape_bytes);
    if (string_blob.len > 0) try writer.interface.writeAll(string_blob);
    try writer.interface.flush();
}

pub fn loadHeaderV2(path: []const u8) !FileHeaderV2 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var reader = file.reader(&buf);

    const header = try reader.interface.takeStruct(FileHeaderV2, .little);
    if (header.magic != MAGIC_V2) return error.InvalidMagic;
    if (header.version != VERSION_V2) return error.UnsupportedVersion;
    return header;
}

/// Read all sections from file after header.
pub const LoadedSections = struct {
    vectors: []u8,
    metadata: []u8,
    levels: []u8,
    tape: []u8,
    string_blob: []u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoadedSections) void {
        if (self.vectors.len > 0) self.allocator.free(self.vectors);
        if (self.metadata.len > 0) self.allocator.free(self.metadata);
        if (self.levels.len > 0) self.allocator.free(self.levels);
        if (self.tape.len > 0) self.allocator.free(self.tape);
        if (self.string_blob.len > 0) self.allocator.free(self.string_blob);
    }
};

pub fn loadSectionsV2(path: []const u8, header: FileHeaderV2, allocator: std.mem.Allocator) !LoadedSections {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buf: [4096]u8 = undefined;
    var reader = file.reader(&buf);

    // Skip header using reader.seekTo (not file.seekTo)
    try reader.seekTo(@sizeOf(FileHeaderV2));

    var sections = LoadedSections{
        .vectors = &.{},
        .metadata = &.{},
        .levels = &.{},
        .tape = &.{},
        .string_blob = &.{},
        .allocator = allocator,
    };
    errdefer sections.deinit();

    const max_usize = std.math.maxInt(usize);

    if (header.vectors_len > 0) {
        if (header.vectors_len > max_usize) return error.LengthOverflow;
        const vlen: usize = @intCast(header.vectors_len);
        sections.vectors = try allocator.alloc(u8, vlen);
        try reader.interface.readSliceAll(sections.vectors);
    }
    if (header.metadata_len > 0) {
        if (header.metadata_len > max_usize) return error.LengthOverflow;
        const mlen: usize = @intCast(header.metadata_len);
        sections.metadata = try allocator.alloc(u8, mlen);
        try reader.interface.readSliceAll(sections.metadata);
    }
    if (header.levels_len > 0) {
        if (header.levels_len > max_usize) return error.LengthOverflow;
        const llen: usize = @intCast(header.levels_len);
        sections.levels = try allocator.alloc(u8, llen);
        try reader.interface.readSliceAll(sections.levels);
    }
    if (header.tape_len > 0) {
        if (header.tape_len > max_usize) return error.LengthOverflow;
        const tlen: usize = @intCast(header.tape_len);
        sections.tape = try allocator.alloc(u8, tlen);
        try reader.interface.readSliceAll(sections.tape);
    }
    if (header.string_blob_len > 0) {
        if (header.string_blob_len > max_usize) return error.LengthOverflow;
        const slen: usize = @intCast(header.string_blob_len);
        sections.string_blob = try allocator.alloc(u8, slen);
        try reader.interface.readSliceAll(sections.string_blob);
    }

    return sections;
}
