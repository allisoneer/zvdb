const std = @import("std");
const testing = std.testing;
const p2 = @import("persistence_v2.zig");

test "loadSectionsV2 rejects lengths larger than usize on 32-bit" {
    // This test only exercises LengthOverflow on 32-bit platforms where usize < u64
    if (@sizeOf(usize) >= @sizeOf(u64)) {
        // On 64-bit, maxInt(u64) fits in usize, so LengthOverflow won't trigger
        // This test documents expected behavior but can only run on 32-bit
        return;
    }

    var header = p2.FileHeaderV2{
        .magic = p2.MAGIC_V2,
        .version = p2.VERSION_V2,
        .metric = 0,
        .t_size = 4,
        .dims = 2,
        .m = 16,
        .m_base = 32,
        .ef_construction = 200,
        .default_ef_search = 64,
        .node_count = 0,
        .entry_point = std.math.maxInt(u64),
        .max_level = 0,
        .vectors_len = @as(u64, std.math.maxInt(usize)) + 1, // Just over usize max
        .metadata_len = 0,
        .levels_len = 0,
        .tape_len = 0,
        .string_blob_len = 0,
    };

    const path = "/tmp/len_overflow.zvd2";
    {
        const f = try std.fs.cwd().createFile(path, .{});
        defer f.close();
        try f.writer().writeAll(std.mem.asBytes(&header));
    }
    defer std.fs.cwd().deleteFile(path) catch {};

    const got_header = try p2.loadHeaderV2(path);
    const res = p2.loadSectionsV2(path, got_header, testing.allocator);
    try testing.expectError(error.LengthOverflow, res);
}

test "persistence_v2 basic header round-trip" {
    // Verify that header loading works correctly
    const header = p2.FileHeaderV2{
        .magic = p2.MAGIC_V2,
        .version = p2.VERSION_V2,
        .metric = 1,
        .t_size = 4,
        .dims = 128,
        .m = 16,
        .m_base = 32,
        .ef_construction = 200,
        .default_ef_search = 64,
        .node_count = 0,
        .entry_point = std.math.maxInt(u64),
        .max_level = 0,
        .vectors_len = 0,
        .metadata_len = 0,
        .levels_len = 0,
        .tape_len = 0,
        .string_blob_len = 0,
    };

    const path = "/tmp/header_test.zvd2";
    {
        const f = try std.fs.cwd().createFile(path, .{});
        defer f.close();
        try f.writer().writeAll(std.mem.asBytes(&header));
    }
    defer std.fs.cwd().deleteFile(path) catch {};

    const loaded = try p2.loadHeaderV2(path);
    try testing.expectEqual(p2.MAGIC_V2, loaded.magic);
    try testing.expectEqual(p2.VERSION_V2, loaded.version);
    try testing.expectEqual(@as(u32, 128), loaded.dims);
    try testing.expectEqual(@as(u8, 1), loaded.metric);
}
