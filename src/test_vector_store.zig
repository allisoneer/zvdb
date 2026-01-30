const std = @import("std");
const testing = std.testing;
const vs = @import("vector_store.zig");

test "VectorStore add/get/grow/alignment" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var store = try vs.VectorStore.init(gpa.allocator(), 2, 4, f32);
    defer store.deinit();

    var v1: [4]f32 = .{ 1, 2, 3, 4 };
    var v2: [4]f32 = .{ 5, 6, 7, 8 };
    var v3: [4]f32 = .{ 9, 10, 11, 12 };

    const s1 = try store.add(std.mem.asBytes(&v1));
    const s2 = try store.add(std.mem.asBytes(&v2));
    // triggers grow
    const s3 = try store.add(std.mem.asBytes(&v3));

    // Verify 64-byte alignment
    try testing.expect(@intFromPtr(store.data.ptr) % 64 == 0);
    try testing.expectEqual(@as(u32, 0), s1);
    try testing.expectEqual(@as(u32, 1), s2);
    try testing.expectEqual(@as(u32, 2), s3);

    // Verify typed retrieval
    const r1 = store.getTyped(f32, s1, 4);
    try testing.expectEqual(@as(f32, 1), r1[0]);
    try testing.expectEqual(@as(f32, 4), r1[3]);

    const r2 = store.getTyped(f32, s2, 4);
    try testing.expectEqual(@as(f32, 7), r2[2]);

    const r3 = store.getTyped(f32, s3, 4);
    try testing.expectEqual(@as(f32, 12), r3[3]);
}

test "VectorStore empty initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var store = try vs.VectorStore.init(gpa.allocator(), 0, 4, f32);
    defer store.deinit();

    try testing.expectEqual(@as(usize, 0), store.capacity);
    try testing.expectEqual(@as(usize, 0), store.count);

    // Add should trigger grow
    var v1: [4]f32 = .{ 1, 2, 3, 4 };
    const s1 = try store.add(std.mem.asBytes(&v1));
    try testing.expectEqual(@as(u32, 0), s1);
    try testing.expectEqual(@as(usize, 1), store.count);
}

test "VectorStore set overwrites existing slot" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var store = try vs.VectorStore.init(gpa.allocator(), 2, 4, f32);
    defer store.deinit();

    var v1: [4]f32 = .{ 1, 2, 3, 4 };
    var v2: [4]f32 = .{ 10, 20, 30, 40 };

    const s1 = try store.add(std.mem.asBytes(&v1));
    store.set(s1, std.mem.asBytes(&v2));

    const r1 = store.getTyped(f32, s1, 4);
    try testing.expectEqual(@as(f32, 10), r1[0]);
    try testing.expectEqual(@as(f32, 40), r1[3]);
}

test "VectorStore get raw bytes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var store = try vs.VectorStore.init(gpa.allocator(), 2, 4, f32);
    defer store.deinit();

    var v1: [4]f32 = .{ 1, 2, 3, 4 };
    const s1 = try store.add(std.mem.asBytes(&v1));

    const raw = store.get(s1);
    try testing.expectEqual(@as(usize, 16), raw.len); // 4 * sizeof(f32)
}

test "VectorStore large dimension alignment" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    // Test with 128 dimensions (typical for embeddings)
    var store = try vs.VectorStore.init(gpa.allocator(), 10, 128, f32);
    defer store.deinit();

    try testing.expect(@intFromPtr(store.data.ptr) % 64 == 0);
    try testing.expectEqual(@as(usize, 512), store.vector_size_bytes); // 128 * 4
}
