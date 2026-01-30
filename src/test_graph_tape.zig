const std = @import("std");
const testing = std.testing;
const tape = @import("graph_tape.zig");
const nr = @import("neighbors_ref.zig");

test "GraphConfig size calculations" {
    const cfg = nr.GraphConfig{ .m = 16, .m_base = 32 };

    // Level 0: count(4) + 32 neighbors * 4 = 132 bytes
    try testing.expectEqual(@as(usize, 132), cfg.neighborsBaseBytes());

    // Level 1+: count(4) + 16 neighbors * 4 = 68 bytes
    try testing.expectEqual(@as(usize, 68), cfg.neighborsLevelBytes());

    // Node at level 0: just base = 132
    try testing.expectEqual(@as(usize, 132), cfg.nodeGraphBytes(0));

    // Node at level 1: base + 1*level = 132 + 68 = 200
    try testing.expectEqual(@as(usize, 200), cfg.nodeGraphBytes(1));

    // Node at level 2: base + 2*level = 132 + 136 = 268
    try testing.expectEqual(@as(usize, 268), cfg.nodeGraphBytes(2));
}

test "GraphTape allocate and neighbor ops" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gt = tape.GraphTape.init(gpa.allocator(), .{ .m = 16, .m_base = 32 });
    defer gt.deinit();

    // Allocate node at level 2 (has levels 0, 1, 2)
    const off = try gt.allocateNode(2);
    try testing.expectEqual(@as(u32, 0), off);

    var n0 = gt.getNeighbors(off, 0);
    var n1 = gt.getNeighbors(off, 1);
    var n2 = gt.getNeighbors(off, 2);

    // Initially empty
    try testing.expectEqual(@as(u32, 0), n0.count());
    try testing.expectEqual(@as(u32, 0), n1.count());
    try testing.expectEqual(@as(u32, 0), n2.count());

    // Add neighbors to level 0
    n0.pushBack(5);
    n0.pushBack(7);
    try testing.expectEqual(@as(u32, 2), n0.count());
    try testing.expectEqual(@as(u32, 5), n0.get(0));
    try testing.expectEqual(@as(u32, 7), n0.get(1));

    // Add neighbor to level 1
    n1.pushBack(9);
    try testing.expectEqual(@as(u32, 1), n1.count());
    try testing.expectEqual(@as(u32, 9), n1.get(0));

    // Clear level 2
    n2.clear();
    try testing.expectEqual(@as(u32, 0), n2.count());

    // Iterator test
    var it = n0.iterator();
    try testing.expectEqual(@as(u32, 5), it.next().?);
    try testing.expectEqual(@as(u32, 7), it.next().?);
    try testing.expect(it.next() == null);
}

test "NeighborsRef contains and set" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gt = tape.GraphTape.init(gpa.allocator(), .{ .m = 16, .m_base = 32 });
    defer gt.deinit();

    const off = try gt.allocateNode(0);
    var n0 = gt.getNeighbors(off, 0);

    n0.pushBack(10);
    n0.pushBack(20);
    n0.pushBack(30);

    try testing.expect(n0.contains(10));
    try testing.expect(n0.contains(20));
    try testing.expect(n0.contains(30));
    try testing.expect(!n0.contains(40));

    // Modify a neighbor
    n0.set(1, 25);
    try testing.expectEqual(@as(u32, 25), n0.get(1));
    try testing.expect(n0.contains(25));
    try testing.expect(!n0.contains(20));
}

test "NeighborsRef slice" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gt = tape.GraphTape.init(gpa.allocator(), .{ .m = 16, .m_base = 32 });
    defer gt.deinit();

    const off = try gt.allocateNode(0);
    var n0 = gt.getNeighbors(off, 0);

    n0.pushBack(100);
    n0.pushBack(200);
    n0.pushBack(300);

    const s = n0.slice();
    try testing.expectEqual(@as(usize, 3), s.len);
    try testing.expectEqual(@as(u32, 100), s[0]);
    try testing.expectEqual(@as(u32, 200), s[1]);
    try testing.expectEqual(@as(u32, 300), s[2]);
}

test "GraphTape multiple nodes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gt = tape.GraphTape.init(gpa.allocator(), .{ .m = 16, .m_base = 32 });
    defer gt.deinit();

    // Allocate multiple nodes at different levels
    const off0 = try gt.allocateNode(0);
    const off1 = try gt.allocateNode(1);
    const off2 = try gt.allocateNode(2);

    try testing.expectEqual(@as(u32, 0), off0);
    try testing.expectEqual(@as(u32, 132), off1); // After level-0 node
    try testing.expectEqual(@as(u32, 332), off2); // After level-1 node (132 + 200)

    // Each node's neighbors are independent
    var n0_0 = gt.getNeighbors(off0, 0);
    var n1_0 = gt.getNeighbors(off1, 0);
    var n2_0 = gt.getNeighbors(off2, 0);

    n0_0.pushBack(1);
    n1_0.pushBack(2);
    n2_0.pushBack(3);

    try testing.expectEqual(@as(u32, 1), n0_0.get(0));
    try testing.expectEqual(@as(u32, 2), n1_0.get(0));
    try testing.expectEqual(@as(u32, 3), n2_0.get(0));
}

test "GraphTape getNodeBytes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gt = tape.GraphTape.init(gpa.allocator(), .{ .m = 16, .m_base = 32 });
    defer gt.deinit();

    const off = try gt.allocateNode(1);
    const bytes = gt.getNodeBytes(off, 1);

    // Level-1 node should be 200 bytes
    try testing.expectEqual(@as(usize, 200), bytes.len);
}
