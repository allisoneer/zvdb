const metadata = @import("metadata.zig");
const Bad = struct { a: [2][]const u8 };
comptime {
    _ = metadata.FixedOf(Bad);
}
