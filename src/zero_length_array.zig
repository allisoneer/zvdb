const metadata = @import("metadata.zig");
const Bad = struct { a: [0]u8 };
comptime {
    _ = metadata.FixedOf(Bad);
}
