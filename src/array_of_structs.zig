const metadata = @import("metadata.zig");
const S = struct { x: u8 };
const Bad = struct { a: [3]S };
comptime {
    _ = metadata.FixedOf(Bad);
}
