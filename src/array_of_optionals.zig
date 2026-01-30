const metadata = @import("metadata.zig");
const Bad = struct { a: [2]?u32 };
comptime {
    _ = metadata.FixedOf(Bad);
}
