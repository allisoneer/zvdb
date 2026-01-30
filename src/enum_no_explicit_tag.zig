const metadata = @import("metadata.zig");
const Bad = struct { e: enum { A, B } };
comptime {
    _ = metadata.FixedOf(Bad);
}
