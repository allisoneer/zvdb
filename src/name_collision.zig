const metadata = @import("metadata.zig");
const Bad = struct { presence_words: u8, a: ?u32 };
comptime {
    _ = metadata.FixedOf(Bad);
}
