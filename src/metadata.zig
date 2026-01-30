const std = @import("std");

pub const StringTable = struct {
    data: []u8 = &.{},

    pub fn slice(self: *const StringTable, off: u32, len: u32) []const u8 {
        return self.data[@as(usize, off)..][0..@as(usize, len)];
    }
};

/// Generate extern struct from Metadata where []const u8 becomes off+len pairs.
/// Supports: []const u8 (strings), int, float, bool, enum types.
pub fn FixedOf(comptime Metadata: type) type {
    if (Metadata == void) return extern struct {};

    const info = @typeInfo(Metadata).@"struct";
    const fields = info.fields;

    // Count output fields
    comptime var out_count: usize = 0;
    inline for (fields) |f| {
        switch (@typeInfo(f.type)) {
            .pointer => |p| {
                if (p.size == .slice and p.child == u8) {
                    out_count += 2; // off + len
                } else {
                    @compileError("Unsupported pointer type: " ++ f.name);
                }
            },
            .int, .float, .bool, .@"enum" => out_count += 1,
            else => @compileError("Unsupported field type: " ++ @typeName(f.type)),
        }
    }

    // Build field arrays
    var names: [out_count][:0]const u8 = undefined;
    var types: [out_count]type = undefined;

    comptime var idx: usize = 0;
    inline for (fields) |f| {
        switch (@typeInfo(f.type)) {
            .pointer => {
                names[idx] = f.name ++ "_off";
                types[idx] = u32;
                idx += 1;
                names[idx] = f.name ++ "_len";
                types[idx] = u32;
                idx += 1;
            },
            .int, .float, .bool, .@"enum" => {
                names[idx] = f.name;
                types[idx] = f.type;
                idx += 1;
            },
            else => unreachable,
        }
    }

    return @Type(.{
        .@"struct" = .{
            .layout = .@"extern",
            .fields = blk: {
                var struct_fields: [out_count]std.builtin.Type.StructField = undefined;
                for (0..out_count) |i| {
                    struct_fields[i] = .{
                        .name = names[i],
                        .type = types[i],
                        .default_value_ptr = null,
                        .is_comptime = false,
                        .alignment = @alignOf(types[i]),
                    };
                }
                break :blk &struct_fields;
            },
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

/// Encode Metadata to FixedOf(Metadata), appending strings to StringTable
pub fn encode(
    comptime Metadata: type,
    meta: Metadata,
    st: *StringTable,
    allocator: std.mem.Allocator,
) !FixedOf(Metadata) {
    if (Metadata == void) return .{};

    var fixed: FixedOf(Metadata) = undefined;
    const info = @typeInfo(Metadata).@"struct";

    inline for (info.fields) |f| {
        switch (@typeInfo(f.type)) {
            .pointer => {
                const s = @field(meta, f.name);
                const off: u32 = @intCast(st.data.len);
                const len: u32 = @intCast(s.len);

                // Append string to blob
                const old_len = st.data.len;
                if (st.data.len == 0) {
                    // First allocation
                    const new_data = try allocator.alloc(u8, s.len);
                    st.data = new_data;
                } else {
                    // Resize existing
                    const new_data = try allocator.realloc(st.data, old_len + s.len);
                    st.data = new_data;
                }
                @memcpy(st.data[old_len..][0..s.len], s);

                @field(fixed, f.name ++ "_off") = off;
                @field(fixed, f.name ++ "_len") = len;
            },
            .int, .float, .bool, .@"enum" => {
                @field(fixed, f.name) = @field(meta, f.name);
            },
            else => unreachable,
        }
    }
    return fixed;
}

/// Decode FixedOf(Metadata) back to Metadata using StringTable for string fields.
/// Note: The returned strings are slices into the StringTable, not owned copies.
pub fn decode(
    comptime Metadata: type,
    fixed: FixedOf(Metadata),
    st: *const StringTable,
) Metadata {
    if (Metadata == void) return {};

    var meta: Metadata = undefined;
    const info = @typeInfo(Metadata).@"struct";

    inline for (info.fields) |f| {
        switch (@typeInfo(f.type)) {
            .pointer => {
                const off = @field(fixed, f.name ++ "_off");
                const len = @field(fixed, f.name ++ "_len");
                @field(meta, f.name) = st.slice(off, len);
            },
            .int, .float, .bool, .@"enum" => {
                @field(meta, f.name) = @field(fixed, f.name);
            },
            else => unreachable,
        }
    }
    return meta;
}
