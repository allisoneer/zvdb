const std = @import("std");

pub const StringTable = struct {
    data: []u8 = &.{},

    pub fn slice(self: *const StringTable, off: u32, len: u32) []const u8 {
        return self.data[@as(usize, off)..][0..@as(usize, len)];
    }

    /// Pre-allocate additional capacity for batch inserts.
    /// This allows a single reallocation for an entire batch.
    pub fn ensureAdditionalCapacity(self: *StringTable, allocator: std.mem.Allocator, add: usize) !void {
        if (add == 0) return;
        const old_len = self.data.len;
        const new_len = old_len + add;
        if (old_len == 0) {
            self.data = try allocator.alloc(u8, new_len);
        } else {
            self.data = try allocator.realloc(self.data, new_len);
        }
    }
};

// =============================================================================
// Type Classification Helpers
// =============================================================================

fn isString(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.size == .slice and p.is_const and p.child == u8,
        else => false,
    };
}

fn isEnumWithExplicitTag(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"enum" => |e| switch (@typeInfo(e.tag_type)) {
            .int => true,
            else => false,
        },
        else => false,
    };
}

fn isPrimitive(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .bool => true,
        .@"enum" => isEnumWithExplicitTag(T),
        else => false,
    };
}

fn isPrimitiveArray(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .array => |a| a.len > 0 and isPrimitive(a.child),
        else => false,
    };
}

// =============================================================================
// Validation Helper
// =============================================================================

fn validateSupportedField(comptime T: type, comptime field_path: []const u8) void {
    switch (@typeInfo(T)) {
        .pointer => |p| {
            if (!(p.size == .slice and p.is_const and p.child == u8)) {
                @compileError("Unsupported pointer type at '" ++ field_path ++ "' (only []const u8 allowed)");
            }
        },
        .array => |a| {
            if (a.len == 0) @compileError("Zero-length arrays [0]T not supported at '" ++ field_path ++ "'");
            if (isString(a.child)) @compileError("Arrays of strings [N][]const u8 not supported at '" ++ field_path ++ "'");
            switch (@typeInfo(a.child)) {
                .@"struct" => @compileError("Arrays of structs [N]S not supported at '" ++ field_path ++ "'"),
                .optional => @compileError("Arrays of optionals [N]?T not supported at '" ++ field_path ++ "'"),
                else => {},
            }
            if (!isPrimitive(a.child)) {
                @compileError("Only primitive arrays [N]T supported (T must be int/float/bool/enum) at '" ++ field_path ++ "'");
            }
        },
        .@"enum" => {
            if (!isEnumWithExplicitTag(T)) {
                @compileError("Enum without explicit integer tag type not supported at '" ++ field_path ++ "'");
            }
        },
        .int, .float, .bool => {},
        .optional => |o| validateSupportedField(o.child, field_path), // Recurse into optional
        .@"struct" => |s| {
            // Validate all fields recursively
            inline for (s.fields) |f| {
                validateSupportedField(f.type, field_path ++ "." ++ f.name);
            }
        },
        else => {
            @compileError("Unsupported field type at '" ++ field_path ++ "': " ++ @typeName(T));
        },
    }
}

// =============================================================================
// Name and Counting Helpers
// =============================================================================

fn joinName(comptime prefix: []const u8, comptime name: []const u8) []const u8 {
    if (prefix.len == 0) return name;
    return prefix ++ "__" ++ name;
}

fn joinNameZ(comptime prefix: [:0]const u8, comptime name: [:0]const u8) [:0]const u8 {
    if (prefix.len == 0) return name;
    return prefix ++ "__" ++ name;
}

const CountResult = struct { fields: usize, optionals: usize };

fn countRec(comptime T: type) CountResult {
    return switch (@typeInfo(T)) {
        .pointer => .{ .fields = 2, .optionals = 0 }, // string: off+len
        .int, .float, .bool, .@"enum" => .{ .fields = 1, .optionals = 0 },
        .array => .{ .fields = 1, .optionals = 0 }, // inline array field
        .optional => |o| blk: {
            const inner = countRec(o.child);
            break :blk .{ .fields = inner.fields, .optionals = inner.optionals + 1 };
        },
        .@"struct" => |s| blk: {
            var fcount: usize = 0;
            var oc: usize = 0;
            inline for (s.fields) |f| {
                const c = countRec(f.type);
                fcount += c.fields;
                oc += c.optionals;
            }
            break :blk .{ .fields = fcount, .optionals = oc };
        },
        else => @compileError("Unsupported type in countRec: " ++ @typeName(T)),
    };
}

fn wordsForOptionals(comptime opt_count: usize) usize {
    return if (opt_count == 0) 0 else (opt_count + 63) / 64;
}

// =============================================================================
// Presence Bit Helpers
// =============================================================================

fn setPresenceBit(comptime W: usize, words: *[W]u64, idx: usize) void {
    const word_idx = idx / 64;
    const bit: u6 = @intCast(idx % 64);
    words[word_idx] |= (@as(u64, 1) << bit);
}

fn testPresenceBit(comptime W: usize, words: [W]u64, idx: usize) bool {
    const word_idx = idx / 64;
    const bit: u6 = @intCast(idx % 64);
    return (words[word_idx] & (@as(u64, 1) << bit)) != 0;
}

// =============================================================================
// FixedOf Generator
// =============================================================================

/// Recursive field emitter - builds names and types arrays
fn emitRec(
    comptime T: type,
    comptime prefix: [:0]const u8,
    comptime names: [][:0]const u8,
    comptime types: []type,
    comptime start_idx: usize,
) usize {
    var idx = start_idx;

    switch (@typeInfo(T)) {
        .pointer => {
            // String: emit off + len fields
            names[idx] = prefix ++ "_off";
            types[idx] = u32;
            idx += 1;
            names[idx] = prefix ++ "_len";
            types[idx] = u32;
            idx += 1;
        },
        .int, .float, .bool, .@"enum" => {
            names[idx] = prefix;
            types[idx] = T;
            idx += 1;
        },
        .array => {
            // Inline array
            names[idx] = prefix;
            types[idx] = T;
            idx += 1;
        },
        .optional => |o| {
            // Optional: recurse into child (no extra field for presence, handled by presence_words)
            idx = emitRec(o.child, prefix, names, types, idx);
        },
        .@"struct" => |s| {
            // Nested struct: recurse on each field with joined name
            inline for (s.fields) |f| {
                idx = emitRec(f.type, joinNameZ(prefix, f.name), names, types, idx);
            }
        },
        else => unreachable,
    }

    return idx;
}

/// Check for name collisions in generated field names
fn checkNameCollisions(comptime names: []const [:0]const u8, comptime has_presence_words: bool) void {
    // Check for reserved "presence_words" collision
    if (has_presence_words) {
        for (names) |name| {
            if (std.mem.eql(u8, name, "presence_words")) {
                @compileError("Field name collision with reserved 'presence_words'");
            }
        }
    }

    // Check for duplicate names
    for (names, 0..) |name, i| {
        for (names[i + 1 ..]) |other| {
            if (std.mem.eql(u8, name, other)) {
                @compileError("Duplicate field name detected: '" ++ name ++ "'");
            }
        }
    }
}

/// Generate extern struct from Metadata with support for:
/// - []const u8 (strings) as off+len pairs
/// - int, float, bool, enum types
/// - Primitive arrays [N]T (inlined)
/// - Optional ?T with presence bits
/// - Nested structs (flattened with __ separator)
pub fn FixedOf(comptime Metadata: type) type {
    if (Metadata == void) return extern struct {};

    const info = @typeInfo(Metadata).@"struct";

    // Validate all fields recursively
    inline for (info.fields) |f| {
        validateSupportedField(f.type, f.name);
    }

    // Count flattened fields and optionals
    const counts = countRec(Metadata);
    const opt_count = counts.optionals;
    const field_count = counts.fields;
    const W = wordsForOptionals(opt_count);

    // Total output fields = presence_words (if any) + flattened fields
    const total_out = if (W > 0) 1 + field_count else field_count;

    // Build field names and types via recursive emission
    comptime var names: [field_count][:0]const u8 = undefined;
    comptime var types: [field_count]type = undefined;

    comptime {
        var idx: usize = 0;
        for (info.fields) |f| {
            idx = emitRec(f.type, f.name, &names, &types, idx);
        }
    }

    // Check for name collisions
    checkNameCollisions(&names, W > 0);

    // Build the extern struct
    return @Type(.{
        .@"struct" = .{
            .layout = .@"extern",
            .fields = blk: {
                var struct_fields: [total_out]std.builtin.Type.StructField = undefined;
                var out_idx: usize = 0;

                // Add presence_words header if needed
                if (W > 0) {
                    struct_fields[out_idx] = .{
                        .name = "presence_words",
                        .type = [W]u64,
                        .default_value_ptr = null,
                        .is_comptime = false,
                        .alignment = @alignOf([W]u64),
                    };
                    out_idx += 1;
                }

                // Add all flattened fields
                for (0..field_count) |i| {
                    struct_fields[out_idx] = .{
                        .name = names[i],
                        .type = types[i],
                        .default_value_ptr = null,
                        .is_comptime = false,
                        .alignment = @alignOf(types[i]),
                    };
                    out_idx += 1;
                }

                break :blk &struct_fields;
            },
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

/// Helper to append string to StringTable
fn appendString(st: *StringTable, allocator: std.mem.Allocator, s: []const u8) !struct { off: u32, len: u32 } {
    const off: u32 = @intCast(st.data.len);
    const len: u32 = @intCast(s.len);

    if (s.len > 0) {
        const old_len = st.data.len;
        if (st.data.len == 0) {
            st.data = try allocator.alloc(u8, s.len);
        } else {
            st.data = try allocator.realloc(st.data, old_len + s.len);
        }
        @memcpy(st.data[old_len..][0..s.len], s);
    }

    return .{ .off = off, .len = len };
}

/// Recursive encoder - writes value to fixed struct fields
fn encodeRec(
    comptime T: type,
    comptime prefix: [:0]const u8,
    comptime W: usize,
    value: T,
    fixed: anytype,
    words: if (W > 0) *[W]u64 else void,
    bit_idx: *usize,
    st: *StringTable,
    allocator: std.mem.Allocator,
) !void {
    switch (@typeInfo(T)) {
        .pointer => {
            // String - encode off+len
            const result = try appendString(st, allocator, value);
            @field(fixed.*, prefix ++ "_off") = result.off;
            @field(fixed.*, prefix ++ "_len") = result.len;
        },
        .int, .float, .bool, .@"enum" => {
            @field(fixed.*, prefix) = value;
        },
        .array => {
            @field(fixed.*, prefix) = value;
        },
        .optional => |o| {
            if (value) |v| {
                // Present: set bit, increment by 1, recurse
                if (W > 0) setPresenceBit(W, words, bit_idx.*);
                bit_idx.* += 1;
                try encodeRec(o.child, prefix, W, v, fixed, words, bit_idx, st, allocator);
            } else {
                // Null: skip this optional plus all nested optionals
                bit_idx.* += 1 + countRec(o.child).optionals;
            }
        },
        .@"struct" => |s| {
            inline for (s.fields) |f| {
                try encodeRec(f.type, joinNameZ(prefix, f.name), W, @field(value, f.name), fixed, words, bit_idx, st, allocator);
            }
        },
        else => unreachable,
    }
}

/// Encode Metadata to FixedOf(Metadata), appending strings to StringTable
pub fn encode(
    comptime Metadata: type,
    meta: Metadata,
    st: *StringTable,
    allocator: std.mem.Allocator,
) !FixedOf(Metadata) {
    if (Metadata == void) return .{};

    const counts = comptime countRec(Metadata);
    const W = comptime wordsForOptionals(counts.optionals);

    // Initialize with zeroes (null optionals = zeroed storage)
    var fixed: FixedOf(Metadata) = std.mem.zeroes(FixedOf(Metadata));

    // Initialize presence words if needed
    var words: if (W > 0) [W]u64 else void = if (W > 0) [_]u64{0} ** W else {};
    var bit_idx: usize = 0;

    // Recursively encode all fields
    const info = @typeInfo(Metadata).@"struct";
    inline for (info.fields) |f| {
        try encodeRec(
            f.type,
            f.name,
            W,
            @field(meta, f.name),
            &fixed,
            if (W > 0) &words else {},
            &bit_idx,
            st,
            allocator,
        );
    }

    // Copy presence words to fixed struct
    if (W > 0) {
        fixed.presence_words = words;
    }

    return fixed;
}

/// Recursive decoder - reads value from fixed struct fields
fn decodeRec(
    comptime T: type,
    comptime prefix: [:0]const u8,
    comptime W: usize,
    fixed: anytype,
    words: if (W > 0) [W]u64 else void,
    bit_idx: *usize,
    st: *const StringTable,
) T {
    switch (@typeInfo(T)) {
        .pointer => {
            // String - decode off+len
            const off = @field(fixed, prefix ++ "_off");
            const len = @field(fixed, prefix ++ "_len");
            return st.slice(off, len);
        },
        .int, .float, .bool, .@"enum" => {
            return @field(fixed, prefix);
        },
        .array => {
            return @field(fixed, prefix);
        },
        .optional => |o| {
            if (W > 0 and testPresenceBit(W, words, bit_idx.*)) {
                // Present: increment bit_idx, decode child
                bit_idx.* += 1;
                const child_val = decodeRec(o.child, prefix, W, fixed, words, bit_idx, st);
                return child_val;
            } else {
                // Null: skip this optional plus all nested optionals
                bit_idx.* += 1 + countRec(o.child).optionals;
                return null;
            }
        },
        .@"struct" => |s| {
            var result: T = undefined;
            inline for (s.fields) |f| {
                @field(result, f.name) = decodeRec(f.type, joinNameZ(prefix, f.name), W, fixed, words, bit_idx, st);
            }
            return result;
        },
        else => unreachable,
    }
}

/// Decode FixedOf(Metadata) back to Metadata using StringTable for string fields.
/// Note: The returned strings are slices into the StringTable, not owned copies.
pub fn decode(
    comptime Metadata: type,
    fixed: FixedOf(Metadata),
    st: *const StringTable,
) Metadata {
    if (Metadata == void) return {};

    const counts = comptime countRec(Metadata);
    const W = comptime wordsForOptionals(counts.optionals);

    // Get presence words (if any)
    const words: if (W > 0) [W]u64 else void = if (W > 0) fixed.presence_words else {};
    var bit_idx: usize = 0;

    var meta: Metadata = undefined;
    const info = @typeInfo(Metadata).@"struct";

    inline for (info.fields) |f| {
        @field(meta, f.name) = decodeRec(
            f.type,
            f.name,
            W,
            fixed,
            words,
            &bit_idx,
            st,
        );
    }

    return meta;
}

/// Recursive encoder for encodeInto - writes to pre-allocated StringTable
fn encodeIntoRec(
    comptime T: type,
    comptime prefix: [:0]const u8,
    comptime W: usize,
    value: T,
    fixed: anytype,
    words: if (W > 0) *[W]u64 else void,
    bit_idx: *usize,
    st: *StringTable,
    cursor: *usize,
) void {
    switch (@typeInfo(T)) {
        .pointer => {
            // String - encode off+len, copy to pre-allocated buffer
            const off: u32 = @intCast(cursor.*);
            const len: u32 = @intCast(value.len);
            if (value.len > 0) {
                @memcpy(st.data[cursor.*..][0..value.len], value);
                cursor.* += value.len;
            }
            @field(fixed.*, prefix ++ "_off") = off;
            @field(fixed.*, prefix ++ "_len") = len;
        },
        .int, .float, .bool, .@"enum" => {
            @field(fixed.*, prefix) = value;
        },
        .array => {
            @field(fixed.*, prefix) = value;
        },
        .optional => |o| {
            if (value) |v| {
                // Present: set bit, increment by 1, recurse
                if (W > 0) setPresenceBit(W, words, bit_idx.*);
                bit_idx.* += 1;
                encodeIntoRec(o.child, prefix, W, v, fixed, words, bit_idx, st, cursor);
            } else {
                // Null: skip this optional plus all nested optionals
                bit_idx.* += 1 + countRec(o.child).optionals;
            }
        },
        .@"struct" => |s| {
            inline for (s.fields) |f| {
                encodeIntoRec(f.type, joinNameZ(prefix, f.name), W, @field(value, f.name), fixed, words, bit_idx, st, cursor);
            }
        },
        else => unreachable,
    }
}

/// Encode metadata into pre-allocated StringTable space, writing at cursor position.
/// Use with totalStringBytesForBatch() and ensureAdditionalCapacity() for batch inserts.
pub fn encodeInto(
    comptime Metadata: type,
    meta: Metadata,
    st: *StringTable,
    cursor: *usize,
) FixedOf(Metadata) {
    if (Metadata == void) return .{};

    const counts = comptime countRec(Metadata);
    const W = comptime wordsForOptionals(counts.optionals);

    // Initialize with zeroes
    var fixed: FixedOf(Metadata) = std.mem.zeroes(FixedOf(Metadata));

    // Initialize presence words if needed
    var words: if (W > 0) [W]u64 else void = if (W > 0) [_]u64{0} ** W else {};
    var bit_idx: usize = 0;

    const info = @typeInfo(Metadata).@"struct";
    inline for (info.fields) |f| {
        encodeIntoRec(
            f.type,
            f.name,
            W,
            @field(meta, f.name),
            &fixed,
            if (W > 0) &words else {},
            &bit_idx,
            st,
            cursor,
        );
    }

    // Copy presence words to fixed struct
    if (W > 0) {
        fixed.presence_words = words;
    }

    return fixed;
}

/// Recursive string byte counter for totalStringBytesForBatch
fn countStringBytesRec(comptime T: type, value: T) usize {
    switch (@typeInfo(T)) {
        .pointer => return value.len,
        .int, .float, .bool, .@"enum", .array => return 0,
        .optional => |o| {
            if (value) |v| {
                return countStringBytesRec(o.child, v);
            } else {
                return 0;
            }
        },
        .@"struct" => |s| {
            var total: usize = 0;
            inline for (s.fields) |f| {
                total += countStringBytesRec(f.type, @field(value, f.name));
            }
            return total;
        },
        else => return 0,
    }
}

/// Calculate total string bytes needed for a batch of metadata records.
/// Used for pre-allocating StringTable space before batch insert.
/// Only counts strings that are present (null optional strings are skipped).
pub fn totalStringBytesForBatch(comptime Metadata: type, metas: []const Metadata) usize {
    if (Metadata == void) return 0;

    var total: usize = 0;
    for (metas) |m| {
        total += countStringBytesRec(Metadata, m);
    }
    return total;
}
