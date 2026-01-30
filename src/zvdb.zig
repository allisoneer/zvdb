// zvdb - High-performance vector database
// Single-threaded HNSW implementation with USearch-style optimizations

pub const distance = @import("distance.zig");
pub const metadata = @import("metadata.zig");
pub const hnsw_v2 = @import("hnsw_v2.zig");
pub const vector_store = @import("vector_store.zig");
pub const neighbors_ref = @import("neighbors_ref.zig");
pub const graph_tape = @import("graph_tape.zig");
pub const search = @import("search2.zig");
pub const persistence = @import("persistence_v2.zig");

pub const DistanceMetric = distance.DistanceMetric;

/// Primary HNSW index type - single-threaded with USearch-style optimizations.
/// Uses dense node storage, contiguous vector matrix (SoA), compact inline
/// neighbor tape, and SIMD-optimized distance calculations.
pub fn HNSW(comptime T: type, comptime metric: DistanceMetric, comptime Metadata: type) type {
    return hnsw_v2.HNSWv2(T, metric, Metadata);
}

/// Search result type containing node ID and distance.
pub fn SearchResult(comptime T: type) type {
    return search.SearchResult(T);
}

// Re-export legacy modules for backward compatibility during migration
// (will be removed in future versions)
pub const legacy = struct {
    pub const hnsw = @import("hnsw.zig");
    pub const search = @import("search.zig");
    pub const persistence = @import("persistence.zig");
};
