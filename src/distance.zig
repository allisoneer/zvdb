const std = @import("std");

pub const DistanceMetric = enum(u8) {
    squared_euclidean,
    euclidean,
    cosine,
    dot_product,
};

/// Returns a comptime-specialized distance function for the given metric.
/// All metrics return "lower is better" for min-heap compatibility.
/// For f32/f64 types, uses SIMD acceleration with automatic lane selection.
pub fn distanceFn(comptime T: type, comptime metric: DistanceMetric) fn ([]const T, []const T) T {
    return switch (metric) {
        .squared_euclidean => squaredEuclidean(T),
        .euclidean => euclideanDist(T),
        .cosine => cosineDist(T),
        .dot_product => dotProductNegative(T),
    };
}

fn squaredEuclidean(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            std.debug.assert(a.len == b.len);
            if (T == f32 or T == f64) {
                return simdSquaredEuclidean(T, a, b);
            }
            var sum: T = 0;
            for (a, b) |av, bv| {
                const d = av - bv;
                sum += d * d;
            }
            return sum;
        }
    }.f;
}

fn euclideanDist(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            if (T == f32 or T == f64) {
                return std.math.sqrt(simdSquaredEuclidean(T, a, b));
            }
            return std.math.sqrt(squaredEuclidean(T)(a, b));
        }
    }.f;
}

fn cosineDist(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            std.debug.assert(a.len == b.len);
            var dot: T = 0;
            var na: T = 0;
            var nb: T = 0;
            if (T == f32 or T == f64) {
                simdDotNorms(T, a, b, &dot, &na, &nb);
            } else {
                for (a, b) |av, bv| {
                    dot += av * bv;
                    na += av * av;
                    nb += bv * bv;
                }
            }
            if (na == 0 or nb == 0) return 1;
            return 1 - (dot / (std.math.sqrt(na) * std.math.sqrt(nb)));
        }
    }.f;
}

fn dotProductNegative(comptime T: type) fn ([]const T, []const T) T {
    return struct {
        fn f(a: []const T, b: []const T) T {
            std.debug.assert(a.len == b.len);
            if (T == f32 or T == f64) {
                return -simdDot(T, a, b);
            }
            var sum: T = 0;
            for (a, b) |av, bv| sum += av * bv;
            return -sum;
        }
    }.f;
}

// =============================================================================
// SIMD Helpers - Zig selects optimal vector width for target CPU
// =============================================================================

/// SIMD squared Euclidean distance: sum((a-b)^2)
/// Uses vector accumulator with FMA for reduced reduction overhead.
fn simdSquaredEuclidean(comptime T: type, a: []const T, b: []const T) T {
    const lanes = std.simd.suggestVectorLength(T) orelse 8;
    const Vec = @Vector(lanes, T);
    var acc_v: Vec = @splat(0);
    const n = a.len / lanes;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const off = i * lanes;
        const va: Vec = a[off..][0..lanes].*;
        const vb: Vec = b[off..][0..lanes].*;
        const d = va - vb;
        acc_v = @mulAdd(Vec, d, d, acc_v); // FMA: acc += d*d
    }
    var acc: T = @reduce(.Add, acc_v);
    // Scalar tail for remaining elements
    const start = n * lanes;
    for (a[start..], b[start..]) |av, bv| {
        const d = av - bv;
        acc += d * d;
    }
    return acc;
}

/// SIMD dot product: sum(a*b)
/// Uses vector accumulator with FMA for reduced reduction overhead.
fn simdDot(comptime T: type, a: []const T, b: []const T) T {
    const lanes = std.simd.suggestVectorLength(T) orelse 8;
    const Vec = @Vector(lanes, T);
    var acc_v: Vec = @splat(0);
    const n = a.len / lanes;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const off = i * lanes;
        const va: Vec = a[off..][0..lanes].*;
        const vb: Vec = b[off..][0..lanes].*;
        acc_v = @mulAdd(Vec, va, vb, acc_v);
    }
    var acc: T = @reduce(.Add, acc_v);
    // Scalar tail
    const start = n * lanes;
    for (a[start..], b[start..]) |av, bv| acc += av * bv;
    return acc;
}

/// SIMD dot product and squared norms in one pass (for cosine distance)
/// Uses three vector accumulators with FMA for reduced reduction overhead.
fn simdDotNorms(comptime T: type, a: []const T, b: []const T, dot_out: *T, na_out: *T, nb_out: *T) void {
    const lanes = std.simd.suggestVectorLength(T) orelse 8;
    const Vec = @Vector(lanes, T);
    var dot_v: Vec = @splat(0);
    var na_v: Vec = @splat(0);
    var nb_v: Vec = @splat(0);

    const n = a.len / lanes;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const off = i * lanes;
        const va: Vec = a[off..][0..lanes].*;
        const vb: Vec = b[off..][0..lanes].*;
        dot_v = @mulAdd(Vec, va, vb, dot_v);
        na_v = @mulAdd(Vec, va, va, na_v);
        nb_v = @mulAdd(Vec, vb, vb, nb_v);
    }
    var dot: T = @reduce(.Add, dot_v);
    var na: T = @reduce(.Add, na_v);
    var nb: T = @reduce(.Add, nb_v);
    // Scalar tail
    const start = n * lanes;
    for (a[start..], b[start..]) |av, bv| {
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    dot_out.* = dot;
    na_out.* = na;
    nb_out.* = nb;
}
