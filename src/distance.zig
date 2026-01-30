const std = @import("std");

pub const DistanceMetric = enum(u8) {
    squared_euclidean,
    euclidean,
    cosine,
    dot_product,
};

/// Returns a comptime-specialized distance function for the given metric.
/// All metrics return "lower is better" for min-heap compatibility.
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
            for (a, b) |av, bv| {
                dot += av * bv;
                na += av * av;
                nb += bv * bv;
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
            var sum: T = 0;
            for (a, b) |av, bv| sum += av * bv;
            return -sum;
        }
    }.f;
}
