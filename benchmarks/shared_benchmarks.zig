const std = @import("std");
const HNSW = @import("zvdb").HNSW;

pub const BenchmarkResult = struct {
    operation: []const u8,
    num_points: usize,
    dimensions: usize,
    num_queries: ?usize,
    k: ?usize,
    num_threads: ?usize,
    total_time_ns: u64,
    operations_per_second: f64,

    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s} Benchmark:\n", .{self.operation});
        try writer.print("  Points: {d}\n", .{self.num_points});
        try writer.print("  Dimensions: {d}\n", .{self.dimensions});
        if (self.num_queries) |queries| {
            try writer.print("  Queries: {d}\n", .{queries});
        }
        if (self.k) |k_value| {
            try writer.print("  k: {d}\n", .{k_value});
        }
        if (self.num_threads) |threads| {
            try writer.print("  Threads: {d}\n", .{threads});
        }
        try writer.print("  Total time: {d:.2} seconds\n", .{@as(f64, @floatFromInt(self.total_time_ns)) / 1e9});
        try writer.print("  {s} per second: {d:.2}\n", .{ self.operation, self.operations_per_second });
    }

    pub fn toCsv(self: BenchmarkResult) []const u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{s},{d},{d},{d},{d},{d},{d},{d:.2}", .{
            self.operation,
            self.num_points,
            self.dimensions,
            self.num_queries orelse 0,
            self.k orelse 0,
            self.num_threads orelse 1,
            self.total_time_ns,
            self.operations_per_second,
        }) catch unreachable;
    }
};

pub fn randomPoint(allocator: std.mem.Allocator, dim: usize) ![]f32 {
    const point = try allocator.alloc(f32, dim);
    for (point) |*v| {
        v.* = std.crypto.random.float(f32);
    }
    return point;
}

pub fn runInsertionBenchmark(allocator: std.mem.Allocator, num_points: usize, dim: usize, num_threads: ?usize) !BenchmarkResult {
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    const end = timer.lap();
    const elapsed_ns = end - start;
    const points_per_second = @as(f64, @floatFromInt(num_points)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    return BenchmarkResult{
        .operation = "Insertion",
        .num_points = num_points,
        .dimensions = dim,
        .num_queries = null,
        .k = null,
        .num_threads = num_threads,
        .total_time_ns = elapsed_ns,
        .operations_per_second = points_per_second,
    };
}

pub fn runSearchBenchmark(allocator: std.mem.Allocator, num_points: usize, dim: usize, num_queries: usize, k: usize, num_threads: ?usize) !BenchmarkResult {
    var hnsw = HNSW(f32).init(allocator, 16, 200);
    defer hnsw.deinit();

    // Insert points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    for (0..num_queries) |_| {
        const query = try randomPoint(allocator, dim);
        defer allocator.free(query);
        const results = try hnsw.search(query, k);
        allocator.free(results);
    }

    const end = timer.lap();
    const elapsed_ns = end - start;
    const queries_per_second = @as(f64, @floatFromInt(num_queries)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);

    return BenchmarkResult{
        .operation = "Search",
        .num_points = num_points,
        .dimensions = dim,
        .num_queries = num_queries,
        .k = k,
        .num_threads = num_threads,
        .total_time_ns = elapsed_ns,
        .operations_per_second = queries_per_second,
    };
}

pub const BenchmarkConfig = struct {
    num_points: usize,
    dimensions: []const usize,
    num_queries: usize,
    k_values: []const usize,
};
