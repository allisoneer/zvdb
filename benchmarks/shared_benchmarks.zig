const std = @import("std");
const zvdb = @import("zvdb");
const HNSW = zvdb.HNSW;
const DistanceMetric = zvdb.DistanceMetric;

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
        writer: anytype,
    ) !void {
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
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, dim, 16, 200);
    defer hnsw.deinit();

    // Prepare batch of points
    const points = try allocator.alloc([]f32, num_points);
    defer allocator.free(points);
    for (points) |*p| {
        p.* = try randomPoint(allocator, dim);
    }
    defer for (points) |p| allocator.free(p);

    // Prepare void metadata
    const metas = try allocator.alloc(void, num_points);
    defer allocator.free(metas);
    for (metas) |*m| m.* = {};

    // Convert to const slices for insertBatch
    const const_points: []const []const f32 = @ptrCast(points);

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    // Use insertBatchThreaded for parallel insertion with explicit thread count
    const ids = try hnsw.insertBatchThreaded(const_points, metas, num_threads);
    defer allocator.free(ids);

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
    var hnsw = HNSW(f32, .squared_euclidean, void).init(allocator, dim, 16, 200);
    defer hnsw.deinit();

    // Insert points using batch
    {
        const points = try allocator.alloc([]f32, num_points);
        defer allocator.free(points);
        for (points) |*p| p.* = try randomPoint(allocator, dim);
        defer for (points) |p| allocator.free(p);

        const metas = try allocator.alloc(void, num_points);
        defer allocator.free(metas);
        for (metas) |*m| m.* = {};

        const const_points: []const []const f32 = @ptrCast(points);
        const ids = try hnsw.insertBatch(const_points, metas);
        allocator.free(ids);
    }

    const threads = num_threads orelse 1;
    const queries_per_thread = (num_queries + threads - 1) / threads;

    const ThreadCtx = struct {
        hnsw: *HNSW(f32, .squared_euclidean, void),
        allocator: std.mem.Allocator,
        query_count: usize,
        dim: usize,
        k: usize,
    };

    const thread_fn = struct {
        fn run(ctx: *const ThreadCtx) void {
            for (0..ctx.query_count) |_| {
                const query = randomPoint(ctx.allocator, ctx.dim) catch continue;
                defer ctx.allocator.free(query);
                const results = ctx.hnsw.searchDefault(query, ctx.k) catch continue;
                ctx.allocator.free(results);
            }
        }
    }.run;

    var timer = try std.time.Timer.start();
    const start = timer.lap();

    if (threads == 1) {
        // Single-threaded path
        for (0..num_queries) |_| {
            const query = try randomPoint(allocator, dim);
            defer allocator.free(query);
            const results = try hnsw.searchDefault(query, k);
            allocator.free(results);
        }
    } else {
        // Multi-threaded path
        var thread_handles = try allocator.alloc(std.Thread, threads);
        defer allocator.free(thread_handles);
        var contexts = try allocator.alloc(ThreadCtx, threads);
        defer allocator.free(contexts);

        for (0..threads) |i| {
            const remaining = num_queries -| (i * queries_per_thread);
            contexts[i] = .{
                .hnsw = &hnsw,
                .allocator = allocator,
                .query_count = @min(queries_per_thread, remaining),
                .dim = dim,
                .k = k,
            };
            thread_handles[i] = try std.Thread.spawn(.{}, thread_fn, .{&contexts[i]});
        }

        for (thread_handles) |*t| t.join();
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
