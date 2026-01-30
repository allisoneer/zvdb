## Starting Situation:

- Single core is maxed out

- Multi-core usage is inefficient due to lock contention

- Performance decreases with more cores due to synchronization overhead

This scenario is common in initial implementations of concurrent systems. The good news is that there's significant room for improvement. Here's how you might approach optimizing your implementation:

1. Parallelization Strategy:

   - Instead of splitting a single operation across cores, parallelize independent operations.

   - For searches: Process multiple queries in parallel, one per thread.

   - For insertions: Batch insertions and process batches in parallel.

2. Lock Granularity:

   - Implement fine-grained locking. Lock at the node level instead of the entire structure.

   - Use read-write locks to allow concurrent reads.

3. Lock-Free Operations:

   - Implement lock-free data structures for the top levels of your HNSW graph.

   - Use atomic operations for updates that don't require complex synchronization.

4. Work Distribution:

   - Implement a work-stealing queue for better load balancing across cores.

   - This helps when operations have varying completion times.

5. SIMD Optimizations:

   - Implement SIMD instructions for distance calculations.

   - This can significantly speed up the core operation in both insertions and searches.

6. Memory Layout:

   - Optimize data layout for cache efficiency.

   - Group frequently accessed data together to improve cache hit rates.

7. Batch Processing:

   - Implement batch insertions to amortize the cost of synchronization.

   - For searches, process multiple queries in a single batch to improve cache utilization.

8. Concurrent Data Structures:

   - Replace standard containers with concurrent versions where appropriate.

   - Consider using a concurrent priority queue for the candidate list in HNSW search.

Expected Improvements:

With these optimizations, you should see:

1. Near-linear scaling with the number of cores for both insertions and searches, up to a point.

2. Significant improvement in single-core performance due to SIMD and cache optimizations.

3. Better utilization of all available cores.

Rough Estimate of Potential Gains:

Assuming an 8-core system and successful implementation of these optimizations:

1. Single-core performance: 2-4x improvement (primarily from SIMD and cache optimizations)

2. Multi-core scaling: 5-7x improvement over the optimized single-core performance

This could potentially lead to:

- Insertion rate: 80,000 - 200,000 points per second

- Search rate: 50,000 - 150,000 queries per second

Implementation Strategy:

1. Start with SIMD optimizations for distance calculations. This will give an immediate boost even in single-threaded scenarios.

2. Implement fine-grained locking and improve your parallelization strategy.

3. Add batch processing for insertions and multi-query processing for searches.

4. Optimize memory layout and implement lock-free structures for hot paths.

5. Fine-tune work distribution and implement work-stealing if necessary.

Remember to profile extensively between each optimization step. This will help you identify the most impactful changes and guide your optimization efforts.

The key is to minimize contention and maximize the amount of work that can be done independently by each core. With Zig's low-level control and performance-oriented features, you're well-positioned to implement these optimizations effectively.

### initial benchmark results
Insertion Benchmark:
  Points: 100000
  Dimensions: 128
  Total time: 11.92 seconds
  Points per second: 8392.22
Search Benchmark:
  Points in index: 100000
  Dimensions: 128
  Queries: 10000
  k: 10
  Total time: 3.73 seconds
  Queries per second: 2678.13


### Theoetical Estimates of memory usage

Below are some rough estimates on memory usage, I decided I'm fine with pretty much all of these numbers. performance is key for me, and as long as the overhead is <5% I'm very happy.


Now, let's calculate how many vectors can fit with each model:

Memory Size | Base Vectors | HNSW | Fine-grained | Lock-free | MVCC
------------|--------------|------|--------------|-----------|------
128MB       | 32,768       | 32,307| 32,153      | 32,029    | 31,781
512MB       | 131,072      | 129,230| 128,615    | 128,119   | 127,128
1GB         | 262,144      | 258,461| 257,231    | 256,239   | 254,257
4GB         | 1,048,576    | 1,033,846| 1,028,926| 1,024,959 | 1,017,031
16GB        | 4,194,304    | 4,135,385| 4,115,707| 4,099,839 | 4,068,126

Percentage of vectors that fit compared to base case:

Concurrency Model | % of Base Vectors
------------------|-------------------
HNSW only         | 98.6%
Fine-grained      | 98.1%
Lock-free         | 97.7%
MVCC              | 97.0%

Observations:
1. HNSW structure itself adds about 1.4% overhead.
2. Fine-grained locking adds another 0.5% overhead.
3. Lock-free structures add an additional 0.4% overhead.
4. MVCC adds the most overhead, about 0.7% more than lock-free.
