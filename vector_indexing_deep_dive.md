# Deep Dive: Vector Indexing - Types, Techniques, Mathematics & Production Patterns

## Table of Contents
1. [Index Fundamentals](#index-fundamentals)
2. [Flat Indexes (Exact Search)](#flat-indexes-exact-search)
3. [Tree-Based Indexes](#tree-based-indexes)
4. [Hash-Based Indexes (LSH)](#hash-based-indexes-lsh)
5. [Graph-Based Indexes (HNSW)](#graph-based-indexes-hnsw)
6. [Clustering-Based Indexes (IVF)](#clustering-based-indexes-ivf)
7. [Quantization Techniques](#quantization-techniques)
8. [Hybrid & Composite Indexes](#hybrid--composite-indexes)
9. [Production Patterns & Benchmarks](#production-patterns--benchmarks)
10. [Decision Framework](#decision-framework)

---

# Index Fundamentals

## The Core Problem

**Given:**
- Database of N vectors: D = {v₁, v₂, ..., vₙ} where vᵢ ∈ ℝᵈ
- Query vector: q ∈ ℝᵈ
- Distance metric: dist(q, vᵢ)

**Goal:** Find k nearest neighbors efficiently

**Naive approach (Brute Force):**
```
for each vector v in D:
    compute dist(q, v)
sort by distance
return top k

Complexity: O(N × d)
For 1M vectors × 768 dims = 768M operations per query!
```

## Index Trade-offs Triangle

```
        Accuracy
           △
          /│\
         / │ \
        /  │  \
       /   │   \
      /    │    \
     /     │     \
    /______|______\
  Speed         Memory

You can optimize for 2, but not all 3:
- Fast + Accurate = High memory (Graph indexes)
- Fast + Low memory = Lower accuracy (Quantized indexes)  
- Accurate + Low memory = Slower (Compressed indexes with decompression)
```

## Distance Metrics

### Euclidean Distance (L2)

**Formula:**
```
dist(a, b) = √(Σᵢ(aᵢ - bᵢ)²)
           = ||a - b||₂
```

**Implementation:**
```python
def euclidean_distance(a, b):
    """L2 distance"""
    diff = a - b
    return np.sqrt(np.sum(diff ** 2))

# Optimized (no sqrt for comparison)
def euclidean_distance_squared(a, b):
    """L2² distance (faster, same ordering)"""
    diff = a - b
    return np.sum(diff ** 2)
```

**Properties:**
- Triangle inequality: dist(a, c) ≤ dist(a, b) + dist(b, c)
- Metric space (enables pruning in tree-based indexes)
- Sensitive to magnitude

### Cosine Similarity → Distance

**Formula:**
```
similarity(a, b) = (a · b) / (||a|| × ||b||)
distance(a, b) = 1 - similarity(a, b)
```

**Implementation:**
```python
def cosine_similarity(a, b):
    """Cosine similarity"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Optimized for unit vectors
def cosine_similarity_normalized(a, b):
    """For pre-normalized vectors, cosine = dot product"""
    return np.dot(a, b)
```

**Properties:**
- Range: [-1, 1] for similarity, [0, 2] for distance
- Invariant to magnitude (only direction matters)
- Most common for text embeddings

### Inner Product (Dot Product)

**Formula:**
```
IP(a, b) = a · b = Σᵢ aᵢbᵢ
```

**Implementation:**
```python
def inner_product(a, b):
    return np.dot(a, b)
```

**Properties:**
- NOT a metric (no triangle inequality)
- Fast to compute
- For normalized vectors: IP = cosine similarity

### Manhattan Distance (L1)

**Formula:**
```
dist(a, b) = Σᵢ |aᵢ - bᵢ|
```

**Properties:**
- Faster to compute than L2
- More robust to outliers
- Used in some specialized applications

---

# Flat Indexes (Exact Search)

## How It Works

**Storage:**
```
Vectors stored in contiguous memory:
[v₁₁, v₁₂, ..., v₁ᵈ, v₂₁, v₂₂, ..., v₂ᵈ, ..., vₙᵈ]
```

**Search:**
```python
def flat_search(query, vectors, k):
    """Brute force search"""
    distances = []
    
    for i, vec in enumerate(vectors):
        dist = compute_distance(query, vec)
        distances.append((i, dist))
    
    # Sort and return top-k
    distances.sort(key=lambda x: x[1])
    return distances[:k]

# Complexity: O(N × d)
```

## Optimizations

### SIMD (Single Instruction, Multiple Data)

**Concept:** Process multiple values in parallel

```cpp
// Scalar (one at a time)
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += a[i] * b[i];  // One multiplication
}

// SIMD (8 at a time with AVX)
__m256 sum_vec = _mm256_setzero_ps();
for (int i = 0; i < n; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&a[i]);  // Load 8 floats
    __m256 b_vec = _mm256_loadu_ps(&b[i]);  // Load 8 floats
    __m256 prod = _mm256_mul_ps(a_vec, b_vec);  // 8 multiplications
    sum_vec = _mm256_add_ps(sum_vec, prod);
}
// Horizontal sum
float sum = horizontal_sum_256(sum_vec);
```

**Speedup:** 4-8× faster on modern CPUs

### Memory Layout Optimization

```python
# Bad: List of vectors (pointer chasing)
vectors = [
    np.array([0.1, 0.2, 0.3]),  # Separate allocation
    np.array([0.4, 0.5, 0.6]),  # Separate allocation
]

# Good: Contiguous array (cache-friendly)
vectors = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
], dtype=np.float32)  # Single contiguous block

# Memory access pattern:
# Bad:  [ptr1] -> [data] ... [ptr2] -> [data] (cache misses)
# Good: [data][data][data] (sequential, cache hits)
```

### Batch Processing

```python
def flat_search_batch(queries, vectors, k):
    """Process multiple queries at once"""
    # queries: [batch_size, d]
    # vectors: [N, d]
    
    # Matrix multiplication: [batch_size, d] × [d, N] = [batch_size, N]
    distances = np.dot(queries, vectors.T)
    
    # Top-k for each query
    top_k_indices = np.argsort(distances, axis=1)[:, -k:]
    
    return top_k_indices

# Speedup: Leverages BLAS (optimized matrix ops)
```

## Pros & Cons

**Pros:**
✅ **100% accuracy** (exact search)
✅ **Simple** (no training, no parameters)
✅ **Deterministic** (same results every time)
✅ **Low memory overhead** (just stores vectors)
✅ **Good for small datasets** (<100k vectors)

**Cons:**
❌ **O(N × d) complexity** (slow for large N)
❌ **No speedup with index** (must scan all vectors)
❌ **Not scalable** to millions of vectors
❌ **Latency grows linearly** with dataset size

**Production Use:**
- Verification/baseline
- Small datasets (<100k)
- When 100% accuracy is required

---

# Tree-Based Indexes

## K-D Tree (K-Dimensional Tree)

### How It Works

**Structure:** Binary tree that recursively partitions space

```
Build algorithm:
1. Choose splitting dimension (e.g., highest variance)
2. Find median value in that dimension
3. Split vectors into left (< median) and right (≥ median)
4. Recurse on both sides

Example (2D):
       (x=5)
      /      \
   (y=3)    (y=7)
   /  \      /  \
  A   B     C   D

A, B, C, D are leaf nodes containing vectors
```

**Implementation:**
```python
class KDNode:
    def __init__(self, point, left=None, right=None, axis=None):
        self.point = point
        self.left = left
        self.right = right
        self.axis = axis

class KDTree:
    def __init__(self, points, depth=0):
        if not points:
            self.root = None
            return
        
        k = len(points[0])  # Dimensionality
        axis = depth % k
        
        # Sort points by current axis
        points.sort(key=lambda x: x[axis])
        median = len(points) // 2
        
        # Create node and recurse
        self.root = KDNode(
            point=points[median],
            left=KDTree(points[:median], depth + 1).root if median > 0 else None,
            right=KDTree(points[median + 1:], depth + 1).root if median + 1 < len(points) else None,
            axis=axis
        )
    
    def search(self, query, k=1):
        """k-NN search"""
        return self._search_recursive(self.root, query, k, 0)
    
    def _search_recursive(self, node, query, k, depth):
        if node is None:
            return []
        
        axis = depth % len(query)
        
        # Decide which branch to explore first
        if query[axis] < node.point[axis]:
            near_branch = node.left
            far_branch = node.right
        else:
            near_branch = node.right
            far_branch = node.left
        
        # Search near branch
        candidates = self._search_recursive(near_branch, query, k, depth + 1)
        
        # Add current node
        dist = euclidean_distance(query, node.point)
        candidates.append((node.point, dist))
        
        # Keep only top-k
        candidates.sort(key=lambda x: x[1])
        candidates = candidates[:k]
        
        # Check if we need to explore far branch
        # (if sphere around query intersects far branch)
        if len(candidates) < k or abs(query[axis] - node.point[axis]) < candidates[-1][1]:
            far_candidates = self._search_recursive(far_branch, query, k, depth + 1)
            candidates.extend(far_candidates)
            candidates.sort(key=lambda x: x[1])
            candidates = candidates[:k]
        
        return candidates
```

**Search Complexity:**
- **Best case:** O(log N) - perfectly balanced tree
- **Average case:** O(log N) for low dimensions (d < 20)
- **Worst case:** O(N) - degenerate tree or high dimensions

### The Curse of Dimensionality

**Why K-D Trees Fail in High Dimensions:**

```
Problem: In high dimensions, all points are almost equidistant

Example (768 dimensions):
- Distance to nearest neighbor: 10.5
- Distance to random point: 10.8
- Relative difference: 2.9%

→ Can't prune branches effectively
→ Must explore most of the tree
→ Degrades to O(N)
```

**Empirical rule:** K-D trees only work well for d < 20

## Ball Tree

### How It Works

**Structure:** Recursively partition points into nested hyperspheres

```
Each node represents a ball:
- Center: c
- Radius: r (contains all child points)

Build:
1. Find centroid of points
2. Find point farthest from centroid (radius)
3. Split points into two balls
4. Recurse

Example:
       Ball(c₁, r₁)
       /           \
  Ball(c₂, r₂)   Ball(c₃, r₃)
  /      \        /      \
 ...    ...     ...     ...
```

**Implementation:**
```python
class BallNode:
    def __init__(self, center, radius, points=None, left=None, right=None):
        self.center = center
        self.radius = radius
        self.points = points  # Leaf node
        self.left = left
        self.right = right

class BallTree:
    def __init__(self, points, leaf_size=40):
        self.leaf_size = leaf_size
        self.root = self._build(points)
    
    def _build(self, points):
        if len(points) <= self.leaf_size:
            # Leaf node
            center = np.mean(points, axis=0)
            radius = np.max([euclidean_distance(center, p) for p in points])
            return BallNode(center, radius, points=points)
        
        # Find centroid
        centroid = np.mean(points, axis=0)
        
        # Find dimension with highest variance
        variances = np.var(points, axis=0)
        split_dim = np.argmax(variances)
        
        # Split by median in that dimension
        median = np.median(points[:, split_dim])
        left_points = points[points[:, split_dim] < median]
        right_points = points[points[:, split_dim] >= median]
        
        # Build child balls
        left_node = self._build(left_points)
        right_node = self._build(right_points)
        
        # Create parent ball
        center = centroid
        radius = max(
            euclidean_distance(center, left_node.center) + left_node.radius,
            euclidean_distance(center, right_node.center) + right_node.radius
        )
        
        return BallNode(center, radius, left=left_node, right=right_node)
    
    def search(self, query, k=1):
        """k-NN search with branch pruning"""
        candidates = []
        self._search_recursive(self.root, query, k, candidates)
        
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
    
    def _search_recursive(self, node, query, k, candidates):
        if node is None:
            return
        
        # Distance from query to ball center
        dist_to_center = euclidean_distance(query, node.center)
        
        # Pruning: if ball is too far, skip
        if len(candidates) >= k:
            max_candidate_dist = max(c[1] for c in candidates)
            if dist_to_center - node.radius > max_candidate_dist:
                return  # Prune this branch
        
        # Leaf node: check all points
        if node.points is not None:
            for point in node.points:
                dist = euclidean_distance(query, point)
                candidates.append((point, dist))
            return
        
        # Internal node: explore children
        # Visit closer child first
        dist_to_left = euclidean_distance(query, node.left.center)
        dist_to_right = euclidean_distance(query, node.right.center)
        
        if dist_to_left < dist_to_right:
            self._search_recursive(node.left, query, k, candidates)
            self._search_recursive(node.right, query, k, candidates)
        else:
            self._search_recursive(node.right, query, k, candidates)
            self._search_recursive(node.left, query, k, candidates)
```

**Complexity:**
- **Build:** O(N log N)
- **Search:** O(log N) for low dimensions, O(N^(1-1/d)) for high dimensions
- **Memory:** O(N)

**Better than K-D Tree for:**
- Higher dimensions (d = 20-100)
- Non-uniform distributions
- Cosine/angular distance

**Still fails for very high dimensions (d > 100)**

## Pros & Cons

**Pros:**
✅ **Exact search** (100% accuracy)
✅ **Fast for low dimensions** (d < 20)
✅ **Good for non-uniform data**
✅ **Interpretable** (geometric structure)

**Cons:**
❌ **Curse of dimensionality** (useless for d > 100)
❌ **Build time** (O(N log N))
❌ **Memory overhead** (tree structure)
❌ **Not used in modern vector DBs** for high-dimensional embeddings

**Production Use:**
- Rarely used for embeddings (too high-dimensional)
- Sometimes for spatial data (2D/3D coordinates)
- Geographic search (lat/lon)

---

# Hash-Based Indexes (LSH)

## Locality-Sensitive Hashing (LSH)

### Core Idea

**Goal:** Hash similar vectors to the same bucket

```
Traditional hash: Similar inputs → Different outputs
LSH hash: Similar inputs → Same output (with high probability)

Example:
vec1 = [0.5, 0.8, 0.3] → hash → bucket_42
vec2 = [0.6, 0.7, 0.4] → hash → bucket_42  ✓ Same!
vec3 = [-0.3, 0.1, 0.9] → hash → bucket_17  Different
```

### Random Projection LSH (for Cosine Similarity)

**Mathematical Foundation:**

**Johnson-Lindenstrauss Lemma:** Random projections preserve distances

**Algorithm:**
1. Generate random hyperplanes
2. For each vector, determine which side of each hyperplane it's on
3. Concatenate bits to form hash

**Implementation:**
```python
class RandomProjectionLSH:
    def __init__(self, n_tables=10, n_bits=12, dim=768):
        """
        n_tables: Number of hash tables (higher = better recall)
        n_bits: Bits per hash (2^n_bits buckets per table)
        dim: Vector dimensionality
        """
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.dim = dim
        
        # Generate random hyperplanes
        # Each hyperplane is a random unit vector
        self.hyperplanes = [
            np.random.randn(n_bits, dim)
            for _ in range(n_tables)
        ]
        
        # Normalize hyperplanes
        for i in range(n_tables):
            norms = np.linalg.norm(self.hyperplanes[i], axis=1, keepdims=True)
            self.hyperplanes[i] /= norms
        
        # Hash tables
        self.tables = [{} for _ in range(n_tables)]
    
    def _hash(self, vector, table_id):
        """Hash a vector for a specific table"""
        # Project onto hyperplanes
        projections = np.dot(self.hyperplanes[table_id], vector)
        
        # Convert to binary (positive = 1, negative = 0)
        binary = (projections > 0).astype(int)
        
        # Convert binary to integer
        hash_value = int(''.join(map(str, binary)), 2)
        
        return hash_value
    
    def add(self, vector, vector_id):
        """Add vector to all hash tables"""
        for table_id in range(self.n_tables):
            hash_value = self._hash(vector, table_id)
            
            if hash_value not in self.tables[table_id]:
                self.tables[table_id][hash_value] = []
            
            self.tables[table_id][hash_value].append(vector_id)
    
    def search(self, query, k=10):
        """Search for nearest neighbors"""
        candidate_ids = set()
        
        # Query all tables
        for table_id in range(self.n_tables):
            hash_value = self._hash(query, table_id)
            
            # Get vectors in this bucket
            if hash_value in self.tables[table_id]:
                candidate_ids.update(self.tables[table_id][hash_value])
        
        # Return candidate IDs
        # Note: Must compute exact distances to rank candidates
        return list(candidate_ids)
```

**Hash collision probability (Johnson-Lindenstrauss):**
```
P(h(a) = h(b)) = 1 - θ/π

Where θ = arccos(cos_similarity(a, b))

Examples:
- cos_sim = 1.0 (identical) → P = 1.0 (always collide)
- cos_sim = 0.9 → P ≈ 0.85
- cos_sim = 0.5 → P ≈ 0.67
- cos_sim = 0.0 (orthogonal) → P = 0.5
```

### Parameter Selection

**Trade-offs:**

```python
# More tables → Better recall, more memory
n_tables = 10  # Standard: 5-20
# - Too few: Miss relevant vectors
# - Too many: Slow, high memory

# More bits → More buckets, better precision
n_bits = 12  # 2^12 = 4096 buckets
# - Too few: Many vectors per bucket (slow post-filtering)
# - Too many: Empty buckets (low recall)

# Rule of thumb:
n_tables = ceil(log(N) / log(1/P))  # N = dataset size, P = target recall
n_bits = ceil(log2(N / average_bucket_size))
```

### Multi-Probe LSH

**Problem:** With one hash, we only check one bucket per table

**Solution:** Check nearby buckets (multi-probe)

```python
def multi_probe_search(self, query, k=10, probe_radius=2):
    """Search with multi-probe"""
    candidate_ids = set()
    
    for table_id in range(self.n_tables):
        # Primary hash
        hash_value = self._hash(query, table_id)
        
        # Generate nearby hashes (flip bits)
        nearby_hashes = self._generate_probes(hash_value, probe_radius)
        
        for h in nearby_hashes:
            if h in self.tables[table_id]:
                candidate_ids.update(self.tables[table_id][h])
    
    return list(candidate_ids)

def _generate_probes(self, hash_value, radius):
    """Generate hashes within Hamming distance = radius"""
    probes = [hash_value]
    
    # Flip up to 'radius' bits
    for num_flips in range(1, radius + 1):
        from itertools import combinations
        
        for bit_positions in combinations(range(self.n_bits), num_flips):
            flipped = hash_value
            for pos in bit_positions:
                flipped ^= (1 << pos)  # Flip bit
            probes.append(flipped)
    
    return probes

# Tradeoff: More probes → better recall, slower query
```

### Cross-Polytope LSH (Simhash)

**Improvement over random projection for high dimensions**

```python
class CrossPolytopeLSH:
    """Faster LSH using cross-polytope hashing"""
    
    def __init__(self, n_tables=10, n_bits=12, dim=768):
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.dim = dim
        
        # Random rotations
        self.rotations = [
            np.random.randn(dim, dim)
            for _ in range(n_tables)
        ]
        
        # Orthogonalize (Gram-Schmidt)
        for i in range(n_tables):
            q, r = np.linalg.qr(self.rotations[i])
            self.rotations[i] = q
    
    def _hash(self, vector, table_id):
        """Hash using cross-polytope"""
        # Rotate vector
        rotated = np.dot(self.rotations[table_id], vector)
        
        # Find largest absolute value in each group
        group_size = self.dim // self.n_bits
        hash_value = 0
        
        for i in range(self.n_bits):
            start = i * group_size
            end = start + group_size
            group = rotated[start:end]
            
            # Index of max absolute value
            max_idx = np.argmax(np.abs(group))
            
            # Sign bit
            sign = 1 if group[max_idx] > 0 else 0
            
            # Combine index and sign
            hash_value = (hash_value << 1) | sign
        
        return hash_value
```

**Advantage:** Better for high dimensions (d > 100)

## Pros & Cons

**Pros:**
✅ **Sub-linear query time** (O(k) candidates)
✅ **Constant build time per vector** (O(1))
✅ **Simple to implement**
✅ **Works for high dimensions**
✅ **Theoretical guarantees** (probabilistic)

**Cons:**
❌ **Low recall** (typically 60-80%)
❌ **Many false positives** (must verify candidates)
❌ **Memory intensive** (multiple hash tables)
❌ **Parameter tuning is difficult**
❌ **Non-deterministic** (probabilistic)
❌ **Not competitive with modern methods** (HNSW, IVF)

**Production Use:**
- **Rarely used alone** in modern vector databases
- Sometimes as **first-stage filter** before HNSW
- **Deduplication** (finding exact duplicates)
- **Streaming scenarios** (constant-time insertion)

---

# Graph-Based Indexes (HNSW)

## Hierarchical Navigable Small World (HNSW)

### Mathematical Foundation

**Small World Property (Watts-Strogatz):**
- Average path length: O(log N)
- High clustering coefficient

**Navigable Small World (NSW):**
- Add long-range connections
- Greedy search converges in O(log N) hops

**HNSW = Hierarchical layers of NSW graphs**

### Architecture

```
Layer 2 (sparse):     A -------------- B
                      |                |
                      |                |
Layer 1 (medium):     A --- C --- D -- B
                      |   /   \   |    |
                      | /       \ |    |
Layer 0 (dense):      A-C-E-F-G-D-H-I-B-J-K-L-M-N
                      All vectors, heavily connected

Properties:
- Layer i has ~N/2^i nodes
- Higher layers = long jumps (zoom out)
- Lower layers = precise navigation (zoom in)
```

### Layer Assignment

**Exponential decay probability:**

```python
def select_layer(max_layer, m_L=1/np.log(2)):
    """
    Assign vector to layer with exponential decay
    
    max_layer: Maximum layer in graph
    m_L: Normalization factor (default = 1/ln(2) ≈ 1.44)
    """
    # Random uniform [0, 1]
    u = np.random.uniform(0, 1)
    
    # Exponential: -ln(u) * m_L
    level = int(-np.log(u) * m_L)
    
    return min(level, max_layer)

# Distribution:
# Layer 0: 100% of nodes
# Layer 1: ~50% of nodes
# Layer 2: ~25% of nodes
# Layer 3: ~12.5% of nodes
# ...
```

### Construction Algorithm

```python
class HNSW:
    def __init__(self, M=16, ef_construction=200, max_layers=16):
        """
        M: Max connections per node (16-48 typical)
        ef_construction: Size of dynamic candidate list during construction
        max_layers: Maximum number of layers
        """
        self.M = M
        self.M_max = M  # Max at layer 0
        self.M_max_0 = M * 2  # Max at upper layers
        self.ef_construction = ef_construction
        self.max_layers = max_layers
        
        # Multi-layer graph
        self.graphs = [defaultdict(set) for _ in range(max_layers)]
        
        # Entry point for search
        self.entry_point = None
        self.entry_point_layer = 0
    
    def add(self, vector, vector_id):
        """Add vector to HNSW graph"""
        # Assign to layer
        layer = self._select_layer()
        
        if self.entry_point is None:
            # First vector
            self.entry_point = vector_id
            self.entry_point_layer = layer
            return
        
        # Search for nearest neighbors at each layer
        nearest = [self.entry_point]
        
        # Phase 1: Zoom out (top layers)
        for lc in range(self.entry_point_layer, layer, -1):
            nearest = self._search_layer(vector, nearest, 1, lc)
        
        # Phase 2: Find neighbors and add connections (insertion layers)
        for lc in range(layer, -1, -1):
            candidates = self._search_layer(
                vector,
                nearest,
                self.ef_construction,
                lc
            )
            
            # Select M best neighbors
            M = self.M_max_0 if lc == 0 else self.M_max
            neighbors = self._select_neighbors(vector, candidates, M, lc)
            
            # Add bidirectional links
            for neighbor_id in neighbors:
                self.graphs[lc][vector_id].add(neighbor_id)
                self.graphs[lc][neighbor_id].add(vector_id)
                
                # Prune neighbor's connections if too many
                max_conn = self.M_max_0 if lc == 0 else self.M_max
                if len(self.graphs[lc][neighbor_id]) > max_conn:
                    # Prune to M best connections
                    self._prune_connections(neighbor_id, lc, max_conn)
            
            nearest = candidates
        
        # Update entry point if necessary
        if layer > self.entry_point_layer:
            self.entry_point = vector_id
            self.entry_point_layer = layer
    
    def _search_layer(self, query, entry_points, num_candidates, layer):
        """
        Greedy search at a specific layer
        
        Returns: List of closest vectors
        """
        visited = set()
        candidates = []  # Min-heap: (-distance, vector_id)
        w = []  # Dynamic list of nearest neighbors (max-heap: distance, vector_id)
        
        # Initialize with entry points
        for ep in entry_points:
            dist = self._distance(query, self.vectors[ep])
            heapq.heappush(candidates, (-dist, ep))
            heapq.heappush(w, (dist, ep))
            visited.add(ep)
        
        while candidates:
            # Get closest unvisited candidate
            current_dist, current_id = heapq.heappop(candidates)
            current_dist = -current_dist
            
            # If current is farther than worst in w, stop
            if current_dist > w[0][0]:
                break
            
            # Check neighbors of current
            for neighbor_id in self.graphs[layer][current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    dist = self._distance(query, self.vectors[neighbor_id])
                    
                    # If better than worst in w, or w not full
                    if dist < w[0][0] or len(w) < num_candidates:
                        heapq.heappush(candidates, (-dist, neighbor_id))
                        heapq.heappush(w, (dist, neighbor_id))
                        
                        # Keep w at most num_candidates elements
                        if len(w) > num_candidates:
                            heapq.heappop(w)
        
        # Return closest neighbors
        return [vector_id for _, vector_id in sorted(w, key=lambda x: x[0])]
    
    def _select_neighbors(self, query, candidates, M, layer):
        """
        Select M best neighbors using heuristic
        
        Heuristic: Prefer diverse neighbors (avoid clustering)
        """
        # Simple heuristic: just take M closest
        # (More sophisticated: Select Neighbors Heuristic from paper)
        return candidates[:M]
    
    def search(self, query, k=10, ef=100):
        """
        Search for k nearest neighbors
        
        ef: Size of dynamic candidate list (ef >= k)
        """
        if self.entry_point is None:
            return []
        
        # Phase 1: Navigate top layers (zoom out)
        nearest = [self.entry_point]
        for layer in range(self.entry_point_layer, 0, -1):
            nearest = self._search_layer(query, nearest, 1, layer)
        
        # Phase 2: Search layer 0 (zoom in)
        nearest = self._search_layer(query, nearest, ef, 0)
        
        # Return top k
        return nearest[:k]
```

### Key Parameters

**M (connections per node):**
```
M = 16 (default)
- Low M (4-8): Fast build, low recall
- Medium M (16-32): Balanced
- High M (48-64): Slow build, high recall, more memory

Memory: O(N × M × avg_layers × d)

Rule of thumb:
- Fast queries, medium recall: M = 16
- High recall: M = 32-48
- Memory constrained: M = 8
```

**ef_construction (build quality):**
```
ef_construction = 200 (default)
- Low (40-100): Fast build, lower quality graph
- Medium (200): Balanced
- High (400-800): Slow build, high quality

Build time ∝ ef_construction
Recall ∝ ef_construction
```

**ef (search quality):**
```
ef = 100 (default, adjustable per query)
- Must satisfy: ef >= k
- Low (50): Fast query, lower recall
- Medium (100-200): Balanced
- High (400-800): Slow query, high recall

Query time ∝ ef
Recall ∝ ef

Typical values:
- 90% recall: ef = 50-100
- 95% recall: ef = 100-200  
- 99% recall: ef = 400-800
```

### Search Complexity

**Theoretical:** O(log N) with high probability

**Proof sketch:**
```
1. Each layer has ~N/2^i nodes
2. Greedy search finds nearest in O(M) steps per layer
3. Total layers: log₂(N)
4. Total steps: O(M × log N) = O(log N) for constant M
```

**Empirical:**
```
Dataset: 1M vectors, 768 dims, M=16, ef=100

Distance computations per query: ~1,500
Compared to brute force: 1,000,000

Speedup: 666×
Recall: 97%
```

### Memory Layout

```
# Naive (bad)
graph = {
    node_id: set([neighbor1, neighbor2, ...])
}

# Optimized (good)
# Store adjacency lists contiguously
neighbors_data = [n1, n2, n3, ..., nK]
offsets = [0, 5, 12, ...]  # Start index for each node
neighbors_per_node = [5, 7, 8, ...]  # Count

def get_neighbors(node_id):
    start = offsets[node_id]
    count = neighbors_per_node[node_id]
    return neighbors_data[start:start+count]

# Cache-friendly, SIMD-friendly
```

## Pros & Cons

**Pros:**
✅ **Excellent accuracy** (95-99% recall typical)
✅ **Fast queries** (O(log N), ~5-10ms for 1M vectors)
✅ **No training required** (unlike IVF)
✅ **Dynamic** (insert/delete without rebuild)
✅ **State-of-the-art** (best accuracy/speed tradeoff)
✅ **Works in high dimensions** (tested up to 4096 dims)

**Cons:**
❌ **High memory** (2-3× vector data due to graph)
❌ **Slow build** (O(N log N × M × ef_construction))
❌ **Complex implementation** (many parameters)
❌ **Non-deterministic** (layer assignment is random)

**Production Use:**
- **Most widely used** in modern vector databases
- **Default choice** for Qdrant, Weaviate, Milvus, Pinecone
- **Best for:** High recall requirements (>90%)

---

(Continuing in next part...)

---

# Clustering-Based Indexes (IVF)

## Inverted File Index (IVF)

### Core Concept

**Idea:** Use k-means clustering to partition vectors, search only relevant clusters

```
Step 1 (Offline): Cluster all vectors
   All N vectors → K clusters (Voronoi cells)
   
Step 2 (Search): 
   Query → Find nearest clusters → Search only those clusters
   
Reduction: Instead of N comparisons, do K + (N/K × nprobe) comparisons
```

### Mathematical Foundation

**Voronoi Decomposition:**
```
Given centroids C = {c₁, c₂, ..., cₖ}
Vector v belongs to cell i if: i = argmin dist(v, cⱼ)
                                      j

Properties:
- Each vector belongs to exactly one cell
- Cell boundaries are hyperplanes
- Cells partition the space
```

### Implementation

```python
class IVF:
    def __init__(self, n_clusters=100, metric='euclidean'):
        """
        n_clusters: Number of Voronoi cells
        """
        self.n_clusters = n_clusters
        self.metric = metric
        
        # Centroids (learned during training)
        self.centroids = None
        
        # Inverted lists: cluster_id → [vector_ids]
        self.inverted_lists = None
        
        # Vector storage
        self.vectors = None
    
    def train(self, vectors):
        """
        Train index using k-means
        
        Complexity: O(N × K × d × iterations)
        """
        from sklearn.cluster import KMeans
        
        print(f"Training IVF with {len(vectors)} vectors...")
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,  # Number of runs
            max_iter=100,
            random_state=42
        )
        
        # Fit and get assignments
        cluster_assignments = kmeans.fit_predict(vectors)
        self.centroids = kmeans.cluster_centers_
        
        # Build inverted lists
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        
        for vector_id, cluster_id in enumerate(cluster_assignments):
            self.inverted_lists[cluster_id].append(vector_id)
        
        self.vectors = vectors
        
        print(f"Training complete. Cluster sizes:")
        for i, inv_list in enumerate(self.inverted_lists):
            print(f"  Cluster {i}: {len(inv_list)} vectors")
    
    def search(self, query, k=10, nprobe=10):
        """
        Search nearest clusters
        
        nprobe: Number of clusters to search (1 to n_clusters)
        """
        if self.centroids is None:
            raise ValueError("Index not trained")
        
        # Step 1: Find nearest centroids
        centroid_distances = []
        for i, centroid in enumerate(self.centroids):
            dist = self._distance(query, centroid)
            centroid_distances.append((i, dist))
        
        # Sort and get top nprobe
        centroid_distances.sort(key=lambda x: x[1])
        nearest_clusters = [cluster_id for cluster_id, _ in centroid_distances[:nprobe]]
        
        # Step 2: Search vectors in nearest clusters
        candidates = []
        
        for cluster_id in nearest_clusters:
            for vector_id in self.inverted_lists[cluster_id]:
                vector = self.vectors[vector_id]
                dist = self._distance(query, vector)
                candidates.append((vector_id, dist))
        
        # Step 3: Sort and return top-k
        candidates.sort(key=lambda x: x[1])
        
        return candidates[:k]
    
    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.linalg.norm(a - b)
        elif self.metric == 'cosine':
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Parameter Selection

**Number of clusters (K):**
```python
# Rule of thumb
K = sqrt(N)  # Square root heuristic

# Examples:
N = 1M → K = 1,000
N = 10M → K = 3,162  
N = 100M → K = 10,000

# Trade-offs:
# Too few clusters (K small):
# - Large clusters → more vectors to scan
# - Fast coarse quantization
# → Use when: Memory is limited

# Too many clusters (K large):
# - Small clusters → fast scanning
# - Slow coarse quantization
# - Risk of empty clusters
# → Use when: Need low latency

# Optimal: Balance between cluster size and scan time
K_optimal = sqrt(N × expected_latency / coarse_quantization_time)
```

**nprobe (clusters to search):**
```python
nprobe = 10  # Default

# Trade-offs:
# nprobe = 1: Fast (K + N/K comparisons), low recall (~60%)
# nprobe = 10: Medium speed, good recall (~90%)
# nprobe = 100: Slow, high recall (~98%)

# Complexity:
# Time = O(K) + O(nprobe × N/K)
# For K = sqrt(N):
# Time = O(sqrt(N)) + O(nprobe × sqrt(N)) = O(nprobe × sqrt(N))

# Recall curve (empirical):
recall = 1 - exp(-alpha × nprobe / K)
# α depends on data distribution (~0.5-2.0)
```

### Cluster Imbalance Problem

**Issue:** K-means creates unbalanced clusters

```python
# Example distribution:
Cluster sizes: [5000, 4800, ..., 500, 200, 50]
                ^^^^^ Large        ^^^^^ Small

Problem:
- Large clusters slow down search
- Small clusters waste capacity

# Mitigation 1: Multiple rounds
def balanced_kmeans(vectors, K, max_cluster_size):
    """Split large clusters recursively"""
    clusters = kmeans(vectors, K)
    
    while max(len(c) for c in clusters) > max_cluster_size:
        # Find largest cluster
        largest = max(clusters, key=len)
        clusters.remove(largest)
        
        # Split it
        sub_clusters = kmeans(largest, 2)
        clusters.extend(sub_clusters)
    
    return clusters

# Mitigation 2: Spherical k-means
def spherical_kmeans(vectors, K):
    """Use cosine similarity instead of L2"""
    # Normalize vectors
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Standard k-means on normalized vectors
    return kmeans(vectors_norm, K)
```

### IVF Variants

**IVFADC (IVF + Asymmetric Distance Computation):**
```python
class IVFADC(IVF):
    """IVF with asymmetric distance computation"""
    
    def search(self, query, k=10, nprobe=10):
        """Use exact query, approximate database vectors"""
        # Find nearest centroids (exact distance)
        nearest_clusters = self._find_nearest_centroids(query, nprobe)
        
        # Search in clusters with exact query distance
        candidates = []
        for cluster_id in nearest_clusters:
            for vector_id in self.inverted_lists[cluster_id]:
                # Exact distance (query is not quantized)
                vector = self.vectors[vector_id]
                dist = self._distance(query, vector)
                candidates.append((vector_id, dist))
        
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
```

**IVFPQ (IVF + Product Quantization):**
Covered in detail in Quantization section below.

## Pros & Cons

**Pros:**
✅ **Fast queries** (O(nprobe × N/K) = O(nprobe × sqrt(N)))
✅ **Low memory** (just centroids + inverted lists)
✅ **Good for large datasets** (>10M vectors)
✅ **Scalable** (can distribute clusters)
✅ **Tunable** (nprobe parameter)

**Cons:**
❌ **Requires training** (K-means is expensive)
❌ **Static** (updates require retraining)
❌ **Lower recall than HNSW** (unless nprobe is very high)
❌ **Cluster imbalance** (some clusters much larger)
❌ **Not great for low-dimensional data** (curse of dimensionality)

**Production Use:**
- **Common in billion-scale systems** (Meta's FAISS, Milvus)
- **Often combined with PQ** (IVFPQ) for extreme compression
- **Good for cost optimization** (less memory than HNSW)

---

# Quantization Techniques

## Product Quantization (PQ)

### Core Idea

**Compress vectors by quantizing subvectors independently**

```
Original: 768 dims × 4 bytes = 3072 bytes
PQ (m=8, k=256): 8 bytes = 384× compression!

Process:
1. Split 768-dim vector into 8 subvectors of 96 dims each
2. For each subvector, find nearest centroid from 256-centroid codebook
3. Store centroid index (1 byte) instead of subvector (96×4 = 384 bytes)
```

### Mathematical Foundation

**Product quantization:**
```
Original vector: x ∈ ℝᵈ

Split: x = [x₁, x₂, ..., xₘ] where xᵢ ∈ ℝ^(d/m)

Quantize each: q(xᵢ) = argmin ||xᵢ - cᵢⱼ||
                        j∈{1..k}

Where:
- m = number of subquantizers (typically 8, 16, 32, 64)
- k = codebook size per subquantizer (typically 256 = 2⁸)
- cᵢⱼ = j-th centroid in i-th codebook

Compressed: [q(x₁), q(x₂), ..., q(xₘ)] ∈ {0..k-1}ᵐ
Storage: m × log₂(k) bits (e.g., 8 × 8 = 64 bits)
```

### Implementation

```python
class ProductQuantizer:
    def __init__(self, m=8, k=256, dim=768):
        """
        m: Number of subquantizers
        k: Codebook size per subquantizer (typically 256 = 2^8)
        dim: Vector dimensionality
        """
        self.m = m
        self.k = k
        self.dim = dim
        self.subvector_dim = dim // m
        
        # Codebooks: m codebooks, each with k centroids
        self.codebooks = None
    
    def train(self, vectors):
        """
        Train codebooks using k-means on subvectors
        
        Complexity: O(m × N × k × d/m × iterations) = O(N × k × d × iterations)
        """
        from sklearn.cluster import KMeans
        
        N = len(vectors)
        self.codebooks = []
        
        print(f"Training PQ with {N} vectors...")
        
        for i in range(self.m):
            # Extract subvectors
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]
            
            # K-means on subvectors
            kmeans = KMeans(n_clusters=self.k, n_init=5, max_iter=50)
            kmeans.fit(subvectors)
            
            self.codebooks.append(kmeans.cluster_centers_)
            
            print(f"  Trained subquantizer {i+1}/{self.m}")
        
        self.codebooks = np.array(self.codebooks)
    
    def encode(self, vector):
        """
        Encode vector to PQ codes
        
        Returns: Array of m codes, each in range [0, k-1]
        """
        codes = np.zeros(self.m, dtype=np.uint8)
        
        for i in range(self.m):
            # Extract subvector
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvector = vector[start:end]
            
            # Find nearest centroid in this codebook
            distances = np.linalg.norm(self.codebooks[i] - subvector, axis=1)
            codes[i] = np.argmin(distances)
        
        return codes
    
    def decode(self, codes):
        """
        Decode PQ codes back to approximate vector
        """
        vector = np.zeros(self.dim)
        
        for i in range(self.m):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            
            # Get centroid
            centroid = self.codebooks[i][codes[i]]
            vector[start:end] = centroid
        
        return vector
    
    def asymmetric_distance(self, query, codes):
        """
        Compute distance between query and PQ-encoded vector
        
        Asymmetric: query is exact, database vector is quantized
        
        This is the KEY to PQ efficiency!
        """
        # Precompute distances from query subvectors to all centroids
        # This is done once per query
        query_to_centroids = np.zeros((self.m, self.k))
        
        for i in range(self.m):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            query_subvector = query[start:end]
            
            # Distance to all centroids in this codebook
            for j in range(self.k):
                centroid = self.codebooks[i][j]
                query_to_centroids[i, j] = np.linalg.norm(
                    query_subvector - centroid
                ) ** 2
        
        # Compute distance using lookup table
        # This is VERY fast: just m lookups!
        distance = 0
        for i in range(self.m):
            code = codes[i]
            distance += query_to_centroids[i, code]
        
        return np.sqrt(distance)
    
    def batch_asymmetric_distance(self, query, codes_matrix):
        """
        Compute distances to multiple PQ-encoded vectors
        
        codes_matrix: [N, m] array of codes
        
        This is extremely fast!
        """
        N = codes_matrix.shape[0]
        
        # Precompute lookup table
        query_to_centroids = np.zeros((self.m, self.k))
        
        for i in range(self.m):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            query_subvector = query[start:end]
            
            for j in range(self.k):
                centroid = self.codebooks[i][j]
                query_to_centroids[i, j] = np.linalg.norm(
                    query_subvector - centroid
                ) ** 2
        
        # Vectorized distance computation
        distances = np.zeros(N)
        for i in range(self.m):
            distances += query_to_centroids[i, codes_matrix[:, i]]
        
        return np.sqrt(distances)

# Usage
pq = ProductQuantizer(m=8, k=256, dim=768)

# Train
train_vectors = np.random.randn(100000, 768)
pq.train(train_vectors)

# Encode
vector = np.random.randn(768)
codes = pq.encode(vector)
print(f"Compressed: {codes}")  # 8 bytes instead of 3072!

# Decode
reconstructed = pq.decode(codes)
error = np.linalg.norm(vector - reconstructed)
print(f"Reconstruction error: {error}")

# Asymmetric distance (fast!)
query = np.random.randn(768)
dist = pq.asymmetric_distance(query, codes)
```

### Optimized Product Quantization (OPQ)

**Problem:** Axis-aligned subspace splitting is suboptimal

**Solution:** Rotate vectors before splitting

```python
class OptimizedProductQuantizer:
    def __init__(self, m=8, k=256, dim=768):
        self.m = m
        self.k = k
        self.dim = dim
        self.subvector_dim = dim // m
        
        # Rotation matrix (learned)
        self.rotation = None
        
        # PQ on rotated space
        self.pq = ProductQuantizer(m, k, dim)
    
    def train(self, vectors):
        """
        Learn optimal rotation + PQ codebooks
        
        Alternating optimization:
        1. Fix rotation, optimize PQ
        2. Fix PQ, optimize rotation
        """
        # Initialize rotation (random orthogonal matrix)
        self.rotation = self._random_orthogonal(self.dim)
        
        for iteration in range(5):  # Alternating optimization
            # Rotate vectors
            rotated = np.dot(vectors, self.rotation)
            
            # Train PQ on rotated vectors
            self.pq.train(rotated)
            
            # Update rotation to minimize quantization error
            self.rotation = self._update_rotation(vectors)
    
    def _random_orthogonal(self, n):
        """Generate random orthogonal matrix"""
        A = np.random.randn(n, n)
        Q, R = np.linalg.qr(A)
        return Q
    
    def encode(self, vector):
        """Encode with rotation"""
        rotated = np.dot(vector, self.rotation)
        return self.pq.encode(rotated)
    
    def asymmetric_distance(self, query, codes):
        """Distance with rotation"""
        rotated_query = np.dot(query, self.rotation)
        return self.pq.asymmetric_distance(rotated_query, codes)
```

**Improvement:** 10-20% better recall for same compression

## Scalar Quantization (SQ)

### Concept

**Compress by reducing precision: float32 → uint8**

```
Original: 768 × 4 bytes = 3072 bytes
SQ8: 768 × 1 byte = 768 bytes (4× compression)

Process:
1. Find min/max values in dataset
2. Quantize: q(x) = round((x - min) / (max - min) × 255)
3. Store as uint8
```

### Implementation

```python
class ScalarQuantizer:
    def __init__(self, dim=768):
        self.dim = dim
        self.min_vals = None
        self.max_vals = None
    
    def train(self, vectors):
        """Learn min/max per dimension"""
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)
    
    def encode(self, vector):
        """Quantize to uint8"""
        # Normalize to [0, 1]
        normalized = (vector - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)
        
        # Scale to [0, 255]
        quantized = np.round(normalized * 255).astype(np.uint8)
        
        return quantized
    
    def decode(self, codes):
        """Dequantize"""
        # Scale back to [0, 1]
        normalized = codes.astype(np.float32) / 255.0
        
        # Denormalize
        vector = normalized * (self.max_vals - self.min_vals) + self.min_vals
        
        return vector
    
    def asymmetric_distance(self, query, codes):
        """Distance using dequantized vector"""
        reconstructed = self.decode(codes)
        return np.linalg.norm(query - reconstructed)

# Compression ratio: 4×
# Accuracy loss: ~5% (very good!)
```

**Variants:**

**SQ4 (4-bit quantization):**
```python
# 8× compression, more accuracy loss (~10%)
quantized = np.round(normalized * 15).astype(np.uint8)
```

**SQ6 (6-bit quantization):**
```python
# 5.3× compression, ~7% accuracy loss
quantized = np.round(normalized * 63).astype(np.uint8)
```

## Binary Quantization

### Concept

**Extreme compression: Reduce each dimension to 1 bit**

```
Original: 768 × 4 bytes = 3072 bytes
Binary: 768 bits = 96 bytes (32× compression!)

Process:
q(x) = 1 if x >= threshold else 0

Distance: Hamming distance (XOR + popcount)
```

### Implementation

```python
class BinaryQuantizer:
    def __init__(self, dim=768):
        self.dim = dim
        self.threshold = None
    
    def train(self, vectors):
        """Learn threshold (median works well)"""
        self.threshold = np.median(vectors)
    
    def encode(self, vector):
        """Quantize to binary"""
        # 1 if above threshold, 0 otherwise
        binary = (vector >= self.threshold).astype(np.uint8)
        
        # Pack bits (8 bits per byte)
        packed = np.packbits(binary)
        
        return packed
    
    def hamming_distance(self, codes_a, codes_b):
        """Hamming distance between binary vectors"""
        # XOR to find differing bits
        xor = np.bitwise_xor(codes_a, codes_b)
        
        # Count 1s (popcount)
        distance = np.unpackbits(xor).sum()
        
        return distance
    
    def batch_hamming_distance(self, query_codes, database_codes):
        """Vectorized Hamming distance (very fast!)"""
        # database_codes: [N, packed_dim]
        N = database_codes.shape[0]
        
        # XOR
        xor = np.bitwise_xor(query_codes, database_codes)
        
        # Popcount using lookup table (fast)
        popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        
        distances = np.zeros(N)
        for i in range(xor.shape[1]):
            distances += popcount_table[xor[:, i]]
        
        return distances

# Ultra-fast search!
# 10-100× faster than float32
# But lower accuracy (85-90% recall typical)
```

## Comparison

| Method | Compression | Accuracy Loss | Speed | Memory |
|--------|------------|---------------|-------|--------|
| None (float32) | 1× | 0% | Baseline | 100% |
| Scalar (SQ8) | 4× | ~5% | 2× faster | 25% |
| Scalar (SQ4) | 8× | ~10% | 4× faster | 12.5% |
| PQ (m=8,k=256) | 384× | ~15% | 10× faster | 0.26% |
| OPQ (m=8,k=256) | 384× | ~10% | 10× faster | 0.26% |
| Binary | 32× | ~20% | 100× faster | 3.1% |

**Trade-off:**
```
Compression ↑ → Accuracy ↓, Speed ↑, Memory ↓
```

---

# Hybrid & Composite Indexes

## IVFPQ (IVF + Product Quantization)

**Combination of two techniques for billion-scale**

```python
class IVFPQ:
    def __init__(self, n_clusters=1000, m=8, k=256):
        # Coarse quantization (IVF)
        self.ivf = IVF(n_clusters=n_clusters)
        
        # Fine quantization (PQ)
        self.pq = ProductQuantizer(m=m, k=k)
    
    def train(self, vectors):
        """Two-stage training"""
        # Stage 1: Train IVF
        self.ivf.train(vectors)
        
        # Stage 2: Train PQ on residuals
        residuals = []
        for i, vector in enumerate(vectors):
            cluster_id = self.ivf.assign_cluster(vector)
            centroid = self.ivf.centroids[cluster_id]
            residual = vector - centroid
            residuals.append(residual)
        
        self.pq.train(np.array(residuals))
    
    def add(self, vector, vector_id):
        """Add compressed vector"""
        # Assign to cluster
        cluster_id = self.ivf.assign_cluster(vector)
        
        # Compute residual
        centroid = self.ivf.centroids[cluster_id]
        residual = vector - centroid
        
        # Encode residual with PQ
        codes = self.pq.encode(residual)
        
        # Store
        self.ivf.inverted_lists[cluster_id].append((vector_id, codes))
    
    def search(self, query, k=10, nprobe=10):
        """Search with two-stage quantization"""
        # Find nearest clusters
        nearest_clusters = self.ivf.find_nearest_clusters(query, nprobe)
        
        # Search in clusters with PQ distance
        candidates = []
        
        for cluster_id in nearest_clusters:
            centroid = self.ivf.centroids[cluster_id]
            
            # Query residual
            query_residual = query - centroid
            
            # Compare to all vectors in cluster
            for vector_id, codes in self.ivf.inverted_lists[cluster_id]:
                # Asymmetric distance
                dist = self.pq.asymmetric_distance(query_residual, codes)
                candidates.append((vector_id, dist))
        
        # Sort and return
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

# Result:
# - 1B vectors × 768 dims = 3TB uncompressed
# - IVFPQ: ~8GB compressed (375× compression!)
# - Query: 10-50ms
# - Recall: 85-90%
```

## HNSW + PQ

**Fast graph navigation + compression**

```python
class HNSWPQ:
    def __init__(self, M=16, m=8, k=256):
        self.hnsw = HNSW(M=M)
        self.pq = ProductQuantizer(m=m, k=k)
        
        # Store PQ codes
        self.pq_codes = {}
    
    def train(self, vectors):
        """Train PQ"""
        self.pq.train(vectors)
    
    def add(self, vector, vector_id):
        """Add to graph + encode"""
        # Add to HNSW (uses full precision for graph construction)
        self.hnsw.add(vector, vector_id)
        
        # Encode and store
        codes = self.pq.encode(vector)
        self.pq_codes[vector_id] = codes
    
    def search(self, query, k=10, ef=100):
        """Navigate graph, compute PQ distances"""
        # HNSW search (returns candidates)
        candidates = self.hnsw.search(query, k=ef)
        
        # Rerank with PQ distance
        scored = []
        for vector_id in candidates:
            codes = self.pq_codes[vector_id]
            dist = self.pq.asymmetric_distance(query, codes)
            scored.append((vector_id, dist))
        
        scored.sort(key=lambda x: x[1])
        return scored[:k]

# Result:
# - HNSW accuracy with PQ memory efficiency
# - Query: ~15ms (slightly slower than HNSW alone)
# - Memory: 4-10× less than HNSW alone
# - Recall: 90-95%
```

## DiskANN

**Disk-based graph index for billion-scale**

### Key Innovations

1. **Graph on disk, search in memory**
2. **Compressed graph** (PQ codes in graph)
3. **SSD-optimized** (prefetch, batch I/O)

```python
class DiskANN:
    """Simplified DiskANN concept"""
    
    def __init__(self, index_path, M=32):
        self.index_path = index_path
        self.M = M
        
        # In-memory: Graph structure (small)
        self.graph = None  # Adjacency lists
        
        # On-disk: Full vectors (large)
        self.vectors_mmap = None
        
        # In-memory: PQ codes (medium)
        self.pq = ProductQuantizer(m=8, k=256)
        self.pq_codes = None
    
    def build(self, vectors):
        """Build index"""
        # Train PQ
        self.pq.train(vectors)
        
        # Encode all vectors
        self.pq_codes = np.array([self.pq.encode(v) for v in vectors])
        
        # Build graph using PQ distances (faster)
        self.graph = self._build_graph_pq(vectors)
        
        # Write full vectors to disk
        self._write_vectors_to_disk(vectors)
        
        # Memory-map vectors
        self.vectors_mmap = np.memmap(
            f"{self.index_path}/vectors.dat",
            dtype='float32',
            mode='r',
            shape=vectors.shape
        )
    
    def search(self, query, k=10, L=100):
        """Search with beam search on graph"""
        # Encode query for PQ distance
        query_codes = self.pq.encode(query)
        
        # Beam search on graph using PQ distance (fast)
        candidates = self._beam_search(query, L)
        
        # Rerank with exact distance (fetch from disk)
        exact_distances = []
        for vector_id in candidates:
            # Fetch from disk (mmap)
            vector = self.vectors_mmap[vector_id]
            dist = np.linalg.norm(query - vector)
            exact_distances.append((vector_id, dist))
        
        exact_distances.sort(key=lambda x: x[1])
        return exact_distances[:k]

# Result:
# - Scales to 1B+ vectors on single machine
# - Index size: ~100GB for 1B vectors
# - Query: 10-30ms from SSD
# - Recall: 95%+
```

---

# Production Patterns & Benchmarks

## Real-World Performance (1M vectors, 768 dims)

| Index Type | Build Time | Memory | Query Latency (P50) | Recall@10 | Best For |
|------------|-----------|---------|---------------------|-----------|----------|
| Flat | 0s | 3GB | 450ms | 100% | <100k vectors |
| IVF (nlist=1000, nprobe=20) | 2min | 3.1GB | 8ms | 92% | Budget systems |
| HNSW (M=16, ef=100) | 8min | 6GB | 5ms | 97% | High recall |
| IVFPQ (nlist=1000, m=8) | 5min | 80MB | 12ms | 85% | Large scale |
| HNSW+SQ8 | 10min | 1.5GB | 7ms | 95% | Memory constrained |

## Billion-Scale Benchmarks (1B vectors, 128 dims)

| Index Type | Machines | Memory/Machine | Query Latency | Recall@10 | Cost/Month |
|------------|----------|----------------|---------------|-----------|------------|
| HNSW | 8 | 64GB | 12ms | 96% | $2,400 |
| IVFPQ | 4 | 16GB | 25ms | 88% | $400 |
| DiskANN | 1 | 32GB + 500GB SSD | 35ms | 94% | $150 |
| HNSW+PQ | 4 | 32GB | 18ms | 93% | $800 |

## Production Recommendations

### For <1M Vectors

**Use: HNSW**
```python
# Config
M = 16
ef_construction = 200
ef_search = 100

# Expected
# - Latency: 5-10ms
# - Recall: 95-98%
# - Memory: 4-6GB
```

### For 1M-10M Vectors

**Use: HNSW with SQ8**
```python
# Config
M = 32
ef_construction = 200
scalar_quantization = True

# Expected
# - Latency: 8-12ms
# - Recall: 93-96%
# - Memory: 1-2GB
```

### For 10M-100M Vectors

**Use: IVF or HNSW distributed**
```python
# Option 1: IVF
nlist = 10000
nprobe = 50

# Option 2: Sharded HNSW
n_shards = 8
M_per_shard = 16
```

### For >100M Vectors

**Use: IVFPQ or DiskANN**
```python
# IVFPQ config
nlist = 65536  # sqrt(N) for N=100M
m = 16
k = 256
nprobe = 64

# Expected
# - Latency: 20-40ms
# - Recall: 85-92%
# - Memory: <1GB
```

---

# Decision Framework

## Quick Decision Tree

```
How many vectors?
├─ <100k → Flat (exact search)
├─ 100k-1M → HNSW (M=16)
├─ 1M-10M → HNSW (M=32) or HNSW+SQ
├─ 10M-100M → IVF or Sharded HNSW
└─ >100M → IVFPQ or DiskANN

What's your priority?
├─ Maximum accuracy (>98% recall)
│   └─ HNSW with high M (32-64), high ef (400+)
│
├─ Balanced (90-95% recall, <20ms)
│   └─ HNSW (M=16-32, ef=100-200)
│
├─ Low latency (<5ms)
│   └─ HNSW with GPU or Binary quantization
│
├─ Low memory
│   ├─ 1-10M vectors: HNSW + SQ8
│   └─ >10M vectors: IVFPQ
│
└─ Billion-scale on single machine
    └─ DiskANN
```

## Summary Table

| Characteristic | Best Index | Notes |
|----------------|-----------|-------|
| Highest accuracy | Flat or HNSW (M=64) | 99%+ recall |
| Best speed/accuracy | HNSW (M=16-32) | Standard choice |
| Lowest memory | IVFPQ or Binary | 100-400× compression |
| Largest scale | DiskANN or Distributed IVFPQ | Billions of vectors |
| Fastest build | Flat or LSH | No training |
| Dynamic updates | HNSW | Insert/delete online |
| Simplest | Flat or IVF | Easy to understand |

## Key Takeaways

1. **HNSW is the default** for most production systems (<100M vectors)
2. **Combine techniques** (IVF+PQ, HNSW+SQ) for best results
3. **Quantization is essential** for large scale (>10M vectors)
4. **No one-size-fits-all** - profile your workload
5. **Start simple** (Flat or HNSW), optimize later

The art is finding the right balance for your **scale, latency, accuracy, and cost requirements**!
