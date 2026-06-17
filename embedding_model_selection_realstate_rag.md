# Embedding Model Selection for Real-Estate AI Platform RAG Pipeline 

---

## 1. Define Retrieval Requirements

Before selecting an embedding model, the first step is to define the platform's retrieval requirements rather than defaulting to vendor recommendations.

### Key Requirements for Real-Estate Platform

#### Multilingual Retrieval
- Property documents may be in **English, Hindi, Kannada, Tamil**, etc.
- Users should be able to ask questions in any supported language
- Critical for a nationwide real-estate platform

#### Cross-Lingual Retrieval
- A document written in English should be retrievable when the query is in Kannada or Hindi
- Essential for user experience across diverse linguistic regions

#### Long-Context Support
- Property documents include: sale deeds, title documents, legal contracts, property brochures, CRM notes
- Embedding model must support reasonably large input lengths
- Reduces excessive fragmentation during chunking

#### High Retrieval Quality
- Legal and property documents require **high recall** (missing a clause = incorrect answers)
- Precision matters, but recall is critical in compliance-heavy domains

#### Large-Scale Deployment
- Expected: hundreds of millions of vectors
- Embedding dimension, memory footprint, and indexing cost become primary constraints
- Must optimize for cost at scale

#### Hybrid Retrieval Compatibility
- Real-estate data contains both semantic concepts and exact identifiers:
  - Property ID
  - Registration Number
  - Survey Number
  - Plot Number
- Architecture must support dense + keyword-based retrieval

---

## 2. Candidate Models Evaluation

| Model | Strengths | Concerns |
|-------|-----------|----------|
| **Cohere Multilingual** | Mature multilingual retrieval, proven in production | Smaller context window limits document sizes |
| **BGE-M3** | Strong retrieval performance, native dense+sparse support | Cross-lingual quality for Indian languages needs validation |
| **Gemini Embedding** | Multilingual, cross-lingual, configurable dimensions, Matryoshka support | Requires benchmark validation on real estate corpus |

---

## 3. Recommendation: Gemini Embedding

### Why Gemini Embedding for This Platform?

1. **Strong Multilingual Support**
   - Supports 100+ languages including Indian regional languages
   - Well-tuned for Hindi, Kannada, Tamil

2. **Cross-Lingual Retrieval**
   - Enables retrieval across language boundaries
   - Critical for nationwide platform with diverse user base

3. **Configurable Embedding Dimensions**
   - Flexibility to optimize storage cost without retraining
   - Supports dimensions from 256 to 1024

4. **Matryoshka Representation Learning**
   - Trade-off between quality and cost without full retraining
   - Allows fine-tuning dimension post-deployment

5. **Cost-Performance Balance**
   - Good retrieval quality at smaller dimensions
   - Suitable for hundreds of millions of vectors

---

## 4. Dimension Selection Strategy

**Key Insight:** Vector count directly impacts infrastructure cost. Do not select dimension based on documentation alone.

### Benchmark Approach

Test multiple dimensions on your corpus:
- 256 dimensions
- 512 dimensions
- 768 dimensions
- 1024 dimensions

### Metrics to Measure

- **Recall@K** (typically K=10)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **MRR** (Mean Reciprocal Rank)
- **Retrieval latency** (milliseconds)
- **Storage footprint** (GB per million vectors)

### Example Decision Framework

```
1024 dimensions → Recall@10 = 92% → Storage = 4 GB per million vectors
768 dimensions  → Recall@10 = 91% → Storage = 3 GB per million vectors
                 Storage reduction = 25% with 1% quality loss

Decision: Choose 768 dimensions
```

**Target:** Select the smallest dimension that meets your retrieval quality SLA.

---

## 5. Hybrid Retrieval Strategy

**Critical Point:** Do not rely solely on dense embeddings.

### Why Hybrid Matters for Real-Estate

Property systems receive many **exact-match queries**:
- "Find Property ID 12345"
- "Search Survey Number ABC-123"
- "Show Registration Number XYZ789"

Dense embeddings are suboptimal for these identifier-based queries.

### Recommended Architecture

```
┌─────────────────────────────────────────┐
│       User Query                        │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┬──────────────┐
        │             │              │
    ┌───▼───┐    ┌───▼────┐    ┌───▼────┐
    │ Dense │    │Keyword │    │Metadata│
    │Retr.  │    │Retr.   │    │Filter  │
    │(Vec)  │    │(BM25)  │    │(Filters)
    └───┬───┘    └────┬───┘    └────┬───┘
        │             │             │
        └─────────────┴─────────────┘
                    │
            ┌───────▼──────────┐
            │  RRF (Reciprocal │
            │  Rank Fusion)    │
            └────────┬─────────┘
                     │
            ┌────────▼─────────────┐
            │ Cross-Encoder        │
            │ Reranking            │
            └──────────┬───────────┘
                       │
            ┌──────────▼────────────┐
            │  Final Ranked Results │
            └──────────────────────┘
```

### Components

1. **Dense Retrieval**
   - Semantic similarity using embeddings
   - Handles conceptual queries like "apartments near schools"
   - Returns top-K results with relevance scores

2. **Keyword Retrieval (BM25)**
   - Exact and fuzzy matching on identifiers
   - Handles property IDs, registration numbers, survey numbers
   - Returns top-K results with BM25 scores

3. **Metadata Filtering**
   - Filter by location, price range, property type
   - Reduces search space before dense retrieval
   - Applies hard constraints to candidate set

4. **RRF (Reciprocal Rank Fusion)**
   - Combines ranking signals from Dense + Keyword + Metadata sources
   - Balances different retrieval modalities without score normalization
   - Formula: RRF Score = Σ(1 / (k + rank)) where k=60 (default)
   - Produces unified ranked list for reranker input

5. **Cross-Encoder Reranking**
   - Fine-grained relevance scoring on fused results
   - Operates on top candidates from RRF (e.g., top 50)
   - Improves precision of final results
   - Returns re-ranked top-K for user presentation

### Query Type Examples

| Query Type | Best Retrieval Method |
|-----------|----------------------|
| "East-facing apartments near schools" | Dense + Metadata |
| "Property ID 12345" | Keyword (BM25) |
| "Survey Number ABC-123" | Keyword (BM25) |
| "2-BHK flats in Bangalore with gym" | Dense + Metadata + Keyword |

---

## 6. Evaluation Framework

**Critical Principle:** Do not choose the embedding model based on vendor documentation alone. Build an offline evaluation dataset using real user queries.

### Evaluation Dataset Structure

```
Query: "Show east-facing apartments near schools"

Expected Results:
- Property A (3 BHK, East-facing, near DPS School)
- Property B (2 BHK, East-facing, near Cathedral School)
- Property C (2 BHK, East-facing, near Springdales School)

Negative Examples:
- Property D (West-facing, good schools nearby)
- Property E (East-facing, no nearby schools)
```

### Evaluation Metrics

For each candidate model, measure:

1. **Recall@K** (K=10, K=20)
   - What percentage of relevant results are in top-K?
   - Critical for legal/compliance domains

2. **Precision@K**
   - What percentage of top-K results are relevant?
   - Affects user experience

3. **MRR (Mean Reciprocal Rank)**
   - Average rank of first relevant result
   - Measures how quickly users find relevant content

4. **NDCG (Normalized Discounted Cumulative Gain)**
   - Considers ranking quality
   - Higher relevance results ranked higher = better NDCG

5. **Cross-Lingual Retrieval Accuracy**
   - Test with queries in Hindi, Kannada, Tamil
   - Measure retrieval accuracy across language pairs

6. **Latency**
   - P50 and P95 query latency
   - Embedding + retrieval time

7. **Memory & Storage Cost**
   - Per-million-vector footprint
   - Index size on disk

### Decision Process

```
Benchmark on representative dataset
        ↓
Compare models on recall, latency, cost
        ↓
Identify top 2-3 performers
        ↓
Run shadow production test (1-2 weeks)
        ↓
Select based on real traffic metrics
        ↓
Deploy with monitoring & fallback
```

---

## 7. Two-Minute Interview Answer

**Prompt:** "Walk us through your approach to selecting an embedding model for a real-estate RAG platform."

**Response:**

"For embedding model selection, I would first define the retrieval requirements rather than starting with vendor documentation. The platform requires:

- Multilingual and cross-lingual retrieval for a nationwide user base
- Support for long property documents (legal contracts, title deeds)
- High recall (missing a relevant clause is costly)
- Cost-efficient operation at hundreds of millions of vectors
- Support for both semantic and exact-identifier matching

I would evaluate models like Cohere Multilingual, BGE-M3, and Gemini Embedding. My initial preference is Gemini Embedding because of its multilingual support, configurable embedding dimensions, and Matryoshka Representation Learning, which allows cost optimization without retraining.

However, the final decision would be data-driven. I would benchmark multiple dimensions (256, 512, 768, 1024) and candidate models on a real-estate retrieval dataset using Recall@K, MRR, and NDCG metrics.

I would also implement hybrid retrieval combining:
- Dense retrieval for semantic queries
- BM25 for exact identifiers (Property IDs, Registration Numbers, Survey Numbers)
- Metadata filtering for structured attributes
- Cross-encoder reranking for final precision

The key principle is not to rely on documentation alone. I would create a representative evaluation dataset from real user queries and select the model that balances retrieval quality, latency, and infrastructure cost at scale."


