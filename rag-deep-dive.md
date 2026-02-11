# RAG Deep Dive: Building Production-Grade Document Q&A Systems

A comprehensive technical guide for staff engineers building Retrieval-Augmented Generation systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Indexing Strategy](#indexing-strategy)
3. [Chunking Strategies](#chunking-strategies)
4. [Embedding Models](#embedding-models)
5. [Vector Database Selection](#vector-database-selection)
6. [Retrieval Strategies](#retrieval-strategies)
7. [Generation Phase](#generation-phase)
8. [Evaluation Methods](#evaluation-methods)
9. [Implementation Guide](#implementation-guide)

---

## Overview

### What is RAG?

Retrieval-Augmented Generation combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG fetches relevant context from your documents at query time.

### End-to-End Flow

```
Documents → Chunk & Embed → Vector DB → Query → Retrieve → LLM → Answer
```

### Key Benefits for Document Q&A

- **Up-to-date information**: Documents can be updated without retraining
- **Source attribution**: Answers can cite specific documents
- **Reduced hallucination**: Grounded in actual content
- **Domain-specific**: Works with proprietary/internal docs
- **Cost-effective**: No need for fine-tuning large models

### Core Challenges

- **Retrieval quality**: Finding the right chunks is critical
- **Context window limits**: Can't retrieve everything
- **Semantic mismatch**: Query phrasing vs document language
- **Multi-hop reasoning**: When answer spans multiple docs
- **Latency**: Balancing speed vs accuracy

---

## Indexing Strategy

### Overview

Indexing is the offline process of preparing your documents for retrieval. The strategy you choose dramatically impacts retrieval quality and system performance.

### Indexing Approaches

#### Dense Vector Indexing

**Pros:**
- Semantic similarity matching
- Handles paraphrasing well
- Works across languages
- State-of-the-art for most use cases

**Best for:** Natural language queries, semantic search

#### Sparse Vector (BM25)

**Pros:**
- Exact keyword matching
- Fast and interpretable
- No model needed
- Good for technical terms

**Best for:** Keyword-heavy queries, code, IDs

#### Hybrid Strategy (Recommended)

Combine dense + sparse vectors for best results:

- Dense vectors capture semantic meaning
- Sparse vectors ensure keyword matches
- Weight: 70% dense, 30% sparse (tune for your domain)
- Fusion methods: RRF (Reciprocal Rank Fusion) or weighted sum

**Impact:** 20-30% improvement over dense-only retrieval

### Metadata Strategy

Store rich metadata with each chunk for filtered retrieval:

```python
metadata = {
    "document_id": "doc-123",
    "document_title": "User Manual v2.0",
    "chunk_index": 5,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-02-10T14:22:00Z",
    "document_type": "pdf",
    "section_heading": "Chapter 3: Installation",
    "author": "Engineering Team",
    "department": "Product",
    "tags": ["installation", "setup", "configuration"]
}
```

### Hierarchical Indexing

```
Document
  ├── Summary embedding (for initial filtering)
  ├── Section embeddings (mid-level retrieval)
  └── Chunk embeddings (fine-grained retrieval)
```

This allows multi-stage retrieval: first find relevant documents, then relevant sections, then specific chunks.

### Incremental Indexing Strategies

| Strategy | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Batch Re-index** | Simple, consistent | Downtime, expensive | Small corpus, infrequent updates |
| **Incremental Updates** | Real-time, efficient | Complexity, eventual consistency | Frequently changing docs |
| **Versioned Indices** | Zero downtime, rollback | Storage overhead | Production systems |

**Recommendation:** Use versioned indices for production systems. Maintain two indices (current + new), switch atomically when ready.

---

## Chunking Strategies

### The Chunking Dilemma

**Too small** = loss of context  
**Too large** = irrelevant content + increased cost

Finding the right balance is critical.

### Chunking Methods

#### 1. Fixed-Size Chunking

**Method:** Split every N tokens with M token overlap

**Example:** 512 tokens, 50 token overlap

**Pros:**
- ✅ Simple to implement
- ✅ Predictable chunk sizes

**Cons:**
- ❌ Breaks semantic boundaries
- ❌ May split sentences

#### 2. Semantic Chunking

**Method:** Split at natural boundaries (paragraphs, sections)

**Pros:**
- ✅ Preserves meaning
- ✅ Better context

**Cons:**
- ❌ Variable chunk sizes
- ❌ Harder to implement

#### 3. Recursive Character Splitting (Recommended)

**Method:** Try splitting at different levels until chunk size is appropriate

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

chunks = splitter.split_text(document_text)
```

**Strategy:**
1. Try to split at section boundaries (headers)
2. If too large, split at paragraphs
3. If still too large, split at sentences
4. Target: 300-500 tokens per chunk
5. Overlap: 50-100 tokens (10-20%)

### Advanced Techniques

#### Context-Enhanced Chunking

Prepend document/section context to each chunk:

```python
enhanced_chunk = f"""[Document: {doc_title}]
[Section: {section_heading}]

{chunk_text}"""
```

**Impact:** Improves retrieval accuracy by 20-30%

#### Sliding Window Chunks

Create overlapping chunks that slide across the document:

```
Chunk 1: [Tokens 0-500]
Chunk 2: [Tokens 400-900]
Chunk 3: [Tokens 800-1300]
```

**Trade-off:** 2-3x more chunks, but better coverage of boundaries

#### Parent-Child Chunking

Store small chunks for retrieval, but return larger parent chunks to LLM:

- **Child chunks:** 128 tokens (for precise retrieval)
- **Parent chunks:** 512 tokens (for LLM context)
- **Benefit:** Better retrieval + better generation

```python
# Create child chunks for retrieval
child_chunks = splitter.split_text(text, chunk_size=128)

# Map each child to its parent
parent_map = {}
for i, child in enumerate(child_chunks):
    parent_start = max(0, i - 1) * 128
    parent_end = min(len(text), (i + 2) * 128)
    parent_map[i] = text[parent_start:parent_end]
```

### Chunking Parameters Guide

| Document Type | Chunk Size | Overlap | Strategy |
|---------------|------------|---------|----------|
| Technical Docs | 400-600 tokens | 100 tokens | Section-aware |
| Legal Docs | 300-400 tokens | 50 tokens | Paragraph-based |
| Chat Logs | 200-300 tokens | 0-20 tokens | Message groups |
| Research Papers | 500-700 tokens | 100 tokens | Section-aware |
| Code Files | 300-500 tokens | 50 tokens | Function-based |

---

## Embedding Models

### What are Embeddings?

Embeddings convert text into dense vectors (arrays of numbers) that capture semantic meaning. Similar text = similar vectors.

### Model Comparison

| Model | Dimensions | Performance | Cost (per 1M tokens) | Best For |
|-------|------------|-------------|----------------------|----------|
| **OpenAI text-embedding-3-large** | 3072 (configurable) | Excellent | $0.13 | Production, high quality |
| **OpenAI text-embedding-3-small** | 1536 | Very Good | $0.02 | Cost-sensitive |
| **Cohere embed-v3** | 1024 | Excellent | $0.10 | Multilingual |
| **Voyage AI voyage-2** | 1024 | Excellent | $0.12 | Domain-specific |
| **BGE-large-en-v1.5** | 1024 | Good | Free (self-hosted) | Budget, privacy |

### Recommendation for Document Q&A

**Start with:** OpenAI text-embedding-3-small
- Excellent quality-to-cost ratio
- Fast inference
- Good for English documents
- Easy integration

**Upgrade to:** text-embedding-3-large if retrieval quality is insufficient

### Critical Considerations

#### Asymmetric Search

Queries and documents have different characteristics:

- **Query:** Short, question format ("What is our refund policy?")
- **Document:** Long, declarative ("Our refund policy states that...")

**Solution:** Use models trained for asymmetric search (most modern models are)

#### Embedding Consistency

**NEVER mix embedding models:**

```
✗ Documents embedded with Model A
✗ Queries embedded with Model B
= Broken retrieval
```

**When upgrading models:** Re-embed entire corpus

### Query Transformation Techniques

#### 1. Hypothetical Document Embeddings (HyDE)

Generate a hypothetical answer, embed that instead of the query:

```python
# Original query
query = "What is our refund policy?"

# Generate hypothetical answer
hypothetical = llm.generate(f"Write a detailed answer to: {query}")
# "Our refund policy allows returns within 30 days..."

# Embed the hypothetical answer
query_embedding = embed(hypothetical)

# Better semantic match with actual documents
results = vector_db.search(query_embedding)
```

**Impact:** 15-25% improvement in retrieval for complex queries

#### 2. Query Expansion

Expand query with synonyms/related terms:

```python
original = "machine learning training"
expanded = """
machine learning training
ML model training
neural network training
deep learning training
"""
```

#### 3. Multi-Query Retrieval

Generate multiple query variations, retrieve for each, merge results:

```python
query = "How do I reset my password?"

variants = llm.generate_variants(query)
# ["password reset procedure",
#  "forgot password steps",
#  "change password instructions"]

all_results = []
for variant in variants:
    results = vector_db.search(variant, top_k=5)
    all_results.extend(results)

# Deduplicate and rerank
final_results = rerank(all_results, query)
```

### Batching Strategy

```python
# Efficient embedding generation
batch_size = 100  # Most APIs support batch requests

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embeddings = embedding_model.embed(batch)
    
    # Store embeddings with metadata
    vectors = [
        {
            "id": f"chunk-{i+j}",
            "values": emb,
            "metadata": metadata[i+j]
        }
        for j, emb in enumerate(embeddings)
    ]
    vector_db.upsert(vectors)
```

**Note:** Batching reduces API calls and cost by ~10x

---

## Vector Database Selection

### Key Requirements

- **Similarity search:** Fast ANN (Approximate Nearest Neighbor)
- **Metadata filtering:** Filter before/after retrieval
- **Scalability:** Handle growing document corpus
- **Hybrid search:** Dense + sparse vectors
- **CRUD operations:** Update/delete documents

### Database Comparison

| Database | Type | Pros | Cons | Best For |
|----------|------|------|------|----------|
| **Pinecone** | Managed | Easy setup, scalable, reliable | Cost, vendor lock-in | Quick start, production |
| **Weaviate** | Open-source/Managed | Hybrid search, GraphQL, modules | Learning curve | Complex retrieval needs |
| **Qdrant** | Open-source/Managed | Fast, Rust-based, good filtering | Smaller ecosystem | Self-hosting, performance |
| **ChromaDB** | Open-source | Simple, embedded mode | Limited scale | Prototyping, small scale |
| **pgvector** | PostgreSQL ext | Use existing PG, ACID | Not optimized for vectors | Existing PG infrastructure |
| **Elasticsearch** | Search engine | Mature, hybrid search, analytics | Heavy, complex | Existing ES infrastructure |

### Recommendation

**Starting out:**
- **Pinecone:** Zero ops, production-ready, $70/month starter plan
- **ChromaDB:** Free, good for dev/prototyping

**Growing system:**
- **Weaviate or Qdrant:** More control, better hybrid search, can self-host for cost savings

### Index Types & Trade-offs

#### HNSW (Hierarchical Navigable Small World)

**Characteristics:**
- ✅ Fast queries (~5-10ms)
- ✅ Good recall (>95%)
- ❌ Slower indexing
- ❌ More memory (~2x vectors)

**Use when:** Query speed is critical, corpus relatively stable

**Configuration:**
```python
index_config = {
    "algorithm": "hnsw",
    "m": 16,  # Number of connections (higher = better recall, more memory)
    "ef_construction": 200,  # Build quality (higher = slower indexing, better quality)
    "ef_search": 100  # Query quality (higher = slower queries, better recall)
}
```

#### IVF (Inverted File Index)

**Characteristics:**
- ✅ Faster indexing
- ✅ Less memory (~1.2x vectors)
- ❌ Slower queries
- ❌ Lower recall (~90%)

**Use when:** Frequent updates, large corpus, can sacrifice some accuracy

### Freshness & Consistency Strategies

#### Real-time Updates

**Pattern:** Update vector DB immediately when document changes

**Pros:**
- ✅ Always fresh
- ✅ Simple to reason about

**Cons:**
- ❌ High embedding cost
- ❌ Potential bottleneck

**Implementation:**
```python
def on_document_update(document):
    # Embed immediately
    chunks = chunk_document(document)
    embeddings = embed_chunks(chunks)
    
    # Delete old chunks
    vector_db.delete(filter={"document_id": document.id})
    
    # Insert new chunks
    vector_db.upsert(embeddings)
```

#### Batch Updates

**Pattern:** Queue changes, process in batches (hourly/daily)

**Pros:**
- ✅ Cost-efficient
- ✅ Better throughput

**Cons:**
- ❌ Eventual consistency
- ❌ Stale data window

**Implementation:**
```python
# Queue updates
update_queue.add(document_id)

# Process in batches
@scheduled(interval="1h")
def process_updates():
    batch = update_queue.get_batch(size=1000)
    documents = load_documents(batch)
    
    # Batch embed
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    embeddings = embed_batch(all_chunks, batch_size=100)
    vector_db.upsert(embeddings)
```

#### Hybrid Approach (Recommended)

```python
# Hot path: Critical documents → real-time updates
if document.priority == "high":
    update_immediately(document)
else:
    # Cold path: Bulk documents → batch updates
    queue_for_batch(document)

# Metadata: Track updated_at, mark stale results in UI
metadata = {
    "updated_at": datetime.now(),
    "version": doc.version,
    "stale": False
}
```

### Deletion & Versioning

```python
# Soft delete pattern (recommended)
metadata = {
    "document_id": "doc-123",
    "deleted": False,  # Toggle instead of hard delete
    "version": 2,      # Track versions
    "valid_until": None  # Expiration for time-sensitive docs
}

# At query time, filter out deleted
filter = {"deleted": {"$eq": False}}
results = vector_db.search(query, filter=filter)

# Hard delete after grace period
@scheduled(interval="1d")
def cleanup_deleted():
    cutoff = datetime.now() - timedelta(days=30)
    vector_db.delete(filter={
        "deleted": True,
        "deleted_at": {"$lt": cutoff}
    })
```

---

## Retrieval Strategies

### Goal

Fetch the most relevant chunks to answer the query while staying within the LLM's context window.

### Basic Retrieval Parameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| `top_k` | 5-20 | Number of chunks to retrieve |
| `similarity_threshold` | 0.7-0.8 | Minimum similarity score |
| `max_tokens` | 4000-8000 | Total context budget |

### Optimal Starting Point

```python
retrieval_config = {
    "top_k": 10,
    "similarity_threshold": 0.75,
    "max_tokens": 6000  # for GPT-4
}
```

**Tune based on evaluation metrics!**

### Advanced Retrieval Patterns

#### 1. Two-Stage Retrieval (Recommended)

```python
# Stage 1: Fast, broad retrieval
initial_results = vector_db.search(
    query_embedding, 
    top_k=100
)

# Stage 2: Rerank with cross-encoder
from cohere import Client
cohere = Client(api_key="...")

reranked = cohere.rerank(
    query=query,
    documents=[r.text for r in initial_results],
    top_n=10,
    model="rerank-english-v3.0"
)

context = [initial_results[r.index] for r in reranked.results]
```

**Impact:** 30-40% improvement in retrieval quality

**Reranker Options:**
- **Cohere Rerank:** Best quality, $2/1000 requests
- **BGE Reranker:** Open-source, self-hostable
- **Cross-Encoder models:** SBERT, MS-MARCO

#### 2. Metadata-Filtered Retrieval

```python
# Filter before vector search for efficiency
filter = {
    "document_type": "technical_manual",
    "department": "engineering",
    "created_at": {"$gte": "2024-01-01"}
}

results = vector_db.search(
    query_embedding,
    filter=filter,
    top_k=10
)
```

**When to use:**
- Multi-tenant systems (filter by user/org)
- Time-sensitive queries (recent docs only)
- Domain-specific queries (filter by category)

#### 3. MMR (Maximum Marginal Relevance)

Diversify results to avoid redundancy:

```python
def mmr(query_embedding, candidates, lambda_param=0.5, top_k=10):
    """
    lambda_param: Balance relevance vs diversity
    - 1.0 = pure relevance
    - 0.0 = pure diversity
    """
    selected = []
    
    while len(selected) < top_k:
        best_score = -float('inf')
        best_idx = None
        
        for i, candidate in enumerate(candidates):
            if i in selected:
                continue
            
            # Relevance score
            relevance = cosine_similarity(
                query_embedding, 
                candidate.embedding
            )
            
            # Diversity penalty (similarity to already selected)
            if selected:
                max_similarity = max([
                    cosine_similarity(
                        candidate.embedding,
                        candidates[j].embedding
                    )
                    for j in selected
                ])
            else:
                max_similarity = 0
            
            # Combined score
            score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        selected.append(best_idx)
    
    return [candidates[i] for i in selected]
```

**Use when:** Documents have overlapping content

#### 4. Context Window Expansion

Retrieve small chunks, but include surrounding context:

```python
# Retrieve precise chunk
top_chunk = vector_db.search(query, top_k=1)[0]

# Fetch neighboring chunks for more context
context_chunks = get_chunks(
    document_id=top_chunk.document_id,
    start_index=top_chunk.chunk_index - 1,  # Previous chunk
    end_index=top_chunk.chunk_index + 1      # Next chunk
)

# Combine for LLM
full_context = "\n\n".join([c.text for c in context_chunks])
```

### Hallucination Reduction Techniques

#### Citation Requirements

Force the LLM to cite sources:

```python
prompt = f"""
Answer based ONLY on the context below.
For each claim, cite the source using [1], [2], etc.
If the answer isn't in the context, respond with "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:
"""
```

#### Confidence Scoring

Ask LLM to rate its confidence:

```python
prompt = f"""
{standard_prompt}

After your answer, provide:
Confidence: [Low/Medium/High]
Reasoning: [Why this confidence level]
"""

response = llm.generate(prompt)

# Parse confidence
if "Confidence: Low" in response:
    add_warning("This answer may not be reliable")
```

#### Retrieval Quality Checks

```python
def check_retrieval_quality(query, results):
    # Check 1: Minimum similarity
    max_similarity = max([r.score for r in results])
    if max_similarity < 0.7:
        return {
            "quality": "low",
            "warning": "No highly relevant documents found"
        }
    
    # Check 2: Result diversity
    similarities = []
    for i in range(len(results)-1):
        sim = cosine_similarity(
            results[i].embedding,
            results[i+1].embedding
        )
        similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities)
    if avg_similarity > 0.95:
        return {
            "quality": "medium",
            "warning": "Results are very similar (possible redundancy)"
        }
    
    return {"quality": "high"}
```

#### Answer Grounding Verification

```python
def verify_answer_grounding(answer, context):
    # Embed answer and context
    answer_embedding = embed(answer)
    context_embedding = embed(context)
    
    # Check semantic similarity
    similarity = cosine_similarity(answer_embedding, context_embedding)
    
    if similarity < 0.6:
        return {
            "grounded": False,
            "warning": "Answer may not be grounded in context",
            "action": "flag_for_review"
        }
    
    return {"grounded": True}
```

### Query Routing

```python
def route_query(query):
    query_type = classify_query(query)
    
    if query_type == "factual":
        # Use precise retrieval with high threshold
        return {
            "top_k": 5,
            "threshold": 0.85,
            "rerank": True
        }
    
    elif query_type == "exploratory":
        # Use broad retrieval with diversity
        return {
            "top_k": 20,
            "threshold": 0.70,
            "mmr": True,
            "lambda": 0.3
        }
    
    elif query_type == "aggregation":
        # Retrieve many documents for summarization
        return {
            "top_k": 50,
            "threshold": 0.65,
            "group_by": "document_id"
        }
    
    else:
        # Default strategy
        return {
            "top_k": 10,
            "threshold": 0.75
        }
```

---

## Generation Phase

### Overview

After retrieving relevant context, we pass it to the LLM to generate the final answer. Prompt engineering is critical here.

### Prompt Template Structure

```python
SYSTEM_PROMPT = """
You are a helpful assistant answering questions about {domain}.
Use ONLY the provided context to answer questions.
If the context doesn't contain the answer, say so clearly.
Always cite your sources using [1], [2], etc.
Be concise but complete in your answers.
"""

USER_PROMPT = """
Context:
{context_chunks}

Question: {user_query}

Instructions:
1. Answer based ONLY on the context above
2. Cite sources for each claim: [1], [2]
3. If unsure or information is missing, say "I don't have enough information"
4. Be concise but complete

Answer:
"""
```

### Context Formatting

#### Option 1: Numbered Chunks

```
Context:
[1] User Manual - Chapter 3: Installation
To install the software, first download the installer from...

[2] FAQ - General Questions
Q: How do I install? A: Download the installer and run...

[3] Release Notes - Version 2.0
This version includes a new installation wizard that...
```

**Pros:** Easy citation, clear structure

#### Option 2: XML Tags (Recommended)

```xml
Context:

<document id="1" title="User Manual" section="Chapter 3: Installation">
To install the software, first download the installer from our website.
Run the installer and follow the on-screen instructions.
</document>

<document id="2" title="FAQ" section="General Questions">
Q: How do I install?
A: Download the installer from our download page and run it.
The installation typically takes 5-10 minutes.
</document>

<document id="3" title="Release Notes" section="Version 2.0">
This version includes a new installation wizard that guides you
through the process step-by-step.
</document>
```

**Pros:** Richer metadata, better parsing, LLMs handle XML well

**Implementation:**
```python
def format_context(chunks):
    context = []
    for i, chunk in enumerate(chunks):
        context.append(f"""
<document id="{i+1}" title="{chunk.metadata['title']}" section="{chunk.metadata.get('section', 'N/A')}">
{chunk.text}
</document>
""")
    return "\n".join(context)
```

### Model Selection

| Model | Cost (Input / Output per 1M tokens) | Context Window | Best For |
|-------|-------------------------------------|----------------|----------|
| **GPT-4 Turbo** | $10 / $30 | 128K | Complex reasoning, accuracy critical |
| **GPT-4o** | $2.50 / $10 | 128K | Balanced cost/quality |
| **GPT-3.5 Turbo** | $0.50 / $1.50 | 16K | Simple Q&A, high volume |
| **Claude 3.5 Sonnet** | $3 / $15 | 200K | Balanced cost/quality, long docs |
| **Claude 3 Opus** | $15 / $75 | 200K | Long documents, nuanced answers |
| **Claude 3 Haiku** | $0.25 / $1.25 | 200K | Cost-sensitive, simple queries |

### Recommendation

**Start with:** Claude 3.5 Sonnet or GPT-4o
- Good balance of cost and quality
- Fast inference
- Sufficient for most Q&A tasks

**Upgrade to:** GPT-4 Turbo or Claude 3 Opus for complex reasoning or critical accuracy

### Advanced Prompting Techniques

#### Few-Shot Examples

```python
SYSTEM_PROMPT = """
You are a helpful assistant. Here are examples of good answers:

Example 1:
Question: What is our refund policy?
Context: [1] Our refund policy allows returns within 30 days of purchase.
Answer: According to the policy [1], you can return items within 30 days of purchase for a full refund.

Example 2:
Question: Do we ship internationally?
Context: [1] We currently only ship within the United States.
Answer: No, we currently only ship within the United States [1].

Now answer the following question:
"""
```

#### Chain-of-Thought Prompting

```python
prompt = f"""
Context:
{context}

Question: {query}

Think through this step by step:
1. What information from the context is relevant?
2. How does this information answer the question?
3. What is the final answer?

Answer:
"""
```

#### Self-Consistency

Generate multiple answers and select the most consistent:

```python
answers = []
for _ in range(3):
    answer = llm.generate(prompt, temperature=0.7)
    answers.append(answer)

# Select most common answer or use voting
final_answer = select_most_consistent(answers)
```

### Response Post-Processing

#### Citation Extraction

```python
import re

def extract_citations(response):
    # Extract citation numbers [1], [2], etc.
    citations = re.findall(r'\[(\d+)\]', response)
    citations = list(set(citations))  # Deduplicate
    citations = [int(c) for c in citations]
    return sorted(citations)

def enhance_response(response, chunks):
    citations = extract_citations(response)
    
    return {
        "answer": response,
        "sources": [
            {
                "id": i,
                "title": chunks[i-1].metadata['title'],
                "text": chunks[i-1].text[:200] + "...",
                "similarity": chunks[i-1].score,
                "url": chunks[i-1].metadata.get('url')
            }
            for i in citations
            if i <= len(chunks)
        ],
        "metadata": {
            "model": "claude-3-sonnet",
            "total_chunks": len(chunks),
            "cited_chunks": len(citations)
        }
    }
```

#### Confidence Filtering

```python
def assess_confidence(response, retrieval_results):
    # Check for uncertainty phrases
    uncertainty_phrases = [
        "I don't have enough information",
        "I'm not sure",
        "The context doesn't mention",
        "It's unclear"
    ]
    
    has_uncertainty = any(
        phrase.lower() in response.lower() 
        for phrase in uncertainty_phrases
    )
    
    # Check retrieval quality
    max_similarity = max([r.score for r in retrieval_results])
    
    if has_uncertainty or max_similarity < 0.7:
        return {
            "confidence": "low",
            "suggestion": "Try rephrasing or providing more context"
        }
    elif max_similarity < 0.8:
        return {"confidence": "medium"}
    else:
        return {"confidence": "high"}
```

#### Fact Verification

```python
def verify_facts(answer, context):
    # Split answer into claims
    claims = split_into_claims(answer)
    
    verification_results = []
    for claim in claims:
        # Check if claim is supported by context
        claim_embedding = embed(claim)
        context_embedding = embed(context)
        
        similarity = cosine_similarity(claim_embedding, context_embedding)
        
        verification_results.append({
            "claim": claim,
            "supported": similarity > 0.7,
            "confidence": similarity
        })
    
    # Flag if any claims are unsupported
    unsupported = [v for v in verification_results if not v['supported']]
    if unsupported:
        return {
            "verified": False,
            "unsupported_claims": unsupported,
            "action": "flag_for_review"
        }
    
    return {"verified": True}
```

### Streaming Responses

```python
async def stream_response(query, context):
    prompt = build_prompt(query, context)
    
    accumulated = ""
    async for chunk in llm.stream(prompt):
        accumulated += chunk
        yield chunk
    
    # Post-processing on complete response
    citations = extract_citations(accumulated)
    confidence = assess_confidence(accumulated, context)
    
    # Send metadata at the end
    yield {
        "type": "metadata",
        "citations": citations,
        "confidence": confidence
    }
```

---

## Evaluation Methods

### Why Evaluation Matters

You can't improve what you don't measure. RAG systems need continuous evaluation to ensure quality.

### Key Metrics

#### Retrieval Metrics

| Metric | Definition | Formula | Target |
|--------|------------|---------|--------|
| **Precision@k** | % of retrieved docs that are relevant | `relevant ∩ retrieved / retrieved` | >80% |
| **Recall@k** | % of relevant docs that were retrieved | `relevant ∩ retrieved / relevant` | >70% |
| **MRR** | Mean Reciprocal Rank of first relevant doc | `1/rank_of_first_relevant` | >0.7 |
| **NDCG@k** | Normalized Discounted Cumulative Gain | `DCG / IDCG` | >0.75 |

#### Generation Metrics

| Metric | Definition | How to Measure |
|--------|------------|----------------|
| **Faithfulness** | Answer grounded in context? | LLM-as-judge or NLI models |
| **Answer Relevance** | Answer addresses the query? | Semantic similarity |
| **Context Utilization** | Uses retrieved context? | Citation analysis |
| **Hallucination Rate** | % of claims not in context | Manual review + LLM-as-judge |

### Evaluation Frameworks

#### RAGAS (Recommended)

Automated RAG evaluation framework:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation dataset
data = {
    "question": ["What is our refund policy?", ...],
    "answer": ["You can return items within 30 days...", ...],
    "contexts": [["Our refund policy allows...", ...], ...],
    "ground_truth": ["30-day return policy", ...]
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,           # Answer grounded in context?
        answer_relevancy,       # Answer relevant to query?
        context_precision,      # Retrieved contexts relevant?
        context_recall          # All relevant contexts retrieved?
    ]
)

print(result)
# {
#   'faithfulness': 0.92,
#   'answer_relevancy': 0.88,
#   'context_precision': 0.85,
#   'context_recall': 0.78
# }
```

#### LangSmith

End-to-end tracing and evaluation:

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define evaluator
def correctness_evaluator(run, example):
    # Compare generated answer to ground truth
    score = compute_similarity(
        run.outputs["answer"],
        example.outputs["ground_truth"]
    )
    return {"score": score}

# Run evaluation
results = evaluate(
    lambda inputs: rag_pipeline(inputs["question"]),
    data="rag-test-set",
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-v1"
)
```

### Creating Test Sets

#### 1. Manual Golden Set (Recommended)

Curate 50-100 query-answer pairs:

```python
test_cases = [
    {
        "query": "What is our refund policy?",
        "expected_answer": "30-day money-back guarantee with no questions asked",
        "relevant_docs": ["doc-123", "doc-456"],
        "difficulty": "easy"
    },
    {
        "query": "How do I configure SSO with Okta?",
        "expected_answer": "Go to Settings > Authentication > Configure SAML...",
        "relevant_docs": ["doc-789"],
        "difficulty": "medium"
    },
    {
        "query": "What are the performance implications of enabling audit logs?",
        "expected_answer": "Approximately 5-10% overhead, scales with write volume...",
        "relevant_docs": ["doc-234", "doc-567", "doc-890"],
        "difficulty": "hard"
    }
]
```

**Effort:** High, but most reliable  
**Coverage:** Focus on common queries and edge cases

#### 2. Synthetic Generation

Use LLMs to generate test cases from documents:

```python
def generate_test_cases(chunk):
    prompt = f"""
Given this document chunk:
{chunk}

Generate 3 questions that can be answered using this chunk.
For each question, provide:
1. The question
2. The expected answer (quote from the chunk)
3. Difficulty level (easy/medium/hard)

Format as JSON.
"""
    
    response = llm.generate(prompt)
    return parse_json(response)

# Generate test cases from all chunks
test_cases = []
for chunk in sample_chunks(corpus, n=100):
    cases = generate_test_cases(chunk)
    test_cases.extend(cases)

# Manual review and filtering
test_cases = review_and_filter(test_cases)
```

**Effort:** Low, but review for quality  
**Coverage:** Good diversity, scales with corpus size

#### 3. User Query Sampling

Sample real user queries and manually label:

```python
# Sample queries from logs
queries = sample_user_queries(n=100, period="last_30_days")

# For each query, manually label:
labeled_data = []
for query in queries:
    # Retrieve documents
    results = rag_pipeline.retrieve(query)
    
    # Manual labeling
    relevant_docs = manually_label_relevant(query, results)
    expected_answer = manually_write_answer(query, results)
    
    labeled_data.append({
        "query": query,
        "relevant_docs": relevant_docs,
        "expected_answer": expected_answer
    })
```

### Continuous Evaluation Pipeline

```
1. Collect user queries + system responses
   ↓
2. Sample random subset (5-10% of traffic)
   ↓
3. Run automated metrics (RAGAS, LLM-as-judge)
   ↓
4. Flag low-scoring responses (score < 0.7)
   ↓
5. Manual review of flagged responses
   ↓
6. Add high-quality examples to golden set
   ↓
7. Analyze failure patterns
   ↓
8. Retrain/tune retrieval parameters
   ↓
9. Deploy and repeat weekly/monthly
```

### Key Performance Indicators

Track these metrics over time:

| KPI | Target | How to Track |
|-----|--------|--------------|
| **Retrieval Precision@10** | >80% | Golden set evaluation |
| **Answer Faithfulness** | >90% | RAGAS / LLM-as-judge |
| **User Satisfaction** | >4.0/5.0 | Thumbs up/down feedback |
| **Hallucination Rate** | <5% | Manual review + automated |
| **Citation Rate** | >85% | Parse [1], [2] tags |
| **Avg Response Time** | <3s | Application monitoring |
| **Cost per Query** | <$0.05 | Track API usage |

### A/B Testing

Test changes systematically:

```python
# Define variants
variants = {
    "control": {
        "chunking": "fixed-512",
        "retrieval": "dense-only",
        "top_k": 10
    },
    "treatment": {
        "chunking": "semantic-500",
        "retrieval": "hybrid",
        "top_k": 15
    }
}

# Route traffic
def get_variant(user_id):
    if hash(user_id) % 2 == 0:
        return "control"
    else:
        return "treatment"

# Track metrics by variant
@log_metrics
def handle_query(user_id, query):
    variant = get_variant(user_id)
    config = variants[variant]
    
    response = rag_pipeline(query, config)
    
    log_event(
        variant=variant,
        query=query,
        response=response,
        user_feedback=None  # Collect later
    )
    
    return response

# Analyze after 1-2 weeks
results = analyze_ab_test(
    control="control",
    treatment="treatment",
    metrics=["precision", "user_satisfaction", "latency"]
)

if results["treatment"]["precision"] > results["control"]["precision"]:
    promote_variant("treatment")
```

### Common Failure Patterns

#### Pattern 1: Retrieval Misses

**Symptom:** Relevant documents not retrieved

**Diagnosis:**
```python
# Check if relevant docs exist
relevant_docs = get_relevant_docs(query)
retrieved_docs = retrieve(query, top_k=20)

missing = set(relevant_docs) - set(retrieved_docs)
if missing:
    print(f"Failed to retrieve: {missing}")
    
    # Check similarity scores
    for doc_id in missing:
        doc = get_document(doc_id)
        score = compute_similarity(query, doc)
        print(f"{doc_id}: score={score}")
```

**Solutions:**
- Query expansion
- HyDE
- Lower similarity threshold
- Better embeddings

#### Pattern 2: Context Fragmentation

**Symptom:** Answer requires information from multiple non-contiguous chunks

**Diagnosis:**
```python
# Check if citations span multiple distant chunks
citations = extract_citations(answer)
chunk_indices = [chunks[c-1].chunk_index for c in citations]

gap = max(chunk_indices) - min(chunk_indices)
if gap > 5:
    print(f"Context fragmented across {gap} chunks")
```

**Solutions:**
- Larger chunks
- Parent-child chunking
- Multi-hop retrieval

#### Pattern 3: Hallucination

**Symptom:** Answer contains information not in context

**Diagnosis:**
```python
# Verify each claim
claims = extract_claims(answer)
for claim in claims:
    grounded = verify_grounding(claim, context)
    if not grounded:
        print(f"Hallucinated claim: {claim}")
```

**Solutions:**
- Stronger citation requirements
- Lower LLM temperature
- Fact verification step
- Better prompts

---

## Implementation Guide

### Production-Ready Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│           API Gateway / Load Balancer        │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────┐
│  RAG API    │  (FastAPI / Flask)
├─────────────┤
│ - Query     │
│ - Retrieve  │
│ - Generate  │
└──────┬──────┘
       │
       ├─────────────────┬──────────────┬───────────────┐
       │                 │              │               │
┌──────▼──────┐   ┌─────▼─────┐   ┌───▼────┐   ┌──────▼──────┐
│Vector DB    │   │Embedding  │   │  LLM   │   │ Cache       │
│(Pinecone)   │   │   API     │   │  API   │   │ (Redis)     │
└─────────────┘   └───────────┘   └────────┘   └─────────────┘
                                                        
┌────────────────────────────────────────────┐
│        Background Jobs (Celery)            │
├────────────────────────────────────────────┤
│ - Document ingestion                       │
│ - Batch embedding                          │
│ - Index updates                            │
└────────────────────────────────────────────┘
```

### Tech Stack Recommendation

| Component | Technology | Alternative |
|-----------|------------|-------------|
| **API Framework** | FastAPI | Flask, Django |
| **Vector DB** | Pinecone | Weaviate, Qdrant |
| **Embeddings** | OpenAI | Cohere, Voyage |
| **LLM** | Claude 3.5 Sonnet | GPT-4o, GPT-3.5 |
| **Orchestration** | LangChain | LlamaIndex, custom |
| **Cache** | Redis | Memcached |
| **Queue** | Celery + Redis | RabbitMQ, AWS SQS |
| **Monitoring** | LangSmith | Prometheus + Grafana |

### Project Structure

```
rag-system/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # Load PDFs, docs, etc.
│   │   ├── chunker.py             # Chunking strategies
│   │   └── embedder.py            # Embedding generation
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py        # Vector DB interface
│   │   ├── reranker.py            # Reranking logic
│   │   └── query_transformer.py   # HyDE, expansion, etc.
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py          # LLM API wrapper
│   │   └── prompt_templates.py    # Prompt management
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── test_sets.py           # Test data management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # API endpoints
│   │   ├── models.py              # Pydantic models
│   │   └── middleware.py          # Auth, logging, etc.
│   └── utils/
│       ├── __init__.py
│       ├── cache.py               # Caching utilities
│       └── logging.py             # Logging setup
├── config/
│   ├── config.yaml                # System configuration
│   └── prompts.yaml               # Prompt templates
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_e2e.py
├── scripts/
│   ├── ingest_documents.py        # Batch ingestion
│   └── evaluate_system.py         # Run evaluations
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Core Implementation

#### 1. Document Ingestion Pipeline

```python
# src/ingestion/document_loader.py
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from typing import List, Dict
import hashlib

class DocumentLoader:
    def load(self, file_path: str) -> Dict:
        """Load document based on file type"""
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'pdf':
            loader = PyPDFLoader(file_path)
        elif ext == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        pages = loader.load()
        
        return {
            "text": "\n\n".join([p.page_content for p in pages]),
            "metadata": {
                "file_path": file_path,
                "file_type": ext,
                "num_pages": len(pages),
                "hash": self._compute_hash(file_path)
            }
        }
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute hash for change detection"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()


# src/ingestion/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def chunk(self, document: Dict) -> List[Dict]:
        """Chunk document with context enhancement"""
        text = document["text"]
        metadata = document["metadata"]
        
        chunks = self.splitter.split_text(text)
        
        # Enhance chunks with context
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced = self._add_context(chunk, metadata)
            enhanced_chunks.append({
                "text": enhanced,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return enhanced_chunks
    
    def _add_context(self, chunk: str, metadata: Dict) -> str:
        """Add document context to chunk"""
        title = metadata.get("title", "Unknown")
        section = metadata.get("section", "")
        
        context = f"[Document: {title}]"
        if section:
            context += f"\n[Section: {section}]"
        
        return f"{context}\n\n{chunk}"


# src/ingestion/embedder.py
from openai import OpenAI
from typing import List
import os

class Embedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings


# Ingestion pipeline
from src.ingestion import DocumentLoader, DocumentChunker, Embedder
from src.retrieval import VectorStore

class IngestionPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        self.embedder = Embedder()
        self.vector_store = VectorStore()
    
    def ingest(self, file_path: str, metadata: Dict = None):
        """Ingest a document"""
        # Load
        document = self.loader.load(file_path)
        if metadata:
            document["metadata"].update(metadata)
        
        # Chunk
        chunks = self.chunker.chunk(document)
        
        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Store
        vectors = [
            {
                "id": f"{document['metadata']['hash']}-{i}",
                "values": embedding,
                "metadata": chunk["metadata"]
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        self.vector_store.upsert(vectors)
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "document_id": document["metadata"]["hash"]
        }
```

#### 2. Query Pipeline

```python
# src/retrieval/vector_store.py
import pinecone
from typing import List, Dict
import os

class VectorStore:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        self.index = pinecone.Index("documents")
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter: Dict = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text"),
                "metadata": match.metadata
            }
            for match in results.matches
        ]
    
    def upsert(self, vectors: List[Dict]):
        """Insert or update vectors"""
        self.index.upsert(vectors=vectors)


# src/retrieval/reranker.py
from cohere import Client
import os

class Reranker:
    def __init__(self):
        self.client = Client(api_key=os.getenv("COHERE_API_KEY"))
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Dict]:
        """Rerank documents using cross-encoder"""
        response = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model="rerank-english-v3.0"
        )
        
        return [
            {
                "index": result.index,
                "score": result.relevance_score,
                "text": documents[result.index]
            }
            for result in response.results
        ]


# src/generation/llm_client.py
from anthropic import Anthropic
import os

class LLMClient:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate response"""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=messages
        )
        
        return response.content[0].text


# RAG Pipeline
class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.llm = LLMClient()
    
    def query(self, question: str, top_k: int = 10) -> Dict:
        """Execute RAG query"""
        # 1. Embed query
        query_embedding = self.embedder.embed_batch([question])[0]
        
        # 2. Retrieve candidates
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=50
        )
        
        # 3. Rerank
        reranked = self.reranker.rerank(
            query=question,
            documents=[r["text"] for r in results],
            top_k=top_k
        )
        
        # Map back to original results
        final_results = [results[r["index"]] for r in reranked]
        
        # 4. Build context
        context = self._format_context(final_results)
        
        # 5. Generate answer
        prompt = self._build_prompt(question, context)
        answer = self.llm.generate(prompt)
        
        # 6. Post-process
        citations = self._extract_citations(answer)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "id": i+1,
                    "title": final_results[i]["metadata"].get("title"),
                    "text": final_results[i]["text"][:200] + "...",
                    "score": final_results[i]["score"]
                }
                for i in citations if i < len(final_results)
            ],
            "metadata": {
                "model": self.llm.model,
                "num_chunks": len(final_results)
            }
        }
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context for LLM"""
        formatted = []
        for i, chunk in enumerate(chunks):
            title = chunk["metadata"].get("title", "Unknown")
            section = chunk["metadata"].get("section", "")
            
            formatted.append(f"""
<document id="{i+1}" title="{title}" section="{section}">
{chunk["text"]}
</document>
""")
        
        return "\n".join(formatted)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""Answer the question based on the context below.
Cite sources using [1], [2], etc.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
    
    def _extract_citations(self, answer: str) -> List[int]:
        """Extract citation numbers from answer"""
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        return sorted(list(set(int(c) - 1 for c in citations)))
```

#### 3. API Implementation

```python
# src/api/routes.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
import logging

app = FastAPI(title="RAG API")
rag = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10

class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Handle query request"""
    try:
        result = rag.query(
            question=request.question,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}
```

### Optimization Strategies

#### Caching

```python
# src/utils/cache.py
import redis
import json
import hashlib
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache(ttl: int = 3600):
    """Cache decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{func.__name__}:{args}:{kwargs}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()
            
            # Try cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=3600)
def embed_query(text: str):
    return embedder.embed_batch([text])[0]
```

#### Rate Limiting

```python
# src/api/middleware.py
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query_request: QueryRequest):
    # ... query logic
    pass
```

### Deployment Checklist

- ✅ **API rate limiting**: Prevent abuse (10-100 req/min per user)
- ✅ **Error handling**: Graceful degradation, user-friendly messages
- ✅ **Monitoring**: Track latency, errors, costs (Prometheus + Grafana)
- ✅ **Logging**: Log all queries/answers for evaluation (structured logs)
- ✅ **Authentication**: API keys or OAuth2
- ✅ **CORS**: Configure for web apps
- ✅ **Health checks**: Monitor Vector DB, LLM, cache availability
- ✅ **Backup strategy**: Regular vector DB snapshots
- ✅ **Cost tracking**: Monitor LLM/embedding API costs
- ✅ **User feedback**: Thumbs up/down on answers
- ✅ **Alerting**: Slack/PagerDuty for critical issues
- ✅ **Documentation**: API docs, architecture diagrams

### Docker Deployment

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  celery:
    build: .
    command: celery -A src.tasks worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
```

---

## Next Steps

### Phase 1: Proof of Concept (Week 1-2)

1. Set up development environment
2. Start with ChromaDB locally (no account needed)
3. Curate 20-30 test Q&A pairs from your documents
4. Implement basic RAG pipeline using LangChain
5. Evaluate baseline performance

### Phase 2: Optimization (Week 3-4)

1. Experiment with chunking strategies
2. Add reranking (Cohere or open-source)
3. Implement caching for embeddings
4. Test different retrieval parameters (top_k, threshold)
5. Measure improvements on test set

### Phase 3: Production (Week 5-6)

1. Migrate to production vector DB (Pinecone)
2. Build FastAPI wrapper
3. Add monitoring and logging
4. Deploy with Docker
5. Set up continuous evaluation pipeline

### Phase 4: Enhancement (Ongoing)

1. Collect user feedback
2. Expand test set with real queries
3. A/B test new features
4. Monitor costs and optimize
5. Keep iterating based on metrics

---

## Resources

### Libraries & Tools

- **LangChain**: https://python.langchain.com/
- **LlamaIndex**: https://www.llamaindex.ai/
- **RAGAS**: https://github.com/explodinggradients/ragas
- **LangSmith**: https://smith.langchain.com/

### Vector Databases

- **Pinecone**: https://www.pinecone.io/
- **Weaviate**: https://weaviate.io/
- **Qdrant**: https://qdrant.tech/
- **ChromaDB**: https://www.trychroma.com/

### Papers & Articles

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Precise Zero-Shot Dense Retrieval" (HyDE paper)
- "Lost in the Middle" (context window optimization)

### Evaluation Frameworks

- **RAGAS**: Automated RAG evaluation
- **TruLens**: LLM observability and evaluation
- **DeepEval**: End-to-end LLM testing

---

This guide provides a comprehensive foundation for building a production-grade RAG system for document Q&A. Start simple, measure everything, and iterate based on data. Good luck!