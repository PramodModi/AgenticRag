# Deep Dive: Hierarchical Chunking & Reranking

## Table of Contents
1. [Hierarchical Chunking](#hierarchical-chunking)
   - Core Concepts & Architecture
   - Implementation Strategies
   - Storage & Indexing
   - Query-Time Retrieval
2. [Reranking](#reranking)
   - Why Reranking Matters
   - Reranking Models & Architectures
   - Implementation Patterns
   - Optimization Techniques
3. [Combining Both](#combining-hierarchical-chunking--reranking)
4. [Production Implementation](#production-implementation)

---

## Hierarchical Chunking

### Core Concept

Traditional chunking treats all chunks equally. Hierarchical chunking creates a **parent-child relationship** where:

- **Parent chunks**: High-level summaries or section headers (metadata-rich, broad context)
- **Child chunks**: Detailed content (specific facts, examples, technical details)

**Why it matters:**
```
❌ Traditional: Retrieve chunk #47 → "The model uses attention mechanisms..."
   Problem: No context about what model, which section, broader topic

✅ Hierarchical: Retrieve Parent → "Section: Neural Architecture" 
                 Child → "The model uses attention mechanisms..."
   Benefit: Context hierarchy preserved
```

### Architecture Patterns

#### Pattern 1: Document → Section → Paragraph

```
Document (Parent L0)
├── Introduction (Parent L1)
│   ├── Background paragraph (Child L2)
│   ├── Problem statement (Child L2)
│   └── Contributions (Child L2)
├── Related Work (Parent L1)
│   ├── Traditional approaches (Child L2)
│   └── Recent advances (Child L2)
└── Methodology (Parent L1)
    ├── Model architecture (Child L2)
    │   ├── Encoder details (Child L3)
    │   └── Decoder details (Child L3)
    └── Training procedure (Child L2)
```

**Retrieval strategy:**
1. Search at parent level (L1) → Find relevant sections
2. Expand to children (L2) → Get detailed content
3. Optionally drill to L3 if needed

#### Pattern 2: Summary → Detail Hierarchy

```
Parent Chunk (Summary):
"This document discusses neural architecture search methods, 
focusing on gradient-based and evolutionary approaches. 
Key contributions include a novel search space and 
efficient training strategy."

Child Chunks:
├── "Gradient-based NAS uses continuous relaxation..."
├── "Evolutionary approaches maintain a population..."
├── "The search space consists of 7 operations..."
└── "Training uses a weight-sharing strategy..."
```

**Retrieval strategy:**
1. Retrieve parent summaries via semantic search
2. If parent is relevant, retrieve all its children
3. LLM uses summary for context + details for facts

#### Pattern 3: Sliding Window with Overlap Hierarchy

```
Document: [A B C D E F G H I J K L M N O P]

Parent chunks (large, 8 tokens, stride 6):
P1: [A B C D E F G H]
P2: [G H I J K L M N]
P3: [M N O P]

Child chunks (small, 4 tokens, stride 3):
C1: [A B C D]  ──┐
C2: [D E F G]    ├─ Link to P1
C3: [G H I J]  ──┘
C4: [J K L M]  ──┐
C5: [M N O P]    ├─ Link to P2
                 └─ Link to P3
```

**Benefits:**
- Captures context across boundaries
- Retrieve at appropriate granularity
- Child chunks never lack context

### Implementation Deep Dive

#### Strategy 1: Explicit Summary Generation

**Best for:** Long documents, technical content, research papers

```python
from typing import List, Dict, Optional
import uuid

class HierarchicalChunker:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embed = embedding_model
    
    def chunk_with_summaries(
        self, 
        document: str,
        chunk_size: int = 512,
        summary_prompt: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Create hierarchical chunks with LLM-generated summaries
        """
        # Step 1: Split into sections (use document structure)
        sections = self._split_by_structure(document)
        
        # Step 2: Create hierarchy
        hierarchy = {
            "parents": [],
            "children": [],
            "relationships": []  # parent_id -> [child_ids]
        }
        
        for section in sections:
            # Generate parent (summary)
            parent_id = str(uuid.uuid4())
            summary = self.llm.generate(
                f"Summarize the following section in 2-3 sentences:\n\n{section.content}"
            )
            
            parent = {
                "id": parent_id,
                "content": summary,
                "metadata": {
                    "type": "summary",
                    "section_title": section.title,
                    "doc_id": document.id,
                    "level": 1
                },
                "embedding": self.embed(summary)
            }
            hierarchy["parents"].append(parent)
            
            # Create children (detailed chunks)
            child_chunks = self._chunk_text(section.content, chunk_size)
            child_ids = []
            
            for i, chunk in enumerate(child_chunks):
                child_id = str(uuid.uuid4())
                child = {
                    "id": child_id,
                    "content": chunk,
                    "metadata": {
                        "type": "detail",
                        "parent_id": parent_id,
                        "section_title": section.title,
                        "chunk_index": i,
                        "level": 2
                    },
                    "embedding": self.embed(chunk)
                }
                hierarchy["children"].append(child)
                child_ids.append(child_id)
            
            hierarchy["relationships"].append({
                "parent_id": parent_id,
                "child_ids": child_ids
            })
        
        return hierarchy
    
    def _split_by_structure(self, document: str) -> List:
        """
        Split document by natural structure (headers, sections)
        """
        # Use markdown headers, or heuristics for plain text
        import re
        
        # Match markdown headers
        sections = []
        current_section = None
        
        for line in document.split('\n'):
            if re.match(r'^#{1,3}\s+', line):  # Headers
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": line.strip('#').strip(),
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Simple sentence-aware chunking
        """
        import nltk
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
```

**Usage:**
```python
chunker = HierarchicalChunker(
    llm=GPT4(),
    embedding_model=OpenAIEmbeddings()
)

document = load_document("research_paper.pdf")
hierarchy = chunker.chunk_with_summaries(document, chunk_size=512)

# Store in vector DB with relationships
vector_db.add_chunks(
    hierarchy["parents"],  
    collection="parents"
)
vector_db.add_chunks(
    hierarchy["children"], 
    collection="children"
)
```

#### Strategy 2: Proposition-based Chunking

**Best for:** Dense factual content, knowledge bases, documentation

**Concept:** Break document into atomic propositions (single facts), then group hierarchically

```python
class PropositionChunker:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic facts from text using LLM
        """
        prompt = f"""
Extract all factual claims from the following text as a list of 
simple, atomic propositions. Each proposition should be self-contained.

Text: {text}

Propositions:
"""
        response = self.llm.generate(prompt)
        return self._parse_propositions(response)
    
    def create_hierarchy(
        self, 
        propositions: List[str],
        cluster_size: int = 5
    ) -> Dict:
        """
        Cluster propositions into hierarchical groups
        """
        # Embed propositions
        embeddings = [self.embed(prop) for prop in propositions]
        
        # Cluster into groups (semantic similarity)
        clusters = self._cluster_embeddings(embeddings, k=len(propositions) // cluster_size)
        
        hierarchy = {"parents": [], "children": []}
        
        for cluster_id, prop_indices in clusters.items():
            # Parent = summary of cluster
            cluster_props = [propositions[i] for i in prop_indices]
            parent_summary = self.llm.generate(
                f"Summarize these related facts in one sentence:\n" +
                "\n".join(f"- {p}" for p in cluster_props)
            )
            
            parent_id = str(uuid.uuid4())
            hierarchy["parents"].append({
                "id": parent_id,
                "content": parent_summary,
                "child_ids": [f"prop_{i}" for i in prop_indices]
            })
            
            # Children = individual propositions
            for i in prop_indices:
                hierarchy["children"].append({
                    "id": f"prop_{i}",
                    "content": propositions[i],
                    "parent_id": parent_id
                })
        
        return hierarchy
```

**Example:**
```python
text = """
The Transformer architecture was introduced in 2017. It uses 
self-attention mechanisms instead of recurrence. BERT is based 
on the Transformer encoder. GPT uses the Transformer decoder. 
Both models achieve state-of-the-art results on NLP tasks.
"""

propositions = [
    "The Transformer architecture was introduced in 2017",
    "Transformers use self-attention mechanisms",
    "Transformers do not use recurrence",
    "BERT is based on the Transformer encoder",
    "GPT uses the Transformer decoder",
    "BERT achieves state-of-the-art results on NLP tasks",
    "GPT achieves state-of-the-art results on NLP tasks"
]

# Cluster into parent groups:
Parent 1: "Transformer architecture fundamentals (2017, self-attention)"
  ├── Child: "Introduced in 2017"
  ├── Child: "Uses self-attention"
  └── Child: "No recurrence"

Parent 2: "Transformer-based models and their performance"
  ├── Child: "BERT uses encoder"
  ├── Child: "GPT uses decoder"
  └── Child: "State-of-the-art results"
```

#### Strategy 3: Recursive Summarization Tree

**Best for:** Very long documents (100+ pages), books, comprehensive guides

```python
class RecursiveSummarizer:
    def __init__(self, llm, max_chunk_size=2000, summary_ratio=0.3):
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.summary_ratio = summary_ratio  # Summary should be 30% of original
    
    def build_tree(self, document: str, level=0) -> Dict:
        """
        Recursively build summary tree
        """
        doc_length = len(document.split())
        
        # Base case: document small enough
        if doc_length <= self.max_chunk_size:
            return {
                "level": level,
                "content": document,
                "is_leaf": True,
                "children": []
            }
        
        # Recursive case: split and summarize
        chunks = self._split_document(document, self.max_chunk_size)
        
        # Build children recursively
        children = []
        summaries = []
        
        for chunk in chunks:
            # Summarize this chunk
            summary = self.llm.generate(
                f"Summarize in {int(len(chunk.split()) * self.summary_ratio)} words:\n{chunk}"
            )
            summaries.append(summary)
            
            # Recurse on the chunk
            child_node = self.build_tree(chunk, level=level + 1)
            children.append(child_node)
        
        # Current node = concatenated summaries
        node_content = "\n\n".join(summaries)
        
        return {
            "level": level,
            "content": node_content,
            "is_leaf": False,
            "children": children
        }
    
    def query_tree(self, tree: Dict, query: str, max_depth: int = 2) -> List[str]:
        """
        Navigate tree to find relevant content
        """
        results = []
        
        # Check relevance at current node
        relevance = self._compute_relevance(tree["content"], query)
        
        if relevance > 0.7:  # Threshold
            if tree["is_leaf"]:
                results.append(tree["content"])
            else:
                # Expand children
                for child in tree["children"]:
                    if max_depth > 0:
                        child_results = self.query_tree(child, query, max_depth - 1)
                        results.extend(child_results)
        
        return results
```

**Tree structure example:**
```
Level 0 (Root): "Book about machine learning covering supervised, 
                 unsupervised, and reinforcement learning"
├── Level 1: "Supervised learning: classification and regression"
│   ├── Level 2: "Linear regression for continuous predictions"
│   │   └── Level 3: [Detailed content about linear regression]
│   └── Level 2: "Logistic regression for binary classification"
│       └── Level 3: [Detailed content about logistic regression]
└── Level 1: "Unsupervised learning: clustering and dimensionality reduction"
    └── Level 2: "K-means clustering algorithm"
        └── Level 3: [Detailed content about k-means]
```

### Storage & Indexing

#### Option 1: Single Collection with Metadata

**Vector DB Schema:**
```python
{
    "id": "chunk_123",
    "content": "The attention mechanism computes...",
    "embedding": [0.1, 0.2, ...],
    "metadata": {
        "level": 2,  # Child
        "parent_id": "chunk_100",
        "doc_id": "paper_456",
        "chunk_type": "detail",
        "section": "Methodology"
    }
}
```

**Retrieval:**
```python
# Search at parent level
parents = vector_db.query(
    query_embedding,
    filter={"level": 1},  # Parents only
    limit=5
)

# Expand to children
all_chunks = []
for parent in parents:
    children = vector_db.query(
        filter={"parent_id": parent.id}
    )
    all_chunks.extend([parent] + children)
```

**Pros:** Simple, single index
**Cons:** Filter overhead, no optimized parent search

#### Option 2: Separate Collections

```python
# Parents collection
parents_collection = {
    "name": "parents",
    "embedding_dim": 1536,
    "chunks": [...]
}

# Children collection
children_collection = {
    "name": "children",
    "embedding_dim": 1536,
    "chunks": [...]
}

# Relationship mapping (in DB or cache)
relationships = {
    "parent_123": ["child_400", "child_401", "child_402"],
    "parent_124": ["child_403", "child_404"]
}
```

**Retrieval:**
```python
# Stage 1: Search parents
parent_results = parents_db.query(query_embedding, limit=5)

# Stage 2: Fetch children via relationships
child_ids = []
for parent in parent_results:
    child_ids.extend(relationships[parent.id])

children_results = children_db.fetch_by_ids(child_ids)

# Combine
context = parent_results + children_results
```

**Pros:** Optimized separate indices, cleaner organization
**Cons:** More complex setup, cross-collection queries

#### Option 3: Graph Database

**Best for:** Complex hierarchies with cross-references

```python
# Neo4j example
CREATE (d:Document {id: 'doc_1', title: 'Neural Nets'})
CREATE (s1:Section {id: 'sec_1', title: 'Introduction'})
CREATE (s2:Section {id: 'sec_2', title: 'Methods'})
CREATE (c1:Chunk {id: 'chunk_1', content: '...'})
CREATE (c2:Chunk {id: 'chunk_2', content: '...'})

CREATE (d)-[:HAS_SECTION]->(s1)
CREATE (d)-[:HAS_SECTION]->(s2)
CREATE (s1)-[:CONTAINS]->(c1)
CREATE (s2)-[:CONTAINS]->(c2)
CREATE (c1)-[:REFERENCES]->(c2)  # Cross-reference
```

**Query:**
```cypher
// Find chunks in same section as relevant chunk
MATCH (relevant:Chunk {id: 'chunk_1'})
MATCH (relevant)<-[:CONTAINS]-(section:Section)
MATCH (section)-[:CONTAINS]->(related:Chunk)
RETURN related
```

**Pros:** Expressive queries, handles complex relationships
**Cons:** Added complexity, need hybrid vector + graph search

### Query-Time Strategies

#### Strategy 1: Parent-First Retrieval

```python
def parent_first_retrieval(query: str, top_k: int = 5):
    """
    Retrieve parents, expand to children
    """
    # Step 1: Search parents
    query_embedding = embed(query)
    parent_candidates = vector_db.search(
        query_embedding,
        filter={"level": 1},
        limit=top_k
    )
    
    # Step 2: Expand each parent to its children
    all_chunks = []
    for parent in parent_candidates:
        # Add parent for context
        all_chunks.append({
            "type": "parent",
            "content": parent.content,
            "score": parent.score
        })
        
        # Fetch all children
        children = vector_db.fetch_by_parent_id(parent.id)
        all_chunks.extend([
            {
                "type": "child",
                "content": child.content,
                "parent_context": parent.content,
                "score": parent.score  # Inherit parent's relevance
            }
            for child in children
        ])
    
    return all_chunks

# Usage in agent
context = parent_first_retrieval(user_query, top_k=3)
# Returns: 3 parents + all their children (maybe 15-20 total chunks)
```

**Pros:** Guaranteed context, simple
**Cons:** May retrieve irrelevant children, token overflow

#### Strategy 2: Child-First with Parent Context

```python
def child_first_retrieval(query: str, top_k: int = 10):
    """
    Retrieve most relevant children, fetch their parents for context
    """
    # Step 1: Search children directly
    query_embedding = embed(query)
    child_candidates = vector_db.search(
        query_embedding,
        filter={"level": 2},  # Children only
        limit=top_k
    )
    
    # Step 2: Fetch parent for each child
    enriched_chunks = []
    seen_parents = set()
    
    for child in child_candidates:
        parent_id = child.metadata["parent_id"]
        
        # Fetch parent if not already retrieved
        if parent_id not in seen_parents:
            parent = vector_db.fetch_by_id(parent_id)
            seen_parents.add(parent_id)
            
            enriched_chunks.append({
                "type": "parent",
                "content": parent.content,
                "is_context": True
            })
        
        enriched_chunks.append({
            "type": "child",
            "content": child.content,
            "parent_id": parent_id,
            "score": child.score
        })
    
    return enriched_chunks
```

**Pros:** Only retrieves relevant children, efficient tokens
**Cons:** May miss relevant context if parent not scored well

#### Strategy 3: Hybrid with Threshold

```python
def hybrid_hierarchical_retrieval(
    query: str, 
    parent_threshold: float = 0.75,
    child_limit: int = 10
):
    """
    Intelligent expansion based on relevance
    """
    query_embedding = embed(query)
    
    # Search parents
    parent_candidates = vector_db.search(
        query_embedding,
        filter={"level": 1},
        limit=5
    )
    
    chunks = []
    
    for parent in parent_candidates:
        # Add parent
        chunks.append(parent)
        
        if parent.score > parent_threshold:
            # High confidence: expand to ALL children
            children = vector_db.fetch_by_parent_id(parent.id)
            chunks.extend(children)
        else:
            # Low confidence: only search children independently
            children = vector_db.search(
                query_embedding,
                filter={"parent_id": parent.id},
                limit=3  # Top 3 children only
            )
            chunks.extend(children)
    
    # Cap at child_limit total children
    child_chunks = [c for c in chunks if c.metadata["level"] == 2]
    if len(child_chunks) > child_limit:
        # Keep top-scoring children
        child_chunks = sorted(child_chunks, key=lambda x: x.score, reverse=True)[:child_limit]
        parent_chunks = [c for c in chunks if c.metadata["level"] == 1]
        chunks = parent_chunks + child_chunks
    
    return chunks
```

#### Strategy 4: Recursive Expansion (On-Demand)

```python
async def recursive_retrieval(
    query: str,
    max_depth: int = 3,
    relevance_threshold: float = 0.7
):
    """
    Expand hierarchy on-demand during agent execution
    """
    # Start at top level
    current_level_chunks = await search_level(query, level=1)
    
    all_chunks = []
    depth = 1
    
    while depth <= max_depth:
        for chunk in current_level_chunks:
            if chunk.score > relevance_threshold:
                all_chunks.append(chunk)
                
                # Check if there are children
                if chunk.has_children:
                    # Expand one level
                    children = await fetch_children(chunk.id)
                    
                    # Re-rank children against query
                    reranked_children = rerank(children, query)
                    
                    # Add to next level for potential expansion
                    current_level_chunks.extend(reranked_children)
        
        depth += 1
    
    return all_chunks
```

---

## Reranking

### Why Reranking Matters

**The Two-Stage Retrieval Problem:**

```
Stage 1 (First-pass retrieval): 
  Vector similarity → Fast but coarse-grained
  Retrieves 100-500 candidates

Stage 2 (Reranking):
  Cross-encoder → Slow but fine-grained
  Reranks to top 10-20
```

**Why first-pass retrieval isn't enough:**

```python
Query: "How does BERT handle long sequences?"

Bi-encoder (traditional vector search) scores:
  Chunk A: 0.82 - "BERT uses self-attention mechanisms"  ❌ Generic
  Chunk B: 0.79 - "BERT has a 512 token limit due to computational..."  ✅ Relevant!
  Chunk C: 0.80 - "BERT processes sequences through multiple layers"  ❌ Generic

Problem: Bi-encoder compares query embedding to chunk embedding separately.
         Loses interaction between query terms and document terms.

Cross-encoder (reranker):
  Chunk A: 0.65  ❌
  Chunk B: 0.95  ✅ Correctly ranks highest
  Chunk C: 0.70  ❌

Why: Cross-encoder processes [query, chunk] together, captures term interactions.
```

### Reranking Model Architectures

#### 1. Cross-Encoder Rerankers

**Architecture:**
```
Input: [CLS] query tokens [SEP] document tokens [SEP]
         ↓
    BERT/RoBERTa Encoder (12-24 layers)
         ↓
    [CLS] token representation
         ↓
    Linear layer → Relevance score
```

**Key properties:**
- **Input:** Query + document concatenated
- **Output:** Single relevance score
- **Inference:** Must run for each (query, document) pair
- **Complexity:** O(n) where n = number of documents

**Popular models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good)
- `cross-encoder/ms-marco-electra-base` (better, slower)
- `BAAI/bge-reranker-large` (state-of-the-art, slow)

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "How to implement attention mechanism?"
candidates = [
    "Attention computes weighted sum of values...",
    "Neural networks use backpropagation...",
    "The attention mechanism was introduced in 2017..."
]

# Score each pair
scores = model.predict([
    [query, candidate] for candidate in candidates
])
# Output: [0.89, 0.23, 0.67]

# Rerank
reranked = sorted(
    zip(candidates, scores), 
    key=lambda x: x[1], 
    reverse=True
)
```

#### 2. Listwise Rerankers

**Architecture:** Considers all candidates together (not pairwise)

```
Input: Query + [Doc1, Doc2, ..., DocN]
       ↓
   Transformer with cross-attention over all docs
       ↓
   Permutation probability distribution
       ↓
   Optimal ordering
```

**Models:**
- MonoT5 / duoT5 (T5-based listwise ranker)
- RankT5
- LiT5 (Listwise T5)

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('castorini/monot5-base-msmarco')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def rerank_with_monot5(query, candidates):
    scores = []
    
    for doc in candidates:
        # Format: "Query: {query} Document: {doc} Relevant:"
        input_text = f"Query: {query} Document: {doc} Relevant:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate "true" or "false"
        outputs = model.generate(input_ids, max_length=2)
        score = outputs[0][0].item()  # Logit for "true" token
        scores.append(score)
    
    return scores
```

**Pros:** Better than pairwise, considers document interactions
**Cons:** Very slow (O(n²) or worse)

#### 3. LLM-based Reranking

**Architecture:** Use LLM to judge relevance

```python
def llm_rerank(query: str, candidates: List[str], llm) -> List[float]:
    """
    Use LLM to score relevance
    """
    scores = []
    
    for doc in candidates:
        prompt = f"""
On a scale of 0-10, how relevant is this document to the query?

Query: {query}
Document: {doc}

Score (0-10):"""
        
        response = llm.generate(prompt, max_tokens=5)
        score = float(response.strip())
        scores.append(score)
    
    return scores
```

**Better approach: Batch scoring**
```python
def llm_batch_rerank(query: str, candidates: List[str], llm) -> List[float]:
    """
    Score multiple candidates in one LLM call
    """
    docs_text = "\n\n".join([
        f"[Doc {i+1}]: {doc}" 
        for i, doc in enumerate(candidates)
    ])
    
    prompt = f"""
Query: {query}

Documents:
{docs_text}

Rank the documents by relevance. Output format:
[Doc ID]: [Score 0-10]
"""
    
    response = llm.generate(prompt)
    return parse_scores(response)
```

**Pros:** Very powerful, handles nuance
**Cons:** Expensive, slow, non-deterministic

#### 4. Cohere Rerank API

**Commercial solution with excellent performance:**

```python
import cohere

co = cohere.Client('your-api-key')

results = co.rerank(
    query="How does photosynthesis work?",
    documents=[
        "Photosynthesis converts light energy into chemical energy...",
        "Plants use chlorophyll to capture sunlight...",
        "The Calvin cycle is the second stage of photosynthesis..."
    ],
    top_n=3,
    model='rerank-english-v2.0'
)

for hit in results:
    print(f"Score: {hit.relevance_score:.2f} - {hit.document['text']}")
```

**Pros:** State-of-the-art, easy API, fast
**Cons:** Costs money, external dependency

### Implementation Patterns

#### Pattern 1: Simple Two-Stage Pipeline

```python
class TwoStageRetrieval:
    def __init__(self, vector_db, reranker):
        self.vector_db = vector_db
        self.reranker = reranker
    
    def retrieve(self, query: str, top_k: int = 10):
        # Stage 1: Vector search (broad recall)
        candidates = self.vector_db.search(
            query, 
            limit=100  # Over-retrieve
        )
        
        # Stage 2: Rerank (precision)
        reranked = self.reranker.rerank(
            query=query,
            documents=[c.content for c in candidates],
            top_n=top_k
        )
        
        return reranked

# Usage
retriever = TwoStageRetrieval(
    vector_db=PineconeDB(),
    reranker=CohereRerank()
)

results = retriever.retrieve("What causes inflation?", top_k=5)
```

#### Pattern 2: Hybrid Retrieval + Reranking

```python
class HybridRetrievalWithReranking:
    def __init__(self, vector_db, bm25_index, reranker):
        self.vector_db = vector_db
        self.bm25 = bm25_index
        self.reranker = reranker
    
    def retrieve(self, query: str, top_k: int = 10):
        # Stage 1a: Dense retrieval
        dense_results = self.vector_db.search(query, limit=50)
        
        # Stage 1b: Sparse retrieval
        sparse_results = self.bm25.search(query, limit=50)
        
        # Combine (union)
        all_candidates = self._merge_results(dense_results, sparse_results)
        
        # Stage 2: Rerank combined candidates
        reranked = self.reranker.rerank(
            query=query,
            documents=[c.content for c in all_candidates],
            top_n=top_k
        )
        
        return reranked
    
    def _merge_results(self, dense, sparse):
        """
        Reciprocal Rank Fusion
        """
        doc_scores = {}
        
        for rank, doc in enumerate(dense):
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + 1 / (rank + 60)
        
        for rank, doc in enumerate(sparse):
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + 1 / (rank + 60)
        
        # Return top 100 by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.get_doc_by_id(doc_id) for doc_id, _ in sorted_docs[:100]]
```

#### Pattern 3: Contextual Reranking

**Include surrounding context when reranking:**

```python
class ContextualReranker:
    def __init__(self, reranker):
        self.reranker = reranker
    
    def rerank_with_context(
        self, 
        query: str, 
        chunks: List[Dict],
        include_neighbors: bool = True
    ):
        """
        Rerank chunks with surrounding context
        """
        enriched_chunks = []
        
        for chunk in chunks:
            content = chunk["content"]
            
            if include_neighbors and chunk.get("neighbors"):
                # Add prev/next chunks for context
                prev_chunk = chunk["neighbors"]["prev"]
                next_chunk = chunk["neighbors"]["next"]
                
                content = f"""
[Previous context]: {prev_chunk}

[Main content]: {chunk["content"]}

[Following context]: {next_chunk}
"""
            
            enriched_chunks.append({
                "original_chunk": chunk,
                "enriched_content": content
            })
        
        # Rerank with enriched content
        scores = self.reranker.predict([
            [query, ec["enriched_content"]] 
            for ec in enriched_chunks
        ])
        
        # Return original chunks with new scores
        reranked = sorted(
            zip([ec["original_chunk"] for ec in enriched_chunks], scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [chunk for chunk, _ in reranked]
```

#### Pattern 4: Multi-Stage Reranking

**Use cheap reranker first, expensive reranker second:**

```python
class MultiStageReranker:
    def __init__(self):
        # Stage 1: Fast, lightweight
        self.fast_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Stage 2: Slow, powerful
        self.slow_reranker = CrossEncoder('BAAI/bge-reranker-large')
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 10):
        # Stage 1: Fast reranking (100 → 30)
        fast_scores = self.fast_reranker.predict([
            [query, doc] for doc in candidates
        ])
        
        # Keep top 30
        top_30 = sorted(
            zip(candidates, fast_scores),
            key=lambda x: x[1],
            reverse=True
        )[:30]
        
        # Stage 2: Slow reranking (30 → 10)
        top_30_docs = [doc for doc, _ in top_30]
        slow_scores = self.slow_reranker.predict([
            [query, doc] for doc in top_30_docs
        ])
        
        # Final ranking
        final = sorted(
            zip(top_30_docs, slow_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [doc for doc, _ in final]
```

**Latency comparison:**
```
No reranking:              50ms   (vector search only)
Single-stage (fast):      150ms   (vector + fast reranker on 100)
Single-stage (slow):      800ms   (vector + slow reranker on 100)
Multi-stage:              250ms   (vector + fast on 100 + slow on 30)
                                  ↑ 3x faster than single slow, better accuracy
```

### Optimization Techniques

#### 1. Batch Processing

```python
def batch_rerank(reranker, query, documents, batch_size=32):
    """
    Process reranking in batches for efficiency
    """
    all_scores = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        pairs = [[query, doc] for doc in batch]
        
        batch_scores = reranker.predict(pairs)
        all_scores.extend(batch_scores)
    
    return all_scores
```

#### 2. Caching

```python
from functools import lru_cache
import hashlib

class CachedReranker:
    def __init__(self, reranker):
        self.reranker = reranker
        self.cache = {}  # Or use Redis for distributed cache
    
    def rerank(self, query: str, documents: List[str]):
        # Create cache key
        doc_hashes = [hashlib.md5(d.encode()).hexdigest() for d in documents]
        cache_key = (query, tuple(doc_hashes))
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compute scores
        scores = self.reranker.predict([
            [query, doc] for doc in documents
        ])
        
        # Cache results
        self.cache[cache_key] = scores
        return scores
```

#### 3. Early Stopping

```python
def adaptive_reranking(
    query: str, 
    candidates: List[str],
    reranker,
    confidence_threshold: float = 0.9
):
    """
    Stop reranking if confident about top results
    """
    scores = []
    
    for i, doc in enumerate(candidates):
        score = reranker.predict([[query, doc]])[0]
        scores.append((doc, score))
        
        # If we have 10 results and top result is very confident, stop
        if i >= 10 and max(s for _, s in scores) > confidence_threshold:
            break
    
    return sorted(scores, key=lambda x: x[1], reverse=True)
```

#### 4. Quantization

```python
from sentence_transformers import CrossEncoder
import torch

# Load model with quantization
model = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device='cuda'
)

# Convert to int8
model.model = torch.quantization.quantize_dynamic(
    model.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# ~4x faster inference with minimal accuracy loss
```

---

## Combining Hierarchical Chunking + Reranking

### Strategy 1: Rerank at Each Level

```python
class HierarchicalRerankedRetrieval:
    def __init__(self, vector_db, reranker):
        self.vector_db = vector_db
        self.reranker = reranker
    
    def retrieve(self, query: str, top_k: int = 10):
        # Stage 1: Retrieve parent candidates
        parent_candidates = self.vector_db.search(
            query,
            filter={"level": 1},
            limit=20
        )
        
        # Stage 2: Rerank parents
        parent_scores = self.reranker.predict([
            [query, p.content] for p in parent_candidates
        ])
        
        top_parents = sorted(
            zip(parent_candidates, parent_scores),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Keep top 5 parents
        
        # Stage 3: Fetch children for top parents
        all_children = []
        for parent, parent_score in top_parents:
            children = self.vector_db.fetch_children(parent.id)
            all_children.extend([
                (child, parent_score, parent) 
                for child in children
            ])
        
        # Stage 4: Rerank children
        child_scores = self.reranker.predict([
            [query, child.content] 
            for child, _, _ in all_children
        ])
        
        # Combine parent and child scores
        final_scores = [
            parent_score * 0.3 + child_score * 0.7
            for (_, parent_score, _), child_score 
            in zip(all_children, child_scores)
        ]
        
        # Stage 5: Final ranking
        reranked = sorted(
            zip([c for c, _, _ in all_children], final_scores, [p for _, _, p in all_children]),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [{
            "content": child.content,
            "score": score,
            "parent_context": parent.content
        } for child, score, parent in reranked]
```

### Strategy 2: Rerank Final Candidates Only

```python
def hierarchical_then_rerank(query: str, top_k: int = 10):
    """
    1. Hierarchical retrieval
    2. Single reranking pass on final candidates
    """
    # Step 1: Hierarchical retrieval (parent-first)
    parent_results = vector_db.search(
        query, 
        filter={"level": 1},
        limit=5
    )
    
    candidates = []
    for parent in parent_results:
        # Add parent
        candidates.append({
            "content": parent.content,
            "type": "parent",
            "id": parent.id
        })
        
        # Add children
        children = vector_db.fetch_children(parent.id)
        candidates.extend([
            {
                "content": f"{parent.content}\n\n{child.content}",  # Concatenate for context
                "type": "child",
                "id": child.id,
                "parent_id": parent.id
            }
            for child in children
        ])
    
    # Step 2: Rerank all candidates
    scores = reranker.predict([
        [query, c["content"]] for c in candidates
    ])
    
    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    return [c for c, _ in reranked]
```

### Strategy 3: Context-Aware Reranking with Hierarchy

```python
def context_aware_hierarchical_reranking(query: str):
    """
    Use parent summaries to help rerank children
    """
    # Retrieve parents
    parents = vector_db.search(query, filter={"level": 1}, limit=5)
    
    all_children = []
    for parent in parents:
        children = vector_db.fetch_children(parent.id)
        
        # For each child, create context-enriched version
        for child in children:
            enriched = f"""
Section Summary: {parent.content}

Detailed Content: {child.content}
"""
            all_children.append({
                "original_child": child,
                "enriched_content": enriched,
                "parent": parent
            })
    
    # Rerank with context
    scores = reranker.predict([
        [query, c["enriched_content"]] 
        for c in all_children
    ])
    
    reranked = sorted(
        zip(all_children, scores),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return [
        {
            "content": item["original_child"].content,
            "parent_context": item["parent"].content,
            "score": score
        }
        for item, score in reranked
    ]
```

---

## Production Implementation

### Complete Pipeline

```python
from typing import List, Dict, Optional
import numpy as np

class ProductionHierarchicalRerankedRAG:
    def __init__(
        self,
        vector_db,
        chunker,
        reranker,
        config: Optional[Dict] = None
    ):
        self.vector_db = vector_db
        self.chunker = chunker
        self.reranker = reranker
        
        # Configuration
        self.config = config or {
            "parent_limit": 10,
            "child_expansion_threshold": 0.7,
            "max_children_per_parent": 5,
            "final_top_k": 10,
            "rerank_batch_size": 32,
            "use_multi_stage_reranking": True
        }
        
        # Caches
        self.query_cache = LRUCache(maxsize=1000)
        self.parent_child_cache = {}
    
    def index_document(self, document: str, doc_id: str):
        """
        Index document with hierarchical chunking
        """
        # Create hierarchy
        hierarchy = self.chunker.chunk_with_summaries(document)
        
        # Index parents
        for parent in hierarchy["parents"]:
            self.vector_db.upsert(
                id=parent["id"],
                embedding=parent["embedding"],
                metadata={
                    **parent["metadata"],
                    "doc_id": doc_id
                }
            )
        
        # Index children
        for child in hierarchy["children"]:
            self.vector_db.upsert(
                id=child["id"],
                embedding=child["embedding"],
                metadata={
                    **child["metadata"],
                    "doc_id": doc_id
                }
            )
        
        # Store relationships
        for rel in hierarchy["relationships"]:
            self.parent_child_cache[rel["parent_id"]] = rel["child_ids"]
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Main retrieval pipeline
        """
        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Step 1: Retrieve parent candidates
        parent_candidates = self._retrieve_parents(query)
        
        # Step 2: Rerank parents (if multi-stage)
        if self.config["use_multi_stage_reranking"]:
            parent_candidates = self._fast_rerank(query, parent_candidates)
        
        # Step 3: Expand to children
        all_candidates = self._expand_to_children(query, parent_candidates)
        
        # Step 4: Final reranking
        final_results = self._final_rerank(query, all_candidates)
        
        # Cache results
        self.query_cache[cache_key] = final_results
        
        return final_results
    
    def _retrieve_parents(self, query: str) -> List[Dict]:
        """
        Retrieve parent chunks
        """
        results = self.vector_db.search(
            query,
            filter={"level": 1},
            limit=self.config["parent_limit"]
        )
        
        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "type": "parent"
            }
            for r in results
        ]
    
    def _fast_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Fast first-stage reranking
        """
        # Use lightweight reranker
        scores = self.reranker.fast_rerank(
            query=query,
            documents=[c["content"] for c in candidates]
        )
        
        # Update scores
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = score
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return candidates
    
    def _expand_to_children(
        self, 
        query: str, 
        parent_candidates: List[Dict]
    ) -> List[Dict]:
        """
        Expand parents to children based on threshold
        """
        all_candidates = []
        
        for parent in parent_candidates:
            # Always include parent for context
            all_candidates.append(parent)
            
            # Decide whether to expand
            should_expand = (
                parent.get("rerank_score", parent["score"]) 
                > self.config["child_expansion_threshold"]
            )
            
            if should_expand:
                # Fetch all children
                child_ids = self.parent_child_cache.get(parent["id"], [])
                children = self.vector_db.fetch_by_ids(child_ids)
                
                # Limit children per parent
                max_children = self.config["max_children_per_parent"]
                children = children[:max_children]
                
                for child in children:
                    all_candidates.append({
                        "id": child.id,
                        "content": child.content,
                        "score": parent["score"],  # Inherit parent score
                        "metadata": child.metadata,
                        "type": "child",
                        "parent_id": parent["id"],
                        "parent_context": parent["content"]
                    })
        
        return all_candidates
    
    def _final_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Final reranking with best model
        """
        # Prepare content for reranking
        rerank_inputs = []
        for c in candidates:
            if c["type"] == "child":
                # Include parent context
                content = f"{c['parent_context']}\n\n{c['content']}"
            else:
                content = c["content"]
            rerank_inputs.append(content)
        
        # Batch reranking
        final_scores = self._batch_rerank(query, rerank_inputs)
        
        # Update scores and sort
        for candidate, score in zip(candidates, final_scores):
            candidate["final_score"] = score
        
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return top-k
        return candidates[:self.config["final_top_k"]]
    
    def _batch_rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Batch reranking for efficiency
        """
        all_scores = []
        batch_size = self.config["rerank_batch_size"]
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            pairs = [[query, doc] for doc in batch]
            
            batch_scores = self.reranker.predict(pairs)
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()


# Usage
rag = ProductionHierarchicalRerankedRAG(
    vector_db=PineconeDB(),
    chunker=HierarchicalChunker(llm=GPT4(), embedding_model=OpenAIEmbeddings()),
    reranker=MultiStageReranker(),
    config={
        "parent_limit": 10,
        "child_expansion_threshold": 0.7,
        "max_children_per_parent": 5,
        "final_top_k": 10,
        "rerank_batch_size": 32,
        "use_multi_stage_reranking": True
    }
)

# Index documents
rag.index_document(research_paper, doc_id="paper_001")

# Retrieve
results = rag.retrieve("How does the model handle long sequences?")

for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"Type: {result['type']}")
    print(f"Content: {result['content'][:200]}...")
    if result['type'] == 'child':
        print(f"Parent context: {result['parent_context'][:100]}...")
    print()
```

### Monitoring & Evaluation

```python
class RAGMonitor:
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.metrics = []
    
    def evaluate_query(self, query: str, ground_truth_docs: List[str]):
        """
        Evaluate retrieval quality for a single query
        """
        # Get results
        results = self.pipeline.retrieve(query)
        
        # Compute metrics
        retrieved_ids = [r["id"] for r in results]
        
        metrics = {
            "query": query,
            "precision": self._precision(retrieved_ids, ground_truth_docs),
            "recall": self._recall(retrieved_ids, ground_truth_docs),
            "mrr": self._mrr(retrieved_ids, ground_truth_docs),
            "latency_ms": results[-1]["latency_ms"] if "latency_ms" in results[-1] else None,
            "num_parents_retrieved": sum(1 for r in results if r["type"] == "parent"),
            "num_children_retrieved": sum(1 for r in results if r["type"] == "child")
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def _precision(self, retrieved, relevant):
        if not retrieved:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(retrieved)
    
    def _recall(self, retrieved, relevant):
        if not relevant:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(relevant)
    
    def _mrr(self, retrieved, relevant):
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def aggregate_metrics(self):
        """
        Compute aggregate statistics
        """
        return {
            "avg_precision": np.mean([m["precision"] for m in self.metrics]),
            "avg_recall": np.mean([m["recall"] for m in self.metrics]),
            "avg_mrr": np.mean([m["mrr"] for m in self.metrics]),
            "avg_latency_ms": np.mean([m["latency_ms"] for m in self.metrics if m["latency_ms"]]),
            "total_queries": len(self.metrics)
        }
```

---

## Key Takeaways

### Hierarchical Chunking
✅ **Use when:** Long documents, need context preservation, multi-granularity retrieval
✅ **Best pattern:** Parent summaries + detailed children
✅ **Storage:** Separate collections or metadata-based filtering
✅ **Retrieval:** Parent-first with conditional expansion

### Reranking
✅ **Use when:** Need high precision, willing to trade latency
✅ **Best model:** Cross-encoders (good balance), Cohere Rerank (best commercial)
✅ **Optimization:** Multi-stage (fast → slow), batching, caching
✅ **Integration:** After initial retrieval, before LLM generation

### Combined Approach
✅ **Pattern:** Hierarchical retrieval → Rerank at child level
✅ **Context:** Include parent summaries during reranking
✅ **Threshold:** Expand to children only for high-scoring parents
✅ **Monitoring:** Track precision, recall, latency, token usage

This gives you production-grade retrieval with both semantic understanding (hierarchical) and precision (reranking).
