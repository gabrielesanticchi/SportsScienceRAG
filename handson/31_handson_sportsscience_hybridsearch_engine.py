# %%
import os
import sys
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Make the sibling ``assets/`` package importable, then load the
# sports-science PDF snippets (one payload-ready dict per paper).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
)
from sports_science_documents import documents  # noqa: E402

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "day3c_hybrid_search"

# Create hybrid collection
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )
    }
)

# Implementing Dense and Sparse Encoding 
# Dense embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Global vocabulary - automatically extends as new texts are processed
global_vocabulary = {}

# Simple sparse encoding (BM25-style)
def create_sparse_vector(text):
    """Create sparse vector from text using term frequency"""
    from collections import Counter
    import re
    
    # Simple tokenization
    words = re.findall(r"\b\w+\b", text.lower())
    word_counts = Counter(words)
        
    
    # Convert to sparse vector format, extending vocabulary as needed
    indices = []
    values = []
    
    for word, count in word_counts.items():
        if word not in global_vocabulary:
            # Add new word to vocabulary with next available index
            global_vocabulary[word] = len(global_vocabulary)
        
        indices.append(global_vocabulary[word])
        values.append(float(count))
        
        # So indices=[4, 6] with values=[2.0, 1.0] means:
        # vocabulary word #4 appeared 2 times (See Counter class)
        # vocabulary word #6 appeared 1 time (See Counter class)

    return models.SparseVector(indices=indices, values=values)

# Upload hybrid data
points = []
for i, item in enumerate(documents):
    dense_vector = encoder.encode(item["text"]).tolist()
    sparse_vector = create_sparse_vector(item["text"])
    
    points.append(models.PointStruct(
            id=i,
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
            },
            payload=item,
        )
    )

client.upload_points(collection_name=collection_name, points=points)
# %%
# Step 3: Implement Hybrid Search with RRF
def hybrid_search_with_rrf(query_text, limit=10):
    """Perform hybrid search using Reciprocal Rank Fusion"""
    
    # Encode query for both dense and sparse
    query_dense = encoder.encode(query_text).tolist()
    query_sparse = create_sparse_vector(query_text)
    
    # Use Qdrant's built-in RRF
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=20
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=20
            )
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )
    
    return response.points

# Test hybrid search
results = hybrid_search_with_rrf("training load and injury risk in soccer players")
for i, point in enumerate(results, 1):
    print(f"{i}. {point.payload.get('title', 'No title')} (Score: {point.score:.3f})")

# Step 4: Compare Search Approaches
def compare_search_methods(query_text):
    """Compare dense, sparse, and hybrid search results"""
    
    print(f"Query: '{query_text}'\n")
    
    # Dense-only search
    dense_results = client.query_points(
        collection_name=collection_name,
        query=encoder.encode(query_text).tolist(),
        using="dense",
        limit=5
    )
    
    # Sparse-only search  
    sparse_results = client.query_points(
        collection_name=collection_name,
        query=create_sparse_vector(query_text),
        using="sparse",
        limit=5
    )
    
    # Hybrid search
    hybrid_results = hybrid_search_with_rrf(query_text, limit=5)
    
    print("DENSE SEARCH:")
    for i, point in enumerate(dense_results.points, 1):
        print(f"  {i}. {point.payload.get('title', 'No title')} ({point.score:.3f})")
    
    print("\nSPARSE SEARCH:")
    for i, point in enumerate(sparse_results.points, 1):
        print(f"  {i}. {point.payload.get('title', 'No title')} ({point.score:.3f})")
    
    print("\nHYBRID SEARCH (RRF):")
    for i, point in enumerate(hybrid_results, 1):
        print(f"  {i}. {point.payload.get('title', 'No title')} ({point.score:.3f})")
    
    print("-" * 50)

# Test with different query types
test_queries = [
    "training load and injury risk in soccer players",
    "match demand in soccer players",
    "RPE in football"
]

for query in test_queries:
    compare_search_methods(query)
# %%
