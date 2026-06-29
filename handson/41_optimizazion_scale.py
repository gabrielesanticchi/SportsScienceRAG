# %%
import os
import sys
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Make the sibling ``assets/`` package importable, then load the
# sports-science PDF snippets (one payload-ready dict per paper).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"))
from sports_science_documents import documents  # noqa: E402
from sports_science_queries import queries  # noqa: E402

client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))


# %%
# Measure baseline performance
def measure_search_performance(collection_name, test_queries, label="Baseline"):
    """Measure search performance across multiple queries"""
    latencies = []

    # Don't forget to warm up caches!

    #response = client.query_points(
    #        collection_name=collection_name,
    #        query=query,
    #        limit=10
    #    )
    
    for query in test_queries:
        start_time = time.time()
        
        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=10
        )
        
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"{label}:")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  P95 latency: {p95_latency:.2f}ms")
    print(f"  Memory usage: Check Qdrant Cloud dashboard")
    
    return {"avg": avg_latency, "p95": p95_latency}
# %%
# Test configurations
quantization_configs = {
    "scalar": {
        "config": models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
        "expected_speedup": "2x",
        "expected_compression": "4x"
    },
    "binary": {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=models.BinaryQuantizationEncoding.ONE_BIT,
                always_ram=True,
            )
        ),
        "expected_speedup": "40x",
        "expected_compression": "32x"
    },
    "binary_2bit": {
        "config": models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                encoding=models.BinaryQuantizationEncoding.TWO_BITS,
                always_ram=True,
            )
        ),
        "expected_speedup": "20x",
        "expected_compression": "16x"
    }
}

# Create quantized collections
for method_name, config_info in quantization_configs.items():
    collection_name = f"day41_optimization_{method_name}"
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  # Adjust to your embedding size
            distance=models.Distance.COSINE,
            on_disk=True,  # Store originals on disk
        ),
        quantization_config=config_info["config"]
    )
    
    print(f"Created {method_name} quantized collection: {collection_name}")

# %%
def benchmark(collection_name, your_test_queries, method_name):
    """Measure quantized search performance"""
    
    # Test without oversampling/rescoring first
    no_rescoring_metrics = measure_search_performance(
        collection_name, 
        queries, 
        f"{method_name} (No Rescoring)"
    )
    
    # Test with oversampling and rescoring
    def search_with_rescoring(collection_name, query, oversampling_factor=3.0):
        start_time = time.time()
        
        response = client.query_points(
            collection_name=collection_name,
            query=query,
            limit=10,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=oversampling_factor,
                )
            ),
        )
        
        return (time.time() - start_time) * 1000, response
    
    # Measure with rescoring
    rescoring_latencies = []
    for query in your_test_queries:
        latency, response = search_with_rescoring(collection_name, query)
        rescoring_latencies.append(latency)
    
    avg_rescoring = np.mean(rescoring_latencies)
    p95_rescoring = np.percentile(rescoring_latencies, 95)
    
    print(f"{method_name} (With Rescoring):")
    print(f"  Average latency: {avg_rescoring:.2f}ms")
    print(f"  P95 latency: {p95_rescoring:.2f}ms")
    
    return {
        "no_rescoring": no_rescoring_metrics,
        "with_rescoring": {"avg": avg_rescoring, "p95": p95_rescoring}
    }


# Upload your data (same as it was done in the previous days for a basic unquantized collection) in each collection

# Test each quantization method
quantization_results = {}
for method_name in quantization_configs.keys():
    collection_name = f"day41_optimization_{method_name}"
    quantization_results[method_name] = benchmark(
        collection_name, queries, method_name
    )