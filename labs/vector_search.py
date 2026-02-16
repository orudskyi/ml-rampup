import time
import numpy as np


def run_vector_search() -> None:
    # 1. Initialize data
    np.random.seed(42)
    print("Generating data...")

    embeddings = np.random.randn(10000, 1536).astype(np.float32)
    query = np.random.randn(1536).astype(np.float32)

    start_time = time.time()

    # 2. Mathematical operations
    # Formula: dot(A, B) / (norm(A) * norm(B))
    dot_product = np.dot(embeddings, query)

    norm_embeddings = np.linalg.norm(embeddings, axis=1) + 1e-10
    norm_query = np.linalg.norm(query) + 1e-10

    cosine_similarities = dot_product / (norm_embeddings * norm_query)

    # Get top 5 indices
    k = 5

    unsorted_top_k = np.argpartition(cosine_similarities, -k)[-k:]
    top_indices = unsorted_top_k[np.argsort(cosine_similarities[unsorted_top_k])][::-1]

    duration = time.time() - start_time

    # 3. Output results:
    print(f"Search completed in {duration:.4f} seconds")
    print(f"Top 5 indices: {top_indices}")


if __name__ == "__main__":
    run_vector_search()
