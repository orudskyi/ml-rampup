import os

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 1. Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is missing in the .env file.")

# 2. Configuration
COLLECTION_NAME = "test_collection"
# Dimension for 'text-embedding-004' is 768
VECTOR_SIZE = 768 
EMBEDDING_MODEL = "text-embedding-004"
QDRANT_URL = "http://localhost:6333"

# 3. Initialize clients
gemini_client = genai.Client(api_key=API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL)

def run_pipeline():
    print("--- Starting Vector Pipeline ---")

    # 4. Create Collection in Qdrant
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("Collection created successfully.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")

    # 5. Define input data
    sentences = [
        "Python is great",
        "I love coding",
        "The sky is blue"
    ]

    print(f"\n--- Generating Embeddings for {len(sentences)} sentences ---")
    points = []

    # 6. Generate embeddings and prepare points
    for idx, text in enumerate(sentences):
        try:
            # Call Gemini API to get the embedding
            response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            # Extract the vector (list of floats)
            embedding = response.embeddings[0].values

            # Create a point for Qdrant
            # We store the original text in the payload to retrieve it later
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={"text": text} 
            )
            points.append(point)
            print(f"Processed: '{text}'")
            
        except Exception as e:
            print(f"Error embedding text '{text}': {e}")

    # 7. Upsert (Upload) vectors to Qdrant
    if points:
        print("\n--- Uploading vectors to Qdrant ---")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print("Upsert completed.")

    # 8. Test Search
    search_query = "programming"
    print(f"\n--- Testing Search for query: '{search_query}' ---")

    # Embed the query
    query_response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=search_query,
    )
    query_vector = query_response.embeddings[0].values

    # Perform the search using 'query_points' instead of 'search'
    # This is the explicit method compliant with recent API changes
    results_object = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector, # Note: argument is 'query', not 'query_vector' here
        limit=2
    )
    
    # query_points returns an object wrapping the points
    search_results = results_object.points

    # Display results
    print("Search Results:")
    for result in search_results:
        # result.score represents the cosine similarity
        print(f"- Found: '{result.payload['text']}' (Score: {result.score:.4f})")

if __name__ == "__main__":
    run_pipeline()