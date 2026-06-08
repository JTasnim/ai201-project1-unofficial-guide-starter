"""
retrieve.py — Retrieval Function and Test Runner
The Unofficial Guide: UC Berkeley CS Professor Reviews

What this script does:
1. Connects to the existing ChromaDB collection (built by embed.py)
2. Provides a retrieve(query, k=5) function for semantic search
3. When run directly, tests retrieval with 3 of your 5 evaluation queries
4. Prints retrieved chunks and distance scores for inspection

Run embed.py first to build the vector store, then run this to test retrieval.
"""

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "reviews_recursive"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load model and collection once at module level
# (so they're reused when retrieve() is called from other scripts)
print("Loading embedding model and connecting to ChromaDB...")
_model = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection = _client.get_collection(name=COLLECTION_NAME)
print(f"Connected to collection '{COLLECTION_NAME}' ({_collection.count()} chunks)\n")


def retrieve(query: str, k: int = 5) -> list:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        query: plain-language question string
        k: number of chunks to return (default 5)

    Returns:
        list of dicts, each with:
        {
            "text": str,        # chunk text
            "source": str,      # filename (e.g. rmp_hug.txt)
            "professor": str,   # professor name
            "course": str,      # course code
            "distance": float,  # cosine distance (lower = more similar)
        }
    """
    # Embed the query using the same model as the chunks
    query_embedding = _model.encode(query).tolist()

    # Query ChromaDB for top-k nearest neighbors
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack results into a clean list
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "professor": results["metadatas"][0][i]["professor"],
            "course": results["metadatas"][0][i]["course"],
            "distance": round(results["distances"][0][i], 4),
        })

    return chunks


def print_retrieval_results(query: str, chunks: list):
    """Pretty-print retrieval results for inspection."""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    for i, chunk in enumerate(chunks, 1):
        distance_flag = ""
        if chunk["distance"] > 0.7:
            distance_flag = "  ⚠️  HIGH — weak match"
        elif chunk["distance"] < 0.4:
            distance_flag = "  ✅ LOW — strong match"

        print(f"\n  Result {i}")
        print(f"  Source:    {chunk['source']}")
        print(f"  Professor: {chunk['professor']}")
        print(f"  Course:    {chunk['course']}")
        print(f"  Distance:  {chunk['distance']}{distance_flag}")
        print(f"  Text:")
        # Wrap text at 60 chars for readability
        text = chunk["text"]
        words = text.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 64:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    # Retrieval quality assessment
    distances = [c["distance"] for c in chunks]
    avg_distance = sum(distances) / len(distances)
    best_distance = min(distances)

    print(f"\n  --- Retrieval Assessment ---")
    print(f"  Best distance score: {best_distance}")
    print(f"  Avg distance score:  {round(avg_distance, 4)}")

    if best_distance < 0.4:
        print(f"  Verdict: ✅ Strong retrieval — top result is highly relevant")
    elif best_distance < 0.6:
        print(f"  Verdict: 🟡 Moderate retrieval — results are related but not precise")
    else:
        print(f"  Verdict: ⚠️  Weak retrieval — chunks may be too small or query too vague")


# ── Test queries (3 of your 5 evaluation plan questions) ──────────────────────

TEST_QUERIES = [
    "What do students say about exam difficulty in CS189?",
    "How do students describe the workload for CS61B?",
    "What do students say about office hours availability for CS70 professors?",
]


if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"RETRIEVAL TEST — 3 Evaluation Plan Queries")
    print(f"{'='*60}")
    print(f"For each query, ask yourself:")
    print(f"  1. Are the returned chunks actually relevant to the question?")
    print(f"  2. Are they from the right professor / course?")
    print(f"  3. Are distance scores below 0.6?")

    for query in TEST_QUERIES:
        chunks = retrieve(query, k=5)
        print_retrieval_results(query, chunks)

    print(f"\n{'='*60}")
    print(f"✅ Retrieval test complete.")
    print(f"   If results look relevant and distances are below 0.6,")
    print(f"   you are ready for Milestone 5 (generation + Gradio UI).")
    print(f"{'='*60}")