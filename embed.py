"""
embed.py — Embedding and Vector Store Pipeline
The Unofficial Guide: UC Berkeley CS Professor Reviews

What this script does:
1. Loads and chunks all documents (via ingest.py + chunk.py)
2. Embeds all chunks using all-MiniLM-L6-v2 (local, no API key)
3. Stores chunks + embeddings in ChromaDB collection: reviews_recursive
4. Reports how many chunks were stored and verifies the collection

Run this ONCE to build the vector store.
After running, the database persists in ./chroma_db/
You do NOT need to re-run this unless you change your documents or chunking.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from chunk import chunk_documents
from ingest import load_and_clean_documents


DOCUMENTS_DIR = "documents"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "reviews_recursive"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the sentence-transformers embedding model locally."""
    print(f"\n{'='*50}")
    print(f"STEP 1: Loading embedding model")
    print(f"{'='*50}")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  (Runs locally — no API key needed)")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✅ Model loaded successfully")
    return model


def setup_chromadb():
    """
    Set up a persistent ChromaDB client and return the collection.
    If the collection already exists, it will be replaced fresh.
    """
    print(f"\n{'='*50}")
    print(f"STEP 2: Setting up ChromaDB")
    print(f"{'='*50}")
    print(f"  Location: {CHROMA_DB_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (clean rebuild)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection (rebuilding fresh)")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    print(f"  ✅ Collection created: {COLLECTION_NAME}")
    return collection


def embed_and_store(chunks: list, model: SentenceTransformer, collection):
    """
    Embed all chunks and store them in ChromaDB with metadata.

    Each chunk is stored with:
    - id: unique string identifier
    - embedding: 384-dim vector from all-MiniLM-L6-v2
    - document: the chunk text
    - metadata: source, professor, course, chunk_index
    """
    print(f"\n{'='*50}")
    print(f"STEP 3: Embedding and storing {len(chunks)} chunks")
    print(f"{'='*50}")
    print(f"  This may take 30–60 seconds...")

    # Extract texts and metadata
    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{chunk['source']}__chunk{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "professor": chunk["professor"],
            "course": chunk["course"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    # Generate embeddings in one batch (faster than one at a time)
    print(f"  Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings_list = embeddings.tolist()

    # Store in ChromaDB in batches of 100 (avoids memory issues on large sets)
    batch_size = 100
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings_list[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_metadatas,
        )
        total_stored += len(batch_texts)

    print(f"  ✅ Stored {total_stored} chunks in ChromaDB")
    return total_stored


def verify_collection(collection):
    """Quick verification that the collection was built correctly."""
    print(f"\n{'='*50}")
    print(f"STEP 4: Verifying collection")
    print(f"{'='*50}")

    count = collection.count()
    print(f"  Total chunks in collection: {count}")

    # Peek at first 3 stored items
    sample = collection.peek(limit=3)
    print(f"\n  Sample stored chunks:")
    for i in range(len(sample["ids"])):
        print(f"\n  [{i+1}] ID: {sample['ids'][i]}")
        print(f"       Source: {sample['metadatas'][i]['source']}")
        print(f"       Professor: {sample['metadatas'][i]['professor']}")
        print(f"       Text preview: {sample['documents'][i][:100]}...")

    print(f"\n  ✅ Collection verified — ready for retrieval")


if __name__ == "__main__":
    # Step 1: Load and chunk documents
    cleaned_docs = load_and_clean_documents(DOCUMENTS_DIR)
    if not cleaned_docs:
        print("❌ No documents loaded. Check your documents/ folder.")
        exit(1)

    chunks = chunk_documents(cleaned_docs)
    print(f"\n  Total chunks to embed: {len(chunks)}")

    # Step 2: Load embedding model
    model = load_embedding_model()

    # Step 3: Set up ChromaDB
    collection = setup_chromadb()

    # Step 4: Embed and store
    embed_and_store(chunks, model, collection)

    # Step 5: Verify
    verify_collection(collection)

    print(f"\n{'='*50}")
    print(f"✅ Milestone 4 embedding complete!")
    print(f"   Database saved to: {CHROMA_DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"{'='*50}")