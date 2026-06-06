"""
chunk.py — Recursive Chunking Pipeline
The Unofficial Guide: UC Berkeley CS Professor Reviews

Strategy: Recursive Chunking
- Target chunk size: 300 characters
- Overlap: 30 characters
- Split hierarchy: \n\n → \n → sentence boundary → word boundary
- Tool: LangChain RecursiveCharacterTextSplitter

What this script does:
1. Calls ingest.py to load and clean all documents
2. Splits each document into chunks using recursive strategy
3. Attaches metadata to every chunk (source, professor, chunk_index)
4. Prints 5 sample chunks for manual inspection
5. Reports total chunk count
"""

import re
from ingest import load_and_clean_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = "documents"

# Chunking config — matches planning.md exactly
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


def extract_professor_name(filename: str) -> str:
    """
    Extract a readable professor name from the filename.
    e.g. 'rmp_hug.txt' → 'hug'
         'rmp_hilfinger.txt' → 'hilfinger'
    """
    name = filename.replace("rmp_", "").replace(".txt", "")
    return name


def extract_course_from_chunk(chunk_text: str) -> str:
    """
    Try to extract a course code mentioned in the chunk text.
    e.g. 'CS61B', 'CS189', 'CS70'
    Returns the first match found, or 'unknown' if none.
    """
    matches = re.findall(r"CS\d+[A-Z]?", chunk_text)
    if matches:
        return matches[0]
    return "unknown"


def chunk_documents(cleaned_docs: dict) -> list:
    """
    Split all cleaned documents into chunks using recursive strategy.

    Returns a list of dicts, each with:
    {
        "text": str,           # the chunk text
        "source": str,         # filename (e.g. rmp_hug.txt)
        "professor": str,      # professor name extracted from filename
        "course": str,         # first course code found in chunk
        "chunk_index": int     # position of this chunk within its document
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # hierarchy from planning.md
        length_function=len,
    )

    all_chunks = []

    for filename, text in cleaned_docs.items():
        professor = extract_professor_name(filename)
        raw_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()

            # Skip empty or near-empty chunks
            if len(chunk_text) < 20:
                continue

            course = extract_course_from_chunk(chunk_text)

            all_chunks.append({
                "text": chunk_text,
                "source": filename,
                "professor": professor,
                "course": course,
                "chunk_index": i,
            })

    return all_chunks


def print_sample_chunks(chunks: list, n: int = 5):
    """
    Print n sample chunks for manual inspection.
    You should read each one and ask:
    - Does this make sense on its own?
    - Is it about one professor/review or mixed?
    - Are there any HTML artifacts or fragments?
    """
    print(f"\n{'='*50}")
    print(f"SAMPLE CHUNKS (showing {n} of {len(chunks)} total)")
    print("Read each one: is it self-contained and clean?")
    print(f"{'='*50}")

    # Pick evenly spaced samples across the full chunk list
    step = max(1, len(chunks) // n)
    samples = [chunks[i * step] for i in range(n) if i * step < len(chunks)]

    for i, chunk in enumerate(samples, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source:    {chunk['source']}")
        print(f"Professor: {chunk['professor']}")
        print(f"Course:    {chunk['course']}")
        print(f"Length:    {len(chunk['text'])} characters")
        print(f"Text:\n{chunk['text']}")


def print_chunk_summary(chunks: list):
    """Print statistics about the chunk set."""
    print(f"\n{'='*50}")
    print("CHUNKING SUMMARY")
    print(f"{'='*50}")
    print(f"Total chunks produced: {len(chunks)}")

    lengths = [len(c["text"]) for c in chunks]
    print(f"Average chunk length:  {sum(lengths) // len(lengths)} characters")
    print(f"Shortest chunk:        {min(lengths)} characters")
    print(f"Longest chunk:         {max(lengths)} characters")

    # Breakdown by professor
    print(f"\nChunks per professor:")
    from collections import Counter
    counts = Counter(c["professor"] for c in chunks)
    for prof, count in sorted(counts.items()):
        print(f"  {prof:<20} {count} chunks")

    # Sanity check
    print(f"\n{'='*50}")
    print("SANITY CHECK")
    print(f"{'='*50}")
    if len(chunks) < 50:
        print("⚠️  WARNING: Fewer than 50 chunks — chunks may be too large.")
        print("   Consider reducing CHUNK_SIZE or check that documents loaded correctly.")
    elif len(chunks) > 2000:
        print("⚠️  WARNING: More than 2000 chunks — chunks may be too small.")
        print("   Consider increasing CHUNK_SIZE.")
    else:
        print(f"✅ Chunk count ({len(chunks)}) is in the healthy range (50–2000).")


if __name__ == "__main__":
    # Step 1: Load and clean documents
    cleaned_docs = load_and_clean_documents(DOCUMENTS_DIR)

    if not cleaned_docs:
        print("❌ No documents loaded. Check your documents/ folder.")
        exit(1)

    # Step 2: Chunk all documents
    print(f"\n{'='*50}")
    print("STEP 3: Chunking documents")
    print(f"Strategy: Recursive | Size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print(f"{'='*50}")

    chunks = chunk_documents(cleaned_docs)
    print(f"  Produced {len(chunks)} chunks from {len(cleaned_docs)} documents")

    # Step 3: Inspect sample chunks
    print_sample_chunks(chunks, n=5)

    # Step 4: Summary and sanity check
    print_chunk_summary(chunks)
