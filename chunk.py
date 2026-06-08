"""
chunk.py — Recursive Chunking Pipeline
The Unofficial Guide: UC Berkeley CS Professor Reviews

Strategy: Recursive Chunking
- Target chunk size: 300 characters
- Overlap: 30 characters
- Split hierarchy: \n\n → \n → sentence boundary → word boundary
- Tool: LangChain RecursiveCharacterTextSplitter
"""

import re
from ingest import load_and_clean_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = "documents"

# Chunking config — matches planning.md exactly
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# Known course mapping — reliable fallback when review text doesn't mention course code
PROFESSOR_COURSES = {
    "hug": "CS61B",
    "hilfinger": "CS61A",
    "sahai": "CS70",
    "garcia": "CS61C",
    "malik": "CS189",
    "gonzalez": "CS189",
    "stoica": "CS162",
    "shewchuk": "CS61C",
    "shenker": "CS168",
    "wawrzynek": "CS152",
    "sinclair": "CS170",
    "song": "CS161",
}


def extract_professor_name(filename: str) -> str:
    """
    Extract a readable professor name from the filename.
    e.g. 'rmp_hug.txt' → 'hug'
    """
    return filename.replace("rmp_", "").replace(".txt", "")


def extract_course_from_chunk(chunk_text: str) -> str:
    """
    Try to extract a course code mentioned in the chunk text.
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
        "text": str,           # chunk text with professor/course prefix
        "source": str,         # filename (e.g. rmp_hug.txt)
        "professor": str,      # professor name
        "course": str,         # course code (from mapping, fallback to text)
        "chunk_index": int     # position within document
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    # Header lines to skip — file metadata, not review content
    header_markers = ["Professor:", "Course:", "University:"]

    all_chunks = []

    for filename, text in cleaned_docs.items():
        professor = extract_professor_name(filename)

        # Use known mapping as primary source, chunk text as fallback
        course = PROFESSOR_COURSES.get(professor, "unknown")

        raw_chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(raw_chunks):
            # Strip leading/trailing whitespace and leading periods
            chunk_text = chunk_text.strip().lstrip(". ")

            # Skip empty or near-empty chunks
            if len(chunk_text) < 40:
                continue

            # Skip chunks that contain only header lines
            lines = [l.strip() for l in chunk_text.splitlines() if l.strip()]
            if lines and all(any(l.startswith(h) for h in header_markers) for l in lines):
                continue

            # Prepend professor + course so embeddings carry this signal
            # This ensures "CS61B" appears in every Hug chunk, even if the
            # review text itself doesn't mention the course code
            contextual_text = (
                f"[Professor {professor.title()} | Course {course}]\n{chunk_text}"
            )

            all_chunks.append({
                "text": contextual_text,
                "source": filename,
                "professor": professor,
                "course": course,
                "chunk_index": i,
            })

    return all_chunks


def print_sample_chunks(chunks: list, n: int = 5):
    """Print n sample chunks for manual inspection."""
    print(f"\n{'='*50}")
    print(f"SAMPLE CHUNKS (showing {n} of {len(chunks)} total)")
    print("Read each one: is it self-contained and clean?")
    print(f"{'='*50}")

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

    print(f"\nChunks per professor:")
    from collections import Counter
    counts = Counter(c["professor"] for c in chunks)
    for prof, count in sorted(counts.items()):
        print(f"  {prof:<20} {count} chunks")

    print(f"\n{'='*50}")
    print("SANITY CHECK")
    print(f"{'='*50}")
    if len(chunks) < 50:
        print("⚠️  WARNING: Fewer than 50 chunks — chunks may be too large.")
    elif len(chunks) > 2000:
        print("⚠️  WARNING: More than 2000 chunks — chunks may be too small.")
    else:
        print(f"✅ Chunk count ({len(chunks)}) is in the healthy range (50–2000).")


if __name__ == "__main__":
    cleaned_docs = load_and_clean_documents(DOCUMENTS_DIR)

    if not cleaned_docs:
        print("❌ No documents loaded. Check your documents/ folder.")
        exit(1)

    print(f"\n{'='*50}")
    print("STEP 3: Chunking documents")
    print(f"Strategy: Recursive | Size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    print(f"{'='*50}")

    chunks = chunk_documents(cleaned_docs)
    print(f"  Produced {len(chunks)} chunks from {len(cleaned_docs)} documents")

    print_sample_chunks(chunks, n=5)
    print_chunk_summary(chunks)

    print(f"\n✅ Chunking complete. Ready for embedding (Milestone 4).")