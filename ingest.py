"""
ingest.py — Document Ingestion and Cleaning Pipeline
The Unofficial Guide: UC Berkeley CS Professor Reviews

What this script does:
1. Loads all rmp_*.txt files from the documents/ folder
2. Cleans each file (removes boilerplate, normalizes whitespace)
3. Returns a dict of {filename: cleaned_text}
4. Prints a summary so you can verify before chunking
"""

import os
import re


DOCUMENTS_DIR = "documents"


def load_raw_documents(folder: str) -> dict:
    """
    Load all rmp_*.txt files from the documents folder.
    Returns a dict: {filename: raw_text}
    """
    documents = {}

    if not os.path.exists(folder):
        print(f"ERROR: '{folder}' folder not found.")
        print("Make sure you have a documents/ folder with your rmp_*.txt files.")
        return documents

    files = sorted([f for f in os.listdir(folder) if f.startswith("rmp_") and f.endswith(".txt")])

    if not files:
        print(f"ERROR: No rmp_*.txt files found in '{folder}/'")
        return documents

    for filename in files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        documents[filename] = raw_text
        print(f"  Loaded: {filename} ({len(raw_text)} characters)")

    return documents


def clean_document(text: str) -> str:
    """
    Clean a single document's raw text.

    Removes:
    - HTML tags (in case any were accidentally included)
    - Excessive whitespace and blank lines (3+ blank lines → 2)
    - Windows-style line endings
    - Leading/trailing whitespace on each line

    Keeps:
    - Review text and opinions
    - Professor name and course headers
    - Single blank lines between reviews (\n\n) — critical for chunking
    """

    # Remove any HTML tags that may have been accidentally copied
    text = re.sub(r"<[^>]+>", "", text)

    # Remove HTML entities (&amp; &nbsp; &#39; etc.)
    text = re.sub(r"&[a-zA-Z]+;", "", text)
    text = re.sub(r"&#\d+;", "", text)

    # Normalize Windows line endings to Unix
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean up each line — strip leading/trailing whitespace
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3+ consecutive blank lines into exactly 2
    # (preserves the \n\n separator between reviews that chunker relies on)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Final strip
    text = text.strip()

    return text


def load_and_clean_documents(folder: str) -> dict:
    """
    Full ingestion pipeline: load + clean all documents.
    Returns a dict: {filename: cleaned_text}
    """
    print(f"\n{'='*50}")
    print("STEP 1: Loading raw documents")
    print(f"{'='*50}")

    raw_docs = load_raw_documents(folder)

    if not raw_docs:
        return {}

    print(f"\n{'='*50}")
    print("STEP 2: Cleaning documents")
    print(f"{'='*50}")

    cleaned_docs = {}
    for filename, raw_text in raw_docs.items():
        cleaned = clean_document(raw_text)
        cleaned_docs[filename] = cleaned
        print(f"  Cleaned: {filename} ({len(raw_text)} → {len(cleaned)} characters)")

    return cleaned_docs


def print_summary(cleaned_docs: dict):
    """Print a summary and a sample of one document for verification."""
    print(f"\n{'='*50}")
    print("INGESTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total documents loaded: {len(cleaned_docs)}")
    total_chars = sum(len(t) for t in cleaned_docs.values())
    print(f"Total characters across all documents: {total_chars:,}")

    print(f"\n{'='*50}")
    print("SAMPLE: First 800 characters of first document")
    print("(Read this to verify cleaning worked correctly)")
    print(f"{'='*50}")

    first_filename = list(cleaned_docs.keys())[0]
    first_text = cleaned_docs[first_filename]
    print(f"\nFile: {first_filename}")
    print("-" * 40)
    print(first_text[:800])
    print("...")


if __name__ == "__main__":
    cleaned_docs = load_and_clean_documents(DOCUMENTS_DIR)

    if cleaned_docs:
        print_summary(cleaned_docs)
        print(f"\n✅ Ingestion complete. {len(cleaned_docs)} documents ready for chunking.")
    else:
        print("\n❌ Ingestion failed. Check that your documents/ folder exists and contains rmp_*.txt files.")