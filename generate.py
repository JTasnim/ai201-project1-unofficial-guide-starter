"""
generate.py — Grounded Response Generation
The Unofficial Guide: UC Berkeley CS Professor Reviews

What this script does:
1. Takes a user query
2. Retrieves top-5 relevant chunks from ChromaDB (via retrieve.py)
3. Builds a grounded prompt — LLM must answer ONLY from retrieved chunks
4. Calls Groq llama-3.3-70b-versatile for generation
5. Returns the answer + source attribution

Grounding is enforced two ways:
- System prompt explicitly forbids using outside knowledge
- Sources are appended programmatically from metadata, not left to the LLM
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retrieve import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


def build_prompt(query: str, chunks: list) -> str:
    """
    Build the user prompt by injecting retrieved chunks as context.
    Each chunk is labeled with its source so the LLM can reference it.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} — Professor {chunk['professor'].title()}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""Here are student reviews retrieved from the UC Berkeley CS professor review database:

{context}

---

Question: {query}

Answer the question using only the information in the student reviews above.
If the reviews do not contain enough information to answer the question,
say exactly: "I don't have enough information in the reviews to answer that."
Cite which sources (e.g. Source 1, Source 3) support each part of your answer."""

    return prompt


SYSTEM_PROMPT = """You are The Unofficial Guide — a helpful assistant that answers questions
about UC Berkeley CS professors and courses based exclusively on student reviews.

STRICT RULES you must follow:
1. Answer ONLY from the student review excerpts provided in the user message.
2. Do NOT use your general knowledge about UC Berkeley, professors, or courses.
3. Do NOT make up or infer information that is not explicitly stated in the reviews.
4. If the provided reviews do not contain enough information, say:
   "I don't have enough information in the reviews to answer that."
5. Always cite which sources your answer draws from (e.g. "According to Source 2...").
6. Keep your tone helpful and neutral — you are summarizing student opinions, not endorsing them."""


def generate_answer(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate → return grounded answer.

    Returns a dict:
    {
        "answer": str,           # LLM's grounded response
        "sources": list[str],    # list of source filenames used
        "chunks": list[dict],    # raw retrieved chunks (for inspection)
    }
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, k=TOP_K)

    # Step 2: Build grounded prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Call Groq API
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,      # lower = more grounded, less creative
        max_tokens=600,
    )

    answer = completion.choices[0].message.content

    # Step 4: Collect sources programmatically from metadata
    # (not left to the LLM — this guarantees attribution is always present)
    seen = set()
    sources = []
    for chunk in chunks:
        src = chunk["source"]
        if src not in seen:
            seen.add(src)
            sources.append(f"{src} (Prof. {chunk['professor'].title()})")

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }


if __name__ == "__main__":
    # Quick end-to-end test
    print("=" * 60)
    print("GENERATION TEST — 2 queries end-to-end")
    print("=" * 60)

    test_queries = [
        "What do students say about the workload for CS70?",
        "What is the best restaurant in Berkeley?",   # out-of-scope test
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        result = generate_answer(query)

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources retrieved:")
        for src in result["sources"]:
            print(f"  • {src}")
        print()