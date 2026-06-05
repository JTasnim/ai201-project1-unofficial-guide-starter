# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

CS professor and course reviews for UC Berkeley's CS department.

This knowledge is valuable because it's the kind of insider information students desperately need which professors give useful feedback, whose exams are fair, which courses have accessible office hours but it's completely absent from official university channels. The CS department website lists faculty bios and research interests. It tells you nothing about whether a professor's midterms are curved, whether they respond to emails, or whether attending lecture actually matters. Rate My Professors aggregates this student-generated knowledge but makes it unsearchable at scale; you'd have to click through dozens of professor pages, read hundreds of individual reviews, and manually synthesize patterns. This RAG system makes that corpus answerable by plain-language question.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

Source: Rate My Professors (ratemyprofessors.com) — UC Berkeley CS department

Collection method: Manual copy-paste into .txt files (one file per professor).
Each file will contain the professor's name, course(s) taught, and 15–25 student reviews in plain text.

| #   | File              | Professor         | Courses      |
| --- | ----------------- | ----------------- | ------------ |
| 1   | rmp_hilfinger.txt | Paul Hilfinger    | CS61A        |
| 2   | rmp_hug.txt       | Josh Hug          | CS61B        |
| 3   | rmp_sahai.txt     | Anant Sahai       | CS70         |
| 4   | rmp_song.txt      | Dawn Song         | CS161, CS294 |
| 5   | rmp_malik.txt     | Jitendra Malik    | CS189, CS280 |
| 6   | rmp_shenker.txt   | Scott Shenker     | CS168        |
| 7   | rmp_gonzalez.txt  | Joseph Gonzalez   | CS189, CS294 |
| 8   | rmp_stoica.txt    | Ion Stoica        | CS162, CS294 |
| 9   | rmp_wawrzynek.txt | John Wawrzynek    | CS152, CS250 |
| 10  | rmp_shewchuk.txt  | Jonathan Shewchuk | CS61C, CS184 |
| 11  | rmp_garcia.txt    | Dan Garcia        | CS10, CS61C  |
| 12  | rmp_sinclair.txt  | Alistair Sinclair | CS170        |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->
<!--
**Chunk size:**

**Overlap:**

**Reasoning:** -->

Strategy: Recursive Chunking

Target chunk size: 300 characters
Overlap: 30 characters
Split hierarchy: \n\n (between reviews) → \n (within a review) →
sentence boundary → words (last resort)

Why recursive over fixed-size:
RMP reviews are natural semantic units, each review is one student's complete
opinion, separated from the next by a blank line (\n\n). Recursive chunking
respects this structure by trying the largest split boundary first. If a review
fits within 300 characters, it stays intact as one chunk. If it's longer, the
splitter falls back to sentence boundaries rather than cutting mid-word or
mid-sentence like fixed-size would.

Fixed-size chunking ignores these boundaries entirely. At 300 characters it will
routinely cut mid-sentence and merge the end of one student's opinion with the
beginning of another's — producing chunks that contain contradictory sentiments
from two different reviewers. Those mixed chunks generate weak, ambiguous
embeddings that match poorly against specific queries.

Why these numbers:

300 characters: Most RMP reviews are 2–5 sentences (150–400 chars). At
300, the majority of reviews fit in a single chunk. Long reviews split at
sentence boundaries, not mid-word.
30 character overlap: Provides one partial sentence of context carry-over
when a review is split, without being large enough to duplicate significant
content across chunks.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

<!-- **Embedding model:**

**Top-k:**

**Production tradeoff reflection:** -->

Embedding model: all-MiniLM-L6-v2 via sentence-transformers
Why this model:

Runs fully locally — no API key, no rate limits, no cost
384-dimension vectors — fast to compute and query
Strong performance on short, opinion-based text (exactly our use case)
Well-documented, widely used baseline for RAG projects

Top-k: k=5

Too few (k=2): may miss the most relevant review if the top results happen to
be about a different course taught by the same professor
Too many (k=10): floods the LLM with loosely related reviews, pulling the
response off-target; also risks hitting the context window limit
k=5 gives enough diversity to capture patterns across multiple student
opinions without overwhelming the prompt

Production tradeoffs (if cost weren't a constraint):

text-embedding-3-large (OpenAI): Higher accuracy on nuanced queries,
1536 dimensions, but per-token API cost and latency make it expensive at scale

multilingual-e5-large: Better for multilingual campuses or international
student reviews, but unnecessary for English-only RMP data

instructor-xl: Task-specific embeddings — likely better accuracy for
domain-specific text, but much slower and heavier locally

For this project, all-MiniLM-L6-v2 is the correct choice: free, fast,
and well-matched to short English text

Vector store: ChromaDB (local, no account needed)

One collection: reviews_recursive
Each chunk stored with metadata: source (filename), professor, course, chunk_index

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| #   | Question                                                                            | Expected answer                                                                                                          |
| --- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 1   | What do students say about exam difficulty in CS189?                                | Reviews should mention Malik or Gonzalez; exams described as challenging and math-heavy; some mention bell-curve grading |
| 2   | Which Berkeley CS professors are known for giving detailed feedback on assignments? | Should surface specific professor names with reviews mentioning feedback quality                                         |
| 3   | How do students describe the workload for CS61B?                                    | Reviews of Hug should mention significant weekly workload, projects being time-consuming, but fair grading               |
| 4   | What do students say about office hours availability for CS70 professors?           | Reviews of Sahai should surface — likely mixed (large class, long office hours lines)                                    |
| 5   | Which CS professors do students most recommend for students new to the CS major?    | Should surface Garcia (CS10/61C) and Hilfinger (CS61A) reviews with beginner-friendly mentions                           |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Professor name disambiguation
   Some professors teach multiple courses. A query about "CS189 difficulty" might
   retrieve chunks about a professor's other course (CS294) if the review doesn't
   explicitly name the course. Mitigation: store course as chunk metadata and
   ensure course names appear in the chunk text during cleaning.

2. Grounding failure on synthesis questions
   Questions like "which professor is best overall" require synthesizing across
   many reviews — the LLM may draw on training knowledge about Berkeley CS rather
   than staying grounded in the retrieved chunks. Mitigation: strong system prompt
   instruction; detection: one of the 5 eval questions will test this explicitly.

3. RMP data volume variance
   Some professors have many more reviews than others. Professors with fewer
   reviews will produce fewer chunks, and retrieval will have less material to work
   with. This creates uneven coverage that will surface in evaluation.
4. Long reviews splitting across sentence boundaries

Reviews longer than 300 characters will be split at sentence boundaries with
30-character overlap. If a key fact (e.g., a specific grade or exam detail)
appears at the very end of a split chunk, the overlap may not carry enough
context to make the next chunk independently retrievable. Detection: inspect
split chunks manually for any that feel incomplete.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

┌─────────────────────────────────────────────────────────────────┐
│ DOCUMENT INGESTION │
│ Source: rmp\_\*.txt files (manual copy from RMP) │
│ Tool: Python file I/O + custom cleaner │
│ Output: cleaned plain-text strings, one per professor file │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ CHUNKING │
│ Strategy: Recursive (\n\n → \n → sentence → word) │
│ Target size: 300 chars | Overlap: 30 chars │
│ Tool: LangChain RecursiveCharacterTextSplitter │
│ Output: list of (chunk_text, metadata) tuples │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ EMBEDDING │
│ Model: all-MiniLM-L6-v2 (sentence-transformers, local) │
│ Output: 384-dim vectors for each chunk │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ VECTOR STORE │
│ Tool: ChromaDB (local, persistent) │
│ Collection: reviews_recursive │
│ Metadata per chunk: source, professor, course, chunk_index │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVAL │
│ Query embedded with same all-MiniLM-L6-v2 model │
│ Top-k=5 chunks returned by cosine similarity │
│ Source metadata returned alongside chunks │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ GENERATION │
│ LLM: Groq llama-3.3-70b-versatile (free tier) │
│ System prompt: answer ONLY from retrieved context; cite source │
│ If insufficient context: explicitly decline to answer │
│ Output: grounded answer + list of source filenames │
└────────────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ QUERY INTERFACE │
│ Tool: Gradio web UI │
│ Inputs: question textbox │
│ Outputs: answer textbox + sources textbox │
│ URL: http://localhost:7860 │
└─────────────────────────────────────────────────────────────────┘

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

# | Pipeline Component | What I'll give the AI | What I expect it to produce

---

1 | Ingestion + Cleaning | This planning.md (Documents + Architecture sections) + sample raw RMP text | ingest.py: loads all rmp\_\*.txt files, strips boilerplate, outputs clean text dict keyed by filename

2 | Chunking | Chunking Strategy section + Architecture diagram | chunk.py: implements recursive chunking with RecursiveCharacterTextSplitter, outputs (chunk_text, metadata) tuples, prints 5 sample chunks for inspection

3 | Embedding + Vector Store | Retrieval Approach section + Architecture diagram | embed.py: embeds all chunks with all-MiniLM-L6-v2, loads into ChromaDB collection reviews_recursive with correct metadata

4 | Retrieval Function | Architecture diagram + top-k spec | retrieve.py: retrieve(query, k=5) returns top-k chunks + source metadata

5 | Generation | Generation box from Architecture + grounding requirement | generate.py: builds prompt with retrieved chunks, calls Groq API, returns grounded answer + sources

6 | Gradio UI | Milestone 5 Gradio skeleton from project spec + my input/output spec | app.py: wires retrieve + generate, runs on port 7860

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
