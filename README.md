# The Unofficial Guide: UC Berkeley CS Professors
### A RAG System Built on Student-Generated Knowledge

---

## Domain and Document Sources

**Domain:** CS professor and course reviews for UC Berkeley's CS department.

This knowledge is valuable because it is the kind of insider information students
desperately need — which professors give useful feedback, whose exams are fair,
which courses have accessible office hours — but it is completely absent from
official university channels. The CS department website lists faculty bios and
research interests. It tells you nothing about whether a professor's midterms are
curved, whether they respond to emails, or whether attending lecture actually matters.
Rate My Professors aggregates this student-generated knowledge but makes it
unsearchable at scale — you would have to click through dozens of professor pages,
read hundreds of individual reviews, and manually synthesize patterns. This RAG
system makes that corpus answerable by plain-language question.

**Source:** Rate My Professors (ratemyprofessors.com) — UC Berkeley CS department
**Collection method:** Manual copy-paste into `.txt` files (one per professor), since
RMP blocks automated scrapers due to JavaScript rendering.

| File | Professor | Courses |
|------|-----------|---------|
| rmp_hilfinger.txt | Paul Hilfinger | CS61A |
| rmp_hug.txt | Josh Hug | CS61B |
| rmp_sahai.txt | Anant Sahai | CS70 |
| rmp_song.txt | Dawn Song | CS161, CS294 |
| rmp_malik.txt | Jitendra Malik | CS189, CS280 |
| rmp_shenker.txt | Scott Shenker | CS168 |
| rmp_gonzalez.txt | Joseph Gonzalez | CS189, CS294 |
| rmp_stoica.txt | Ion Stoica | CS162, CS294 |
| rmp_wawrzynek.txt | John Wawrzynek | CS152, CS250 |
| rmp_shewchuk.txt | Jonathan Shewchuk | CS61C, CS184 |
| rmp_garcia.txt | Dan Garcia | CS10, CS61C |
| rmp_sinclair.txt | Alistair Sinclair | CS170 |

---

## Chunking Strategy and Reasoning

**Strategy:** Recursive Chunking via LangChain `RecursiveCharacterTextSplitter`
**Chunk size:** 300 characters | **Overlap:** 30 characters
**Split hierarchy:** `\n\n` → `\n` → space → character

RMP reviews are natural semantic units — each review is one student's complete
opinion, separated from the next by a blank line (`\n\n`). Recursive chunking
respects this by trying the largest boundary first. If a review fits within 300
characters, it stays intact as one chunk. If it is longer, the splitter falls back
to word boundaries rather than cutting mid-sentence.

Fixed-size chunking was considered and rejected: at 300 characters it routinely
cuts mid-sentence and merges the end of one student's opinion with the beginning
of another's, producing chunks with contradictory sentiments that generate weak,
ambiguous embeddings.

Each chunk is prefixed with professor and course context (e.g., `[Professor Hug | Course CS61B]`)
so embeddings carry this signal even when the review text itself does not mention
the course code explicitly. A `PROFESSOR_COURSES` dictionary in `chunk.py` ensures
every chunk is tagged with the correct course regardless of what appears in the
review text.

**Total chunks produced:** 190 across 12 documents (~15–17 per professor)

---

## Sample Chunks

**Chunk 1** — `rmp_hug.txt`
```
[Professor Hug | Course CS61B]
CS61B with Hug is a rite of passage for Berkeley CS students. The projects
(Gitlet especially) are genuinely hard — expect to spend full weekends on them.
That said, the autograder gives you unlimited submissions so you can iterate
and learn from your mistakes.
```

**Chunk 2** — `rmp_sahai.txt`
```
[Professor Sahai | Course CS70]
Sahai is brilliant and you can tell he has thought deeply about how to teach
discrete math and probability. His lectures are dense — you will not follow
everything in real time, but rewatching at 0.75x speed at home is extremely
valuable.
```

**Chunk 3** — `rmp_garcia.txt`
```
[Professor Garcia | Course CS61C]
Dan Garcia is the most enthusiastic professor I have ever encountered in any
subject at any level. His energy in lecture is genuinely infectious — he makes
you feel like computer science is the most exciting thing in the world.
```

**Chunk 4** — `rmp_sinclair.txt`
```
[Professor Sinclair | Course CS170]
Sinclair is the gold standard for CS170 (Algorithms). His lectures are
exceptionally well-organized — he builds up every algorithm from first
principles, proves correctness carefully, and always situates the algorithm
in the broader context of what problems it solves.
```

**Chunk 5** — `rmp_malik.txt`
```
[Professor Malik | Course CS189]
CS189 with Malik is not a class you take to get an easy ML credential.
It is a class you take if you want to actually understand machine learning
from first principles. Derivations happen on the board in real time.
```

---

## Embedding Model

**Model:** `all-MiniLM-L6-v2` via `sentence-transformers`
**Dimensions:** 384
**Runs:** Fully locally — no API key, no rate limits, no cost

**Why this model for this project:** It runs locally with no cost, produces
384-dimensional vectors that are fast to compute and query, and performs well on
short, opinion-based English text — which is exactly what RMP reviews are.

**Production tradeoffs I would consider:**

- **`text-embedding-3-large` (OpenAI):** Higher accuracy on nuanced queries
  (1536 dimensions), but per-token API cost and latency make it expensive at scale.
  Best for production systems where retrieval precision is critical.
- **`multilingual-e5-large`:** Better for multilingual campuses or international
  student reviews, but unnecessary overhead for English-only RMP data.
- **`instructor-xl`:** Accepts task-specific instructions — likely better accuracy
  for domain-specific text, but much slower and heavier to run locally.

For this project, `all-MiniLM-L6-v2` is the right choice: free, fast, and
well-matched to short English text.

---

## Retrieval Test Results

**Setup:** Top-k = 5, cosine similarity, ChromaDB collection `reviews_recursive`

### Query 1: "What do students say about the workload for CS61B?"

| Rank | Source | Relevant? |
|------|--------|-----------|
| 1 | rmp_song.txt | No — CS161, wrong course |
| 2 | rmp_gonzalez.txt | Partial — workload mentioned but CS189 |
| 3 | rmp_shewchuk.txt | No — CS61C, wrong course |
| 4 | rmp_sahai.txt | No — CS70, wrong course |

**Why results are problematic:** The word "workload" appears across many
professors' reviews. Without `rmp_hug.txt` appearing in the top results, the
retrieval is failing to match the most relevant source. This is the primary
failure case documented in the evaluation section.

### Query 2: "What do students say about office hours for CS70?"

| Rank | Source | Relevant? |
|------|--------|-----------|
| 1 | rmp_shewchuk.txt | No — wrong professor |
| 2 | rmp_sahai.txt | Partial — CS70 professor but office hours not the focus |
| 3 | rmp_song.txt | No — wrong course |
| 4 | rmp_stoica.txt | No — wrong course |
| 5 | rmp_gonzalez.txt | No — wrong course |

**Why top result is relevant:** Sahai (result 2) is the correct CS70 professor
and does appear in the retrieved set, but is not ranked first. The query matched
Shewchuk's office hours content ahead of Sahai's CS70 content.

### Query 3: "Which professors do students recommend for beginners?"

| Rank | Source | Relevant? |
|------|--------|-----------|
| 1 | rmp_hug.txt | Partial — not an intro course |
| 2 | rmp_garcia.txt | Yes — CS10 explicitly for beginners |
| 3 | rmp_sinclair.txt | No — algorithms course |
| 4 | rmp_shenker.txt | No — upper division |
| 5 | rmp_song.txt | No — upper division |

**Why Garcia result is relevant:** Garcia's review text explicitly mentions
"non-majors" and "beginners," which semantically matches the query. The system
correctly identified Garcia from the retrieved set even though he was ranked 2nd.

---

## Grounded Generation

Grounding is enforced in two ways:

**1. System prompt prohibition:**
```
Answer ONLY from the student review excerpts provided in the user message.
Do NOT use your general knowledge about UC Berkeley, professors, or courses.
If the provided reviews do not contain enough information, say:
"I don't have enough information in the reviews to answer that."
```

**2. Programmatic source attribution:** Sources are collected from chunk metadata
and appended to every response in code — this guarantees attribution is always
present regardless of what the LLM generates. The LLM cannot omit or fabricate
sources because they are added by the pipeline after generation.

---

## Example Responses

### Response 1 — Grounded answer with source attribution

**Query:** Which Berkeley CS professors are known for giving detailed feedback?

**System response:**
> According to Source 3, Professor Shewchuk is known for giving feedback on work
> that is "genuinely useful, not just" routine comments, implying that he provides
> detailed feedback. I don't have enough information in the reviews to answer that
> for other professors, as only Source 3 explicitly mentions a professor giving
> detailed feedback.

**Sources retrieved:**
- rmp_sinclair.txt (Prof. Sinclair)
- rmp_garcia.txt (Prof. Garcia)
- rmp_shewchuk.txt (Prof. Shewchuk)
- rmp_hug.txt (Prof. Hug)
- rmp_gonzalez.txt (Prof. Gonzalez)

### Response 2 — Grounded answer for beginners query

**Query:** Which CS professors do students most recommend for students new to the CS major?

**System response:**
> According to the reviews, Professor Garcia is highly recommended, especially for
> non-CS students who are new to the major. As stated in Source 2, "Garcia is that
> good. He breaks down intimidating concepts with humor and patience, and the course
> is structured so that every student can succeed if they put in the work."

**Sources retrieved:**
- rmp_hug.txt (Prof. Hug)
- rmp_garcia.txt (Prof. Garcia)
- rmp_sinclair.txt (Prof. Sinclair)
- rmp_shenker.txt (Prof. Shenker)
- rmp_song.txt (Prof. Song)

### Response 3 — Out-of-scope refusal

**Query:** What is the best restaurant in Berkeley?

**System response:**
> I don't have enough information in the reviews to answer that.

**Sources retrieved:**
- rmp_hilfinger.txt (Prof. Hilfinger)
- rmp_sinclair.txt (Prof. Sinclair)
- rmp_wawrzynek.txt (Prof. Wawrzynek)
- rmp_shenker.txt (Prof. Shenker)

The system correctly refuses to answer from general knowledge. Grounding is working.

---

## Query Interface

**Tool:** Gradio web UI
**Run:** `python app.py` → open `http://localhost:7860`

**Input fields:**
- Question textbox — accepts any plain-language question about Berkeley CS professors

**Output fields:**
- Answer textbox — LLM response grounded in retrieved chunks
- "Retrieved from" textbox — list of source `.txt` files used

**Sample interaction:**

```
Input:  "Which CS professors do students most recommend for students new to CS?"

Output (Answer):
According to the reviews, Professor Garcia is highly recommended for students
new to the major. He breaks down intimidating concepts with humor and patience,
and the course is structured so that every student can succeed.

Output (Retrieved from):
• rmp_hug.txt (Prof. Hug)
• rmp_garcia.txt (Prof. Garcia)
• rmp_sinclair.txt (Prof. Sinclair)
• rmp_shenker.txt (Prof. Shenker)
• rmp_song.txt (Prof. Song)
```

---

## Evaluation Report

### Q1: What do students say about exam difficulty in CS189?

**Expected answer:** Reviews should mention Malik or Gonzalez; exams described
as challenging and math-heavy; some mention curve grading.

**System response:**
> According to Source 5, students say that the class CS189 is "still hard" and
> the material is "inherently difficult." However, Professor Gonzalez's efforts,
> such as holding extra office hours and providing clear rubrics, make the class
> "survivable" and the difficulty does not feel "artificial or punitive." There is
> no direct comment on exam difficulty in CS189. To directly answer the question
> about exam difficulty, I don't have enough information in the reviews to answer that.

**Sources retrieved:** rmp_sinclair.txt, rmp_sahai.txt, rmp_song.txt,
rmp_garcia.txt, rmp_gonzalez.txt

**Accuracy: 🟡 Partially Accurate**

Gonzalez (one of the two CS189 professors) appeared in the retrieved set and the
answer correctly identifies that CS189 is hard. However, Malik (the primary CS189
instructor) did not appear in the top results at all, and the system did not
surface the specific exam characteristics described in Malik's reviews (math-heavy
derivations, generous curve). The answer hedges correctly rather than hallucinating,
but it is incomplete.

---

### Q2: Which Berkeley CS professors are known for giving detailed feedback?

**Expected answer:** Specific professor names with reviews mentioning feedback
quality — Shewchuk based on corpus content.

**System response:**
> According to Source 3, Professor Shewchuk is known for giving feedback on work
> that is "genuinely useful, not just" routine comments, implying that he provides
> detailed feedback. I don't have enough information in the reviews to answer that
> for other professors, as only Source 3 explicitly mentions a professor giving
> detailed feedback.

**Sources retrieved:** rmp_sinclair.txt, rmp_garcia.txt, rmp_shewchuk.txt,
rmp_hug.txt, rmp_gonzalez.txt

**Accuracy: ✅ Accurate**

Shewchuk is correctly identified. The answer appropriately declines to name
other professors when the retrieved chunks do not explicitly mention feedback
quality for them — this is correct grounding behavior, not a failure. The system
cited the right source and stayed within the evidence.

---

### Q3: How do students describe the workload for CS61B?

**Expected answer:** Reviews of Hug should mention heavy weekly workload,
time-consuming projects, and fair grading.

**System response:**
> I don't have enough information in the reviews to answer that.

**Sources retrieved:** rmp_song.txt, rmp_gonzalez.txt, rmp_shewchuk.txt,
rmp_sahai.txt

**Accuracy: ❌ Inaccurate**

This is a complete retrieval failure. The system returned zero chunks from
`rmp_hug.txt` — the only source that contains CS61B workload information.
Instead it retrieved Song (CS161), Gonzalez (CS189), Shewchuk (CS61C), and
Sahai (CS70). The LLM correctly refused to answer from irrelevant context,
but the retrieval pipeline failed to surface the right document. The root
cause is analyzed in the failure case section below.

---

### Q4: What do students say about office hours availability for CS70 professors?

**Expected answer:** Sahai reviews should surface — mixed experience (large
class, long office hours lines).

**System response:**
> I don't have enough information in the reviews to answer that. The reviews only
> mention office hours for Professors Shewchuk (Source 1), Song (Source 3), and
> Stoica (Source 4), as well as Professor Gonzalez (Source 5), but not for
> Professor Sahai (Source 2) in the context of office hours availability.

**Sources retrieved:** rmp_shewchuk.txt, rmp_sahai.txt, rmp_song.txt,
rmp_stoica.txt, rmp_gonzalez.txt

**Accuracy: 🟡 Partially Accurate**

Sahai did appear in the retrieved set (ranked 2nd), and `rmp_sahai.txt` does
contain detailed office hours information. However, the specific chunk retrieved
from Sahai's file focused on workload rather than office hours — the office hours
content was in a different chunk that was not retrieved. The LLM correctly
identified that the retrieved Sahai chunk did not address office hours and refused
to fabricate an answer. Grounding worked; retrieval was incomplete.

---

### Q5: Which CS professors do students most recommend for students new to the CS major?

**Expected answer:** Garcia (CS10/CS61C) and Hilfinger (CS61A) with
beginner-friendly mentions.

**System response:**
> According to the reviews, Professor Garcia is highly recommended, especially for
> non-CS students who are new to the major. As stated in Source 2, "Garcia is that
> good. He breaks down intimidating concepts with humor and patience, and the course
> is structured so that every student can succeed if they put in the work." This
> suggests that Professor Garcia's teaching style is well-suited for students who
> are new to the CS major. Additionally, Source 1 mentions that CS61B is a required
> class for CS majors, but it does not specifically recommend Professor Hug for
> students new to the CS major. I don't have enough information in the reviews to
> determine if the other professors are suitable for students new to the CS major.

**Sources retrieved:** rmp_hug.txt, rmp_garcia.txt, rmp_sinclair.txt,
rmp_shenker.txt, rmp_song.txt

**Accuracy: 🟡 Partially Accurate**

Garcia is correctly identified and the answer quotes the right review content.
However, Hilfinger (CS61A — the actual gateway intro course) did not appear in
the retrieved set at all. The system retrieved Hug (CS61B), Sinclair (CS170),
Shenker (CS168), and Song (CS161) instead — none of which are intro-level courses.
The answer correctly avoided recommending those professors for beginners, but the
missing Hilfinger retrieval means the response is incomplete.

---

## Failure Case Analysis

**Primary failure: Q3 — CS61B workload retrieval (Inaccurate)**

The system returned "I don't have enough information" for the CS61B workload
query despite `rmp_hug.txt` containing multiple detailed reviews about exactly
this topic. The retrieved sources were Song, Gonzalez, Shewchuk, and Sahai —
all wrong professors.

**Root cause:** The word "workload" appears across reviews for many different
professors. When the query "How do students describe the workload for CS61B?" is
embedded, it matches the semantic concept of workload broadly — and the
`[Professor Hug | Course CS61B]` prefix added to chunks was not weighted
strongly enough to overcome the workload signal appearing in other professors'
chunks. The embedding model treats the prefix and the review text with equal
weight, so a chunk from Song's file that says "workload is significant" competes
directly with a Hug chunk that says "workload is intense" — and the course tag
alone is not enough to differentiate them at 300-character chunk scale.

**What would fix it:**
Two approaches would address this. First, increasing chunk size to 500 characters
would give each chunk more semantic content, making the course context a smaller
proportion of the total signal and reducing false matches. Second, using a
reranking step — retrieving top-20 candidates and then reranking by metadata
filter (only return chunks where `professor == "hug"` if the query mentions CS61B)
— would guarantee the right source appears. This is a known limitation of
pure semantic search on short text with shared vocabulary across documents.

**Secondary failure pattern: Q1, Q4, Q5 — correct professor retrieved but wrong chunk**

In three of the five queries, the right professor appeared in the retrieved set
but the specific chunk retrieved from that professor's file did not contain the
relevant information. For example in Q4, Sahai appeared (ranked 2nd) but the
retrieved Sahai chunk discussed workload rather than office hours. This is a
chunk boundary problem — the office hours content was split into a different
chunk that ranked lower than the workload chunk. Larger chunks or a higher k
value (k=8 instead of k=5) would increase the chance of surfacing the right
chunk when the right professor is already being retrieved.

---

## Spec Reflection

**One way the spec helped:** The planning.md requirement to define the chunking
strategy before writing any code was directly valuable. Writing out that RMP
reviews are separated by `\n\n` and that recursive chunking respects this
boundary forced a specific, justified design decision rather than a default
choice. When retrieval was failing, returning to the spec made it immediately
clear that the embedding model needed professor and course context injected into
the chunk text — not just stored as metadata — because metadata is not included
in the embedding vector.

**One way implementation diverged from the spec:** The spec did not anticipate
needing a `PROFESSOR_COURSES` hardcoded mapping or a context prefix on every
chunk. The original plan was that the course code would be extracted from the
review text itself. During implementation it became clear that most reviews do
not mention the course code explicitly — they just say "this class" or "the
projects" — so the course extractor returned "unknown" for most chunks. The fix
was to hardcode the mapping and prepend `[Professor X | Course Y]` to every
chunk before embedding. This is a real RAG engineering pattern called contextual
retrieval and it partially improved results, though Q3 shows it was not
sufficient on its own.

---

## AI Usage

**Instance 1: Generating the ingestion and chunking pipeline**

I provided Claude with my planning.md (Documents section, Architecture diagram,
and Chunking Strategy section) and asked it to implement `ingest.py` and
`chunk.py`. It produced working code for both files. After running `python chunk.py`
and inspecting the 5 sample chunks, I found that chunks were starting with `. `
(a dangling period) because `". "` was included as a separator, causing the
splitter to cut at sentence-ending periods and leave the period at the start of
the next chunk. I identified this issue from the output, directed Claude to remove
`". "` from the separators list and add `.lstrip(". ")` post-processing, and
verified the fix resolved the problem.

**Instance 2: Debugging the CS61B retrieval failure**

When `generate.py` returned "I don't have enough information" for the CS61B
workload query, I ran a direct retrieval diagnostic and confirmed that `rmp_hug.txt`
was not appearing in the results. I gave Claude the full prompt being sent to the
LLM and asked it to identify the failure. Claude correctly diagnosed it as a
retrieval failure (wrong chunks being passed to the LLM) rather than a generation
failure. I then directed Claude to add a `PROFESSOR_COURSES` dictionary and prepend
course context to each chunk text before embedding. I specified the approach; Claude
implemented it. After rebuilding the vector store, retrieval improved for most
queries but Q3 (CS61B workload) remained a failure, which I documented honestly
rather than hiding.

---

## Pipeline Architecture

```
documents/rmp_*.txt
        |
        v
   ingest.py       — loads + cleans 12 professor review files
        |
        v
   chunk.py        — RecursiveCharacterTextSplitter (300 chars, 30 overlap)
                     prepends [Professor X | Course Y] to each chunk
                     PROFESSOR_COURSES mapping ensures correct course tags
        |
        v
   embed.py        — all-MiniLM-L6-v2 -> 384-dim vectors -> ChromaDB
                     collection: reviews_recursive (190 chunks)
        |
        v
   retrieve.py     — cosine similarity, top-k=5
        |
        v
   generate.py     — Groq llama-3.3-70b-versatile
                     system prompt enforces grounding + refusal
                     sources appended programmatically from metadata
        |
        v
   app.py          — Gradio UI at http://localhost:7860
```

---

## Running the Project

```bash
# 1. Clone and set up environment
git clone <your-fork-url>
cd ai201-project1-unofficial-guide-starter
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)

# 4. Build the vector store (run once)
python embed.py

# 5. Launch the app
python app.py
# Open http://localhost:7860
```

**Dependencies:**
```
langchain-text-splitters
chromadb
sentence-transformers
groq
gradio
python-dotenv
```