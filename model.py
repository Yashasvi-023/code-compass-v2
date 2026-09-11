"""
model.py — GitHub RAG pipeline using google-genai SDK.
Includes exponential backoff for rate-limit (429) errors.
"""
import os
import shutil
import tempfile
import math
import time
import random
from git import Repo
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.6-flash"
CHUNK_SIZE       = 1500
CHUNK_OVERLAP    = 200
TOP_K            = 8
BATCH_SIZE       = 20    # smaller batches = fewer tokens per call = less rate limiting

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css",
    ".java", ".cpp", ".c", ".h", ".go", ".rs", ".md", ".txt",
    ".sh", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".rb", ".php", ".swift", ".kt"
}

# README-like files prioritized for "what is this about?" questions
PRIORITY_FILES = {"readme.md", "readme.txt", "readme", "description.md"}


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def _with_backoff(fn, max_retries: int = 6, base_delay: float = 2.0):
    """
    Call fn(). If a 429 / rate-limit / quota error is raised, wait with
    exponential backoff and retry up to max_retries times.

    Backoff schedule (seconds): 2, 4, 8, 16, 32, 64  (+ small random jitter)
    """
    for attempt in range(max_retries):
        try:
            return fn()

        except Exception as e:
            err = str(e).lower()
            is_rate_limit = (
                "429"        in err or
                "rate limit" in err or
                "quota"      in err or
                "exhausted"  in err or
                "resource_exhausted" in err
            )

            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[backoff] Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise   # not a rate-limit error, or out of retries → re-raise


# ─────────────────────────────────────────────────────────────────────────────
# RAG Model
# ─────────────────────────────────────────────────────────────────────────────

class GitHubRAGModel:
    """
    A RAG pipeline — no LangChain.

    How it works:
      1. Clone the GitHub repo.
      2. Read every code/doc file → split into overlapping text chunks.
         README files are indexed first so they score highest for overview questions.
      3. Embed every chunk using Gemini embeddings (with backoff on rate limits).
      4. Store chunks + embeddings in plain Python lists in memory.
      5. On a question:
         a. Embed the question.
         b. Compute cosine similarity vs all stored embeddings.
         c. Pick the TOP_K most similar chunks as context.
         d. Ask Gemini to answer using that context.
    """

    def __init__(self, api_key: str):
        self.api_key    = api_key
        self.client     = genai.Client(api_key=api_key)
        self.chunks     = []
        self.embeddings = []
        self.sources    = []
        self.repo_name  = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Process repository
    # ─────────────────────────────────────────────────────────────────────────

    def process_repository(self, repo_url: str):
        temp_dir = tempfile.mkdtemp()
        try:
            self.repo_name = repo_url.rstrip("/").split("/")[-1] \
                             .replace("-", " ").replace("_", " ")

            # 1. Clone
            Repo.clone_from(repo_url, temp_dir)

            # 2. Walk and chunk files (README first)
            priority_chunks, regular_chunks = [], []

            for root, _, files in os.walk(temp_dir):
                if ".git" in root:
                    continue
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in ALLOWED_EXTENSIONS:
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                        if not text:
                            continue
                        rel_path = os.path.relpath(file_path, temp_dir)
                        target   = priority_chunks if file.lower() in PRIORITY_FILES \
                                   else regular_chunks
                        for chunk in self._split_text(text):
                            target.append((chunk, rel_path))
                    except Exception:
                        pass

            raw_chunks = priority_chunks + regular_chunks
            if not raw_chunks:
                return False, "No readable files found in this repository."

            # 3. Embed in batches with backoff
            texts   = [c[0] for c in raw_chunks]
            sources = [c[1] for c in raw_chunks]
            all_embeddings = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]

                def _embed_batch(b=batch):
                    return self.client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=b,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )

                response = _with_backoff(_embed_batch)
                for emb in response.embeddings:
                    all_embeddings.append(emb.values)

                # Small pause between batches to stay under RPM limit
                if i + BATCH_SIZE < len(texts):
                    time.sleep(0.5)

            # 4. Store
            self.chunks     = texts
            self.embeddings = all_embeddings
            self.sources    = sources

            num_files = len(set(sources))
            return True, (
                f"✅ Loaded **{self.repo_name}** — "
                f"{num_files} files, {len(self.chunks)} chunks indexed."
            )

        except Exception as e:
            return False, f"Error processing repository: {str(e)}"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Ask a question
    # ─────────────────────────────────────────────────────────────────────────

    def ask_question(self, question: str) -> str:
        if not self.chunks:
            return "No repository loaded. Please load a repository first."

        # Embed question (with backoff)
        def _embed_question():
            return self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )

        q_resp = _with_backoff(_embed_question)
        q_vec  = q_resp.embeddings[0].values

        # Rank chunks by cosine similarity
        scores = sorted(
            enumerate(self.embeddings),
            key=lambda x: self._cosine_similarity(q_vec, x[1]),
            reverse=True
        )
        top_indices = [idx for idx, _ in scores[:TOP_K]]

        # Build context
        context = "\n\n---\n\n".join(
            f"[File: {self.sources[i]}]\n{self.chunks[i]}"
            for i in top_indices
        )

        prompt = f"""You are an expert code assistant for the repository "{self.repo_name}".

You have been given relevant excerpts from the codebase as context. Use them to answer the user's question.

Rules:
- Base your answer primarily on the provided context.
- For high-level questions ("what is this project about?"), summarize what you can infer from the context (README, file names, code structure).
- If you cannot determine the answer from the context, say so clearly.
- Be concise but thorough. Use code blocks where appropriate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        def _generate():
            return self.client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )

        response = _with_backoff(_generate)
        return response.text

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _split_text(self, text: str) -> list:
        """Split text into fixed-size overlapping chunks."""
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + CHUNK_SIZE])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Cosine similarity between two vectors (0.0 – 1.0)."""
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
