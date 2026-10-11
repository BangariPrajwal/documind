"""
Core RAG logic for DocuMind.

Kept separate from the Streamlit UI so that every step can be unit-tested
and reused by the evaluation script:

    PDF -> pages -> chunks -> embeddings -> FAISS index -> retrieve -> LLM -> answer + citations
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are DocuMind, an assistant that answers questions ONLY from the context "
    "taken from the user's documents. Each context passage starts with a label "
    "like [source.pdf, page N].\n"
    "Rules:\n"
    "1. Use only the context. If the answer is not in the context, reply exactly: "
    "\"I could not find this in the documents.\"\n"
    "2. Be concise.\n"
    "3. End your answer with the pages you used, in the form: Sources: page N, page M\n\n"
    "Context:\n{context}"
)

REFUSAL_PATTERN = re.compile(
    r"could not find|couldn't find|not (?:mentioned|found|present|available|stated|provided) in the (?:documents?|context)",
    re.IGNORECASE,
)
PAGE_PATTERN = re.compile(r"\bp(?:age)?s?\.?\s*(\d+(?:\s*(?:,|and|&|-|–)\s*(?:p(?:age)?\.?\s*)?\d+)*)", re.IGNORECASE)


# ---------------------------------------------------------------- data types
@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int  # 1-based physical page number in the PDF


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]
    cited_pages: set[int]
    retrieval_seconds: float
    generation_seconds: float
    refused: bool = False
    retrieved_pages: set[int] = field(default_factory=set)

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds

    @property
    def citations_grounded(self) -> bool:
        """True if every page the model cited was actually retrieved (no invented citations)."""
        return self.cited_pages.issubset(self.retrieved_pages)


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------- loading
def load_pdf(data: bytes | str, source_name: str | None = None) -> list[Chunk]:
    """Read a PDF (path or raw bytes) into one Chunk per page. Empty pages are skipped."""
    if isinstance(data, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(data))
        name = source_name or "document.pdf"
    else:
        reader = PdfReader(data)
        name = source_name or str(data).replace("\\", "/").split("/")[-1]
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Chunk(text=text, source=name, page=i))
    return pages


def split_pages(pages: list[Chunk], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping chunks. Chunks never cross page boundaries,
    so every chunk keeps an exact page number for citations."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            chunks.append(Chunk(text=piece, source=page.source, page=page.page))
    return chunks


# ---------------------------------------------------------------- embeddings
class SentenceTransformerEmbedder:
    """Real semantic embeddings (downloads the model on first use)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, show_progress_bar=False), dtype="float32")


class HashingEmbedder:
    """Tiny keyword-based embedder with no downloads. Used in tests and as an offline fallback."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        return out


# ---------------------------------------------------------------- index
class VectorIndex:
    """Cosine-similarity search over chunks using FAISS."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("No text found to index. Is the PDF scanned (image-only)?")
        self.chunks = chunks
        self.embedder = embedder
        vectors = embedder.encode([c.text for c in chunks]).astype("float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(q)
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


def build_index(pages: list[Chunk], embedder: Embedder, chunk_size: int = 1000, chunk_overlap: int = 150) -> VectorIndex:
    return VectorIndex(split_pages(pages, chunk_size, chunk_overlap), embedder)


# ---------------------------------------------------------------- prompting
def format_context(results: list[tuple[Chunk, float]]) -> str:
    return "\n\n---\n\n".join(f"[{c.source}, page {c.page}]\n{c.text}" for c, _ in results)


def build_messages(question: str, results: list[tuple[Chunk, float]]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(results))},
        {"role": "user", "content": question},
    ]


def extract_cited_pages(text: str) -> set[int]:
    """Find page numbers the model cited, e.g. 'page 7', 'pages 3 and 5', 'p. 12', 'pages 4-6'."""
    pages: set[int] = set()
    for match in PAGE_PATTERN.finditer(text):
        group = match.group(1)
        for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", group):
            lo, hi = int(a), int(b)
            if 0 < hi - lo <= 20:
                pages.update(range(lo, hi + 1))
        pages.update(int(n) for n in re.findall(r"\d+", group))
    return pages


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERN.search(text))


# ---------------------------------------------------------------- LLM + answering
LLMFn = Callable[[list[dict]], str]


def groq_llm(api_key: str, model: str = DEFAULT_LLM_MODEL, temperature: float = 0.0) -> LLMFn:
    from groq import Groq

    client = Groq(api_key=api_key)

    def call(messages: list[dict]) -> str:
        resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return resp.choices[0].message.content or ""

    return call


def answer_question(question: str, index: VectorIndex, llm: LLMFn, k: int = 4) -> Answer:
    t0 = time.perf_counter()
    results = index.search(question, k=k)
    t1 = time.perf_counter()
    text = llm(build_messages(question, results))
    t2 = time.perf_counter()
    refused = is_refusal(text)
    return Answer(
        text=text,
        sources=results,
        cited_pages=set() if refused else extract_cited_pages(text),
        retrieval_seconds=t1 - t0,
        generation_seconds=t2 - t1,
        refused=refused,
        retrieved_pages={c.page for c, _ in results},
    )
