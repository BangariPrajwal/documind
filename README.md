# 📄 DocuMind: Chat with your PDFs, with citations you can trust

![tests](https://github.com/BangariPrajwal/documind/actions/workflows/tests.yml/badge.svg)

DocuMind is a Retrieval-Augmented Generation (RAG) app: upload PDFs, ask questions in plain English, and get answers grounded only in your documents, with the exact page each answer came from.

Most RAG demos stop at "it answers questions". DocuMind also **checks its own citations on every answer** and ships with a **labelled benchmark** that measures how often it retrieves the right page, answers correctly, cites the right page, and admits when the answer is not in the document.

**Live demo:** https://documind-prajwal.streamlit.app

## Features

- **Page-level citations.** Chunks never cross page boundaries, so every passage keeps an exact page number.
- **Citation checking.** Each answer is labelled ✅ when every cited page was actually retrieved, or ⚠️ when the model cites a page it was never shown (an invented citation).
- **Honest refusals.** If the answer is not in the documents, the model is instructed to say so instead of guessing, and this is measured.
- **Live monitoring tab.** Response time (retrieval vs generation), p95 latency, grounded-citation rate and a downloadable query log for the session.
- **Evaluation tab.** Benchmark results across chunk sizes, shown inside the app.
- **Automated tests.** 26 unit tests run on every push via GitHub Actions, with no API key or model download needed.

## How it works

1. **Load:** PDFs are read page by page with `pypdf`.
2. **Chunk:** each page is split into overlapping chunks (LangChain `RecursiveCharacterTextSplitter`), keeping the page number.
3. **Embed:** chunks become vectors with the `all-MiniLM-L6-v2` sentence-transformer.
4. **Index:** vectors go into a FAISS index (cosine similarity).
5. **Retrieve:** the top 4 chunks for each question are fetched.
6. **Generate:** GPT-OSS 20B (via Groq) answers from those chunks only and lists the pages it used.
7. **Verify:** cited pages are extracted from the answer and compared with the retrieved pages.

```
PDF → pages → chunks → embeddings → FAISS → top-k chunks → LLM → answer + cited pages → grounding check
```

## Evaluation

The benchmark (`eval/questions.json`) has 20 questions about an 82-page research report: 17 answerable questions with the correct page numbers labelled, and 3 questions the document cannot answer.

| Metric | Meaning |
|---|---|
| Retrieval hit | The search found a chunk from a page that contains the answer |
| Answer accuracy | The answer contains the expected fact |
| Citation accuracy | The model cited a page that contains the answer |
| Citation grounding | Every cited page was in the retrieved context |
| Refusal accuracy | For unanswerable questions, the model said so instead of guessing |

**Results:** see [`eval/RESULTS.md`](eval/RESULTS.md).

Reproduce:

```bash
python evaluate.py --pdf your_report.pdf --chunk-sizes 500 1000 1500
python evaluate.py --pdf your_report.pdf --retrieval-only   # no API key needed
```

## Project structure

```
app.py              Streamlit UI: chat, monitoring and evaluation tabs
rag.py              Core logic: loading, chunking, embeddings, FAISS, prompting, citation checks
evaluate.py         Benchmark runner and scoring
eval/questions.json Labelled question set
tests/test_rag.py   Unit tests (offline)
.github/workflows/  Runs the tests on every push
```

## Run locally

```bash
git clone https://github.com/BangariPrajwal/documind.git
cd documind
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then paste your Groq key into .env
streamlit run app.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest -v
```

## Tech stack

Python · Streamlit · pypdf · LangChain text splitters · Sentence Transformers · FAISS · Groq (GPT-OSS 20B) · pytest · GitHub Actions

## Limitations and next steps

- The benchmark is small (20 questions on one document); a larger, multi-document set would give tighter estimates.
- Answer accuracy uses keyword matching, which can miss correct answers phrased differently.
- Monitoring is per session; a production version would store logs persistently.
- Next: hybrid (keyword + semantic) search and a re-ranker, compared using the same benchmark.
