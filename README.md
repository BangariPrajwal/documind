# 📄 DocuMind — Chat with your PDFs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them in plain English. Answers are grounded only in your documents and come with source file and page references, which reduces hallucinations.

**Live demo:** [add your Streamlit link here]

## How it works

1. **Load:** PDFs are read page by page with PyPDF.
2. **Chunk:** Text is split into 1,000-character chunks with 150-character overlap (LangChain `RecursiveCharacterTextSplitter`).
3. **Embed:** Each chunk is converted into a vector using the `all-MiniLM-L6-v2` sentence-transformer.
4. **Index:** Vectors are stored in a FAISS vector index for fast semantic search.
5. **Retrieve:** For each question, the top 4 most similar chunks are retrieved.
6. **Generate:** GPT-OSS 20B (via Groq) answers using only the retrieved context and cites the source pages.

## Tech stack

Python · Streamlit · LangChain · FAISS · Sentence Transformers · Groq (GPT-OSS 20B) · PyPDF

## Run locally

```bash
git clone https://github.com/[username]/documind.git
cd documind
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
# create a .env file containing: GROQ_API_KEY=your_key
streamlit run app.py
```

## Future improvements

- Conversation memory for follow-up questions
- Hybrid search (keyword + semantic) for better retrieval
- Support for Word and web-page documents
