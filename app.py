"""
DocuMind - Chat with your PDFs (RAG chatbot)
Stack: Streamlit + LangChain + FAISS + HuggingFace embeddings + Groq (GPT-OSS 20B)
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Config ----------
load_dotenv()  # reads GROQ_API_KEY from a local .env file


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:  # when deployed on Streamlit Cloud, the key lives in st.secrets
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


LLM_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are DocuMind, a helpful assistant that answers questions ONLY using the "
     "context from the user's documents below. If the answer is not in the context, "
     "say you could not find it in the documents - do not make things up. "
     "Keep answers clear and concise, and mention the source file and page you used.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


# ---------- RAG building blocks ----------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vector_store(uploaded_files):
    """PDF -> pages -> chunks -> embeddings -> FAISS index."""
    documents = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        for page in pages:
            page.metadata["source"] = uploaded.name
        documents.extend(pages)
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, load_embeddings())
    return store, len(documents), len(chunks)


def format_context(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page", 0) + 1  # PyPDF pages start at 0
        parts.append(f"[{d.metadata.get('source', 'document')}, page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question, store, api_key):
    docs = store.similarity_search(question, k=TOP_K)
    llm = ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0)
    messages = PROMPT.format_messages(context=format_context(docs), question=question)
    response = llm.invoke(messages)
    return response.content, docs


# ---------- UI ----------
st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind")
st.caption("Upload PDFs and ask questions. Answers come only from your documents, with sources.")

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file (local) or Streamlit secrets (deployed).")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not files):
        with st.spinner("Reading and indexing your PDFs..."):
            store, n_pages, n_chunks = build_vector_store(files)
            st.session_state.store = store
            st.session_state.messages = []
        st.success(f"Indexed {n_pages} pages into {n_chunks} chunks.")
    if st.session_state.store is not None and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    if st.session_state.store is None:
        st.warning("Please upload and process at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(question, st.session_state.store, api_key)
            st.markdown(answer)
            with st.expander("Sources used"):
                for d in sources:
                    page = d.metadata.get("page", 0) + 1
                    st.markdown(f"**{d.metadata.get('source')} (page {page})**")
                    st.caption(d.page_content[:300] + "...")
        st.session_state.messages.append({"role": "assistant", "content": answer})
