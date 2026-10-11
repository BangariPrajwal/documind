"""
DocuMind - chat with your PDFs, with page-level citations, live monitoring
and a published evaluation of how accurate the citations are.

Stack: Streamlit · pypdf · LangChain text splitter · Sentence Transformers · FAISS · Groq (GPT-OSS 20B)
"""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()
RESULTS_FILE = Path(__file__).parent / "eval" / "results.json"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 150, 4


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_embedder():
    return rag.SentenceTransformerEmbedder()


@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return rag.groq_llm(api_key)


st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
st.title("📄 DocuMind")
st.caption("Ask questions about your PDFs. Every answer cites the pages it came from, and every citation is checked.")

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file (local) or Streamlit secrets (deployed).")
    st.stop()

for key, default in {"messages": [], "index": None, "log": []}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary", disabled=not files):
        with st.spinner("Reading and indexing your PDFs..."):
            pages = []
            for f in files:
                pages.extend(rag.load_pdf(f.getvalue(), f.name))
            try:
                st.session_state.index = rag.build_index(pages, load_embedder(), CHUNK_SIZE, CHUNK_OVERLAP)
                st.session_state.messages = []
                st.success(f"Indexed {len(pages)} pages into {len(st.session_state.index.chunks)} chunks.")
            except ValueError as err:
                st.error(str(err))
    if st.session_state.index is not None and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Settings: chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}, top-{TOP_K} retrieval")

chat_tab, monitor_tab, eval_tab = st.tabs(["💬 Chat", "📈 Monitoring", "🧪 Evaluation"])

# ------------------------------------------------------------------ chat
with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    question = st.chat_input("Ask something about your documents...")
    if question:
        if st.session_state.index is None:
            st.warning("Please upload and process at least one PDF first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ans = rag.answer_question(question, st.session_state.index, load_llm(api_key), k=TOP_K)
                    except Exception as err:  # network / rate-limit errors from the LLM provider
                        st.error(f"The language model could not answer right now: {err}")
                        st.stop()
                st.markdown(ans.text)
                if ans.refused:
                    badge = "ℹ️ Not found in the documents"
                elif not ans.cited_pages:
                    badge = "⚠️ No page cited"
                elif ans.citations_grounded:
                    badge = f"✅ Citations verified against retrieved pages · {ans.total_seconds:.1f}s"
                else:
                    bad = sorted(ans.cited_pages - ans.retrieved_pages)
                    badge = f"⚠️ Cited page(s) {bad} were not in the retrieved context"
                st.caption(badge)
                with st.expander("Sources used"):
                    for chunk, score in ans.sources:
                        st.markdown(f"**{chunk.source}, page {chunk.page}** · similarity {score:.2f}")
                        st.caption(chunk.text[:300] + "...")
            st.session_state.messages.append({"role": "assistant", "content": ans.text, "badge": badge})
            st.session_state.log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "retrieval_s": round(ans.retrieval_seconds, 3),
                "generation_s": round(ans.generation_seconds, 3),
                "total_s": round(ans.total_seconds, 3),
                "retrieved_pages": sorted(ans.retrieved_pages),
                "cited_pages": sorted(ans.cited_pages),
                "refused": ans.refused,
                "grounded": ans.citations_grounded if ans.cited_pages else None,
            })

# ------------------------------------------------------------------ monitoring
with monitor_tab:
    log = st.session_state.log
    if not log:
        st.info("Ask a question in the Chat tab and live metrics for this session will appear here.")
    else:
        df = pd.DataFrame(log)
        cited = df[df["grounded"].notna()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", len(df))
        c2.metric("Avg response", f"{df['total_s'].mean():.2f}s")
        c3.metric("p95 response", f"{df['total_s'].quantile(0.95):.2f}s")
        c4.metric("Grounded citations", f"{cited['grounded'].mean() * 100:.0f}%" if len(cited) else "–")
        st.caption("Grounded = every page the model cited was actually in the retrieved context.")
        st.subheader("Latency per question")
        st.bar_chart(df.set_index("time")[["retrieval_s", "generation_s"]], stack=True)
        st.subheader("Query log")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download log (CSV)", df.to_csv(index=False), "documind_log.csv", "text/csv")

# ------------------------------------------------------------------ evaluation
with eval_tab:
    if not RESULTS_FILE.exists():
        st.info("No evaluation results yet. Run `python evaluate.py --pdf <file> --chunk-sizes 500 1000 1500`.")
    else:
        report = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        st.markdown(
            f"Benchmark on **{report['question_count']} labelled questions** (including questions the document "
            f"cannot answer). Model `{report['llm_model']}`, embeddings `{report['embed_model']}`, run {report['run_at']}."
        )
        rows = []
        for r in report["runs"]:
            s = r["summary"]
            rows.append({
                "Chunk size": r["config"]["chunk_size"],
                "Retrieval hit": s["retrieval_hit_rate"],
                "Answer accuracy": s["answer_accuracy"],
                "Citation accuracy": s["citation_accuracy"],
                "Citation grounding": s["citation_grounding"],
                "Refusal accuracy": s["refusal_accuracy"],
                "Avg seconds": s.get("avg_seconds"),
            })
        table = pd.DataFrame(rows)
        st.dataframe(
            table.style.format({c: "{:.0%}" for c in table.columns if c not in ("Chunk size", "Avg seconds")}, na_rep="–"),
            width="stretch", hide_index=True,
        )
        metric_cols = ["Retrieval hit", "Answer accuracy", "Citation accuracy"]
        st.bar_chart(table.set_index("Chunk size")[metric_cols].astype(float))
        with st.expander("What do these metrics mean?"):
            st.markdown(
                "- **Retrieval hit**: the search found a chunk from a page that contains the answer.\n"
                "- **Answer accuracy**: the answer contains the expected fact.\n"
                "- **Citation accuracy**: the model cited a page that actually contains the answer.\n"
                "- **Citation grounding**: every cited page was in the retrieved context (no invented citations).\n"
                "- **Refusal accuracy**: for questions the document cannot answer, the model said so instead of guessing."
            )
