"""Unit tests for DocuMind. They run offline: no API key, no model downloads."""
import sys
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate  # noqa: E402
import rag  # noqa: E402

PAGE_TEXTS = [
    "Introduction. This report studies solar panels installed in Guildford.",
    "Results. The best model reached an accuracy of 0.93 on the test set.",
    "Budget. The project cost 4,500 pounds in total, mostly for sensors.",
]


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in PAGE_TEXTS:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return path


@pytest.fixture
def index(sample_pdf):
    return rag.build_index(rag.load_pdf(str(sample_pdf)), rag.HashingEmbedder(), chunk_size=200, chunk_overlap=20)


def fake_llm(reply):
    return lambda messages: reply


# ------------------------------------------------------------- loading & chunking
def test_load_pdf_keeps_page_numbers(sample_pdf):
    pages = rag.load_pdf(str(sample_pdf))
    assert [p.page for p in pages] == [1, 2, 3]
    assert "0.93" in pages[1].text
    assert pages[0].source == "report.pdf"


def test_load_pdf_from_bytes(sample_pdf):
    pages = rag.load_pdf(sample_pdf.read_bytes(), "upload.pdf")
    assert len(pages) == 3 and pages[0].source == "upload.pdf"


def test_chunks_never_cross_pages():
    pages = [rag.Chunk("word " * 300, "a.pdf", 1), rag.Chunk("other " * 300, "a.pdf", 2)]
    chunks = rag.split_pages(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 2
    assert all(len(c.text) <= 200 for c in chunks)
    assert {c.page for c in chunks if "word" in c.text} == {1}
    assert {c.page for c in chunks if "other" in c.text} == {2}


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        rag.split_pages([rag.Chunk("x", "a.pdf", 1)], chunk_size=100, chunk_overlap=100)


def test_empty_index_gives_clear_error():
    with pytest.raises(ValueError, match="No text"):
        rag.VectorIndex([], rag.HashingEmbedder())


# ------------------------------------------------------------- retrieval
def test_search_finds_the_right_page(index):
    top_chunk, _ = index.search("What accuracy did the best model reach?", k=1)[0]
    assert top_chunk.page == 2


def test_search_never_returns_more_than_available(index):
    assert len(index.search("anything", k=50)) == len(index.chunks)


# ------------------------------------------------------------- citations
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Sources: page 7", {7}),
        ("See page 3 and page 12.", {3, 12}),
        ("Sources: pages 4, 5 and 9", {4, 5, 9}),
        ("[report.pdf, page 2]", {2}),
        ("p. 14", {14}),
        ("pages 4-6", {4, 5, 6}),
        ("The model scored 0.93 with p = 0.82", set()),
        ("No citation here.", set()),
    ],
)
def test_extract_cited_pages(text, expected):
    assert rag.extract_cited_pages(text) == expected


def test_prompt_examples_contain_no_real_page_numbers():
    # A concrete example like "page 7" in the instructions can be copied by the model as a fake citation.
    assert rag.extract_cited_pages(rag.SYSTEM_PROMPT) == set()


def test_refusal_detection():
    assert rag.is_refusal("I could not find this in the documents.")
    assert rag.is_refusal("That is not mentioned in the documents.")
    assert not rag.is_refusal("The accuracy was 0.93. Sources: page 2")


def test_prompt_contains_page_labels(index):
    results = index.search("accuracy", k=2)
    messages = rag.build_messages("What is the accuracy?", results)
    assert messages[0]["role"] == "system" and messages[1]["content"] == "What is the accuracy?"
    assert "page 2]" in messages[0]["content"]


# ------------------------------------------------------------- answering
def test_grounded_answer(index):
    ans = rag.answer_question("What accuracy did the best model reach?", index,
                              fake_llm("It reached 0.93. Sources: page 2"), k=1)
    assert ans.cited_pages == {2} and ans.citations_grounded and not ans.refused
    assert ans.total_seconds >= 0


def test_invented_citation_is_flagged(index):
    ans = rag.answer_question("What accuracy did the best model reach?", index,
                              fake_llm("It reached 0.93. Sources: page 99"), k=1)
    assert not ans.citations_grounded


def test_refusal_has_no_citations(index):
    ans = rag.answer_question("Who is the mayor?", index, fake_llm("I could not find this in the documents."), k=1)
    assert ans.refused and ans.cited_pages == set()


# ------------------------------------------------------------- evaluation scoring
def test_number_normalisation():
    assert evaluate.answer_is_correct("There were 284,807 rows", {"must_contain": ["284807"]})
    assert not evaluate.answer_is_correct("About 280k rows", {"must_contain": ["284807"]})
    assert evaluate.answer_is_correct("an 80/20 split", {"must_contain_any": ["80/20", "80-20"]})


def test_score_answerable_question(index):
    q = {"id": "t1", "question": "accuracy?", "gold_pages": [2], "must_contain": ["0.93"]}
    ans = rag.answer_question(q["question"], index, fake_llm("0.93. Sources: page 2"), k=1)
    row = evaluate.score_question(q, ans.retrieved_pages, ans)
    assert row["retrieval_hit"] and row["answer_correct"] and row["citation_correct"] and row["citation_grounded"]


def test_score_unanswerable_question(index):
    q = {"id": "u1", "question": "Who is the mayor?", "answerable": False}
    ans = rag.answer_question(q["question"], index, fake_llm("I could not find this in the documents."), k=1)
    row = evaluate.score_question(q, ans.retrieved_pages, ans)
    assert row["refusal_correct"] and row["retrieval_hit"] is None


def test_summary_rates():
    rows = [
        {"retrieval_hit": True, "answer_correct": True, "seconds": 1.0},
        {"retrieval_hit": False, "answer_correct": False, "seconds": 3.0},
        {"retrieval_hit": None, "refusal_correct": True, "seconds": 2.0},
    ]
    s = evaluate.summarise(rows)
    assert s["retrieval_hit_rate"] == 0.5 and s["answer_accuracy"] == 0.5
    assert s["refusal_accuracy"] == 1.0 and s["avg_seconds"] == 2.0


def test_question_set_is_well_formed():
    import json

    data = json.loads((Path(__file__).resolve().parents[1] / "eval" / "questions.json").read_text())
    ids = [q["id"] for q in data["questions"]]
    assert len(ids) == len(set(ids))
    for q in data["questions"]:
        if q.get("answerable", True):
            assert q["gold_pages"], q["id"]
            assert q.get("must_contain") or q.get("must_contain_any"), q["id"]
