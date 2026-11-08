"""Local retrieval, mock grounded answers, and document-level evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
PERSIST_DIR = ROOT / ".chroma"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FIXED_COLLECTION = "jobapps_fixed_chunks"
SENTENCE_COLLECTION = "jobapps_sentence_chunks"
FIXED_CHUNK_SIZE = 320
FIXED_OVERLAP = 60
SENTENCES_PER_CHUNK = 2
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class QueryCase:
    query: str
    relevant_doc_ids: frozenset[str]


EVALUATION_QUERIES = (
    QueryCase(
        "How long do I have to respond to a written offer and discuss pay?",
        frozenset({"offer-negotiation"}),
    ),
    QueryCase(
        "When can I ask to reschedule an interview, and how do I confirm its time zone?",
        frozenset({"interview-scheduling"}),
    ),
    QueryCase(
        "What consent and checks are needed for a candidate background check?",
        frozenset({"background-verification"}),
    ),
    QueryCase(
        "How long is probation, and when will my manager review progress?",
        frozenset({"probation-period"}),
    ),
    QueryCase(
        "How long are unsuccessful applications kept, and can I request deletion?",
        frozenset({"applicant-data-retention"}),
    ),
)
CALIBRATION_IN_SCOPE = tuple(case.query for case in EVALUATION_QUERIES)
CALIBRATION_OUT_OF_SCOPE = (
    "How do I prepare sourdough bread at home?",
    "What is the boiling point of pure water at sea level?",
)


def load_documents(directory: Path = DOCS_DIR) -> list[Document]:
    documents = [
        Document(path.stem, path.read_text(encoding="utf-8").strip())
        for path in sorted(directory.glob("*.md"))
    ]
    if not documents:
        raise FileNotFoundError(f"No Markdown knowledge-base documents in {directory}")
    return documents


def fixed_size_chunks(
    documents: Iterable[Document],
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_OVERLAP,
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks: list[Chunk] = []
    for document in documents:
        start = 0
        while start < len(document.text):
            end = min(start + chunk_size, len(document.text))
            text = document.text[start:end].strip()
            if text:
                chunks.append(
                    Chunk(f"{document.doc_id}-fixed-{start}", document.doc_id, text)
                )
            if end == len(document.text):
                break
            start = end - overlap
    return chunks


def sentence_chunks(
    documents: Iterable[Document],
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
) -> list[Chunk]:
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be positive.")

    chunks: list[Chunk] = []
    for document in documents:
        body = "\n".join(
            line for line in document.text.splitlines() if not line.lstrip().startswith("#")
        )
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", body.strip())
            if sentence.strip()
        ]
        for index in range(0, len(sentences), sentences_per_chunk):
            text = " ".join(sentences[index : index + sentences_per_chunk])
            chunks.append(
                Chunk(
                    f"{document.doc_id}-sentence-{index // sentences_per_chunk}",
                    document.doc_id,
                    text,
                )
            )
    return chunks


def load_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return SentenceTransformer(MODEL_NAME)


def _chroma_client(persist_directory: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return chromadb.PersistentClient(path=str(persist_directory))


def upsert_collection(
    client: Any,
    model: Any,
    name: str,
    chunks: list[Chunk],
) -> Any:
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"doc_id": chunk.doc_id} for chunk in chunks],
        embeddings=embeddings.tolist(),
    )
    return collection


def build_collections(
    persist_directory: Path = PERSIST_DIR,
) -> tuple[Any, Any, Any]:
    documents = load_documents()
    model = load_embedding_model()
    client = _chroma_client(persist_directory)
    fixed = upsert_collection(
        client, model, FIXED_COLLECTION, fixed_size_chunks(documents)
    )
    sentence = upsert_collection(
        client, model, SENTENCE_COLLECTION, sentence_chunks(documents)
    )
    return model, fixed, sentence


def retrieve(
    model: Any, collection: Any, query: str, top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "doc_id": metadata["doc_id"],
            "similarity": 1.0 - float(distance),
        }
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def answer_query(
    model: Any,
    collection: Any,
    query: str,
    threshold: float,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    hits = retrieve(model, collection, query, top_k)
    if not hits or hits[0]["similarity"] < threshold:
        return {
            "answer": "I don't know based on the available job-application documents.",
            "fallback": True,
            "hits": hits,
        }

    # This mock generator is extractive: its answer contains only retrieved text.
    answer = "\n".join(
        f"[{hit['doc_id']}] {hit['text']}" for hit in hits
    )
    return {"answer": answer, "fallback": False, "hits": hits}


def calibrate_threshold(
    model: Any,
    collection: Any,
    in_scope_queries: Iterable[str] = CALIBRATION_IN_SCOPE,
    out_of_scope_queries: Iterable[str] = CALIBRATION_OUT_OF_SCOPE,
) -> tuple[float, list[float], list[float]]:
    in_scope_scores = [
        retrieve(model, collection, query, top_k=1)[0]["similarity"]
        for query in in_scope_queries
    ]
    out_of_scope_scores = [
        retrieve(model, collection, query, top_k=1)[0]["similarity"]
        for query in out_of_scope_queries
    ]
    lowest_in_scope = min(in_scope_scores)
    highest_out_of_scope = max(out_of_scope_scores)
    if highest_out_of_scope >= lowest_in_scope:
        raise ValueError(
            "Calibration clusters overlap: "
            f"max out-of-scope={highest_out_of_scope:.4f}, "
            f"min in-scope={lowest_in_scope:.4f}. "
            "Choose better representative calibration queries and measure again."
        )
    return (highest_out_of_scope + lowest_in_scope) / 2, in_scope_scores, out_of_scope_scores


def evaluate_collection(
    model: Any, collection: Any, top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:
    rows = []
    for case in EVALUATION_QUERIES:
        hits = retrieve(model, collection, case.query, top_k)
        retrieved_docs = list(dict.fromkeys(hit["doc_id"] for hit in hits))
        relevant = set(case.relevant_doc_ids)
        true_positives = len(set(retrieved_docs) & relevant)
        precision = true_positives / len(retrieved_docs) if retrieved_docs else 0.0
        recall = true_positives / len(relevant) if relevant else 0.0
        rows.append(
            {
                "query": case.query,
                "relevant": sorted(relevant),
                "retrieved": retrieved_docs,
                "tp": true_positives,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def print_evaluation(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{label} document-level evaluation:")
    for row in rows:
        print(
            f"  {row['query']}\n"
            f"    relevant={row['relevant']}; retrieved={row['retrieved']}; "
            f"P={row['tp']}/{len(row['retrieved'])}={row['precision']:.3f}; "
            f"R={row['tp']}/{len(row['relevant'])}={row['recall']:.3f}"
        )
    mean_precision = sum(row["precision"] for row in rows) / len(rows)
    mean_recall = sum(row["recall"] for row in rows) / len(rows)
    print(f"  Macro-average precision={mean_precision:.3f}, recall={mean_recall:.3f}")


def run_demo(
    persist_directory: Path = PERSIST_DIR, top_k: int = DEFAULT_TOP_K
) -> None:
    model, fixed, sentence = build_collections(persist_directory)
    fixed_threshold, fixed_in, fixed_out = calibrate_threshold(model, fixed)
    sentence_threshold, sentence_in, sentence_out = calibrate_threshold(
        model, sentence
    )
    for label, threshold, in_scores, out_scores in (
        ("Fixed-size-with-overlap", fixed_threshold, fixed_in, fixed_out),
        ("Sentence-based", sentence_threshold, sentence_in, sentence_out),
    ):
        print(f"{label} collection calibration:")
        print(
            "  In-scope top-1 similarities:",
            json.dumps([round(score, 4) for score in in_scores]),
        )
        print(
            "  Out-of-scope top-1 similarities:",
            json.dumps([round(score, 4) for score in out_scores]),
        )
        print(f"  Empirical threshold: {threshold:.4f}")
    print("\nGrounded answer demonstrations (sentence collection):")
    for query in CALIBRATION_IN_SCOPE[:5] + CALIBRATION_OUT_OF_SCOPE[:1]:
        result = answer_query(model, sentence, query, sentence_threshold, top_k)
        print(
            f"  Query: {query}\n"
            f"  Top-1 similarity: {result['hits'][0]['similarity']:.4f}\n"
            f"  Fallback: {result['fallback']}\n"
            f"  Answer: {result['answer']}"
        )
    print_evaluation(
        "Fixed-size-with-overlap", evaluate_collection(model, fixed, top_k)
    )
    print_evaluation(
        "Sentence-based", evaluate_collection(model, sentence, top_k)
    )


def run_query(
    query: str,
    strategy: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: Path = PERSIST_DIR,
) -> None:
    model, fixed, sentence = build_collections(persist_directory)
    collection = fixed if strategy == "fixed" else sentence
    threshold, in_scores, out_scores = calibrate_threshold(model, collection)
    print(f"Strategy: {strategy}")
    print(f"In-scope top-1 similarities: {json.dumps(in_scores)}")
    print(f"Out-of-scope top-1 similarities: {json.dumps(out_scores)}")
    print(f"Empirical threshold: {threshold:.4f}")
    result = answer_query(model, collection, query, threshold, top_k)
    print(f"Top-1 similarity: {result['hits'][0]['similarity']:.4f}")
    print(f"Fallback: {result['fallback']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=PERSIST_DIR,
        help="Directory for the persistent ChromaDB collections.",
    )
    parser.add_argument(
        "--query",
        help="Answer one query instead of running the calibration/evaluation demo.",
    )
    parser.add_argument(
        "--strategy",
        choices=("fixed", "sentence"),
        default="sentence",
        help="Collection to query when --query is supplied.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.query:
        run_query(args.query, args.strategy, args.top_k, args.persist_directory)
    else:
        run_demo(args.persist_directory, args.top_k)
