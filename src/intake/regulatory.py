from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .settings import load_env_file

RETRIEVAL_WORKERS = 6  # concurrent Pinecone queries per retrieval batch


CORPUS_VERSION = "regulatory-v1-2026-11-10"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536
PINECONE_API_VERSION = "2025-10"
PINECONE_NAMESPACE = "model-intake-brief-regulatory"


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    document_version: str
    official_url: str
    provision: str
    text: str | None
    status: str = "verified"
    extraction_source: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class IndexStatus:
    configured: bool
    index_name: str | None
    namespace: str
    index_dimension: int | None
    embedding_dimension: int
    compatible: bool
    needed: list[str]
    blockers: list[str]


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    provision: str
    official_url: str
    score: float
    document_version: str
    excerpt: str


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    item_id: str | None
    hits: list[RetrievalHit]
    blockers: list[str]


def load_rag_config_from_env() -> dict[str, str]:
    env_values = load_env_file(Path('.env'))
    merged = {**env_values, **os.environ}
    keys = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_NAMESPACE",
        "PINECONE_API_VERSION",
    ]
    return {key: merged[key] for key in keys if merged.get(key)}


def build_regulatory_corpus() -> list[CorpusChunk]:
    transparency_page = "https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai"
    transparency_pdf = "https://www.editionmultimedia.fr/wp-content/uploads/2025/07/Code-of-Practice-for-GPAI-Transparency-10-07-25.pdf"
    act_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ%3AL_202401689"
    act_version = "Regulation (EU) 2024/1689, OJ L, 12.7.2024"
    transparency_version = "GPAI Code of Practice Transparency chapter, published 2025-07-10"

    chunks: list[CorpusChunk] = [
        CorpusChunk(
            chunk_id="ai-act-art53-1a",
            document_version=act_version,
            official_url=act_url,
            provision="Article 53(1)(a)",
            text=(
                "Providers of general-purpose AI models must draw up and keep up to date technical documentation "
                "for the model, including its training and testing process and evaluation results, with the minimum "
                "information set out in Annex XI for the AI Office and national competent authorities."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-art53-1b",
            document_version=act_version,
            official_url=act_url,
            provision="Article 53(1)(b)",
            text=(
                "Providers must also prepare and keep up to date information and documentation for AI system providers "
                "who integrate the model, so they can understand capabilities and limitations and meet their own obligations; "
                "the documentation must contain at least the elements in Annex XII."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-art53-1d",
            document_version=act_version,
            official_url=act_url,
            provision="Article 53(1)(d)",
            text=(
                "Providers must draw up and publicly make available a sufficiently detailed summary of the content used "
                "for training the general-purpose AI model using the AI Office template."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xi-1c",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XI, Section 1, point 1(c)",
            text=(
                "The general description must identify the model's date of release and methods of distribution."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xi-2e",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XI, Section 1, point 2(e)",
            text=(
                "Providers must document the known or estimated energy consumption of the model; if it is unknown, the "
                "estimate may rely on the computational resources used."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xii-1a",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XII, point 1(a)",
            text=(
                "The downstream-provider documentation must include the tasks the model is intended to perform and the "
                "type and nature of AI systems into which it can be integrated."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xii-1h",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XII, point 1(h)",
            text="The downstream-provider documentation must state the licence for the model.",
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xii-2a",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XII, point 2(a)",
            text=(
                "The documentation must describe the technical means required for the model to be integrated into AI systems."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xii-2b",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XII, point 2(b)",
            text=(
                "The documentation must include the modality and format of inputs and outputs and their maximum size, "
                "such as context window length."
            ),
        ),
        CorpusChunk(
            chunk_id="ai-act-annex-xii-2c",
            document_version=act_version,
            official_url=act_url,
            provision="Annex XII, point 2(c)",
            text=(
                "The documentation must cover the data used for training, testing and validation, including type, provenance "
                "and curation methodology."
            ),
        ),
        CorpusChunk(
            chunk_id="cop-transparency-m1-1",
            document_version=transparency_version,
            official_url=transparency_page,
            provision="Transparency chapter, Measure 1.1",
            extraction_source=transparency_pdf,
            text=(
                "Signatories must draw up and keep up to date model documentation when they place a general-purpose AI model on the market, "
                "documenting at least the information in the Model Documentation Form and retaining earlier versions for ten years."
            ),
            notes="Verified from the readable mirror of the official Transparency chapter PDF.",
        ),
        CorpusChunk(
            chunk_id="cop-transparency-m1-2",
            document_version=transparency_version,
            official_url=transparency_page,
            provision="Transparency chapter, Measure 1.2",
            extraction_source=transparency_pdf,
            text=(
                "Signatories must publicly disclose contact information for the AI Office and downstream providers to request access to relevant information, "
                "provide up-to-date information on request from the AI Office, and give downstream providers the documentation they need to understand capabilities, limitations, and integration."
            ),
            notes="Verified from the readable mirror of the official Transparency chapter PDF.",
        ),
        CorpusChunk(
            chunk_id="ai-act-training-summary-template",
            document_version="Commission template page, last update 2026-11-10",
            official_url="https://digital-strategy.ec.europa.eu/en/faqs/template-general-purpose-ai-model-providers-summarise-their-training-content",
            provision="Article 53(1)(d) support",
            text=(
                "The Commission template and explanatory notice facilitate compliance with the public summary of training content "
                "requirement under Article 53(1)(d)."
            ),
        ),
    ]
    return chunks


def corpus_manifest() -> dict[str, Any]:
    chunks = build_regulatory_corpus()
    return {
        "corpus_version": CORPUS_VERSION,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": OPENAI_EMBEDDING_DIMENSION,
        "namespace": PINECONE_NAMESPACE,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_version": chunk.document_version,
                "official_url": chunk.official_url,
                "provision": chunk.provision,
                "status": chunk.status,
                "extraction_source": chunk.extraction_source,
                "notes": chunk.notes,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


def _extract_openai_embedding(response_json: dict[str, Any]) -> list[float]:
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("OpenAI embeddings response did not contain any vectors.")
    first = data[0]
    if not isinstance(first, dict) or "embedding" not in first:
        raise ValueError("OpenAI embeddings response was malformed.")
    embedding = first["embedding"]
    if not isinstance(embedding, list):
        raise ValueError("OpenAI embeddings response embedding was malformed.")
    return [float(value) for value in embedding]


def embed_texts(texts: list[str], api_key: str, timeout_seconds: float) -> list[list[float]]:
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": OPENAI_EMBEDDING_MODEL, "input": texts},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenAI embeddings response did not contain a data list.")
    embeddings: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("OpenAI embeddings response contained an invalid item.")
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("OpenAI embeddings response contained an invalid embedding.")
        embeddings.append([float(value) for value in embedding])
    if len(embeddings) != len(texts):
        raise ValueError("OpenAI embeddings response count did not match the request.")
    return embeddings


def _pinecone_headers(api_key: str, api_version: str | None = None) -> dict[str, str]:
    headers = {"Api-Key": api_key, "Accept": "application/json"}
    headers["X-Pinecone-Api-Version"] = api_version or PINECONE_API_VERSION
    return headers


def describe_index(index_name: str, api_key: str, api_version: str | None = None, timeout_seconds: float = 15.0) -> dict[str, Any]:
    response = requests.get(
        f"https://api.pinecone.io/indexes/{index_name}",
        headers=_pinecone_headers(api_key, api_version),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Pinecone describe index response was malformed.")
    return payload


def resolve_index_status(config: dict[str, str], timeout_seconds: float = 15.0) -> IndexStatus:
    needed: list[str] = []
    blockers: list[str] = []
    index_name = config.get("PINECONE_INDEX_NAME")
    namespace = config.get("PINECONE_NAMESPACE") or PINECONE_NAMESPACE
    api_key = config.get("PINECONE_API_KEY")

    if not api_key:
        needed.append("PINECONE_API_KEY")
    if not index_name:
        needed.append("PINECONE_INDEX_NAME")
    if not namespace:
        needed.append("PINECONE_NAMESPACE")

    if needed:
        return IndexStatus(False, index_name, namespace, None, OPENAI_EMBEDDING_DIMENSION, False, needed, blockers)

    try:
        index_info = describe_index(index_name, api_key, config.get("PINECONE_API_VERSION"), timeout_seconds)
    except requests.RequestException as exc:
        blockers.append(f"Pinecone describe index request failed: {exc.__class__.__name__}.")
        return IndexStatus(True, index_name, namespace, None, OPENAI_EMBEDDING_DIMENSION, False, needed, blockers)

    index_dimension = index_info.get("dimension")
    if not isinstance(index_dimension, int):
        blockers.append("Pinecone index did not report a numeric dimension.")
        return IndexStatus(True, index_name, namespace, None, OPENAI_EMBEDDING_DIMENSION, False, needed, blockers)

    compatible = index_dimension == OPENAI_EMBEDDING_DIMENSION
    if not compatible:
        needed.append(
            f"An existing Pinecone index with dimension {OPENAI_EMBEDDING_DIMENSION} and metric cosine (current dimension: {index_dimension})."
        )
    return IndexStatus(True, index_name, namespace, index_dimension, OPENAI_EMBEDDING_DIMENSION, compatible, needed, blockers)


def _pinecone_upsert_url(index_info: dict[str, Any]) -> str:
    host = index_info.get("host")
    if not isinstance(host, str) or not host:
        raise ValueError("Pinecone index host is missing.")
    return f"https://{host}/vectors/upsert"


def _pinecone_query_url(index_info: dict[str, Any]) -> str:
    host = index_info.get("host")
    if not isinstance(host, str) or not host:
        raise ValueError("Pinecone index host is missing.")
    return f"https://{host}/query"


def index_regulatory_corpus(config: dict[str, str], timeout_seconds: float = 15.0) -> dict[str, Any]:
    corpus = [chunk for chunk in build_regulatory_corpus() if chunk.text]
    status = resolve_index_status(config, timeout_seconds=timeout_seconds)
    report: dict[str, Any] = {
        "corpus_version": CORPUS_VERSION,
        "indexed": 0,
        "skipped_blocked": sum(1 for chunk in build_regulatory_corpus() if chunk.status != "verified"),
        "needed": status.needed,
        "blockers": status.blockers,
        "compatible": status.compatible,
        "index_name": status.index_name,
        "namespace": status.namespace,
        "index_dimension": status.index_dimension,
        "embedding_dimension": status.embedding_dimension,
    }

    if not status.configured or not status.compatible or status.blockers:
        return report

    api_key = config["PINECONE_API_KEY"]
    openai_key = config.get("OPENAI_API_KEY")
    if not openai_key:
        report["needed"].append("OPENAI_API_KEY")
        return report

    index_info = describe_index(status.index_name, api_key, config.get("PINECONE_API_VERSION"), timeout_seconds)
    embeddings = embed_texts([chunk.text or "" for chunk in corpus], openai_key, timeout_seconds)
    vectors = []
    for chunk, embedding in zip(corpus, embeddings, strict=True):
        metadata = {
            "chunk_id": chunk.chunk_id,
            "document_version": chunk.document_version,
            "official_url": chunk.official_url,
            "provision": chunk.provision,
            "corpus_version": CORPUS_VERSION,
        }
        if chunk.extraction_source:
            metadata["extraction_source"] = chunk.extraction_source
        vectors.append(
            {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": metadata,
            }
        )

    response = requests.post(
        _pinecone_upsert_url(index_info),
        headers=_pinecone_headers(api_key, config.get("PINECONE_API_VERSION")),
        json={"namespace": status.namespace, "vectors": vectors},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    report["indexed"] = len(vectors)
    return report


def _make_query_vector(query: str, api_key: str, timeout_seconds: float) -> list[float]:
    return embed_texts([query], api_key, timeout_seconds)[0]


def _query_regulatory_hits(
    index_info: dict[str, Any],
    namespace: str,
    api_key: str,
    api_version: str | None,
    query_vector: list[float],
    timeout_seconds: float,
    top_k: int,
) -> tuple[list[RetrievalHit], list[str]]:
    response = requests.post(
        _pinecone_query_url(index_info),
        headers=_pinecone_headers(api_key, api_version),
        json={
            "namespace": namespace,
            "vector": query_vector,
            "topK": top_k,
            "includeMetadata": True,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return [], ["Pinecone query response did not contain matches."]

    hits: list[RetrievalHit] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        metadata = match.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = metadata.get("chunk_id") or match.get("id") or "unknown"
        provision = metadata.get("provision") or "unknown"
        official_url = metadata.get("official_url") or ""
        document_version = metadata.get("document_version") or ""
        score = float(match.get("score") or 0.0)
        excerpt = metadata.get("excerpt") or ""
        if not excerpt:
            excerpt = next((chunk.text for chunk in build_regulatory_corpus() if chunk.chunk_id == chunk_id), "") or ""
        hits.append(
            RetrievalHit(
                chunk_id=str(chunk_id),
                provision=str(provision),
                official_url=str(official_url),
                score=score,
                document_version=str(document_version),
                excerpt=str(excerpt),
            )
        )
    return hits, []


def retrieve_regulatory_context(
    query: str,
    config: dict[str, str],
    item_id: str | None = None,
    timeout_seconds: float = 15.0,
    top_k: int = 3,
) -> RetrievalResult:
    status = resolve_index_status(config, timeout_seconds=timeout_seconds)
    blockers = list(status.blockers)
    if not status.configured:
        blockers.append("Pinecone configuration is incomplete.")
        return RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers)
    if not status.compatible:
        blockers.append("Configured Pinecone index dimension is incompatible with text-embedding-3-small.")
        return RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers)

    openai_key = config.get("OPENAI_API_KEY")
    if not openai_key:
        blockers.append("OPENAI_API_KEY is required for retrieval embeddings.")
        return RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers)

    api_key = config["PINECONE_API_KEY"]
    index_info = describe_index(status.index_name, api_key, config.get("PINECONE_API_VERSION"), timeout_seconds)
    query_vector = _make_query_vector(query, openai_key, timeout_seconds)
    hits, query_blockers = _query_regulatory_hits(
        index_info,
        status.namespace,
        api_key,
        config.get("PINECONE_API_VERSION"),
        query_vector,
        timeout_seconds,
        top_k,
    )
    return RetrievalResult(query=query, item_id=item_id, hits=hits, blockers=blockers + query_blockers)


def retrieve_regulatory_context_batch(
    item_ids: list[str | None],
    config: dict[str, str],
    timeout_seconds: float = 15.0,
    top_k: int = 3,
) -> dict[str | None, RetrievalResult]:
    status = resolve_index_status(config, timeout_seconds=timeout_seconds)
    blockers = list(status.blockers)
    results: dict[str | None, RetrievalResult] = {}
    for item_id in item_ids:
        query = default_query_for_item(item_id)
        if not status.configured:
            results[item_id] = RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers + ["Pinecone configuration is incomplete."])
            continue
        if not status.compatible:
            results[item_id] = RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers + ["Configured Pinecone index dimension is incompatible with text-embedding-3-small."])
            continue

    openai_key = config.get("OPENAI_API_KEY")
    if not openai_key:
        for item_id in item_ids:
            query = default_query_for_item(item_id)
            results[item_id] = RetrievalResult(query=query, item_id=item_id, hits=[], blockers=blockers + ["OPENAI_API_KEY is required for retrieval embeddings."])
        return results

    api_key = config["PINECONE_API_KEY"]
    index_info = describe_index(status.index_name, api_key, config.get("PINECONE_API_VERSION"), timeout_seconds)
    queries = [default_query_for_item(item_id) for item_id in item_ids]
    embeddings = embed_texts(queries, openai_key, timeout_seconds)
    # The queries are independent, so they run concurrently. map() keeps the input order,
    # and the first failure (if any) is raised exactly as the sequential loop raised it.
    def run_query(query_vector: list[float]) -> tuple[list[RetrievalHit], list[str]]:
        return _query_regulatory_hits(
            index_info,
            status.namespace,
            api_key,
            config.get("PINECONE_API_VERSION"),
            query_vector,
            timeout_seconds,
            top_k,
        )

    with ThreadPoolExecutor(max_workers=min(RETRIEVAL_WORKERS, max(len(embeddings), 1))) as pool:
        outcomes = list(pool.map(run_query, embeddings))
    for item_id, query, (hits, query_blockers) in zip(item_ids, queries, outcomes, strict=True):
        results[item_id] = RetrievalResult(query=query, item_id=item_id, hits=hits, blockers=list(blockers) + query_blockers)
    return results


def default_query_for_item(item_id: str | None) -> str:
    mapping = {
        "ID-01": "official provider and exact model identity",
        "ID-02": "release date for the exact model",
        "ID-03": "base model or upstream checkpoint",
        "DS-01": "distribution and access for the exact model",
        "DS-02": "licence or access terms",
        "TP-01": "architecture type",
        "TP-02": "parameter count",
        "TP-03": "input and output modalities",
        "TP-04": "input and output limits",
        "US-01": "intended uses and representative tasks",
        "US-02": "restrictions on use",
        "US-03": "official integration instructions and API call",
        "DT-01": "training data types and provenance",
        "DT-02": "training data curation",
        "DT-03": "public summary of training content",
        "CT-01": "training duration",
        "CT-02": "training compute",
        "CT-03": "training energy",
    }
    return mapping.get(item_id or "", "relevant regulatory context")

