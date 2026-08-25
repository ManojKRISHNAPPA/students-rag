"""Steps 4 & 5 of the pipeline: retrieval with confidence scoring, then
augmented generation via an LLM (OpenAI)."""
import re
from dataclasses import dataclass, field
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config

STUDENT_ID_QUERY_PATTERN = re.compile(r"STU\d+", re.IGNORECASE)

PROMPT_TEMPLATE = """You are an assistant answering questions about student records.
Use ONLY the context below to answer the question. If the context does not
contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedChunk:
    document: Document
    score: float  # relevance score, 0 (irrelevant) - 1 (identical)


@dataclass
class RagResult:
    answer: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0


def retrieve_with_confidence(
    vector_store: Chroma,
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K,
) -> List[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks along with a 0-1 confidence score.

    If the query mentions an explicit Student ID (e.g. STU20260001), retrieval is
    filtered to that student's chunks first, since dense similarity alone struggles
    to discriminate exact IDs across near-identical structured records.
    """
    id_match = STUDENT_ID_QUERY_PATTERN.search(query)
    if id_match:
        student_id = id_match.group(0).upper()
        results = vector_store.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"student_id": student_id}
        )
        if results:
            return [RetrievedChunk(document=doc, score=score) for doc, score in results]

    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    return [RetrievedChunk(document=doc, score=score) for doc, score in results]


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.document.metadata.get("source", "unknown")
        page = chunk.document.metadata.get("page_label", chunk.document.metadata.get("page", "?"))
        parts.append(f"[Source {i}: {source}, page {page}]\n{chunk.document.page_content}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    openai_api_key: str,
    model_name: str = config.OPENAI_MODEL_NAME,
) -> str:
    """Augment the LLM prompt with retrieved context and generate an answer."""
    llm = ChatOpenAI(api_key=openai_api_key, model=model_name, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    context = _build_context(chunks)
    return chain.invoke({"context": context, "question": query})


def answer_question(
    vector_store: Chroma,
    query: str,
    openai_api_key: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    model_name: str = config.OPENAI_MODEL_NAME,
) -> RagResult:
    """Full retrieval-augmented-generation flow for a single question."""
    chunks = retrieve_with_confidence(vector_store, query, top_k)

    if not chunks:
        return RagResult(answer="No relevant documents found.", chunks=[], confidence=0.0)

    confidence = max(chunk.score for chunk in chunks)

    if confidence < config.CONFIDENCE_THRESHOLD:
        return RagResult(
            answer="I couldn't find sufficiently relevant information in the student records to answer that confidently.",
            chunks=chunks,
            confidence=confidence,
        )

    answer = generate_answer(query, chunks, openai_api_key, model_name)
    return RagResult(answer=answer, chunks=chunks, confidence=confidence)
