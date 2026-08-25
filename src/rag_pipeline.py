"""STAGES 3 & 4 of the RAG pipeline: RETRIEVAL (with confidence) + GENERATION.

Teaching note: this is the "online" half of RAG, run once per user question:
  1. RETRIEVE: embed the question, ask Chroma for the top-k most similar
     chunks, and compute a confidence score for how relevant they are.
  2. AUGMENT: stitch those chunks together into a "context" block of text.
  3. GENERATE: send the LLM a prompt that says "only answer using this
     context" — this is what stops the model from making things up
     (hallucinating) and keeps answers grounded in our actual PDFs.
"""
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

# The PROMPT_TEMPLATE is the actual instruction sent to the LLM. {context} and
# {question} are placeholders that get filled in at run time (see
# generate_answer). Telling the model to answer "ONLY" from the context, and
# to admit when it doesn't know, is called "grounding" — a core RAG technique
# to reduce hallucination.
PROMPT_TEMPLATE = """You are an assistant answering questions about student records.
Use ONLY the context below to answer the question. If the context does not
contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""


# --- What is @dataclass? -----------------------------------------------------
# @dataclass is a decorator (a function that wraps another and adds behaviour)
# built into Python's standard library. Normally, to store a small bundle of
# related values you'd write a class like:
#
#     class RetrievedChunk:
#         def __init__(self, document, score):
#             self.document = document
#             self.score = score
#         def __repr__(self):
#             return f"RetrievedChunk(document={self.document!r}, score={self.score!r})"
#         def __eq__(self, other):
#             return (self.document, self.score) == (other.document, other.score)
#
# @dataclass auto-generates that boilerplate (the __init__, __repr__, __eq__
# methods) just from the type-annotated fields you list below. It's purely a
# convenience for classes that are mainly "data containers" — you still get a
# normal Python class you can use exactly like any other.
@dataclass
class RetrievedChunk:
    """One retrieved chunk of text plus how relevant it was to the query."""

    document: Document
    score: float  # relevance/confidence score, 0 (irrelevant) - 1 (identical)


@dataclass
class RagResult:
    """The final output of a RAG query: the answer plus the evidence used.

    `field(default_factory=list)` is needed (instead of writing `= []`
    directly) because Python evaluates default arguments once, at class
    definition time. If we wrote `chunks: List[...] = []`, every RagResult
    instance would accidentally SHARE the exact same list object. A
    default_factory tells @dataclass to call list() fresh for each new
    instance instead.
    """

    answer: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0


def retrieve_with_confidence(
    vector_store: Chroma,
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K,
) -> List[RetrievedChunk]:
    """RETRIEVAL step: find the top-k chunks most similar in meaning to the query.

    Under the hood, Chroma: (1) embeds `query` with the same embedding model
    used during ingestion, (2) compares that vector to every stored chunk
    vector using cosine similarity, and (3) returns the k closest ones along
    with a similarity score we treat as our "confidence".

    If the query mentions an explicit Student ID (e.g. STU20260001), retrieval is
    filtered to that student's chunks first, since dense similarity alone struggles
    to discriminate exact IDs across near-identical structured records.
    """
    id_match = STUDENT_ID_QUERY_PATTERN.search(query)
    if id_match:
        student_id = id_match.group(0).upper()
        # `filter` restricts the search to only chunks whose metadata matches
        # exactly — this is a metadata (keyword) filter, used alongside vector
        # similarity search. Combining both is often called "hybrid search".
        results = vector_store.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"student_id": student_id}
        )
        if results:
            return [RetrievedChunk(document=doc, score=score) for doc, score in results]

    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    return [RetrievedChunk(document=doc, score=score) for doc, score in results]


def _build_context(chunks: List[RetrievedChunk]) -> str:
    """AUGMENT step: turn retrieved chunks into one text block for the prompt.

    We label each chunk with its source PDF and page number so the LLM (and a
    human reading the answer) can trace facts back to where they came from.
    """
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
    """GENERATION step: ask the LLM to answer using only the retrieved context.

    The `prompt | llm | StrOutputParser()` line is LangChain's "LCEL" syntax:
    the `|` pipes the output of one step into the next, like a Unix pipeline.
    Here: fill the prompt template -> send it to the chat model -> extract
    the plain text string from the model's response object.
    """
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
    """Full retrieval-augmented-generation flow for a single question.

    We only call the (paid, slower) LLM if the retrieved chunks look relevant
    enough (confidence >= CONFIDENCE_THRESHOLD). This is a simple but useful
    guardrail: if nothing relevant was found, saying "I don't know" is more
    honest and cheaper than letting the LLM guess from weak context.
    """
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
