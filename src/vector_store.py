"""STAGE 2 of the RAG pipeline: EMBEDDING + VECTOR STORAGE.

Teaching note: an "embedding" is just a list of numbers (a vector) that
represents the *meaning* of a piece of text. Texts with similar meaning end up
with vectors that are close together in that number-space, even if they don't
share the exact same words (e.g. "GPA" and "grade point average"). This lets
us search by MEANING instead of by exact keyword matching.

Chroma is our "vector database": a store that is specialised for saving many
vectors and quickly finding the ones closest to a new query vector.
"""
import os
import shutil
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src import config

# "hnsw:space": "cosine" tells Chroma to compare vectors using cosine
# similarity (the angle between two vectors) instead of raw distance. Cosine
# similarity naturally lands in a 0-1 range for our embeddings, which is what
# lets us treat the retrieval score as a "confidence" number later on.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_embedding_function(openai_api_key: str) -> OpenAIEmbeddings:
    """Return the embedding model used to turn text into vectors.

    We use OpenAI's `text-embedding-3-large` — a paid API model, hence the
    api_key — because it captures meaning more accurately than smaller local
    models, which improves how well our retrieval finds the right chunks.
    """
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME, api_key=openai_api_key)


def build_vector_store(
    documents: List[Document],
    openai_api_key: str,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> Chroma:
    """Embed every chunk and save the resulting vectors to disk (Chroma).

    `Chroma.from_documents` does two things under the hood for each chunk:
      1. Calls the embedding model to turn chunk.page_content into a vector.
      2. Stores {vector, page_content, metadata} together so we can later
         retrieve the original text once we find a matching vector.
    `persist_directory` is where Chroma writes its on-disk database files, so
    we don't have to re-embed everything every time the app restarts.
    """
    embeddings = get_embedding_function(openai_api_key)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        collection_metadata=COLLECTION_METADATA,
    )


def load_vector_store(
    openai_api_key: str,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> Chroma:
    """Re-open an already-built Chroma collection from disk (no re-embedding)."""
    embeddings = get_embedding_function(openai_api_key)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata=COLLECTION_METADATA,
    )


def vector_store_exists(persist_directory: str = config.CHROMA_PERSIST_DIR) -> bool:
    """True if a Chroma index has already been built and saved to disk."""
    return os.path.isdir(persist_directory) and len(os.listdir(persist_directory)) > 0


def get_or_build_vector_store(
    openai_api_key: str,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
    documents: Optional[List[Document]] = None,
    force_rebuild: bool = False,
) -> Chroma:
    """Load the vector store if it already exists, otherwise build it from documents.

    Set force_rebuild=True to discard any existing index and re-embed from scratch
    (required whenever the embedding model or ingestion metadata changes).
    """
    if force_rebuild and os.path.isdir(persist_directory):
        shutil.rmtree(persist_directory)

    if not force_rebuild and vector_store_exists(persist_directory):
        return load_vector_store(openai_api_key, persist_directory, collection_name)

    if documents is None:
        raise ValueError("No existing vector store found and no documents provided to build one.")
    return build_vector_store(documents, openai_api_key, persist_directory, collection_name)
