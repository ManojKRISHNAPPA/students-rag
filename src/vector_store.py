"""Step 3 of the pipeline: embed chunks and persist/load them in Chroma."""
import os
import shutil
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src import config

# Cosine space gives a properly calibrated 0-1 relevance/confidence score.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_embedding_function(openai_api_key: str) -> OpenAIEmbeddings:
    """OpenAI's large embedding model for high-quality semantic retrieval."""
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME, api_key=openai_api_key)


def build_vector_store(
    documents: List[Document],
    openai_api_key: str,
    persist_directory: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> Chroma:
    """Embed chunks and persist them to a Chroma collection on disk."""
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
    """Load an existing Chroma collection from disk."""
    embeddings = get_embedding_function(openai_api_key)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata=COLLECTION_METADATA,
    )


def vector_store_exists(persist_directory: str = config.CHROMA_PERSIST_DIR) -> bool:
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
