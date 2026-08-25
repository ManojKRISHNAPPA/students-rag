"""STAGE 1 of the RAG pipeline: INGESTION = LOAD + SPLIT.

Teaching note: a Large Language Model (LLM) has never seen our private student
PDFs, so we cannot just "ask" it questions about them. RAG (Retrieval-Augmented
Generation) works around this in two phases:
  1. OFFLINE (this file + vector_store.py): read the documents once, break them
     into small "chunks", turn each chunk into a numeric vector (embedding),
     and store those vectors in a searchable database (Chroma).
  2. ONLINE (rag_pipeline.py, run per question): embed the user's question,
     find the most similar chunks, and hand them to the LLM as context.
This file only handles step 1: LOAD the PDFs, then SPLIT them into chunks.
"""
import re
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# Regex = a pattern used to find text. Here we scan each PDF's plain text for
# lines like "Student ID: STU20260001" and "Full Name: Esha Joshi" so we can
# save those values as structured metadata (see _tag_student_metadata below).
STUDENT_ID_PATTERN = re.compile(r"Student ID:\s*(STU\d+)")
FULL_NAME_PATTERN = re.compile(r"Full Name:\s*(.+)")


def _tag_student_metadata(documents: List[Document]) -> None:
    """Extract Student ID / Name from the text and store them as metadata.

    Why this matters: with ~1000 near-identical student records, plain
    semantic similarity search can confuse one student with another (their
    text "looks" alike to an embedding model). By tagging each document with
    an exact student_id, we can later filter precisely instead of only
    relying on "fuzzy" similarity — a classic RAG lesson: combine metadata
    filtering with vector search for structured data.
    """
    for doc in documents:
        id_match = STUDENT_ID_PATTERN.search(doc.page_content)
        if id_match:
            doc.metadata["student_id"] = id_match.group(1)
        name_match = FULL_NAME_PATTERN.search(doc.page_content)
        if name_match:
            doc.metadata["student_name"] = name_match.group(1).strip()


def load_documents(pdf_directory: str = config.PDF_DIRECTORY) -> List[Document]:
    """LOAD step: read every PDF in a folder into LangChain `Document` objects.

    Each `Document` has two parts:
      - page_content: the raw extracted text of that PDF page.
      - metadata: a dict of extra facts about the document (filename, page
        number, and — after tagging — student_id / student_name).
    """
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    _tag_student_metadata(documents)
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """SPLIT step: break long documents into small overlapping chunks.

    Why split at all? Two reasons students should know:
      1. Embedding models and LLM context windows work best with short,
         focused pieces of text rather than huge documents.
      2. Smaller chunks make retrieval more precise — we want to fetch just
         the few sentences that answer the question, not an entire page.
    `chunk_overlap` repeats a bit of text between consecutive chunks so we
    don't accidentally cut a sentence/fact in half at a chunk boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest(pdf_directory: str = config.PDF_DIRECTORY) -> List[Document]:
    """Convenience wrapper: LOAD -> SPLIT, run once when (re)building the index."""
    documents = load_documents(pdf_directory)
    return split_documents(documents)
