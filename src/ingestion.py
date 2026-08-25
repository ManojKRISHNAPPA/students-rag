"""Step 1 & 2 of the pipeline: load raw PDFs and split them into chunks."""
import re
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

STUDENT_ID_PATTERN = re.compile(r"Student ID:\s*(STU\d+)")
FULL_NAME_PATTERN = re.compile(r"Full Name:\s*(.+)")


def _tag_student_metadata(documents: List[Document]) -> None:
    """Attach student_id / student_name metadata so exact-match lookups are possible."""
    for doc in documents:
        id_match = STUDENT_ID_PATTERN.search(doc.page_content)
        if id_match:
            doc.metadata["student_id"] = id_match.group(1)
        name_match = FULL_NAME_PATTERN.search(doc.page_content)
        if name_match:
            doc.metadata["student_name"] = name_match.group(1).strip()


def load_documents(pdf_directory: str = config.PDF_DIRECTORY) -> List[Document]:
    """Load every PDF page in the given directory as a LangChain Document."""
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    _tag_student_metadata(documents)
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Split loaded documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest(pdf_directory: str = config.PDF_DIRECTORY) -> List[Document]:
    """Full ingestion: load -> split."""
    documents = load_documents(pdf_directory)
    return split_documents(documents)
