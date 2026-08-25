"""Central configuration for the Students RAG pipeline."""
import os

# Ingestion
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "student_pdfs")

# Splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embeddings
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# Vector store
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "student_documents"

# Retrieval
RETRIEVAL_TOP_K = 4
CONFIDENCE_THRESHOLD = 0.5  # min relevance score (0-1) to treat context as trustworthy

# LLM
OPENAI_MODEL_NAME = "gpt-4o-mini"
