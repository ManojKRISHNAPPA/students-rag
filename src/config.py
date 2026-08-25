"""Central configuration for the Students RAG pipeline.

Teaching note: keeping every "tunable" value in one file (instead of
scattered magic numbers across the codebase) makes it easy for students to
experiment — e.g. change CHUNK_SIZE and see how it affects answer quality —
without hunting through every module.
"""
import os

# Ingestion
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "student_pdfs")

# Splitting: how big each chunk is, and how much consecutive chunks overlap,
# both measured in characters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embeddings: the model that turns text into vectors (see vector_store.py).
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# Vector store: where Chroma persists its on-disk database.
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "student_documents"

# Retrieval: how many chunks to fetch per question, and the minimum
# confidence score required before we trust them enough to call the LLM.
RETRIEVAL_TOP_K = 4
CONFIDENCE_THRESHOLD = 0.5  # min relevance score (0-1) to treat context as trustworthy

# LLM: the chat model used for the final "generation" step.
OPENAI_MODEL_NAME = "gpt-4o-mini"
