# students-rag

A Retrieval-Augmented Generation (RAG) app over student PDF records, built with
LangChain, Chroma and Streamlit.

## Architecture

```
student_pdfs/ ──▶ ingestion (load ▶ split) ──▶ embeddings ──▶ Chroma vector store
                                                                     │
                                                                     ▼
                                                     retrieval + confidence score
                                                                     │
                                                                     ▼
                                                context-augmented prompt ──▶ OpenAI LLM ──▶ answer
```

- **Ingestion** ([src/ingestion.py](src/ingestion.py)): loads all PDFs in `student_pdfs/`
  with `PyPDFDirectoryLoader`, then splits them into overlapping chunks with
  `RecursiveCharacterTextSplitter`.
- **Embeddings & storage** ([src/vector_store.py](src/vector_store.py)): embeds chunks
  with OpenAI's `text-embedding-3-large` model and persists them in a Chroma
  collection (cosine similarity space) under `chroma_db/`.
- **Retrieval + confidence** ([src/rag_pipeline.py](src/rag_pipeline.py)): retrieves the
  top-k most relevant chunks and their similarity-based confidence scores.
- **Augmented generation**: builds a context-grounded prompt from the retrieved
  chunks and sends it to an OpenAI LLM (`langchain-openai`) to produce the final answer.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the sidebar (used for both embeddings and answers), click
**Build / Rebuild Index** the first time (embeds all PDFs in `student_pdfs/`), then ask
questions in the main panel. Rebuilding always re-embeds from scratch.
