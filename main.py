"""
Entry point for the Multi-User Conversational RAG system.

On first run:
  - Creates data/knowledge_base.txt (sample text)
  - Builds the FAISS vectorstore at data/faiss_index/
  - Starts the interactive CLI

Subsequent runs skip the build step if the index already exists.

Usage:
    python main.py
"""
from app.data_setup import ensure_sample_knowledge_base
from app.vectorstore import build_vectorstore, vectorstore_exists
from app.chat import run_cli


def bootstrap() -> None:
    kb_path = ensure_sample_knowledge_base()
    if not vectorstore_exists():
        print("Building FAISS vectorstore from sample knowledge base...")
        print("(Downloading sentence-transformers model on first run — ~90 MB)")
        build_vectorstore(kb_path)
        print("Vectorstore created.\n")


if __name__ == "__main__":
    bootstrap()
    run_cli()
