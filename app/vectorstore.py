from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# ⚠️ Updated from deprecated Chroma → FAISS (latest as of 2025)
from langchain_community.vectorstores import FAISS

from app.config import EMBEDDING_MODEL, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(doc_path: str) -> FAISS:
    """
    Load a plain-text document, split it into chunks,
    build a FAISS index, and persist it to FAISS_INDEX_PATH.
    """
    loader   = TextLoader(doc_path, encoding="utf-8")
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks     = splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we built this index ourselves
    )


def vectorstore_exists() -> bool:
    return Path(FAISS_INDEX_PATH, "index.faiss").exists()
