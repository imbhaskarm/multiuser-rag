import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ⚠️ Updated from deprecated OpenAI → Groq (latest as of 2025)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL      = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Use an absolute path for the SQLite DB so it always resolves correctly
# regardless of which directory the user runs the script from.
_BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH   = f"sqlite:///{_BASE_DIR / 'chat_history.db'}"

FAISS_INDEX_PATH = str(_BASE_DIR / "data" / "faiss_index")
CHUNK_SIZE       = 700
CHUNK_OVERLAP    = 120
TOP_K            = 4
MEMORY_WINDOW_K  = 10
