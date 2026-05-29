# Multi-User Conversational RAG

A conversational RAG system built with LangChain, FAISS, and Groq that supports multiple isolated user sessions simultaneously. Each session has its own chat history stored in SQLite, so switching between users never mixes up context.

Built while learning history-aware retrieval and multi-session memory management as part of my transition from .NET development into GenAI engineering.

---

## How It Works

```
User question (+ session ID)
        ↓
[History-aware retriever]
  └─ Rewrites follow-up questions into standalone questions
  └─ Retrieves top-K chunks from FAISS
        ↓
[Question-answer chain]
  └─ Groq LLM generates answer grounded in retrieved context
        ↓
[SQLChatMessageHistory]
  └─ Saves (question, answer) to SQLite under the session ID
        ↓
Answer returned to user
```

Each session ID (e.g. `alice`, `bob`) gets a completely separate history row in `chat_history.db`, so two users can run concurrent conversations without any cross-contamination.

---

## Features

- **Per-session memory** — SQLite-backed chat history isolated by session ID
- **History-aware retrieval** — follow-up questions are rephrased before search so context is preserved correctly
- **Sliding memory window** — only the last N turns are injected into the prompt, preventing unbounded token growth
- **Session commands** — switch users, view history, and clear sessions from the CLI without restarting
- **Zero-setup demo** — a sample knowledge base is written automatically on first run

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone https://github.com/imbhaskarm/multiuser-rag.git
cd multiuser-rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**

```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

Get a free Groq API key at: https://console.groq.com

**3. Run**

```bash
python main.py
```

On first run, the system will:
- Create `data/knowledge_base.txt` (sample text about Indian wildlife and RAG concepts)
- Download the sentence-transformers model (~90 MB) and build the FAISS index
- Start the interactive CLI

Subsequent runs skip the build step automatically.

---

## CLI Commands

| Command | Effect |
|---|---|
| `/switch <id>` | Switch to a different user session |
| `/history` | Show recent messages for the active session |
| `/clear` | Delete all history for the active session |
| `/exit` | Quit the program |

---

## Example Session

```
Enter session ID to begin: alice

Active session: alice

[alice] You: What is RAG?
Assistant: RAG stands for Retrieval-Augmented Generation. It retrieves relevant
document chunks from a vector database and passes them to a language model so
answers are grounded in external knowledge.

[alice] You: How does the multi-user version differ?
Assistant: A multi-user conversational RAG system stores separate conversation
history for each session ID so that different users can talk to the same system
without mixing up context.

[alice] You: /switch bob
Switched to session: bob

[bob] You: What is the fastest land animal?
Assistant: The fastest land animal is the cheetah.
```

---

## Project Structure

```
multiuser-rag/
├── app/
│   ├── __init__.py
│   ├── config.py        # all settings and paths
│   ├── data_setup.py    # creates sample knowledge base on first run
│   ├── vectorstore.py   # FAISS build and load functions
│   ├── history.py       # SQLite-backed per-session chat history
│   ├── chains.py        # history-aware retriever + RAG chain
│   └── chat.py          # CLI runner and ask_question() helper
├── main.py              # entry point: bootstrap + run CLI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Notes

- LLM: `llama-3.1-8b-instant` via Groq (free tier, fast inference)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (local, no API key needed)
- The SQLite database `chat_history.db` is created automatically in the project root and is excluded from version control
- The FAISS index at `data/faiss_index/` is also excluded — it is rebuilt from the knowledge base on first run
- To use your own document instead of the sample text, replace the content written by `data_setup.py` with a `TextLoader` pointing to your file, or drop your `.txt` file into `data/` and update the path in `main.py`
