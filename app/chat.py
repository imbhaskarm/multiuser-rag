from app.chains import build_multiuser_chain
from app.history import clear_session, get_recent_chat_history

# Chain is built once at first use and reused for all sessions.
_chain = None


def get_chain():
    global _chain
    if _chain is None:
        _chain = build_multiuser_chain()
    return _chain


def ask_question(session_id: str, question: str):
    """
    Invoke the RAG chain for a specific user session.
    Returns (answer_text, context_docs).
    """
    runnable = get_chain()
    config   = {"configurable": {"session_id": session_id}}
    result   = runnable.invoke({"input": question}, config=config)
    return result["answer"], result.get("context", [])


def print_sources(context_docs: list) -> None:
    if not context_docs:
        print("No sources returned.\n")
        return
    print("Retrieved sources:")
    for i, doc in enumerate(context_docs, start=1):
        snippet = doc.page_content[:220].replace("\n", " ")
        print(f"  {i}. {snippet}...")
    print()


def run_cli() -> None:
    get_chain()  # warm up on startup so the first query is fast

    print("=" * 68)
    print("  Multi-User Conversational RAG")
    print("=" * 68)
    print("Commands:")
    print("  /switch <id>   switch to a different user session")
    print("  /history       show recent history for the active session")
    print("  /clear         clear history for the active session")
    print("  /exit          quit")
    print()

    session_id = input("Enter session ID to begin (e.g. alice): ").strip() or "default_user"
    print(f"\nActive session: {session_id}\n")

    while True:
        question = input(f"[{session_id}] You: ").strip()
        if not question:
            continue

        if question.lower() == "/exit":
            print("Goodbye!")
            break

        if question.lower().startswith("/switch "):
            new_id = question.split(" ", 1)[1].strip()
            if new_id:
                session_id = new_id
                print(f"Switched to session: {session_id}\n")
            continue

        if question.lower() == "/clear":
            clear_session(session_id)
            print("Session history cleared.\n")
            continue

        if question.lower() == "/history":
            messages = get_recent_chat_history(session_id)
            if not messages:
                print("No history for this session yet.\n")
                continue
            print("Recent history:")
            for msg in messages:
                role = msg.__class__.__name__.replace("Message", "")
                print(f"  {role}: {msg.content}")
            print()
            continue

        answer, context = ask_question(session_id, question)
        print(f"\nAssistant: {answer}\n")
        print_sources(context)
