from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.config import GROQ_API_KEY, LLM_MODEL, TOP_K, MEMORY_WINDOW_K
from app.vectorstore import load_vectorstore
from app.history import get_session_history_db


def _trim_chat_history(payload: dict) -> dict:
    """
    Keep only the most recent MEMORY_WINDOW_K turns in context.
    This prevents the prompt from growing unbounded across long sessions.
    """
    payload = dict(payload)
    payload["chat_history"] = payload.get("chat_history", [])[-(2 * MEMORY_WINDOW_K):]
    return payload


def build_multiuser_chain() -> RunnableWithMessageHistory:
    """
    Build the full conversational RAG chain.

    Two-stage design:
    1. history_aware_retriever  — rewrites the follow-up question into a
       standalone question before retrieval, so search quality stays high
       even mid-conversation.
    2. question_answer_chain    — generates the final answer from retrieved
       context and the full chat history.
    """
    # ⚠️ Updated from deprecated OpenAI → Groq via langchain-groq (latest as of 2025)
    llm        = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL)
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question which can be "
            "understood without the chat history. Do not answer the question — just "
            "reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, rephrase_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you do not know the answer, say you do not know. "
            "Keep the answer concise but helpful.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain             = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    trimmed_rag_chain = RunnableLambda(_trim_chat_history) | rag_chain

    return RunnableWithMessageHistory(
        trimmed_rag_chain,
        get_session_history_db,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
