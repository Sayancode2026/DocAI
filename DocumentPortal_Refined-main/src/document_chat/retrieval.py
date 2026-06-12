"""
retrieval.py  (updated — in-memory chat history)
-------------------------------------------------
ConversationalRAG now automatically reads and writes chat history
from/to the module-level MEMORY_STORE singleton.

Key change:  invoke() no longer requires the caller to pass chat_history.
             It pulls history from MEMORY_STORE and saves the new turn
             back after getting the answer — all transparently.

The old signature still works if you pass chat_history explicitly
(e.g. for tests), but for normal API calls just call:

    rag.invoke(user_input)   ← history handled automatically
"""

import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from src.model.models import PromptType

# ── NEW: import the singleton memory store ──────────────────────────────────
from memory_store import MEMORY_STORE


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with in-memory chat history.

    History is stored in the module-level MEMORY_STORE keyed by session_id.
    Each invoke() call:
      1. Loads history from MEMORY_STORE
      2. Runs the LCEL chain with that history
      3. Saves the new Q/A turn back to MEMORY_STORE

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss("faiss_index/abc", k=5)
        answer = rag.invoke("What is ...?")   # history auto-managed
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info(
                "ConversationalRAG initialized",
                session_id=self.session_id,
                prior_turns=MEMORY_STORE.get_turn_count(session_id or ""),
            )
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Load FAISS vectorstore from disk and build retriever + LCEL chain."""
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,  # kept for backwards-compat / tests
    ) -> str:
        """
        Invoke the LCEL pipeline.

        If chat_history is NOT passed (normal API usage), history is
        loaded automatically from MEMORY_STORE and saved back after the call.

        If chat_history IS passed explicitly (tests / manual calls),
        MEMORY_STORE is NOT used — caller is in full control.
        """
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )

            # ── Decide which history to use ──────────────────────────────
            use_store = chat_history is None  # True → auto-manage via MEMORY_STORE
            if use_store:
                history = MEMORY_STORE.get_history(self.session_id or "")
                log.info(
                    "Loaded chat history from MEMORY_STORE",
                    session_id=self.session_id,
                    turns=MEMORY_STORE.get_turn_count(self.session_id or ""),
                )
            else:
                history = chat_history  # caller-supplied (e.g. tests)

            # ── Run the chain ─────────────────────────────────────────────
            payload = {"input": user_input, "chat_history": history}
            answer = self.chain.invoke(payload)

            if not answer:
                log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                answer = "I could not generate an answer based on the provided documents."

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )

            # ── Persist the new turn ──────────────────────────────────────
            if use_store and self.session_id:
                MEMORY_STORE.add_exchange(self.session_id, user_input, answer)
                log.info(
                    "Chat history updated in MEMORY_STORE",
                    session_id=self.session_id,
                    total_turns=MEMORY_STORE.get_turn_count(self.session_id),
                )

            return answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    def clear_history(self) -> None:
        """Clear in-memory chat history for this session."""
        if self.session_id:
            MEMORY_STORE.clear(self.session_id)
            log.info("Chat history cleared", session_id=self.session_id)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
