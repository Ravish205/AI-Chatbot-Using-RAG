"""
RAG orchestration pipeline connecting retriever, LLM providers (Ollama, OpenAI, Mock),
prompt engineering, conversational memory, and source citation extraction.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

from config import settings
from src.prompts import (
    CONDENSE_QUESTION_PROMPT,
    DOCUMENT_SUMMARY_PROMPT,
    RAG_QA_PROMPT,
    RAG_SYSTEM_PROMPT,
    format_chat_history,
    format_context_chunks,
)
from src.retriever import RetrievedChunk, Retriever
from src.utils import get_logger, sanitize_user_input

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Represents the complete answer output from the RAG chain."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    standalone_query: str = ""
    raw_prompt: str = ""
    latency_seconds: float = 0.0
    model_name: str = ""


class BaseLLMClient:
    """Abstract interface for LLM completion calls."""

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM for testing without running local/cloud servers."""

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        if "No relevant context chunks found" in prompt or "<context>\n\n</context>" in prompt:
            return "Based on the provided documents, I do not have enough information to answer this question."

        # Simulate strict grounding check
        if "<context>" in prompt and "</context>" in prompt and "User Question:" in prompt:
            context_part = prompt.split("<context>")[1].split("</context>")[0].lower()
            question_part = prompt.split("User Question:")[1].split("\n")[0].lower()

            # Check overlap of non-trivial words
            import re
            q_words = [w for w in re.findall(r"\b\w{4,}\b", question_part) if w not in {"what", "which", "where", "when", "does", "have", "that", "this"}]
            has_match = any(w in context_part for w in q_words)
            if not has_match and q_words:
                return "Based on the provided documents, I do not have enough information to answer this question."

        return "Based on the provided documents: This is a verified answer grounded directly in the provided document excerpts. [Document, Page 1]"


class OllamaLLMClient(BaseLLMClient):
    """Client for local inference via Ollama server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_MODEL if hasattr(settings, "OLLAMA_MODEL") else getattr(settings, "ollama_model", "llama3.2")
        self.timeout = timeout or settings.ollama_request_timeout

    def is_available(self) -> bool:
        """Checks if the Ollama service is running and responsive."""
        if requests is not None:
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
                return resp.status_code == 200
            except Exception:
                return False
        else:
            try:
                req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    return resp.status == 200
            except Exception:
                return False

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        # First attempt using langchain_ollama if installed
        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(
                base_url=self.base_url,
                model=self.model_name,
                temperature=temperature,
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            pass

        # Fallback direct REST API call to Ollama /api/generate
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if requests is not None:
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_exception()
                data = resp.json()
                return data.get("response", "").strip()
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is installed "
                    f"and running ('ollama serve') and model '{self.model_name}' is pulled ('ollama pull {self.model_name}')."
                )
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}") from e
        else:
            try:
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "").strip()
            except (urllib.error.URLError, ConnectionRefusedError):
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is installed "
                    f"and running ('ollama serve') and model '{self.model_name}' is pulled ('ollama pull {self.model_name}')."
                )
            except Exception as e:
                raise RuntimeError(f"Ollama generation failed: {e}") from e


class OpenAILLMClient(BaseLLMClient):
    """Client for cloud inference via OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model_name = model_name or settings.openai_model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=temperature,
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except ImportError:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()


class LLMFactory:
    """Instantiates the appropriate LLM client."""

    @staticmethod
    def create(provider: Optional[str] = None, **kwargs) -> BaseLLMClient:
        chosen_provider = (provider or settings.llm_provider).lower()
        logger.info("Initializing LLM provider: %s", chosen_provider)

        if chosen_provider == "ollama":
            return OllamaLLMClient(**kwargs)
        elif chosen_provider == "openai":
            return OpenAILLMClient(**kwargs)
        elif chosen_provider in ("mock", "test"):
            return MockLLMClient()
        else:
            logger.warning("Unknown LLM provider '%s'. Falling back to Ollama.", chosen_provider)
            return OllamaLLMClient(**kwargs)


class RAGChain:
    """
    Core RAG chain coordinating question rephrasing, context retrieval,
    strict grounded answer generation, and source citation extraction.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.retriever = retriever
        self.llm = llm_client or LLMFactory.create()

    def _condense_question(self, query: str, chat_history: List[dict]) -> str:
        """
        Reformulates follow-up queries using conversation history.
        """
        if not chat_history:
            return query

        formatted_history = format_chat_history(chat_history, max_turns=3)
        if not formatted_history.strip():
            return query

        prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=formatted_history,
            question=query,
        )

        try:
            standalone = self.llm.generate(prompt, temperature=0.0)
            standalone = standalone.strip().replace('"', '')
            if standalone:
                logger.info("Reformulated query: '%s' -> '%s'", query, standalone)
                return standalone
        except Exception as e:
            logger.warning("Query reformulation failed (%s); using original query.", e)

        return query

    def answer_question(
        self,
        query: str,
        chat_history: Optional[List[dict]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        Processes a user question end-to-end:
        1. Sanitizes user input.
        2. Reformulates query if conversation history exists.
        3. Retrieves relevant context chunks.
        4. If no relevant chunks match, returns explicit admission of missing context.
        5. Formats grounded QA prompt.
        6. Queries LLM and returns structured response with citations.
        """
        start_time = time.time()
        clean_query = sanitize_user_input(query)
        history = chat_history or []

        # Step 1: Rephrase question for conversational follow-ups
        standalone_query = self._condense_question(clean_query, history)

        # Step 2: Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=standalone_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Step 3: Handle empty retrieval (prevent hallucinations)
        if not chunks:
            logger.info("No relevant document chunks retrieved for query: '%s'", standalone_query)
            return RAGResponse(
                answer=(
                    "Based on the provided documents, I could not find information "
                    "relevant to your question. Please verify that the relevant document "
                    "is uploaded or try rephrasing your question."
                ),
                sources=[],
                standalone_query=standalone_query,
                raw_prompt="N/A (No relevant chunks passed similarity threshold)",
                latency_seconds=round(time.time() - start_time, 2),
                model_name=getattr(self.llm, "model_name", "Mock"),
            )

        # Step 4: Format Context and Prompt
        context_str = format_context_chunks(chunks)
        history_str = format_chat_history(history, max_turns=3) if history else ""

        qa_prompt = RAG_QA_PROMPT.format(
            system_prompt=RAG_SYSTEM_PROMPT,
            context=context_str,
            chat_history_section=history_str,
            question=clean_query,
        )

        # Step 5: Generate Answer
        try:
            answer = self.llm.generate(qa_prompt, temperature=0.1)
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            answer = f"Error generating answer from language model: {e}"

        # Step 6: Package Sources
        sources = [
            {
                "citation": c.citation,
                "score": c.score,
                "content_preview": c.content[:250] + ("..." if len(c.content) > 250 else ""),
                "full_content": c.content,
                "metadata": c.metadata,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]

        latency = round(time.time() - start_time, 2)
        return RAGResponse(
            answer=answer,
            sources=sources,
            standalone_query=standalone_query,
            raw_prompt=qa_prompt,
            latency_seconds=latency,
            model_name=getattr(self.llm, "model_name", "LLM"),
        )

    def summarize_documents(self, max_chunks: int = 8) -> str:
        """Generates an executive summary of the currently indexed documents."""
        all_chunks = self.retriever.vector_store.get_all_chunks()
        if not all_chunks:
            return "No documents are currently indexed in the knowledge base."

        # Take sample chunks distributed across the document
        step = max(1, len(all_chunks) // max_chunks)
        sampled = all_chunks[::step][:max_chunks]

        sample_context = "\n\n---\n\n".join(
            f"[{item.get('metadata', {}).get('source_filename', 'Doc')}]:\n{item.get('document', '')}"
            for item in sampled
        )

        prompt = DOCUMENT_SUMMARY_PROMPT.format(context=sample_context)
        try:
            return self.llm.generate(prompt, temperature=0.2)
        except Exception as e:
            return f"Failed to generate document summary: {e}"
