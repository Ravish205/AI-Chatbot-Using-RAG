"""
Streamlit Web Application for AI-Powered Document Chatbot using RAG.
Features conversational chat, multi-format document ingestion, persistent ChromaDB,
hybrid retrieval, source citations, chat history export, and document summarization.
"""

from __future__ import annotations

import os
import time
from typing import List

import streamlit as st

from config import settings
from src.document_loader import (
    CorruptedFileError,
    DocumentLoader,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.embeddings import EmbeddingFactory
from src.rag_chain import LLMFactory, OllamaLLMClient, RAGChain
from src.retriever import Retriever
from src.text_splitter import TextSplitter
from src.utils import export_chat_history_json, export_chat_history_txt, get_logger
from src.vector_store import VectorStoreManager

logger = get_logger("app")

# Set Page Config
st.set_page_config(
    page_title="DocuMind RAG — AI Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean modern UI
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    /* Header typography */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    /* Badge pills */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-success {
        background-color: #ecfdf5;
        color: #065f46;
        border-color: #a7f3d0;
    }
    .badge-info {
        background-color: #eff6ff;
        color: #1e40af;
        border-color: #bfdbfe;
    }
    .badge-warning {
        background-color: #fffbeb;
        color: #92400e;
        border-color: #fde68a;
    }
    /* Source citation box */
    .citation-card {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    /* Sidebar header */
    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_vector_store(collection_name: str) -> VectorStoreManager:
    """Initializes and caches the VectorStoreManager singleton."""
    embeddings = EmbeddingFactory.get_embeddings(settings.embedding_provider)
    return VectorStoreManager(
        collection_name=collection_name,
        persist_directory=settings.chroma_persist_directory,
        embedding_provider=embeddings,
    )


def initialize_session_state():
    """Initializes Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = settings.default_collection_name
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = settings.llm_provider
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = settings.ollama_model
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = settings.openai_model
    if "top_k" not in st.session_state:
        st.session_state.top_k = settings.top_k
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = settings.similarity_threshold
    if "enable_hybrid" not in st.session_state:
        st.session_state.enable_hybrid = settings.enable_hybrid_search
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False


initialize_session_state()

# Load Vector Store
vector_store = get_vector_store(st.session_state.collection_name)
retriever = Retriever(
    vector_store=vector_store,
    top_k=st.session_state.top_k,
    similarity_threshold=st.session_state.similarity_threshold,
    enable_hybrid_search=st.session_state.enable_hybrid,
)

# Initialize LLM Client
if st.session_state.llm_provider == "ollama":
    llm_client = LLMFactory.create(
        provider="ollama",
        base_url=settings.ollama_base_url,
        model_name=st.session_state.ollama_model,
        timeout=settings.ollama_request_timeout,
    )
    ollama_online = isinstance(llm_client, OllamaLLMClient) and llm_client.is_available()
else:
    llm_client = LLMFactory.create(
        provider="openai",
        model_name=st.session_state.openai_model,
        api_key=settings.openai_api_key,
    )
    ollama_online = False

rag_chain = RAGChain(retriever=retriever, llm_client=llm_client)

# ==============================================================================
# SIDEBAR CONFIGURATION & INGESTION
# ==============================================================================
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")

    # Document Upload Section
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload PDF, Plain Text, or Microsoft Word documents (up to 25MB each).",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Uploaded Files", type="primary", use_container_width=True):
            loader = DocumentLoader()
            splitter = TextSplitter()
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count = 0
            skipped_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                file_bytes = uploaded_file.getvalue()
                status_text.text(f"Processing ({i + 1}/{len(uploaded_files)}): {file_name}...")

                try:
                    # 1. Extract documents
                    raw_docs = loader.load_from_bytes(file_bytes, filename=file_name)

                    # 2. Chunk text
                    chunks = splitter.split_documents(raw_docs)

                    # 3. Embed and store
                    result = vector_store.add_chunks(chunks, skip_if_exists=True)

                    if result["status"] == "skipped":
                        skipped_count += 1
                        st.info(f"ℹ️ {file_name} was already indexed.")
                    else:
                        success_count += 1
                        st.success(f"✅ {file_name} ({len(chunks)} chunks indexed)")

                except UnsupportedFileTypeError as e:
                    st.error(f"❌ Unsupported format: {e}")
                except FileTooLargeError as e:
                    st.error(f"❌ File too large: {e}")
                except EmptyDocumentError as e:
                    st.warning(f"⚠️ {file_name} contained no readable text.")
                except CorruptedFileError as e:
                    st.error(f"❌ Could not read {file_name}: {e}")
                except Exception as e:
                    st.error(f"❌ Ingestion error on {file_name}: {e}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text(f"Done! Indexed: {success_count}, Skipped (Duplicate): {skipped_count}")
            time.sleep(1)
            st.rerun()

    # Indexed Documents Overview
    indexed_docs = vector_store.get_indexed_documents()
    total_chunks = vector_store.count()

    st.markdown("---")
    st.markdown("#### 📁 Indexed Documents")
    if indexed_docs:
        st.markdown(
            f"<span class='metric-badge badge-success'>📚 {len(indexed_docs)} Documents</span>"
            f"<span class='metric-badge badge-info'>🧩 {total_chunks} Chunks</span>",
            unsafe_allow_html=True,
        )
        for doc in indexed_docs:
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                st.caption(f"**Chunks:** {doc['chunk_count']}")
                st.caption(f"**Characters:** {doc['total_characters']:,}")
                st.caption(f"**Type:** {doc['file_type'].upper()}")
                st.caption(f"**Checksum:** `{doc['file_hash'][:10]}...`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Summarize", use_container_width=True, help="Generate executive summary of all docs"):
                with st.spinner("Generating document summary..."):
                    summary_text = rag_chain.summarize_documents()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"### 📋 Document Collection Executive Summary\n\n{summary_text}",
                            "sources": [],
                            "model_name": getattr(llm_client, "model_name", "LLM"),
                        }
                    )
                    st.rerun()

        with col2:
            if st.button("🗑️ Reset DB", use_container_width=True, help="Clear all stored embeddings"):
                vector_store.reset_collection()
                st.toast("Knowledge base reset successfully!", icon="🧹")
                time.sleep(0.5)
                st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF, TXT, or DOCX above to begin!")

    # Retrieval Tuning
    st.markdown("---")
    st.markdown("#### ⚙️ Retrieval Parameters")
    st.session_state.top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of relevant text passages passed to the language model.",
    )
    st.session_state.similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=0.8,
        value=float(st.session_state.similarity_threshold),
        step=0.05,
        help="Relevance score cutoff. Passages below this score are discarded.",
    )
    st.session_state.enable_hybrid = st.checkbox(
        "Enable Hybrid Search (BM25 + Dense)",
        value=st.session_state.enable_hybrid,
        help="Combines semantic vector embeddings with keyword BM25 ranking for optimal retrieval.",
    )

    # Model Configuration
    st.markdown("---")
    st.markdown("#### 🤖 Model Provider")
    provider_choice = st.selectbox(
        "Active LLM",
        options=["ollama", "openai"],
        index=0 if st.session_state.llm_provider == "ollama" else 1,
    )
    if provider_choice != st.session_state.llm_provider:
        st.session_state.llm_provider = provider_choice
        st.rerun()

    if st.session_state.llm_provider == "ollama":
        st.session_state.ollama_model = st.text_input("Ollama Model", value=st.session_state.ollama_model)
        if ollama_online:
            st.markdown("<span class='metric-badge badge-success'>● Ollama Online</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='metric-badge badge-warning'>▲ Ollama Offline</span>", unsafe_allow_html=True)
            st.caption("Start Ollama locally: `ollama serve`")
    else:
        st.session_state.openai_model = st.text_input("OpenAI Model", value=st.session_state.openai_model)
        has_key = bool(settings.openai_api_key or os.getenv("OPENAI_API_KEY"))
        if has_key:
            st.markdown("<span class='metric-badge badge-success'>● API Key Configured</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='metric-badge badge-warning'>▲ Missing OPENAI_API_KEY</span>", unsafe_allow_html=True)

    # Advanced Settings
    st.markdown("---")
    st.session_state.debug_mode = st.toggle("🐞 Debug / Inspect Mode", value=st.session_state.debug_mode)

    # Chat Actions
    st.markdown("#### 💬 Conversation")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c_btn2:
        if st.session_state.messages:
            txt_export = export_chat_history_txt(st.session_state.messages)
            st.download_button(
                label="📥 Export TXT",
                data=txt_export,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================

st.title("📚 DocuMind AI")
st.caption(
    "Grounding conversational answers strictly in your uploaded PDF, DOCX, and TXT documents "
    "using semantic search and open-source models."
)

# Display Knowledge Base Status Banner
if total_chunks == 0:
    st.info(
        "👋 **Welcome!** Upload documents using the left sidebar to index your knowledge base. "
        "Once indexed, you can ask specific questions and receive verified, cited answers."
    )
elif not ollama_online and st.session_state.llm_provider == "ollama":
    st.warning(
        "⚠️ **Ollama server is not detected on localhost:11434.** "
        "Make sure Ollama is running (`ollama serve`) and model is pulled (`ollama pull "
        f"{st.session_state.ollama_model}`). Responses will show connection alerts until Ollama starts."
    )

# Render Chat History
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    latency = msg.get("latency")
    model_name = msg.get("model_name")
    debug_info = msg.get("debug_info", {})

    with st.chat_message(role):
        st.markdown(content)

        # Show model & latency badge if available
        if role == "assistant" and (latency or model_name):
            badge_html = ""
            if model_name:
                badge_html += f"<span class='metric-badge badge-info'>🤖 {model_name}</span>"
            if latency:
                badge_html += f"<span class='metric-badge'>{latency}s</span>"
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)

        # Show expandable source citations
        if role == "assistant" and sources:
            with st.expander(f"📚 Source Citations & Evidence ({len(sources)} cited)", expanded=False):
                for idx, src in enumerate(sources, start=1):
                    citation_text = src.get("citation", "Document")
                    score = src.get("score", 0.0)
                    preview = src.get("full_content", src.get("content_preview", ""))
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <strong>[{idx}] {citation_text}</strong> &nbsp;
                            <span class="metric-badge badge-success">Relevance: {score:.1%}</span>
                            <p style="margin-top: 6px; font-style: italic; color: #475569;">
                                "{preview}"
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Show debug inspector if debug mode is active
        if role == "assistant" and st.session_state.debug_mode and debug_info:
            with st.expander("🛠️ Debug Inspector (Retrieval & Prompt)", expanded=False):
                st.markdown(f"**Standalone Query:** `{debug_info.get('standalone_query', '')}`")
                st.markdown(f"**Retrieved Chunks Count:** `{len(sources)}`")
                if "raw_prompt" in debug_info:
                    st.markdown("**Raw Prompt:**")
                    st.code(debug_info["raw_prompt"], language="markdown")


# User Query Input
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate grounded response
    with st.chat_message("assistant"):
        with st.spinner("Searching document context and generating answer..."):
            try:
                response = rag_chain.answer_question(
                    query=user_query,
                    chat_history=st.session_state.messages[:-1],
                    top_k=st.session_state.top_k,
                    similarity_threshold=st.session_state.similarity_threshold,
                )

                st.markdown(response.answer)

                # Badges
                badge_html = f"<span class='metric-badge badge-info'>🤖 {response.model_name}</span>"
                badge_html += f"<span class='metric-badge'>{response.latency_seconds}s</span>"
                st.markdown(badge_html, unsafe_allow_html=True)

                # Source citations
                if response.sources:
                    with st.expander(f"📚 Source Citations & Evidence ({len(response.sources)} cited)", expanded=False):
                        for idx, src in enumerate(response.sources, start=1):
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <strong>[{idx}] {src['citation']}</strong> &nbsp;
                                    <span class="metric-badge badge-success">Relevance: {src['score']:.1%}</span>
                                    <p style="margin-top: 6px; font-style: italic; color: #475569;">
                                        "{src['full_content']}"
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Debug panel
                debug_info = {
                    "standalone_query": response.standalone_query,
                    "raw_prompt": response.raw_prompt,
                }
                if st.session_state.debug_mode:
                    with st.expander("🛠️ Debug Inspector (Retrieval & Prompt)", expanded=False):
                        st.markdown(f"**Standalone Query:** `{response.standalone_query}`")
                        st.markdown("**Raw Prompt Sent to LLM:**")
                        st.code(response.raw_prompt, language="markdown")

                # Save assistant response to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "latency": response.latency_seconds,
                        "model_name": response.model_name,
                        "debug_info": debug_info,
                    }
                )

            except Exception as e:
                err_msg = f"An error occurred while answering your question: {e}"
                logger.error(err_msg, exc_info=True)
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ {err_msg}",
                        "sources": [],
                    }
                )
