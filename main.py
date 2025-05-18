"""
Main entry point for the GitLab Handbook RAG application.
"""
import time
import logging
import streamlit as st
from config import MAIN_URLS, TOP_K_LINKS, MAX_LINKS_PER_SITE

from src.crawling import extract_link_contexts
from src.embedding import EmbeddingManager
from src.document_processor import DocumentProcessor
from src.vectorstore import VectorStore
from src.rag_chain import RAGChain

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='(%(levelname)s) - [%(funcName)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def run_rag_pipeline(query, source=None):
    """
    Run the complete RAG pipeline.

    Args:
        query (str): The query to answer
        source (str, optional): Specific URL to search

    Returns:
        tuple: (answer, sources, elapsed_time)
    """
    start_time = time.time()
    progress_placeholder = st.empty()

    # Step 1: Extract links from main pages
    progress_placeholder.info("Step 1: Extracting links from main pages...")
    logger.info("Step 1: Extracting links from main pages")

    if source:
        # If specific source provided, only use that URL
        relevant_links = [source]
        progress_placeholder.info(f"Using specific source URL: {source}")
    else:
        all_links = []
        for main_url in MAIN_URLS:
            link_data = extract_link_contexts(main_url, MAX_LINKS_PER_SITE)
            all_links.extend(link_data)

        # Step 2: Rank links by relevance to query
        progress_placeholder.info(
            "Step 2: Ranking links by relevance to query...")
        logger.info("Step 2: Ranking links by relevance to query")
        embedding_manager = EmbeddingManager()
        relevant_links = embedding_manager.rank_links_by_query_relevance(
            query, all_links, top_k=TOP_K_LINKS)
        logger.info(f"Selected {len(relevant_links)} relevant links")

    # Step 3: Load and process documents
    progress_placeholder.info("Step 3: Loading and processing documents...")
    logger.info("Step 3: Loading and processing documents")
    doc_processor = DocumentProcessor()

    if source:
        # For single source, use direct method
        docs = doc_processor.load_single_url(source)
    else:
        # For multiple sources, use parallel processing
        docs = doc_processor.load_and_split_documents_parallel(relevant_links)

    # Step 4: Build vector store
    progress_placeholder.info("Step 4: Building vector store...")
    logger.info("Step 4: Building vector store")
    vectorstore = VectorStore()
    vectorstore.build_from_documents(docs)

    # Step 5: Create and run RAG chain
    progress_placeholder.info("Step 5: Generating answer...")
    logger.info("Step 5: Creating and running RAG chain")
    rag_chain = RAGChain(vectorstore)
    result = rag_chain.invoke(query)

    # Format results
    answer, sources = rag_chain.format_results(result)

    # Calculate time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

    # Clear the progress placeholder
    progress_placeholder.empty()

    return answer, sources, elapsed_time


def main():
    st.title("GenAI Gitlab Chatbot built on Gemini Model")

    # Add a source URL selector in the sidebar
    with st.sidebar:
        st.header("Options")
        use_source = st.checkbox("Search specific URL")
        source_url = None
        if use_source:
            source_url = st.text_input("Enter URL to search:",
                                       placeholder="https://handbook.gitlab.com/handbook/")

    # Containers for question and answer
    question_container = st.container()
    answer_container = st.container()

    # Get user input
    query = st.chat_input("Ask a question about GitLab:")

    if query:
        # Display user question
        with question_container:
            st.subheader("Your Question:")
            st.markdown(query)

        # Generate and display response
        with answer_container:
            with st.spinner("Thinking..."):
                try:
                    answer, sources, elapsed_time = run_rag_pipeline(
                        query, source_url)

                    # Display answer
                    st.subheader("Answer:")
                    st.markdown(answer)

                    # Display sources
                    if sources:
                        st.subheader("Sources:")
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"{i}. [{source}]({source})")

                    # Display elapsed time
                    st.markdown(f"*Response time: {elapsed_time:.2f} seconds*")

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error in main process: {e}", exc_info=True)


if __name__ == "__main__":
    main()
