import logging
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GOOGLE_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for document retrieval."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        self.db = None
        logger.info("Initialized vector store")
    
    def build_from_documents(self, docs):
        """
        Build the vector store from a list of documents.
        
        Args:
            docs (list): List of document chunks
            
        Returns:
            VectorStore: Self, for method chaining
        """
        if not docs:
            logger.warning("No documents provided to build vector store")
            return self
            
        logger.info(f"Building vector store from {len(docs)} documents")
        try:
            self.db = FAISS.from_documents(docs, self.embeddings)
            logger.info("Vector store built successfully")
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            raise
            
        return self
    
    def as_retriever(self, k=3):
        """
        Get a retriever from the vector store.
        
        Args:
            k (int): Number of documents to retrieve
            
        Returns:
            Retriever: Document retriever
        """
        if not self.db:
            raise ValueError("Vector store not initialized. Call build_from_documents first.")
            
        return self.db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k}
        )