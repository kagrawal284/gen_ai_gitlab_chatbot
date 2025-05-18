import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS
from src.cache_manager import PageCache

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processor for loading and chunking documents."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info(f"Initialized document processor with chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    
    def load_single_url(self, url):
        """
        Load and chunk a document from a single URL.
        
        Args:
            url (str): URL to load
            
        Returns:
            list: List of document chunks
        """
        # Check cache first
        cached = PageCache.load(url)
        if cached:
            logger.info(f"Loaded {url} from cache")
            doc = Document(page_content=cached, metadata={"source": url})
            return self.text_splitter.split_documents([doc])
        
        # If not cached, load from web
        try:
            logger.info(f"Loading {url} from web")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Cache the full document
            full_text = "\n\n".join(doc.page_content for doc in docs)
            PageCache.save(url, full_text)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Loaded {url} with {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load {url}: {e}")
            return []
    
    def load_and_split_documents_parallel(self, urls):
        """
        Load and chunk documents from multiple URLs in parallel.
        
        Args:
            urls (list): List of URLs to load
            
        Returns:
            list: Combined list of document chunks from all URLs
        """
        all_docs = []
        logger.info(f"Loading {len(urls)} URLs with max {MAX_WORKERS} workers")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.load_single_url, url): url for url in urls}
            
            for future in as_completed(futures):
                url = futures[future]
                try:
                    chunks = future.result()
                    all_docs.extend(chunks)
                    logger.info(f"Processed {url}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        logger.info(f"Total document chunks loaded: {len(all_docs)}")
        return all_docs