import logging
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.api_core.exceptions import ResourceExhausted, Unauthorized, Forbidden

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, TOTAL_EMBEDDING_BUDGET
from src.cache_manager import EmbeddingCache

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manager for embedding generation and handling."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info(f"Initialized embedding manager with model: {EMBEDDING_MODEL}")
    
    def embed_query(self, query):
        """
        Generate an embedding for a query.
        
        Args:
            query (str): The query text
            
        Returns:
            list: The embedding vector
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def embed_document(self, text):
        """
        Generate an embedding for a document.
        
        Args:
            text (str): The document text
            
        Returns:
            list: The embedding vector
        """
        try:
            # Using embed_query as it's meant for single items
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            raise

    def rank_links_by_query_relevance(self, query, link_data, top_k=20, min_links=10):
        """
        Rank links by relevance to a query using embeddings.
        
        Args:
            query (str): The query text
            link_data (list): List of dictionaries with 'url' and 'context' keys
            top_k (int): Number of top links to return
            min_links (int): Minimum number of links to return
            
        Returns:
            list: List of URLs ranked by relevance
        """
        try:
            # First rank all links using naive keyword matching
            from src.crawling import naive_relevance_score
            scored = [(naive_relevance_score(link["context"], query), link) for link in link_data]
            scored.sort(reverse=True, key=lambda x: x[0])
            
            # Select up to max_docs links based on naive score to respect API limits
            max_docs = TOTAL_EMBEDDING_BUDGET - 1  # Leave 1 for query
            selected_links = [link for score, link in scored[:max_docs]]
            
            # Backfill if needed to reach min_links
            if len(selected_links) < min_links:
                logger.warning(f"Only {len(selected_links)} relevant links found, backfilling to minimum {min_links}")
                selected_urls = set(link["url"] for link in selected_links)
                
                for link in link_data:
                    if link["url"] not in selected_urls:
                        selected_links.append(link)
                        if len(selected_links) >= min_links:
                            break
            
            # Embed query
            query_embedding = self.embed_query(query)
            scored_links = []
            
            # Process each selected link
            for link in selected_links:
                # Check if embedding is cached
                cached_vec = EmbeddingCache.load(link["url"])
                
                if cached_vec is None:
                    # Generate new embedding
                    emb = self.embed_document(link["context"])
                    EmbeddingCache.save(link["url"], emb)
                else:
                    emb = cached_vec
                
                # Calculate similarity score
                score = cosine_similarity([query_embedding], [emb])[0][0]
                scored_links.append((score, link["url"]))
            
            # Sort by score and return top_k
            top_links = sorted(scored_links, reverse=True)[:top_k]
            return [url for score, url in top_links]
            
        except Unauthorized:
            logger.error("API key unauthorized or expired. Please check your API key.")
            return []
        except Forbidden:
            logger.error("Access forbidden. Check your API key permissions.")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred in ranking: {e}")
            return []