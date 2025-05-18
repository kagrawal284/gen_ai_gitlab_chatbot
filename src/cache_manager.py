import os
import json
import logging
from typing import Optional

# Import from utils - utils doesn't import from cache_manager
from src.utils import hash_url, ensure_directory_exists

# Import constants directly, not the whole module
from config import PAGE_CACHE_DIR, EMBEDDING_CACHE_DIR

logger = logging.getLogger(__name__)

class CacheManager:
    """Manager for different types of caches."""
    
    @staticmethod
    def initialize():
        """Initialize cache directories."""
        ensure_directory_exists(PAGE_CACHE_DIR)
        ensure_directory_exists(EMBEDDING_CACHE_DIR)
        logger.info("Cache directories initialized")

class PageCache:
    """Cache for web page content."""
    
    @staticmethod
    def save(url, text):
        """Save page content to cache."""
        try:
            ensure_directory_exists(PAGE_CACHE_DIR)
            path = os.path.join(PAGE_CACHE_DIR, f"{hash_url(url)}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.debug(f"Saved page cache for {url}")
        except Exception as e:
            logger.error(f"Failed to save cache for {url}: {e}")

    @staticmethod
    def load(url) -> Optional[str]:
        """Load page content from cache."""
        try:
            path = os.path.join(PAGE_CACHE_DIR, f"{hash_url(url)}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    logger.debug(f"Loaded page cache for {url}")
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to load cache for {url}: {e}")
        return None

class EmbeddingCache:
    """Cache for embeddings."""
    
    @staticmethod
    def save(url, embedding):
        """Save embedding to cache."""
        try:
            ensure_directory_exists(EMBEDDING_CACHE_DIR)
            path = os.path.join(EMBEDDING_CACHE_DIR, f"{hash_url(url)}.emb")
            with open(path, "w") as f:
                json.dump(embedding, f)
            logger.debug(f"Saved embedding cache for {url}")
        except Exception as e:
            logger.error(f"Failed to save embedding cache for {url}: {e}")

    @staticmethod
    def load(url) -> Optional[list]:
        """Load embedding from cache."""
        try:
            path = os.path.join(EMBEDDING_CACHE_DIR, f"{hash_url(url)}.emb")
            if os.path.exists(path):
                with open(path, "r") as f:
                    logger.debug(f"Loaded embedding cache for {url}")
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load embedding cache for {url}: {e}")
        return None