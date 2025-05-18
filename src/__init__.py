"""Package initialization file."""
import logging
logger = logging.getLogger(__name__)

# Initialize cache in lazy imported function to avoid circular imports
def initialize_cache():
    from src.cache_manager import CacheManager
    from config import PAGE_CACHE_DIR, EMBEDDING_CACHE_DIR
    CacheManager.initialize()
    logger.info("Initialized cache directories")

# Run initialization
initialize_cache()