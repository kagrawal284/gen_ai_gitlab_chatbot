import os
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='(%(levelname)s) - [%(funcName)s] - %(message)s'
)

logger = logging.getLogger(__name__)

def hash_url(url):
    """Create a hash of a URL for caching purposes."""
    return hashlib.md5(url.encode()).hexdigest()

def hash_text(text):
    """Create a hash of text content."""
    return hashlib.md5(text.encode()).hexdigest()

def ensure_directory_exists(directory):
    """Ensure a directory exists, creating it if necessary."""
    if os.path.exists(directory):
        if not os.path.isdir(directory):
            logger.warning(f"Removing file {directory} to create directory")
            os.remove(directory)
            
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")