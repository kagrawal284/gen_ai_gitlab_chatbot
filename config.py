import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Web crawling settings
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; CustomBot/1.0)")
HEADERS = {
    "User-Agent": USER_AGENT
}

# Data directory settings
CACHE_DIR = "cache"
PAGE_CACHE_DIR = os.path.join(CACHE_DIR, "pages")
EMBEDDING_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")

# Crawling limits
MAX_LINKS_PER_SITE = 250  # Maximum links to extract from each main page
TOP_K_LINKS = 20  # Number of top ranked links to process
TOTAL_EMBEDDING_BUDGET = 30  # API call limit for embeddings

# Document processing
CHUNK_SIZE = 500  # Size of text chunks for splitting
CHUNK_OVERLAP = 100  # Overlap between chunks

# RAG settings
RETRIEVER_K = 3  # Number of documents to retrieve
TEMPERATURE = 0.3  # LLM temperature
MAX_TOKENS = 100  # Maximum tokens for response

# Models
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"

# Sources
MAIN_URLS = [
    "https://handbook.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/"
]

# Parallel processing
MAX_WORKERS = 5  # Maximum number of parallel workers for document loading
