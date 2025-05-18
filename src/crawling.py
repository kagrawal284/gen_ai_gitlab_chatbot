import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import HEADERS

logger = logging.getLogger(__name__)

def extract_link_contexts(main_url, max_links=250):
    """
    Extract links and their surrounding context from a web page.
    
    Args:
        main_url (str): URL of the page to extract links from
        max_links (int): Maximum number of links to extract
        
    Returns:
        list: List of dictionaries with 'url' and 'context' keys
    """
    logger.info(f"Extracting links from {main_url}")
    
    try:
        resp = requests.get(main_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch main page: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    link_data = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = urljoin(main_url, a["href"])
        
        # Skip non-http links and already seen links
        if not href.startswith("http") or href in seen:
            continue
            
        # Extract context (link text and title)
        text = a.get_text(strip=True)
        title = a.get("title", "")
        context = " ".join([text, title, href])
        
        seen.add(href)
        link_data.append({"url": href, "context": context})

        if len(link_data) >= max_links:
            logger.info(f"Reached max links limit ({max_links})")
            break

    logger.info(f"Extracted {len(link_data)} links from {main_url}")
    return link_data

def naive_relevance_score(context, query):
    """
    Calculate a simple relevance score based on keyword overlap.
    
    Args:
        context (str): The context text
        query (str): The query text
        
    Returns:
        int: Number of overlapping keywords
    """
    query_keywords = set(query.lower().split())
    context_words = set(context.lower().split())
    return len(query_keywords & context_words)