import logging
import time
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted, InternalServerError

from config import GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS, RETRIEVER_K

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG (Retrieval Augmented Generation) chain."""

    def __init__(self, vectorstore):
        """
        Initialize the RAG chain.

        Args:
            vectorstore (VectorStore): Vector store for document retrieval
        """
        self.retriever = vectorstore.as_retriever(k=RETRIEVER_K)
        self.chain = self.create_chain()
        logger.info(f"Initialized RAG chain with model: {LLM_MODEL}")

    def create_chain(self):
        """
        Create the RAG chain.

        Returns:
            Chain: The complete RAG chain
        """
        # Create the system prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant for question-answering tasks about GitLab's handbook and direction. 
                          Use the following pieces of retrieved context to answer the question. 
                          If you don't know the answer, say you don't know. 
                          Keep your answers concise and focused on the question.

                        Context:
                                {context}"""),
            ("human", "{input}")
        ])

        # Create the LLM
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            google_api_key=GOOGLE_API_KEY
        )

        # Create the document chain
        qa_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retrieval chain
        return create_retrieval_chain(self.retriever, qa_chain)

    def invoke(self, query):
        """
        Invoke the RAG chain with a query.

        Args:
            query (str): The query text

        Returns:
            dict: Response containing answer and context documents
        """
        return self.safe_invoke(query)

    def invoke_once(self, query):
        """Single attempt to invoke the chain without retries."""
        return self.chain.invoke({"input": query})

    def safe_invoke(self, query):
        """
        Safely invoke the RAG chain with retry logic.

        Args:
            query (str): The query text

        Returns:
            dict: Response containing answer and context documents, or error message
        """
        delay = 1  # Start with 1 second delay
        max_delay = 60  # Maximum delay between retries (60 seconds)

        for attempt in range(3):
            try:
                logger.info(f"Running query (attempt {attempt+1}): {query}")

                result = self.invoke_once(query)

                logger.info(f"Query successful: {query}")
                return result

            except ResourceExhausted:
                logger.warning(
                    "Quota exhausted. Waiting 60s before retrying...")
                time.sleep(60)

            except InternalServerError:
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = delay + jitter

                logger.warning(
                    f"Internal server error. Waiting {sleep_time:.2f}s before retrying...")
                time.sleep(sleep_time)

                # Exponential backoff
                delay = min(delay * 2, max_delay)
            except Exception as e:
                logger.error(f"Error during invocation: {e}")
                return {
                    "answer": f"An error occurred: {str(e)}",
                    "context": []
                }

        return {
            "answer": "Failed to get a response after multiple attempts. Please try again later.",
            "context": []
        }

    def format_results(self, result):
        """
        Format the results for display.

        Args:
            result (dict): The result from the RAG chain

        Returns:
            tuple: (answer, sources)
        """
        answer = result.get("answer", "No answer generated")

        # Extract sources
        sources = []
        context_docs = result.get("context", [])
        if context_docs:
            for doc in context_docs:
                source = doc.metadata.get("source", "Unknown source")
                sources.append(source)

        return answer, sources
