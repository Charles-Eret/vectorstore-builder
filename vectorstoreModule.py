# This module embeds the given txt file containing URLs into a vectorstore using LangChain and FAISS.
import time
import os
import logging
import random
import requests
from dotenv import load_dotenv
import tempfile
from playwright.sync_api import sync_playwright
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import fitz  # PyMuPDF
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def load_urls_from_file(file_path):
    """
    Reads URLs from a text file, stripping whitespace and empty lines.
    
    Args:
        file_path (str): Path to the file containing URLs
        
    Returns:
        list: List of cleaned URLs
    """

    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading the URL file: {e}")
        return []

def needs_js_rendering(page_content):
    """
    Determines if a webpage requires JavaScript rendering based on its content.
    Checks for suspicious phrases that indicate JS dependencies or access restrictions.
    
    Args:
        page_content (str): Raw HTML content of the page
        
    Returns:
        bool: True if the page needs JavaScript rendering
    """

    if page_content is None or page_content.strip() is None:
        return True  # Empty content indicates a potential issue
    suspicious_phrases = [
        "javascript",
        "access denied",
        "cloudflare",
        "robot check",
        "js"
    ]
    content_lower = page_content.lower()
    if any(phrase in content_lower for phrase in suspicious_phrases):
        return True
    return False

def extract_text_with_playwright(url, selector="body"):
    """
    Extracts text from a URL using Playwright and waits for a specific element to load.
    
    Args:
        url (str): The URL to scrape.
        selector (str): The CSS selector to wait for before scraping.
    
    Returns:
        List[Document]: A list of Document objects with the page content and metadata.
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # Set headless=False for debugging
            page = browser.new_page()
            page.goto(url, timeout=60000)  # Wait for the page to load (60 seconds timeout)
            
            # Wait for the specific element to load
            page.wait_for_selector(selector, timeout=30000)  # Wait for up to 30 seconds
            
            # Get the fully rendered HTML content
            content = page.content()
            title = page.title()
            browser.close()

        # Return the content as a Document object
        return [Document(page_content=content, metadata={"source": url, "title": title})]
    except Exception as e:
        print(f"Failed to load {url} with Playwright: {e}")
        return []

def extract_text_from_dynamic_url(url):
    """
    Uses PlaywrightURLLoader for URLs requiring JavaScript rendering.
    Serves as a fallback for the basic WebBaseLoader.
    
    Args:
        url (str): URL to process
        
    Returns:
        List[Document] or None: Extracted documents or None if failed
    """

    try:
        loader = PlaywrightURLLoader(
            urls=[url],
        )
        docs = loader.load()
        return docs
    except Exception as e:
        logger.warning(f"Failed to load {url} with PlayWright: {e}")
        return None

def extract_text_from_pdf(url):
    """
    Downloads and extracts text from PDF files using PyMuPDF.
    Handles temporary file creation and cleanup.
    
    Args:
        url (str): URL of the PDF file
        
    Returns:
        List[Document]: Documents containing PDF text and page metadata
    """

    try:
        # 1. Download the PDF
        response = requests.get(
            url, 
            headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/114.0.0.0 Safari/537.36"
            }
        )

        response.raise_for_status()

        # 2. Create a temporary file and write the downloaded content
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name

        # 3. Open the PDF using fitz
        doc = fitz.open(tmp_pdf_path)
        documents = []

        # 4. Extract text from the PDF pages
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": url, "page": page_num + 1}
                ))

        # Explicitly close the temporary file here to avoid deletion before usage
        #tmp_file.close()

        doc.close()

        return documents

    except Exception as e:
        logger.warning(f"Failed to load {url} with fitz: {e}")
        return []
    finally:
        # Clean up the temp file if it exists
        if os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)

def extract_text_from_url(url):
    """
    Extracts text from webpages using WebBaseLoader with JS fallback.
    Detects if JavaScript rendering is needed and switches to dynamic loader.
    
    Args:
        url (str): URL to process
        
    Returns:
        List[Document]: Extracted documents
    """

    try:
        docs = WebBaseLoader(url).load()
        if not docs:
            logger.warning(f"No documents found for {url}.")
            return []
        content = docs[0].page_content
        if needs_js_rendering(content):
            # fallback to heavy
            print(f"Falling back to Selenium for {url}")
            docs = extract_text_from_dynamic_url(url)
            print(f"need js rendering for {url}")
        return docs
    except Exception as e:
        print(f"Failed to load webpage {url}: {e}")
        return []

# --- Classify and load content based on type ---
def load_document_from_url(url):
    """
    Routes URLs to appropriate handler based on file type.
    Handles both PDF files and web pages differently.
    
    Args:
        url (str): URL to process
        
    Returns:
        List[Document] or None: Processed documents or None if failed
    """

    try:
        if url.lower().endswith(".pdf"):
            return extract_text_from_pdf(url)
        else:
            return extract_text_from_url(url)
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return None

# --- Embedding a single URL ---
def process_url(url):
    """
    Processes a single URL with rate limiting.
    Loads document and splits into chunks for embedding.
    
    Args:
        url (str): URL to process
        
    Returns:
        List[Document]: Chunked documents ready for embedding
    """

    time.sleep(random.uniform(0.2, 0.5))
    #logger.info(f"Processing URL: {url}")
    docs = load_document_from_url(url)
    if docs:
        for doc in docs:
            doc.metadata['source'] = url
        # Split documents into chunks
        return TEXT_SPLITTER.split_documents(docs)
    return []

def process_url_with_retries(url, retries=3):
    """
    Implements retry logic for URL processing with exponential backoff.
    
    Args:
        url (str): URL to process
        retries (int): Number of retry attempts
        
    Returns:
        List[Document]: Processed document chunks
    """

    for attempt in range(retries):
        try:
            return process_url(url)
        except Exception as e:
            logger.warning(f"Retry {attempt + 1}/{retries} failed for {url}: {e}")
            time.sleep(2** attempt)  # Wait before retrying
    logger.error(f"Failed to process {url} after {retries} retries.")
    return []

# --- Main embedding pipeline ---
def embed_urls(input_file_path, output_vectorstore_path, batch_size=10):
    """
    Main pipeline for embedding URLs into FAISS vectorstore.
    Handles concurrent processing, batching, and vectorstore updates.
    
    Args:
        input_file_path (str): Path to file containing URLs
        output_vectorstore_path (str): Path to save the vectorstore
        batch_size (int): Number of URLs to process before updating vectorstore
    """

    urls = load_urls_from_file(input_file_path)
    all_chunks = []
    processed_count = 0

    vectorstore = None

    def update_vectorstore(chunks):
        """
        Updates FAISS vectorstore with new document chunks.
        Creates new vectorstore or merges with existing one.
        
        Args:
            chunks (List[Document]): Document chunks to add to vectorstore
        """
        nonlocal vectorstore
        if not chunks:
            return
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        else:
            new_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
            vectorstore.merge_from(new_store)
        try:
            vectorstore.save_local(output_vectorstore_path)
            logger.info(f"💾 Vectorstore updated with {len(chunks)} new chunks.")
        except Exception as e:
            logger.error(f"Failed to update vectorstore: {e}")
            

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_url_with_retries, url): url for url in urls}
        try:
            for future in tqdm(as_completed(futures), total=len(urls), desc="Processing URLs"):
                url = futures[future]
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_count += 1

                    if processed_count % batch_size == 0:
                        update_vectorstore(all_chunks)
                        all_chunks.clear()

                except Exception as e:
                    logger.warning(f"Error processing {url}: {e}")

        except KeyboardInterrupt:
            logger.warning("🛑 KeyboardInterrupt detected. Attempting graceful shutdown...")

            # Cancel futures that haven't started yet
            for f in futures.keys():
                if not f.done():
                    f.cancel()
        finally:
            # Final save for remaining chunks
            if all_chunks:
                update_vectorstore(all_chunks)
            logger.info(f"✅ Final vectorstore save completed. Total URLs processed: {processed_count}")


if __name__ == "__main__":
    load_dotenv()
    embed_urls("filtered_urls/sunlife_urls_test.txt", "vectorstores/sunlife")
