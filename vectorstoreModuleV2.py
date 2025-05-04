import time
import logging
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
OUTPUT_VECTORSTORE_PATH = "vectorstores/sunflife"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load list of URLs from text file ---
def load_urls_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading the URL file: {e}")
        return []

# --- Classify and load content based on type ---
def load_document_from_url(url):
    try:
        if url.lower().endswith(".pdf"):
            return UnstructuredPDFLoader(url).load()
        else:
            return WebBaseLoader(url).load()
    except Exception as e:
        logger.warning(f"Failed to load {url}: {e}")
        return None

# --- Embedding a single URL ---
def process_url(url):
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
    for attempt in range(retries):
        try:
            return process_url(url)
        except Exception as e:
            logger.warning(f"Retry {attempt + 1}/{retries} failed for {url}: {e}")
            time.sleep(2** attempt)  # Wait before retrying
    logger.error(f"Failed to process {url} after {retries} retries.")
    return []

# --- Main embedding pipeline ---
def embed_urls(file_path, batch_size=10):
    urls = load_urls_from_file(file_path)
    all_chunks = []
    processed_count = 0

    # Check if a vectorstore already exists to append to
    """
    if os.path.exists(f"{OUTPUT_VECTORSTORE_PATH}/index.faiss"):
        logger.info("Loading existing vectorstore for appending...")
        vectorstore = FAISS.load_local(OUTPUT_VECTORSTORE_PATH, EMBEDDING_MODEL)
    else:
        vectorstore = None
    """

    vectorstore = None

    def update_vectorstore(chunks):
        nonlocal vectorstore
        if not chunks:
            return
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        else:
            new_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
            vectorstore.merge_from(new_store)
        try:
            vectorstore.save_local(OUTPUT_VECTORSTORE_PATH)
            logger.info(f"💾 Vectorstore updated with {len(chunks)} new chunks.")
        except Exception as e:
            logger.error(f"Failed to update vectorstore: {e}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_url_with_retries, url): url for url in urls}

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

    # Final save for remaining chunks
    if all_chunks:
        update_vectorstore(all_chunks)

    logger.info(f"✅ Final vectorstore save completed. Total URLs processed: {processed_count}")


if __name__ == "__main__":
    load_dotenv()
    embed_urls("filtered_urls/sunlife_urls_test.txt")
