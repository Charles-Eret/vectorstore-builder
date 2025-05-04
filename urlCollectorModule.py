# This module implements an intelligent web crawler that collects relevant URLs from a website's sitemap. 
# It only collects English-language content related to investing and insurance.
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from vectorstoreModule import extract_text_from_pdf

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def filter_url_with_gpt(url, title, description, model="gpt-3.5-turbo"):
    """
    Uses GPT to evaluate if a URL's content is relevant based on its metadata.
    
    Args:
        url (str): The URL to evaluate
        title (str): Page title
        description (str): Meta description
        model (str): GPT model to use
        
    Returns:
        bool: True if content is in English and related to investing/insurance
    """

    prompt = f"""
    Given the following webpage information, determine if it is relevant to investing or insurance **and** if the content is likely in English. Respond with "yes" if both conditions are met, otherwise respond with "no".

    URL: {url}
    Title: {title}
    Description: {description}
    """.strip()

    #print(f"prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        #print(f"answer: {answer}")
        return answer in {"yes", "yes."}
    
    except Exception as e:
        print(f"[WARN] GPT filtering failed for {url}: {e}")
        return True # Default to True to avoid missing URLs
    
def normalize_url(url):
    """
    Standardizes URLs by converting to lowercase, removing query parameters and fragments.
    
    Args:
        url (str): URL to normalize
        
    Returns:
        str: Normalized URL without query strings and fragments
    """

    parsed = urlparse(url)
    # Normalize scheme and netloc to lowercase, strip trailing slashes from path
    normalized_path = parsed.path
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=normalized_path,
        query='',  # Optional: remove query strings to treat as same URL
        fragment=''  # Optional: remove fragments like #section
    )

    return urlunparse(normalized)

def is_valid_url(url, domain):
    """
    Validates if a URL belongs to the target domain and uses http(s) protocol.
    
    Args:
        url (str): URL to validate
        domain (str): Target domain to check against
        
    Returns:
        bool: True if URL is valid and belongs to domain
    """

    parsed = urlparse(url)
    return parsed.netloc.endswith(domain) and parsed.scheme in {"http", "https"}

def extract_data(url):
    """
    Extracts links, title, and description from a webpage or PDF.
    Handles both HTML pages and PDF documents differently.
    
    Args:
        url (str): URL to extract data from
        
    Returns:
        tuple: (list of links, title, description)
    """

    try:
        # Check if the URL is a PDF
        if url.lower().endswith(".pdf"):
            documents = extract_text_from_pdf(url)
            return [], "", documents[0].page_content[:500] if documents else ""
        
        # Else it is a webpage
        resp = requests.get(
            url, 
            headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/114.0.0.0 Safari/537.36"
            }, 
            timeout=10
        )

        if resp.status_code in {403, 429}:
            print(f"[WARN] Skipped {url} due to status code {resp.status_code}")
            return [], "", ""

        soup = BeautifulSoup(resp.content, "lxml")

        # Extract links
        links = [a.get("href") for a in soup.find_all("a", href=True)]

        # Extract metadata
        title = soup.title.string.strip() if soup.title else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag["content"].strip() if desc_tag and "content" in desc_tag.attrs else ""

        if not title and not description:
            print(f"[WARN] No title or description found for {url}")
        
        return links, title, description
    
    except Exception as e:
        print(f"[WARN] Failed to crawl {url}: {e}")
        return [], "", ""

def parse_sitemap(sitemap_url):
    """
    Retrieves and parses XML sitemap to extract URLs.
    Uses session-based requests with browser-like headers.
    
    Args:
        sitemap_url (str): URL of the sitemap
        
    Returns:
        list: URLs found in sitemap
    """

    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/xml, text/xml, application/xhtml+xml, text/html;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }
        
        resp = session.get(sitemap_url, headers=headers, timeout=10)
        #print(f"[DEBUG] Sitemap response status code: {resp.status_code}")
        #print(f"[DEBUG] Response headers: {dict(resp.headers)}")
        
        if resp.status_code != 200:
            print(f"[ERROR] Failed to fetch sitemap. Status code: {resp.status_code}")
            return []
        
        soup = BeautifulSoup(resp.content, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        #print(f"[INFO] Found {len(urls)} URLs in sitemap")
        return urls
    
    except Exception as e:
        print(f"[ERROR] Failed to parse sitemap: {e}")
        return []

def crawl_domain(sitemap_url, max_pages):
    """
    Performs breadth-first crawl of a domain starting from sitemap URLs.
    Filters relevant content using GPT and tracks progress.
    
    Args:
        sitemap_url (str): Starting sitemap URL
        max_pages (int): Maximum number of pages to crawl
        
    Returns:
        list: Filtered relevant URLs
    """

    start_urls = parse_sitemap(sitemap_url)
    if not start_urls:
        print(f"[ERROR] No URLs found in the sitemap: {sitemap_url}")
        return []

    domain = urlparse(sitemap_url).netloc.lower()
    visited = set()
    queue = deque(start_urls[:])  # BFS queue
    relevant_urls = []  # Store filtered URLs

    print(f"🌐 Starting crawl on domain: {domain}")
    with tqdm(total=max_pages) as pbar:
        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            norm_url = normalize_url(current_url)
            if norm_url in visited:
                continue
            visited.add(norm_url)
            pbar.update(1)

            # Extract links and metadata
            try:
                links, title, description = extract_data(current_url)

                # Filter URL using GPT
                if filter_url_with_gpt(current_url, title, description):
                    relevant_urls.append(norm_url)

                # Enqueue links for further crawling
                for link in links:
                    full_url = urljoin(current_url, link)
                    norm_full_url = normalize_url(full_url)
                    if is_valid_url(norm_full_url, domain) and norm_full_url not in visited and norm_full_url not in queue:
                        queue.append(norm_full_url)
            except Exception as e:
                print(f"[WARN] Failed to process {current_url}: {e}")
                continue
            
            # Update tqdm description dynamically
            pbar.set_description(f"Visited: {len(visited)} | Relevant: {len(relevant_urls)}")

            time.sleep(0.2)  # Be respectful

    print(f"\n✅ Collected {len(visited)} unique URLs.\n")
    print(f"✅ Collected {len(relevant_urls)} relevant URLs")

    return relevant_urls

def collect_urls(sitemap_url, output_file_path, max_pages=100000):
    """
    Main entry point for URL collection process.
    Crawls domain, filters URLs, and saves results to file.
    
    Args:
        sitemap_url (str): Starting sitemap URL
        output_file_path (str): Path to save filtered URLs
        max_pages (int): Maximum pages to crawl (default: 100000)
    """
    
    # Check if writing to the output file is possible
    try:
        with open(output_file_path, "w") as test_file:
            print(f"Able to write to the output file '{output_file_path}'!")
            pass  # Just testing if the file can be opened for writing
    except Exception as e:
        print(f"[ERROR] Unable to write to the output file '{output_file_path}': {e}")
        return
    
    filtered_urls = crawl_domain(sitemap_url, max_pages)
    with open(output_file_path, "w") as f:
        for url in sorted(filtered_urls):
            f.write(url + "\n")
    print(f"✔ URLs saved to {output_file_path}")
    
if __name__ == "__main__":
    collect_urls("https://www.sunlife.com/en/sitemap.xml",r"filtered_urls\sunlife_urls.txt")
