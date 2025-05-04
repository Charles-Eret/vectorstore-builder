# vectorstore-builder
# 💬 Domain-Aware RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on the specific content scraped from a specific domain. It includes two main modules: one for **collecting relevant URLs** from a sitemap, and another for **scraping and embedding** the content into a FAISS vectorstore for fast retrieval.

---

## 🧱 Project Structure

### `urlCollectorModule.py`
A GPT-augmented web crawler that:
- Parses XML sitemaps and traverses the domain
- Uses GPT (via OpenAI API) to filter URLS with relevant content.
- Saves filtered URLs to a text file.

### `vectorstoreModule.py`
A scraping and embedding pipeline that:
- Loads the filtered URLs from the saved file.
- Extracts and embeds the raw text.
- Stores them in a local **FAISS vectorstore** for fast retrieval.

---

## 🚀 How to Use
```bash
pip install -r requirements.txt
playwright install
```
### 1. Collect Relevant URLs
```bash
python urlCollectorModule.py
```
This crawls a sitemap, filters for relevant content using GPT, and saves valid URLs

### 2. Create Vectorstore
```bash
python vectorstoreModule.py
```
This loads the filtered URLs, scrapes content, splits and embeds the documents, and saves the FAISS vectorstore

## 🧠 Next Steps
- clean data before embedding
- captcha bypass functionality
- dynamic page extraction
