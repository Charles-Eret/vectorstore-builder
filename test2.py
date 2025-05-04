from vectorstoreModule import load_document_from_url, extract_text_with_playwright

docs = load_document_from_url("https://www.sunlife.ca/en/insurance/life/term-vs-perm/")
print(docs)