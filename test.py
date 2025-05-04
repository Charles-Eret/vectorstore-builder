from urlCollectorModule import extract_data, filter_url_with_gpt
if __name__ == "__main__":
    url = "https://www.sunlife.ca/en/investments/resp/"
    links, title, description = extract_data(url)
    answer = filter_url_with_gpt(url, title, description)
    print(f"title: {title}")
    print(f"description: {description}")
    print(f"answer: {answer}")