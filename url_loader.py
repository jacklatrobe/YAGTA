# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

##  url_loader.py - URL Loader functions for YAGTA

# Imports
import logging
import validators
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

# URL Loader Function
def get_url_content(url: str) -> str:
    if validators.url(url) == True:
        try:
            logging.info("get_url_content: Loading page content from {url}")
            loader = AsyncHtmlLoader(url)
            docs = loader.load()
            html2text = Html2TextTransformer()
            doc_transformed = html2text.transform_documents(docs)
            page_content = doc_transformed[0].page_content
            return f"Content from {url}:\n{page_content}"
        except Exception as ex:
            logging.warning(f"get_url_content: Error loading page content from {url}: {str(ex)}")
            return f"Error loading page content from {url}: {str(ex)}"
    else:
        logging.warning (f"{url} is not a valid URL - unable to load page content")
        return f"{url} is not a valid URL - unable to load page content"