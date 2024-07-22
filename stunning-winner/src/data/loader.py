# src/data/loader.py
from langchain_community.document_loaders.text import TextLoader
from config.config import FILE_PATHS  # Import FILE_PATHS


def load_documents():
    docs = []
    for file_path in FILE_PATHS:
        try:
            loaded_docs = TextLoader(file_path).load()
            docs.extend(loaded_docs)  # Ensure correct structure
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return docs
