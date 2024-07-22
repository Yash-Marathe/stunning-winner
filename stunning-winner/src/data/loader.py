from langchain.document_loaders import TextLoader

from config.config import FILE_PATHS

def load_documents():
    """Load documents from the specified file paths."""
    docs = [TextLoader(file_path).load() for file_path in FILE_PATHS]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list