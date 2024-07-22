from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config.config import COLLECTION_NAME
from src.data.loader import load_documents
from src.embeddings.embeddings import get_embeddings

def get_retriever():
    """Create and return a Chroma vectorstore retriever."""
    embd = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(load_documents())
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=COLLECTION_NAME,
        embedding=embd,
    )
    return vectorstore.as_retriever()