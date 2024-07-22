from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    """Initialize and return OpenAI embeddings."""
    return OpenAIEmbeddings()