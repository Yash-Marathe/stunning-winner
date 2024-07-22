import os
from getpass import getpass

# --- Configuration ---

# Prompt for API keys if they are not set as environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or getpass(
    'Enter Open AI API Key: ')
COHERE_API_KEY = os.getenv('COHERE_API_KEY') or getpass(
    'Enter Cohere API Key: ')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY') or getpass(
    'Enter Tavily Search API Key: ')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY') or getpass(
    'Enter LangChain API Key: ')

# Set environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['COHERE_API_KEY'] = COHERE_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Data paths
# Data paths
DATA_PATH = "data/"
FILE_PATHS = [
    os.path.join(DATA_PATH, "HDFC.txt"),
    os.path.join(DATA_PATH, "ICICI.txt"),
    os.path.join(DATA_PATH, "terms.txt"),
]

# Vectorstore params
COLLECTION_NAME = "rag-chroma"
