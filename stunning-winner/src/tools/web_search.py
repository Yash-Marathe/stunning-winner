from langchain_community.tools.tavily_search import TavilySearchResults

def get_web_search_tool():
    """Initialize and return the Tavily web search tool."""
    return TavilySearchResults(k=3)