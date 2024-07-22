from langchain.schema import Document
from src.tools.web_search import get_web_search_tool

def web_search(state):
    """Perform a web search using the provided question."""
    print("---WEB SEARCH---")
    question = state["question"]
    docs = get_web_search_tool().invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {
        "documents": web_results,
        "question": question,
        "history": state["history"]
    }