from typing_extensions import TypedDict
from typing import List, Dict

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    documents: List[Dict]
    history: List[Dict[str, str]]