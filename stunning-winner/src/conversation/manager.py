from src.utils.state import GraphState

class ConversationManager:
    """Manages the conversation history and state."""
    def __init__(self):
        self.history = []

    def add_turn(self, question, answer):
        """Add a turn to the conversation history."""
        self.history.append({"question": question, "answer": answer})

    def get_context(self):
        """Get the current conversation history."""
        return self.history

    def new_interaction(self, question) -> GraphState:
        """Initiate a new interaction with the user."""
        initial_state = GraphState(
            question=question,
            generation="",
            documents=[],
            history=self.history.copy(),
        )
        return initial_state