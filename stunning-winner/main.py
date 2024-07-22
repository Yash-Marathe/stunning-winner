from pprint import pprint
from src.graph.workflow import create_workflow
from src.conversation.manager import ConversationManager

if __name__ == "__main__":
    app = create_workflow()
    conversation_manager = ConversationManager()

    while True:
        question = input("Enter your question (or type 'exit'): ")
        if question.lower() == 'exit':
            break

        inputs = conversation_manager.new_interaction(question)
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
        pprint("\n---\n")
        pprint(value["generation"])
        conversation_manager.add_turn(question, value["generation"])