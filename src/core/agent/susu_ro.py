from typing import Annotated
from typing_extensions import TypedDict
 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
 
from langchain_ollama.llms import OllamaLLM

from langchain_core.messages import HumanMessage

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

class SusuRo():
    def __init__(self):
        # Create the chat model
        self.llm = OllamaLLM(model="qwen3:4b")

        # Start to build the graph
        self.graph_builder = StateGraph(State)
        
        # Add nodes to the graph
        self.graph_builder.add_node("chatbot_node", self.chatbot_function)
        
        # Add edges
        self.graph_builder.add_edge(START, "chatbot_node")
        self.graph_builder.add_edge("chatbot_node", END)
        
        # Compile the graph
        self.graph = self.graph_builder.compile()
        
        # Display the graph
        try:
            print(self.graph.get_graph().draw_ascii())
        except Exception as e:
            print(f"Error al visualizar el grafo: {e}")
    
    # Function
    def chatbot_function(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}
    
    # Invoke 
    def invoke(self, prompt: str):
        messages = [HumanMessage(content=prompt)]

        result = self.graph.invoke({"messages": messages}, {"number": None})
        
        return result["messages"]


