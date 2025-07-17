from typing import Annotated
from typing_extensions import TypedDict
 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
 
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from langsmith import traceable

from core.stt.whisper import Whisper

import os
from dotenv import load_dotenv

load_dotenv()

# Load the environment variables
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# State
class State(TypedDict):
    """
    State of the SusuRo agent.
    """
    messages: Annotated[list, add_messages]

class SusuRo():
    def __init__(self, chat_model: str = "qwen3:4b", chat_reasoning: bool = False, whisper_model_size: str = "large", whisper_device: str = "cpu"):
        """
        Initialize the SusuRo agent.

        Args:
            whisper_model_size (str, optional): The size of the Whisper model. Defaults to "large".
            whisper_device (str, optional): The device to use for the Whisper model. Defaults to "cpu".
        """
        # Create the chat model
        self.llm = ChatOllama(model=chat_model, reasoning=chat_reasoning)

        # Create the stt model
        self.stt = Whisper(model_size=whisper_model_size, device=whisper_device)

        # Create tools list
        self.tools_list = [self.stt.transcribe]

        # Create LLM with tools
        self.llm_with_tools = self.llm.bind_tools(self.tools_list)

        # Create tool node
        stt_tool_node = ToolNode(tools=self.tools_list)

        # Start to build the graph
        self.graph_builder = StateGraph(State)
        
        # Add nodes to the graph
        self.graph_builder.add_node("chatbot_node", self.chatbot_function)
        self.graph_builder.add_node("stt_tool", stt_tool_node)

        # Add edges
        self.graph_builder.add_edge(START, "chatbot_node")
        self.graph_builder.add_conditional_edges(
            "chatbot_node",
            tools_condition,
            {
                "tools": "stt_tool",  # Cuando hay tool calls, va a stt_tool
                "__end__": END,       # Cuando no hay tool calls, termina
            }
        )
        self.graph_builder.add_edge("stt_tool", "chatbot_node")

        # Compile the graph
        self.graph = self.graph_builder.compile()
        
        # Display the graph
        try:
            print(self.graph.get_graph().draw_ascii())
        except Exception as e:
            print(f"Error al visualizar el grafo: {e}")
    
    # Functions
    @traceable(run_type="llm")
    def chatbot_function(self, state: State):
        input_messages = state["messages"]
        response = self.llm_with_tools.invoke(input_messages)
        print(input_messages)
        print(response)
        return {"messages": [self.llm_with_tools.invoke(state["messages"])]}
    
    @traceable(run_type="tool")
    def stt_function(self, state: State):
        return {"messages": [self.stt.transcribe(state["messages"])]}
    
    # Invoke 
    def invoke(self, prompt: str):
        messages = [HumanMessage(content=prompt)]

        result = self.graph.invoke({"messages": messages}, {"number": None})
        
        return result["messages"]


