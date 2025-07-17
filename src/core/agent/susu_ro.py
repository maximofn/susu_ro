from typing import Annotated
from typing_extensions import TypedDict
 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
 
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

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
    def __init__(self, chat_model: str = "qwen3:4b", chat_reasoning: bool = False, whisper_model_size: str = "large", whisper_device: str = "cpu") -> None:
        """
        Initialize the SusuRo agent.

        Args:
            chat_model (str, optional): The model to use for the chat. Defaults to "qwen3:4b".
            chat_reasoning (bool, optional): Whether to use reasoning for the chat. Defaults to False.
            whisper_model_size (str, optional): The size of the Whisper model. Defaults to "large".
            whisper_device (str, optional): The device to use for the Whisper model. Defaults to "cpu".

        Returns:
            None
        """
        # Create llm model
        self.llm = ChatOllama(model=chat_model, reasoning=chat_reasoning)

        # Create the stt model
        self.stt = Whisper(model_size=whisper_model_size, device=whisper_device)

        # Create tools list
        self.tools_list = [self.stt.transcribe]

        # Create LLM with tools
        self.transcriptor = self.llm.bind_tools(self.tools_list)

        # Create tool node
        stt_tool_node = ToolNode(tools=self.tools_list)

        # Start to build the graph
        self.graph_builder = StateGraph(State)
        
        # Add nodes to the graph
        self.graph_builder.add_node("transcriptor_node", self.transcriptor_function)
        self.graph_builder.add_node("evaluator_node", self.evaluator_function)
        self.graph_builder.add_node("stt_tool", stt_tool_node)

        # Add edges
        self.graph_builder.add_edge(START, "transcriptor_node")
        self.graph_builder.add_conditional_edges(
            "transcriptor_node",
            tools_condition,
            {
                "tools": "stt_tool",           # When there are tool calls, go to stt_tool
                "__end__": "evaluator_node",   # When there are no tool calls, go to evaluator_node
            }
        )
        self.graph_builder.add_edge("stt_tool", "transcriptor_node")
        self.graph_builder.add_conditional_edges(
            "evaluator_node",
            self.evaluator_decision,
            {
                "retry": "transcriptor_node",  # If transcription needs retry, go back to transcriptor
                "good": END,                   # If transcription is good, end the graph
            }
        )

        # Compile the graph
        self.graph = self.graph_builder.compile()
        
        # Display the graph
        try:
            print(self.graph.get_graph().draw_ascii())
        except Exception as e:
            print(f"Error al visualizar el grafo: {e}")
    
    # Functions
    @traceable(run_type="llm")
    def transcriptor_function(self, state: State) -> State:
        """
        Handles audio transcription with dual behavior based on input source.
        
        This function operates in two modes:
        1. **Literal Mode**: When receiving transcription results from the STT tool,
           returns the exact transcription without modifications to preserve accuracy.
        2. **Correction Mode**: When receiving feedback from the evaluator indicating
           the transcription needs improvement, applies context-aware corrections to
           fix technical terms, acronyms, and phonetic errors.

        Args:
            state (State): The current state containing message history. The function
                          analyzes the last message to determine its source (tool vs evaluator).

        Returns:
            State: Updated state with either literal transcription or corrected text,
                   depending on the input source.
                   
        Examples:
            - Tool input: Returns Whisper transcription exactly as provided
            - Evaluator retry: Corrects "rac" → "RAG", "sequel" → "SQL", etc.
        """
        input_messages = state["messages"]
        
        # Check if the last message is from a tool (ToolMessage) or from evaluator
        last_message = input_messages[-1]
        is_from_tool = hasattr(last_message, 'tool_call_id')
        
        if is_from_tool:
            # If coming from tool, return literal transcription
            literal_prompt = SystemMessage(content="""You are an audio transcriptor. Your role is to return the literal transcription from the tool.

When you receive a transcription result from the transcribe tool, you must:
1. Return the transcribed text exactly as provided by the tool
2. Do not modify, correct, or interpret the text
3. Simply format it for readability without changing the content
4. Preserve the original transcription accuracy

Return only the literal transcription text.""")
            
            messages_with_system = [literal_prompt] + input_messages
            response = self.transcriptor.invoke(messages_with_system)
            return {"messages": [response]}
        else:
            # Check if this is the initial request to transcribe audio
            user_message = input_messages[-1]
            if hasattr(user_message, 'content') and 'transcribe' in user_message.content.lower():
                # This is an initial transcription request, use the transcriptor with tools
                system_prompt = SystemMessage(content="""You are an audio transcriptor with access to transcription tools. When asked to transcribe audio, you must use the transcribe tool with the provided audio file path.

When you receive a request to transcribe audio:
1. Use the transcribe tool with the audio file path provided
2. The transcribe tool will handle the actual transcription
3. Do not attempt to transcribe manually

Use the transcribe tool now.""")
                
                messages_with_system = [system_prompt] + input_messages
                response = self.transcriptor.invoke(messages_with_system)
                return {"messages": [response]}
            else:
                # If coming from evaluator (retry), perform context-aware correction
                correction_prompt = SystemMessage(content="""You are an intelligent transcription corrector. Your role is to analyze and improve transcriptions using context.

When you receive a transcription that needs correction, you must:
1. Analyze the context and meaning of the text
2. Identify words that might be incorrectly transcribed based on context
3. Correct technical terms, acronyms, and specialized vocabulary
4. Fix obvious phonetic errors (e.g., "rac" should be "RAG" in technical contexts)
5. Maintain the original meaning while improving accuracy
6. Consider domain-specific terminology (AI, programming, etc.)

Examples of corrections:
- "rac" → "RAG" (in AI/ML context)
- "sequel" → "SQL" (in database context)
- "pie-torch" → "PyTorch" (in ML context)

Return the corrected transcription text.""")
                
                messages_with_system = [correction_prompt] + input_messages
                response = self.llm.invoke(messages_with_system)
                return {"messages": [response]}
    
    @traceable(run_type="llm")
    def evaluator_function(self, state: State) -> State:
        """
        Evaluates transcription quality using contextual analysis and domain knowledge.
        
        This function acts as an intelligent transcription evaluator that:
        1. Analyzes the context and semantic coherence of the transcription
        2. Identifies potential transcription errors based on domain knowledge
        3. Detects technical terms, acronyms, and specialized vocabulary that might be incorrectly transcribed
        4. Evaluates grammar, flow, and logical consistency
        5. Returns a binary decision for transcription quality
        
        The evaluator uses contextual understanding to identify issues like:
        - Technical terms that sound similar but have different meanings
        - Acronyms that might be transcribed as regular words
        - Domain-specific vocabulary in AI, programming, etc.
        - Grammatical inconsistencies that suggest transcription errors

        Args:
            state (State): The current state containing the transcription to evaluate.

        Returns:
            State: Updated state with evaluation result ("GOOD" or "RETRY").
        """
        # Define system prompt for intelligent transcription evaluation
        evaluator_system_prompt = SystemMessage(content="""You are an intelligent transcription evaluator with expertise in technical and specialized content. Your role is to evaluate transcription quality using contextual analysis and domain knowledge.

When evaluating a transcription, analyze:

**Context and Coherence:**
- Does the text make logical sense in its context?
- Are there words that seem out of place or contextually incorrect?
- Is the overall flow and meaning coherent?

**Technical and Domain-Specific Terms:**
- Are technical terms, acronyms, and specialized vocabulary correctly transcribed?
- Look for common transcription errors in technical domains:
  - "rac" instead of "RAG" (Retrieval Augmented Generation)
  - "sequel" instead of "SQL" (database query language)
  - "pie-torch" instead of "PyTorch" (ML framework)
  - "java script" instead of "JavaScript"
  - "react" vs "React" (framework vs verb)
  - "node" vs "Node.js" in programming contexts

**Grammar and Structure:**
- Are there grammatical inconsistencies that suggest transcription errors?
- Does the sentence structure make sense?
- Are there obvious phonetic errors?

**Evaluation Criteria:**
- GOOD: Transcription is contextually accurate, technically correct, and coherent
- RETRY: Transcription contains likely errors, unclear technical terms, or contextual inconsistencies

**Response Format:**
Return ONLY one word:
- "GOOD" if the transcription is accurate and well-transcribed
- "RETRY" if the transcription likely contains errors or needs correction

Do not provide explanations or additional text, just return "GOOD" or "RETRY".""")
        
        input_messages = state["messages"]
        # Add system prompt to the beginning of messages
        messages_with_system = [evaluator_system_prompt] + input_messages
        response = self.llm.invoke(messages_with_system)
        return {"messages": [response]}
    
    @traceable(run_type="llm")
    def evaluator_decision(self, state: State) -> str:
        """
        Function to determine if the transcription is good or needs retry.
        
        Args:
            state (State): The state of the agent.
            
        Returns:
            str: "good" if transcription is acceptable, "retry" if it needs to be redone
        """
        last_message = state["messages"][-1]
        response_content = last_message.content.strip().upper()
        
        if "GOOD" in response_content:
            return "good"
        elif "RETRY" in response_content:
            return "retry"
        else:
            # Default to retry if response is unclear
            return "retry"
    
    @traceable(run_type="tool")
    def stt_function(self, state: State) -> State:
        """
        Function to handle the audio transcription logic.

        Args:
            state (State): The state of the agent.

        Returns:
            State: The state of the agent.
        """
        return {"messages": [self.stt.transcribe(state["messages"])]}
    
    # Invoke 
    def __call__(self, prompt: str) -> list[HumanMessage]:
        """
        Invoke the agent with a prompt.

        Args:
            prompt (str): The prompt to invoke the agent with.

        Returns:
            list[HumanMessage]: The messages from the agent.
        """
        messages = [HumanMessage(content=prompt)]

        result = self.graph.invoke({"messages": messages}, {"number": None})
        
        return result["messages"]


