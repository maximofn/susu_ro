from typing import Annotated
from typing_extensions import TypedDict
 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
 
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_ENDPOINT = "https://models.github.ai/inference"

PRINT_DEBUG = True

# State
class State(TypedDict):
    """
    State of the SusuRo agent.
    """
    messages: Annotated[list, add_messages]
    transcription: Annotated[str, ""]
    first_message: Annotated[bool, True]

class SusuRo():
    def __init__(
            self,
            chat_model: str = "qwen3:4b",
            chat_reasoning: bool = False,
            whisper_model_size: str = "large",
            whisper_device: str = "auto",
            enable_streaming: bool = True
        ) -> None:
        """
        Initialize the SusuRo agent.

        Args:
            chat_model (str, optional): The model to use for the chat. Defaults to "qwen3:4b".
            chat_reasoning (bool, optional): Whether to use reasoning for the chat. Defaults to False.
            whisper_model_size (str, optional): The size of the Whisper model. Defaults to "large".
            whisper_device (str, optional): The device to use for the Whisper model. Defaults to "auto".
            enable_streaming (bool, optional): Whether to enable streaming of tokens. Defaults to True.

        Returns:
            None
        """
        # Create llm model
        if "gpt" in chat_model or "o3" in chat_model:
            if not GITHUB_TOKEN:
                raise ValueError("GITHUB_TOKEN environment variable is required for GitHub Models")
            self.llm = ChatOpenAI(model=chat_model, base_url=GITHUB_ENDPOINT, api_key=GITHUB_TOKEN)
        else:
            self.llm = ChatOllama(model=chat_model, reasoning=chat_reasoning)
        
        # Enable streaming
        self.enable_streaming = enable_streaming

        # Create the stt model
        self.stt = Whisper(model_size=whisper_model_size, device=whisper_device)

        # Create tools list
        self.tools_list = [self.stt_function]

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
            if PRINT_DEBUG: print(self.graph.get_graph().draw_ascii())
        except Exception as e:
            if PRINT_DEBUG: print(f"Error al visualizar el grafo: {e}")
        
        # Save the graph as PNG
        try:
            self.graph.get_graph().draw_mermaid_png(output_file_path="susu_ro_graph.png")
            if PRINT_DEBUG: print("Graph saved as 'susu_ro_graph.png'")
        except Exception as e:
            if PRINT_DEBUG: print(f"Error al guardar el grafo como PNG: {e}")
    
    # Functions
    def print_initial_debug_block(self) -> None:
        """
        Print the state of the agent.
        """
        if PRINT_DEBUG: print("\n\n")
        if PRINT_DEBUG: print("+" * 100)
    
    def print_end_debug_block(self) -> None:
        """
        Print the end of the debug block.
        """
        if PRINT_DEBUG: print("-" * 100)
    
    @traceable(run_type="llm")
    def stream_llm_response(self, messages: list, use_tools: bool = False, function_name: str = None) -> str:
        """
        Stream LLM response with real-time token display, including reasoning tokens.
        
        Args:
            messages: List of messages to send to the LLM
            use_tools: Whether to use tools (enables streaming with tool response)
            function_name: Optional name to display in debug output
            
        Returns:
            str or AIMessage: Complete response content or full message object if tools used
        """
        # If llm can use tools, use the transcriptor
        llm = self.transcriptor if use_tools else self.llm

        # If streaming is disabled, use the normal invocation
        if not self.enable_streaming:
            # Fallback to normal invocation when streaming disabled
            response = llm.invoke(messages)
            if PRINT_DEBUG and hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
                print(f"\t[{function_name}] reasoning: {response.additional_kwargs['reasoning_content']}")
            if PRINT_DEBUG and hasattr(response, 'content') and response.content:
                print(f"\t[{function_name}] response: {response.content}")
            return response if use_tools else response.content
        
        # Streaming enabled - works for both with and without tools        
        if function_name:
            if PRINT_DEBUG: print(f"\t[{function_name}]\tresponse: ", end="", flush=True)
        else:
            if PRINT_DEBUG: print("\tresponse: ", end="", flush=True)
        
        full_response = ""
        reasoning_content = ""
        in_reasoning_phase = False
        tool_calls = []
        response_id = None
        
        try:
            for chunk in llm.stream(messages):
                
                # Capture response metadata
                if hasattr(chunk, 'id') and chunk.id:
                    response_id = chunk.id
                
                # Handle tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
                
                # Print reasoning content
                if hasattr(chunk, 'additional_kwargs'):
                    additional_kwargs = chunk.additional_kwargs
                    if 'reasoning_content' in additional_kwargs:
                        reasoning_text = additional_kwargs['reasoning_content']
                        if reasoning_text and PRINT_DEBUG:
                            if not in_reasoning_phase:
                                print(f"\n🤔 [REASONING] ", end="", flush=True)
                                in_reasoning_phase = True
                            print(reasoning_text, end="", flush=True)
                            reasoning_content += reasoning_text
                
                # Handle regular content tokens
                content = chunk.content
                if content:
                    # If we were in reasoning phase, switch to response phase
                    if in_reasoning_phase and PRINT_DEBUG:
                        print(f"\n💭 [RESPONSE] ", end="", flush=True)
                        in_reasoning_phase = False
                    
                    if PRINT_DEBUG: 
                        print(content, end="", flush=True)
                    full_response += content
                
        except Exception as e:
            if PRINT_DEBUG: print(f"\n❌ Streaming error: {e}")
            # Fallback to normal invocation
            llm = self.transcriptor if use_tools else self.llm
            response = llm.invoke(messages)
            return response if use_tools else response.content
        
        if PRINT_DEBUG: print()  # New line after streaming
        
        # If tools were used, construct and return a complete AIMessage-like response
        if use_tools:
            # Create a complete response object similar to what invoke() would return
            response = AIMessage(
                content=full_response,
                tool_calls=tool_calls,
                id=response_id,
                additional_kwargs={'reasoning_content': reasoning_content} if reasoning_content else {}
            )
            
            return response
        
        return full_response
    
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
        transcription = state["transcription"]
        
        # Check if the last message is from a tool (ToolMessage) or from evaluator
        last_message = input_messages[-1]
        is_from_tool = hasattr(last_message, 'tool_call_id')
        self.print_initial_debug_block()
        if PRINT_DEBUG: print(f"\t[transcriptor_function] is_from_tool: {is_from_tool}")
        
        if is_from_tool:
            # If coming from tool, store transcription separately from logic
            transcription_result = last_message.content
            
            literal_prompt = SystemMessage(content="""You are an audio transcriptor. Your role is to provide analysis and context about the transcription process.

When you receive a transcription result from the transcribe tool, you must:
1. Analyze the transcription quality and provide context
2. Discuss the transcription process and methodology
3. Provide insights about the audio content structure
4. Do not include the actual transcribed text in your response

Focus on the analytical and contextual aspects of the transcription work.""")
            
            messages_with_system = [literal_prompt] + input_messages
            response = self.stream_llm_response(messages_with_system, use_tools=False, function_name="transcriptor_function")
            self.print_end_debug_block()
            return {"messages": [response], "transcription": transcription_result, "first_message": False}
        else:
            # Check if this is the initial request to transcribe audio
            is_first_message = state["first_message"]
            if PRINT_DEBUG: print(f"\t[transcriptor_function] [state] is first message: {is_first_message}")
            if is_first_message:
                # This is an initial transcription request, use the transcriptor with tools
                system_prompt = SystemMessage(content="""You are an audio transcriptor with access to transcription tools. When asked to transcribe audio, you must use the transcribe tool with the provided audio file path.

When you receive a request to transcribe audio:
1. Use the transcribe tool with the audio file path provided
2. The transcribe tool will handle the actual transcription
3. Do not attempt to transcribe manually

Use the transcribe tool now.""")
                
                messages_with_system = [system_prompt] + input_messages
                response = self.stream_llm_response(messages_with_system, use_tools=True, function_name="transcriptor_function")
                self.print_end_debug_block()
                return {"messages": [response], "transcription": transcription, "first_message": False}
            else:
                # If coming from evaluator (retry), perform context-aware correction
                correction_prompt = SystemMessage(content="""You are an intelligent transcription analyst. Your role is to analyze and discuss transcription improvements.

When you receive feedback that a transcription needs correction, you must:
1. Analyze the context and reasoning behind the correction need
2. Discuss potential issues with technical terms, acronyms, and specialized vocabulary
3. Provide analytical insights about transcription challenges
4. Explain the methodology for improving transcription accuracy
5. Do not include the actual corrected transcription text in your response

Focus on the analytical and methodological aspects of transcription improvement.""")
                
                # Apply corrections to the transcription field using LLM
                corrected_transcription = self._apply_transcription_corrections(transcription)
                
                messages_with_system = [correction_prompt] + input_messages
                response_content = self.stream_llm_response(messages_with_system, use_tools=False, function_name="transcriptor_function")
                response = type(input_messages[-1])(content=response_content)
                self.print_end_debug_block()
                return {"messages": [response], "transcription": corrected_transcription, "first_message": False}
    
    @traceable(run_type="llm")
    def _apply_transcription_corrections(self, transcription: str) -> str:
        """
        Apply context-aware corrections to transcription text using LLM.
        
        Args:
            transcription (str): The original transcription text
            
        Returns:
            str: The corrected transcription text
        """
        correction_prompt = f"""Fix only clear transcription errors in this text. Keep the original wording and meaning intact.

Original text: "{transcription}"

Fix only these types of errors:
1. **Garbled words**: Fix nonsensical text that should be real words
2. **Wrong homophones**: Fix words that sound similar but are contextually wrong
3. **Negatives**: Fix "can't" to "can" when context is clearly positive
4. **Broken sentences**: Fix sentence structure that doesn't make grammatical sense
5. **Missing punctuation**: Add quotes for dialogue and basic punctuation

Preserve everything else:
- Keep all correctly spelled words even if you might prefer synonyms
- Keep the original vocabulary choices
- Keep the same sentence structure when it's already correct
- Don't change writing style or word preferences

Return only the corrected text:"""
        
        correction_message = HumanMessage(content=correction_prompt)
        response_content = self.stream_llm_response([correction_message], use_tools=False, function_name="apply_transcription_corrections")

        return response_content.strip()
    
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
        evaluator_system_prompt = SystemMessage(content="""Evaluate this transcription for clear transcription errors only.

Consider these as errors requiring RETRY:
1. **Garbled text**: Nonsensical words or letter combinations that aren't real words
2. **Context homophones**: Words that sound similar but are clearly wrong in context
3. **Wrong negatives**: "can't" when context clearly means "can" (positive meaning)
4. **Broken grammar**: Sentence fragments or structure that doesn't make sense
5. **Missing dialogue punctuation**: Speech without proper quotes when clearly dialogue

DO NOT consider as errors:
- Word choices that are correct but could be different
- Complete sentences with proper grammar
- Repetitive but meaningful language
- Style preferences

Standards:
- GOOD: Transcription conveys clear meaning with proper grammar
- RETRY: Contains obvious transcription failures that impair understanding

Provide brief analysis then end with: GOOD or RETRY""")
        
        transcription = state["transcription"]

        self.print_initial_debug_block()
        
        # Create evaluation prompt that includes the transcription for analysis
        evaluation_prompt = HumanMessage(content=f"""Please evaluate the following transcription:

Transcription: "{transcription}"

Provide your analytical assessment of the transcription quality.""")
        
        # Add system prompt and evaluation prompt
        messages_with_system = [evaluator_system_prompt, evaluation_prompt]
        response_content = self.stream_llm_response(messages_with_system, use_tools=False, function_name="evaluator_function")
        response = type(evaluation_prompt)(content=response_content)
        self.print_end_debug_block()
        return {"messages": [response], "transcription": transcription, "first_message": False}
    
    @traceable(run_type="llm")
    def evaluator_decision(self, state: State) -> str:
        """
        Function to determine if the transcription is good or needs retry.
        
        Analyzes the evaluator's response to determine the next action.
        Returns "good" if the response contains "GOOD", "retry" if it contains "RETRY",
        or defaults to "retry" if the response is unclear.
        
        Args:
            state (State): The state of the agent.
            
        Returns:
            str: "good" if transcription is acceptable, "retry" if it needs to be redone
        """
        last_message = state["messages"][-1]
        response_content = last_message.content.strip().upper()

        self.print_initial_debug_block()
        
        # More robust checking for GOOD or RETRY
        if "RETRY" in response_content:
            if PRINT_DEBUG: print(f"\t[evaluator_decision] return: retry")
            if PRINT_DEBUG: print(f"\t[evaluator_decision] found 'RETRY' in response")
            self.print_end_debug_block()
            return "retry"
        elif "GOOD" in response_content:
            if PRINT_DEBUG: print(f"\t[evaluator_decision] return: good")
            if PRINT_DEBUG: print(f"\t[evaluator_decision] found 'GOOD' in response")
            if PRINT_DEBUG: print(f"\t[evaluator_decision] final transcription: {state['transcription']}")
            self.print_end_debug_block()
            return "good"
        else:
            # Default to retry if response is unclear
            if PRINT_DEBUG: print(f"\t[evaluator_decision] return: retry (default - no clear GOOD or RETRY found)")
            if PRINT_DEBUG: print(f"\t[evaluator_decision] response content: {response_content[:100]}...")
            self.print_end_debug_block()
            return "retry"
    
    @traceable(run_type="tool")
    def stt_function(self, audio_path: str, language: str = None) -> str:
        """
        Function to handle the audio transcription logic.

        Args:
            audio_path (str): Path to the audio file to transcribe.
            language (str, optional): Language code for transcription. Defaults to None.

        Returns:
            str: The transcribed text from the audio file.
        """
        transcription = self.stt.transcribe(audio_path, language)
        
        self.print_initial_debug_block()
        if PRINT_DEBUG: print(f"\t[stt_function] audio_path: {audio_path}")
        if PRINT_DEBUG: print(f"\t[stt_function] language: {language}")
        if PRINT_DEBUG: print(f"\t[stt_function] transcription: {transcription}")
        self.print_end_debug_block()

        return transcription
    
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
        state = {"messages": messages, "transcription": "", "first_message": True}

        result = self.graph.invoke(state, {"number": None})
        
        return result["messages"]


