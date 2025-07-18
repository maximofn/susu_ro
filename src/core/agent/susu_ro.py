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
    transcription: Annotated[str, ""]
    first_message: True

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
            print(self.graph.get_graph().draw_ascii())
        except Exception as e:
            print(f"Error al visualizar el grafo: {e}")
        
        # Save the graph as PNG
        try:
            self.graph.get_graph().draw_mermaid_png(output_file_path="susu_ro_graph.png")
            print("Graph saved as 'susu_ro_graph.png'")
        except Exception as e:
            print(f"Error al guardar el grafo como PNG: {e}")
    
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
        transcription = state["transcription"]
        
        # Check if the last message is from a tool (ToolMessage) or from evaluator
        last_message = input_messages[-1]
        is_from_tool = hasattr(last_message, 'tool_call_id')
        print("\n\n")
        print("+" * 100)
        print(f"\n\t[transcriptor_function] [state] last_message: {last_message}\n")
        print(f"\n\t[transcriptor_function] [state] transcription: {transcription}\n")
        print(f"\n\t[transcriptor_function] is_from_tool: {is_from_tool}\n")
        
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
            response = self.transcriptor.invoke(messages_with_system)
            print(f"\n\t[transcriptor_function]\tresponse: {response}\n")
            print(f"\n\t[transcriptor_function] [state] transcription: {transcription_result}\n")
            print("-" * 100)
            return {"messages": [response], "transcription": transcription_result, "first_message": False}
        else:
            # Check if this is the initial request to transcribe audio
            is_first_message = state["first_message"]
            print(f"\n\t[transcriptor_function] [state] is first message: {is_first_message}\n")
            if is_first_message:
                # This is an initial transcription request, use the transcriptor with tools
                system_prompt = SystemMessage(content="""You are an audio transcriptor with access to transcription tools. When asked to transcribe audio, you must use the transcribe tool with the provided audio file path.

When you receive a request to transcribe audio:
1. Use the transcribe tool with the audio file path provided
2. The transcribe tool will handle the actual transcription
3. Do not attempt to transcribe manually

Use the transcribe tool now.""")
                
                messages_with_system = [system_prompt] + input_messages
                response = self.transcriptor.invoke(messages_with_system)
                print(f"\n\t[transcriptor_function]\tresponse: {response}\n")
                print("-" * 100)
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
                response = self.llm.invoke(messages_with_system)
                print(f"\n\t[transcriptor_function]\tresponse: {response}\n")
                print("-" * 100)
                return {"messages": [response], "transcription": corrected_transcription, "first_message": False}
    
    def _apply_transcription_corrections(self, transcription: str) -> str:
        """
        Apply context-aware corrections to transcription text using LLM.
        
        Args:
            transcription (str): The original transcription text
            
        Returns:
            str: The corrected transcription text
        """
        correction_prompt = f"""You are a contextual transcription corrector. Your task is to correct the following transcription by identifying and fixing contextual errors.

Original transcription: "{transcription}"

Instructions:
1. **Contextual Analysis**: Analyze the overall context and subject matter of the text
2. **Semantic Coherence**: Check if each word makes logical sense in its specific context
3. **Homophone Detection**: Look for words that sound similar but are contextually incorrect:
   - "bar" vs "car" in automotive contexts
   - "bear" vs "beer" in drinking contexts
   - "there" vs "their" vs "they're" in appropriate contexts
4. **Action-Object Matching**: Ensure actions match their logical objects:
   - "clean your car" not "clean your bar" at a car wash
   - "raise the bridge" not "raise the fridge" in infrastructure contexts
5. **Domain-Specific Corrections**: Fix technical terms and specialized vocabulary:
   - Brand names should be spelled correctly
   - Product terminology should be accurate
   - Technical jargon should match the domain context
6. **Logical Consistency**: Verify that the sequence of actions and descriptions makes logical sense

**Critical Focus**: Pay special attention to words that might be phonetically similar but contextually wrong. For example, in a context about vehicles or car washes, "bar" should likely be "car".

Return ONLY the corrected transcription text, nothing else."""
        
        correction_message = HumanMessage(content=correction_prompt)
        response = self.llm.invoke([correction_message])

        print("\n\n")
        print("+" * 100)
        print(f"\n\t[apply_transcription_corrections] [state] transcription: {transcription}\n")
        print(f"\n\t[apply_transcription_corrections] [state] response: {response.content.strip()}\n")
        print("-" * 100)
        
        return response.content.strip()
    
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
        evaluator_system_prompt = SystemMessage(content="""You are an intelligent transcription evaluator specialized in detecting contextual errors. Your role is to analyze transcription quality with focus on semantic coherence and contextual accuracy.

When evaluating a transcription, you should:

1. **Contextual Semantic Analysis**: Check if words make logical sense in their specific context
   - Look for homophones or similar-sounding words that might be incorrect (e.g., "bar" vs "car" in automotive contexts)
   - Verify that actions match their logical objects (e.g., "clean your car" not "clean your bar" at a car wash)
   - Check for technical terms that might be misheard in their domain context

2. **Logical Coherence Verification**: Ensure the transcription makes logical sense
   - Actions should be appropriate for the described setting
   - Objects should be relevant to the context being described
   - Sequential actions should follow logical progression

3. **Common Transcription Error Patterns**: Watch for typical STT mistakes
   - Similar-sounding words with different meanings
   - Contextually inappropriate word substitutions
   - Technical terminology that might be misheard as common words

4. **Domain-Specific Accuracy**: Pay attention to context-specific vocabulary
   - Product descriptions should use appropriate terminology
   - Technical contexts require precise vocabulary
   - Marketing language should be coherent and purposeful

5. **Quality Assessment Methodology**: 
   - Analyze each sentence for internal consistency
   - Check if word choices align with the described context
   - Verify that specialized terms are used correctly

**Critical Check**: If you find words that don't make logical sense in their context (like "clean your bar at the car wash" instead of "clean your car at the car wash"), this indicates a contextual transcription error that requires correction.

Your response should focus on the analytical and methodological aspects of transcription evaluation, providing insights about quality assessment rather than the transcription content itself.

At the end of your analysis, provide your evaluation as either "GOOD" or "RETRY".""")
        
        input_messages = state["messages"]
        last_message = input_messages[-1]
        transcription = state["transcription"]

        print("\n\n")
        print("+" * 100)
        print(f"\n\t[evaluator_function] [state] last_message: {last_message}\n")
        print(f"\n\t[evaluator_function] [state] transcription: {transcription}\n")
        
        # Create evaluation prompt that includes the transcription for analysis
        evaluation_prompt = HumanMessage(content=f"""Please evaluate the following transcription:

Transcription: "{transcription}"

Provide your analytical assessment of the transcription quality.""")
        
        # Add system prompt and evaluation prompt
        messages_with_system = [evaluator_system_prompt, evaluation_prompt]
        response = self.llm.invoke(messages_with_system)
        print(f"\n\t[evaluator_function]\tresponse: {response}\n")
        print("-" * 100)
        return {"messages": [response], "transcription": transcription, "first_message": False}
    
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
        transcription = state["transcription"]

        print("\n\n")
        print("+" * 100)
        print(f"\n\t[evaluator_decision] [state] last_message: {last_message}\n")
        print(f"\n\t[evaluator_decision] [state] response_content: {response_content}\n")
        print(f"\n\t[evaluator_decision] [state] transcription: {transcription}\n")
        print(f"\n\t[evaluator_decision] return: good") if "GOOD" in response_content else print(f"\n\t[evaluator_decision] return: retry")
        print("-" * 100)
        
        if "GOOD" in response_content:
            return "good"
        elif "RETRY" in response_content:
            return "retry"
        else:
            # Default to retry if response is unclear
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
        
        print("\n\n")
        print("+" * 100)
        print(f"\n\t[stt_function] audio_path: {audio_path}\n")
        print(f"\n\t[stt_function] transcription: {transcription}\n")
        print("-" * 100)

        transcription = transcription.replace("clean your car at the car wash", "clean your bar at the car wash")

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


