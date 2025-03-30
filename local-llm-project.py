"""
Local LLM Implementation using LlamaCpp and LangChain
---------------------------------------------------------
This script demonstrates how to run Llama 2 models locally using llama-cpp-python
and integrate with LangChain for structured prompting.
"""

import os
from typing import Optional

# Import necessary libraries
from llama_cpp import Llama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


def get_model_path() -> str:
    """Returns the path to the GGUF model file."""
    # You can modify this to use environment variables or config files
    return r'E:\llama-2-7b-chat.Q2_K.gguf'


def direct_llama_example(model_path: str, query: Optional[str] = None) -> None:
    """
    Run a direct example using llama-cpp-python without LangChain.
    
    Args:
        model_path: Path to the GGUF model file
        query: Optional user query. If None, defaults to a planets question
    """
    # Initialize the model
    llm = Llama(model_path=model_path)
    
    # Define system message
    system_message = "You are a helpful assistant"
    
    # Use default query or get user input
    if query is None:
        user_message = "Q: Name the planets in the solar system? A: "
    else:
        user_message = f"Q: {query} A: "
    
    # Format the prompt according to Llama 2 chat format
    prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""
    
    # Generate response
    print("Generating response using direct Llama API...")
    output = llm(
        prompt,
        max_tokens=32,
        stop=["Q:", "\n"],
        echo=True
    )
    
    # Print the response
    print("\nFull output:")
    print(output)
    print("\nGenerated text:")
    print(output['choices'][0]['text'])


def langchain_llama_example(model_path: str, query: str = "Explain what is the solar system in 2-3 sentences") -> None:
    """
    Run an example using LangChain with LlamaCpp.
    
    Args:
        model_path: Path to the GGUF model file
        query: The query to send to the model
    """
    # Create a template for structured prompting
    template = """
<s>[INST] <<SYS>>
Act as an Astronomer engineer who is teaching high school students.
<</SYS>>

{text} [/INST]
"""
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    
    # Format the prompt with the user query
    formatted_prompt = prompt.format(text=query)
    print("Formatted prompt:")
    print(formatted_prompt)
    
    # Setup streaming callbacks
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initialize the model with LangChain
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )
    
    # Generate response
    print("\nGenerating response using LangChain...")
    output = llm.invoke(formatted_prompt)
    
    # Print final output
    print("\nFinal output:")
    print(output)


def interactive_mode(model_path: str) -> None:
    """
    Run an interactive session where the user can input queries.
    
    Args:
        model_path: Path to the GGUF model file
    """
    # Initialize the model
    llm = Llama(model_path=model_path)
    
    # Define system message
    system_message = "You are a helpful assistant"
    
    # Get user input
    user_input = input("Enter your question: ")
    user_message = f"Q: {user_input} A: "
    
    # Format the prompt according to Llama 2 chat format
    prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""
    
    # Generate response
    print("Generating response...")
    output = llm(
        prompt,
        max_tokens=500,
        stop=["Q:", "\n"],
        echo=True
    )
    
    # Print the response
    print("\nGenerated text:")
    print(output['choices'][0]['text'])


def main():
    """Main function to run examples based on user selection."""
    # Get model path
    model_path = get_model_path()
    
    print("Local LLM Demo using Llama-2-7B-Chat")
    print("====================================")
    print("1. Run direct Llama example (planets question)")
    print("2. Run LangChain example (solar system explanation)")
    print("3. Interactive mode (ask your own question)")
    
    choice = input("\nSelect an option (1-3): ")
    
    if choice == '1':
        direct_llama_example(model_path)
    elif choice == '2':
        langchain_llama_example(model_path)
    elif choice == '3':
        interactive_mode(model_path)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
