# local_llm_project
# Local LLM with Llama 2 and LangChain

This repository demonstrates how to run Llama 2 language models locally on your PC without relying on external services like Ollama. It uses llama-cpp-python for direct model interaction and LangChain for structured prompting and chain management.

## Features

- Run Llama 2 models locally without API calls or external services
- Direct interaction with the model using llama-cpp-python
- Integration with LangChain for advanced prompting and chains
- Interactive mode for real-time conversations
- Works with quantized GGUF models for efficiency on consumer hardware

## Requirements

- Python 3.8+
- llama-cpp-python
- langchain and langchain-community
- A compatible Llama 2 GGUF model file

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Model

Download the Llama 2 7B Chat GGUF model from Hugging Face:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main

For this project, we use the `llama-2-7b-chat.Q2_K.gguf` quantized model, which offers a good balance between performance and resource usage.

### 3. Update Model Path

In `main.py`, update the model path in the `get_model_path()` function to point to your downloaded model file:

```python
def get_model_path() -> str:
    return r'path/to/your/llama-2-7b-chat.Q2_K.gguf'
```

## Usage

Run the script:

```bash
python main.py
```

The program will present you with three options:
1. Run a direct example using llama-cpp-python (planet names)
2. Run an example using LangChain (solar system explanation)
3. Enter interactive mode to ask your own questions

## Example Prompts

The repository includes examples of properly formatted prompts for Llama 2 chat models:

```python
prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""
```

## Project Structure

- `main.py` - Main script with examples and interactive mode
- `requirements.txt` - Required Python packages
- `README.md` - This documentation file

## Notes on Performance

- The model's performance depends on your hardware. A modern CPU will work, but a GPU provides much better performance.
- Quantized models (like Q2_K) run faster and use less RAM but may have slightly lower quality than higher precision models.
- Adjust the `max_tokens` parameter for longer or shorter responses.

## Resources

- [TheBloke's Llama-2-7B-Chat-GGUF models](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python)
- [LangChain documentation](https://python.langchain.com/docs/get_started/introduction)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

