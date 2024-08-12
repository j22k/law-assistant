
# Law Assistant Chatbot

This project involves the development of a law assistant chatbot, fine-tuned using Meta's LLaMA-2 model on legal data sourced from the internet. The chatbot is designed to assist with answering legal queries, providing information on various laws, and interpreting legal concepts. The model was trained using a custom dataset and has been optimized for efficient performance on resource-constrained environments.

## Project Overview

The chatbot leverages a fine-tuned version of the `NousResearch/Llama-2-7b-chat-hf` model. The fine-tuning process involved adapting the model to handle specific legal queries, focusing on cyber law and other legal domains. The project is implemented using Hugging Face's Transformers library, with training conducted on Google Colab.

### Key Features

- **Model Architecture**: LLaMA-2 with LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- **Training Framework**: Hugging Face Transformers and Accelerate, with bitsandbytes for quantization.
- **Data**: Fine-tuned on custom legal datasets, including cyber law.
- **Precision**: Uses 4-bit quantization to optimize memory usage and improve inference speed.
- **Deployment**: The model is ready for deployment and can be integrated into legal software or used as a standalone chatbot.

## Installation

To install and run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/law-assistant-chatbot.git
    cd law-assistant-chatbot
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment and download the model:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Junaidjk/Model"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ```

## Usage

To interact with the chatbot, you can use the following script:

```python
from transformers import pipeline

def get_model_res(prompt, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=300)
    result = pipe(f"[INST] {prompt} [/INST]")
    return result[0]['generated_text']

y_or_n = True
while y_or_n:
    prompt = input("Ask your question: ")
    res = get_model_res(prompt, model, tokenizer)
    print(res)
    user_input = input("Do you want to continue? (y/n): ")
    if user_input.lower() != 'y':
        y_or_n = False
```

### Example Prompts

- "What is the legal weight of an electronic signature in India?"
- "Can the Information Technology Act, 2000 be applied to offences committed outside India?"
- "What are the valid legal documents recognized under Indian law?"

## Model Training Details

The model was fine-tuned using the following configuration:

- **LoRA Attention Dimension**: 64
- **Alpha Parameter**: 16
- **Dropout**: 0.1
- **Training Epochs**: 1
- **Learning Rate**: 2e-4
- **Batch Size**: 4

For detailed code and training parameters, please refer to the Colab notebook available in the repository.

## Model Deployment

The model and tokenizer have been uploaded to the Hugging Face Hub:

- Model: [Law-assistant-final](https://huggingface.co/Junaidjk/Model)
- Tokenizer: [Law-assistant-final](https://huggingface.co/Junaidjk/Model)

You can also run the model on your local machine or deploy it on cloud platforms for real-time inference.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to Hugging Face for providing the Transformers library and Google Colab for the compute resources.
