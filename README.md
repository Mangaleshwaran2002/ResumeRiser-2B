# ResumeRiser-2B

## Description
Welcome to **ResumeRiser-2B**, a fine-tuned version of Microsoft's Phi-2 model (2.7 billion parameters) designed to answer questions about my personal and professional journey. By leveraging **LoRA (Low-Rank Adaptation)** and **4-bit quantization**, this model was tailored using a custom dataset, optimized for efficiency and performance with tools like Hugging Face's Transformers, PEFT, and TRL libraries. This repository showcases the code, model details, and usage instructions for interacting with the fine-tuned model hosted on Hugging Face.

## Features
- **Custom Fine-Tuning**: Adapted Phi-2 to provide accurate responses about my academic and professional background.
- **Efficient Training**: Utilizes LoRA and 4-bit quantization for resource-efficient fine-tuning.
- **Easy Inference**: Includes sample code to run inference using Hugging Face's Transformers library.
- **High Performance**: Inherits Phi-2's near state-of-the-art capabilities in common sense, language understanding, and logical reasoning for models under 13 billion parameters.


## Screenshot
https://github.com/user-attachments/assets/1d8f4ec7-a60f-4cd3-85f1-d98e1eba8fbb

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face Transformers, PEFT, and TRL libraries
- CUDA-enabled GPU (recommended for inference)

### Installation
Install the required dependencies:
```bash
pip install transformers torch peft trl
```

### Usage
Below is a sample script to run inference with the fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model repository
repo_id = "Mangal-404/phi-2-ResumeRiser-2B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(repo_id)


https://github.com/user-attachments/assets/4935b6c3-d8ed-48c6-8cda-d1b5c470e998


# Prepare input
input_text = "Where did Mangaleshwaran pursue his Master's degree?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate response
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Project Details
This project involved fine-tuning the **Microsoft Phi-2** model using a custom dataset I curated to reflect my personal and professional experiences. The fine-tuning process employed **LoRA (Low-Rank Adaptation)** for efficient parameter updates and **4-bit quantization** to reduce memory usage, making the model suitable for deployment on resource-constrained environments. The Hugging Face ecosystem, including **Transformers**, **PEFT**, and **TRL** libraries, was instrumental in streamlining the training and inference pipeline.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the model, documentation, or code. Please follow the standard GitHub workflow for contributions.

## Contact
For questions or inquiries, reach out via [GitHub Issues](https://github.com/Mangaleshwaran2002/ResumeRiser-7B/issues) or connect with me on [LinkedIn](https://www.linkedin.com/in/mangaleshwaran-k/).

## Acknowledgments
- [Microsoft](https://www.microsoft.com) for developing the Phi-2 model.
- [Hugging Face](https://huggingface.co) for providing the Transformers, PEFT, and TRL libraries.
- The open-source community for continuous inspiration and support.
