# LangQA Documentation

## Project Overview
**LangQA** is a state-of-the-art chatbot system designed to answer medical questions with precision and reliability. Built using LangChain and a fine-tuned open-source Large Language Model (LLM), LangQA leverages custom data integration and advanced fine-tuning techniques to deliver context-aware, domain-specific responses. The system is tailored for healthcare professionals, patients, and researchers who require accurate and trustworthy information retrieval in the medical domain.

---

## Purpose and Business Problem
The primary goal of LangQA is to address the challenge of providing accurate answers to complex medical questions in real-time. Medical professionals and patients often face difficulties in retrieving reliable and contextually relevant information due to the vast and complex nature of medical knowledge. LangQA aims to solve this problem by:

1. **Delivering Reliable Information**: Leveraging fine-tuned medical datasets to ensure responses are accurate and aligned with medical standards.
2. **Enhancing Accessibility**: Providing instant answers through an intuitive chatbot interface.
3. **Improving Decision-Making**: Assisting healthcare professionals with evidence-based responses to support clinical decisions.

---

## Key Features
- **Domain-Specific Knowledge**: Fine-tuned on medical instruction datasets to ensure relevance and reliability.
- **Custom Prompt Engineering**: Tailored prompts for medical inquiries to optimize context comprehension.
- **Efficient Deployment**: Lightweight model with 4-bit quantization for faster inference and reduced resource consumption.
- **LangChain Integration**: Utilizes advanced chaining techniques for handling complex multi-turn conversations.

---

## How LangQA Works
1. **Fine-Tuning with Medical Data**: LangQA uses the `nlpie/Llama2-MedTuned-Instructions` dataset to train the model on medical-specific tasks.
2. **Prompt Templates**: Custom templates are designed to guide the model in understanding and responding to medical queries effectively.
3. **Inference Pipeline**: The chatbot processes user inputs through a LangChain-powered pipeline, combining contextual memory and precise generation.

---

## Installation
To deploy LangQA, follow these steps:

### Prerequisites
- Python 3.12.3
- Poetry package manager
- CUDA-enabled GPU (for training and inference)

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/Krupique/lang-qa
   cd lang-qa
   ```
2. Install Poetry:
   ```bash
   pip install poetry
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Activate the Poetry environment:
   ```bash
   poetry shell
   ```

---

## Configuration
LangQA's behavior is controlled via the `config/config.json` file. This configuration includes model settings, training parameters, and quantization options. Below is an example:

```json
{
    "llm": "NousResearch/Llama-2-7b-chat-hf",

    "BitsAndBytesConfig": {
        "use_4bit": true,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "use_nested_quant": false
    },

    "LoraConfig": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },

    "TrainingArguments": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "optim": "paged_adamw_32bit",
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "logging_steps": 10,
        "num_train_epochs": 3,
        "max_steps": 150,
        "fp16": true
    }
}
```

---

## Training
LangQA employs LoRa (Low-Rank Adaptation) and 4-bit quantization techniques to fine-tune a pre-trained LLM efficiently.

### Steps to Train
1. Ensure the configuration file is properly set.
2. Execute the training script:
   ```bash
   poetry run python train.py
   ```
3. Monitor the training process through the logs.
4. The fine-tuned model will be saved in `models/final_model`.

---

## Evaluation
LangQA evaluates its performance by responding to medical questions and measuring the quality of its outputs.

### Steps to Evaluate
1. Use the evaluation script to test the model:
   ```bash
   poetry run python evaluate.py --prompt "What are the symptoms of diabetes?"
   ```
2. Example output:
   ```
   > Prompt: What are the symptoms of diabetes?
   > Answer: Common symptoms of diabetes include increased thirst, frequent urination, and unexplained weight loss.
   ```
3. The evaluation script ensures the model's ability to understand medical questions and provide accurate responses.

---

## Project Structure
```
LangQA/
├── app/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── utils/
│       └── utils_functions.py      # Utility functions (e.g., prompt creation)
├── config/
│   └── config.json                 # Configuration file
├── models/
│   ├── adjusted_model/             # Directory for the adjusted model
│   └── final_model/                # Directory for the fine-tuned model
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for data exploration
├── .gitignore                      # Ignored files
├── README.md                       # Project README
├── pyproject.toml                  # Poetry configuration
└── pyproject.lock                  # Poetry lock
```

---

## Business Impact
LangQA aims to transform how medical professionals and patients access information by:

1. **Reducing Search Time**: Instantly answering questions that otherwise require extensive searches in medical literature.
2. **Improving Patient Care**: Assisting healthcare providers with evidence-based answers to enhance decision-making.
3. **Enhancing Education**: Supporting medical students and researchers with a reliable tool for learning and exploration.

---

## Future Enhancements
To further improve LangQA, future developments may include:
- **Multi-Language Support**: Expanding the model to answer medical questions in multiple languages.
- **Voice Integration**: Enabling voice-based input and responses for accessibility.
- **Dynamic Updates**: Regularly updating the model with the latest medical knowledge and guidelines.

---

## Conclusion
LangQA provides an innovative solution to the problem of medical information retrieval by combining cutting-edge LLM fine-tuning techniques with a focus on domain-specific expertise. By leveraging the power of LangChain and a fine-tuned open-source LLM, LangQA ensures accurate, context-aware, and reliable answers to complex medical questions, making it an invaluable tool for the healthcare industry.

