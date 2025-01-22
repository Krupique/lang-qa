import json
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline, TrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

from utils.utils_functions import create_prompt


import warnings


class LangQA():

    def __init__(self, config):
        self.config = config
        

    def set_bitsandbytes(self, config):
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        
        bnb_config = BitsAndBytesConfig(load_in_4bit = config["use_4bit"],
                                        bnb_4bit_quant_type = config["bnb_4bit_quant_type"],
                                        bnb_4bit_compute_dtype = compute_dtype,
                                        bnb_4bit_use_double_quant = config["use_nested_quant"])
        
        # Verifying if the GPU supports bfloat16
        if compute_dtype == torch.float16 and config["use_4bit"]:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("The GPU suporrts bfloat16. You can accelerate the train using bf16=True")
                print("=" * 80)

        return bnb_config
    

    def set_lora(self, config):
        # LoRa Parameters
        peft_config = LoraConfig(r = config["r"],
                                lora_alpha = config["lora_alpha"],
                                lora_dropout = config["lora_dropout"],
                                bias = config["bias"],
                                task_type = config["task_type"])
        
        return peft_config


    def create_model(self):
        # Defining the BitsAndBytes config
        bnb_config = self.set_bitsandbytes(self.config["BitsAndBytesConfig"])

        # LLM
        llm_name = config["llm"]
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # Use the EOS token from the tokenizer to pad at the end of each sequence
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Enable padding at the end of each sentence
        self.tokenizer.padding_side = "right"

        # Load the base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(llm_name,
                                                    quantization_config = bnb_config,
                                                    device_map = "auto",
                                                    use_cache = False)

        self.peft_config = self.set_lora(self.config["LoraConfig"])
        # Prepare the model to  train
        self.model = prepare_model_for_kbit_training(self.model)
        # Merge the quantized model with the LoRa adapters
        self.model = get_peft_model(self.model, peft_config=self.peft_config)

        print(self.model)


    def set_training_arguments(self, config):
        output_model = 'models/adjusted_model'
        # Train arguments
        training_arguments = TrainingArguments(output_dir = output_model,
                                            per_device_train_batch_size = config["per_device_train_batch_size"],
                                            gradient_accumulation_steps = config["gradient_accumulation_steps"],
                                            optim = config["optim"],
                                            learning_rate = config["learning_rate"],
                                            lr_scheduler_type = config["lr_scheduler_type"],
                                            save_strategy = config["save_strategy"],
                                            logging_steps = config["logging_steps"],
                                            num_train_epochs = config["num_train_epochs"],
                                            max_steps = config["max_steps"],
                                            fp16 = config["fp16"])
        return training_arguments


    def load_dataset(self):
        self.dataset = load_dataset('nlpie/Llama2-MedTuned-Instructions')
        self.train_data = self.dataset['train'].select(indices=range(1000))
        # Selecting the lines to test the model
        self.test_data = self.dataset['train'].select(indices=range(1000, 1200))


    def create_trainer(self):
        training_arguments = self.set_training_arguments(config["TrainingArguments"])

        # Creates the Trainer
        # Optimized for fine-tuning pre-trained models with smaller datasets on supervised learning tasks.
        self.trainer = SFTTrainer(model = self.model,
                            peft_config = self.peft_config,
                            #  max_seq_length = 512,
                            tokenizer = self.tokenizer,
                            #  packing = True,
                            formatting_func = create_prompt,
                            args = training_arguments,
                            train_dataset = self.train_data,
                            eval_dataset = self.test_data)


    def train(self):
        self.trainer.train()

        # Model save
        self.trainer.save_model('models/final_model')


if __name__ == "__main__":
    with open("config/config.json", "r") as f:
        config = json.load(f)

    print(config)


    langqa = LangQA(config)
    langqa.create_model()
    langqa.load_dataset()
    langqa.create_trainer()

    print('Chegou até aqui')