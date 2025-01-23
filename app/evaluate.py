import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


class LangQAEval():
    def __init__(self):
        pass


    def load_model(self):
        self.model = AutoPeftModelForCausalLM.from_pretrained('models/final_model')
        # Merge
        self.model = self.model.merge_and_unload()
        self.tokenizer = AutoTokenizer.from_pretrained('models/final_model')


    def create_prompt_template(self):
        # Create pre-prompt with the instruction
        pre_prompt = """[INST] <<SYS>>\nAnalyze the question and answer with the best option.\n"""

        # Create the prompt adding the input
        prompt_template = pre_prompt + "Here is my question {context}" + "[\INST]"

        # Create the prompt template with LangChain
        self.prompt_template = PromptTemplate(template = prompt_template, input_variables=["context"])

        # Create the pipeline object
        pipe = pipeline("text-generation",
                        model = self.model,
                        tokenizer = self.tokenizer,
                        max_new_tokens = 512,
                        use_cache = False,
                        do_sample = True,
                        pad_token_id = self.tokenizer.eos_token_id,
                        top_p = 0.7,
                        temperature = 0.5)
        
        # Create the Hugging Face Pipeline
        self.llm_pipeline = HuggingFacePipeline(pipeline = pipe)

        self.memory = ConversationBufferMemory()


    def predict(self, prompt):
        # Create the LLM Chain
        chat_llm_chain = LLMChain(llm = self.llm_pipeline,
                                  prompt = self.prompt_template,
                                  verbose = False,
                                  memory = self.memory)
        
        return chat_llm_chain.predict(prompt)





if __name__ == "__main__":
    # Argument parser for dynamic input
    parser = argparse.ArgumentParser(description="Generate text using an LLM")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The initial text prompt for the Lang QA model."
    )

    args = parser.parse_args()
    
    print(args.prompt)

    langqaeval = LangQAEval()
    langqaeval.load_model()
    langqaeval.create_prompt_template()
    output_text = langqaeval.predict(args.prompt)

    print(output_text)