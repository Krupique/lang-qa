from langchain.prompts import PromptTemplate

# Defines a function that takes a dictionary named sample
def create_prompt(sample):

    # Defines a pre_prompt string that serves as a template for the first part of the prompt
    pre_prompt = """[INST]<<SYS>> {instruction}\n"""

    # Concatenates pre_prompt with additional strings to form the complete prompt
    prompt = pre_prompt + "{input}" +"[/INST]"+"\n{output}"

    # Assigns the value of the 'instruction' key of the dictionary sample to the variable example_instruction
    example_instruction = sample['instruction']

    # Assigns the value of the 'input' key of the dictionary sample to the variable example_input
    example_input = sample['input']

    # Assigns the value of the 'output' key of the dictionary sample to the variable example_output
    example_output = sample['output']

    # Creates an instance of PromptTemplate with the previously defined prompt and input variables
    prompt_template = PromptTemplate(template = prompt,
    input_variables = ["instruction", "input", "output"])

    # Uses the format method of the prompt_template instance to replace the variables
    # in the template with the specified values
    prompt_unico = prompt_template.format(instruction = example_instruction,
                                          input = example_input,
                                          output = example_output)

    # Returns the formatted prompt
    return [prompt_unico]