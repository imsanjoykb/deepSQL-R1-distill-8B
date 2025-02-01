---
license: apache-2.0
language:
- en
base_model:
- deepseek-ai/DeepSeek-R1
new_version: imsanjoykb/deepSQL-R1-distill-8B
pipeline_tag: text-generation
library_name: adapter-transformers
library_name2: transformers
tags:
- unsloth,
- pytorch,
- deepsee-R1,
- inference-endpoint,
- sql-code-generation,
---
[🤗 HF Repo](https://huggingface.co/imsanjoykb/deepSQL-R1-distill-8B) | [♾️ Colab](https://drive.google.com/file/d/145PP-oW50OMS1bYJaYuUphfufpsuOGWl/view?usp=sharing) | [📶 Kaggle](https://www.kaggle.com/code/imsanjoykb/inference-deepsql-r1-distill-8b)

Introducing the latest fine-tuned version of Qwen2.5-Coder-14B-Instruct, specifically tailored for SQL code generation. Built on the robust 14-billion parameter Qwen2.5-Coder architecture, this model leverages advanced configurations like bfloat16 precision and a custom quantization setup, optimized for efficient 4-bit computation. With a maximum context window of 32K tokens, this model supports extensive SQL sequences and complex query generation without compromising accuracy or performance.

Our fine-tuning process has enriched this model with domain-specific SQL patterns and nuanced query constructions, making it exceptionally adept at handling real-world SQL requirements, from query creation to debugging and optimization. By combining Qwen2.5's foundational strengths with targeted training on custom SQL data, this model achieves a powerful balance of general-purpose code understanding and SQL-specific precision, making it an ideal tool for developers and data engineers seeking top-tier SQL generation capabilities.


## Inference

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
# Import necessary libraries
from unsloth import FastLanguageModel
import torch

# Define the model name and other parameters
model_name = "imsanjoykb/deepSQL-R1-distill-8B"
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load the model and tokenizer from Hugging Face
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define the prompt template
odoo_text2sql_prompt = """Below is an instruction describing a task related to generating a SQL query specifically for Odoo's database structure. The input provides relevant context about Odoo models or data fields from {db_schema}. Write a SQL query that fulfills the given task using Odoo's database schema.

### Instruction:
Generate a SQL query in the context of Odoo to {}

### Input:
{}

### Response:
{}
"""

# Optionally, use a TextStreamer for continuous inference
from transformers import TextStreamer

# Prepare the input text for continuous inference
instruction = ""
input_text = "What is the top profitable product?"
output_text = ""

# Tokenize the input text
inputs = tokenizer(
    [
        odoo_text2sql_prompt.format(instruction, input_text, output_text)
    ],
    return_tensors="pt"
).to("cuda")

# Initialize the TextStreamer
text_streamer = TextStreamer(tokenizer)

# Generate the output using the model with TextStreamer
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=350)
```
## Model Download
|            **Model**            | **#Total Params** | **#Active Params** | **Context Length** |                         **Download**                         |
| :-----------------------------: | :---------------: | :----------------: | :----------------: | :----------------------------------------------------------: |
|   deepSQL-R1-distill-8B         |        14B        |        2.4B        |        128k        | [🤗 HuggingFace](https://huggingface.co/imsanjoykb/deepSQL-R1-distill-8B) |

# Uploaded  model

- **Developed by:** [Sanjoy Biswas](https://www.linkedin.com/in/imsanjoykb/) 
- **License:** apache-2.0