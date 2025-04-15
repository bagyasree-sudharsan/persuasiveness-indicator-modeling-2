# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="llama3.1")

# response = llm.invoke("The first man on the moon was ...")
# print(response)

# import transformers
# import torch

# model_id = "meta-llama/Meta-Llama-3-8B"

# pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
# pipeline("What is the capital of India? Why?")

from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)