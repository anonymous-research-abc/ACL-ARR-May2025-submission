from __future__ import annotations

from functools import partial
import attr
from transformers import AutoModelForCausalLM, AutoTokenizer, BlenderbotForConditionalGeneration, BlenderbotTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

import os 
import openai 
from llm.text import get_print_func
import requests
from llm.configure import hf_api_token
import torch


@attr.define
class ChatBotsParams:
    name: str
    tokenizer_func: callable
    model_func: callable
    device: str = 'cpu'
    model_path: str = None

    def __attrs_post_init__(self):
        if self.model_path is None:
            self.model_path = self.name

# 1. Blenderbot
blenderbot = ChatBotsParams(
    name="facebook/blenderbot-3B",
    tokenizer_func = BlenderbotTokenizer.from_pretrained,
    model_func = BlenderbotForConditionalGeneration.from_pretrained,
)

# 2. FLAN-T5 Large
flant5_large = ChatBotsParams(
    name="google/flan-t5-large",
    tokenizer_func=T5Tokenizer.from_pretrained,
    model_func=T5ForConditionalGeneration.from_pretrained,
)

# 3. FLAN-T5 XXL
flant5_xxl = ChatBotsParams(
    name="google/flan-t5-xxl",
    tokenizer_func=T5Tokenizer.from_pretrained,
    model_func=T5ForConditionalGeneration.from_pretrained,
)

# 4. GPT-2 XL
gpt2_xl = ChatBotsParams(
    name="openai-community/gpt2-xl",
    tokenizer_func=GPT2Tokenizer.from_pretrained,
    model_func=GPT2LMHeadModel.from_pretrained,
)

# 5. GPT-J
gptj = ChatBotsParams(
    name="EleutherAI/gpt-j-6b",
    tokenizer_func=AutoTokenizer.from_pretrained,
    model_func=AutoModelForCausalLM.from_pretrained,
)

# 6. DialoGPT
dialogpt = ChatBotsParams(
    name="microsoft/DialoGPT-large",
    tokenizer_func=AutoTokenizer.from_pretrained,
    model_func=AutoModelForCausalLM.from_pretrained,
)

# 7. GPT-Neo
gpt_neo = ChatBotsParams(
    name="EleutherAI/gpt-neo-1.3B",
    tokenizer_func=AutoTokenizer.from_pretrained,
    model_func=AutoModelForCausalLM.from_pretrained,
)

# 8. Zephyr-7B
zephyr_7b = ChatBotsParams(
    name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_func=AutoTokenizer.from_pretrained,
    model_func=AutoModelForCausalLM.from_pretrained,
)

# 9. Llama-7B
llama_7b = ChatBotsParams(
    name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token
    ),
)

# 9-1. LLaMA-7b
llama_7b_4bit = ChatBotsParams(
    name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token,
        load_in_4bit=True
    ),
)

# 10. Llama-13B
llama_13b = ChatBotsParams(
    name="meta-llama/Llama-2-13b-chat-hf",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token,
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token
    ),
)

# 10-1. Llama-13B
llama_13b_4bit = ChatBotsParams(
    name="meta-llama/Llama-2-13b-chat-hf",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token,
        load_in_4bit=True
    ),
)

# 11. LLaMA-3-8b
llama3_8b = ChatBotsParams(
    name="meta-llama/Meta-Llama-3-8B",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token
    ),
)

# 12. LLaMA-70b-4bit

llama_70b_4bit = ChatBotsParams(
    name="meta-llama/Llama-2-70b-chat-hf",
    tokenizer_func=partial(
        AutoTokenizer.from_pretrained,
        use_auth_token=hf_api_token
    ),
    model_func=partial(
        AutoModelForCausalLM.from_pretrained,
        use_auth_token=hf_api_token,
        load_in_4bit=True
    ),
)

# 13. gpt3.5
gpt_35 = ChatBotsParams(
    name="gpt-3.5-turbo",
    tokenizer_func = None,
    model_func = None,
)


ALL_MODELS = [
    blenderbot,
    flant5_large,
    flant5_xxl,
    gpt2_xl,
    gptj,
    dialogpt,
    gpt_neo,
    zephyr_7b,
    llama_7b,
    llama_7b_4bit,
    llama_13b,
    llama_13b_4bit,
    llama3_8b,
    llama_70b_4bit,
    gpt_35
]


def build_tokenizer_n_model(params: ChatBotsParams, device: str = None):
    tokenizer = params.tokenizer_func(params.model_path)    
    model_kwargs = {'device_map': device} if device else {}
    model = params.model_func(params.model_path, **model_kwargs)    
    return tokenizer, model

def build_tokenizer_n_model_legacy(params: ChatBotsParams, device:str=None):
    if device is None:
        return params.tokenizer_func(params.model_path), params.model_func(params.model_path)
    return params.tokenizer_func(params.model_path), params.model_func(params.model_path, device_map = device)


def send_request(model_name: str, question: str, model_names_n_ports: dict, verbose: bool =True):
    """Function to send a POST request to a given URL."""
    port = model_names_n_ports[model_name]
    url = f'http://127.0.0.1:{port}/generate/'
    verbose_print = get_print_func(verbose)
    verbose_print(f"{url=}")
    payload = {'text': question}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        generated_text = response.json().get('generated_text', '')
    else:
        generated_text = f"Failed to generate text from {url}. Status code: {response.status_code}"
    return generated_text

