from __future__ import annotations

from llm.configure import _DIRECTORY_DIR, _STUB_DIR, SentenceTransformerServerParams, ServerParams

from typing import Union, Callable, Any
import requests
import glob
import json
import attr
import random 
import openai
import llm.configure as _
from logging import Logger
from functools import partial
from llm.util import write 
import time
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Literal

_MAX_REPEAT = 50 

@attr.define(kw_only=True)
class BotBase:
    model: Literal[
        'blenderbot', 'flant5_large', 'flant5_xxl', 'gpt2_xl', 'gptj',
        'dialogpt', 'gpt_neo', 'zephyr_7b', 'llama_7b', 'llama_13b',
        'llama_7b_4bit', 'llama_13b_4bit', 'llama3_8b', 'llama_70b_4bit',
        'gpt_35'
    ] = attr.field(
        validator=attr.validators.in_([
            'blenderbot', 'flant5_large', 'flant5_xxl', 'gpt2_xl', 'gptj',
            'dialogpt', 'gpt_neo', 'zephyr_7b', 'llama_7b', 'llama_13b',
            'llama_7b_4bit', 'llama_13b_4bit', 'llama3_8b', 'llama_70b_4bit',
            'gpt_35'
        ])
    )
    logger: Logger | None = attr.field(default=None)
    max_length: int = attr.field(default=800)
    maybe_write_log: Callable[[str], None] = attr.field(init=False)
    response_func: Callable[[str, str], str] = attr.field(init=False)
    

    def __attrs_post_init__(self):
        self.maybe_write_log = partial(write, logger=self.logger)

        if self.model == 'gpt_35':
            self.response_func = partial(call_gpt, logger=self.logger, max_length=self.max_length)
        else: 
            sparams = find_llm_server(self.model)    
            self.response_func = partial(
                call_server, sparams=sparams, logger=self.logger, max_length=self.max_length
            )


def _sleep_func(attempted: int, logger: Logger | None):
    _write = partial(write, logger=logger) 
    if 5 < attempted < 30:
        _write("sleeping for 10 sec")
        print("sleeping for 10 sec")
        time.sleep(10)
    elif attempted >= 30:
        _write("sleeping for 180 sec")
        print("sleeping for 180 sec")
        time.sleep(180)

def _call_gpt(
    message_or_prompt: str, 
    system_content: str, 
    max_length: int = 800, 
    logger: Logger | None  = None
):
    _write = partial(write, logger=logger) 
    message_or_prompt = f"{message_or_prompt}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": message_or_prompt}
        ]
    )
    generated_text = completion.choices[0].message.get("content", "Text extraction error")
    limited_text = generated_text[:max_length]
    return limited_text


def call_gpt(
    message_or_prompt: str, 
    system_content: str, 
    max_length: int = 800, 
    logger: Logger | None  = None, 
    sleep_func: Callable[[int, Logger|None], None] = _sleep_func
):
    _write = partial(write, logger=logger)
    for j in range(_MAX_REPEAT):
        try:
            response = _call_gpt(
                message_or_prompt=message_or_prompt,  
                system_content=system_content,
                max_length=max_length, 
                logger=logger
            )
        except Exception as e:
            _write(f"GPT error: {e}")
            sleep_func(j, logger)
            continue
        break
    return response

def _call_server(
    message_or_prompt: str, 
    system_content: str,
    sparams: ServerParams, 
    max_length: int = 800, 
    logger: Logger | None  = None
):
    _write = partial(write, logger=logger) 
    
    url = f"http://127.0.0.1:{sparams.port}/generate/"
    payload = {'text': f"{message_or_prompt} "}
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        completion_text = response.json().get('generated_text', '')
    else:
        raise ValueError(f"Failed at {url}: {response.status_code}, {response.text}")

    if completion_text.startswith(message_or_prompt):
        completion_text = completion_text[len(message_or_prompt):]

    return completion_text[:max_length]

def call_server(
    message_or_prompt: str, 
    system_content: str,
    sparams: ServerParams, 
    max_length: int = 800, 
    logger: Logger | None  = None, 
    sleep_func: Callable[[int, Logger|None], None] = _sleep_func
):
    _write = partial(write, logger=logger)
    for j in range(_MAX_REPEAT):
        try:
            response = _call_server(
                message_or_prompt=message_or_prompt,  
                system_content=system_content,
                max_length=max_length, 
                sparams=sparams, 
                logger=logger
            )
        except Exception as e:
            _write(f"Server error: {e}")
            sleep_func(j, logger)
            continue
        break
    return response

def check_alive(params: Union[ServerParams, SentenceTransformerServerParams]):
    url = f"http://127.0.0.1:{params.port}/heartbeat/"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return False
        if isinstance(params, ServerParams):
            model_name = response.json().get('model_name', '')
            return model_name == params.model_name
        return True
    except:
        return False
    

def probe_servers():
    files = glob.glob(_DIRECTORY_DIR + "/*.json")
    files += glob.glob(_STUB_DIR + "/*.json")
    server_params_list = []
    for filepath in files:
        with open(filepath, 'r') as file:
            data = json.load(file)
            server_params_list.append(attr.evolve(ServerParams(**data), device='na'))
    unique_params = list(set(server_params_list))
    return [p for p in unique_params if check_alive(p)]


def find_llm_server(model_name, randomized: bool =True):
    alive_servers = probe_servers()
    matched_servers = [c for c in alive_servers if c.model_name == model_name]
    if len(matched_servers) == 0:
        return None
    if randomized:
        return random.choice(matched_servers)
    return matched_servers[0]
