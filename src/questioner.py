from __future__ import annotations

import attr 
from llm.llm_bot import BotBase
from llm.sampler import Sampler
from typing import Any, Generator

@attr.define(kw_only=True)
class Questioner(BotBase):
    num_topics: int = attr.field()
    num_samples: int = attr.field()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # handle logger first. 
        assert self.model in ["llama_13b", "gpt_35"]
        
    def get_sampler(self) -> Sampler:
        raise NotImplementedError("Please build a subclass and implement it")
        
    def generate_questions(self, x: Any = None, **kwargs)->Generator[tuple[str, Any], None, None]: 
        sampler = self.get_sampler()
        for topic_list in sampler.sample(self.num_topics, self.num_samples, x, **kwargs):
            response, topic_list = self._generate_question(topic_list)
            yield response, topic_list


    def _generate_question(self, topic_list: tuple):
        prompt_info = self.generate_prompt(topic_list)
        prompt_info["prompt"] += (
            "\n Please generate response with no more than "
            f"{self.max_length} tokens"
        )
        self.maybe_write_log(f"prompt: {prompt_info['prompt']}")
        return self.response_func(
            prompt_info["prompt"], prompt_info["system_content"]
        ), topic_list

    def generate_prompt(self, topic_list: tuple): 
        try:
            if self.model == 'gpt_35':
                return self._generate_prompt_chatgpt(topic_list)
            else:
                return self._generate_prompt_llama13(topic_list)
        except NotImplementedError as e:
            return self._generate_prompt_default(topic_list)


    def _generate_prompt_default(self, topic_list: tuple): 
        raise NotImplementedError("Please build a subclass and implement it")
    
    def _generate_prompt_chatgpt(self, topic_list: tuple): 
        raise NotImplementedError("Please build a subclass and implement it")

    def _generate_prompt_llama13(self, topic_list: tuple): 
        raise NotImplementedError("Please build a subclass and implement it")

