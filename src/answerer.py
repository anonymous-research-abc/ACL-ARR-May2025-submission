from __future__ import annotations

import attr 
from llm.llm_bot import BotBase

@attr.define(kw_only=True)
class Answerer(BotBase):
    def generate_response(self, question: str) -> str:
        if self.model == 'gpt_35':
            return self._generate_response_chatgpt(question)
        else:
            return self._generate_response_non_chatgpt(question)
    
    
    def _generate_response_chatgpt(self, question: str) -> str:
        raise NotImplementedError("Please build a subclass and implement it")
    
    def _generate_response_non_chatgpt(self, question: str) -> str:
        raise NotImplementedError("Please build a subclass and implement it")