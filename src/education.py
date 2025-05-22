from __future__ import annotations
import os 

from llm.configure import USE_DIR
import pandas as pd 
import time
from logging import Logger 
from llm.util import write 

from functools import partial 
import openai
from logging import Logger 
from llm.util import write 

from llm.questioner import Questioner
import attr
from llm.sampler import Sampler, RandomSampler, SamplerMethod
from llm.answerer import Answerer


_EDU_DIR = os.path.join(USE_DIR, "edu")
_EDU_FILE = os.path.join(_EDU_DIR, "NGSS_Table_All_3C_utf8.pqt")



def get_default_topics_edu():
    """
    read NGSS dataset 
    """
    edu_dataset = pd.read_csv(_EDU_FILE)
    topics_list_edu = edu_dataset[['Code-1', 'Code-2', 'Describe']].fillna('').values.tolist()
    return topics_list_edu



@attr.define(kw_only=True)
class EducationQuestioner(Questioner):
    sampler: Sampler = attr.field()

    @classmethod
    def make(
        cls, 
        model: str, 
        logger: Logger | None,
        num_topics: int,
        num_samples: int,
        sample_method: SamplerMethod = SamplerMethod.RANDOM, 
        max_length: int = 800,
    ) -> EducationQuestioner:
        if sample_method == SamplerMethod.RANDOM:
            df = pd.read_parquet(_EDU_FILE)
            topic_list = df['full_description'].to_list()
            sampler = RandomSampler(
                    topic_list=topic_list,
                )
            return cls(
                model=model,
                logger=logger,
                max_length=max_length,
                num_topics=num_topics,
                num_samples=num_samples,
                sampler=sampler 
            )
        raise NotImplementedError(f"{sample_method.name=} not implemented")

    def get_sampler(self) -> RandomSampler:
        return self.sampler

    def _generate_prompt_default(self, topic_list: tuple): 
        return self._generate_prompt_chatgpt(topic_list)
    
    def _generate_prompt_chatgpt(self, topic_list: tuple):
        augmented_topic_tuple = [f"TOPIC {i}: {elem}" for i, elem in enumerate(topic_list)]
        full_topic_description = "; ".join(augmented_topic_tuple)
        prompt = (
            f"You are an experienced K-12 science educator designing short-answer or open-ended quiz questions based on the Next Generation Science Standards (NGSS).\n\n"
            f"**Task:** Create a single, open-ended question related to the following NGSS topic(s):\n\n"
            f"**Topics (can be multiple):** \"{full_topic_description}\"\n\n"
            f"**Question Requirements:**\n"
            f"- The question should be in a short-answer or open-ended format.\n"
            f"- It should encourage students to explain, describe, or analyze a concept rather than provide a simple factual answer.\n"
            f"- Use age-appropriate language and ensure clarity.\n"
            f"- The question should assess students’ understanding of the topic and promote critical thinking.\n"
            f"- Do **not** provide the answer—only generate the question itself.\n\n"
            f"**Note:** Use the NGSS code(s) to infer the appropriate grade level. If multiple topics are mentioned, select the highest grade level among them.\n\n"
            f"**Additional Instruction:**\n"
            f"After generating the question, specify the most appropriate K-12 grade level for which this question is best suited.\n\n"
            f"Format your response as follows:\n"
            f"- **Question:** <Insert question here>\n"
            f"- **Recommended Grade Level:** <Insert grade level here>"
        )
        system_content = "You are an experienced science educator."
        return {"prompt": prompt, "system_content": system_content}
    
    def _generate_prompt_llama13(self, topic_list: tuple):
        prompt_info = self._generate_prompt_chatgpt(topic_list)
        prompt_info["prompt"] = prompt_info["prompt"] + prompt_info["system_content"]
        prompt_info["system_content"] = ""
        return prompt_info
    
@attr.define(kw_only=True)
class EducationAnswerer(Answerer):
    def _generate_response_chatgpt(self, question):
        prompt = (
            f"As an experienced teacher for the grade, "
            f"please answer the following question and explain in a way that is engaging, "
            f"clear, and suit for students at this grade level to understand.  \n\n{question} "
        )
    
        system_content = "You are an education professional and you are answering questions for students."
        return self.response_func(prompt, system_content)
    
    def _generate_response_non_chatgpt(self, question: str) -> str:
        prompt = (
            f"As an experienced teacher for the grade, "
            f"please answer the following question and explain in a way that is engaging, "
            f"clear, and suit for students at this grade level to understand.  \n\n{question} "
        )
        system_content = ""
        return self.response_func(prompt, system_content)
