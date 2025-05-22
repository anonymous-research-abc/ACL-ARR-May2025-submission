from __future__ import annotations

import pandas as pd
import os
import json
import random
import time 
from functools import partial 
import attr
import openai
from logging import Logger
from llm.questioner import Questioner
import llm.configure as _
from llm.configure import USE_DIR
from llm.util import write 
from llm.sampler import Sampler, RandomSampler, SamplerMethod
from llm.answerer import Answerer

_DISEASE_FILE = os.path.join(USE_DIR, "icd_blocks.csv")
_DISEASE_FILE_LEAVES = os.path.join(USE_DIR, "icd_only_blocks.csv")
_QUESTIONS_FILE_SCENARIOS = os.path.join(USE_DIR, "phrases_train_scenarios_only.json")

def get_default_q_and_d():
    disease_dataset = pd.read_csv(_DISEASE_FILE)
    disease_list = disease_dataset["Description"].tolist()
    with open(_QUESTIONS_FILE_SCENARIOS, "r", encoding="utf-8") as f:
        question_dataset = json.load(f)
    question_templates = [record["scenario"] for record in question_dataset]
    return question_templates, disease_list

def get_default_q_and_d_only_leaves():
    disease_dataset = pd.read_csv(_DISEASE_FILE_LEAVES)
    disease_list = disease_dataset["Description"].tolist()
    with open(_QUESTIONS_FILE_SCENARIOS, "r", encoding="utf-8") as f:
        question_dataset = json.load(f)
    question_templates = [record["scenario"] for record in question_dataset]
    return question_templates, disease_list


@attr.define(kw_only=True)
class DiseaseQuestioner(Questioner):
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
    ) -> DiseaseQuestioner:
        if sample_method == SamplerMethod.RANDOM:
            _, topic_list = get_default_q_and_d()
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
        q, _ = get_default_q_and_d()
        topic_text = ", ".join(topic_list)
        template = random.choice(q)
        
        prompt = (
            f"Please refer to the medical scenario described in the template below, "
            f"and create a patient case scenario involving the two medical topics: "
            f"'{topic_text}'. The case should interweave these themes into a "
            f"coherent and medically plausible scenario, and generate a question "
            f"concerning this patient. Build your own case.\n"
            f"Here is a template scenario: {template}.\n"
        )
        system_content = (
            "You are a medical school professor and you are " 
            "designing questions for medical school students"
        )
        return {
            "prompt": prompt, 
            "system_content": system_content
        }

    def _generate_prompt_llama13(self, topic_list):
        q, _ = get_default_q_and_d()
        topic_text = ", ".join(topic_list)
        template = random.choice(q)

        prompt = (
            f"Your role: You are a medical school professor and you are designing questions for medical school students. "
            f"Please refer to the medical scenario described in the template below, "
            f"and create a patient case scenario involving the two medical topics: '{topic_text}'. "
            f"The case should interweave these themes into a coherent and medically plausible scenario, "
            f"and generate a question concerning this patient. Build your own case.\n"
            f"Here is a template scenario: {template}.\n "
            f"Note: For the cases in the template, only learn the format here, please do not copy the scenarios here."
        )
        system_content = ""
        return {
            "prompt": prompt, 
            "system_content": system_content
        }

def _generate_question_gpt(
    disease_tuple: list,  
    max_length: int = 800, 
    logger: Logger | None  = None
):
    _ = partial(write, logger=logger)
    q, _ = get_default_q_and_d()

    topic_text = ", ".join(disease_tuple)
    template = random.choice(q)
    prompt = (f"Please refer to the medical scenario described in the template below, "
            f"and create a patient case scenario involving the two medical topics: '{topic_text}'. "
            f"The case should interweave these themes into a coherent and medically plausible scenario, "
            f"and generate a question concerning this patient. Build your own case.\n"
            f"Here is a template scenario: {template}.\n ")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical school professor and you are designing questions for medical school students"},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = completion.choices[0].message["content"] if "content" in completion.choices[0].message else "Text extraction error"

    limited_text = generated_text[:max_length]
    return limited_text, disease_tuple


def generate_question(
    disease_tuple: list,  
    max_length=800,
    logger: Logger | None  = None
):
    _write = partial(write, logger=logger)

    for j in range(50):
        try:
            question, disease_tuple = _generate_question_gpt(
                disease_tuple,  
                max_length= max_length, 
                logger=logger
            )
        except Exception as e:
            print(e)
            _write(f"{e}")
            if j > 5 and j < 30:
                print("sleeping for 10 sec")
                time.sleep(10)
            elif j >= 30:
                print("sleeping for 180 sec")
                time.sleep(180)
            continue
        break
    return question, disease_tuple


def _generate_answer_gpt(question:str, max_length:int=500):
    prompt = f"As a clinical expert, how would you respond to this medical case?\n\n{question}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical doctor and you are seeing the patient"},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = completion.choices[0].message["content"] if "content" in completion.choices[0].message else "Text extraction error"

    limited_text = generated_text[:max_length]
    return limited_text


def generate_answer_gpt(question:str, max_length:int=500):
     #print("-"*10, "calling gpt to answer")
     for j in range(50):
        try:
            question = _generate_answer_gpt(question,  max_length= max_length)
        except Exception as e:
            print(e)
            # TODO: no commit info needs to be changed. 
            # _info(f"{e}")
            if j > 5 and j < 30:
                print("sleeping for 10 sec")
                time.sleep(10)
            elif j >= 30:
                print("sleeping for 180 sec")
                time.sleep(180)
            continue
        break
     return question


@attr.define(kw_only=True)
class DiseaseAnswerer(Answerer):
    def _generate_response_chatgpt(self, question):
        prompt = f"As a clinical expert, how would you respond to this medical case?\n\n{question}"
        system_content = "You are a medical doctor and you are seeing the patient"
        return self.response_func(prompt, system_content)
    
    def _generate_response_non_chatgpt(self, question: str) -> str:
        prompt = f"As a clinical expert, how would you respond to this medical case?\n\n{question}"
        system_content = ""
        return self.response_func(prompt, system_content)
