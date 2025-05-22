from __future__ import annotations
from logging import Logger
from typing import Literal

from llm.medical import DiseaseQuestioner
from llm.education import EducationQuestioner
from llm.sampler import SamplerMethod
import attr 

@attr.define(kw_only=True)
class ExpParams:
    left_name: str
    right_name: str
    num_topics: int
    num_rounds: int  
    num_repeats: int
    questioner_device: str | None = None
    left_device: str | None = None
    right_device: str | None = None
    optimizer: str = "random"  # other options: bayes, embed
    questioner: str = "llama_13b"  # other options gpt3.5 / llama_13b
    num_sample: int = None

    questioner_type: Literal['disease', 'education'] = attr.field(
        default='education',
        validator=attr.validators.in_(['disease', 'education'])
    )

    def __attrs_post_init__(self):
        if self.optimizer == "random":
            assert self.num_sample is None, "num_sample should be none for **random** policy"

        assert self.left_device is None and self.right_device is None and self.questioner_device is None, (
            "The current version does not load models dynamically; device fields must be None"
        )

    def to_string(self):
        return "_".join(str(getattr(self, f.name)) for f in attr.fields(self.__class__))

    def to_long_string(self):
        return "_".join(f"{f.name}:{getattr(self, f.name)}" for f in attr.fields(self.__class__))


def get_questioner_from_exp_param(
    params: ExpParams,
    logger: Logger | None = None
):
    maker = {
        'education': EducationQuestioner.make,
        "disease": DiseaseQuestioner.make
    }[params.questioner_type]

    questioner = maker(
        model=params.questioner,
        logger=logger,
        num_topics=params.num_topics,
        num_samples=params.num_sample or 1,
        sample_method=SamplerMethod.RANDOM
    )
    return questioner

