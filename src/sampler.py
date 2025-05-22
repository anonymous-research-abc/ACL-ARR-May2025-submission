from __future__ import annotations

import attr 
from typing import Any 
import random 

from enum import Enum

class SamplerMethod(Enum):
    RANDOM = "random"
    BAYES = "bayes"
    EMBED = "embed"


@attr.define(kw_only=True, frozen=True)
class Sampler: 
    topic_list: tuple = attr.field(converter=tuple)

    def sample(self, num_topics: int, num_samples: int, x: Any = None, **kwargs) -> list[list[str]]:
        raise NotImplementedError("Please build a subclass and implement it")


@attr.define(kw_only=True, frozen=True) 
class RandomSampler(Sampler):
    def sample(self, num_topics: int, num_samples: int, x: Any = None, **kwargs) -> list[list[str]]:
        return [
            sorted(random.sample(self.topic_list, num_topics))
            for _ in range(num_samples)
        ]
