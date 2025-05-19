from __future__ import annotations

import socket
import sys 
import os 
import openai 
import attr
from collections import defaultdict

api_key = "your-openai-api-key"
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = api_key

hf_api_token = "your-huggingface-api-token"
print("OpenAI and HuggingFace keys set up.")

hostname = socket.gethostname()

if hostname.startswith('anonymous-env-1'): # Generic paths based on environment
    sys.path.append('/path/to/workspace/ml_code')
    USE_DIR = "/path/to/workspace/ml_code/data"
    _SCRATCH_DIR = "/path/to/scratch"
    EXP_DIR = '/path/to/scratch/exp'

elif hostname.startswith('anonymous-env-2'):
    sys.path.append("/path/to/another/git_code/ml")
    USE_DIR = "/path/to/another/"
    _SCRATCH_DIR = "/path/to/another/scratch"
    EXP_DIR = '/path/to/another/scratch/exp'

else:  # default or cloud environment
    sys.path.append("/path/to/cloud-env/ml")
    USE_DIR = "/path/to/cloud-env/ml/data"
    _SCRATCH_DIR = "/path/to/cloud-env/scratch"
    EXP_DIR = "/path/to/cloud-env/scratch/exp"

_DIRECTORY_DIR = f"{_SCRATCH_DIR}/active_server"
_STUB_DIR = f"{_SCRATCH_DIR}/stub_active_server"
_DIRECTORY_ST_DIR = f"{_SCRATCH_DIR}/st_active_server"


@attr.define(kw_only=True, frozen=True)
class ServerParams:
    port:int
    device:str
    model_name:str


@attr.define(kw_only=True, frozen=True)
class SentenceTransformerServerParams:
    port:int
    device:str
