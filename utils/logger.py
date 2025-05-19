import random
import string
from datetime import datetime 
import logging 
import os
import sys
import secrets
import socket

def rnd_str(size=10):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(size))
    return random_string


def timed_rnd_str(size: int = 10):
    return f'{datetime.now().strftime("%Y-%m-%d %H:%M")}_{rnd_str(size=size)}'


def get_print_func(verbose: bool):
    if verbose:
        return print
    def _no_opt(*arg, **kwargs):
        return
    return _no_opt

LOG_DIR = '/path/to/log/directory'

def get_log_dir():
    hostname = socket.gethostname()
    if hostname.startswith("anonymous-env-1"):
        return '/path/to/env1/log'
    elif hostname.startswith("anonymous-env-2"):
        return '/path/to/env2/log'
    else:  # default or cloud environment
        return '/path/to/cloud-env/log'

def get_logger(suffix: str = None): 
    print("Logger initialized")
    logger = logging.getLogger('ml.llm')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
            
    current_time = datetime.now()
    suffix = suffix or "_"
    random_str = secrets.token_hex(4)
    filename = f"logfile_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_{random_str}{suffix}.log"

    log_filepath = os.path.join(get_log_dir(), filename)
    file_handler = logging.FileHandler(log_filepath) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Ensure root logger uses the same formatter
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    return logger 


def get_logger_with_filename(filename: str, group: str = "default", suffix: str = "html"): 
    print(f"Creating logger for file: {filename}")
    logger = logging.getLogger('ml.llm')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logdir = os.path.join(get_log_dir(), group, filename)
    os.makedirs(logdir, exist_ok=True)
    log_filepath = os.path.join(logdir, f"log.{suffix}")     
    file_handler = logging.FileHandler(log_filepath) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Ensure root logger uses the same formatter
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    return logger 
