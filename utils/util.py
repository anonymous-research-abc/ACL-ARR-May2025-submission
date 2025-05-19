
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from logging import Logger
import torch 
import gc
import time 

def build_logger(dir_name: str | Path):
    """
    Builds and configures a logger that logs messages to a file and optionally to the console.
    
    Args:
        dir_name (str | Path): The directory where the log file will be created.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory exists
    dir_path = Path(dir_name)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Create log file name with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = dir_path / f"{timestamp}.log"

    # Get the project logger
    logger = logging.getLogger('ml.llm')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Remove all existing file handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    # Formatter to include current time in each log message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Console handler (add only if not already present)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add the new file handler
    logger.addHandler(file_handler)

    return logger


def compute_distance(answer1: str, answer2: str, model_s: SentenceTransformer):
    embeddings = model_s.encode([answer1, answer2])
    return cosine(embeddings[0], embeddings[1])

def write(message: str, logger: Logger | None,): 
    if logger is not None:
        logger.info(message)


def _repeat_to_device(_model: torch.nn.Module, _device: str):
    for _ in range(4000):
        try:
            res = _model.to(_device)
        except Exception as e:
            print(f"load failed {e}; sleeping for 10 seconds v2")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
            continue
        return res


class DataCollector:
    def __init__(self, columns:list, output_dir:str) -> None:
        self.rows = []
        self.count = 0
        self.columns = columns
        self.output_dir = output_dir

    def add_row(self, row:list):
        self.rows.append([self.count]+row)

    def increment_and_save(self)->pd.DataFrame:
        self.count += 1
        fname = f"data_round_{self.count}.pqt"
        return self.save(fname)

    def save(self, fname:str = None)->pd.DataFrame:
        fname = fname or "data.pqt"
        df = pd.DataFrame(self.rows, columns=self.columns)
        df = df.set_index("Topic index")
        df["Mean Distance"] = df['Distance'].groupby(df.index).mean().reindex(df.index)
        df["Var Distance"] = df['Distance'].groupby(df.index).var().reindex(df.index)
        fname = os.path.join(self.output_dir,fname)
        df.to_parquet(fname)
        return df
    
    