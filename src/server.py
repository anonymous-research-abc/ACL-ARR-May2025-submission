from __future__ import annotations

import os
import gc
import attr
import torch
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
from typing import Any, Literal, Optional
import concurrent.futures
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from skopt import gp_minimize

import llm.configure as _ 
from llm.llm_bot import find_llm_server
from llm.distance import get_compute_distance_func
from llm.embed_opt import MLP, EmbedExpParams, find_latest_checkpoint, get_root_disease, tree_to_vector_map
from llm.experiments import ExpParams, get_questioner_from_exp_param

from llm.chatbots import (
    ChatBotsParams, 
    blenderbot,
    build_tokenizer_n_model,
    dialogpt, 
    flant5_large, 
    flant5_xxl,
    gpt2_xl,
    gpt_35,
    gpt_neo,
    gptj,
    llama3_8b,
    llama_13b,
    llama_13b_4bit,
    llama_70b_4bit,
    llama_7b,
    llama_7b_4bit,
    zephyr_7b, 
    ALL_MODELS
)
from llm.configure import (
    EXP_DIR, 
    ServerParams
)
from llm.disease import generate_answer_gpt, generate_question, get_default_q_and_d, get_default_q_and_d_only_leaves
from llm.util import DataCollector, _repeat_to_device, compute_distance
import llm.configure as _ 
from llm.medical import DiseaseAnswerer
from llm.education import EducationAnswerer
from llm.answerer import Answerer

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

__all__ = [
    # Model related
    "blenderbot",
    "flant5_large",
    "flant5_xxl",
    "gpt2_xl",
    "dialogpt",
    "gpt_neo",
    "zephyr_7b",
    "llama_7b",
    "llama_7b_4bit",
    "llama_13b",
    "llama_13b_4bit",
    "llama3_8b",

    # Question generation
    "generate_question",

    # Sampling methods
    "sigmoid",
    "sample_topics",
    "sample_topics_embed",

    # Objective functions
    "objective_function_bayes",
    "objective_function_emb",

    # Experimental utilities
    "run_exp_random",
    "run_exp_bayes",
    "run_exp_embed",

    # Embedding utilities
    "create_siamese_jobs",
    "load_decoder_model",

    # Parallel execution
    "parallel_execute",

    # Server-related utilities
    "ServerParams",
    "generate_answer_from_server",
    "build_answer_func_from_server",
    "find_llm_server",

    # Parameter Classes
    "ExpParamsEHP",

    # Data Collection utility
    "DataCollector",
]


logger = None 


def _info(message: str):
    global logger
    if logger is None:
        logger = get_logger()
    logger.info(message)


QUESTIONER_TOKENIZER = None
QUESTIONER_MODEL = None
QUESTIONER_NAME:ChatBotsParams = None

MAP_MODELS = {
    "blenderbot": blenderbot,
    "flant5_large": flant5_large,
    "flant5_xxl": flant5_xxl,
    "gpt2_xl": gpt2_xl,
    "gptj": gptj,
    "dialogpt": dialogpt,
    "gpt_neo": gpt_neo,
    "zephyr_7b": zephyr_7b,
    "llama_7b": llama_7b,
    "llama_13b": llama_13b,
    "llama_7b_4bit": llama_7b_4bit,
    "llama_13b_4bit": llama_13b_4bit,
    "llama3_8b" : llama3_8b,
    "llama_70b_4bit" : llama_70b_4bit,
    "gpt_35" : gpt_35,
}

K_LIST = [10, 30]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_topics(x, disease_list, num_topic, num_sample):
    probabilities = sigmoid(x)
    probabilities /= np.sum(probabilities)
    return [np.random.choice(disease_list, size=num_topic, replace=False, p=probabilities).tolist() for _ in range(num_sample)]

def sample_topics_embed(x, disease_list, num_topic, num_sample):
    assert np.all(x>=0)
    probabilities = x
    probabilities /= np.sum(probabilities + 1e-10)
    probabilities += 1e-6
    probabilities /= np.sum(probabilities)
    return [np.random.choice(disease_list, size=num_topic, replace=False, p=probabilities).tolist() for _ in range(num_sample)]


def objective_function_bayes(
    x: list[float], 
    disease_list: list[str], 
    params, 
    questioner, 
    left_answerer, 
    right_answerer, 
    dist_func, 
    data_collector
):
    """
    Objective function for Bayesian optimization experiments.
    """
    sampled_topics = sample_topics(
        np.array(x), 
        disease_list, 
        params.num_topics, 
        params.num_sample
    )

    distances = []
    for topics in sampled_topics:
        topics_repr = "***".join(topics)
        
        for question_number in range(params.num_repeats):
            question, _ = questioner.generate_question(topics)
            answer_left, answer_right = parallel_execute(
                left_answerer, right_answerer, question
            )

            distance = dist_func(answer_left, answer_right)
            distances.append(distance)

            row = [
                topics_repr, 
                question_number, 
                question, 
                answer_left, 
                answer_right, 
                distance, 
                np.nan, 
                np.nan
            ]

            data_collector.add_row(row)
            _info(
                f"Topics={topics_repr}, "
                f"QuestionNum={question_number}, "
                f"Distance={distance}"
            )

    data_collector.increment_and_save()
    return -np.mean(distances)

@attr.define(kw_only=True)
class ExpParamsEHP(ExpParams):
    kappa: float = 1.96
    acq_optimizer: str = "auto"

def parallel_execute(left_answer: Answerer, right_answer: Answerer, question: str):
    futures_dict = {}
    result_map = {}

    left_func = partial(left_answer.generate_response, question=question)
    right_func = partial(right_answer.generate_response, question=question)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for name, func in zip(['left', 'right'], [left_func, right_func]):
            future = executor.submit(func)
            futures_dict[future] = name

        for future in concurrent.futures.as_completed(futures_dict):
            name = futures_dict[future]
            try:
                result = future.result()
                result_map[name] = result
            except Exception as e:
                print(f"{name}: Failed with {e}")
                _info(f"{name}: Failed with {e}")
                raise e 

    return result_map['left'], result_map['right']

def get_answerer(
    model_name: str, 
    questioner_type: str, 
    logger: Logger | None, 
):
    assert questioner_type in ["disease","education"]
    if questioner_type == "disease":
        return DiseaseAnswerer(model=model_name, logger=logger, max_length=65536)
    elif questioner_type == "education":
        return EducationAnswerer(model=model_name, logger=logger, max_length=65536)
    assert False

def run_exp_random(
    params:ExpParams, 
    *, 
    group: str = 'default'
):
    """
    Run a random experiment using specified experimental parameters.

    Args:
        params (ExpParams): Experiment configuration parameters.
        group (str, optional): Group identifier for logging purposes. Defaults to 'default'.

    Returns:
        pd.DataFrame: DataFrame containing experiment results.
    """
    global logger
    filename = params.to_long_string()
    logger = get_logger_with_filename(filename, group=group)

    gc.collect()
    torch.cuda.empty_cache()

    assert params.optimizer == "random"
    assert params.questioner in ["llama_13b", "gpt_35"]

    output_dir = os.path.join(EXP_DIR, params.to_string())
    os.makedirs(output_dir, exist_ok=True)

    dist_func = get_compute_distance_func()
    questioner = get_questioner_from_exp_param(params, logger)

    left_answer = get_answerer(params.left_name, params.questioner_type, logger)
    left_name = MAP_MODELS[params.left_name]

    right_answer = get_answerer(params.right_name, params.questioner_type, logger)
    right_name = MAP_MODELS[params.right_name]
    

    _info(f"{output_dir=}")
    _info(f"{left_name.name=}, {right_name.name=}")

    left_repr = left_name.name.replace("/", "_")
    right_repr = right_name.name.replace("/", "_")
    if right_repr == left_repr:
        right_repr += ".2"

    columns = [
        'Topic index',
        'Topic',
        'Question Number',
        'Question',
        f'{left_repr} answer',
        f'{right_repr} answer',
        'Distance',
        'Mean Distance',
        'Var Distance'
    ]

    data = []

    for i in range(params.num_rounds):
        distances = []
        for question_number in tqdm(range(params.num_repeats)): 
            question, topic_list = list(questioner.generate_questions())[0]
            answer_left, answer_right = parallel_execute(left_answer, right_answer, question=question)
            distance = dist_func(answer_left, answer_right)
            distances.append(distance)
            
            topics_repr = "***".join(topic_list)
            new_row = [
                i,
                topics_repr,
                question_number,
                question,
                answer_left,
                answer_right,
                distance,
                np.nan,
                np.nan
            ]
            data.append(new_row)
            _info(f"{i=}, {question_number=}, {distance=}")

        fname = f"data_round_{i}.pqt"
        df = pd.DataFrame(data, columns=columns)
        df = df.set_index("Topic index")
        df["Mean Distance"] = df['Distance'].groupby(df.index).mean().reindex(df.index)
        df["Var Distance"] = df['Distance'].groupby(df.index).var().reindex(df.index)

        fname = os.path.join(output_dir, fname)
        _info(f"{fname=}")
        df.to_parquet(fname)

    # Save final results to a summary file
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index("Topic index")
    df["Mean Distance"] = df['Distance'].groupby(df.index).mean().reindex(df.index)
    df["Var Distance"] = df['Distance'].groupby(df.index).var().reindex(df.index)

    final_fname = os.path.join(output_dir, "data.pqt")
    os.makedirs(output_dir, exist_ok=True)
    _info(f"{final_fname=}")
    df.to_parquet(final_fname)

    gc.collect()
    torch.cuda.empty_cache()

    return df



def run_exp_bayes(
    params: ExpParams,
    verbose: bool = True,
    *,
    group: str = "default"
):
    """
    Run a Bayesian optimization experiment based on provided parameters.

    Args:
        params (ExpParams): Configuration for the Bayesian optimization experiment.
        verbose (bool, optional): Verbosity flag for logging details. Defaults to True.
        group (str, optional): Group identifier for logging. Defaults to "default".

    Returns:
        tuple: DataFrame containing experiment results and optimization results object.
    """
    global logger 
    filename = params.to_long_string()
    logger = get_logger_with_filename(filename, group=group)
    
    gc.collect()
    torch.cuda.empty_cache()

    assert params.optimizer == "bayes"
    assert params.questioner in ["llama_13b","gpt3.5"]

    output_dir = os.path.join(EXP_DIR, params.to_string())
    os.makedirs(output_dir, exist_ok=True)

    dist_func = get_compute_distance_func()

    _, disease_list = get_default_q_and_d()
    num_dimensions = len(disease_list)
    
    if params.questioner == "llama_13b":
        question_f = get_generate_question_func(
            params, model_s=SentenceTransformer('all-MiniLM-L6-v2')
        )
    elif params.questioner == "gpt3.5":
        question_f = generate_question

    left_name = None
    left_model = None
    right_name = None
    right_model = None

    def _get_answer_function(_model_name: str, _device: str):
        if _model_name == "gpt_35":
            return MAP_MODELS[_model_name], None, None, generate_answer_gpt
        f = build_answer_func_from_server(_model_name)
        _name = MAP_MODELS[_model_name]
        if f is None:
            _tokenizer, _model = build_tokenizer_n_model(_name)
            func = partial(
                generate_answer_from_server,
                model=_model,
                tokenizer=_tokenizer,
                device=_device
            )
            return _name, _tokenizer, _model.to(_device), func
        else:
            return _name, None, None, f

    left_name, left_model, left_q = _get_answer_function(
        params.left_name, params.left_device
    )
    right_name, right_model, right_q = _get_answer_function(
        params.right_name, params.right_device
    )


    _info(f"{output_dir=}")
    _info(f"{left_name.name=}, {right_name.name=}")


    left_repr = left_name.name.replace("/", "_")
    right_repr = right_name.name.replace("/", "_")
    if right_repr == left_repr:
        right_repr += ".2"

    columns = [
        'Topic index',
        'Topic',
        'Question Number',
        'Question',
        f'{left_repr} answer',
        f'{right_repr} answer',
        'Distance',
        'Mean Distance',
        'Var Distance'
    ]

    data_collector = DataCollector(
        columns=columns,
        output_dir=output_dir,
    )

    new_obj = partial(
        objective_function_bayes, 
        disease_list = disease_list,
        num_topics = params.num_topics,
        num_sample = params.num_sample,
        num_repeats = params.num_repeats,
        left_model = left_model,
        left_q = left_q,
        right_model = right_model,
        right_q = right_q,
        dist_func=dist_func, 
        question_f = question_f,
        data_collector = data_collector,
        logger = logger,
        verbose = verbose,
        )
    
    space = [(-3, 3)]*num_dimensions

    if isinstance(params, ExpParamsEHP):    
        kappa = params.kappa
        acq_optimizer = params.acq_optimizer

    else:
        kappa = 1.96
        acq_optimizer = "auto"

    print(f"{kappa=}, {acq_optimizer=}")
    _info(f"{kappa=}, {acq_optimizer=}")

    result = gp_minimize(
        new_obj, 
        space, 
        n_calls=params.num_rounds, 
        random_state=0, 
        kappa=kappa, 
        acq_optimizer=acq_optimizer,
    )

    df = data_collector.save()

    if left_model is not None:
        del left_model
    if right_model is not None:
        del right_model

    gc.collect()
    torch.cuda.empty_cache()

    return df, result


def load_decoder_model(params:EmbedExpParams, state_file:str = None):
    if params.device != 'cpu':
        _info(f"warning; not honoring {params.device}; always to cpu")
    output_dir = os.path.join(params.data_dir, params.to_string(),"decoder")
    if state_file is None:
        state_file = find_latest_checkpoint(output_dir)
    else: 
        state_file = os.path.join(output_dir, state_file)
    if state_file is None:
        raise ValueError(f"no state file in {output_dir}")
    
    root = get_root_disease()
    vector_map = tree_to_vector_map(root)
    leaves_position = np.array([n.is_leaf for n in vector_map]).astype(bool)
    decoder = MLP(params.embed_dim, params.decoder_layer_sizes, np.sum(leaves_position))#.to(params.device)

    checkpoint = torch.load(state_file, map_location='cpu')
    decoder.load_state_dict(checkpoint['state_dict'])
    
    return decoder

def create_siamese_jobs(
        training_mode: Literal['end2end', 'sequential'] = 'end2end'
        ):
    params = []
    k_list = [2, 3, 4, 5, 6, 7, 8]
    use_sigmoid_list = [False, True]
    take_sqrt_list = [False, True]
    dim = 50
    epochs = 200
    hidden_list = [[350, 150], [200], [500, 250, 100]]

    gpu_map = {
        2: 'cuda:0',
        3: 'cuda:0',
        4: 'cuda:0',
        5: 'cuda:1',
        6: 'cuda:1',
        7: 'cuda:1',
        8: 'cuda:1',
    }
    data_dir_map = {
        'end2end': '/path/to/embed_test_network_n2n',
        'sequential': '/path/to/embed_test_network_seq'
    }
    data_dir = data_dir_map[training_mode]

    for num_topics, use_sigmoid, take_sqrt, hidden in product(
        k_list, use_sigmoid_list, take_sqrt_list, hidden_list
    ):
        param = EmbedExpParams(
            num_topics=num_topics,
            train_n=500000,
            train_cmp=2000000,
            test_n=500000,
            test_cmp=2000000,
            embed_dim=dim,
            layer_sizes=hidden,
            decoder_layer_sizes=hidden[::-1],
            train_batch_size=64,
            test_batch_size=512,
            lr=0.001,
            epochs=epochs,
            data_dir=data_dir,
            device=gpu_map[num_topics],
            use_sigmoid=use_sigmoid,
            take_sqrt=take_sqrt,
            training_mode=training_mode,
        )
        params.append(param)
    return params

def objective_function_emb(
    x: list[tuple],
    disease_list: list[str],
    num_topics: int,
    num_sample: int,
    num_repeats: int,
    left_model: torch.nn.Module,
    left_q: callable,
    right_model: torch.nn.Module,
    right_q: callable,
    decoder: torch.nn.Module,
    dist_func: callable,
    question_f: callable[[str, int], [str, list]] = None,
    data_collector: DataCollector = None,
    logger: Logger = None,
    verbose: bool = True,
):
    def _add_row(row: str):
        if data_collector:
            data_collector.add_row(row)

    def _log(msg: str):
        if logger:
            logger.info(msg)

    question_f = question_f or generate_question

    x_tensor = torch.from_numpy(np.array(x)).reshape(1, -1).float()
    decoder.eval()

    with torch.no_grad():
        decoded_x = decoder(x_tensor).cpu().numpy().reshape(-1)
        decoded_x_clipped = np.clip(decoded_x, 0, None)

    sampled_topics_list = sample_topics_embed(decoded_x_clipped, disease_list, num_topics, num_sample)

    distances = []

    for disease_tuple in sampled_topics_list:
        topics_repr = "***".join(disease_tuple)
        for question_number in tqdm(range(num_repeats), disable=not verbose):
            question, _ = question_f(disease_tuple)
            answer_left, answer_right = parallel_execute(
                left_q,
                right_q,
                question=question,
                left_model=left_model,
                right_model=right_model
            )
            distance = dist_func(answer_left, answer_right)
            distances.append(distance)

            new_row = [
                f'{topics_repr}',
                question_number,
                question,
                answer_left,
                answer_right,
                distance,
                np.nan,
                np.nan
            ]

            i = -1 if data_collector is None else data_collector.count
            _add_row(new_row)
            _log(f"{i=}, {question_number=}, {distance=}")

    if data_collector:
        data_collector.increment_and_save()

    average_distance = np.mean(distances)
    return -average_distance 

def run_exp_embed(
    params: ExpParams,
    params_e: EmbedExpParams,
    verbose: bool = True,
    *,
    group: str = 'default',
    new_behavior: bool = True
):
    """
    Run embedding-based Bayesian optimization experiment.

    Args:
        params (ExpParams): General experimental parameters.
        params_e (EmbedExpParams): Embedding-specific experimental parameters.
        verbose (bool, optional): Controls verbosity. Defaults to True.
        group (str, optional): Group identifier for logging. Defaults to 'default'.
        new_behavior (bool, optional): Adjust search space dynamically. Defaults to True.

    Returns:
        tuple: DataFrame of collected data and optimization result.
    """
    global logger
    filename = params.to_long_string() + "/NETWORK:" + params_e.to_string()
    logger = get_logger_with_filename(filename, group=group)

    assert params.num_topics == params_e.num_topics
    gc.collect()
    torch.cuda.empty_cache()

    assert params.optimizer == "embed"
    assert params.questioner in ["llama_13b", "gpt3.5"]

    output_dir = os.path.join(EXP_DIR, params.to_string(), params_e.to_string())
    os.makedirs(output_dir, exist_ok=True)

    dist_func = get_compute_distance_func()
    _info(f"{output_dir=}")

    _, disease_list = get_default_q_and_d_only_leaves()
    num_dimensions = params_e.embed_dim

    if params.questioner == "llama_13b":
        question_f = get_generate_question_func(
            params, model_s=SentenceTransformer('all-MiniLM-L6-v2')
        )
    elif params.questioner == "gpt3.5":
        question_f = generate_question

    left_name = None
    left_model = None
    right_name = None
    right_model = None 

    def _get_answer_function(model_name: str, device: str):
        if model_name == "gpt_35":
            return MAP_MODELS[model_name], None, None, generate_answer_gpt
        f = build_answer_func_from_server(model_name)
        name = MAP_MODELS[model_name]
        if f is None:
            tokenizer, model = build_tokenizer_n_model(name)
            func = partial(generate_answer_from_server, model=model, tokenizer=tokenizer, device=device)
            return name, tokenizer, model.to(device), func
        else:
            return name, None, None, f

    left_name, left_tokenizer, left_model, left_q = _get_answer_function(params.left_name, params.left_device)
    right_name, right_tokenizer, right_model, right_q = _get_answer_function(params.right_name, params.right_device)

    _info(f"{output_dir=}")
    _info(f"{left_name.name=}, {right_name.name=}")

    left_repr = left_name.name.replace("/", "_")
    right_repr = right_name.name.replace("/", "_")
    if right_repr == left_repr:
        right_repr += ".2"

    columns = [
        'Topic index', 'Topic', 'Question Number', 'Question',
        f'{left_repr} answer', f'{right_repr} answer',
        'Distance', 'Mean Distance', 'Var Distance'
    ]

    data_collector = DataCollector(
        columns=columns,
        output_dir = output_dir,
    )
    decoder = load_decoder_model(params_e)

    new_obj = partial(
        objective_function_emb, 
        disease_list = disease_list,
        num_topics = params.num_topics,
        num_sample = params.num_sample,
        num_repeats = params.num_repeats,
        left_model = left_model,
        left_q = left_q,
        right_model = right_model,
        right_q = right_q,
        decoder = decoder,
        dist_func=dist_func,
        question_f = question_f,
        data_collector = data_collector,
        logger = logger,
        verbose = verbose,
        )

    if new_behavior:
        if params_e.use_sigmoid:
            space = [(0, 1)] * num_dimensions
        else:
            space = [(-10, 10)] * num_dimensions
    else:
        space = [(-10, 10)] * num_dimensions

    print(space)
    _info(space)

    if isinstance(params, ExpParamsEHP):
        kappa = params.kappa
        acq_optimizer = params.acq_optimizer

    else:
        kappa = 1.96
        acq_optimizer = "auto"
        initial_points = 5
        n_iter = 25
        
    result = gp_minimize(
        new_obj,
        space,
        n_calls=params.num_rounds,
        random_state=0,
        kappa=kappa,
        acq_optimizer=acq_optimizer,
    )

    df = data_collector.save()

    if left_model is not None:
        del left_model
    if right_model is not None:
        del right_model

    gc.collect()
    torch.cuda.empty_cache()

    return df, result


def generate_answer_from_server(question: str, sparams: ServerParams):
    """
    Generate an answer for the given question using a remote server.
    """
    ever_failed = False
    for attempt in range(10):
        try:
            answer_text = _generate_answer_from_server(question, sparams)
        except Exception as e:
            print(e)
            _info(f"Error encountered: {question=}, {sparams=}, {e}")
            _info("Retrying...")
            ever_failed = True
            if attempt > 3:
                print("Sleeping for 3 seconds before retrying...")
                time.sleep(3)
            continue
        break

    if ever_failed:
        _info(f"Successful after retries; final response: {answer_text=}")

    return answer_text

def _generate_answer_from_server(question: str, sparams: ServerParams):
    """
    Helper function to send the request to the server and retrieve the generated answer.
    """
    prompt = f"How would you respond to the following question as {role}?\n\n{question}"

    url = f'http://127.0.0.1:{sparams.port}/generate/'
    payload = {'text': prompt}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        # Get the generated text from the response
        generated_text = response.json().get('generated_text', '')
    else:
        raise ValueError(f"Failed to generate text from {url}. Status code: {response.status_code} \n {response}")
    return generated_text

def build_answer_func_from_server(model_name: str):
    """
    Builds a function to generate answers from a remote server, based on model availability.
    """
    server_params = find_llm_server(model_name)
    if server_params is None:
        return None

    def _answer_q(question: str):
        return generate_answer_from_server(question, server_params)

    return _answer_q