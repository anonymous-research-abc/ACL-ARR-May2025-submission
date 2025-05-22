from __future__ import annotations

import os
import attr 
import pickle
import pandas as pd
import numpy as np
from itertools import product
from typing import Optional
import llm.configure as config 
from llm.server import ExpParamsEHP, get_path_map


def get_experiment_params_and_paths(
    model_pairs: list,
    method: str,
    kappas: Optional[list] = None,
    acq_optimizers: Optional[list] = None,
    use_ehp: bool = True,
    left_device: str = "cuda:2",
    right_device: str = "cuda:3",
    num_rounds: Optional[int] = None,
    exp_dir: Optional[str] = None
):
    """
    Unified experiment parameter and path generation function
    """
    exp_dir = exp_dir or config.EXP_DIR
    embed_path, loaded_data = _get_loaded_data()

    if method == "random":
        assert not use_ehp and kappas is None and acq_optimizers is None
    else:
        if use_ehp:
            kappas = kappas or ([0.5] if method == "bayes" else [5])
            acq_optimizers = acq_optimizers or ["auto"]

    num_rounds = num_rounds or (100 if method in ['embed', 'bayes'] else 300)
    num_sample = 3 if method in ['embed', 'bayes'] else None

    all_params = []
    for config_set in product(
        kappas if use_ehp else [None],
        acq_optimizers if use_ehp else [None],
        model_pairs,
        [2, 3, 5]
    ):
        kappa, acq_optimizer, pair, num_topics = config_set
        params_e = loaded_data[num_topics]
        params_e = attr.evolve(params_e, data_dir=embed_path)

        common_params = {
            "left_name": pair[0],
            "right_name": pair[1],
            "num_topics": num_topics,
            "num_rounds": num_rounds,
            "num_repeats": 2,
            "questioner_device": "cpu",
            "left_device": left_device,
            "right_device": right_device,
            "optimizer": method,
            "questioner": "gpt3.5",
            "num_sample": num_sample,
        }

        if use_ehp:
            params = ExpParamsEHP(**common_params, kappa=kappa, acq_optimizer=acq_optimizer)
        else:
            params = ExpParams(**common_params)

        if method == 'embed':
            all_params.append([params, params_e])
        else:
            all_params.append(params)

    path_map = {}
    for i, params in enumerate(all_params):
        if isinstance(params, list):
            output_dir = os.path.join(exp_dir, params[0].to_string(), params[1].to_string())
        else:
            output_dir = os.path.join(exp_dir, params.to_string())

        files = glob.glob(output_dir + "/*.pqt")
        round_numbers = [
            int(os.path.basename(f).split("_")[-1].replace(".pqt", ""))
            for f in files if "data.pqt" not in f
        ]
        if not round_numbers:
            continue
        latest_round = min(max(round_numbers), 100 if method in ['bayes', 'embed'] else 300)
        filename = os.path.join(output_dir, f"data_round_{latest_round}.pqt")
        path_map[i] = filename

    return all_params, path_map

def _get_loaded_data():
    embed_path = '/your/generalized/path/embed_result'
    res_pkl = os.path.join(embed_path, 'res.pkl') 

    with open(res_pkl, 'rb') as file:
        loaded_data = pickle.load(file)
    return embed_path, loaded_data

def _normalize(df_res, dfb, normalize):
    """
    Standardize or normalize the avg_dist values based on baseline statistics.
    """
    if normalize:
        df_res['avg_dist'] = df_res['avg_dist'].div(df_res['metric'], axis=0)
    df_res = df_res.set_index(['left', 'right', 'Topic'])
    df_res[dfb.columns] = dfb
    df_res['avg_dist'] = (df_res['avg_dist'] - df_res['mean']).div(df_res['std'], axis=0)
    df_res = df_res[['metric', 'method', 'avg_dist']].reset_index()
    return df_res.set_index(['left', 'right', 'metric', "Topic", "method"]).unstack("Topic").unstack("method").droplevel(0, axis=1)

def aggregate_metric(
    params, 
    mp, 
    col: str = "Distance", 
    k_list: Optional[list] = None, 
    *, 
    standardize: bool =False, 
    normalize: bool=False, 
    dfb: pd.DataFrame=None
):
    """
    Metric aggregation function
    """
    if standardize or normalize:
        assert dfb is not None 
    if k_list is None:
        k_list = K_LIST
    entries = []
    for idx, k in product(range(len(params)), k_list):
        a_params = params[idx]
        if isinstance(a_params, list):
            a_params = a_params[0]
        if idx not in mp:
            continue
        df = pd.read_parquet(mp[idx])
        df_ = df[["Topic", col]].groupby("Topic").mean().sort_values(ascending=False, by=col).reset_index(drop=True).cumsum().add_suffix("_r")
        left, right = sorted([a_params.left_name, a_params.right_name]) 

        entries.append({"Topic": a_params.num_topics, "left": left, "right": right, "metric": k, "method": a_params.optimizer, "avg_dist": df_.iloc[k, 0]})
    df_res = pd.DataFrame(entries) 
    if standardize or normalize:
        return _normalize(df_res, dfb, normalize) 
    else:
        return df_res.set_index(['left', 'right', 'metric', "Topic", "method"]).unstack("Topic").unstack("method").droplevel(0, axis=1)

def cumulative_boosting(
    params, 
    mp_e, 
    mp_b, 
    k_list: Optional[list] = None,
    *, 
    standardize: bool =False, 
    normalize: bool=False, 
    dfb: pd.DataFrame=None
):
    if standardize or normalize:
        assert dfb is not None 
        
    assert len(mp_e) == len(mp_b) == len(params) 
    if k_list is None:
        k_list = K_LIST
    entries = []
    for idx, k in product(range(len(params)), k_list):
        a_params = params[idx]
        if isinstance(a_params, list):
            a_params = a_params[0]
        df = pd.read_parquet(mp_e[idx])
        df2 = pd.read_parquet(mp_b[idx]) 
        df2.index = df.index + df.index.get_level_values(0).max() + 1

        df = pd.concat([df, df2], axis=0)
        ddd = df[["Topic", "Distance"]].groupby("Topic").mean()
        dff = df.reset_index().set_index("Topic")
        dff[ddd.columns] = ddd
        df_ = dff.reset_index().set_index(["Topic index"])[["Distance"]].groupby("Topic index").max().sort_values(by="Distance", ascending=False).cumsum()
        left, right = sorted([a_params.left_name, a_params.right_name]) 
        entries.append({"Topic": a_params.num_topics, "left": left, "right": right, "metric": k, "method": "boosting", "avg_dist": df_.iloc[k, 0]})
    df_res = pd.DataFrame(entries)
    if standardize or normalize:
        return _normalize(df_res, dfb, normalize) 
    else:
        return df_res.set_index(['left', 'right', 'metric', "Topic", "method"]).unstack("Topic").unstack("method").droplevel(0, axis=1)

def get_mean_std(model_pairs, left_device=None, right_device=None, exp_dir=None):
    method = 'random'
    use_ehp = False
    params, mp = get_path_map(model_pairs, method, use_ehp=use_ehp, left_device=left_device, right_device=right_device, exp_dir=exp_dir)
    return _get_mean_std(params, mp)

def _get_mean_std(all_params, mp):
    entries = []
    for idx in range(len(all_params)):
        params = all_params[idx]
        if idx not in mp:
            continue
        df = pd.read_parquet(mp[idx])
        left, right = sorted([params.left_name, params.right_name]) 
        entries.append({"Topic": params.num_topics, "left": left, "right": right, "mean": df['Distance'].mean(), 'std': df['Distance'].std()})
    dfb = pd.DataFrame(entries)
    return dfb.set_index(['left', 'right', 'Topic'])