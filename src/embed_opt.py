from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm.education import _EDU_FILE
from llm.medical import _DISEASE_FILE
import torch
import attr 
import os
from typing import Callable

import glob
import re
from llm.util import build_logger

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.is_leaf = False
        self.indicator = 0

    def add_child(self, child):
        self.children.append(child)


def reset_tree_indicators(root):
    stack = [root]
    while stack:
        node = stack.pop()
        node.indicator = 0
        stack.extend(node.children)


def set_leaves_subset(root, subset):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            if node in subset:
                node.indicator = 1
            else:
                node.indicator = 0
        stack.extend(node.children)


def encode_tree(root):
    if root.is_leaf:
        return root.indicator
    else:
        count = 0
        for child in root.children:
            count += encode_tree(child)
        root.indicator = count
        return count


def tree_to_vector(root):
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.indicator)
        stack.extend(reversed(node.children))
    return result


def tree_to_vector_map(root):
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(reversed(node.children))
    return result


def pre_order_traversal(node):
    result = [node]
    for child in node.children:
        result.extend(pre_order_traversal(child))
    return result


def create_random_subsets(root, num_subsets, subset_size):
    all_leaves = [node for node in pre_order_traversal(root) if node.is_leaf]
    subsets = []
    for _ in range(num_subsets):
        selected_leaves = random.sample(all_leaves, subset_size)
        subsets.append(selected_leaves)
    return subsets


def vectorize_subsets(root, subsets):
    vectors = []
    for subset in subsets:
        reset_tree_indicators(root)
        set_leaves_subset(root, subset)
        encode_tree(root)
        vectors.append(tree_to_vector(root))
    return vectors


def compute_l1_distances(vectors):
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = torch.tensor([torch.sum(torch.abs(torch.tensor(vectors[i]) - torch.tensor(vectors[j]))).item()], dtype=torch.float)
            distances.append((vectors[i], vectors[j], dist))
    return distances


def create_sample_tree():
    root = TreeNode("Root")
    child1 = TreeNode("Child1")
    child2 = TreeNode("Child2")
    root.add_child(child1)
    root.add_child(child2)
    leaf1 = TreeNode("Leaf1")
    leaf2 = TreeNode("Leaf2")
    leaf3 = TreeNode("Leaf3")
    child1.add_child(leaf1)
    child2.add_child(leaf2)
    child2.add_child(leaf3)
    leaf1.is_leaf = True
    leaf2.is_leaf = True
    leaf3.is_leaf = True
    return root


class TreeDataset(Dataset):
    def __init__(self, idx_pairs, data):
        self.data = data
        self.idx_pairs = idx_pairs

    def __len__(self):
        return len(self.idx_pairs)

    def __getitem__(self, idx):
        l, r = self.idx_pairs[idx]
        x1, x2 = self.data[l], self.data[r]
        y = np.sum(np.abs(x1 - x2))
        return x1, x2, y


def construct_tree(df):
    # Create the root TreeNode
    root = TreeNode("Root")

    # Dictionary to keep track of created chapters
    chapters = {}

    # Iterate over the DataFrame and construct the tree
    for _, row in df.iterrows():
        chapter_name = row['Chapter']
        description = row['Description']  # Use description as the name of the leaf nodes

        # Skip rows where 'Block' is NaN (i.e., description is None)
        if pd.isna(row['Block']):
            continue

        # Check if chapter TreeNode exists, if not create it and add to root
        if chapter_name not in chapters:
            chapter_node = TreeNode(chapter_name)
            chapters[chapter_name] = chapter_node
            root.add_child(chapter_node)

        # Create a leaf TreeNode with the description as its name and add it to its chapter
        leaf_node = TreeNode(description)
        leaf_node.is_leaf = True
        chapters[chapter_name].add_child(leaf_node)

    return root


def construct_edu_tree(df):
    # Create the root TreeNode
    root = TreeNode("Root")

    # Dictionaries to keep track of created grades and topics
    grades = {}
    topics = {}

    # Iterate over the DataFrame and construct the tree
    for (grade, topic), group in df.groupby(['grade', 'topic']):
        # Check if grade TreeNode exists, if not create it and add to root
        if grade not in grades:
            grade_node = TreeNode(grade)
            grades[grade] = grade_node
            root.add_child(grade_node)

        # Check if topic TreeNode exists, if not create it and add to grade
        topic_key = (grade, topic)
        if topic_key not in topics:
            topic_node = TreeNode(topic)
            topics[topic_key] = topic_node
            grades[grade].add_child(topic_node)

        # Add merged_code nodes with descriptions as leaf nodes
        for _, row in group.iterrows():
            merged_code = row['merged_code']
            description = row['full_description']

            # Create a merged_code TreeNode and add it to the topic
            leaf_node = TreeNode(f"{merged_code}:{description}")
            leaf_node.is_leaf = True

            topics[topic_key].add_child(leaf_node)

    return root


class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes, embedding_dim, add_sigmoid=False):
        super(MLP, self).__init__()
        layers = []
        previous_dim = input_dim
        for layer_size in layer_sizes:
            layers.append(nn.Linear(previous_dim, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            previous_dim = layer_size

        layers.append(nn.Linear(previous_dim, embedding_dim))
        if add_sigmoid:
            print("added sigmoid")
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, layer_sizes, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.mlp = MLP(input_dim, layer_sizes, embedding_dim)

    def forward(self, input1, input2):
        output1 = self.mlp(input1)
        output2 = self.mlp(input2)
        return output1, output2


def lst2str(ll):
    return "-".join([str(e) for e in ll])


@attr.define(kw_only=True)
class EmbedExpParams:
    # scenario  
    num_topics:int

    # data processing
    train_n:int
    train_cmp:int
    test_n:int
    test_cmp:int

    # network 
    embed_dim:int
    layer_sizes:list
    decoder_layer_sizes:list

    # optimizer 
    train_batch_size:int
    test_batch_size:int
    lr:float
    epochs:int

    #output path
    data_dir:str

    #environment
    device:str = "cpu"

    # auxilary 
    use_sigmoid: bool = False
    take_sqrt: bool = True

    def to_string(self):
        list_members = ["layer_sizes", "decoder_layer_sizes"]
        excluded_members = ["data_dir"]
        s1 = "_".join([str(getattr(self, f.name)) for f in attr.fields(self.__class__) if f.name not in list_members + excluded_members])
        s2 = "_".join([lst2str(getattr(self,e)) for e in list_members])
        return s1+"_"+s2


def get_root_disease():
    df = pd.read_csv(_DISEASE_FILE)
    return construct_tree(df)

def get_root_education():
    df = pd.read_parquet(_EDU_FILE)
    return construct_edu_tree(df)


def eval_embedding(loader, model, device, verbose = True):
    model.eval()
    predys = []
    ys = []
    with torch.no_grad():
        for input1, input2, distance in tqdm(loader, disable = not verbose):
            input1, input2, distance = input1.to(device), input2.to(device), distance.to(device)
            output1, output2 = model(input1, input2)
            predy = ((output1 - output2)**2).mean(axis=1).reshape(-1)
            y = distance.reshape(-1)
            predys.append(predy.to('cpu'))
            ys.append(y.to('cpu'))

    pyy=torch.cat(predys).detach().cpu().numpy()
    yy=torch.cat(ys).detach().cpu().numpy()

    return np.corrcoef(pyy,yy)[0,1]

def train_model(params: EmbedExpParams):
    if params.training_mode == 'end2end':
        end2end_train(params)
    elif params.training_mode == 'sequential':
        sequential_train(params)
    else:
        raise ValueError(f"Unknown training mode: {params.training_mode}")

def train_siamese(
    params:EmbedExpParams, 
    get_root:Callable | None = None,
    verbose:bool = True
):
    
    output_dir = os.path.join(params.data_dir, params.to_string())
    logger_dir = os.path.join(output_dir, 'log')
    logger = build_logger(logger_dir)
    
    if get_root is None:
        get_root = get_root_disease

    range_n = range(params.train_n)
    idx_pairs = np.array([sorted(random.sample(range_n, 2)) for _ in range(params.train_cmp)])
    root = get_root_disease()
    subsets = create_random_subsets(root, params.train_n, params.num_topics)  # 100 random subsets of 2 leaves each
    vectors = vectorize_subsets(root, subsets)
    np_vectors = np.array(vectors).astype(np.float32)
    td = TreeDataset(idx_pairs, np_vectors)
    train_loader = DataLoader(td, batch_size = params.train_batch_size, shuffle=True)

    subsets_test = create_random_subsets(root, params.test_n, params.num_topics)  # 100 random subsets of 2 leaves each
    vectors_test = vectorize_subsets(root, subsets_test)
    range_n_test = range(params.test_n)
    idx_pairs_test = np.array([sorted(random.sample(range_n_test, 2)) for _ in range(params.test_cmp)])
    np_vectors_test = np.array(vectors_test).astype(np.float32)
    td_test = TreeDataset(idx_pairs_test, np_vectors_test)
    test_loader = DataLoader(td_test, batch_size=params.test_batch_size, shuffle=False)

    device = params.device
    siamese_model = SiameseNetwork(
        input_dim=np_vectors.shape[1],
        layer_sizes = params.layer_sizes,
        embedding_dim = params.embed_dim,
    ).to(device)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{output_dir=}")

    optimizer_siamese = optim.Adam(siamese_model.parameters(), lr=params.lr)
    siamese_model.train()
    criterion = nn.MSELoss()
    eval_results = []

    for epoch in range(params.epochs):
        total_loss = 0
        for input1, input2, distance in tqdm(train_loader, disable = not verbose):
            input1, input2, distance = input1.to(device), input2.to(device), distance.to(device)
            optimizer_siamese.zero_grad()
            output1, output2 = siamese_model(input1, input2)
            loss = criterion(((output1 - output2)**2).mean(axis=1).reshape(-1), distance.reshape(-1))
            loss.backward()
            optimizer_siamese.step()
            total_loss += loss.item()
        logger.info(f'Epoch {epoch+1}/{params.epochs}, Loss: {total_loss/len(train_loader)}')
        state = {
            'epoch': epoch + 1,
            'state_dict': siamese_model.state_dict(),
            'optimizer': optimizer_siamese.state_dict(),
        }
        filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        logger.info(f'saving state to {filename}')
        torch.save(state, filename)
        if epoch%5 == 0:
            c = eval_embedding(train_loader, siamese_model, params.device, verbose = verbose)
            logger.info(f"{epoch=}, correlation coefficient={c}#")
            ct = eval_embedding(test_loader, siamese_model, params.device, verbose = verbose)
            logger.info(f"{epoch=}, correlation coefficient={ct}#")
            eval_results.append({"epoch":epoch, "train corr":c, "test corr":ct})
            df = pd.DataFrame(eval_results)
            filename = os.path.join(output_dir, f'corr_{epoch+1}.pqt')
            df.to_parquet(filename)
            logger.info(f"correlation saving to {filename}")
    df_res = pd.DataFrame(eval_results)
    return siamese_model, df_res


def eval_decoder(loader, encoder, decoder, device, verbose = True):
    encoder.eval()
    decoder.eval()
    predys = []
    ys = []

    root = get_root_disease()
    vector_map = tree_to_vector_map(root)
    leaves_position = np.array([n.is_leaf for n in vector_map]).astype(bool)
    tensor_leaves_position = torch.from_numpy(leaves_position)
    tl = tensor_leaves_position.to(device)

    with torch.no_grad():
        for input1, input2, _ in tqdm(loader, disable = not verbose):
            input1, input2 = input1.to(device), input2.to(device)
            embedding1, embedding2 = encoder(input1, input2)
            reconstructed1 = decoder(embedding1)
            reconstructed2 = decoder(embedding2)
            predys.append(reconstructed1.reshape(-1).to('cpu'))
            ys.append(input1[:,tl].reshape(-1).to('cpu'))
            predys.append(reconstructed2.reshape(-1).to('cpu'))
            ys.append(input2[:,tl].reshape(-1).to('cpu'))

    pyy=torch.cat(predys).detach().cpu().numpy()
    yy=torch.cat(ys).detach().cpu().numpy()

    return np.corrcoef(pyy,yy)[0,1]


def find_latest_checkpoint(directory):
    pattern = os.path.join(directory, "checkpoint_epoch_*.pth")
    files = glob.glob(pattern)

    if not files:
        return None

    def get_epoch_number(filename):
        match = re.search(r"checkpoint_epoch_(\d+).pth", filename)
        if match:
            return int(match.group(1))
        return -1
    latest_file = max(files, key=get_epoch_number)
    return latest_file


def load_siamese_model(params:EmbedExpParams, state_file:str = None):
    output_dir = os.path.join(params.data_dir, params.to_string())
    if state_file is None:
        state_file = find_latest_checkpoint(output_dir)
    else:
        state_file = os.path.join(output_dir, state_file)
    if state_file is None:
        raise ValueError(f"no state file in {output_dir}")

    root = get_root_disease()
    subsets = create_random_subsets(root, 1, params.num_topics)
    vectors = vectorize_subsets(root, subsets)
    np_vectors = np.array(vectors).astype(np.float32)
    device = params.device
    input_dim=np_vectors.shape[1]

    siamese_model = SiameseNetwork(
        input_dim=input_dim,
        layer_sizes = params.layer_sizes,
        embedding_dim = params.embed_dim,
    )
    checkpoint = torch.load(state_file, map_location='cpu')
    siamese_model.load_state_dict(checkpoint['state_dict'])
    siamese_model = siamese_model.to(device)
    return siamese_model


def train_decoder(params:EmbedExpParams, verbose:bool = True):
    output_dir = os.path.join(params.data_dir, params.to_string(),"decoder")
    logger_dir = os.path.join(output_dir, 'log')
    logger = build_logger(logger_dir)
    

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{output_dir=}")

    range_n = range(params.train_n)
    idx_pairs = np.array([sorted(random.sample(range_n, 2)) for _ in range(params.train_cmp)])
    root = get_root_disease()
    subsets = create_random_subsets(root, params.train_n, params.num_topics) 
    vectors = vectorize_subsets(root, subsets)
    np_vectors = np.array(vectors).astype(np.float32)
    td = TreeDataset(idx_pairs, np_vectors)
    train_loader = DataLoader(td, batch_size = params.train_batch_size, shuffle=True)

    #related to test data
    subsets_test = create_random_subsets(root, params.test_n, params.num_topics) 
    vectors_test = vectorize_subsets(root, subsets_test)
    range_n_test = range(params.test_n)
    idx_pairs_test = np.array([sorted(random.sample(range_n_test, 2)) for _ in range(params.test_cmp)])
    np_vectors_test = np.array(vectors_test).astype(np.float32)
    td_test = TreeDataset(idx_pairs_test, np_vectors_test)
    test_loader = DataLoader(td_test, batch_size=params.test_batch_size, shuffle=False)

    encoder = load_siamese_model(params)
    root = get_root_disease()
    vector_map = tree_to_vector_map(root)
    leaves_position = np.array([n.is_leaf for n in vector_map]).astype(bool)
    decoder = MLP(params.embed_dim, params.decoder_layer_sizes, np.sum(leaves_position)).to(params.device)

    optimizer_decoder = optim.Adam(decoder.parameters(), lr=params.lr)
    tensor_leaves_position = torch.from_numpy(leaves_position)
    tl = tensor_leaves_position.to(params.device)

    encoder.eval()
    decoder.train()
    criterion = nn.MSELoss()
    eval_results = []

    for epoch in range(params.epochs):
        total_loss = 0
        for input1, input2, _ in train_loader:
            input1, input2 = input1.to(params.device), input2.to(params.device)
            optimizer_decoder.zero_grad()
            with torch.no_grad():
                embedding1, embedding2 = encoder(input1, input2)
            reconstructed1 = decoder(embedding1)
            reconstructed2 = decoder(embedding2)
            loss1 = criterion(reconstructed1, input1[:,tl])
            loss2 = criterion(reconstructed2, input2[:,tl])
            loss = loss1 + loss2
            loss.backward()
            optimizer_decoder.step()
            total_loss += loss.item()
        logger.info(f'Epoch {epoch+1}/{params.epochs}, Loss: {total_loss/len(train_loader)}')
        state = {
            'epoch': epoch + 1,
            'state_dict': decoder.state_dict(),
            'optimizer': optimizer_decoder.state_dict(),
        }
        filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        logger.info(f'saving state to {filename}')
        torch.save(state, filename)

        if epoch%5 == 0:
            c = eval_decoder(train_loader, encoder, decoder, params.device, verbose = verbose)
            logger.info(f"{epoch=}, train correlation coefficient={c}#")
            ct = eval_decoder(test_loader, encoder, decoder, params.device, verbose = verbose)
            logger.info(f"{epoch=}, test correlation coefficient={ct}#")
            eval_results.append({"epoch":epoch, "train corr":c, "test corr":ct})
            df = pd.DataFrame(eval_results)
            filename = os.path.join(output_dir, f'corr_{epoch+1}.pqt')
            df.to_parquet(filename)
            logger.info(f"correlation saving to {filename}")
    df_res = pd.DataFrame(eval_results)

    return decoder, df_res

def sequential_train(params: EmbedExpParams, verbose: bool = True):
    train_siamese(params)
    train_decoder(params)


def end2end_train(params: EmbedExpParams, verbose: bool = True):
    output_dir = os.path.join(params.data_dir, params.to_string())
    os.makedirs(output_dir, exist_ok=True)
    
    logger_dir = os.path.join(output_dir, 'log')
    logger = build_logger(logger_dir)
    
    range_n = range(params.train_n)
    idx_pairs = np.array([sorted(random.sample(range_n, 2)) for _ in range(params.train_cmp)])
    root = get_root_disease()
    subsets = create_random_subsets(root, params.train_n, params.num_topics)
    vectors = vectorize_subsets(root, subsets)
    np_vectors = np.array(vectors).astype(np.float32)
    logger.info("creating dataset done")
    td = TreeDataset(idx_pairs, np_vectors)
    train_loader = DataLoader(td, batch_size = params.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # related to test data. 
    subsets_test = create_random_subsets(root, params.test_n, params.num_topics)
    vectors_test = vectorize_subsets(root, subsets_test)
    range_n_test = range(params.test_n)
    idx_pairs_test = np.array([sorted(random.sample(range_n_test, 2)) for _ in range(params.test_cmp)])
    np_vectors_test = np.array(vectors_test).astype(np.float32)
    td_test = TreeDataset(idx_pairs_test, np_vectors_test)
    test_loader = DataLoader(td_test, batch_size=params.test_batch_size, shuffle=False)


    device = params.device
    siamese_model = SiameseNetwork(
        input_dim=np_vectors.shape[1],
        layer_sizes = params.layer_sizes,
        embedding_dim = params.embed_dim,
    ).to(device)
    logger.info(f"{output_dir=}")

    root = get_root_disease()
    vector_map = tree_to_vector_map(root)
    leaves_position = np.array([n.is_leaf for n in vector_map]).astype(bool)
    decoder = MLP(params.embed_dim, params.decoder_layer_sizes, np.sum(leaves_position), add_sigmoid=params.use_sigmoid).to(params.device)

    output_dir_decoder = os.path.join(params.data_dir, params.to_string(),"decoder")
    os.makedirs(output_dir_decoder, exist_ok=True)
    logger.info(f"{output_dir_decoder=}")


    optimizer = optim.Adam(list(siamese_model.parameters()) + list(decoder.parameters()), lr=params.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=.7, threshold=5e-4,patience=10)
    siamese_model.train()
    decoder.train()
    criterion_encoder = nn.MSELoss()
    eval_results_encoder = []

    criterion_decoder = nn.MSELoss()
    eval_results_decoder = []

    tensor_leaves_position = torch.from_numpy(leaves_position)
    tl = tensor_leaves_position.to(params.device)

    for epoch in range(params.epochs):
        total_loss_encoder = 0
        total_loss_decoder = 0
        for input1, input2, distance in tqdm(train_loader, disable = not verbose):
            input1, input2, distance = input1.to(device), input2.to(device), distance.to(device)
            optimizer.zero_grad()
            output1, output2 = siamese_model(input1, input2)
            if params.take_sqrt:
                loss_encoder = criterion_encoder(torch.sqrt(((output1 - output2)**2).mean(axis=1)).reshape(-1), distance.reshape(-1))
            else:
                loss_encoder = criterion_encoder(((output1 - output2)**2).mean(axis=1).reshape(-1), distance.reshape(-1))

            reconstructed1 = decoder(output1)
            reconstructed2 = decoder(output2)
            loss1 = criterion_decoder(reconstructed1, input1[:,tl])
            loss2 = criterion_decoder(reconstructed2, input2[:,tl])

            loss = 0.1 * loss_encoder + (loss1 + loss2) * 1.0
            loss.backward()
            optimizer.step()
            total_loss_encoder += loss_encoder.item()
            total_loss_decoder += loss1.item() + loss2.item()

        logger.info(f'Epoch {epoch+1}/{params.epochs}, Loss encoder: {total_loss_encoder/len(train_loader)}, Loss decoder: {total_loss_decoder/len(train_loader)}')


        # now save encoder. 
        state = {
            'epoch': epoch + 1,
            'state_dict': siamese_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        logger.info(f'saving state to {filename}')
        torch.save(state, filename)

        # then save decoder
        state = {
            'epoch': epoch + 1,
            'state_dict': decoder.state_dict(),
        }
        filename = os.path.join(output_dir_decoder, f'checkpoint_epoch_{epoch+1}.pth')
        logger.info(f'saving state to {filename}')
        torch.save(state, filename)

        if epoch%5 == 0:
            c = eval_embedding(train_loader, siamese_model, params.device, verbose = verbose)
            logger.info(f"{epoch=}, encoder train correlation coefficient={c}#")
            ct = eval_embedding(test_loader, siamese_model, params.device, verbose = verbose)
            logger.info(f"{epoch=}, encoder test correlation coefficient={ct}#")
            eval_results_encoder.append({"epoch":epoch, "train corr":c, "test corr":ct})
            df = pd.DataFrame(eval_results_encoder)
            filename = os.path.join(output_dir, f'corr_{epoch+1}.pqt')
            df.to_parquet(filename)
            logger.info(f"correlation saving to {filename}")


            c = eval_decoder(train_loader, siamese_model, decoder, params.device, verbose = verbose)
            logger.info(f"{epoch=}, decoder train correlation coefficient={c}#")
            ct = eval_decoder(test_loader, siamese_model, decoder, params.device, verbose = verbose)
            logger.info(f"{epoch=}, decoder test correlation coefficient={ct}#")
            eval_results_decoder.append({"epoch":epoch, "train corr":c, "test corr":ct})
            df = pd.DataFrame(eval_results_decoder)
            filename = os.path.join(output_dir_decoder, f'corr_{epoch+1}.pqt')
            df.to_parquet(filename)
            logger.info(f"correlation saving to {filename}")
            scheduler.step(c)



    df_res_encoder = pd.DataFrame(eval_results_encoder)
    df_res_decoder = pd.DataFrame(eval_results_decoder)

    return {
        'encoder': siamese_model,
        'decoder': decoder,
        'stat_encoder': df_res_encoder,
        'stat_decoder': df_res_decoder
    }