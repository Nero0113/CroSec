from transformers import AutoTokenizer
import sys
# sys.path.append('/home/public_space/yanmeng/zhangjingrui/projects/OFA/')
import torch
from tqdm import tqdm
import json
import re
import math
import Levenshtein
import matplotlib.pyplot as plt
import argparse
import os
from utils import coder_model_name

# Define the space symbol used by the tokenizer
space_symbol = "Ġ"  # or "▁", depending on the tokenizer

def get_args():
    """
    Parse command-line arguments.
    Returns:
        args: Parsed arguments.
    """
    # Find trg_model to sec_model token_map
    parser = argparse.ArgumentParser()
    parser.add_argument('--sec_model_path', type=str, default= '/home/yanmeng/zhangjingrui/models/Qwen2.5-Coder-0.5B-Instruct') 
    parser.add_argument('--trg_model_path', type=str, default= '/home/yanmeng/zhangjingrui/models/deepseek-coder-6.7b-base')  
    parser.add_argument('--base_dir', type=str, default='../token_map_files')  # Output directory for token maps
    args = parser.parse_args()

    return args

def is_all_space(string):
    """
    Check if a string consists entirely of space symbols.
    Args:
        string (str): The string to check.
    Returns:
        bool: True if the string is all spaces, False otherwise.
    """
    return re.match(f'^{space_symbol}*$', string) is not None


def token_maps(sec_model_path, trg_model_path, base_dir):
    """
    Generate token maps between two models.
    Args:
        sec_model_path (str): Path to the sec model.
        trg_model_path (str): Path to the source model.
        base_dir (str): Output directory for token maps.
    """
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }

    # Determine model names
    sec_model_path , _ = coder_model_name(sec_model_path)
    trg_model_name , _ = coder_model_name(trg_model_path)
    if sec_model_path == trg_model_name:
        raise NotImplementedError("Model name must be different!")

    # Load tokenizers for both models
    tokenizer_model1 = AutoTokenizer.from_pretrained(sec_model_path, **config_kwargs)
    tokenizer_model2 = AutoTokenizer.from_pretrained(trg_model_path, **config_kwargs)

    # Get and sort vocabularies for both models
    vocab_model1 = tokenizer_model1.get_vocab()
    vocab_model1 = sorted(vocab_model1.items(), key=lambda x: x[1])
    model1 = [x[0] for x in vocab_model1]

    vocab_model2 = tokenizer_model2.get_vocab()
    vocab_model2 = sorted(vocab_model2.items(), key=lambda x: x[1])
    model2 = [x[0] for x in vocab_model2]

    # Get all special tokens from the source model
    all_special_tokens = tokenizer_model2.all_special_tokens

    # Create output directory if it doesn't exist
    directory = f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize lists and dictionaries for token mapping
    list_distance = []
    model22model1_id = []
    model22model1_match = []
    model22model1_token = {}
    cnt = 0


    # Iterate over the vocabulary of the source model
    for i in tqdm(range(len(vocab_model2))):
        item = vocab_model2[i]
        token = item[0]

        # Handle special tokens
        if token in all_special_tokens:
            # Special tokens like <s>, </s>, <unk> are mapped to themselves
            model22model1_id.append(item[1])
            list_distance.append(0)
            model22model1_token[token] = token
            model22model1_match.append(1)
        elif token.startswith('<') and token.endswith('>') and len(token) > 2:
            # Handle hex tokens
            model22model1_id.append(item[1])
            list_distance.append(0)
            model22model1_token[token] = token
            model22model1_match.append(1)
        else:
            # For regular tokens, find the closest match in the target model's vocabulary
            candidate = []
            match_flag = 0
            for related_tokens in vocab_model1:
                if token == related_tokens[0]:
                    # Exact match found
                    candidate = [related_tokens]
                    match_flag = 1
                    break
                if (token.startswith(related_tokens[0]) or related_tokens[0].startswith(token)):
                    # Partial match found
                    if related_tokens[0] == space_symbol and is_all_space(token) == False:
                        continue
                    candidate.append(related_tokens)

            # Find the closest token based on Levenshtein distance
            min_distance = float('inf')
            closest_token = None
            closest_id = -1
            for candidate_item in candidate:
                distance = Levenshtein.distance(token, candidate_item[0])
                if distance < min_distance:
                    min_distance = distance
                    closest_token = candidate_item[0]
                    closest_id = candidate_item[1]

            # Store the mapping results
            model22model1_id.append(closest_id)
            model22model1_token[token] = closest_token
            model22model1_match.append(match_flag)
            if closest_token is None:
                cnt += 1
            else:
                list_distance.append(min_distance)

        # Save intermediate results every 1000 tokens
        if i % 1000 == 0:
            with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_id.json", "w") as f:
                json.dump(model22model1_id, f, indent=4)
            with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_token.json", "w") as f:
                json.dump(model22model1_token, f, indent=4)
            with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_match.json", "w") as f:
                json.dump(model22model1_match, f, indent=4)

    # Print statistics
    print(cnt)
    print(sum(list_distance) / len(list_distance))

    # Plot and save the histogram of distances
    plt.hist(list_distance, bins=100, align='left')
    plt.savefig(f"{directory}/histogram_distance.png")

    # Save final results
    with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_id.json", "w") as f:
        json.dump(model22model1_id, f, indent=4)
    with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_token.json", "w") as f:
        json.dump(model22model1_token, f, indent=4)
    with open(f"{base_dir}/{sec_model_path}_2_{trg_model_name}/token_maps/one2one_{trg_model_name}2{sec_model_path}_match.json", "w") as f:
        json.dump(model22model1_match, f, indent=4)

if __name__ == '__main__':
    # Parse arguments and run the token mapping function
    args = get_args()
    # Find model2 to model1 token_map
    token_maps(args.sec_model_path, args.trg_model_path, args.base_dir)