import ast
import logging
import os
import random
import subprocess
import sys

import numpy as np
import peft
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from Ensemble_model import EnsembleModel
from constants import PRETRAINED_MODELS, SECURITY_MODELS
from try_plugin_like.my_model import myQwenForCausalLM

logger = logging.getLogger()

def set_devices(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.logger.info('Device: %s, n_gpu: %s', device, args.n_gpu)


def set_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_logging(args, log_file):
    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    args.logger = logger

def load_model(model_name=None, args=None, ref=False):
    # Check if model is available in PRETRAINED_MODELS
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Model {model_name} not found in PRETRAINED_MODELS")

    # Load models based on target type and ref flag
    if args.target == "base_model" or ref:
        model = AutoModelForCausalLM.from_pretrained(
            PRETRAINED_MODELS[model_name], device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODELS[model_name])

        # Ensure the token embeddings are resized at the end
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
    elif args.target == "sec_model":
        model = AutoModelForCausalLM.from_pretrained(
            SECURITY_MODELS[model_name], device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(SECURITY_MODELS[model_name])

        # Ensure the token embeddings are resized at the end
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

    elif args.target == "one4all":
        trg_model = AutoModelForCausalLM.from_pretrained(args.trg_model, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.trg_model)
        trg_model.resize_token_embeddings(len(tokenizer))

        src_model = AutoModelForCausalLM.from_pretrained(args.src_model, torch_dtype=torch.bfloat16)
        src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
        src_model.resize_token_embeddings(len(src_tokenizer))
        src_model = peft.PeftModel.from_pretrained(src_model, args.lora)

        model = EnsembleModel(
            src_model=src_model,
            trg_model=trg_model,
            src_tokenizer=src_tokenizer,
            trg_tokenizer=tokenizer,
            sparse_matrix_path=args.sparse_matrix_path,
            token_map=args.token_map,
        )

    return tokenizer, model



def try_parse(code, lang):
    if lang == 'py':
        try:
            ast.parse(code)
            return 0
        except:
            return 1
    elif lang == 'c':
        cmd = 'gcc -c -x c -'
        process = subprocess.run(cmd, shell=True, timeout=5, input=code.encode(), stderr=subprocess.DEVNULL)
        if process.returncode == 0:
            return 0
        else:
            return 1
    else:
        raise NotImplementedError()
