import sys

from transformers import set_seed

sys.path.append('../')
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.human_eval.problem_yaml import Problem
from utils import load_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_name', type=str, default='StarCoder_qwen_lora_w03')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--eval_type', type=str, default='human_eval')
    parser.add_argument('--target', type=str, choices=['base_model','sec_model', 'one4all'], default='one4all')
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_new_len', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_samples_per_gen', type=int, default=25)
    parser.add_argument('--ensemble_weight', type=float, default=0.3)

    parser.add_argument('--trg_model', type=str,
                        default='../models/starcoder2-7b')
    parser.add_argument('--sec_model', type=str,
                        default='../models/Qwen2.5-Coder-0.5B-Instruct')
    parser.add_argument('--lora', type=str,
                        default='../trained/Qwen2.5-Coder-0.5b/checkpoint-last')

    parser.add_argument('--sparse_matrix_path', type=str,
                        default='../map_files/Qwen_to_DeepSeek_sim_matrix.npz')
    parser.add_argument('--token_map', type=str,
                        default='../token_map/main2assist/DeepSeek2Qwen/one2one_DeepSeek2Qwen_id.json')

    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.experiments_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args

def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion

def main(args):
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    tokenizer, model = load_model(model_name=args.model_name, args=args)


    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        prompt = problem.prompt.strip()
        inputs = tokenizer(prompt.strip(), return_tensors='pt').to(model.trg_model.device)
        seed = args.seed
        for i in range(args.num_samples // args.num_samples_per_gen):
            set_seed(seed + i)
            with torch.no_grad():

                samples = model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=args.num_samples_per_gen,
                    temperature=args.temp,
                    max_new_tokens=args.max_new_len,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            for sample in samples.tolist():
                # print(tokenizer.decode(sample))
                # print('*'*150)
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion)
                completion = trim_code(completion, problem.stop_tokens)
                # print(completion)
                # print('='*150)
                problem.completions.append(completion)
        with problem_yaml_path.open('w') as f:
            f.write(Problem.dump(problem))


if __name__ == '__main__':
    args = get_args()
    main(args)