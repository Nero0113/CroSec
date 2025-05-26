import argparse
import os
import sys

import torch

sys.path.append('../')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from trainer import Trainer
from utils import set_logging, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, default='Qwen2.5-Coder-0.5B-lora-sec_1')

    parser.add_argument('--model', type=str, default='/home/public_space/yanmeng/lidong/models/Qwen2.5-Coder-0.5B-Instruct')

    # training arguments
    parser.add_argument('--loss_type', type=str, default='CoSec')
    parser.add_argument('--num_train_epochs', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_acc_steps', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--sec_loss_weight', type=int, default=0)
    parser.add_argument('--kl_loss_weight', type=int, default=0)
    parser.add_argument('--exclude_neg', action='store_true', default=False)
    parser.add_argument('--no_weights', action='store_true', default=False)

    # lora arguments
    parser.add_argument('--lora', action='store_true', default=False, help='Toggle to use lora in training')
    parser.add_argument('--r', type=int, default=16, help='Lora hidden dimensions')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Alpha param, see Lora doc.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout in the learned extensions')

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--data_path', type=str, default='/home/public_space/yanmeng/lidong/code/one4all/data_train_val/train', help='训练数据路径')
    parser.add_argument('--val_path', type=str, default='/home/public_space/yanmeng/lidong/code/one4all/data_train_val/val', help='评估数据路径')
    parser.add_argument('--model_dir', type=str, default='../trained/')

    args = parser.parse_args()

    # adjust the naming to make sure that it is in the expected format for loading
    if args.lora and not args.output_name.startswith(f'{args.pretrain_name}-lora'):
        args.output_name = f'{args.pretrain_name}-lora-' + args.output_name

    args.output_dir = os.path.join(args.model_dir, args.output_name)
    if args.loss_type == 'CoSec':
        args.sec_loss_weight = 1
        args.kl_loss_weight = 1

    args.vul_type = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476', 'cwe-416', 'cwe-022', 'cwe-787', 'cwe-079', 'cwe-190']
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def main(args):
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_seed(args.seed)
    Trainer(args).run()

if __name__ == '__main__':
    args = get_args()
    main(args)