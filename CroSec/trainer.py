import os
from collections import OrderedDict

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, \
    get_cosine_schedule_with_warmup
import torch.nn.functional as F
from train_delta.dataset import CodeDataset
from train_delta.timer import Timer
from utils import load_model, set_seed


class LossDict:
    def __init__(self, keys):
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps
                p.append(f'{k}: {round(s, 6)}')
        return ', '.join(p)

    def clear(self):
        for key in self.keys:
            self.d[key].clear()

    def __getitem__(self, k):
        return self.d[k]


def get_logits_from_lm(lm, inputs):
    past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)


def token_weighted_loss(loss_type, inputs, targets, weights):
    if loss_type == 'ce':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'nll':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1)
        weights = weights.view(-1)
        loss_fct = torch.nn.NLLLoss(reduction='none')
        loss = loss_fct(inputs, targets)
    elif loss_type == 'ul':
        probs = F.softmax(inputs, dim=-1)
        probs = torch.gather(probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.clamp((1.0 - probs), min=1e-5)
        loss = -torch.log(probs)
    elif loss_type == 'kl':
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1, targets.size(-1))
        weights = weights.view(-1)
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(inputs, targets)
        loss = loss.sum(dim=1)
    else:
        assert False

    loss = loss[weights != 0]
    return loss.mean()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        if self.args.loss_type == 'CoSec':
            self.loss_keys = ['sec', 'kl']

    def load_model(self):
        # self.tokenizer, self.model = load_model(self.args.pretrain_name, self.args, ref=False)
        # if self.args.kl_loss_weight > 0:
        #     _, self.ref_model = load_model(self.args.pretrain_name, self.args, ref=True)
        #     self.ref_model.resize_token_embeddings(len(self.tokenizer))
        #     # self.ref_model.eval()
        #     assert self.ref_model is not None
        # assert self.model is not None
        # # self.model.train()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model, device_map='auto', trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.args.model, device_map='auto', trust_remote_code=True
        )
        self.ref_model.resize_token_embeddings(len(self.tokenizer))
        self.model.resize_token_embeddings(len(self.tokenizer))
        # target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['qwen2']
        # target_modules = list(set([name for name in re.findall(r'\((\w+)\): Linear', str(model.modules))]))
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[  # 查询（q）、键（k）、值（v）
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
            ],
            lora_dropout=0.05,
            bias='none',
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)

        self.model.train()
        self.ref_model.eval()

    def load_dataset(self):
        self.dataset = CodeDataset(self.args, self.tokenizer, True)
        self.val_dataset = CodeDataset(self.args, self.tokenizer, False)
        # print(self.dataset.__getitem__(0))

    def CoSec_step(self, batch):
        loss_dict = LossDict(self.loss_keys)

        inputs = batch['input_ids'].to(self.model.device)
        weights = batch['weights'].to(self.model.device)
        inputs = inputs.to(self.model.device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:].squeeze(0)

        correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs)
        lm_loss = token_weighted_loss('ce', correct_logits, shift_inputs, shift_weights)
        loss_dict['sec'].append(lm_loss.item())

        assert self.args.kl_loss_weight > 0
        correct_log_probs = F.log_softmax(correct_logits, dim=-1)
        with torch.no_grad():
            ref_logits, _ = get_logits_from_lm(self.ref_model, inputs)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        kl_loss = token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1 - shift_weights)

        loss_dict['kl'].append(kl_loss.item())

        loss_total = lm_loss + kl_loss

        return loss_total, loss_dict

    def step(self, batch):
        pass

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict = self.CoSec_step(batch) if self.args.loss_type == 'CoSec' else self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def save(self, path):
        """
        For normal models this saves the whole set of weights, for LoRA models it saves the adapter.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def run(self):
        self.load_model()
        self.load_dataset()

        if self.args.lora:
            pass

        self.args.logger.info(f'Training args {self.args}')

        batch_size = self.args.batch_size
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if
                        (not any(nd in n for nd in no_decay)) and p.requires_grad],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if
                        any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]

        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
        #                                             num_training_steps=total_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_steps
        )
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)

        global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
        set_seed(self.args.seed)
        timer = Timer(total_steps)
        timer.start()
        self.model.train()
        self.ref_model.eval()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):

                loss, loss_dict = self.CoSec_step(batch) if self.args.loss_type == 'CoSec' else self.step(batch)
                loss /= self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                acc_loss_dict.step(loss_dict)

                if (step + 1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s, %s', idx + 1,
                                              self.args.num_train_epochs, global_step, total_steps, acc_loss_pp, timer)
                        acc_loss_dict.clear()

                    timer.end()
                    timer.start()

            if self.args.save_epochs > 0 and (idx + 1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    eval_loss_pp = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx + 1, eval_loss_pp)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx + 1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir)
                self.save(last_output_dir)

        if (idx + 1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                eval_loss_pp = self.do_eval()
            self.args.logger.info('final eval loss: %s', eval_loss_pp)
            output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.args.logger.info('Saving model checkpoint to %s', last_output_dir)
            self.save(output_dir)
            self.save(last_output_dir)
