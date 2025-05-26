import re

import peft
from peft import PeftModel
from transformers import AutoTokenizer, CodeGenForCausalLM, AutoModelForCausalLM

from Ensemble_model import EnsembleModel
#from scripts.co_generation import CodegenModelLM
from utils import load_model, try_parse


class LM_Evaler:

    def __init__(self, args):
        self.args = args
        self.load_model()

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lora' if self.args.model_type == 'lora' else 'lm', self.args.model_name_or_path,
                                              False, self.args)
        self.model.eval()

    def truncate(self, completion, lang):
        if lang == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                last_comment_str = '\n    #'
                if last_comment_str in completion:
                    completion = completion[:completion.rfind(last_comment_str)]
        elif lang == 'c':
            if '\n}' in completion:
                completion = completion[:completion.find('\n}')+2]
            else:
                last_comment_strs = ['\n    //', '\n    /*']
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[:completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + '\n}'

            lines = completion.split('\n')
            final_lines = []
            for line in lines:
                if '->name = "' in line: continue
                final_lines.append(line)
            completion = '\n'.join(final_lines)
        else:
            raise NotImplementedError()

        return completion

    def process_completions(self, input_src, input_ids_len, gen_output, lang):

        tokens = gen_output[:, input_ids_len:, ...]
        completions = self.tokenizer.batch_decode(tokens)

        output_srcs, output_ids = [], []
        dup_srcs, non_parsed_srcs = [], []
        for i, completion in enumerate(completions):
            #如果eos在文中，就在eos处停止
            if self.tokenizer.eos_token in completion:
                completion = completion[:completion.find(self.tokenizer.eos_token)]

            completion = self.truncate(completion, lang)
            completion_len = len(self.tokenizer.encode(completion))

            output_src = input_src + completion
            output_src = output_src.rstrip() + '\n'

            if output_src in output_srcs:
                dup_srcs.append(output_src)
            elif try_parse(output_src, lang) != 0:
                non_parsed_srcs.append(output_src)
            else:
                output_srcs.append(output_src)
                output_ids.append((gen_output[i][:input_ids_len].tolist(), gen_output[i][input_ids_len:input_ids_len+completion_len].tolist()))

        return output_srcs, output_ids, dup_srcs, non_parsed_srcs


    def sample(self, file_context, func_context, control, lang):
        #input_src: 完整的python或者c/c++文件，从import 到函数定义
        input_src = file_context + func_context
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            # return_dict_in_generate=True,
            # output_scores=True,
        )

        return self.process_completions(input_src, input_ids_len, gen_output, lang)

class CO_Evaler:

    def __init__(self, args):
        self.args = args
        self.load_model()

    def load_model(self):
        self.trg_model = AutoModelForCausalLM.from_pretrained(self.args.trg_model)
        self.trg_tokenizer = AutoTokenizer.from_pretrained(self.args.trg_model)

        #
        # self.trg_tokenizer.eos_token = self.trg_tokenizer.bos_token

        self.src_model = AutoModelForCausalLM.from_pretrained(self.args.src_model)
        self.src_tokenizer = AutoTokenizer.from_pretrained(self.args.src_model)
        self.src_model.resize_token_embeddings(len(self.src_tokenizer))
        self.src_model = peft.PeftModel.from_pretrained(self.src_model, self.args.lora)

        self.model = EnsembleModel(
            src_model=self.src_model,
            trg_model=self.trg_model,
            src_tokenizer=self.src_tokenizer,
            trg_tokenizer=self.trg_tokenizer,
            sparse_matrix_path=self.args.sparse_matrix_path,
            token_map=self.args.token_map,
            ensemble_weight=self.args.ensemble_weight  # Equal weight to both models
        )

        self.input_device = self.args.device


    def truncate(self, completion, lang):
        if lang == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                last_comment_str = '\n    #'
                if last_comment_str in completion:
                    completion = completion[:completion.rfind(last_comment_str)]
        elif lang == 'c':
            if '\n}' in completion:
                completion = completion[:completion.find('\n}')+2]
            else:
                last_comment_strs = ['\n    //', '\n    /*']
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[:completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + '\n}'

            lines = completion.split('\n')
            final_lines = []
            for line in lines:
                if '->name = "' in line: continue
                final_lines.append(line)
            completion = '\n'.join(final_lines)
        else:
            raise NotImplementedError()

        return completion

    def replace_gen_prompt(self, prompt: str, model_path: str) -> str:
        if "starcoder" in model_path.lower():
            prompt = prompt.replace("<fim_prefix>", "")
            prompt = prompt.replace("<fim_suffix><fim_middle>", "")

        else:
            prompt = prompt.replace(
                "Please complete the following Python code without providing any additional tasks such as testing or explanations\n",
                "")
        if "starchat" in model_path.lower():
            prompt = prompt.replace("<|system|>\n<|end|>\n<|user|>", "")
            prompt = prompt.replace("<|end|>\n<|assistant|>", "")
        return prompt

    def process_completions(self, input_src, input_ids_len, gen_output, lang):

        tokens = gen_output[:, input_ids_len:, ...]
        completions = self.trg_tokenizer.batch_decode(tokens)
        # add for starcoder
        # if "starcoder" in self.args.trg_model.lower():
        #     completions = [self.replace_gen_prompt(c, self.args.trg_model) for c in completions]
        output_srcs, output_ids = [], []
        dup_srcs, non_parsed_srcs = [], []
        for i, completion in enumerate(completions):
            #如果eos在文中，就在eos处停止
            if self.trg_tokenizer.eos_token in completion:
                completion = completion[:completion.find(self.trg_tokenizer.eos_token)]

            completion = self.truncate(completion, lang)
            completion_len = len(self.trg_tokenizer.encode(completion))

            output_src = input_src + completion
            output_src = output_src.rstrip() + '\n'

            if output_src in output_srcs:
                dup_srcs.append(output_src)
            elif try_parse(output_src, lang) != 0:
                non_parsed_srcs.append(output_src)
            else:
                output_srcs.append(output_src)
                output_ids.append((gen_output[i][:input_ids_len].tolist(), gen_output[i][input_ids_len:input_ids_len+completion_len].tolist()))

        return output_srcs, output_ids, dup_srcs, non_parsed_srcs


    def sample(self, file_context, func_context, control, lang):
        #input_src: 完整的python或者c/c++文件，从import 到函数定义
        input_src = file_context + func_context
        input_trg = self.trg_tokenizer(input_src, return_tensors='pt').to(self.trg_model.device)
        input_ids_len = input_trg.input_ids.shape[1]
        # if "starcoder" in self.args.trg_model.lower():
        #     input_ids_len = input_ids_len + 3

        gen_output = self.model.generate(
            **input_trg,
            do_sample=True,
            num_return_sequences=25,  # Generate 5 different completions
            temperature=self.args.temp,
            max_new_tokens=self.args.max_new_len,
            top_p=self.args.top_p,
            pad_token_id=self.trg_tokenizer.eos_token_id,
            eos_token_id=self.trg_tokenizer.eos_token_id,
            use_cache=True,
        )

        return self.process_completions(input_src, input_ids_len, gen_output, lang)


