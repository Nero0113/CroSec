import json
import os

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria,
    TopPLogitsWarper,
    TemperatureLogitsWarper
)
from transformers.generation import EosTokenCriteria
from transformers.generation.utils import GenerateOutput, GenerateDecoderOnlyOutput
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Callable, List, Union, Dict, Any



class EnsembleModel:
    def __init__(
            self,
            src_model_path=None,
            trg_model_path=None,
            sparse_matrix_path=None,
            token_map=None,
            src_model=None,
            trg_model=None,
            src_tokenizer=None,
            trg_tokenizer=None,
            ensemble_weight=0.5
    ):
        """
        Initialize the ensemble model with source and target models

        Args:
            src_model_path: Path to the source model
            trg_model_path: Path to the target model
            sparse_matrix_path: Path to the sparse similarity matrix for vocabulary mapping
            src_model: Pre-loaded source model (optional)
            trg_model: Pre-loaded target model (optional)
            src_tokenizer: Pre-loaded source tokenizer (optional)
            trg_tokenizer: Pre-loaded target tokenizer (optional)
            ensemble_weight: Weight for ensemble (0.5 means equal weight to both models)
        """
        # Load tokenizers
        self.src_tokenizer = src_tokenizer or AutoTokenizer.from_pretrained(src_model_path)
        self.trg_tokenizer = trg_tokenizer or AutoTokenizer.from_pretrained(trg_model_path)

        # Load models
        self.src_model = src_model.to('cuda') or AutoModelForCausalLM.from_pretrained(src_model_path).to('cuda')
        self.trg_model = trg_model.to('cuda') or AutoModelForCausalLM.from_pretrained(trg_model_path).to('cuda')

        self.trg_model.resize_token_embeddings(len(self.trg_tokenizer))
        self.src_model.resize_token_embeddings(len(self.src_tokenizer))

        self.ensemble_weight = ensemble_weight
        
        self.trg2src = torch.tensor(json.load(open(token_map)), dtype=torch.long, device=self.trg_model.device)
        # self.token_map = torch.tensor(
        #     json.load(open('/home/public_space/yanmeng/lidong/code/one4all/try_EVA_like/mapping_starcoder/token_map_full.json')), dtype=torch.long, device=self.trg_model.device)
        spmat = sp.load_npz(sparse_matrix_path)
        # Load sparse mapping matrix
        self.sparse_matrix = torch.sparse_csr_tensor(
            torch.tensor(spmat.indptr, dtype=torch.int32, device=self.src_model.device),
            torch.tensor(spmat.indices, dtype=torch.int32, device=self.src_model.device),
            torch.tensor(spmat.data, dtype=torch.float32, device=self.src_model.device),
            size=spmat.shape, device=self.src_model.device)

        # 断言
        assert self.sparse_matrix.size(0) == len(self.src_tokenizer)
        # assert len(self.token_map) == len(self.src_tokenizer)
        assert self.src_model.get_input_embeddings().num_embeddings == len(self.src_tokenizer)
        assert self.trg_model.get_input_embeddings().num_embeddings == len(self.trg_tokenizer)

        # self.token_map = json.load(open(token_map, 'r'))
        # self.one2one = True
        # self.token_map = torch.tensor(self.token_map).to(self.trg_model.device)

        # Set ensemble weight
        # self.ensemble_weight = ensemble_weight

        # Set models to evaluation mode
        self.src_model.eval()
        self.trg_model.eval()

    def format_starcoder_prompt(self, prompt):

        # 组织Starcoder的FIM格式输入
        fim_prefix = "<fim_prefix>"
        fim_suffix = "<fim_suffix>"
        fim_middle = "<fim_middle>"

        fim_input = fim_prefix + prompt + fim_suffix + fim_middle
        return fim_input

    @staticmethod
    def _expand_inputs_for_generation(
            expand_size: int = 1,
            input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        return input_ids

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences in the batch
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        """Extract past key/values from model output"""
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Standardize cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """Update model kwargs for the next generation step"""
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        # update attention mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def _get_topk_mask(self, logits, top_k=40):
        """Create a mask for top-k filtering"""
        filter_value = -float("Inf")
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        return indices_to_remove, filter_value

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            do_sample: bool = False,
            num_return_sequences: Optional[int] = 1,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            max_new_tokens: Optional[int] = None,
            max_length: Optional[int] = None,
            top_p: float = 1.0,
            top_k: int = 0,
            temperature: float = 1.0,
            synced_gpus: bool = False,
            use_cache: bool = True,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate sequences with ensemble model.

        Args:
            input_ids: Input token ids for target model
            attention_mask: Attention mask for input
            do_sample: Whether to sample instead of greedy decoding
            num_return_sequences: Number of sequences to generate per input
            logits_processor: Custom logits processors
            stopping_criteria: Custom stopping criteria
            logits_warper: Custom logits warpers
            pad_token_id: Token ID for padding
            eos_token_id: Token ID to indicate end of sequence
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum length of output sequence (overrides max_new_tokens)
            top_p: Top-p for nucleus sampling
            top_k: Top-k for top-k sampling
            temperature: Temperature for sampling
            synced_gpus: Whether to sync GPUs
            use_cache: Whether to use KV-cache for faster generation
            **kwargs: Additional arguments for model forward pass

        Returns:
            Generated token IDs
        """
        # Set up base kwargs for both models
        trg_model_kwargs = kwargs.copy()
        src_model_kwargs = kwargs.copy()

        if use_cache:
            trg_model_kwargs["use_cache"] = use_cache
            src_model_kwargs["use_cache"] = use_cache

        # Add attention mask if provided
        if attention_mask is not None:
            trg_model_kwargs["attention_mask"] = attention_mask
            src_model_kwargs["attention_mask"] = attention_mask

        # Set up logits processors and warpers
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        # Add temperature and top-p if not default values
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_p != 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        # Determine stopping criteria
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        elif max_new_tokens is not None:
            stopping_criteria.append(
                MaxLengthCriteria(max_length=input_ids.shape[-1] + max_new_tokens)
            )

        # Add EOS token stopping criteria if provided
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        # Default pad_token_id if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.trg_tokenizer.pad_token_id

        # Prepare storage for generated outputs
        scores = ()
        raw_logits = ()

        # # Expand inputs for multiple sequences if needed
        # if num_return_sequences > 1:
        #     input_ids = self._expand_inputs_for_generation(num_return_sequences, input_ids)
        #     if attention_mask is not None:
        #         trg_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        #         src_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        text = self.trg_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # src_input = self.src_tokenizer(text, return_tensors='pt')

        # ---- Starcoder: FIM prompt formatting ----for src_model
        if "starcoder" in self.src_model.config.name_or_path.lower() or "codeshell" in self.src_model.config.name_or_path.lower():
            fim_input = self.format_starcoder_prompt(text)
            src_input = self.src_tokenizer(fim_input, return_tensors="pt").to(self.src_model.device)
            src_input_ids = src_input.input_ids.to(self.src_model.device)
            if attention_mask is not None:
                src_model_kwargs["attention_mask"] = src_input.attention_mask
        else:
            src_input = self.src_tokenizer(text, return_tensors='pt')
            src_input_ids = src_input.input_ids.to(self.src_model.device)
            if attention_mask is not None:
                src_model_kwargs["attention_mask"] = src_input.attention_mask

        # ---- Starcoder: FIM prompt formatting ----for trg_model
        if "starcoder" in self.trg_model.config.name_or_path.lower() or "codeshell" in self.trg_model.config.name_or_path.lower():
            fim_input = self.format_starcoder_prompt(text)
            inputs = self.trg_tokenizer(fim_input, return_tensors="pt").to(self.trg_model.device)
            input_ids = inputs.input_ids
            if attention_mask is not None:
                attention_mask = inputs.attention_mask
                trg_model_kwargs["attention_mask"] = attention_mask
                
        # Expand inputs for multiple sequences if needed
        if num_return_sequences > 1:
            input_ids = self._expand_inputs_for_generation(num_return_sequences, input_ids)
            src_input_ids = self._expand_inputs_for_generation(num_return_sequences, src_input_ids)
            src_input_ids = src_input_ids.to(self.src_model.device)
            if attention_mask is not None:
                trg_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)
                src_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        else:

            src_input_ids = src_input.input_ids.to(self.src_model.device)
            if attention_mask is not None:
                trg_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)
                src_model_kwargs["attention_mask"] = attention_mask.repeat_interleave(num_return_sequences, dim=0)
        # Move inputs to correct devices
        input_ids = input_ids.to(self.trg_model.device)
        trg_model_kwargs = {k: v.to(self.trg_model.device) if hasattr(v, 'to') else v for k, v in trg_model_kwargs.items()}
        src_input_ids = src_input_ids.to(self.src_model.device)
        src_model_kwargs = {k: v.to(self.src_model.device) if hasattr(v, 'to') else v for k, v in src_model_kwargs.items()}

        # Keep track of which sequences are finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Generate tokens auto-regressively
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if synced_gpus and this_peer_finished:
                continue  # Skip computation if this GPU is done

            # Prepare inputs for both models
            trg_model_inputs = self.trg_model.prepare_inputs_for_generation(input_ids, **trg_model_kwargs)

            # Create position IDs explicitly for source model to avoid shape mismatch
            src_position_ids = torch.arange(src_input_ids.shape[1], dtype=torch.long, device=src_input_ids.device)
            src_position_ids = src_position_ids.unsqueeze(0).expand_as(src_input_ids)

            # Prepare source model inputs with explicit position IDs
            src_model_kwargs_copy = src_model_kwargs.copy()
            src_model_kwargs_copy["position_ids"] = src_position_ids
            src_model_inputs = self.src_model.prepare_inputs_for_generation(src_input_ids, **src_model_kwargs_copy)

            ids = src_model_inputs["input_ids"]
            if torch.any(ids >= self.src_tokenizer.vocab_size) or torch.any(ids < 0):
                bad = ids[(ids >= self.src_tokenizer.vocab_size) | (ids < 0)]
                print("[FATAL] illegal token ids:", bad[:10], "  max_id:", ids.max(), "  min_id:", ids.min())
                raise ValueError("found out-of-range token ids")

            # Get outputs from both models
            trg_outputs = self.trg_model(**trg_model_inputs, return_dict=True)
            src_outputs = self.src_model(**src_model_inputs, return_dict=True)

            # Get next token logits
            trg_next_token_logits = trg_outputs.logits[:, -1, :]
            src_next_token_logits = src_outputs.logits[:, -1, :]

            # Map source logits to target vocabulary space using sparse matrix
            src_next_token_logits = src_next_token_logits.to(torch.float32)

            # Convert to probability distributions
            src_probs = nn.functional.softmax(src_next_token_logits, dim=-1)

            mapped_src_probs = torch.spmm(src_probs, self.sparse_matrix)  # [B, V_trg]
            mapped_src_probs = mapped_src_probs.clamp_min_(1e-20).to(trg_next_token_logits.device)

            # Convert target logits to probabilities
            trg_probs = nn.functional.softmax(trg_next_token_logits, dim=-1)


            invalid_mask = (self.trg2src == -1)

            ensemble_probs = (1 - self.ensemble_weight) * trg_probs + self.ensemble_weight * mapped_src_probs
            ensemble_probs[:, invalid_mask] = 0
            assert self.sparse_matrix.size(1) == ensemble_probs.size(-1)  # 若已放在 generate 里

            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True)  # ②
            next_token_logits = torch.log(ensemble_probs)

            # Process logits
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Sample or greedy decode
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Store scores
            scores += (next_token_scores,)
            raw_logits += (next_token_logits,)

            # Set next tokens for finished sequences to pad token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)


            # Convert to tensor
            next_src_tokens = self.trg2src[next_tokens]
            next_src_tokens = next_src_tokens.to(self.src_model.device)
            # Update source input IDs
            src_input_ids = torch.cat([src_input_ids, next_src_tokens[:, None]], dim=-1)

            # Update target model kwargs for next generation step
            trg_model_kwargs = self._update_model_kwargs_for_generation(
                trg_outputs, trg_model_kwargs,
                is_encoder_decoder=False,
                standardize_cache_format=False
            )

            # For source model, we'll reset most kwargs since we're re-processing tokens
            # But we'll keep the cache if it exists
            if use_cache and "past_key_values" in src_model_kwargs:
                # Extract past key values
                past_key_values = self._extract_past_from_model_output(
                    src_outputs, standardize_cache_format=False
                )
                src_model_kwargs = {"past_key_values": past_key_values, "use_cache": True}
            else:
                # Reset to empty dict if not using cache
                src_model_kwargs = {"use_cache": use_cache}

            # Update which sequences are finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # Clean up memory for this iteration
            del trg_outputs
            del src_outputs

            # Break if all sequences are finished
            if this_peer_finished:
                break

        # Return the generated sequences
        return input_ids

class EnsembleFamilyModel:
    def __init__(
            self,
            src_model_path=None,
            trg_model_path=None,
            sparse_matrix_path=None,
            token_map=None,
            src_model=None,
            trg_model=None,
            src_tokenizer=None,
            trg_tokenizer=None,
            ensemble_weight=0.5
    ):
        """
        Initialize the ensemble model with source and target models

        Args:
            src_model_path: Path to the source model
            trg_model_path: Path to the target model
            sparse_matrix_path: Path to the sparse similarity matrix for vocabulary mapping
            src_model: Pre-loaded source model (optional)
            trg_model: Pre-loaded target model (optional)
            src_tokenizer: Pre-loaded source tokenizer (optional)
            trg_tokenizer: Pre-loaded target tokenizer (optional)
            ensemble_weight: Weight for ensemble (0.5 means equal weight to both models)
        """
        # Load tokenizers
        self.src_tokenizer = src_tokenizer or AutoTokenizer.from_pretrained(src_model_path)
        self.trg_tokenizer = trg_tokenizer or AutoTokenizer.from_pretrained(trg_model_path)

        # Load models
        self.src_model = src_model.to('cuda') or AutoModelForCausalLM.from_pretrained(src_model_path).to('cuda')
        self.trg_model = trg_model.to('cuda') or AutoModelForCausalLM.from_pretrained(trg_model_path).to('cuda')

        self.trg_model.resize_token_embeddings(len(self.trg_tokenizer))
        self.src_model.resize_token_embeddings(len(self.src_tokenizer))

        self.ensemble_weight = ensemble_weight

        # Set models to evaluation mode
        self.src_model.eval()
        self.trg_model.eval()

    def format_starcoder_prompt(self, prompt):

        # 组织Starcoder的FIM格式输入
        fim_prefix = "<fim_prefix>"
        fim_suffix = "<fim_suffix>"
        fim_middle = "<fim_middle>"

        fim_input = fim_prefix + prompt + fim_suffix + fim_middle
        return fim_input

    @staticmethod
    def _expand_inputs_for_generation(
            expand_size: int = 1,
            input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        return input_ids

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences in the batch
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        """Extract past key/values from model output"""
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Standardize cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """Update model kwargs for the next generation step"""
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        # update attention mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def _get_topk_mask(self, logits, top_k=40):
        """Create a mask for top-k filtering"""
        filter_value = -float("Inf")
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        return indices_to_remove, filter_value

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            do_sample: bool = False,
            num_return_sequences: Optional[int] = 1,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            max_new_tokens: Optional[int] = None,
            max_length: Optional[int] = None,
            top_p: float = 1.0,
            top_k: int = 0,
            temperature: float = 1.0,
            synced_gpus: bool = False,
            use_cache: bool = True,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate sequences with ensemble model.

        Args:
            input_ids: Input token ids for target model
            attention_mask: Attention mask for input
            do_sample: Whether to sample instead of greedy decoding
            num_return_sequences: Number of sequences to generate per input
            logits_processor: Custom logits processors
            stopping_criteria: Custom stopping criteria
            logits_warper: Custom logits warpers
            pad_token_id: Token ID for padding
            eos_token_id: Token ID to indicate end of sequence
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum length of output sequence (overrides max_new_tokens)
            top_p: Top-p for nucleus sampling
            top_k: Top-k for top-k sampling
            temperature: Temperature for sampling
            synced_gpus: Whether to sync GPUs
            use_cache: Whether to use KV-cache for faster generation
            **kwargs: Additional arguments for model forward pass

        Returns:
            Generated token IDs
        """
        # Set up base kwargs for both models
        trg_model_kwargs = kwargs.copy()
        src_model_kwargs = kwargs.copy()

        if use_cache:
            trg_model_kwargs["use_cache"] = use_cache
            src_model_kwargs["use_cache"] = use_cache

        # Add attention mask if provided
        if attention_mask is not None:
            trg_model_kwargs["attention_mask"] = attention_mask
            src_model_kwargs["attention_mask"] = attention_mask

        # Set up logits processors and warpers
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        # Add temperature and top-p if not default values
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_p != 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

        # Determine stopping criteria
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        elif max_new_tokens is not None:
            stopping_criteria.append(
                MaxLengthCriteria(max_length=input_ids.shape[-1] + max_new_tokens)
            )

        # Add EOS token stopping criteria if provided
        if eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        # Default pad_token_id if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.trg_tokenizer.pad_token_id

        # Prepare storage for generated outputs
        scores = ()
        raw_logits = ()

        text = self.trg_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        src_input = self.src_tokenizer(text, return_tensors='pt').to(self.src_model.device)
        src_input_ids = src_input.input_ids.to(self.src_model.device)

        # ---- Starcoder: FIM prompt formatting ----
        if "starcoder" in self.trg_model.config.name_or_path.lower() or "codeshell" in self.trg_model.config.name_or_path.lower():
            fim_input = self.format_starcoder_prompt(text)         
            inputs = self.trg_tokenizer(fim_input, return_tensors="pt").to(self.trg_model.device)
            src_input = self.src_tokenizer(fim_input, return_tensors='pt').to(self.src_model.device)
            input_ids = inputs.input_ids.to(self.trg_model.device)
            src_input_ids = src_input.input_ids.to(self.src_model.device)
            if attention_mask is not None:
                trg_model_kwargs["attention_mask"] = inputs.attention_mask
                src_model_kwargs["attention_mask"] = src_input.attention_mask

        # Expand inputs for multiple sequences if needed
        if num_return_sequences > 1:
            input_ids = self._expand_inputs_for_generation(num_return_sequences, input_ids)
            src_input_ids = self._expand_inputs_for_generation(num_return_sequences, src_input.input_ids)
            src_input_ids = src_input_ids.to(self.src_model.device)
            if attention_mask is not None:
                trg_model_kwargs["attention_mask"] = trg_model_kwargs["attention_mask"].repeat_interleave(num_return_sequences, dim=0)
                src_model_kwargs["attention_mask"] = src_model_kwargs["attention_mask"].repeat_interleave(num_return_sequences, dim=0)

        # Move inputs to correct devices
        input_ids = input_ids.to(self.trg_model.device)
        trg_model_kwargs = {k: v.to(self.trg_model.device) if hasattr(v, 'to') else v for k, v in trg_model_kwargs.items()}
        src_input_ids = src_input_ids.to(self.src_model.device)
        src_model_kwargs = {k: v.to(self.src_model.device) if hasattr(v, 'to') else v for k, v in src_model_kwargs.items()}

        # Keep track of which sequences are finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Generate tokens auto-regressively
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if synced_gpus and this_peer_finished:
                continue  # Skip computation if this GPU is done

            # Prepare inputs for both models
            trg_model_inputs = self.trg_model.prepare_inputs_for_generation(input_ids, **trg_model_kwargs)

            # Create position IDs explicitly for source model to avoid shape mismatch
            src_position_ids = torch.arange(src_input_ids.shape[1], dtype=torch.long, device=src_input_ids.device)
            src_position_ids = src_position_ids.unsqueeze(0).expand_as(src_input_ids)

            # Prepare source model inputs with explicit position IDs
            src_model_kwargs_copy = src_model_kwargs.copy()
            src_model_kwargs_copy["position_ids"] = src_position_ids
            src_model_inputs = self.src_model.prepare_inputs_for_generation(src_input_ids, **src_model_kwargs_copy)

            # Get outputs from both models
            trg_outputs = self.trg_model(**trg_model_inputs, return_dict=True)
            src_outputs = self.src_model(**src_model_inputs, return_dict=True)

            # Get next token logits
            trg_next_token_logits = trg_outputs.logits[:, -1, :]
            src_next_token_logits = src_outputs.logits[:, -1, :]

            # Map source logits to target vocabulary space using sparse matrix
            src_next_token_logits = src_next_token_logits.to(torch.float32)

            # Top-k filtering for source logits before mapping (不能少)
            # if top_k > 0:
            #     src_indices_to_remove, filter_value = self._get_topk_mask(src_next_token_logits, top_k=top_k)
            #     src_next_token_logits = src_next_token_logits.masked_fill(src_indices_to_remove, filter_value)

            # Convert to probability distributions
            src_probs = nn.functional.softmax(src_next_token_logits, dim=-1)

            # Transform source probabilities to target vocabulary space
            # src_probs_t = src_probs.t()
            # mapped_src_probs = torch.spmm(self.sparse_matrix.to(src_probs.device), src_probs_t)
            # mapped_src_probs = mapped_src_probs.t().to(trg_next_token_logits.device)
            # mapped_src_probs = torch.spmm(src_probs, self.sparse_matrix)  # [B, V_trg]
            mapped_src_probs = src_probs
            mapped_src_probs = mapped_src_probs.clamp_min_(1e-20).to(trg_next_token_logits.device)

            # Convert target logits to probabilities
            trg_probs = nn.functional.softmax(trg_next_token_logits, dim=-1)

            #Ensemble the probabilities
            ensemble_probs = (1 - self.ensemble_weight) * trg_probs + self.ensemble_weight * mapped_src_probs

            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=-1, keepdim=True)  # ②
            next_token_logits = torch.log(ensemble_probs)

            # Process logits
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Sample or greedy decode
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Store scores
            scores += (next_token_scores,)
            raw_logits += (next_token_logits,)

            # Set next tokens for finished sequences to pad token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)


            # Convert to tensor
            # next_src_tokens = self.trg2src[next_tokens]
            next_src_tokens = next_tokens.to(self.src_model.device)
   
            # Update source input IDs
            src_input_ids = torch.cat([src_input_ids, next_src_tokens[:, None]], dim=-1)

            # Update target model kwargs for next generation step
            trg_model_kwargs = self._update_model_kwargs_for_generation(
                trg_outputs, trg_model_kwargs,
                is_encoder_decoder=False,
                standardize_cache_format=False
            )

            # For source model, we'll reset most kwargs since we're re-processing tokens
            # But we'll keep the cache if it exists
            if use_cache and "past_key_values" in src_model_kwargs:
                # Extract past key values
                past_key_values = self._extract_past_from_model_output(
                    src_outputs, standardize_cache_format=False
                )
                src_model_kwargs = {"past_key_values": past_key_values, "use_cache": True}
            else:
                # Reset to empty dict if not using cache
                src_model_kwargs = {"use_cache": use_cache}

            # Update which sequences are finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # Clean up memory for this iteration
            del trg_outputs
            del src_outputs

            # Break if all sequences are finished
            if this_peer_finished:
                break

        # Return the generated sequences
        return input_ids
