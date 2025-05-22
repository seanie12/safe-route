from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import torchbnn as bnn

class BNN(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.cls = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=512
            ),
            nn.LayerNorm(512),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=512, out_features=512
            ),
            nn.LayerNorm(512),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=512, out_features=1
            ),
        )
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

        def weights_init(m):
            if isinstance(m, bnn.BayesLinear):
                nn.init.kaiming_normal_(m.weight_mu)

        self.cls.apply(weights_init)

    def get_features(self, x):
        num_layers = len(self.cls)
        features = []
        for i in range(num_layers-1):
            x = self.cls[i](x)
            if isinstance(self.cls[i], nn.ReLU):
                features.append(x)
        return features

    def train(self, mode: bool = True):
        self.cls.train(mode)
        self.training = True
        for module in self.cls:
            if isinstance(module, bnn.BayesLinear):
                module.unfreeze()

    def eval(self):
        self.cls.eval()
        self.training = False
        for module in self.cls:
            if isinstance(module, bnn.BayesLinear):
                module.freeze(zero=True)

    def forward(self, x):
        return self.cls(x).squeeze(-1)

    def get_kl(self):
        return self.kl_loss(self.cls).sum()


class VLLMGuard(object):
    def __init__(self, device, version=3, pbar=False) -> None:
        if version == 1:
            model_id = "meta-llama/LlamaGuard-7b"
            max_model_len = 4096
        elif version == 2:
            print("version-2")
            model_id = "meta-llama/Meta-Llama-Guard-2-8B"
            max_model_len = 8192
        else:
            print("version-3")
            model_id = "meta-llama/Llama-Guard-3-8B"
            max_model_len = 8192

        self.version = version
        dtype = "bfloat16"
        self.pbar = pbar

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(
            model_id, dtype=dtype, max_model_len=max_model_len, device=device
        )

        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)

        unsafe_token = self.tokenizer.tokenize("unsafe")
        safe_token = self.tokenizer.tokenize("safe")

        self.unsafe_id = self.tokenizer.convert_tokens_to_ids(unsafe_token)[0]
        self.safe_id = self.tokenizer.convert_tokens_to_ids(safe_token)[0]

    def make_chat(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = []
            for x, y in zip(prompts, responses):
                if y is not None:
                    chat = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x},
                            {"role": "assistant", "content": y},
                        ],
                        tokenize=False,
                    )
                else:
                    chat = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x},
                        ],
                        tokenize=False,
                    )
                chats.append(chat)
            assert len(chats) == len(prompts) & len(chats) == len(responses)
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}], tokenize=False
                )
                for x in prompts
            ]
        if self.version == 3:
            chats = [chat + "\n\n" for chat in chats]
        return chats

    def compute(self, prompts, responses=None):
        system_prompts = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)

        llm_outputs = self.llm.generate(
            system_prompts, sampling_params, use_tqdm=self.pbar
        )
        logits = []

        for output in llm_outputs:
            safe_log_prob = output.outputs[0].logprobs[0][self.safe_id].logprob
            unsafe_log_prob = output.outputs[0].logprobs[0][self.unsafe_id].logprob
            log_probs = [safe_log_prob, unsafe_log_prob]
            logits.append(log_probs)

        logits = torch.tensor(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        unsafe_log_prob = log_probs[:, 1]

        return unsafe_log_prob

    def get_soft_label(self, prompts, responses=None, temperature=1.0):
        system_prompts = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=2)

        llm_outputs = self.llm.generate(
            system_prompts, sampling_params, use_tqdm=self.pbar
        )

        logits = []

        for output in llm_outputs:
            safe_log_prob = output.outputs[0].logprobs[0][self.safe_id].logprob
            unsafe_log_prob = output.outputs[0].logprobs[0][self.unsafe_id].logprob
            log_probs = [safe_log_prob, unsafe_log_prob]
            logits.append(log_probs)
        logits = torch.tensor(logits, dtype=torch.float)
        soft_labels = F.softmax(logits / temperature, -1)
        unsafe_labels = soft_labels[:, 1]
        unsafe_log_probs = torch.log(unsafe_labels)

        return unsafe_log_probs

    def generate_label(self, prompts, responses=None):
        chats = self.make_chat(prompts, responses)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=20, min_tokens=1)
        llm_outputs = self.llm.generate(
            chats, sampling_params, use_tqdm=self.pbar)
        labels = []

        for output in llm_outputs:
            label = output.outputs[0].text
            labels.append(label)

        return labels


class Guardian(object):
    def __init__(self, device):
        self.device = device
        model_path = "ibm-granite/granite-guardian-3.0-8b"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left")

        self.guardian_config = {"risk_name": "harm"}

        safe_token = "No"
        unsafe_token = "Yes"
        self.safe_token_id = self.tokenizer.convert_tokens_to_ids(safe_token)
        self.unsafe_token_id = self.tokenizer.convert_tokens_to_ids(
            unsafe_token)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def make_chat(self, prompts, responses=None):
        if responses is not None:
            chats = []
            assert len(prompts) == len(responses)
            for x, y in zip(prompts, responses):
                if y is not None:
                    chat = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": x},
                         {"role": "assistant", "content": y}],
                        guardian_config=self.guardian_config,
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                else:
                    chat = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": x}],
                        guardian_config=self.guardian_config,
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                chats.append(chat)
        else:
            chats = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": x}],
                guardian_config=self.guardian_config,
                add_generation_prompt=True, 
                tokenize=False
            ) for x in prompts
            ]
        return chats
    
    def get_inputs(self, prompts, responses=None):
        chats = self.make_chat(prompts, responses)

        inputs = self.tokenizer(
            chats, return_tensors="pt", padding=True)
        return inputs

    @torch.no_grad()
    def compute(self, prompts, responses=None):
        inputs = self.get_inputs(prompts, responses).to(self.device)

        logits = self.model(**inputs).logits[:, -1]
        logits = torch.stack(
            [logits[:, self.safe_token_id], logits[:, self.unsafe_token_id]],
            dim=1
        )
        log_probs = F.log_softmax(logits, -1)
        unsafe_logprob = log_probs[:, 1]

        return unsafe_logprob


    @torch.no_grad()
    def forward(self, prompts, responses=None, t=1, layer_idx=-1):
        inputs = self.get_inputs(prompts, responses)
        inputs = inputs.to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[layer_idx]  # token reps
        features = hidden[:, -1, :]  # last token

        logits = outputs.logits[:, -1, :]
        logits = torch.stack(
            [logits[:, self.safe_token_id], logits[:, self.unsafe_token_id]], dim=1
        )
        log_probs = F.log_softmax(logits / t, -1)
        unsafe_logprob = log_probs[:, 1]

        output_dict = {
            "features": features,
            "logits": logits,
            "unsafe_logprob": unsafe_logprob,
        }

        return output_dict

class LlamaToxicClassifier(object):
    def __init__(self, device, version=3, custom_template=None) -> None:
        if version == 1:
            model_id = "meta-llama/LlamaGuard-7b"
        elif version == 2:
            model_id = "meta-llama/Meta-Llama-Guard-2-8B"
        elif version == 3:
            model_id = "meta-llama/Llama-Guard-3-8B"
        elif version == "1b":
            model_id = "meta-llama/Llama-Guard-3-1B"
        print(f"version: {version}")
        self.version = version
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if custom_template is None:
            self.chat_template = self.tokenizer.chat_template  # default chat template
        else:
            self.chat_template = custom_template

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
        )
        self.hidden_size = self.model.config.hidden_size
        self.model.eval()
        unsafe_token = self.tokenizer.tokenize("unsafe")
        self.unsafe_id = self.tokenizer.convert_tokens_to_ids(unsafe_token)[0]
        safe_token = self.tokenizer.tokenize("safe")
        self.safe_id = self.tokenizer.convert_tokens_to_ids(safe_token)[0]

        if version == 3:
            assert len(self.tokenizer.encode("\n\n")) == 1
            self.newline_id = self.tokenizer.encode("\n\n")[0]
        elif version == "1b":
            self.newline_id = 271

    def make_chat(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = []
            for x, y in zip(prompts, responses):
                if y is not None:
                    chat = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x},
                            {"role": "assistant", "content": y},
                        ],
                        tokenize=False,
                        chat_template=self.chat_template,
                    )
                else:
                    chat = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x},
                        ],
                        tokenize=False,
                        chat_template=self.chat_template,
                    )
                chats.append(chat)
            assert len(chats) == len(prompts) & len(chats) == len(responses)
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": x}],
                    tokenize=False,
                    chat_template=self.chat_template,
                )
                for x in prompts
            ]
        return chats

    def make_chat_v2(self, prompts, responses=None):
        if responses is not None:
            assert len(prompts) == len(responses)
            chats = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": [
                            {"type": "text", "text": x}]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": y}]},
                    ],
                    tokenize=False,
                    chat_template=self.chat_template,
                )
                for x, y in zip(prompts, responses)
            ]
        else:
            chats = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": x}]}],
                    tokenize=False,
                    chat_template=self.chat_template,
                )
                for x in prompts
            ]

        return chats

    @torch.no_grad()
    def compute(self, prompts, responses=None, ts=1.0):
        input_ids, attention_mask = self.get_inputs(prompts, responses)

        self.model.eval()
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        logits = torch.stack(
            [logits[:, self.safe_id], logits[:, self.unsafe_id]], dim=1
        )

        log_probs = F.log_softmax(logits / ts, -1)
        unsafe_logprob = log_probs[:, 1]

        return unsafe_logprob

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @torch.no_grad()
    def forward(self, prompts, responses=None, t=1, layer_idx=-1):
        input_ids, attention_mask = self.get_inputs(prompts, responses)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[layer_idx]  # token reps
        features = hidden[:, -1, :]  # last token

        logits = outputs.logits[:, -1, :].float()  # cast to float32
        logits = torch.stack(
            [logits[:, self.safe_id], logits[:, self.unsafe_id]], dim=1
        )
        log_probs = F.log_softmax(logits / t, -1)
        unsafe_logprob = log_probs[:, 1]

        output_dict = {
            "features": features,
            "logits": logits,
            "unsafe_logprob": unsafe_logprob,
        }

        return output_dict

    def get_inputs(self, prompts, responses=None):
        if self.version == "1b":
            chats = self.make_chat_v2(prompts, responses)
        else:
            chats = self.make_chat(prompts, responses)
        inputs = self.tokenizer(chats, padding=True, return_tensors="pt")

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Llama-guard-3 predict safe/unsafe after the token "\n\n"
        if self.version == 3 or self.version == "1b":
            bs = input_ids.size(0)
            _new_id = (
                torch.ones((bs, 1), dtype=torch.long, device=self.device)
                * self.newline_id
            )
            _new_mask = torch.ones_like(_new_id)
            input_ids = torch.cat([input_ids, _new_id], dim=1)
            attention_mask = torch.cat([attention_mask, _new_mask], dim=1)

        return input_ids, attention_mask


@dataclass
class ModelResults:
    probs: Optional[torch.Tensor]
    preds: Optional[torch.Tensor]
    final_labels: Optional[torch.Tensor]

class SafetyGuard(nn.Module):
    def __init__(
        self,
        router_ckpt: str,
        small_model: LlamaToxicClassifier,
        large_model: LlamaToxicClassifier,
    ) -> None:
        super().__init__()
        self.device = small_model.device
        self.small_model = small_model
        self.large_model = large_model
        
        # Initialize router
        self.router = self._init_router(router_ckpt)
        
    def _init_router(self, router_ckpt: str) -> nn.Module:
        """Initialize and load the router model."""
        router = BNN(self.small_model.hidden_size).to(self.device)
        # Load checkpoint directly to device
        ckpt = torch.load(router_ckpt)
        self.layer_idx = ckpt["layer_idx"]
        msg = router.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Router initialization: {msg}")
        return router

    def eval(self) -> None:
        """Set all models to evaluation mode."""
        for model in [self.router, self.small_model, self.large_model]:
            model.eval()

    def _process_model_outputs(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> ModelResults:
        """Process model outputs and create results structure."""
        if mask is not None:
            if not mask.any():
                return ModelResults(None, None, None)
            probs = probs[mask]
            labels = labels[mask]
            
        preds = (probs > 0.5).long()
        return ModelResults(probs, preds, labels)

    def _process_large_model_inputs(
        self,
        prompts: List[str],
        responses: Optional[List[str]],
        mask: torch.Tensor
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Process inputs for large model based on routing mask.
        Optimized to minimize CPU-GPU transfers.
        """
        # Get indices in single GPU->CPU transfer
        
        # Use indices to select from lists
        mask_np = mask.cpu().numpy()
        masked_prompts = np.array(prompts)[mask_np].tolist()
        masked_responses = np.array(responses)[mask_np].tolist() if responses is not None else None
        
        return masked_prompts, masked_responses

    def _prepare_final_results(
        self,
        small_results: ModelResults,
        large_results: ModelResults,
        num_large: int
    ) -> Dict[str, Any]:
        """
        Prepare final results dictionary with optimized tensor operations.
        Moves tensors to CPU only at the final step.
        """
        final_results = {"num_large": num_large}
        
        for key in vars(small_results).keys():
            small_val = getattr(small_results, key)
            large_val = getattr(large_results, key)
            
            if small_val is not None or large_val is not None:
                tensors_to_cat = []
                if small_val is not None:
                    tensors_to_cat.append(small_val)
                if large_val is not None:
                    tensors_to_cat.append(large_val)
                    
                final_results[key] = torch.cat(tensors_to_cat).cpu()
                
        return final_results

    @torch.no_grad()
    def forward(
        self,
        prompts: List[str],
        responses: Optional[List[str]],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Forward pass through the safety guard system.
        Optimized for minimal CPU-GPU transfers and efficient tensor operations.
        """
        # Move labels to GPU once
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        # Small model processing
        small_output = self.small_model.forward(
            prompts, responses, layer_idx=self.layer_idx)
        small_features = small_output["features"].float()
        small_probs = small_output["unsafe_logprob"].exp().float()

        # Routing decision (keep on GPU)
        routing_probs = torch.sigmoid(self.router(small_features))
        large_mask = routing_probs > 0.5
        small_mask = ~large_mask
        num_large = large_mask.sum().item()

        # Process small model results
        small_results = self._process_model_outputs(small_probs, labels, small_mask)
        
        # Process large model if needed
        if num_large > 0:
            large_prompts, large_responses = self._process_large_model_inputs(
                prompts, responses, large_mask)
            large_probs = self.large_model.compute(
                large_prompts, large_responses).exp().float()
            large_results = self._process_model_outputs(large_probs, labels[large_mask])
        else:
            large_results = ModelResults(None, None, None)

        # Prepare final results
        return self._prepare_final_results(small_results, large_results, num_large)

    @torch.no_grad()
    def get_decision(self, prompts, responses):
        self.eval()
        output = self.small_model.forward(
            prompts, responses, layer_idx=self.layer_idx)
        small_features = output["features"].float()

        routing_probs = torch.sigmoid(self.router(small_features)).cpu()
        large_mask = routing_probs > 0.5

        return large_mask.long(), routing_probs




class Oracle(nn.Module):
    def __init__(self, small_model, large_model) -> None:
        super().__init__()
        self.device = small_model.device
        self.small_model = small_model
        self.large_model = large_model

    def eval(self):
        self.small_model.eval()
        self.large_model.eval()

    
    
    def _process_model_outputs(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> ModelResults:
        """Process model outputs and create results structure."""
        if mask is not None:
            if not mask.any():
                return ModelResults(None, None, None)
            probs = probs[mask]
            labels = labels[mask]
            
        preds = (probs > 0.5).long()
        return ModelResults(probs, preds, labels)

    def _process_large_model_inputs(
        self,
        prompts: List[str],
        responses: Optional[List[str]],
        mask: torch.Tensor
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Process inputs for large model based on routing mask.
        Optimized to minimize CPU-GPU transfers.
        """
        # Get indices in single GPU->CPU transfer
        
        # Use indices to select from lists
        masked_prompts = [p for p, m in zip(prompts, mask) if m]
        
        if responses is None:
            return masked_prompts, None
            
        masked_responses = [r for r,m in zip(responses, mask) if m]
        return masked_prompts, masked_responses

    def _prepare_final_results(
        self,
        small_results: ModelResults,
        large_results: ModelResults,
        num_large: int
    ) -> Dict[str, Any]:
        """
        Prepare final results dictionary with optimized tensor operations.
        Moves tensors to CPU only at the final step.
        """
        final_results = {"num_large": num_large}
        
        for key in vars(small_results).keys():
            small_val = getattr(small_results, key)
            large_val = getattr(large_results, key)
            
            if small_val is not None or large_val is not None:
                tensors_to_cat = []
                if small_val is not None:
                    tensors_to_cat.append(small_val)
                if large_val is not None:
                    tensors_to_cat.append(large_val)
                    
                final_results[key] = torch.cat(tensors_to_cat).cpu()
                
        return final_results
    
    
    @torch.no_grad()
    def forward(self, prompts, responses, labels):
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        # forward prompts with small and large model
        small_probs = self.small_model.compute(
            prompts, responses).exp().float()
        
        small_preds = (small_probs > 0.5).long()
        
        large_probs = self.large_model.compute(
            prompts, responses).exp().float()
        large_preds = (large_probs > 0.5).long()
        
        large_mask = torch.logical_and(
            small_preds != labels, large_preds == labels)
        small_mask = ~large_mask
        num_large = large_mask.sum().item()

        if small_mask.any():
            small_probs = small_probs[small_mask].cpu()
            small_preds = small_preds[small_mask].cpu()    
            small_labels = labels[small_mask].cpu()

            small_results = {
                "probs": small_probs,
                "preds": small_preds,
                "final_labels": small_labels,
            }

        else:
            small_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
            }
        if num_large > 0:
            large_probs = large_probs[large_mask].cpu()
            large_preds = large_preds[large_mask].cpu()
            large_labels = labels[large_mask].cpu()

            large_results = {
                "probs": large_probs,
                "preds": large_preds,
                "final_labels": large_labels,
            }
        else:
            large_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
                "num_seen_responses": 0,
            }
        final_results = dict()
        final_results["num_large"] = num_large
        for k in small_results.keys():
            agg = [small_results[k], large_results[k]]
            final_results[k] = torch.cat([x for x in agg if x is not None])

        return final_results


class RandomModel(nn.Module):
    def __init__(self, small_model, large_model) -> None:
        super().__init__()
        self.device = small_model.device
        self.small_model = small_model
        self.large_model = large_model

    def eval(self):
        self.small_model.eval()
        self.large_model.eval()

    @torch.no_grad()
    def forward(self, prompts, responses, labels):
        labels = torch.LongTensor(labels)

        # forward prompts with small and large model
        small_probs = self.small_model.compute(
            prompts, responses).exp().float().cpu()
        small_preds = (small_probs > 0.5).long()

        large_mask = torch.rand(len(prompts)) < 0.5
        small_mask = ~large_mask
        num_large = torch.sum(large_mask).item()
        if small_mask.any():
            small_results = {
                "probs": small_probs[small_mask],
                "preds": small_preds[small_mask],
                "final_labels": labels[small_mask],
            }

        else:
            small_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
            }
        if num_large > 0:
            large_prompts = [p for p, m in zip(prompts, large_mask) if m]
            if responses is not None:
                large_responses = [r for r, m in zip(responses, large_mask) if m]
            else:
                large_responses = None
            large_probs = self.large_model.compute(
                large_prompts, large_responses).exp().float().cpu()
            large_preds = (large_probs > 0.5).long()

            large_results = {
                "probs": large_probs,
                "preds": large_preds,
                "final_labels": labels[large_mask],
            }

        else:
            large_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
            }
        final_results = dict()
        final_results["num_large"] = num_large
        for k in small_results.keys():
            agg = [small_results[k], large_results[k]]
            final_results[k] = torch.cat([x for x in agg if x is not None])

        return final_results

    def get_decision(self, prompts, responses):
        large_mask = torch.rand(len(prompts)) < 0.5
        probs = torch.ones_like(large_mask) * 0.5
        return large_mask.long(), probs


class CalibrationModel(nn.Module):
    def __init__(self, small_model, large_model) -> None:
        super().__init__()
        self.device = small_model.device
        self.small_model = small_model
        self.large_model = large_model

    def eval(self):
        self.small_model.eval()
        self.large_model.eval()

    def entropy(self, prob):
        ret = -prob * torch.log2(prob) - (1 - prob) * torch.log2(1 - prob)
        return ret

    def calibrate_py(self, p_y, p_cf, mode="diagonal"):
        """
        Calibrate p_y with a 1D p_cf shared across all batches.
        Args:
            p_y (torch.Tensor): A 2D tensor of shape (batch_size, num_classes) containing probabilities.
            p_cf (torch.Tensor): A 1D tensor of shape (num_classes,) for calibration factors shared across the batch.
            mode (str): Either 'diagonal' or 'identity', determining the calibration strategy.

        Returns:
            torch.Tensor: Calibrated probabilities, shape (batch_size, num_classes).
        """
        batch_size, num_classes = p_y.shape

        if p_cf is None:
            # Do not calibrate
            W = torch.eye(num_classes, device=p_y.device).expand(
                batch_size, -1, -1
            )  # Shape: (batch_size, num_classes, num_classes)
            b = torch.zeros(
                batch_size, num_classes, 1, device=p_y.device
            )  # Shape: (batch_size, num_classes, 1)
        else:
            # Calibrate
            if mode == "diagonal":
                W = torch.diag(1.0 / p_cf).to(
                    p_y.device
                )  # Shape: (num_classes, num_classes)
                W = W.expand(
                    batch_size, -1, -1
                )  # Shape: (batch_size, num_classes, num_classes)
                b = torch.zeros(
                    batch_size, num_classes, 1, device=p_y.device
                )  # Shape: (batch_size, num_classes, 1)
                cal_py = torch.matmul(W, p_y.unsqueeze(-1)) + b
            elif mode == "identity":
                W = torch.eye(
                    num_classes, device=p_y.device
                )  # Shape: (num_classes, num_classes)
                W = W.expand(
                    batch_size, -1, -1
                )  # Shape: (batch_size, num_classes, num_classes)
                b = (
                    -torch.log(p_cf).unsqueeze(0).expand(batch_size,
                                                         num_classes, 1)
                )  # Shape: (batch_size, num_classes, 1)
                cal_py = torch.matmul(W, torch.log(
                    p_y + 1e-6).unsqueeze(-1)) + b
                cal_py = torch.exp(cal_py)
            else:
                raise ValueError(
                    "Unsupported mode. Choose either 'diagonal' or 'identity'."
                )

        cal_py = cal_py.squeeze(-1)  # Remove the last dimension
        # Normalize across classes
        cal_py = cal_py / cal_py.sum(dim=-1, keepdim=True)

        return cal_py

    @torch.no_grad()
    def forward(
        self,
        prompts,
        responses,
        labels,
        batch_prob=None,
        ts=1.0,
    ):
        labels = torch.LongTensor(labels)

        # forward prompts with small and large model
        small_probs = (
            self.small_model.compute(
                prompts, responses, ts=ts).exp().float().cpu()
        )
        
        if batch_prob is not None:
            p = torch.stack([1 - small_probs, small_probs], dim=1)
            p = self.calibrate_py(p, batch_prob)[:, 1]

        else:
            p = small_probs
 
        large_mask = self.entropy(p) > 0.5
        small_mask = ~large_mask
        if small_mask.any():  # small model
            small_labels = labels[small_mask]
            small_probs = small_probs[small_mask]
            small_preds = (small_probs > 0.5).long()
            
            small_results = {
                "probs": small_probs,
                "preds": small_preds,
                "final_labels": small_labels,
            }

        else:
            small_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
            }

        if large_mask.any():  # large model
            large_prompts = [p for p, m in zip(prompts, large_mask) if m]
            large_labels = labels[large_mask]

            if responses is not None:
                large_responses = [r for r, m in zip(responses, large_mask) if m]
            else:
                large_responses = None
            large_probs = self.large_model.compute(
                large_prompts, large_responses).exp().float().cpu()
            large_preds = (large_probs > 0.5).long()

            large_results = {
                "probs": large_probs,
                "preds": large_preds,
                "final_labels": large_labels,
            }

        else:
            large_results = {
                "probs": None,
                "preds": None,
                "final_labels": None,
            }

        num_large = large_mask.sum().item()
        final_results = dict()
        final_results["num_large"] = num_large
        for k in small_results.keys():
            agg = [small_results[k], large_results[k]]
            final_results[k] = torch.cat([x for x in agg if x is not None])

        return final_results

    @torch.no_grad()
    def get_decision(self, prompts, responses, batch_prob=None, ts=1.0):
        # forward prompts with small and large model
        small_probs = (
            self.small_model.compute(
                prompts, responses, ts=ts).exp().float().cpu()
        )
        if batch_prob is not None:
            p = torch.stack([1 - small_probs, small_probs], dim=1)
            p = self.calibrate_py(p, batch_prob)[:, 1]

        else:
            p = small_probs

        large_mask = self.entropy(p) > 0.5

        return large_mask.long(), p
