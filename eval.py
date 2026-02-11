import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.__main__ import cli_evaluate

from transformers import AutoTokenizer

from hnet import HNetConfig, HNetLM
from data.utils import process_byte_last

class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs, add_bos=False, add_eos=False, **kwargs):
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)

def seq2args(str_sequences) -> Tuple[torch.Tensor, torch.Tensor, int]:
    samples = [torch.tensor(bytearray(b), dtype=torch.long).pin_memory().to('cuda', non_blocking=True)
                for b in str_sequences]
    values = torch.cat(samples, dim=0)

    lengths = [len(s) for s in samples]
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long, device='cuda')
    offsets[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.long, device='cuda'), dim=0)

    max_seqlen = max(lengths) if lengths else 0
    return values, offsets, max_seqlen

@register_model("hnet")
class HNetEvalModel(LM):
    def __init__(
        self,
        config_path: str,
        pretrain_transformer_name: str,
        checkpoint_path: str,
        mode: str,
        device: str = 'cuda',
        max_length: int = 2048,
        custom_batch_size: int = 1,
        **kwargs
    ):
        super().__init__()

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length
        self.batch_size = custom_batch_size
        self.pretrain_transformer_name = pretrain_transformer_name
        self.mode = mode
        self.tokenizer = ByteTokenizer()
        self.pretrain_tokenizer = AutoTokenizer.from_pretrained(pretrain_transformer_name)
        
        self.config = HNetConfig.load_config(config_path)
        
        accelerator_kwargs = InitProcessGroupKwargs()
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        if self.accelerator.num_processes > 1:
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
            self._device = self.accelerator.device
            print(f"Accelerator DP. Rank={self._rank}, WorldSize={self._world_size}, Device={self._device}")
        else:
            self._rank = 0
            self._world_size = 1
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f"Single process. Rank={self._rank}, WorldSize={self._world_size}, Device={self._device}")
    
        self._load_model()
        self.model = self.model.eval()
        self.model = self.accelerator.prepare(self.model)
        
        print("Model loaded successfully!")
        
        print("HNet model loaded:")
        print(f"  - config: {config_path}")
        print(f"  - checkpoint: {checkpoint_path}")
        
    @property
    def tokenizer_name(self) -> str:
        return self.pretrain_transformer_name
    
    def _load_model(self):            
        self.model = HNetLM(self.config, pretrain_transformer_name=self.pretrain_transformer_name).to(torch.bfloat16).to(self.device)
            
        if self.checkpoint_path:
            print(f"Loading checkpoint: {self.checkpoint_path}")
            model_ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
                
            for k, v in model_ckpt.items():
                new_key = k.replace("._orig_mod", "")
                    
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict, strict=True)

            print("Checkpoint loaded")
    
    def tok_encode(self, string: str, add_bos: bool = False) -> List[int]:
        """Encode text into a token sequence."""
        return self.tokenizer.encode([string], add_bos=add_bos)[0]["input_ids"]
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode a token sequence into text."""
        return self.tokenizer.decode(tokens)
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:        
        if self.mode == "stage1":
            return self.loglikelihood_stage1(requests)
        elif self.mode == "stage2":
            return self.loglikelihood_stage2(requests)
        else:
            return self.loglikelihood_stage2(requests)
    
    def pretrain_tok_encode(self, string: str, add_bos: bool = False) -> List[int]:
        """Encode text into a token sequence."""
        return self.pretrain_tokenizer(string, add_special_tokens=add_bos)["input_ids"]
    
    def pretrain_tok_decode(self, tokens: List[int]) -> str:
        """Decode a token sequence into text."""
        return self.pretrain_tokenizer.decode(tokens)
    
    def loglikelihood_stage1(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        
        disable_tqdm = (self.rank != 0)
        desc = f"Evaluating (Rank {self.rank})" if self.rank != 0 else "Evaluating"

        for req in tqdm(requests, desc=desc, disable=disable_tqdm):
            context = req.args[0]
            continuation = req.args[1]
                    
            context_tokens = self.tok_encode(context, add_bos=True)
            continuation_tokens = self.tok_encode(continuation, add_bos=False)
            
            context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0).to(self.device).long()
            continuation_tokens_tensor = torch.tensor(continuation_tokens).unsqueeze(0).to(self.device).long()
            
            full_tokens = torch.cat([context_tokens_tensor, continuation_tokens_tensor], dim=1)

            continuation_boundary_mask, _, _ = process_byte_last(
                continuation.encode("utf-8"), self.pretrain_tokenizer, return_ids=True, return_emb=False
            )
            
            full_seq_len = full_tokens.shape[1]
            full_offsets = torch.tensor(
                [0, full_seq_len],
                dtype=torch.int,
                device=self.device
            )
            
            with torch.inference_mode():
                _, bpred, _ = self.model.module(
                    "stage_1_inference",
                    iids_values=full_tokens,
                    iids_offsets=full_offsets,
                    max_seqlen=full_seq_len,
                )
            
            continuation_boundary_mask = torch.tensor(
                continuation_boundary_mask,
                dtype=torch.bool,
                device=self.device
            ).unsqueeze(0)
            context_seq_len = context_tokens_tensor.shape[1]
            context_bpred = bpred[:, :context_seq_len]
            bpred = torch.cat([context_bpred, continuation_boundary_mask], dim=1)
            
            pretrain_target_tokens = self.pretrain_tok_encode(continuation, add_bos=False)
            
            with torch.inference_mode():
                full_seq_len = full_tokens.shape[1]
                full_offsets = torch.tensor(
                    [0, full_seq_len],
                    dtype=torch.int,
                    device=self.device
                )
                token_mask = torch.ones(
                    (1, full_seq_len),
                    device=self.device,
                    dtype=torch.bool
                )
                
                logits, _, _ = self.model.module(
                    "stage_1_inference",
                    iids_values=full_tokens,
                    iids_offsets=full_offsets,
                    max_seqlen=full_seq_len,
                    bpred=bpred,
                    emb=None,
                    bpred_mask=token_mask
                )
                
                logits = logits[:, :-1, :]
                start_idx = logits.shape[1] - len(pretrain_target_tokens)
                cont_logits = logits[0, start_idx:]
                
                if cont_logits.shape[0] == 0:
                    ll = -99.0
                    is_greedy = False
                    if self.rank == 0: # Only warn on Rank 0.
                        print("Warning, continuation len is 0!")
                    results.append((ll, is_greedy))
                    continue

                log_probs = F.log_softmax(cont_logits, dim=-1)
                
                ll = 0.0
                is_greedy = True

                for i, target_token in enumerate(pretrain_target_tokens):
                    if i < cont_logits.shape[0]:
                        token_ll = log_probs[i, target_token].item()
                        ll += token_ll
                        
                        greedy_token = cont_logits[i].argmax().item()
                        if target_token != greedy_token:
                            is_greedy = False
            
            results.append((ll, is_greedy))
        return results
    
    def loglikelihood_stage2(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Calculate log-likelihood for each request using batched byte-level inference."""
        results = []
        if not requests:
            return results

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size),
            desc="Evaluating loglikelihood"
        ):
            batch_requests = requests[batch_start:batch_start + self.batch_size]
            
            batch_sequences = []
            context_lengths = []
            continuation_token_ids = []

            for req in batch_requests:
                context = req.args[0]
                continuation = req.args[1]

                context_tokens = self.tok_encode(context, add_bos=True)
                continuation_tokens = self.tok_encode(continuation, add_bos=False)

                context_lengths.append(len(context_tokens))
                continuation_token_ids.append(continuation_tokens.tolist())

                full_bytes = b'\xfe' + (context + continuation).encode("utf-8")
                batch_sequences.append(full_bytes)

            iids_values, iids_offsets, max_seqlen = seq2args(batch_sequences)

            with torch.inference_mode():
                logits, _, _ = self.model(
                    "byte_inference",
                    iids_values=iids_values,
                    iids_offsets=iids_offsets,
                    max_seqlen=max_seqlen,
                )

            offsets = iids_offsets.tolist()

            for idx, req in enumerate(batch_requests):
                seq_start = offsets[idx]
                seq_end = offsets[idx + 1]

                start_idx = seq_start + context_lengths[idx] - 1
                end_idx = seq_end - 1
                cont_logits = logits[start_idx:end_idx].clone()
                
                part1 = cont_logits[:, :256]
                part2 = cont_logits[:, 256:]

                cont_logits[:, :256] = torch.maximum(part1, part2)

                if cont_logits.shape[0] == 0:
                    results.append((-99.0, False))
                    continue

                continuation_tokens = continuation_token_ids[idx]
                target_len = min(len(continuation_tokens), cont_logits.shape[0])
                logits_slice = cont_logits[:target_len]

                target_tensor = torch.tensor(
                    continuation_tokens[:target_len],
                    dtype=torch.long,
                    device=logits.device
                )

                log_probs = F.log_softmax(logits_slice, dim=-1)
                token_lls = log_probs.gather(1, target_tensor.unsqueeze(-1)).squeeze(-1)
                ll = token_lls.sum().item()

                greedy_tokens = logits_slice.argmax(dim=-1)
                is_greedy = bool(torch.equal(greedy_tokens, target_tensor))

                results.append((ll, is_greedy))

        return results
    
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

if __name__ == "__main__":
    cli_evaluate()
