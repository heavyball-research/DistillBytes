import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from hnet import HNetConfig, HNetLM


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs: List[str], add_bos: bool = False, add_eos: bool = False):
        outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")
            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte_ids = np.array(bytearray(text_byte), dtype=self.dtype)
            outputs.append({"input_ids": text_byte_ids})
        return outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)


class HNetGenerator:
    def __init__(
        self,
        config_path: str,
        pretrain_transformer_name: str,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        self.config = HNetConfig.load_config(config_path)
        self.byte_tokenizer = ByteTokenizer()
        self.pretrain_tokenizer = AutoTokenizer.from_pretrained(pretrain_transformer_name)

        model = HNetLM(self.config, pretrain_transformer_name=pretrain_transformer_name)
        model = model.to(self.device)
        if self.dtype is not None:
            model = model.to(self.dtype)

        self.model = model
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[Warn] CUDA unavailable, fallback to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "none": None,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = OrderedDict()

        for key, value in ckpt.items():
            new_key = key.replace("._orig_mod", "")
            new_state_dict[new_key] = value

        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"[Warn] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[Warn] Unexpected keys: {len(unexpected_keys)}")

    def _sample_next_token_id(self, logits: torch.Tensor, do_sample: bool, temperature: float) -> int:
        if not do_sample:
            return torch.argmax(logits, dim=-1).item()

        temperature = max(temperature, 1e-5)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate_stage1(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        prompt_ids = self.byte_tokenizer.encode([prompt], add_bos=True)[0]["input_ids"]
        prompt_ids = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)

        seq_len = prompt_ids.shape[1]
        offsets = torch.tensor([0, seq_len], device=self.device)
        cache = self.model.backbone.allocate_inference_cache(batch_size=1, max_seqlen=seq_len + max_new_tokens)

        generated_text = ""
        eos_token_id = self.pretrain_tokenizer.eos_token_id

        with torch.inference_mode():
            logits, _, _, inference_params = self.model(
                "stage_1_inference",
                iids_values=prompt_ids,
                iids_offsets=offsets,
                max_seqlen=seq_len,
                inference_params=cache,
            )

            for _ in range(max_new_tokens):
                next_token_logits = logits[0, -1, :]
                next_token_id = self._sample_next_token_id(next_token_logits, do_sample=do_sample, temperature=temperature)

                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                next_piece = self.pretrain_tokenizer.decode([next_token_id], skip_special_tokens=False)
                generated_text += next_piece

                next_bytes = bytearray(next_piece.encode("utf-8"))
                if not next_bytes:
                    continue

                next_tensor = torch.tensor(next_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
                logits, inference_params = self.model.stage1_step(next_tensor, len(next_bytes), inference_params)

        return generated_text

    def generate_stage2(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        prompt_ids = self.byte_tokenizer.encode([prompt], add_bos=True)[0]["input_ids"]
        prompt_ids = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)

        seq_len = prompt_ids.shape[1]
        offsets = torch.tensor([0, seq_len], device=self.device)

        cache = self.model.backbone.allocate_inference_cache(batch_size=1, max_seqlen=seq_len + max_new_tokens)

        generated_bytes = []
        prefill_boundary = None

        with torch.inference_mode():
            logits, inference_params, prefill_boundary = self.model(
                "byte_inference",
                iids_values=prompt_ids,
                iids_offsets=offsets,
                max_seqlen=seq_len,
                prefill_boundary=prefill_boundary,
                inference_params=cache,
            )

            for _ in range(max_new_tokens):
                if logits.dim() == 3:
                    logits = logits.squeeze(0)

                next_logits = logits[-1, :]
                next_token_id = self._sample_next_token_id(next_logits, do_sample=do_sample, temperature=temperature)

                next_boundary = False
                if next_token_id >= 256:
                    next_token_id = next_token_id - 256
                    next_boundary = True

                if next_token_id == self.byte_tokenizer.bos_idx:
                    break

                generated_bytes.append(next_token_id)

                next_tensor = torch.tensor(next_token_id, dtype=torch.long, device=self.device).view(1, 1)
                logits, inference_params = self.model.stage2_step(next_tensor, next_boundary, inference_params)

        return self.byte_tokenizer.decode(generated_bytes, errors="ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Example generation script for HNet")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--pretrain-transformer-name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="stage2", choices=["stage1", "stage2"])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    generator = HNetGenerator(
        config_path=args.config_path,
        pretrain_transformer_name=args.pretrain_transformer_name,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    if args.mode == "stage1":
        output = generator.generate_stage1(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
    else:
        output = generator.generate_stage2(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )

    print(f"[Prompt]\n{args.prompt}")
    print(f"[Output]\n{output}")


if __name__ == "__main__":
    main()
