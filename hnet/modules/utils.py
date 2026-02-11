from dataclasses import asdict

import torch

def get_seq_idx(cu_seqlens, device=None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }

def packed_to_padded(packed, cu_seqlens, max_seqlen=None, pad_value=0, return_mask=True):
    """
    Vectorized conversion from packed (sum of lengths, ...) to padded (B, max_seqlen, ...).
    Returns (padded, mask) by default, where mask is True for valid positions.
    """
    device = packed.device
    cu_seqlens = torch.as_tensor(cu_seqlens, device=device)
    batch_size = cu_seqlens.numel() - 1
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if max_seqlen is None:
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0

    total = int(cu_seqlens[-1].item()) if cu_seqlens.numel() > 0 else 0
    out_shape = (batch_size, max_seqlen) + packed.shape[1:]
    padded = packed.new_zeros(out_shape)
    if pad_value != 0:
        padded.fill_(pad_value)

    if total == 0:
        if return_mask:
            mask = torch.zeros((batch_size, max_seqlen), device=device, dtype=torch.bool)
            return padded, mask
        return padded

    seq_idx = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)
    pos_idx = torch.arange(total, device=device) - cu_seqlens[seq_idx]

    padded[seq_idx, pos_idx] = packed
    if return_mask:
        mask = torch.zeros((batch_size, max_seqlen), device=device, dtype=torch.bool)
        mask[seq_idx, pos_idx] = True
        return padded, mask
    return padded