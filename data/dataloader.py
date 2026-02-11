from pathlib import Path
import random
import torch

from .utils import process_byte_last

OUTPUT_DIR = Path("../../data/fineweb-edu/350BT")

def iter_fineweb_random(output_dir: Path = OUTPUT_DIR):
    """Iterate files randomly from all folders."""
    folders = [f for f in output_dir.iterdir() if f.is_dir()]
    folder_iters = {folder: iter(folder.iterdir()) for folder in folders}

    while folder_iters:
        folder = random.choice(list(folder_iters.keys()))
        try:
            path = next(folder_iters[folder])
            if path.suffix == ".txt":
                yield path
        except StopIteration:
            del folder_iters[folder]


def seqlen_8192_fineweb(msl, seq_len, tokenizer, bos=b'\xfe', model=None,
                        bos_emb=None, return_ids=False, return_emb=True,
                        output_dir=None, rank=0, world_size=1, use_precomputed=False, packed_mode=True,
                        tokenizer_name=None):
    """Process fineweb data with random sampling and fixed sequence length."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    # Buffers for accumulating data
    str_buf, boundary_buf, emb_buf, token_ids_buf = bytearray(), [], [], []

    # Batch sequences
    seqs, boundary_seqs, emb_seqs, token_id_seqs = [], [], [], []
    
    if packed_mode:
        batch_size = 1
        seq_len = msl
    else:
        batch_size = msl // seq_len

    for i, path in enumerate(iter_fineweb_random(output_dir)):
        if i % world_size != rank:
            continue

        # Read and process document
        content = path.read_bytes()

        boundary = embedding = token_input_ids = None
        if use_precomputed and not return_emb:
            meta_path = path.with_suffix(".meta.pt")
            if meta_path.exists():
                meta = torch.load(meta_path, map_location="cpu")
                assert tokenizer_name and meta.get("tokenizer_name") == tokenizer_name

                boundary = meta.get("boundary_mask")
                token_input_ids = meta.get("token_ids")
                embedding = []

        if boundary is None:
            boundary, embedding, token_input_ids = process_byte_last(
                content, tokenizer, bos_emb, model,
                add_bos=True, return_ids=return_ids, return_emb=return_emb
            )

        str_buf.extend(bos + content)
        boundary_buf.extend(boundary)
        emb_buf.extend(embedding)
        token_ids_buf.extend(token_input_ids)

        while len(str_buf) >= seq_len: 
            pos = seq_len
            
            seqs.append(bytes(str_buf[:pos]))
            boundary_seqs.append(boundary_buf[:pos])
            emb_seqs.append(emb_buf[:pos])
            token_id_seqs.append(token_ids_buf[:pos])

            str_buf = str_buf[pos:]
            boundary_buf = boundary_buf[pos:]
            emb_buf = emb_buf[pos:]
            token_ids_buf = token_ids_buf[pos:]

            if len(seqs) >= batch_size:
                flat_token_ids = [id for seq in token_id_seqs for id in seq]
                yield seqs, boundary_seqs, emb_seqs, flat_token_ids
                seqs, boundary_seqs, emb_seqs, token_id_seqs = [], [], [], []
