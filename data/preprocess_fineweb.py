import argparse
import os
from pathlib import Path
from hashlib import sha256
from queue import Queue
from threading import Thread
from typing import Optional

from pyarrow import dataset as ds
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import torch

from utils import process_byte_last

# ===========================================
OUTPUT_DIR = Path("../data/fineweb")
TEMP_DOWNLOAD_DIR = Path("./temp_fineweb_download")
MAX_DOC_BYTES = 4 * 131072
SEQ_LEN = 8192
BOS_BYTE = b'\xfe'
# ===========================================

def _default_consumer_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError, OSError):
        return os.cpu_count() or 4

def grab_fineweb():
    DATA_DIR = Path(snapshot_download(
        "HuggingFaceFW/fineweb", repo_type="dataset", allow_patterns="sample/10BT/*"
    )) / 'sample/10BT'
    dataset = ds.dataset(DATA_DIR, format="parquet")
    return dataset.to_batches(columns=["text"], batch_size=4096)

def dump_one(txt: str) -> Path:
    """Write a file and return the path."""
    raw = txt.encode("utf-8")
    hash_name = sha256(raw).hexdigest()
    p = OUTPUT_DIR / str(len(raw))
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{hash_name}.txt"
    path.write_bytes(raw)
    return path

def dump_fineweb_to_disk(tokenizer_name: str, consumers: Optional[int] = None):
    if not tokenizer_name:
        raise ValueError("tokenizer_name is required to build precomputed metadata")
    if consumers is None:
        consumers = _default_consumer_count()
    consumers = max(consumers, 1)

    batches = grab_fineweb()
    
    qsize = consumers * 4
    
    def producer(q, pbar):
        current_offset = 0
        
        for batch in batches:
            texts = batch['text'].to_pylist()
            for s in texts:
                q.put(s)
            
            current_offset += len(texts)
            pbar.update(len(texts))
        
        for _ in range(consumers): 
            q.put(None)

    def consumer(q):
        local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        while True:
            text = q.get()
            if text is None:
                q.task_done()
                break
            
            path = dump_one(text)
            raw = text.encode("utf-8")

            boundary_mask, _, token_ids = process_byte_last(
                raw,
                local_tokenizer,
                add_bos=True,
                return_ids=True,
                return_emb=False
            )

            meta = {
                "tokenizer_name": tokenizer_name,
                "seq_len": SEQ_LEN,
                "bos": BOS_BYTE[0],
                "doc_len_bytes": len(raw),
                "boundary_mask": boundary_mask,
                "token_ids": token_ids,
            }
            torch.save(meta, path.with_suffix(".meta.pt"))
            q.task_done()
    
    q = Queue(maxsize=qsize)
    progress_bar = tqdm(desc="Processing", unit="doc")

    threads = [
        Thread(target=producer, args=(q, progress_bar)),
        *[Thread(target=consumer, args=(q,)) for _ in range(consumers)]
    ]
    for t in threads: t.start()
    
    q.join() 
    for t in threads: t.join()
    progress_bar.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer-name', type=str, required=True,
                    help='Tokenizer name or path (must match training pretrain-transformer-name)')
    ap.add_argument('--consumers', type=int, default=None, help='Number of consumer threads')
    args = ap.parse_args()
    dump_fineweb_to_disk(tokenizer_name=args.tokenizer_name, consumers=args.consumers)
