import torch

def create_byte_to_char_map(text: str) -> dict:
    char_to_byte_map = {}
    byte_offset = 0
    for i, char in enumerate(text):
        char_to_byte_map[i] = byte_offset
        byte_offset += len(char.encode('utf-8'))
    # Add an entry for the position after the last character
    char_to_byte_map[len(text)] = byte_offset

    return char_to_byte_map

def process_byte_last(byte_seq: bytes, tokenizer, bos_embedding=None, model=None,
                      add_bos=False, return_ids=False, return_emb=True):
    if return_emb:
        assert bos_embedding is not None and model is not None

    # Tokenize
    text = byte_seq.decode('utf-8', errors='ignore')
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_offsets, token_ids_list = enc['offset_mapping'], enc['input_ids']
    char_to_byte_map = create_byte_to_char_map(text)

    # Initialize outputs
    num_positions = len(byte_seq) + (1 if add_bos else 0)
    boundary_mask = [False] * num_positions
    token_input_ids = [0] * num_positions if return_ids else None

    # Get embeddings if needed
    embeddings = None
    if return_emb:
        device = bos_embedding.device
        embeddings = torch.zeros(num_positions, bos_embedding.shape[0],
                                dtype=bos_embedding.dtype, device=device)
        with torch.no_grad():
            token_ids = torch.as_tensor(token_ids_list, dtype=torch.long, device=device)
            tok_emb = model.backbone.main_network.main_network.model.embed_tokens(token_ids)

    # Process each token
    for token_idx, (char_start, char_end) in enumerate(token_offsets):
        if char_end == 0:
            continue

        last_byte_idx = char_to_byte_map[char_end] - 1 + (1 if add_bos else 0)
        start_byte_idx = char_to_byte_map[char_start] + (1 if add_bos else 0)

        if return_emb:
            embeddings[last_byte_idx] = tok_emb[token_idx]
        boundary_mask[last_byte_idx] = True

        if return_ids:
            for i in range(start_byte_idx, last_byte_idx + 1):
                if i < num_positions:
                    token_input_ids[i] = token_ids_list[token_idx]

    # Add BOS
    if add_bos:
        boundary_mask[0] = True
        if return_emb:
            embeddings[0] = bos_embedding
        if return_ids:
            token_input_ids[0] = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id

    # Return
    emb_list = [e for e in embeddings] if return_emb else []
    return boundary_mask, emb_list, token_input_ids


def process_byte_last_batch(byte_seqs, tokenizer, bos_embedding=None, model=None,
                            add_bos_flags=None, return_ids=False, return_emb=True):
    if return_emb:
        assert bos_embedding is not None and model is not None

    if add_bos_flags is None:
        add_bos_flags = [False] * len(byte_seqs)
    elif isinstance(add_bos_flags, bool):
        add_bos_flags = [add_bos_flags] * len(byte_seqs)

    if len(add_bos_flags) != len(byte_seqs):
        raise ValueError("add_bos_flags length must match byte_seqs length")

    texts = [byte_seq.decode('utf-8', errors='ignore') for byte_seq in byte_seqs]
    enc = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
    offsets_batch = enc['offset_mapping']
    token_ids_batch = enc['input_ids']

    tok_emb_batch = None
    if return_emb:
        device = bos_embedding.device
        flat_token_ids = [tid for seq in token_ids_batch for tid in seq]
        if flat_token_ids:
            with torch.no_grad():
                token_ids = torch.as_tensor(flat_token_ids, dtype=torch.long, device=device)
                flat_emb = model.backbone.main_network.main_network.model.embed_tokens(token_ids)
        else:
            flat_emb = torch.zeros((0, bos_embedding.shape[0]),
                                   dtype=bos_embedding.dtype, device=device)

        tok_emb_batch = []
        idx = 0
        for seq_ids in token_ids_batch:
            length = len(seq_ids)
            tok_emb_batch.append(flat_emb[idx:idx + length])
            idx += length

    boundary_masks = []
    emb_lists = []
    token_input_ids_list = []

    for seq_idx, (text, byte_seq, offsets, token_ids, add_bos) in enumerate(
        zip(texts, byte_seqs, offsets_batch, token_ids_batch, add_bos_flags)
    ):
        char_to_byte_map = create_byte_to_char_map(text)
        num_positions = len(byte_seq) + (1 if add_bos else 0)
        boundary_mask = [False] * num_positions
        token_input_ids = [0] * num_positions if return_ids else None

        embeddings = None
        if return_emb:
            embeddings = torch.zeros(num_positions, bos_embedding.shape[0],
                                     dtype=bos_embedding.dtype, device=bos_embedding.device)
            seq_tok_emb = tok_emb_batch[seq_idx]

        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_end == 0:
                continue

            last_byte_idx = char_to_byte_map[char_end] - 1 + (1 if add_bos else 0)
            start_byte_idx = char_to_byte_map[char_start] + (1 if add_bos else 0)

            if return_emb:
                embeddings[last_byte_idx] = seq_tok_emb[token_idx]
            boundary_mask[last_byte_idx] = True

            if return_ids:
                for i in range(start_byte_idx, last_byte_idx + 1):
                    if i < num_positions:
                        token_input_ids[i] = token_ids[token_idx]

        if add_bos:
            boundary_mask[0] = True
            if return_emb:
                embeddings[0] = bos_embedding
            if return_ids:
                token_input_ids[0] = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id

        emb_list = [e for e in embeddings] if return_emb else []
        boundary_masks.append(boundary_mask)
        emb_lists.append(emb_list)
        token_input_ids_list.append(token_input_ids)

    return boundary_masks, emb_lists, token_input_ids_list
