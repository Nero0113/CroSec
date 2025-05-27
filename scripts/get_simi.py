import os

import numpy as np, scipy.sparse as sp, json, time
from tqdm import tqdm
from transformers import AutoTokenizer


src_model = '/home/public_space/yanmeng/lidong/models/Qwen2.5-Coder-0.5B-Instruct'
trg_model = '/home/public_space/liuchao/shushanfu/LMOps/checkpoints/codegen-6B'
emb_src_file = '/home/public_space/yanmeng/lidong/code/one4all/try_EVA_like/embeddings_codegen/emb_src.npy'   # 形状 [V_src, d]
emb_trg_file = '/home/public_space/yanmeng/lidong/code/one4all/try_EVA_like/embeddings_codegen/emb_trg.npy'   # 形状 [V_trg, d]
top_k      = 3
sim_thres  = 0.5               # CSLS < 0.5 视为无映射
output_dir = './mapping_codegen/'


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, time, argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from transformers import AutoTokenizer



def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main(args):
    t0 = time.time()
    print(">>> Loading tokenizers …")
    tok_src = AutoTokenizer.from_pretrained(args.src_model, trust_remote_code=True)
    tok_trg = AutoTokenizer.from_pretrained(args.trg_model, trust_remote_code=True)
    V_src, V_trg = len(tok_src), len(tok_trg)
    print(f"    src_vocab={V_src}  trg_vocab={V_trg}")

    print(">>> Loading aligned embeddings (.npy) …")
    src_emb = np.load(args.src_emb).astype(np.float32)   # [V_src, d]
    trg_emb = np.load(args.trg_emb).astype(np.float32)   # [V_trg, d]
    print(src_emb.shape[0])
    assert src_emb.shape[0] == V_src, "src_emb 行数 ≠ src_vocab_size"
    assert trg_emb.shape[0] == V_trg, "trg_emb 行数 ≠ trg_vocab_size"
    assert src_emb.shape[1] == trg_emb.shape[1], "两端向量维度不一致"

    d = src_emb.shape[1]
    print(f"    embedding dim = {d}")


    src_emb = l2_normalize(src_emb)
    trg_emb = l2_normalize(trg_emb)


    trg_tok2id = {t: i for i, t in enumerate(tok_trg.convert_ids_to_tokens(range(V_trg)))}
    shared_src, shared_trg = [], []
    for s_id, tok in enumerate(tok_src.convert_ids_to_tokens(range(V_src))):
        if tok in trg_tok2id:
            shared_src.append(s_id)
            shared_trg.append(trg_tok2id[tok])
    shared_src = np.array(shared_src, dtype=np.int32)
    shared_trg = np.array(shared_trg, dtype=np.int32)
    print(f"    common tokens = {len(shared_src)}")


    print(">>> Computing CSLS backward KNN …")
    top_k = args.top_k
    sim_trg_src = trg_emb @ src_emb.T                     # [V_trg, V_src]
    trg_knn = np.partition(sim_trg_src, -top_k, axis=1)[:, -top_k:].mean(1)  # [V_trg]


    rows, cols, data = [], [], []            # 用于构造 COO
    batch = 1024
    remain_src = np.setdiff1d(np.arange(V_src), shared_src, assume_unique=True)

    print(">>> CSLS forward pass …")
    for i in tqdm(range(0, len(remain_src), batch)):
        idx = remain_src[i:i+batch]
        sim = src_emb[idx] @ trg_emb.T                      # [b, V_trg]
        src_knn = np.partition(sim, -top_k, axis=1)[:, -top_k:].mean(1)  # [b]
        csls = 2 * sim - src_knn[:, None] - trg_knn[None, :]

        top_idx = np.argmax(csls, axis=1)                   # [b]
        top_val = csls[np.arange(len(idx)), top_idx]

        mask = top_val >= args.thres
        rows.extend(idx[mask])
        cols.extend(top_idx[mask])
        data.extend(top_val[mask].astype(np.float32))


    rows.extend(shared_src.tolist())
    cols.extend(shared_trg.tolist())
    data.extend(np.ones_like(shared_src, dtype=np.float32))


    src2trg = sp.coo_matrix((data, (rows, cols)), shape=(V_src, V_trg), dtype=np.float32).tocsr()
    print(f"    sparse matrix: shape={src2trg.shape}  nnz={src2trg.nnz}")


    token_map = [-1] * V_src
    for s, t in zip(rows, cols):
        if token_map[s] == -1:
            token_map[s] = int(t)


    os.makedirs(args.out_dir, exist_ok=True)
    sp.save_npz(os.path.join(args.out_dir, "src2trg_full.npz"), src2trg)
    json.dump(token_map, open(os.path.join(args.out_dir, "token_map_full.json"), "w"))
    meta = {
        "src_model": args.src_model,
        "trg_model": args.trg_model,
        "dim": int(d),
        "top_k": top_k,
        "threshold": args.thres,
        "common_tokens": int(len(shared_src)),
        "coverage": float(np.mean(np.array(token_map) != -1))
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"✓ Done!  elapsed {time.time()-t0:.1f}s")
    print(f"   coverage = {meta['coverage']*100:.1f}%   files saved to {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_model", default=src_model)
    ap.add_argument("--trg_model", default=trg_model)
    ap.add_argument("--src_emb",   default=emb_src_file, help=".npy from src (aligned)")
    ap.add_argument("--trg_emb",   default=emb_trg_file, help=".npy from trg (aligned)")
    ap.add_argument("--out_dir",   default=output_dir)
    ap.add_argument("--top_k",     type=int, default=3)
    ap.add_argument("--thres",     type=float, default=0.5,
                    help="CSLS threshold; < thres → no mapping (-1)")
    args = ap.parse_args()
    main(args)
