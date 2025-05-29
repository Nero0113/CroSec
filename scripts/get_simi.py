import os, json, time, argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from transformers import AutoTokenizer


def get_args():
    """
    Parse command-line arguments.
    Returns:
        args: Parsed arguments.
    """
    # Find trg_model to sec_model token_map
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec_model", default='/home/yanmeng/zhangjingrui/models/Qwen2.5-Coder-7B')
    parser.add_argument("--trg_model", default='/home/yanmeng/zhangjingrui/models/deepseek-coder-6.7b-base')
    parser.add_argument("--sec_emb",   default='/home/yanmeng/zhangjingrui/projects/CroSec/embeddings_d2q/emb_sec.npy', help=".npy from sec (aligned)")
    parser.add_argument("--trg_emb",   default='/home/yanmeng/zhangjingrui/projects/CroSec/embeddings_d2q/emb_trg.npy', help=".npy from trg (aligned)")
    parser.add_argument("--out_dir",   default='./mapping_deepseek/')
    parser.add_argument("--top_k",     type=int, default=3)
    parser.add_argument("--thres",     type=float, default=0.5,help="CSLS threshold; < thres → no mapping (-1)")
    args = parser.parse_args()

    return args

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def main(args):
    t0 = time.time()
    print(">>> Loading tokenizers …")
    tok_sec = AutoTokenizer.from_pretrained(args.sec_model, trust_remote_code=True)
    tok_trg = AutoTokenizer.from_pretrained(args.trg_model, trust_remote_code=True)
    V_sec, V_trg = len(tok_sec), len(tok_trg)
    print(f"    sec_vocab={V_sec}  trg_vocab={V_trg}")

    print(">>> Loading aligned embeddings (.npy) …")
    sec_emb = np.load(args.sec_emb).astype(np.float32)   # [V_sec, d]
    trg_emb = np.load(args.trg_emb).astype(np.float32)   # [V_trg, d]
    print(sec_emb.shape[0])
    assert sec_emb.shape[0] == V_sec, "sec_emb 行数 ≠ sec_vocab_size"
    assert trg_emb.shape[0] == V_trg, "trg_emb 行数 ≠ trg_vocab_size"
    assert sec_emb.shape[1] == trg_emb.shape[1], "两端向量维度不一致"

    d = sec_emb.shape[1]
    print(f"    embedding dim = {d}")


    sec_emb = l2_normalize(sec_emb)
    trg_emb = l2_normalize(trg_emb)


    trg_tok2id = {t: i for i, t in enumerate(tok_trg.convert_ids_to_tokens(range(V_trg)))}
    shared_sec, shared_trg = [], []
    for s_id, tok in enumerate(tok_sec.convert_ids_to_tokens(range(V_sec))):
        if tok in trg_tok2id:
            shared_sec.append(s_id)
            shared_trg.append(trg_tok2id[tok])
    shared_sec = np.array(shared_sec, dtype=np.int32)
    shared_trg = np.array(shared_trg, dtype=np.int32)
    print(f"    common tokens = {len(shared_sec)}")


    print(">>> Computing CSLS backward KNN …")
    top_k = args.top_k
    sim_trg_sec = trg_emb @ sec_emb.T                     # [V_trg, V_sec]
    trg_knn = np.partition(sim_trg_sec, -top_k, axis=1)[:, -top_k:].mean(1)  # [V_trg]


    rows, cols, data = [], [], []            # 用于构造 COO
    batch = 1024
    remain_sec = np.setdiff1d(np.arange(V_sec), shared_sec, assume_unique=True)

    print(">>> CSLS forward pass …")
    for i in tqdm(range(0, len(remain_sec), batch)):
        idx = remain_sec[i:i+batch]
        sim = sec_emb[idx] @ trg_emb.T                      # [b, V_trg]
        sec_knn = np.partition(sim, -top_k, axis=1)[:, -top_k:].mean(1)  # [b]
        csls = 2 * sim - sec_knn[:, None] - trg_knn[None, :]

        top_idx = np.argmax(csls, axis=1)                   # [b]
        top_val = csls[np.arange(len(idx)), top_idx]

        mask = top_val >= args.thres
        rows.extend(idx[mask])
        cols.extend(top_idx[mask])
        data.extend(top_val[mask].astype(np.float32))


    rows.extend(shared_sec.tolist())
    cols.extend(shared_trg.tolist())
    data.extend(np.ones_like(shared_sec, dtype=np.float32))


    sec2trg = sp.coo_matrix((data, (rows, cols)), shape=(V_sec, V_trg), dtype=np.float32).tocsr()
    print(f"    sparse matrix: shape={sec2trg.shape}  nnz={sec2trg.nnz}")


    token_map = [-1] * V_sec
    for s, t in zip(rows, cols):
        if token_map[s] == -1:
            token_map[s] = int(t)


    os.makedirs(args.out_dir, exist_ok=True)
    sp.save_npz(os.path.join(args.out_dir, "sec2trg_full.npz"), sec2trg)
    json.dump(token_map, open(os.path.join(args.out_dir, "token_map_full.json"), "w"))
    meta = {
        "sec_model": args.sec_model,
        "trg_model": args.trg_model,
        "dim": int(d),
        "top_k": top_k,
        "threshold": args.thres,
        "common_tokens": int(len(shared_sec)),
        "coverage": float(np.mean(np.array(token_map) != -1))
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"✓ Done!  elapsed {time.time()-t0:.1f}s")
    print(f"   coverage = {meta['coverage']*100:.1f}%   files saved to {args.out_dir}")


if __name__ == "__main__":
    args = get_args()
    main(args)
