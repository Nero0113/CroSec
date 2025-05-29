import os, json, argparse, time, warnings
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm


def l2_normalize(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def get_args():
    """
    Parse command-line arguments.
    Returns:
        args: Parsed arguments.
    """
    # Find trg_model to sec_model token_map
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec_model", default='/home/yanmeng/zhangjingrui/models/Qwen2.5-Coder-7B',
                        help="HF Hub name or local path of *source* model")
    parser.add_argument("--trg_model", default='/home/yanmeng/zhangjingrui/models/deepseek-coder-6.7b-base',
                        help="HF Hub name or local path of *target* model")
    parser.add_argument("--out_dir",   default="./embeddings_d2q",
                        help="Directory to dump emb_sec.npy / emb_trg.npy")
    parser.add_argument("--align_method", choices=["procrustes", "none"],
                        default="procrustes",
                        help="How to align sec to trg space after PCA")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="Force common dim for PCA; default=min(dim_sec, dim_trg)")
    args = parser.parse_args()

    return args


def main(a):
    t0 = time.time()
    print(">>> loading models")
    tok_sec = AutoTokenizer.from_pretrained(a.sec_model, trust_remote_code=True)
    tok_trg = AutoTokenizer.from_pretrained(a.trg_model, trust_remote_code=True)
    V_sec, V_trg = len(tok_sec), len(tok_trg)
    m_sec = AutoModelForCausalLM.from_pretrained(a.sec_model, trust_remote_code=True, torch_dtype=torch.float32)
    m_sec.resize_token_embeddings(V_sec)
    m_trg = AutoModelForCausalLM.from_pretrained(a.trg_model, trust_remote_code=True, torch_dtype=torch.float32)
    m_trg.resize_token_embeddings(V_trg)
    emb_sec = m_sec.get_input_embeddings().weight.detach().cpu().numpy()  # [V_sec,d_sec]
    emb_trg = m_trg.get_input_embeddings().weight.detach().cpu().numpy()  # [V_trg,d_trg]
    d_sec, d_trg = emb_sec.shape[1], emb_trg.shape[1]
    print(f"    sec: {V_sec}×{d_sec}   trg: {V_trg}×{d_trg}")


    emb_sec = l2_normalize(emb_sec)
    emb_trg = l2_normalize(emb_trg)


    if d_sec != d_trg or a.pca_dim:
        d_common = a.pca_dim or min(d_sec, d_trg)
        print(f">>> PCA to d_common = {d_common}")
        emb_sec = PCA(d_common, svd_solver="auto", random_state=0).fit_transform(emb_sec)
        emb_trg = PCA(d_common, svd_solver="auto", random_state=0).fit_transform(emb_trg)
    else:
        d_common = d_sec


    print(">>> collecting common tokens")
    trg_tok2id = {t: i for i, t in enumerate(tok_trg.convert_ids_to_tokens(range(V_trg)))}
    common_sec_idx, common_trg_idx = [], []
    for s_id, tok in enumerate(tok_sec.convert_ids_to_tokens(range(V_sec))):
        t_id = trg_tok2id.get(tok)
        if t_id is not None:
            common_sec_idx.append(s_id)
            common_trg_idx.append(t_id)

    n_com = len(common_sec_idx)
    print(f"    common tokens = {n_com}")
    if n_com < 10:
        warnings.warn("公共词太少，Procrustes 质量可能受影响")


    print(">>> Procrustes alignment")
    X = emb_sec[common_sec_idx]          # [n_com,d_common]
    Y = emb_trg[common_trg_idx]          # [n_com,d_common]
    W, _ = orthogonal_procrustes(X, Y)   # d_common×d_common
    emb_sec_aligned = emb_sec @ W        # [V_sec,d_common]
    emb_trg_aligned = emb_trg            # 目标侧保持不变


    ensure_dir(a.out_dir)
    np.save(os.path.join(a.out_dir, "emb_sec.npy"), emb_sec_aligned.astype(np.float32))
    np.save(os.path.join(a.out_dir, "emb_trg.npy"), emb_trg_aligned.astype(np.float32))
    meta = dict(sec_model=a.sec_model, trg_model=a.trg_model,
                dim=d_common, common_tokens=n_com,
                sec_vocab=V_sec, trg_vocab=V_trg, pca_used=(d_sec!=d_trg or a.pca_dim is not None))
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(f"✓ saved to {a.out_dir}  | elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    args = get_args()
    main(args)