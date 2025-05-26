import os, json, argparse, time, warnings
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm


def l2_normalize(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
def ensure_dir(d): os.makedirs(d, exist_ok=True)


def main(a):
    t0 = time.time()
    print(">>> loading models")
    tok_src = AutoTokenizer.from_pretrained(a.src_model, trust_remote_code=True)
    tok_trg = AutoTokenizer.from_pretrained(a.trg_model, trust_remote_code=True)
    V_src, V_trg = len(tok_src), len(tok_trg)
    m_src = AutoModelForCausalLM.from_pretrained(a.src_model, trust_remote_code=True, torch_dtype=torch.float32)
    m_src.resize_token_embeddings(V_src)
    m_trg = AutoModelForCausalLM.from_pretrained(a.trg_model, trust_remote_code=True, torch_dtype=torch.float32)
    m_trg.resize_token_embeddings(V_trg)
    emb_src = m_src.get_input_embeddings().weight.detach().cpu().numpy()  # [V_src,d_src]
    emb_trg = m_trg.get_input_embeddings().weight.detach().cpu().numpy()  # [V_trg,d_trg]
    d_src, d_trg = emb_src.shape[1], emb_trg.shape[1]
    print(f"    src: {V_src}×{d_src}   trg: {V_trg}×{d_trg}")


    emb_src = l2_normalize(emb_src)
    emb_trg = l2_normalize(emb_trg)


    if d_src != d_trg or a.pca_dim:
        d_common = a.pca_dim or min(d_src, d_trg)
        print(f">>> PCA to d_common = {d_common}")
        emb_src = PCA(d_common, svd_solver="auto", random_state=0).fit_transform(emb_src)
        emb_trg = PCA(d_common, svd_solver="auto", random_state=0).fit_transform(emb_trg)
    else:
        d_common = d_src


    print(">>> collecting common tokens")
    trg_tok2id = {t: i for i, t in enumerate(tok_trg.convert_ids_to_tokens(range(V_trg)))}
    common_src_idx, common_trg_idx = [], []
    for s_id, tok in enumerate(tok_src.convert_ids_to_tokens(range(V_src))):
        t_id = trg_tok2id.get(tok)
        if t_id is not None:
            common_src_idx.append(s_id)
            common_trg_idx.append(t_id)

    n_com = len(common_src_idx)
    print(f"    common tokens = {n_com}")
    if n_com < 10:
        warnings.warn("公共词太少，Procrustes 质量可能受影响")


    print(">>> Procrustes alignment")
    X = emb_src[common_src_idx]          # [n_com,d_common]
    Y = emb_trg[common_trg_idx]          # [n_com,d_common]
    W, _ = orthogonal_procrustes(X, Y)   # d_common×d_common
    emb_src_aligned = emb_src @ W        # [V_src,d_common]
    emb_trg_aligned = emb_trg            # 目标侧保持不变


    ensure_dir(a.out_dir)
    np.save(os.path.join(a.out_dir, "emb_src.npy"), emb_src_aligned.astype(np.float32))
    np.save(os.path.join(a.out_dir, "emb_trg.npy"), emb_trg_aligned.astype(np.float32))
    meta = dict(src_model=a.src_model, trg_model=a.trg_model,
                dim=d_common, common_tokens=n_com,
                src_vocab=V_src, trg_vocab=V_trg, pca_used=(d_src!=d_trg or a.pca_dim is not None))
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=2)
    print(f"✓ saved to {a.out_dir}  | elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_model", default='/home/public_space/yanmeng/zhangjingrui/models/deepseek-coder-6.7b-base',
                        help="HF Hub name or local path of *source* model")
    parser.add_argument("--trg_model", default='/home/public_space/yanmeng/zhangjingrui/models/Qwen2.5-Coder-7B',
                        help="HF Hub name or local path of *target* model")
    parser.add_argument("--out_dir",   default="./embeddings_d2q",
                        help="Directory to dump emb_src.npy / emb_trg.npy")
    parser.add_argument("--align_method", choices=["procrustes", "none"],
                        default="procrustes",
                        help="How to align src to trg space after PCA")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="Force common dim for PCA; default=min(dim_src, dim_trg)")
    args = parser.parse_args()

    main(args)