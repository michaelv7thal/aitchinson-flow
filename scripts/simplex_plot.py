"""
POC v2: Clustering next-token probability distributions on the simplex.
Key fix: centroids fitted ONCE on pooled corpus; all per-sequence labels
derived by assignment only (no re-fitting per sequence).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scipy.spatial.distance import cdist
from typing import Literal

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K = 128
N_SEQUENCES = 100
SEQ_LEN = 32
SCRAMBLE_FRAC = 0.15
N_CLUSTERS = 8
N_INIT = 5
MAX_ITER = 50
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-10

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── 1. Data ───────────────────────────────────────────────────────────────────
def load_text8_sequences(n_sequences, seq_len):
    dataset = load_dataset("afmck/text8", split="train")
    text = dataset[0]["text"]
    chunk_size = seq_len * 5
    return [text[i : i + chunk_size] for i in range(0, chunk_size * n_sequences * 2, chunk_size)][
        :n_sequences
    ]


def scramble_sequence(tokens, frac):
    tokens = tokens.clone()
    n = tokens.shape[-1]
    n_sc = max(1, int(n * frac))
    idx = torch.randperm(n)[:n_sc]
    perm = torch.randperm(n)[:n_sc]
    tokens[0, idx] = tokens[0, perm]
    return tokens


# ── 2. GPT-2 extraction ───────────────────────────────────────────────────────
def get_topk_probs(model, tokenizer, text, top_k, seq_len, scramble=False):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len + 1)
    ids = enc["input_ids"].to(DEVICE)
    if scramble:
        ids = scramble_sequence(ids, SCRAMBLE_FRAC)
    with torch.no_grad():
        logits = model(ids).logits
    next_logits = logits[0, :-1, :]
    probs = torch.softmax(next_logits, dim=-1)
    topk, _ = torch.topk(probs, top_k, dim=-1)
    topk = topk / topk.sum(dim=-1, keepdim=True)
    return topk.cpu().numpy()


# ── 3. Distances (vectorised) ─────────────────────────────────────────────────
def hilbert_distance_vec(X, Y):
    """(n,k) x (m,k) -> (n,m) Hilbert distance matrix."""
    lr = np.log(X[:, None, :] + EPS) - np.log(Y[None, :, :] + EPS)
    return lr.max(axis=2) - lr.min(axis=2)


def aitchison_clr(X):
    lx = np.log(X + EPS)
    return lx - lx.mean(axis=1, keepdims=True)


# ── 4. ILR transform ──────────────────────────────────────────────────────────
def ilr_transform(X):
    """Δ^{k-1} -> R^{k-1} via Helmert contrasts."""
    X = X + EPS
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    n, k = X.shape
    Z = np.zeros((n, k - 1))
    for j in range(1, k):
        s = np.sqrt(j / (j + 1))
        Z[:, j - 1] = s * (log_X[:, :j].mean(axis=1) - log_X[:, j])
    return Z


# ── 5. Simplex k-means ────────────────────────────────────────────────────────
def simplex_kmeans(X, n_clusters, metric="hilbert", n_init=N_INIT, max_iter=MAX_ITER):
    """
    Fit k-means on the simplex.
    - Init  : k-means++ in ILR space (fast, avoids O(n^2 k) Hilbert loop)
    - Assign: vectorised Hilbert or Aitchison
    - Update: geometric mean (log-space arithmetic mean), renormalised
    """
    n, k = X.shape
    X_ilr = ilr_transform(X)
    best_labels, best_centroids, best_inertia = None, None, np.inf

    for _ in range(n_init):
        # k-means++ in ILR space
        idx = [np.random.randint(n)]
        for _ in range(n_clusters - 1):
            D = cdist(X_ilr, X_ilr[idx]).min(axis=1) ** 2
            idx.append(np.random.choice(n, p=D / D.sum()))
        centroids = X[idx].copy()
        labels = np.full(n, -1, dtype=int)

        for it in range(max_iter):
            if metric == "hilbert":
                Dm = hilbert_distance_vec(X, centroids)
            else:
                Dm = cdist(aitchison_clr(X), aitchison_clr(centroids))
            new_labels = Dm.argmin(axis=1)
            if np.array_equal(new_labels, labels) and it > 0:
                break
            labels = new_labels
            for c in range(n_clusters):
                m = X[labels == c]
                if len(m) == 0:
                    centroids[c] = X[np.random.randint(n)]
                else:
                    g = np.exp(np.log(m + EPS).mean(axis=0))
                    centroids[c] = g / g.sum()

        inertia = Dm[np.arange(n), labels].sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids


def assign_to_centroids(X, centroids, metric="hilbert"):
    """Assign-only — no re-fitting. Use this for all per-sequence labelling."""
    if metric == "hilbert":
        D = hilbert_distance_vec(X, centroids)
    else:
        D = cdist(aitchison_clr(X), aitchison_clr(centroids))
    return D.argmin(axis=1)


def seq_cluster_freq(probs, centroids, metric="hilbert"):
    """(L, k) token probs -> (n_clusters,) normalised frequency using fixed centroids."""
    labels = assign_to_centroids(probs, centroids, metric=metric)
    counts = np.bincount(labels, minlength=N_CLUSTERS).astype(float)
    return counts / counts.sum()


# ── 6. Plots ──────────────────────────────────────────────────────────────────
CMAP = plt.get_cmap("tab10")


def pca2(X, pca=None):
    if pca is None:
        pca = PCA(n_components=min(2, X.shape[1]))
        return pca.fit_transform(X), pca
    return pca.transform(X), pca


def ax_style(ax, pca, title):
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(True, alpha=0.2, ls="--", lw=0.5)


def make_all_plots(
    valid_pool, invalid_pool, tok_lv, tok_li, centroids, valid_sf, invalid_sf, metric
):

    # PCA for token-level
    all_pool = np.vstack([valid_pool, invalid_pool])
    pool_ilr = ilr_transform(all_pool)
    cent_ilr = ilr_transform(centroids)
    pool_2d, pca_tok = pca2(pool_ilr)
    cent_2d, _ = pca2(cent_ilr, pca=pca_tok)
    v2d = pool_2d[: len(valid_pool)]
    i2d = pool_2d[len(valid_pool) :]

    # PCA for sequence-level
    all_sf = np.vstack([valid_sf, invalid_sf])
    sf_ilr = ilr_transform(all_sf)
    sf_2d, pca_sf = pca2(sf_ilr)
    vsf2d = sf_2d[: len(valid_sf)]
    isf2d = sf_2d[len(valid_sf) :]
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=5)
    sf_labels = km.fit_predict(sf_ilr)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # [0,0] per-token by cluster
    ax = fig.add_subplot(gs[0, 0])
    all_tok_labels = np.concatenate([tok_lv, tok_li])
    for c in range(N_CLUSTERS):
        m = all_tok_labels == c
        if m.any():
            ax.scatter(*pool_2d[m].T, s=15, alpha=0.45, color=CMAP(c), lw=0, label=f"C{c}")
    ax.scatter(
        *cent_2d.T, marker="*", s=300, color="k", edgecolors="w", lw=1, zorder=5, label="Centroids"
    )
    ax.legend(fontsize=7, ncol=2)
    ax_style(ax, pca_tok, "Per-token: by cluster")

    # [0,1] per-token valid vs invalid
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(*v2d.T, s=20, alpha=0.5, color="#1f77b4", lw=0, label=f"Valid (n={len(v2d)})")
    ax.scatter(
        *i2d.T,
        marker="x",
        s=35,
        alpha=0.6,
        lw=1.5,
        color="#d62728",
        label=f"Invalid (n={len(i2d)})",
    )
    ax.legend(fontsize=8)
    ax_style(ax, pca_tok, "Per-token: valid vs invalid")

    # [0,2] mean cluster occupancy (token level)
    ax = fig.add_subplot(gs[0, 2])
    L = valid_pool.shape[0] // N_SEQUENCES
    v_occ = (
        np.vstack(
            [
                np.bincount(tok_lv[i * L : (i + 1) * L], minlength=N_CLUSTERS)
                for i in range(N_SEQUENCES)
            ]
        )
        / L
    )
    i_occ = (
        np.vstack(
            [
                np.bincount(tok_li[i * L : (i + 1) * L], minlength=N_CLUSTERS)
                for i in range(N_SEQUENCES)
            ]
        )
        / L
    )
    x = np.arange(N_CLUSTERS)
    w = 0.35
    ax.bar(x - w / 2, v_occ.mean(0), w, color="#1f77b4", alpha=0.75, label="Valid")
    ax.bar(x + w / 2, i_occ.mean(0), w, color="#d62728", alpha=0.75, label="Invalid")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x], fontsize=8)
    ax.set_ylabel("Mean token fraction")
    ax.legend(fontsize=8)
    ax.set_title("Mean token cluster occupancy", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")

    # [1,0] per-sequence by cluster
    ax = fig.add_subplot(gs[1, 0])
    for c in range(N_CLUSTERS):
        m = sf_labels == c
        if m.any():
            ax.scatter(*sf_2d[m].T, s=30, alpha=0.55, color=CMAP(c), lw=0, label=f"C{c}")
    ax.legend(fontsize=7, ncol=2)
    ax_style(ax, pca_sf, "Per-sequence freq: by cluster")

    # [1,1] per-sequence valid vs invalid
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(*vsf2d.T, s=40, alpha=0.6, color="#1f77b4", lw=0, label=f"Valid (n={len(vsf2d)})")
    ax.scatter(
        *isf2d.T,
        marker="x",
        s=60,
        alpha=0.7,
        lw=1.5,
        color="#d62728",
        label=f"Invalid (n={len(isf2d)})",
    )
    ax.legend(fontsize=8)
    ax_style(ax, pca_sf, "Per-sequence freq: valid vs invalid")

    # [1,2] mean cluster occupancy (sequence level)
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x - w / 2, valid_sf.mean(0), w, color="#1f77b4", alpha=0.75, label="Valid")
    ax.bar(x + w / 2, invalid_sf.mean(0), w, color="#d62728", alpha=0.75, label="Invalid")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x], fontsize=8)
    ax.set_ylabel("Mean sequence fraction")
    ax.legend(fontsize=8)
    ax.set_title("Mean sequence cluster occupancy", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        f"Simplex clustering | TOP_K={TOP_K}, K={N_CLUSTERS}, metric={metric}",
        fontsize=13,
        weight="bold",
    )
    plt.savefig("simplex_clustering.png", dpi=150, bbox_inches="tight")
    print("Saved simplex_clustering.png")

    # Centroid entropy
    cent_H = -(centroids * np.log(centroids + EPS)).sum(axis=1) / np.log(TOP_K)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x, cent_H, color=[CMAP(i) for i in x], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{i}" for i in x])
    ax2.set_ylabel("Normalised entropy (0=peaked, 1=uniform)")
    ax2.set_title("Centroid entropy — overconfidence per cluster", weight="bold")
    ax2.grid(axis="y", alpha=0.3, ls="--")
    plt.tight_layout()
    plt.savefig("centroid_entropy.png", dpi=150, bbox_inches="tight")
    print("Saved centroid_entropy.png")


# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    print(
        f"Config: TOP_K={TOP_K}, N_SEQUENCES={N_SEQUENCES}, "
        f"SEQ_LEN={SEQ_LEN}, N_CLUSTERS={N_CLUSTERS}, device={DEVICE}"
    )

    print("\nLoading GPT-2 …")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

    print("Loading text8 …")
    texts = load_text8_sequences(N_SEQUENCES, SEQ_LEN)

    # Step 1: extract
    print(f"\nExtracting top-{TOP_K} probs for {N_SEQUENCES} sequences …")
    vl, il = [], []
    for i, t in enumerate(texts):
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_SEQUENCES}")
        vl.append(get_topk_probs(model, tokenizer, t, TOP_K, SEQ_LEN, scramble=False))
        il.append(get_topk_probs(model, tokenizer, t, TOP_K, SEQ_LEN, scramble=True))

    valid_pool = np.vstack(vl)
    invalid_pool = np.vstack(il)
    all_pool = np.vstack([valid_pool, invalid_pool])
    print(f"Pooled: {all_pool.shape}")

    # Step 2: fit centroids ONCE
    print(f"\nFitting {N_CLUSTERS} centroids on {len(all_pool)} points (Hilbert) …")
    _, centroids = simplex_kmeans(all_pool, N_CLUSTERS, metric="hilbert")

    cent_H_norm = -(centroids * np.log(centroids + EPS)).sum(1) / np.log(TOP_K)
    print("\nCentroid summary (norm-entropy, top-3 probs):")
    for i, (c, h) in enumerate(zip(centroids, cent_H_norm)):
        print(f"  C{i}: H={h:.3f}  top3={np.sort(c)[::-1][:3].round(3)}")

    # Step 3: assign using fixed centroids (no re-fitting!)
    print("\nAssigning tokens to fixed centroids …")
    tok_lv = assign_to_centroids(valid_pool, centroids)
    tok_li = assign_to_centroids(invalid_pool, centroids)

    # Step 4: per-sequence cluster frequency vectors
    print("Computing per-sequence cluster frequencies …")
    valid_sf = np.array([seq_cluster_freq(vl[i], centroids) for i in range(N_SEQUENCES)])
    invalid_sf = np.array([seq_cluster_freq(il[i], centroids) for i in range(N_SEQUENCES)])

    # Step 5: ILR → GP-ready
    valid_ilr = ilr_transform(valid_sf)
    invalid_ilr = ilr_transform(invalid_sf)
    print(f"\nGP-ready ILR features: {valid_ilr.shape} / {invalid_ilr.shape}")

    # Step 6: separation summary
    print("\nCluster occupancy (mean ± std per sequence):")
    print(f"{'':5} | {'Valid':^20} | {'Invalid':^20} | {'Δ':>6}")
    for c in range(N_CLUSTERS):
        vm, vs = valid_sf[:, c].mean(), valid_sf[:, c].std()
        im, is_ = invalid_sf[:, c].mean(), invalid_sf[:, c].std()
        print(f"  C{c}  | {vm:.4f} ± {vs:.4f}     | {im:.4f} ± {is_:.4f}     | {im - vm:+.4f}")

    # Step 7: plot
    print("\nPlotting …")
    make_all_plots(
        valid_pool, invalid_pool, tok_lv, tok_li, centroids, valid_sf, invalid_sf, metric="hilbert"
    )
    print("Done.")


if __name__ == "__main__":
    main()
