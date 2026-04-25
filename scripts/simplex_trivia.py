"""
POC v4: Simplex clustering of next-token probability distributions
        on TriviaQA — valid (correct answers) vs invalid (hallucinated answers).
        Model: Qwen/Qwen3-4B in bfloat16 (~8GB VRAM).

The key difference from GPT-2:
  - Qwen3-4B actually knows many TriviaQA answers, so when it hallucinates
    it does so with genuine overconfidence — a much stronger signal.
  - Thinking mode is disabled (enable_thinking=False) for deterministic
    next-token distributions without chain-of-thought tokens inflating the
    answer length.

Pipeline
--------
1.  Load TriviaQA (validation split, rc subset)
2.  For each question:
      a. Build chat-formatted prompt + correct answer  → valid token ids
      b. Generate Qwen3's own greedy answer            → invalid token ids
3.  Get top-k next-token probs for ANSWER tokens only
4.  Fit simplex k-means ONCE on the pooled corpus (Hilbert metric)
5.  Assign all tokens to fixed centroids → per-sequence frequency vectors
6.  ILR-transform → GP-ready features
7.  Plot: per-token PCA, per-sequence PCA, cluster occupancy, centroid entropy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cdist

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
TOP_K = 50  # simplex lives in Δ^{TOP_K - 1}
N_QUESTIONS = 200  # number of QA pairs to use
MAX_ANS_TOK = 30  # max new tokens for the generated answer
MAX_Q_TOK = 128  # max tokens for the question prompt
N_CLUSTERS = 8
N_INIT = 5
MAX_ITER = 50
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-10

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── 1. TriviaQA loading ───────────────────────────────────────────────────────


def load_triviaqa(n_questions: int):
    """
    Load TriviaQA rc validation split.
    Returns list of (question_str, correct_answer_str) pairs.
    Filters to questions whose shortest correct answer is short enough
    to keep sequences manageable.
    """
    ds = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)
    pairs = []
    for row in ds:
        q = row["question"].strip()
        # aliases contains all acceptable answers; take the shortest one
        aliases = row["answer"]["aliases"]
        if not aliases:
            continue
        ans = min(aliases, key=len).strip()
        # skip very long answers
        if len(ans.split()) > 6:
            continue
        pairs.append((q, ans))
        if len(pairs) >= n_questions:
            break
    print(f"Loaded {len(pairs)} QA pairs from TriviaQA")
    return pairs


# ── 2. Sequence construction ──────────────────────────────────────────────────


def make_chat_prompt(tokenizer, question: str) -> str:
    """
    Apply Qwen3 chat template for a bare question (no answer).
    thinking=False disables the <think>...</think> preamble so answer
    tokens start immediately — cleaner signal extraction.
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3-specific: skip CoT tokens
    )


def build_prompt_ids(tokenizer, question: str) -> torch.Tensor:
    """Tokenise the chat-formatted question prompt only (no answer)."""
    prompt = make_chat_prompt(tokenizer, question)
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_Q_TOK,
    )["input_ids"].to(DEVICE)
    return ids


def build_valid_ids(tokenizer, question: str, answer: str) -> torch.Tensor:
    """
    Tokenise the chat prompt with the CORRECT answer appended.
    Valid sequence: model sees the right answer tokens.
    """
    prompt = make_chat_prompt(tokenizer, question)
    full_text = prompt + answer
    ids = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_Q_TOK + MAX_ANS_TOK,
    )["input_ids"].to(DEVICE)
    return ids


def generate_invalid_ids(model, tokenizer, question: str) -> torch.Tensor:
    """
    Let Qwen3 greedily generate its own answer.
    Greedy decoding (do_sample=False) maximises per-token confidence,
    making overconfident-but-wrong distributions most visible.
    This is the INVALID (potentially hallucinated) sequence.
    """
    prompt_ids = build_prompt_ids(tokenizer, question)
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_new_tokens=MAX_ANS_TOK,
            do_sample=False,
            temperature=None,  # must be None when do_sample=False for Qwen3
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return generated


# ── 3. Top-k probability extraction ──────────────────────────────────────────


def get_topk_probs_from_ids(
    model,
    input_ids: torch.Tensor,
    top_k: int,
    answer_start_idx: int,
) -> np.ndarray:
    """
    Run a forward pass and return renormalised top-k next-token probabilities
    for ANSWER tokens only (positions answer_start_idx onwards).

    We focus on answer tokens because:
      - Question tokens are identical for valid and invalid → no signal
      - Answer tokens are where valid vs hallucinated diverges

    Returns
    -------
    probs : (L, top_k) where L = number of answer token positions
    """
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, T, V)

    # next-token logits: position t predicts token t+1
    # answer positions start at answer_start_idx - 1 (predicting first answer token)
    start = max(0, answer_start_idx - 1)
    next_logits = logits[0, start:-1, :]  # (L, V)

    if next_logits.shape[0] == 0:
        # fallback: use all positions
        next_logits = logits[0, :-1, :]

    probs_full = torch.softmax(next_logits, dim=-1)
    topk_vals, _ = torch.topk(probs_full, top_k, dim=-1)
    topk_renorm = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    return topk_renorm.cpu().numpy()  # (L, top_k)


def find_answer_start(
    tokenizer,
    full_ids: torch.Tensor,
    question: str,
) -> int:
    """
    Find the token index where the answer begins (after the chat prompt).
    Uses the prompt-only encoding to measure prompt length precisely.
    """
    prompt = make_chat_prompt(tokenizer, question)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_Q_TOK)[
        "input_ids"
    ].shape[1]
    return min(prompt_len, full_ids.shape[1] - 2)


# ── 4. Distances (vectorised) ─────────────────────────────────────────────────


def hilbert_distance_vec(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(n,k) x (m,k) -> (n,m) Hilbert distance matrix."""
    lr = np.log(X[:, None, :] + EPS) - np.log(Y[None, :, :] + EPS)
    return lr.max(axis=2) - lr.min(axis=2)


def aitchison_clr(X: np.ndarray) -> np.ndarray:
    lx = np.log(X + EPS)
    return lx - lx.mean(axis=1, keepdims=True)


# ── 5. ILR transform ──────────────────────────────────────────────────────────


def ilr_transform(X: np.ndarray) -> np.ndarray:
    """Δ^{k-1} -> R^{k-1} via Helmert contrasts (Egozcue et al. 2003)."""
    X = X + EPS
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    n, k = X.shape
    Z = np.zeros((n, k - 1))
    for j in range(1, k):
        s = np.sqrt(j / (j + 1))
        Z[:, j - 1] = s * (log_X[:, :j].mean(axis=1) - log_X[:, j])
    return Z


# ── 6. Simplex k-means ────────────────────────────────────────────────────────


def simplex_kmeans(
    X: np.ndarray,
    n_clusters: int,
    metric: str = "hilbert",
    n_init: int = N_INIT,
    max_iter: int = MAX_ITER,
) -> tuple:
    """
    Fit simplex k-means.
    Init    : k-means++ in ILR space (fast)
    Assign  : vectorised Hilbert distance
    Update  : geometric mean centroid (log-space mean, renormalised)
    """
    n, k = X.shape
    X_ilr = ilr_transform(X)
    best_labels, best_centroids, best_inertia = None, None, np.inf

    for _ in range(n_init):
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


def assign_to_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    metric: str = "hilbert",
) -> np.ndarray:
    """Assign-only — fixed centroids, no re-fitting."""
    if metric == "hilbert":
        D = hilbert_distance_vec(X, centroids)
    else:
        D = cdist(aitchison_clr(X), aitchison_clr(centroids))
    return D.argmin(axis=1)


def seq_cluster_freq(
    probs: np.ndarray,
    centroids: np.ndarray,
    metric: str = "hilbert",
) -> np.ndarray:
    """(L, k) -> (n_clusters,) normalised frequency using fixed centroids."""
    labels = assign_to_centroids(probs, centroids, metric=metric)
    counts = np.bincount(labels, minlength=N_CLUSTERS).astype(float)
    return counts / counts.sum()


# ── 7. Plots ──────────────────────────────────────────────────────────────────

CMAP = plt.get_cmap("tab10")


def pca2(X: np.ndarray, pca: PCA = None):
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
    valid_pool: np.ndarray,  # (N_ans_total, TOP_K)
    invalid_pool: np.ndarray,
    tok_lv: np.ndarray,  # token-level cluster labels
    tok_li: np.ndarray,
    centroids: np.ndarray,  # (N_CLUSTERS, TOP_K)
    valid_sf: np.ndarray,  # (N_QUESTIONS, N_CLUSTERS) sequence freq
    invalid_sf: np.ndarray,
    valid_lens: list,  # answer lengths per question (valid)
    invalid_lens: list,  # answer lengths per question (invalid)
    metric: str,
):
    # ── ILR + PCA: token level ────────────────────────────────────────────────
    all_pool = np.vstack([valid_pool, invalid_pool])
    pool_ilr = ilr_transform(all_pool)
    cent_ilr = ilr_transform(centroids)
    pool_2d, pca_tok = pca2(pool_ilr)
    cent_2d, _ = pca2(cent_ilr, pca=pca_tok)
    v2d = pool_2d[: len(valid_pool)]
    i2d = pool_2d[len(valid_pool) :]
    all_tok_labels = np.concatenate([tok_lv, tok_li])

    # ── ILR + PCA: sequence level ─────────────────────────────────────────────
    all_sf = np.vstack([valid_sf, invalid_sf])
    sf_ilr = ilr_transform(all_sf)
    sf_2d, pca_sf = pca2(sf_ilr)
    vsf2d = sf_2d[: len(valid_sf)]
    isf2d = sf_2d[len(valid_sf) :]
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=5)
    sf_labels = km.fit_predict(sf_ilr)

    x = np.arange(N_CLUSTERS)
    w = 0.35

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

    # [0,0] per-token by cluster
    ax = fig.add_subplot(gs[0, 0])
    for c in range(N_CLUSTERS):
        m = all_tok_labels == c
        if m.any():
            ax.scatter(*pool_2d[m].T, s=10, alpha=0.35, color=CMAP(c), lw=0, label=f"C{c}")
    ax.scatter(
        *cent_2d.T, marker="*", s=300, color="k", edgecolors="w", lw=1, zorder=5, label="Centroids"
    )
    ax.legend(fontsize=7, ncol=2)
    ax_style(ax, pca_tok, "Per-token: by cluster")

    # [0,1] per-token valid vs invalid
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(*v2d.T, s=12, alpha=0.4, color="#1f77b4", lw=0, label=f"Valid answer (n={len(v2d)})")
    ax.scatter(
        *i2d.T,
        marker="x",
        s=20,
        alpha=0.5,
        lw=1.2,
        color="#d62728",
        label=f"Qwen3 answer (n={len(i2d)})",
    )
    ax.legend(fontsize=8)
    ax_style(ax, pca_tok, "Per-token: correct vs Qwen3 generated")

    # [0,2] mean token cluster occupancy
    ax = fig.add_subplot(gs[0, 2])
    # Reconstruct per-question occupancy using valid_lens / invalid_lens
    v_occ_rows, i_occ_rows = [], []
    vstart = istart = 0
    for vl, il in zip(valid_lens, invalid_lens):
        vc = np.bincount(tok_lv[vstart : vstart + vl], minlength=N_CLUSTERS).astype(float)
        ic = np.bincount(tok_li[istart : istart + il], minlength=N_CLUSTERS).astype(float)
        v_occ_rows.append(vc / vc.sum() if vc.sum() > 0 else vc)
        i_occ_rows.append(ic / ic.sum() if ic.sum() > 0 else ic)
        vstart += vl
        istart += il
    v_occ = np.vstack(v_occ_rows)
    i_occ = np.vstack(i_occ_rows)
    ax.bar(x - w / 2, v_occ.mean(0), w, color="#1f77b4", alpha=0.75, label="Valid")
    ax.bar(x + w / 2, i_occ.mean(0), w, color="#d62728", alpha=0.75, label="Qwen3")
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
            ax.scatter(*sf_2d[m].T, s=30, alpha=0.6, color=CMAP(c), lw=0, label=f"C{c}")
    ax.legend(fontsize=7, ncol=2)
    ax_style(ax, pca_sf, "Per-sequence freq: by cluster")

    # [1,1] per-sequence valid vs invalid
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(*vsf2d.T, s=45, alpha=0.65, color="#1f77b4", lw=0, label=f"Valid (n={len(vsf2d)})")
    ax.scatter(
        *isf2d.T,
        marker="x",
        s=65,
        alpha=0.75,
        lw=1.8,
        color="#d62728",
        label=f"Qwen3 (n={len(isf2d)})",
    )
    ax.legend(fontsize=8)
    ax_style(ax, pca_sf, "Per-sequence freq: correct vs Qwen3 generated")

    # [1,2] mean sequence cluster occupancy
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x - w / 2, valid_sf.mean(0), w, color="#1f77b4", alpha=0.75, label="Valid")
    ax.bar(x + w / 2, invalid_sf.mean(0), w, color="#d62728", alpha=0.75, label="Qwen3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x], fontsize=8)
    ax.set_ylabel("Mean sequence fraction")
    ax.legend(fontsize=8)
    ax.set_title("Mean sequence cluster occupancy", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.3, ls="--")

    fig.suptitle(
        f"TriviaQA | Correct answer vs {MODEL_NAME} generated answer\n"
        f"TOP_K={TOP_K}, K={N_CLUSTERS} clusters, metric={metric}",
        fontsize=13,
        weight="bold",
    )
    plt.savefig("simplex_clustering_triviaqa.png", dpi=150, bbox_inches="tight")
    print("Saved simplex_clustering_triviaqa.png")

    # ── Centroid entropy plot ──────────────────────────────────────────────────
    cent_H = -(centroids * np.log(centroids + EPS)).sum(axis=1) / np.log(TOP_K)
    # Sort clusters by entropy for readability
    order = np.argsort(cent_H)
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))

    axes2[0].bar(np.arange(N_CLUSTERS), cent_H[order], color=[CMAP(i) for i in order], alpha=0.85)
    axes2[0].set_xticks(np.arange(N_CLUSTERS))
    axes2[0].set_xticklabels([f"C{i}" for i in order])
    axes2[0].set_ylabel("Normalised entropy (0=peaked, 1=uniform)")
    axes2[0].set_title("Centroid entropy (sorted)", weight="bold")
    axes2[0].grid(axis="y", alpha=0.3, ls="--")

    # Δ occupancy: invalid - valid, sorted by entropy
    delta = invalid_sf.mean(0) - valid_sf.mean(0)
    colours = ["#d62728" if d > 0 else "#1f77b4" for d in delta[order]]
    axes2[1].bar(np.arange(N_CLUSTERS), delta[order], color=colours, alpha=0.85)
    axes2[1].axhline(0, color="k", lw=0.8)
    axes2[1].set_xticks(np.arange(N_CLUSTERS))
    axes2[1].set_xticklabels([f"C{i}\nH={cent_H[i]:.2f}" for i in order], fontsize=8)
    axes2[1].set_ylabel("Δ mean occupancy (Qwen3 − valid)")
    axes2[1].set_title(
        "Which clusters does Qwen3 over/under-use?\n(red = Qwen3 higher, blue = valid higher)",
        weight="bold",
    )
    axes2[1].grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig("centroid_entropy_triviaqa.png", dpi=150, bbox_inches="tight")
    print("Saved centroid_entropy_triviaqa.png")


# ── 8. Main ───────────────────────────────────────────────────────────────────


def main():
    print(
        f"Config: MODEL={MODEL_NAME}, TOP_K={TOP_K}, N_QUESTIONS={N_QUESTIONS}, "
        f"N_CLUSTERS={N_CLUSTERS}, device={DEVICE}\n"
    )

    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # native Qwen3 dtype, ~8GB VRAM for 4B
        device_map="auto",  # handles multi-GPU or CPU offload if needed
        attn_implementation="eager",  # use "flash_attention_2" if available
    ).eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    print(
        f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        if torch.cuda.is_available()
        else "Running on CPU"
    )

    print("Loading TriviaQA …")
    qa_pairs = load_triviaqa(N_QUESTIONS)

    # ── Step 1: extract per-token probabilities ───────────────────────────────
    print(f"\nExtracting probabilities for {len(qa_pairs)} QA pairs …")

    valid_pool_list = []  # per-question (L_v, TOP_K) arrays
    invalid_pool_list = []  # per-question (L_i, TOP_K) arrays
    valid_lens = []
    invalid_lens = []
    skipped = 0

    for i, (question, answer) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(qa_pairs)}  (skipped={skipped})")

        try:
            # ── Valid: correct answer ────────────────────────────────────────
            valid_ids = build_valid_ids(tokenizer, question, answer)
            ans_start = find_answer_start(tokenizer, valid_ids, question)
            valid_probs = get_topk_probs_from_ids(model, valid_ids, TOP_K, ans_start)

            if valid_probs.shape[0] == 0:
                skipped += 1
                continue

            # ── Invalid: Qwen3 generated answer ─────────────────────────────
            invalid_ids = generate_invalid_ids(model, tokenizer, question)
            inv_ans_start = find_answer_start(tokenizer, invalid_ids, question)
            invalid_probs = get_topk_probs_from_ids(model, invalid_ids, TOP_K, inv_ans_start)

            if invalid_probs.shape[0] == 0:
                skipped += 1
                continue

            valid_pool_list.append(valid_probs)
            invalid_pool_list.append(invalid_probs)
            valid_lens.append(valid_probs.shape[0])
            invalid_lens.append(invalid_probs.shape[0])

        except Exception as e:
            skipped += 1
            continue

    N = len(valid_pool_list)
    print(f"\nUsable pairs: {N}  (skipped: {skipped})")

    valid_pool = np.vstack(valid_pool_list)
    invalid_pool = np.vstack(invalid_pool_list)
    all_pool = np.vstack([valid_pool, invalid_pool])
    print(f"Token-level pool: valid={valid_pool.shape}, invalid={invalid_pool.shape}")

    # ── Step 2: fit centroids ONCE on pooled corpus ───────────────────────────
    print(f"\nFitting {N_CLUSTERS} centroids on {len(all_pool)} points (Hilbert) …")
    _, centroids = simplex_kmeans(all_pool, N_CLUSTERS, metric="hilbert")

    cent_H_norm = -(centroids * np.log(centroids + EPS)).sum(1) / np.log(TOP_K)
    print("\nCentroid summary:")
    for i, (c, h) in enumerate(zip(centroids, cent_H_norm)):
        top3 = np.sort(c)[::-1][:3].round(3)
        print(f"  C{i}: norm-H={h:.3f}  top-3={top3}")

    # ── Step 3: assign using fixed centroids ──────────────────────────────────
    print("\nAssigning tokens to fixed centroids …")
    tok_lv = assign_to_centroids(valid_pool, centroids)
    tok_li = assign_to_centroids(invalid_pool, centroids)

    # ── Step 4: per-sequence cluster frequency vectors ────────────────────────
    print("Computing per-sequence cluster frequencies …")
    valid_sf = np.array([seq_cluster_freq(valid_pool_list[i], centroids) for i in range(N)])
    invalid_sf = np.array([seq_cluster_freq(invalid_pool_list[i], centroids) for i in range(N)])

    # ── Step 5: ILR → GP-ready features ──────────────────────────────────────
    valid_ilr = ilr_transform(valid_sf)
    invalid_ilr = ilr_transform(invalid_sf)
    print(f"\nGP-ready ILR features: {valid_ilr.shape} / {invalid_ilr.shape}")

    # ── Step 6: separation statistics ─────────────────────────────────────────
    print("\nCluster occupancy (mean ± std per sequence):")
    print(f"{'':5} | {'Valid':^22} | {'Qwen3':^22} | {'Δ':>7}")
    for c in range(N_CLUSTERS):
        vm, vs = valid_sf[:, c].mean(), valid_sf[:, c].std()
        im, is_ = invalid_sf[:, c].mean(), invalid_sf[:, c].std()
        marker = " ◄" if abs(im - vm) > 0.03 else ""
        print(
            f"  C{c}  | {vm:.4f} ± {vs:.4f}       | "
            f"{im:.4f} ± {is_:.4f}       | {im - vm:+.4f}{marker}"
        )

    # ── Step 7: plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    make_all_plots(
        valid_pool,
        invalid_pool,
        tok_lv,
        tok_li,
        centroids,
        valid_sf,
        invalid_sf,
        valid_lens,
        invalid_lens,
        metric="hilbert",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
