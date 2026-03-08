
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from fcmeans import FCM
from scipy.stats import entropy as scipy_entropy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── config 
N_CLUSTERS   = 15      
FCM_M        = 2.0     
                       
FCM_MAX_ITER = 150
FCM_ERROR    = 1e-5
RANDOM_STATE = 42
BOUNDARY_GAP = 0.15    
TOP_N_DOCS   = 8       

OUTPUT_DIR = Path("./clustering_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── helpers 
def load_data():
    embeddings  = np.load("embeddings.npy")
    categories  = json.load(open("categories.json"))
    docs        = json.load(open("docs_cleaned.json"))
    return embeddings, categories, docs



def reduce_dims(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Reduce 384-dim embeddings to 50 dims via PCA before FCM.

    Why PCA before FCM?
      In high dimensions (384-d), all pairwise Euclidean distances converge
      to the same value (the "curse of dimensionality").  FCM uses distance
      to centroids — when all distances are equal, all memberships become
      1/k (= 0.067 for k=15), producing the degenerate uniform output seen
      without this step.  50 PCA components retain ~85%+ of variance while
      making distances meaningful again.
    """
    log.info(f"Reducing embeddings from {embeddings.shape[1]}-d to {n_components}-d via PCA ...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    log.info(f"PCA retains {explained:.1%} of variance")
    return reduced


def fpc_sweep(embeddings: np.ndarray, ks: list[int]) -> dict:
    """Compute Fuzzy Partition Coefficient for a range of k values."""
    results = {}
    log.info(f"FPC sweep over k = {ks} …")
    for k in ks:
        fcm = FCM(n_clusters=k, m=FCM_M, max_iter=FCM_MAX_ITER,
                  error=FCM_ERROR, random_state=RANDOM_STATE)
        fcm.fit(embeddings)
        results[k] = fcm.partition_coefficient
        log.info(f"  k={k:2d}  FPC={results[k]:.4f}")
    return results


def plot_fpc(fpc_dict: dict):
    ks   = sorted(fpc_dict)
    fpcs = [fpc_dict[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, fpcs, marker="o", linewidth=2)
    ax.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"chosen k={N_CLUSTERS}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Fuzzy Partition Coefficient")
    ax.set_title("FPC vs k — elbow determines cluster count")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "fpc_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info(f"Saved {path}")


def run_fcm(embeddings: np.ndarray) -> tuple[FCM, np.ndarray]:
    """Fit FCM and return (model, membership_matrix U)."""
    log.info(f"Fitting Fuzzy C-Means with k={N_CLUSTERS}, m={FCM_M} …")
    fcm = FCM(n_clusters=N_CLUSTERS, m=FCM_M, max_iter=FCM_MAX_ITER,
              error=FCM_ERROR, random_state=RANDOM_STATE)
    fcm.fit(embeddings)
    U = fcm.u              # shape (N, k) – membership distribution per document
    log.info(f"FCM converged. FPC={fcm.partition_coefficient:.4f}")
    return fcm, U


def cluster_analysis(U: np.ndarray, docs: list[str], categories: list[str]) -> dict:
    """
    Return rich per-cluster stats.
    dominant_cluster[i] = argmax of row i in U  (used only for visualisation,
    NOT as a hard label in the rest of the system).
    """
    dominant = U.argmax(axis=1)   
    report   = {}

    for c in range(N_CLUSTERS):
        members_mask = dominant == c
        member_idx   = np.where(members_mask)[0]

        
        memberships  = U[member_idx, c]

        
        cat_counts: dict[str, int] = {}
        for i in member_idx:
            cat_counts[categories[i]] = cat_counts.get(categories[i], 0) + 1

        # average entropy of membership vectors (measures fuzziness)
        avg_entropy = float(np.mean([scipy_entropy(U[i]) for i in member_idx])) \
            if len(member_idx) > 0 else 0.0

        # top representative docs = those with highest membership to c
        top_idx = member_idx[np.argsort(memberships)[::-1][:TOP_N_DOCS]]
        top_docs = [(docs[i][:120], float(U[i, c])) for i in top_idx]

        report[c] = {
            "n_dominant": int(members_mask.sum()),
            "avg_membership": float(memberships.mean()) if len(memberships) else 0.0,
            "avg_entropy": avg_entropy,
            "category_distribution": cat_counts,
            "top_docs": top_docs,
        }

    return report, dominant


def find_boundary_docs(U: np.ndarray, docs: list[str], gap: float = BOUNDARY_GAP):
    """
    Boundary documents: top-2 membership scores are within `gap` of each other.
    These are the semantically ambiguous posts that sit between topics.
    """
    sorted_u = np.sort(U, axis=1)[:, ::-1]
    gaps     = sorted_u[:, 0] - sorted_u[:, 1]
    boundary_mask = gaps < gap
    boundary_idx  = np.where(boundary_mask)[0]

    results = []
    for i in boundary_idx:
        top2_clusters = np.argsort(U[i])[::-1][:2]
        results.append({
            "doc_idx": int(i),
            "text_preview": docs[i][:150],
            "cluster_a": int(top2_clusters[0]),
            "score_a": float(U[i, top2_clusters[0]]),
            "cluster_b": int(top2_clusters[1]),
            "score_b": float(U[i, top2_clusters[1]]),
            "gap": float(gaps[i]),
        })

    
    results.sort(key=lambda x: x["gap"])
    log.info(f"Found {len(results):,} boundary documents (gap < {gap})")
    return results


def plot_tsne(embeddings: np.ndarray, dominant: np.ndarray):
    """2-D t-SNE projection coloured by dominant cluster."""
    log.info("Computing t-SNE projection (may take ~2 min on CPU) …")
    # Use a random subsample if corpus is huge to keep t-SNE tractable
    MAX_TSNE = 5000
    if len(embeddings) > MAX_TSNE:
        rng   = np.random.default_rng(RANDOM_STATE)
        idx   = rng.choice(len(embeddings), MAX_TSNE, replace=False)
        emb_s = embeddings[idx]
        dom_s = dominant[idx]
    else:
        emb_s, dom_s = embeddings, dominant

    proj = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE,
                max_iter=800, init="pca").fit_transform(emb_s)

    cmap   = cm.get_cmap("tab20", N_CLUSTERS)
    colors = [cmap(c) for c in dom_s]

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=dom_s, cmap="tab20",
                         s=4, alpha=0.6, vmin=0, vmax=N_CLUSTERS - 1)
    plt.colorbar(scatter, ax=ax, label="Dominant cluster")
    ax.set_title(f"t-SNE projection — {N_CLUSTERS} fuzzy clusters (dominant assignment)")
    ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / "tsne_clusters.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info(f"Saved {path}")


def print_report(report: dict):
    print("\n" + "=" * 70)
    print(f"CLUSTER ANALYSIS REPORT  (k={N_CLUSTERS})")
    print("=" * 70)
    for c, info in report.items():
        print(f"\n── Cluster {c:2d}  "
              f"(dominant docs: {info['n_dominant']:4d}, "
              f"avg_membership: {info['avg_membership']:.3f}, "
              f"avg_entropy: {info['avg_entropy']:.3f})")
        # top 3 categories
        top_cats = sorted(info["category_distribution"].items(),
                          key=lambda x: -x[1])[:3]
        print(f"   Top categories: {top_cats}")
        print(f"   Representative doc snippet:")
        if info["top_docs"]:
            text, score = info["top_docs"][0]
            print(f"     [{score:.3f}] {text!r}")


def main():
    embeddings, categories, docs = load_data()

    
    
    embeddings_reduced = reduce_dims(embeddings, n_components=50)

    
    fpc_dict = fpc_sweep(embeddings_reduced, ks=[5, 8, 10, 12, 15, 18, 20])
    plot_fpc(fpc_dict)

    
    fcm, U = run_fcm(embeddings_reduced)

    
    np.save("membership_matrix.npy", U)
    np.save("embeddings_reduced.npy", embeddings_reduced)
    log.info("Saved membership_matrix.npy  (shape: N × k)")

   
    report, dominant = cluster_analysis(U, docs, categories)
    np.save("dominant_clusters.npy", dominant)

    
    np.save("fcm_centroids.npy", fcm.centers)

    print_report(report)

    
    boundary = find_boundary_docs(U, docs)
    print("\n── TOP 5 BOUNDARY / AMBIGUOUS DOCUMENTS ──")
    for b in boundary[:5]:
        print(f"  Cluster {b['cluster_a']} ({b['score_a']:.3f}) ↔ "
              f"Cluster {b['cluster_b']} ({b['score_b']:.3f})  "
              f"gap={b['gap']:.3f}")
        print(f"  {b['text_preview']!r}\n")

    
    plot_tsne(embeddings, dominant)

    
    import json

    def make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serialisable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open("cluster_report.json", "w") as f:
        json.dump(make_serialisable(report), f, indent=2)

    log.info("Part 2 complete ✓")


if __name__ == "__main__":
    main()