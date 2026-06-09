"""
Evaluate the accuracy of the Level-1 CoarseGenerator.

Four checks:
  1. Next-step accuracy   -- teacher-forced top-1/top-3 on val elephant
  2. Transition matrix    -- real vs model transition probabilities
  3. Visit frequency      -- real vs generated cell visit proportions
  4. Free-run degradation -- how accuracy and KL divergence evolve as the
                             model runs on its own predictions (no ground truth)

Usage:
    python scripts/evaluate_coarse.py
    python scripts/evaluate_coarse.py --val-elephant AG005 --steps 500 --output results/coarse_eval/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elephantgraph.models.coarse_generator import CoarseGenerator

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")


# ---------------------------------------------------------------------------
# Helpers (same as generate_coarse.py)
# ---------------------------------------------------------------------------

def build_h3_mapping(hourly_csv_path, h3_graph_dir):
    from elephantgraph.training.dataset_coarse import load_h3_zoom
    zoom = load_h3_zoom(h3_graph_dir)
    df = pd.read_csv(hourly_csv_path, parse_dates=["timestamp"])
    df["h3_idx"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], zoom), axis=1
    )
    all_h3 = sorted(df["h3_idx"].unique())
    h3_to_id = {h: i for i, h in enumerate(all_h3)}
    id_to_h3 = {i: h for h, i in h3_to_id.items()}
    print(f"  H3 zoom={zoom}  ({len(all_h3)} unique cells)")
    return h3_to_id, id_to_h3, df


def load_model(checkpoint_path, num_h3_nodes, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ms   = ckpt["model_state"]
    d_model    = ms["encoder.node_embed.weight"].shape[0]
    num_layers = sum(1 for k in ms if k.startswith("encoder.encoder.layers.")
                     and k.endswith(".norm1.weight"))
    nhead = 8 if d_model % 8 == 0 else 4
    model = CoarseGenerator(
        d_model=d_model, nhead=nhead,
        num_layers=num_layers, num_h3_nodes=num_h3_nodes,
    ).to(device)
    model.load_state_dict(ms)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. Next-step accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_next_step_accuracy(model, node_embeddings, val_sequences,
                                behavior_codes, season_codes,
                                context_len, device):
    """
    For every (context → next_node) pair in val_sequences, run the model
    and check whether the true next node is in the top-1 / top-3 predictions.

    Returns dict with top1_acc, top3_acc, per_cell_top1 counts.
    """
    node_embeddings = node_embeddings.to(device)
    correct_top1 = 0
    correct_top3 = 0
    total        = 0
    per_cell_correct = Counter()
    per_cell_total   = Counter()

    for seq, beh, sea in zip(val_sequences, behavior_codes, season_codes):
        behavior = torch.tensor([beh], dtype=torch.long, device=device)
        season   = torch.tensor([sea], dtype=torch.long, device=device)

        for i in range(len(seq) - 1):
            ctx        = seq[max(0, i - context_len + 1): i + 1]
            true_next  = int(seq[i + 1])
            node_inp   = torch.tensor([ctx], dtype=torch.long, device=device)

            logits, _  = model(node_inp, node_embeddings, season, behavior)
            pred_probs  = torch.softmax(logits[0, -1, :], dim=-1)
            top3        = torch.topk(pred_probs, k=3).indices.tolist()

            if top3[0] == true_next:
                correct_top1 += 1
                per_cell_correct[true_next] += 1
            if true_next in top3:
                correct_top3 += 1

            per_cell_total[true_next] += 1
            total += 1

    return {
        "top1_acc": correct_top1 / total if total else 0,
        "top3_acc": correct_top3 / total if total else 0,
        "total_predictions": total,
        "per_cell_correct": per_cell_correct,
        "per_cell_total":   per_cell_total,
    }


# ---------------------------------------------------------------------------
# 2. Transition matrix comparison
# ---------------------------------------------------------------------------

def real_transition_matrix(sequences, num_nodes):
    """Empirical transition probability matrix from real val sequences."""
    counts = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


@torch.no_grad()
def model_transition_matrix(model, node_embeddings, num_nodes,
                              behavior_code, season_code, device):
    """
    Model's transition probability matrix:
    For each possible current node, run a length-1 context and record
    the softmax distribution over next nodes.
    """
    node_embeddings = node_embeddings.to(device)
    behavior = torch.tensor([behavior_code], dtype=torch.long, device=device)
    season   = torch.tensor([season_code],   dtype=torch.long, device=device)
    matrix   = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src in range(num_nodes):
        node_inp   = torch.tensor([[src]], dtype=torch.long, device=device)
        logits, _  = model(node_inp, node_embeddings, season, behavior)
        probs      = torch.softmax(logits[0, -1, :], dim=-1).cpu().numpy()
        matrix[src] = probs

    return matrix


# ---------------------------------------------------------------------------
# 3. Visit frequency
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sequence(model, node_embeddings, seed, steps,
                       behavior_code, season_code, context_len,
                       temperature, device):
    node_embeddings = node_embeddings.to(device)
    behavior = torch.tensor([behavior_code], dtype=torch.long, device=device)
    season   = torch.tensor([season_code],   dtype=torch.long, device=device)
    generated = [seed]
    for _ in range(steps):
        ctx      = generated[-context_len:]
        node_inp = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(node_inp, node_embeddings, season, behavior)
        probs    = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
        nxt      = torch.multinomial(probs, 1).item()
        generated.append(nxt)
    return generated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_transition_matrices(real_mat, model_mat, id_to_h3, output_dir):
    n = real_mat.shape[0]
    labels = [id_to_h3[i][-7:] for i in range(n)]  # last 7 chars of H3 ID

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, mat, title in zip(axes[:2],
                               [real_mat, model_mat],
                               ["Real transitions (val)", "Model transitions"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Next cell"); ax.set_ylabel("Current cell")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Difference plot
    diff = model_mat - real_mat
    im3  = axes[2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[2].set_yticks(range(n)); axes[2].set_yticklabels(labels, fontsize=7)
    axes[2].set_xlabel("Next cell"); axes[2].set_ylabel("Current cell")
    axes[2].set_title("Difference (model - real)\nBlue = underestimates, Red = overestimates")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    path = os.path.join(output_dir, "transition_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_visit_frequency(real_freq, gen_freq, id_to_h3, num_nodes, output_dir):
    labels = [id_to_h3[i][-7:] for i in range(num_nodes)]
    x = np.arange(num_nodes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, num_nodes), 4))
    ax.bar(x - width/2, real_freq, width, label="Real (val)", color="#3498db", alpha=0.85)
    ax.bar(x + width/2, gen_freq,  width, label="Generated",  color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Visit proportion")
    ax.set_title("Cell visit frequency: real vs generated")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = os.path.join(output_dir, "visit_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_per_cell_accuracy(per_cell_correct, per_cell_total, id_to_h3, output_dir):
    cells = sorted(per_cell_total.keys())
    accs  = [per_cell_correct[c] / per_cell_total[c] for c in cells]
    tots  = [per_cell_total[c] for c in cells]
    labels = [id_to_h3[c][-7:] for c in cells]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(cells)), 6), sharex=True)
    ax1.bar(range(len(cells)), accs, color="#2ecc71", alpha=0.85)
    ax1.set_ylabel("Top-1 accuracy")
    ax1.set_title("Per-cell top-1 next-step accuracy")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.axhline(np.mean(accs), color="red", linestyle="--", label=f"mean={np.mean(accs):.2%}")
    ax1.legend()

    ax2.bar(range(len(cells)), tots, color="#9b59b6", alpha=0.85)
    ax2.set_ylabel("# predictions")
    ax2.set_title("Prediction count per cell")
    ax2.set_xticks(range(len(cells)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_cell_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# 4. Free-run evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def free_run_evaluation(model, node_embeddings, val_seq,
                         behavior_code, season_code,
                         context_len, num_nodes,
                         n_runs, window_size, device):
    """
    Repeatedly seed the model with the first `context_len` real steps, then
    let it run completely free (no ground truth) for `len(val_seq)` steps.

    At each position t we record:
      - Whether the free-run prediction matches the real value (hit/miss)
      - KL divergence of cell frequencies in a rolling window centred on t

    Running n_runs times and averaging removes sampling noise.

    Returns:
        step_accuracy : np.ndarray (T,)  — fraction of runs that matched real at t
        rolling_kl    : np.ndarray (T,)  — mean KL(real_window || gen_window)
    """
    node_embeddings = node_embeddings.to(device)
    behavior = torch.tensor([behavior_code], dtype=torch.long, device=device)
    season   = torch.tensor([season_code],   dtype=torch.long, device=device)

    T = len(val_seq) - context_len  # steps evaluated after the seed
    hit_counts  = np.zeros(T, dtype=np.float32)
    kl_sums     = np.zeros(T, dtype=np.float32)

    half_w = window_size // 2

    for run in range(n_runs):
        # Seed with real context, then run completely free
        generated = list(val_seq[:context_len])

        for t in range(T):
            ctx      = generated[-context_len:]
            node_inp = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = model(node_inp, node_embeddings, season, behavior)
            probs    = torch.softmax(logits[0, -1, :], dim=-1)
            nxt      = torch.multinomial(probs, 1).item()
            generated.append(nxt)

            # Hit at this position
            true_val = int(val_seq[context_len + t])
            if nxt == true_val:
                hit_counts[t] += 1

            # Rolling KL: window of real vs window of generated around t
            real_start = max(0, context_len + t - half_w)
            real_end   = min(len(val_seq), context_len + t + half_w)
            gen_start  = max(0, len(generated) - 1 - half_w)

            real_window = val_seq[real_start:real_end]
            gen_window  = np.array(generated[gen_start:], dtype=np.int64)

            real_dist = np.bincount(real_window, minlength=num_nodes).astype(float)
            gen_dist  = np.bincount(gen_window,  minlength=num_nodes).astype(float)

            # Smooth to avoid log(0), then normalise
            real_dist = (real_dist + 1e-8) / (real_dist.sum() + num_nodes * 1e-8)
            gen_dist  = (gen_dist  + 1e-8) / (gen_dist.sum()  + num_nodes * 1e-8)

            kl = float(np.sum(real_dist * np.log(real_dist / gen_dist)))
            kl_sums[t] += kl

    step_accuracy = hit_counts / n_runs
    rolling_kl    = kl_sums   / n_runs
    return step_accuracy, rolling_kl


def plot_free_run(step_accuracy, rolling_kl, context_len, output_dir):
    T      = len(step_accuracy)
    steps  = np.arange(T)
    # Smooth with a rolling mean for readability
    def smooth(x, w=20):
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Accuracy decay
    ax1.plot(steps, step_accuracy,          color="#bdc3c7", lw=0.8, alpha=0.6, label="per-step")
    ax1.plot(steps, smooth(step_accuracy),  color="#2980b9", lw=2,   label="smoothed (w=20)")
    ax1.axhline(1 / 15, color="#e74c3c", linestyle="--", lw=1.2,
                label="random baseline (1/15)")
    ax1.set_ylabel("Fraction of runs correct")
    ax1.set_title("Free-run accuracy decay\n"
                  "(seeded with real context, then model runs on own predictions)")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(fontsize=8)

    # KL divergence growth
    ax2.plot(steps, rolling_kl,          color="#bdc3c7", lw=0.8, alpha=0.6)
    ax2.plot(steps, smooth(rolling_kl),  color="#e67e22", lw=2,   label="smoothed (w=20)")
    ax2.axhline(0, color="#2ecc71", linestyle="--", lw=1.2, label="perfect match (KL=0)")
    ax2.set_ylabel("KL divergence (real || generated)")
    ax2.set_xlabel(f"Steps after seed (seed length = {context_len})")
    ax2.set_title("Rolling cell-frequency KL divergence over time")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "free_run_degradation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default=os.path.join(PROJECT_DIR, "checkpoints", "coarse", "best_coarse_model.pt"))
    parser.add_argument("--hourly-csv",  default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "hourly", "hourly.csv"))
    parser.add_argument("--node-emb",    default=os.path.join(PROJECT_DIR, "elephantgraph", "data", "processed", "h3_graph", "node_embeddings.npy"))
    parser.add_argument("--train-frac",  type=float, default=0.8)
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--context-len", type=int,   default=23)
    parser.add_argument("--steps",       type=int,   default=500,  help="Generated sequence length for frequency comparison")
    parser.add_argument("--free-runs",   type=int,   default=20,   help="Number of free-run trials to average (check 4)")
    parser.add_argument("--window-size", type=int,   default=48,   help="Rolling window size for KL divergence (check 4)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--output",      default=os.path.join(PROJECT_DIR, "results", "coarse_eval"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading data ...")
    h3_graph_dir = os.path.dirname(args.node_emb)
    h3_to_id, id_to_h3, df = build_h3_mapping(args.hourly_csv, h3_graph_dir)
    num_nodes = len(h3_to_id)

    emb_dict = np.load(args.node_emb, allow_pickle=True).item()
    embed_dim = next(iter(emb_dict.values())).shape[0]
    node_embeddings = torch.zeros(num_nodes, embed_dim)
    for hex_str, emb in emb_dict.items():
        if hex_str in h3_to_id:
            node_embeddings[h3_to_id[hex_str]] = torch.tensor(emb, dtype=torch.float32)

    # Val sequences: last 10% of each elephant's timeline
    val_sequences, behavior_codes, season_codes = [], [], []
    for eid, edf in df.groupby("elephant_id"):
        edf = edf.sort_values("timestamp").reset_index(drop=True)
        n = len(edf)
        i_val  = int(n * args.train_frac)
        i_test = int(n * (args.train_frac + args.val_frac))
        seg = edf.iloc[i_val:i_test]
        if len(seg) < 2:
            continue
        seq = seg["h3_idx"].map(h3_to_id).values.astype(np.int64)
        val_sequences.append(seq)
        behavior_codes.append(int(seg["behavior_code"].mode()[0]))
        season_codes.append(int(seg["season_code"].mode()[0]))
        print(f"  {eid}: {len(seq)} val steps")

    total_val_steps = sum(len(s) for s in val_sequences)
    print(f"Total val steps across {len(val_sequences)} elephants: {total_val_steps}")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading model ...")
    model = load_model(args.checkpoint, num_nodes, device)

    # ── 1. Next-step accuracy ────────────────────────────────────────────────
    print("\n[1] Next-step accuracy ...")
    acc = compute_next_step_accuracy(
        model, node_embeddings,
        val_sequences=val_sequences,
        behavior_codes=behavior_codes,
        season_codes=season_codes,
        context_len=args.context_len,
        device=device,
    )
    print(f"  Top-1 accuracy : {acc['top1_acc']:.2%}  ({acc['total_predictions']} predictions)")
    print(f"  Top-3 accuracy : {acc['top3_acc']:.2%}")
    plot_per_cell_accuracy(acc["per_cell_correct"], acc["per_cell_total"], id_to_h3, args.output)

    # ── 2. Transition matrices ────────────────────────────────────────────────
    # Use dominant behavior/season across all val sequences for the model matrix
    all_val_beh = max(set(behavior_codes), key=behavior_codes.count)
    all_val_sea = max(set(season_codes),   key=season_codes.count)
    print("\n[2] Transition matrices ...")
    real_mat  = real_transition_matrix(val_sequences, num_nodes)
    model_mat = model_transition_matrix(model, node_embeddings, num_nodes,
                                         all_val_beh, all_val_sea, device)
    mae = np.abs(real_mat - model_mat).mean()
    print(f"  Transition MAE : {mae:.4f}  (0 = perfect, 1 = worst)")
    plot_transition_matrices(real_mat, model_mat, id_to_h3, args.output)

    # ── 3. Visit frequency ────────────────────────────────────────────────────
    print(f"\n[3] Visit frequency ({args.steps} generated steps) ...")
    seed = int(val_sequences[0][0])
    gen_seq = generate_sequence(
        model, node_embeddings, seed, args.steps,
        all_val_beh, all_val_sea, args.context_len, args.temperature, device,
    )

    all_val_nodes = np.concatenate(val_sequences)
    real_counts = Counter(all_val_nodes.tolist())
    gen_counts  = Counter(gen_seq)
    real_freq   = np.array([real_counts.get(i, 0) for i in range(num_nodes)], dtype=float)
    gen_freq    = np.array([gen_counts.get(i,  0) for i in range(num_nodes)], dtype=float)
    real_freq  /= real_freq.sum()
    gen_freq   /= gen_freq.sum()

    freq_mae = np.abs(real_freq - gen_freq).mean()
    print(f"  Frequency MAE  : {freq_mae:.4f}  (0 = perfect match)")
    for i in range(num_nodes):
        if real_freq[i] > 0 or gen_freq[i] > 0:
            print(f"    {id_to_h3[i][-10:]}  real={real_freq[i]:.3f}  gen={gen_freq[i]:.3f}  "
                  f"diff={gen_freq[i]-real_freq[i]:+.3f}")
    plot_visit_frequency(real_freq, gen_freq, id_to_h3, num_nodes, args.output)

    # ── 4. Free-run degradation (use longest val sequence) ───────────────────
    longest_idx = max(range(len(val_sequences)), key=lambda i: len(val_sequences[i]))
    longest_seq = val_sequences[longest_idx]
    longest_beh = behavior_codes[longest_idx]
    longest_sea = season_codes[longest_idx]
    print(f"\n[4] Free-run degradation ({args.free_runs} runs x "
          f"{len(longest_seq) - args.context_len} steps) ...")
    step_accuracy, rolling_kl = free_run_evaluation(
        model, node_embeddings, longest_seq,
        longest_beh, longest_sea,
        context_len=args.context_len,
        num_nodes=num_nodes,
        n_runs=args.free_runs,
        window_size=args.window_size,
        device=device,
    )
    # Accuracy at different horizons
    horizons = [1, 6, 24, 72, len(step_accuracy) - 1]
    for h in horizons:
        if h < len(step_accuracy):
            print(f"  Accuracy at step +{h:4d} : {step_accuracy[h]:.2%}  "
                  f"(KL={rolling_kl[h]:.4f})")
    plot_free_run(step_accuracy, rolling_kl, args.context_len, args.output)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"  Top-1 next-step accuracy (teacher-forced) : {acc['top1_acc']:.2%}")
    print(f"  Top-3 next-step accuracy (teacher-forced) : {acc['top3_acc']:.2%}")
    print(f"  Transition matrix MAE                     : {mae:.4f}")
    print(f"  Visit frequency MAE                       : {freq_mae:.4f}")
    print(f"  Free-run acc at  +1 step                  : {step_accuracy[min(1,  len(step_accuracy)-1)]:.2%}")
    print(f"  Free-run acc at +24 steps                 : {step_accuracy[min(24, len(step_accuracy)-1)]:.2%}")
    print(f"  Free-run KL  at +24 steps                 : {rolling_kl[min(24,  len(rolling_kl)-1)]:.4f}")
    print(f"  Plots saved to                            : {args.output}/")


if __name__ == "__main__":
    main()
