#!/usr/bin/env python3
"""Workshop-paper figures from the finished 45-cell grid (OPUS_TASKS T11, $0).

Reads results/grid/grid_*.json (9 safety bins x 5 goal bins per model) and renders:
  F9_grid_heatmaps.png  — per model, 3 heatmaps (HA, CP, MB) over the 9x5 cells.
  F3_grid_marginals.png — knob marginals: HA vs safety bin, CP vs goal bin,
                           HA vs goal bin (push erodes HA), one line per model.

No API. Saves to paper/figures/. Does not touch the other paper's F3_sensitivity.png.
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
NI, NJ = 9, 5
SHORT = {"qwen/qwen3-32b": "qwen", "meta-llama/llama-3.3-70b-instruct": "llama",
         "google/gemini-2.5-flash-lite": "gemini",
         "mistralai/mistral-small-3.2-24b-instruct": "mistral"}
ORDER = ["qwen", "llama", "gemini", "mistral"]


def harmonic(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def load_grids():
    grids = {}
    for f in glob.glob(os.path.join(ROOT, "results", "grid", "grid_*.json")):
        d = json.load(open(f))
        model = d["model"]; short = SHORT.get(model, model)
        HA = [[None] * NJ for _ in range(NI)]
        CP = [[None] * NJ for _ in range(NI)]
        MB = [[None] * NJ for _ in range(NI)]
        for r in d["records"]:
            i, j = r["cell"]
            HA[i][j] = r["ha"]; CP[i][j] = r["cp"]; MB[i][j] = harmonic(r["ha"], r["cp"])
        grids[short] = {"HA": HA, "CP": CP, "MB": MB, "model": model}
    return grids


def f9(grids):
    models = [m for m in ORDER if m in grids]
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.1 * len(models)))
    if len(models) == 1:
        axes = [axes]
    metrics = ["HA", "CP", "MB"]
    for rr, m in enumerate(models):
        for cc, met in enumerate(metrics):
            ax = axes[rr][cc]
            mat = grids[m][met]
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis",
                           vmin=0, vmax=100)
            for i in range(NI):
                for j in range(NJ):
                    v = mat[i][j]
                    if v is not None:
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                                fontsize=6, color="white" if v < 55 else "black")
            ax.set_title(f"{m} — {met}", fontsize=9)
            ax.set_xlabel("goal bin j (push)", fontsize=7)
            ax.set_ylabel("safety bin i", fontsize=7)
            ax.set_xticks(range(NJ)); ax.set_yticks(range(NI))
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("45-cell proxy landscape per model (stratified proxy, b10/h5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{FIG}/F9_grid_heatmaps.png", dpi=150); plt.close(fig); print("F9 done")


def _mean_over(mat, axis):
    """axis='j' -> mean over goal for each safety bin i (len NI);
       axis='i' -> mean over safety for each goal bin j (len NJ)."""
    if axis == "j":
        return [sum(v for v in mat[i] if v is not None) /
                max(1, sum(1 for v in mat[i] if v is not None)) for i in range(NI)]
    return [sum(mat[i][j] for i in range(NI) if mat[i][j] is not None) /
            max(1, sum(1 for i in range(NI) if mat[i][j] is not None)) for j in range(NJ)]


def f3(grids):
    models = [m for m in ORDER if m in grids]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.5))
    for m in models:
        a1.plot(range(NI), _mean_over(grids[m]["HA"], "j"), "-o", label=m)
        a2.plot(range(NJ), _mean_over(grids[m]["CP"], "i"), "-o", label=m)
        a3.plot(range(NJ), _mean_over(grids[m]["HA"], "i"), "-o", label=m)
    a1.set_title("HA vs safety bin (mean over goal)"); a1.set_xlabel("safety bin i (sw↑)")
    a1.set_ylabel("HA %"); a1.set_ylim(0, 100); a1.grid(alpha=.3); a1.legend(fontsize=8)
    a2.set_title("CP vs goal bin (mean over safety)"); a2.set_xlabel("goal bin j (push↑)")
    a2.set_ylabel("CP %"); a2.set_ylim(0, 100); a2.grid(alpha=.3); a2.legend(fontsize=8)
    a3.set_title("HA vs goal bin (push erodes HA)"); a3.set_xlabel("goal bin j (push↑)")
    a3.set_ylabel("HA %"); a3.set_ylim(0, 100); a3.grid(alpha=.3); a3.legend(fontsize=8)
    fig.suptitle("Knob marginals from the 45-cell grid", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG}/F3_grid_marginals.png", dpi=150); plt.close(fig); print("F3 marginals done")


if __name__ == "__main__":
    grids = load_grids()
    print("loaded grids:", sorted(grids))
    f9(grids); f3(grids)
    print("FIGS:", [x for x in sorted(os.listdir(FIG)) if x.startswith(("F9_grid", "F3_grid"))])
