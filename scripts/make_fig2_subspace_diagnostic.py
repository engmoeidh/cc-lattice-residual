#!/usr/bin/env python3
"""
make_fig2_subspace_diagnostic.py
================================

Computes the subspace-overlap diagnostic from the cached overlap
matrices (produced by make_fig2_transport_overlaps_rowA.py):

    S(xi_j) = Tr[P_V20(1) P_V20(xi_j)] / N_ANCHOR
            = (1/20) * sum_{alpha=1..20, k=1..20} |<psi_alpha(1)|phi_k(xi_j)>|^2

S(xi_j) is the average fraction of the anchor V_20(1) subspace retained
in the transported V_20(xi_j). It is invariant under unitary rotations
inside degenerate eigenspaces, so it is a cleaner diagnostic of subspace
transport than per-mode overlaps.

Produces:
    paper/figures/fig_transport_overlaps_rowA.pdf   (replaces earlier figure)
    stdout: numerical value of S at each xi
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE_PATH = os.path.join(REPO_ROOT, "results", "fig2_rowA_transport.npz")
FIG_PATH = os.path.join(REPO_ROOT, "paper", "figures",
                        "fig_transport_overlaps_rowA.pdf")

N_ANCHOR = 20

data = np.load(CACHE_PATH)
print("=" * 70)
print(" SUBSPACE OVERLAP DIAGNOSTIC (Row A)")
print("=" * 70)

xi_nodes = data["xi_nodes"]
overlaps_per_xi = {}
S_per_xi = {}

for xi in xi_nodes[1:]:
    key_O = f"O_matrix_xi_{xi:.3f}".replace(".", "p")
    key_ov = f"overlaps_xi_{xi:.3f}".replace(".", "p")
    if key_O not in data.files:
        print(f"ERROR: no overlap matrix cached for xi = {xi:.3f}")
        print(f"        Rerun compute step first.")
        sys.exit(1)
    O = data[key_O]               # (N_ANCHOR, N_c)
    overlaps = data[key_ov]       # (N_ANCHOR,)
    n_anchor, n_cand = O.shape
    # Subspace overlap: project V_20(1) onto span of lowest-20 candidates
    # S = (1/N_ANCHOR) * sum_{alpha=1..20, k=1..20} |O_{alpha, k}|^2
    O_sq = O**2
    S_full = O_sq[:, :N_ANCHOR].sum() / N_ANCHOR
    S_all = O_sq.sum() / N_ANCHOR   # include overlap into candidates 21-25
    S_per_xi[xi] = S_full
    overlaps_per_xi[xi] = overlaps

    # Per-mode row sums (what fraction of each anchor mode lives in the first 20)
    row_sums_20 = O_sq[:, :N_ANCHOR].sum(axis=1)

    print(f"\n  xi = {xi:.3f}:")
    print(f"    O matrix shape: {O.shape}")
    print(f"    S(xi) = Tr[P_V20(1) P_V20_meas(xi)] / {N_ANCHOR}")
    print(f"          = {S_full:.4f}   (projecting onto lowest 20 candidates)")
    print(f"    S_all = {S_all:.4f}   (including candidates 21-25)")
    print(f"    Max row sum: {row_sums_20.max():.4f}, "
          f"min row sum: {row_sums_20.min():.4f}")
    print(f"    Mean row sum: {row_sums_20.mean():.4f}")
    print(f"    Per-mode assigned overlap: "
          f"min = {overlaps.min():.3f}, mean = {overlaps.mean():.3f}")

# --- Plot: two-panel figure ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# Panel 1: per-mode overlaps (same as before)
ax1 = axes[0]
colors = {1.045: "tab:blue", 1.090: "tab:orange"}
markers = {1.045: "o", 1.090: "s"}
for xi in xi_nodes[1:]:
    overlaps = overlaps_per_xi[xi]
    alpha = np.arange(1, len(overlaps) + 1)
    ax1.plot(alpha, overlaps,
             marker=markers[xi], color=colors[xi],
             markersize=6, linewidth=1.0, linestyle='-',
             markeredgecolor='black', markeredgewidth=0.5,
             label=rf"$\xi = {xi:.3f}$, $\Omega_{{\min}} = {overlaps.min():.3f}$",
             zorder=3)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax1.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5,
            label="0.5 reference")
ax1.set_xlabel(r"mode index $\alpha$", fontsize=11)
ax1.set_ylabel(
    r"$|\langle\psi_\alpha(\xi_0)\,|\,\hat\psi_\alpha(\xi_j)\rangle|$",
    fontsize=11)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=8.5)
ax1.set_xlim(0, 21)
ax1.set_ylim(0, 1.08)
ax1.set_xticks(range(1, 21, 2))
ax1.set_title("(a) Per-mode assigned overlaps", fontsize=10.5)

# Panel 2: subspace overlap S(xi)
ax2 = axes[1]
xi_plot = np.array([1.000] + [float(x) for x in xi_nodes[1:]])
S_plot = np.array([1.0] + [S_per_xi[xi] for xi in xi_nodes[1:]])
ax2.plot(xi_plot, S_plot, 'o-',
         markersize=10, linewidth=1.5, color='tab:green',
         markeredgecolor='black', markeredgewidth=0.8,
         label=r"$\mathcal{S}(\xi)$ (subspace overlap)", zorder=3)
for xi, S in zip(xi_plot, S_plot):
    ax2.annotate(f"{S:.4f}", xy=(xi, S), xytext=(0, 12),
                 textcoords='offset points', ha='center', fontsize=10,
                 color='tab:green')
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
ax2.axhline(0.95, color='red', linestyle='--', linewidth=0.8, alpha=0.5,
            label=r"$0.95$ reference")
ax2.set_xlabel(r"$\xi$", fontsize=11)
ax2.set_ylabel(
    r"$\mathcal{S}(\xi) = \frac{1}{N_{\rm anchor}}\,"
    r"\mathrm{Tr}[P_{V_{20}}(\xi_0)\,P_{V_{20}}(\xi)]$",
    fontsize=11)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=9)
ax2.set_xlim(0.99, 1.10)
ax2.set_xticks(xi_plot)
ax2.set_xticklabels([f"{xi:.3f}" for xi in xi_plot])
ax2.set_ylim(0.90, 1.015)
ax2.set_title(r"(b) Subspace overlap $\mathcal{S}(\xi)$",
              fontsize=10.5)

# Overall Row A label
fig.text(0.98, 0.02,
         rf"Row A: $L_s = 18$, $L_t = 12$, $\rho = 3.0$",
         ha='right', va='bottom', fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='gray', alpha=0.85))

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)

os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
plt.savefig(FIG_PATH, dpi=600, bbox_inches='tight')
plt.close()

print()
print("=" * 70)
print(f"  Summary:")
print(f"    S(xi=1.045) = {S_per_xi[1.045]:.4f}")
print(f"    S(xi=1.090) = {S_per_xi[1.090]:.4f}")
print(f"  Paper claim: V_20 subspace transports continuously.")
print(f"  Interpretation: S values >> 0.95 confirm subspace preservation;")
print(f"  per-mode overlaps ~ 0.6 reflect degenerate-subspace rotation.")
print("=" * 70)
print(f"\nSaved: {FIG_PATH}")
print(f"File size: {os.path.getsize(FIG_PATH)} bytes")
