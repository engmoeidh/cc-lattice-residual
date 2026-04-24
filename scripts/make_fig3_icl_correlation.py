#!/usr/bin/env python3
"""
make_fig3_icl_correlation.py
============================

Generates Figure 3 of the paper: I_cl-degraded reference rows (D1-D4)
with the weighted-linear-regression guide, using chi^2/dof error
inflation to account for scatter.

Produces:
    paper/figures/fig_icl_correlation.pdf

Data source:
    Table 11 of CC_2.tex (four reference rows, Simpson measurements).

Conventions:
    y = delta_xi log zeta / epsilon       (dlz_full convention)
    x = I_cl - 1
    eps = 0.09

Fit:
    y(x) = a + b*x, weighted least squares with weights w_i = 1/sigma_i^2
    chi^2/dof reported; extrapolation uncertainty inflated by
    sqrt(chi^2/dof) when > 1 to account for un-modeled scatter.

Cross-check:
    At I_cl = 1, extrapolation gives delta_xi log zeta = a*eps,
    compared against primary result -1.669 +/- 0.125 (Eq. 18).

Runtime:
    < 3 seconds on any CPU. No GPU, no lattice computation.

Usage:
    python make_fig3_icl_correlation.py

Author: M. M. Hanash
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Data: Table 11 of CC_2.tex ---
rows = [
    {"label": "D1", "dim": "8^3 x 8",   "rho": 3.0, "Icl": 1.201,
     "dlz": -2.240, "err": 0.168},
    {"label": "D2", "dim": "12^3 x 8",  "rho": 3.0, "Icl": 1.068,
     "dlz": -2.095, "err": 0.174},
    {"label": "D3", "dim": "18^3 x 12", "rho": 4.5, "Icl": 1.218,
     "dlz": -2.802, "err": 0.168},
    {"label": "D4", "dim": "24^3 x 16", "rho": 6.0, "Icl": 1.329,
     "dlz": -3.600, "err": 0.384},
]

epsilon = 0.09
primary_dlz      = -1.669    # Eq. 18 of CC_2.tex (three-row weighted)
primary_dlz_err  =  0.125

Icl  = np.array([r["Icl"]  for r in rows])
y    = np.array([r["dlz"]  for r in rows]) / epsilon
yerr = np.array([r["err"]  for r in rows]) / epsilon
x    = Icl - 1.0

# --- Weighted least-squares fit ---
w = 1.0 / yerr**2
Sw   = w.sum()
Swx  = (w * x).sum()
Swy  = (w * y).sum()
Swxx = (w * x**2).sum()
Swxy = (w * x * y).sum()

det   = Sw * Swxx - Swx**2
slope = (Sw * Swxy - Swx * Swy) / det
intr  = (Swxx * Swy - Swx * Swxy) / det

var_a = Swxx / det
var_b = Sw / det
sig_a_raw = np.sqrt(var_a)
sig_b_raw = np.sqrt(var_b)

y_pred   = intr + slope * x
residual = y - y_pred
sigmas   = residual / yerr
chi2     = (sigmas**2).sum()
dof      = len(rows) - 2
chi2_red = chi2 / dof if dof > 0 else float('nan')

# --- Chi^2/dof inflation (Option X) ---
inflation = np.sqrt(max(chi2_red, 1.0))
sig_a_inflated = sig_a_raw * inflation
sig_b_inflated = sig_b_raw * inflation

# --- Extrapolation with inflated uncertainty ---
y_extrap           = intr
dlz_extrap         = y_extrap * epsilon
dlz_extrap_err_raw      = sig_a_raw * epsilon
dlz_extrap_err_inflated = sig_a_inflated * epsilon

tension_raw      = abs(dlz_extrap - primary_dlz) / np.sqrt(
    dlz_extrap_err_raw**2 + primary_dlz_err**2)
tension_inflated = abs(dlz_extrap - primary_dlz) / np.sqrt(
    dlz_extrap_err_inflated**2 + primary_dlz_err**2)

# --- Stdout diagnostic ---
print("=" * 70)
print(" WEIGHTED LINEAR REGRESSION: delta_xi log zeta / eps  vs  I_cl - 1")
print("=" * 70)
print()
print(f"  a (intercept) = {intr:+8.3f} +/- {sig_a_raw:5.3f}  (raw)")
print(f"                = {intr:+8.3f} +/- {sig_a_inflated:5.3f}  (inflated by sqrt(chi^2/dof))")
print(f"  b (slope)     = {slope:+8.3f} +/- {sig_b_raw:5.3f}  (raw)")
print(f"                = {slope:+8.3f} +/- {sig_b_inflated:5.3f}  (inflated)")
print(f"  chi^2 = {chi2:6.3f}   dof = {dof}   chi^2/dof = {chi2_red:5.3f}")
print(f"  inflation factor sqrt(chi^2/dof) = {inflation:5.3f}")
print()
print("  Per-row residuals against fit:")
print(f"    {'Row':<4} {'I_cl':>6} {'y_meas':>8} {'y_pred':>8} "
      f"{'resid':>7} {'sigma':>7}")
for i, r in enumerate(rows):
    print(f"    {r['label']:<4} {Icl[i]:>6.3f} {y[i]:>+8.3f} "
          f"{y_pred[i]:>+8.3f} {residual[i]:>+7.3f} {sigmas[i]:>+7.2f}")
print()
print("  Continuum extrapolation at I_cl = 1:")
print(f"    delta_xi log zeta = {dlz_extrap:+7.4f} +/- {dlz_extrap_err_raw:6.4f}  (raw)")
print(f"    delta_xi log zeta = {dlz_extrap:+7.4f} +/- {dlz_extrap_err_inflated:6.4f}  (inflated)")
print(f"    primary result     = {primary_dlz:+7.4f} +/- {primary_dlz_err:6.4f}")
print(f"    tension (raw)      = {tension_raw:5.3f} sigma")
print(f"    tension (inflated) = {tension_inflated:5.3f} sigma")
print("=" * 70)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6.2, 4.6))

ax.errorbar(x, y, yerr=yerr, fmt='o', color='black',
            markersize=7, capsize=4, elinewidth=1.2,
            markeredgecolor='black', markerfacecolor='white',
            label="Reference rows (D1-D4)", zorder=3)

x_line = np.linspace(-0.05, 0.37, 200)
y_line = intr + slope * x_line
guide_label = (rf"Fit: ${intr:.1f} {slope:+.1f}\,(I_{{\rm cl}} - 1)$"
               f",  $\\chi^2/\\mathrm{{dof}} = {chi2_red:.1f}$")
ax.plot(x_line, y_line, '-', color='tab:blue', linewidth=1.4,
        label=guide_label, zorder=2)

# Vertical line at I_cl = 1 with inflated-error annotation
ax.axvline(0, color='tab:red', linestyle='--', linewidth=1.0, alpha=0.7,
           zorder=1)

# Show inflated error band at extrapolation point
ax.errorbar([0], [intr], yerr=[sig_a_inflated], fmt='s',
            color='tab:red', markersize=6, capsize=4, elinewidth=1.2,
            markeredgecolor='tab:red', markerfacecolor='tab:red',
            zorder=3, alpha=0.85)

ax.annotate(
    f"$I_{{\\rm cl}} = 1$:\n"
    f"$\\delta_\\xi\\!\\log\\zeta = {dlz_extrap:.2f} \\pm {dlz_extrap_err_inflated:.2f}$",
    xy=(0, intr), xytext=(0.03, intr + 6),
    fontsize=9.5, color='tab:red',
    arrowprops=dict(arrowstyle='->', color='tab:red', lw=0.8))

label_offsets = {"D1": (8, -8), "D2": (8, 5), "D3": (8, 6), "D4": (8, 5)}
for r, xi_, yi_ in zip(rows, x, y):
    dx, dy = label_offsets.get(r["label"], (7, 5))
    ax.annotate(r["label"], xy=(xi_, yi_),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=11)

ax.set_xlabel(r"$I_{\rm cl} - 1$", fontsize=12)
ax.set_ylabel(r"$\delta_\xi \log\zeta \,/\, \epsilon$", fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=9.0)

ax.set_xlim(-0.04, 0.36)
ax.set_ylim(-45, -10)

plt.tight_layout()

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "paper", "figures")
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "fig_icl_correlation.pdf")

plt.savefig(outpath, dpi=600, bbox_inches='tight')
plt.close()

print()
print(f"Saved: {outpath}")
print(f"File size: {os.path.getsize(outpath)} bytes")
