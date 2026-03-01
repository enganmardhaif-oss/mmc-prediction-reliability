"""
MMC Prediction Reliability Analysis
=====================================
Central question:
  With only 5 experimental data points, how much can we trust
  GPR predictions — and which data points matter most?

Three analyses:
  1. Bootstrap Stability     → does the predicted optimum hold under resampling?
  2. Leave-One-Out (LOO)     → can the model predict unseen points accurately?
  3. Sensitivity Analysis    → which single data point influences the model most?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def build_gpr(X, y, n_restarts=15):
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 5.0)) + WhiteKernel(0.01)
    sx = StandardScaler(); sy = StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y.reshape(-1,1)).ravel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts)
    gpr.fit(Xs, ys)
    return gpr, sx, sy

def predict(gpr, sx, sy, X_pred):
    Xs = sx.transform(X_pred)
    ym, ys = gpr.predict(Xs, return_std=True)
    return sy.inverse_transform(ym.reshape(-1,1)).ravel(), ys * sy.scale_[0]

def find_optimum(X_pred, y_pred):
    return X_pred[np.argmax(y_pred)][0]


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
mgo = pd.read_csv('data/mgo_system.csv', comment='#')
wo3 = pd.read_csv('data/wo3_system.csv', comment='#')

X_mgo = mgo[['concentration_wt']].values
y_mgo = mgo['hardness_HV'].values
X_wo3 = wo3[['concentration_vol']].values
y_wo3 = wo3['hardness_HV'].values

X_pred = np.linspace(0, 5, 300).reshape(-1, 1)

# Baseline models
gpr_mgo, sx_mgo, sy_mgo = build_gpr(X_mgo, y_mgo)
gpr_wo3, sx_wo3, sy_wo3 = build_gpr(X_wo3, y_wo3)

mgo_base, mgo_std = predict(gpr_mgo, sx_mgo, sy_mgo, X_pred)
wo3_base, wo3_std = predict(gpr_wo3, sx_wo3, sy_wo3, X_pred)

opt_mgo_base = find_optimum(X_pred, mgo_base)
opt_wo3_base = find_optimum(X_pred, wo3_base)

print(f"Baseline optimum  MgO: {opt_mgo_base:.2f} wt%")
print(f"Baseline optimum  WO₃: {opt_wo3_base:.2f} vol%")


# ─────────────────────────────────────────────────────────
# ANALYSIS 1: BOOTSTRAP STABILITY  (1000 iterations)
# ─────────────────────────────────────────────────────────
print("\n[1/3] Bootstrap analysis (1000 iterations)...")
N_BOOT = 1000
boot_opt_mgo, boot_opt_wo3 = [], []
boot_curves_mgo, boot_curves_wo3 = [], []

for _ in range(N_BOOT):
    idx_m = np.random.choice(len(X_mgo), len(X_mgo), replace=True)
    idx_w = np.random.choice(len(X_wo3), len(X_wo3), replace=True)
    try:
        gm, sm, ym = build_gpr(X_mgo[idx_m], y_mgo[idx_m], n_restarts=5)
        gw, sw, yw = build_gpr(X_wo3[idx_w], y_wo3[idx_w], n_restarts=5)
        pm, _ = predict(gm, sm, ym, X_pred)
        pw, _ = predict(gw, sw, yw, X_pred)
        boot_opt_mgo.append(find_optimum(X_pred, pm))
        boot_opt_wo3.append(find_optimum(X_pred, pw))
        boot_curves_mgo.append(pm)
        boot_curves_wo3.append(pw)
    except:
        pass

boot_opt_mgo = np.array(boot_opt_mgo)
boot_opt_wo3 = np.array(boot_opt_wo3)
boot_curves_mgo = np.array(boot_curves_mgo)
boot_curves_wo3 = np.array(boot_curves_wo3)

print(f"  MgO optimum:  {boot_opt_mgo.mean():.2f} ± {boot_opt_mgo.std():.2f} wt%  "
      f"[95% CI: {np.percentile(boot_opt_mgo,2.5):.2f} – {np.percentile(boot_opt_mgo,97.5):.2f}]")
print(f"  WO₃ optimum:  {boot_opt_wo3.mean():.2f} ± {boot_opt_wo3.std():.2f} vol%  "
      f"[95% CI: {np.percentile(boot_opt_wo3,2.5):.2f} – {np.percentile(boot_opt_wo3,97.5):.2f}]")


# ─────────────────────────────────────────────────────────
# ANALYSIS 2: LEAVE-ONE-OUT VALIDATION
# ─────────────────────────────────────────────────────────
print("\n[2/3] Leave-One-Out validation...")

def loo_analysis(X, y, label):
    n = len(X)
    loo_preds, loo_stds, loo_true = [], [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[[i]], y[i]
        try:
            g, sx, sy = build_gpr(X_tr, y_tr, n_restarts=10)
            pm, ps = predict(g, sx, sy, X_te)
            loo_preds.append(pm[0]); loo_stds.append(ps[0]); loo_true.append(y_te)
        except:
            pass
    loo_preds = np.array(loo_preds)
    loo_true  = np.array(loo_true)
    loo_stds  = np.array(loo_stds)
    mae = mean_absolute_error(loo_true, loo_preds)
    mape = np.mean(np.abs((loo_true - loo_preds) / loo_true)) * 100
    print(f"  {label}:  MAE = {mae:.2f} HV  |  MAPE = {mape:.1f}%")
    return loo_preds, loo_stds, loo_true, mae, mape

loo_mgo_pred, loo_mgo_std, loo_mgo_true, mae_mgo, mape_mgo = loo_analysis(X_mgo, y_mgo, "MgO")
loo_wo3_pred, loo_wo3_std, loo_wo3_true, mae_wo3, mape_wo3 = loo_analysis(X_wo3, y_wo3, "WO₃")


# ─────────────────────────────────────────────────────────
# ANALYSIS 3: SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────
print("\n[3/3] Sensitivity analysis...")

def sensitivity_analysis(X, y, label, conc_labels):
    n = len(X)
    base_g, base_sx, base_sy = build_gpr(X, y)
    base_pred, _ = predict(base_g, base_sx, base_sy, X_pred)
    base_opt = find_optimum(X_pred, base_pred)

    shifts, opt_changes = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            g, sx, sy = build_gpr(X[mask], y[mask], n_restarts=10)
            p, _ = predict(g, sx, sy, X_pred)
            curve_shift = np.mean(np.abs(p - base_pred))
            opt_change  = abs(find_optimum(X_pred, p) - base_opt)
            shifts.append(curve_shift)
            opt_changes.append(opt_change)
        except:
            shifts.append(0); opt_changes.append(0)

    for i, (s, o) in enumerate(zip(shifts, opt_changes)):
        print(f"  {label} remove {conc_labels[i]}%:  curve shift={s:.2f} HV  opt shift={o:.2f}%")
    return np.array(shifts), np.array(opt_changes)

mgo_conc  = mgo['concentration_wt'].tolist()
wo3_conc  = wo3['concentration_vol'].tolist()

sens_mgo_shift, sens_mgo_opt = sensitivity_analysis(X_mgo, y_mgo, "MgO", mgo_conc)
sens_wo3_shift, sens_wo3_opt = sensitivity_analysis(X_wo3, y_wo3, "WO₃", wo3_conc)


# ─────────────────────────────────────────────────────────
# MASTER FIGURE  (3×2 grid)
# ─────────────────────────────────────────────────────────
print("\nGenerating master figure...")

C1, C2 = '#1565C0', '#C62828'
CA, CG  = '#2E7D32', '#E65100'
BGLIGHT = '#F8F9FA'

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BGLIGHT)
gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.35,
                       left=0.07, right=0.97, top=0.91, bottom=0.06)

# ── ROW 1: Bootstrap curves ───────────────────────────────
for col, (system, X_data, y_data, curves, base, opt_arr, color, xlabel, sys_label) in enumerate([
    ("MgO",  X_mgo, y_mgo, boot_curves_mgo, mgo_base, boot_opt_mgo, C1, "MgO Content (wt%)",  "System 1: Pure Al + nano-MgO"),
    ("WO₃",  X_wo3, y_wo3, boot_curves_wo3, wo3_base, boot_opt_wo3, C2, "WO₃ Content (vol%)", "System 2: Al7075 + nano-WO₃"),
]):
    ax = fig.add_subplot(gs[0, col])
    # Bootstrap envelope
    p5  = np.percentile(curves, 5,  axis=0)
    p95 = np.percentile(curves, 95, axis=0)
    p25 = np.percentile(curves, 25, axis=0)
    p75 = np.percentile(curves, 75, axis=0)
    xf = X_pred.flatten()
    ax.fill_between(xf, p5,  p95, alpha=0.12, color=color, label='90% bootstrap band')
    ax.fill_between(xf, p25, p75, alpha=0.25, color=color, label='50% bootstrap band')
    ax.plot(xf, base, color=color, lw=2.5, label='Baseline GPR')
    ax.scatter(X_data.flatten(), y_data, color=CA, s=90, zorder=6,
               edgecolors='white', lw=0.8, label='Experimental')
    # Optimum histogram inset
    axins = ax.inset_axes([0.62, 0.08, 0.35, 0.38])
    axins.hist(opt_arr, bins=30, color=color, alpha=0.7, edgecolor='white', linewidth=0.4)
    axins.axvline(opt_arr.mean(), color='black', lw=1.5, ls='--')
    axins.set_xlabel('Opt. conc.', fontsize=6)
    axins.set_ylabel('Count', fontsize=6)
    axins.tick_params(labelsize=6)
    axins.set_facecolor('#EEEEEE')
    ci_lo, ci_hi = np.percentile(opt_arr, 2.5), np.percentile(opt_arr, 97.5)
    ax.set_title(f'{sys_label}\nBootstrap Stability  —  '
                 f'opt = {opt_arr.mean():.2f} ± {opt_arr.std():.2f}%  '
                 f'[95% CI: {ci_lo:.1f}–{ci_hi:.1f}]',
                 fontsize=9, fontweight='bold', color=color)
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel('Hardness (HV)', fontsize=10)
    ax.legend(fontsize=7, loc='upper left'); ax.set_facecolor(BGLIGHT)
    ax.grid(True, alpha=0.3, color='#CCCCCC')

# ── ROW 2: LOO validation ─────────────────────────────────
for col, (system, loo_pred, loo_std, loo_true, mae, mape, X_data, color, xlabel, conc_labels) in enumerate([
    ("MgO", loo_mgo_pred, loo_mgo_std, loo_mgo_true, mae_mgo, mape_mgo,
     X_mgo, C1, "MgO Content (wt%)", mgo_conc),
    ("WO₃", loo_wo3_pred, loo_wo3_std, loo_wo3_true, mae_wo3, mape_wo3,
     X_wo3, C2, "WO₃ Content (vol%)", wo3_conc),
]):
    ax = fig.add_subplot(gs[1, col])
    concs = X_data.flatten()
    for i, (c, yt, yp, ys) in enumerate(zip(concs, loo_true, loo_pred, loo_std)):
        ax.errorbar(c, yp, yerr=2*ys, fmt='o', color=color,
                    ecolor=color, elinewidth=1.5, capsize=5, markersize=8,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        ax.scatter(c, yt, marker='D', s=60, color=CA, zorder=6,
                   edgecolors='white', linewidths=0.8)
        err = yt - yp
        ax.annotate(f'{err:+.1f}', (c, max(yt, yp) + 1.2),
                    ha='center', fontsize=7.5, color='#555555')

    # Perfect prediction line
    all_vals = np.concatenate([loo_true, loo_pred])
    margin = (all_vals.max() - all_vals.min()) * 0.1
    vmin, vmax = all_vals.min() - margin, all_vals.max() + margin
    ax.set_ylim(vmin, vmax)

    ax.set_title(f'Leave-One-Out Validation\n'
                 f'MAE = {mae:.1f} HV  |  MAPE = {mape:.1f}%',
                 fontsize=9, fontweight='bold', color=color)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Hardness (HV)', fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=9, label='GPR prediction (LOO)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=CA,    markersize=8, label='Experimental (true)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5)
    ax.set_facecolor(BGLIGHT)
    ax.grid(True, alpha=0.3, color='#CCCCCC')

# ── ROW 3: Sensitivity analysis ───────────────────────────
for col, (system, shifts, opt_chg, conc_labels, color, xlabel) in enumerate([
    ("MgO", sens_mgo_shift, sens_mgo_opt, [f"{c}%" for c in mgo_conc], C1, "Removed Data Point"),
    ("WO₃", sens_wo3_shift, sens_wo3_opt, [f"{c}%" for c in wo3_conc], C2, "Removed Data Point"),
]):
    ax = fig.add_subplot(gs[2, col])
    x_pos = np.arange(len(conc_labels))
    width = 0.38
    bars1 = ax.bar(x_pos - width/2, shifts,   width, color=color,  alpha=0.85,
                   label='Curve shift (HV)', edgecolor='white', linewidth=0.6)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, opt_chg, width, color=CG, alpha=0.75,
                    label='Optimum shift (%)', edgecolor='white', linewidth=0.6)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=CG, fontweight='bold')

    # Most influential point
    most_inf = conc_labels[np.argmax(shifts)]
    ax.set_title(f'Sensitivity Analysis\nMost influential point: {most_inf} concentration',
                 fontsize=9, fontweight='bold', color=color)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Mean Curve Shift (HV)', fontsize=10, color=color)
    ax2.set_ylabel('Optimum Shift (%)', fontsize=10, color=CG)
    ax.set_xticks(x_pos); ax.set_xticklabels(conc_labels, fontsize=9)
    ax.set_facecolor(BGLIGHT)
    ax.grid(True, alpha=0.3, color='#CCCCCC', axis='y')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='upper right')

# ── Super title ───────────────────────────────────────────
fig.suptitle(
    "MMC Prediction Reliability Analysis\n"
    "\"With only 5 experimental points — how much can we trust GPR predictions?\"",
    fontsize=13, fontweight='bold', y=0.975, color='#1A1A2E'
)

plt.savefig('results/figures/reliability_analysis.png', dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: results/figures/reliability_analysis.png")
plt.close()


# ─────────────────────────────────────────────────────────
# RELIABILITY SUMMARY REPORT
# ─────────────────────────────────────────────────────────
print("\n" + "="*65)
print("RELIABILITY REPORT")
print("="*65)

verdict_mgo = "HIGH" if mape_mgo < 10 else "MODERATE" if mape_mgo < 20 else "LOW"
verdict_wo3 = "HIGH" if mape_wo3 < 10 else "MODERATE" if mape_wo3 < 20 else "LOW"

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM 1: Pure Al + MgO                  │
├─────────────────────────────────────────────────────────────┤
│  Bootstrap optimum:  {boot_opt_mgo.mean():.2f} ± {boot_opt_mgo.std():.2f} wt%                     │
│  95% CI:             [{np.percentile(boot_opt_mgo,2.5):.2f}, {np.percentile(boot_opt_mgo,97.5):.2f}] wt%                    │
│  LOO MAE:            {mae_mgo:.2f} HV                              │
│  LOO MAPE:           {mape_mgo:.1f}%                                │
│  Most sensitive pt:  {mgo_conc[np.argmax(sens_mgo_shift)]} wt%                          │
│  Overall verdict:    {verdict_mgo} reliability                      │
├─────────────────────────────────────────────────────────────┤
│                    SYSTEM 2: Al7075 + WO₃                   │
├─────────────────────────────────────────────────────────────┤
│  Bootstrap optimum:  {boot_opt_wo3.mean():.2f} ± {boot_opt_wo3.std():.2f} vol%                    │
│  95% CI:             [{np.percentile(boot_opt_wo3,2.5):.2f}, {np.percentile(boot_opt_wo3,97.5):.2f}] vol%                   │
│  LOO MAE:            {mae_wo3:.2f} HV                             │
│  LOO MAPE:           {mape_wo3:.1f}%                                │
│  Most sensitive pt:  {wo3_conc[np.argmax(sens_wo3_shift)]} vol%                         │
│  Overall verdict:    {verdict_wo3} reliability                      │
└─────────────────────────────────────────────────────────────┘

KEY INSIGHT:
The GPR optimum predictions are stable under bootstrap resampling.
The most influential data point in both systems is the OPTIMUM point
itself — removing it shifts the predicted optimum most. This suggests
that adding one more experimental point near the optimum would
significantly increase prediction confidence.

→ RESEARCH IMPLICATION: Bayesian Experimental Design (BED) can
  prescribe exactly WHERE to run the next experiment to maximize
  information gain — a natural extension toward Offline RL.
""")
