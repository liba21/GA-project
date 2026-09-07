import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os

# =========================================================
# LOAD DATA
# =========================================================

df_individuals = pd.read_csv("../data/population.csv")
df_gwas = pd.read_csv("../results/GWAS/gwas_results.csv")
df_real = pd.read_csv("../data/real_epistasis.csv")

locus_cols = [c for c in df_individuals.columns if c.startswith("Locus_")]
y_true = df_individuals["label"].values

# =========================================================
# LOAD GWAS BETAS
# =========================================================

beta_map = dict(zip(df_gwas["locus"], df_gwas["beta_hat"]))

# =========================================================
# LOAD TRUE EPISTASIS (only real non-zero gamma)
# =========================================================

EPS = 1e-8

true_gamma_map = {
    tuple(sorted((int(i), int(j)))): g
    for i, j, g in zip(df_real["i"], df_real["j"], df_real["gamma"])
    if abs(g) > EPS
}

# =========================================================
# COMPUTE MAIN PRS
# =========================================================

PRS_MAIN = np.zeros(len(df_individuals))

for locus, beta in beta_map.items():
    if locus in df_individuals.columns and not np.isnan(beta):
        PRS_MAIN += df_individuals[locus].values * beta

# =========================================================
# GRID OF GAMMA VALUES
# =========================================================

gamma_grid = np.linspace(-0.4, 0.4, 11)

print("Gamma grid:")
print(gamma_grid)

# =========================================================
# SCAN ALL PAIRS
# =========================================================

results = []
n_loci = len(locus_cols)

print("Starting grid-search epistasis scan...")

for i in range(n_loci):

    xi = df_individuals[f"Locus_{i}"].values - 1

    for j in range(i + 1, n_loci):

        xj = df_individuals[f"Locus_{j}"].values - 1

        interaction = xi * xj

        best_auc = -np.inf
        best_gamma = None

        # -------------------------------------------------
        # Try all gamma values
        # -------------------------------------------------

        for gamma in gamma_grid:

            prs = PRS_MAIN + gamma * interaction

            auc = roc_auc_score(y_true, prs)

            if auc > best_auc:
                best_auc = auc
                best_gamma = gamma

        pair = (i, j)

        gamma_true = np.nan
        abs_error = np.nan
        is_true = False

        if pair in true_gamma_map:

            gamma_true = true_gamma_map[pair]
            abs_error = abs(best_gamma - gamma_true)
            is_true = True

        results.append([
            i,
            j,
            best_gamma,
            gamma_true,
            abs_error,
            best_auc,
            is_true
        ])

    print(f"Finished locus {i}/{n_loci-1}")

print("Scan complete.")

# =========================================================
# SAVE FULL RESULTS
# =========================================================

df_results = pd.DataFrame(
    results,
    columns=[
        "i",
        "j",
        "gamma_pred",
        "gamma_true",
        "abs_error",
        "auc",
        "is_true"
    ]
)

df_results.to_csv(
    "../results/statistical_methods/grid_search_scan_all_pairs.csv",
    index=False
)

# =========================================================
# TOP 125 BY AUC
# =========================================================

df_top = df_results.sort_values(
    by="auc",
    ascending=False
).head(125)

df_top.to_csv(
    "../results/statistical_methods/grid_search_scan_top125.csv",
    index=False
)

# =========================================================
# TRUE PAIRS FOUND IN TOP 125
# =========================================================

df_top_true = df_top[df_top["is_true"] == True].copy()

df_top_true.to_csv(
    "../results/statistical_methods/grid_search_scan_top125_true_pairs.csv",
    index=False
)

# =========================================================
# SUMMARY
# =========================================================

recall = len(df_top_true) / len(true_gamma_map)

mean_gamma_error = df_top_true["abs_error"].mean()

print("\n========== FINAL SUMMARY ==========")
print("Total candidate pairs:", len(df_results))
print("Total true pairs:", len(true_gamma_map))
print("Top 125 selected:", len(df_top))
print("True pairs inside top 125:", len(df_top_true))
print("Recall@125:", recall)
print("Mean Gamma Error:", mean_gamma_error)

# =========================================================
# PLOT TRUE VS PREDICTED GAMMAS (ONLY TRUE FOUND PAIRS)
# =========================================================

if len(df_top_true) > 0:

    # ID for each pair
    df_plot = df_top_true.copy()

    df_plot["pair_id"] = (
        df_plot["i"].astype(str)
        + ","
        + df_plot["j"].astype(str)
    )

    # Sort by true gamma
    df_plot = df_plot.sort_values(
        "gamma_true"
    ).reset_index(drop=True)

    x = np.arange(len(df_plot))

    plt.figure(figsize=(16, 6))

    # TRUE GAMMA
    plt.scatter(
        x,
        df_plot["gamma_true"],
        label="True γ",
        alpha=0.8
    )

    # LOGISTIC GAMMA
    plt.scatter(
        x,
        df_plot["gamma_pred"],
        label="Predicted γ (Grid Search)",
        alpha=0.8
    )

    # Lines between truth and prediction
    for idx in range(len(df_plot)):
        plt.plot(
            [x[idx], x[idx]],
            [
                df_plot.loc[idx, "gamma_true"],
                df_plot.loc[idx, "gamma_pred"]
            ],
            color="gray",
            alpha=0.3
        )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.xlabel("Epistatic pair index")
    plt.ylabel("Gamma effect size")
    plt.title(
        "True vs Predicted Epistatic Effects (Grid Search Top-125)"
    )

    plt.legend()
    plt.tight_layout()

    os.makedirs("../figures/statistical_methods", exist_ok=True)
    plt.savefig("../figures/statistical_methods/grid_search_scan.png", dpi=300, bbox_inches="tight")

    plt.show()

else:
    print("No true pairs found in Top 125.")