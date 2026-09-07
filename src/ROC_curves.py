from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.genetic_utils import compute_epistasis_effect
import os

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df_individuals = pd.read_csv("../data/population.csv")
df_gwas = pd.read_csv("../results/GWAS/gwas_results.csv")
df_epistasis = pd.read_csv("../data/real_epistasis.csv")

# -----------------------------------------
# build genotype matrix
# -----------------------------------------
locus_cols = [
    c for c in df_individuals.columns
    if c.startswith("Locus_")
]

genotypes = df_individuals[locus_cols].values

# -----------------------------------------
# build pairs and gamma
# -----------------------------------------
pairs = list(
    zip(
        df_epistasis["i"],
        df_epistasis["j"]
    )
)

gamma = df_epistasis["gamma"].values

# ---------------------------------------------------------
# Function 1: Compute PRS from GWAS beta_hat
# ---------------------------------------------------------
def compute_prs_main(df_individuals, df_gwas, p_threshold=0.05):

    significant_snps = df_gwas[df_gwas["p_adj"] < p_threshold]

    prs = np.zeros(len(df_individuals))

    for _, row in significant_snps.iterrows():

        locus = row["locus"]
        beta_hat = row["beta_hat"]

        if not np.isnan(beta_hat):
            prs += df_individuals[locus] * beta_hat

    return prs
# ---------------------------------------------------------
# Function 2: Add epistatic effects
# ---------------------------------------------------------
def add_epistasis(
        prs,
        df_individuals,
        df_epistasis,
        use_true_gamma=True,
        gamma_scale=1.0
):
    prs_epi = prs.copy()

    for _, row in df_epistasis.iterrows():

        gamma = row["gamma"]

        # skip non-epistatic pairs
        if gamma == 0:
            continue

        i = int(row["i"])
        j = int(row["j"])

        locus_i = f"Locus_{i}"
        locus_j = f"Locus_{j}"

        xi = df_individuals[locus_i] - 1
        xj = df_individuals[locus_j] - 1

        interaction = xi * xj

        if use_true_gamma:
            prs_epi += gamma_scale * gamma * interaction

    return prs_epi

def compute_top_epistasis(
        genotypes,
        pairs,
        gamma,
        prs_main,
        known_fraction
):
    """
    known_fraction:
        0.0 -> no epistasis known
        0.5 -> top 50% strongest interactions known
        1.0 -> all interactions known
    """

    nonzero_idx = np.where(gamma != 0)[0]

    ranked_idx = nonzero_idx[
        np.argsort(
            np.abs(gamma[nonzero_idx])
        )[::-1]
    ]

    n_known = int(
        len(ranked_idx) * known_fraction
    )

    selected_idx = ranked_idx[:n_known]

    gamma_partial = np.zeros_like(gamma)

    gamma_partial[selected_idx] = gamma[selected_idx]

    epi_effect, _ = compute_epistasis_effect(
        genotypes,
        pairs,
        gamma_partial,
        prs_main,
        epi_variance_fraction=0.3,
        apply_scaling=False
    )

    return epi_effect

# ---------------------------------------------------------
# Function 3: Evaluate ROC
# ---------------------------------------------------------
def evaluate_model(y_true, y_score, title="ROC Curve", filename="roc.png"):

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 7))

    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    os.makedirs("../figures/GWAS", exist_ok=True)
    plt.savefig(f"../figures/GWAS/{filename}", dpi=300, bbox_inches="tight")

    plt.show()

    return auc

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

y_true = df_individuals["label"]

# ---------------------------------------------------------
# Model 1: GWAS only
# ---------------------------------------------------------
prs_main = compute_prs_main(df_individuals, df_gwas)
knowledge_levels = [
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00
]

results = []

auc_main = evaluate_model(
    y_true,
    prs_main,
    title="ROC - GWAS PRS only",
    filename="roc_gwas_only.png"
)

for frac in knowledge_levels:

    epi_effect = compute_top_epistasis(
        genotypes,
        pairs,
        gamma,
        prs_main,
        known_fraction=frac
    )

    prs_model = prs_main + epi_effect

    auc = roc_auc_score(
        y_true,
        prs_model
    )

    results.append(
        {
            "known_fraction": frac,
            "auc": auc
        }
    )

    print(
        f"Known strongest epistasis = {frac:.0%}"
        f" | AUC = {auc:.4f}"
    )
results_df = pd.DataFrame(results)
baseline_auc = results_df.loc[0, "auc"]

results_df["auc_improvement_%"] = (
    (results_df["auc"] - baseline_auc) / baseline_auc
) * 100

print(results_df)
# ---------------------------------------------------------
# Model 2: GWAS + Epistasis
# ---------------------------------------------------------
epi_effect, _ = compute_epistasis_effect(
    genotypes,
    pairs,
    gamma,
    prs_main,
    epi_variance_fraction=0.3,
    apply_scaling=False
)

prs_epi = prs_main + epi_effect

auc_epi = evaluate_model(
    y_true,
    prs_epi,
    title="ROC - GWAS + Epistasis",
    filename="roc_with_epistasis.png"
)

plt.figure(figsize=(7,5))

plt.plot(
    results_df["known_fraction"],
    results_df["auc"],
    marker="o"
)

plt.xlabel(
    "Fraction of strongest epistatic pairs known"
)

plt.ylabel("AUC")

plt.title(
    "AUC vs Knowledge of Strongest Epistatic Interactions"
)

plt.grid(True)

os.makedirs("../figures/GWAS", exist_ok=True)
plt.savefig("../figures/GWAS/auc_with_epistasis.png", dpi=300, bbox_inches="tight")

plt.show()

# ---------------------------------------------------------
# Print comparison
# ---------------------------------------------------------
print("\n==============================")
print(f"AUC GWAS only:       {auc_main:.3f}")
print(f"AUC GWAS + epistasis:{auc_epi:.3f}")
print("==============================")




