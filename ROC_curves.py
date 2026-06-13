from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from genetic_utils import compute_epistasis_effect

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df_individuals = pd.read_csv("population.csv")
df_gwas = pd.read_csv("gwas_results.csv")
df_epistasis = pd.read_csv("real_epistasis.csv")

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
"""
y_true = df_individuals["label"]
y_score = PRS_gwas
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"GWAS-PRS ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve based on GWAS-estimated PRS")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ROC using true PRS (with epistasis)
auc_true = roc_auc_score(df_individuals["label"], df_individuals["logit"])

# ROC using GWAS PRS
auc_gwas = roc_auc_score(df_individuals["label"], PRS_gwas)

print(f"AUC true model: {auc_true:.3f}")
print(f"AUC GWAS PRS: {auc_gwas:.3f}")
"""
# ---------------------------------------------------------
# Function 3: Evaluate ROC
# ---------------------------------------------------------
def evaluate_model(y_true, y_score, title="ROC Curve"):

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
    title="ROC - GWAS PRS only"
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
    title="ROC - GWAS + Epistasis"
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

plt.show()

# ---------------------------------------------------------
# Print comparison
# ---------------------------------------------------------
print("\n==============================")
print(f"AUC GWAS only:       {auc_main:.3f}")
print(f"AUC GWAS + epistasis:{auc_epi:.3f}")
print("==============================")
"""
# ------------------------------------------------------------------------------------
# recall and precision: 2 different ways to calculate them with different thresholds
# ------------------------------------------------------------------------------------
from sklearn.metrics import (
    roc_curve,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Option 1: Threshold via ROC (Youden’s J)
# ------------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
threshold_youden = roc_thresholds[best_idx]

y_pred_youden = (y_score >= threshold_youden).astype(int)

precision_youden = precision_score(y_true, y_pred_youden)
recall_youden = recall_score(y_true, y_pred_youden)

print("=== Threshold via Youden’s J ===")
print(f"Threshold: {threshold_youden:.4f}")
print(f"Precision: {precision_youden * 100:.2f}%")
print(f"Recall:    {recall_youden * 100:.2f}%")

# ------------------------------------------------------------
# Precision–Recall Curve
# ------------------------------------------------------------
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
    y_true, y_score
)

avg_precision = average_precision_score(y_true, y_score)

plt.figure(figsize=(7, 6))
plt.plot(
    recall_curve,
    precision_curve,
    label=f"PRS (AP = {avg_precision:.3f})"
)

# Mark the two operating points
plt.scatter(
    recall_youden,
    precision_youden,
    color="red",
    label="Youden threshold",
    zorder=3
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (GWAS PRS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""



