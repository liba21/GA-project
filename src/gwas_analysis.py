import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# reading from the population file
df_individuals = pd.read_csv("../data/population.csv")
df_betas = pd.read_csv("../data/real_betas.csv")
df_epistasis = pd.read_csv("../data/real_epistasis.csv")

#--------------------------------------
# Part 1: Selection of variables for GWAS
#--------------------------------------
# df_individuals.columns → List of names of all the columns that called "Locus_".
# df_gwas → Table with all Locus columns and the label column.
locus_cols = [c for c in df_individuals.columns if c.startswith("Locus_")]
df_gwas = df_individuals[locus_cols + ["label"]]

#---------------------------------------
# Part 2: GWAS loop (one locus at a time)
#---------------------------------------
results = [] # An empty list where we will store the GWAS results from each locus.

# for each locus in locus_cols:
for locus in locus_cols:
    # - Extracts one column only (the current locus)
    # - Double brackets [[...]] keep X as a DataFrame, not a Series
    # - Values represent genotype dosage (0/1/2)
    X = df_gwas[[locus]]

    # - Adds a column of 1’s called const
    # - This is the intercept of the logistic regression
    X = sm.add_constant(X)  # intercept

    # label is the phenotype - 0 for healthy, 1 for disease
    y = df_gwas["label"]

    # the model becomes: log(P(Y=1)/P(Y=0)) = a + β1⋅x1
    # where:
    # - a = baseline prevalence of the disease in the population
    # - β1 = the beta effect of this locus
    # - x1 = genotype dosage (0/1/2)
    model = sm.Logit(y, X)

    # GWAS often fails for some loci because of:
    # - Perfect separation
    # - Rare alleles
    # - No variation in genotypes
    # This block prevents the entire loop from crashing.
    try:
        # Fits the logistic regression using maximum likelihood
        # disp=False suppresses verbose output
        # If the model converges, res contains:
        # - Estimated coefficients
        # - Standard errors
        # - p-values
        res = model.fit(disp=False)

        # beta_hat = estimated regression coefficient
        # Represents log-odds change per 1 unit increase in genotype
        # Interpretation:
        # - Positive → risk allele
        # - Negative → protective allele
        beta_hat = res.params[locus]

        # Wald test p-value for 𝐻0 : 𝛽 =0
        # Tests whether this locus is significantly associated with the phenotype.
        # This is the key GWAS statistic used in Manhattan plots.
        p_value = res.pvalues[locus]

        # Converts log-odds to odds ratio.
        # Interpretation:
        # - OR > 1 → increased risk
        # - OR < 1 → protective effect
        # - OR = 1 → no effect
        OR = np.exp(beta_hat)

        # Appends one dictionary per locus.
        # Each row represents one GWAS test.
        results.append({
            "locus": locus,
            "beta_hat": beta_hat,
            "OR": OR,
            "p_value": p_value
        })

    except:
        # Catches any fitting error (e.g. non-convergence).
        # Stores NaN instead of crashing.
        # Keeps locus count consistent.
        results.append({
            "locus": locus,
            "beta_hat": np.nan,
            "OR": np.nan,
            "p_value": np.nan
        })

# Turns the list of dictionaries into a structured table.
# Columns: locus, beta_hat, OR, p_value
df_gwas_results = pd.DataFrame(results)

# Bonferroni correction: p(adj)=p×m, where:
# m = number of loci tested
# Purpose:
# - Control family-wise error rate
# - Very conservative (common in GWAS)
df_gwas_results["p_adj"] = df_gwas_results["p_value"] * len(df_gwas_results)

# sort loci by significance
#df_gwas_results = df_gwas_results.sort_values("p_value")

# save to file, index=False avoids adding row numbers
df_gwas_results.to_csv("../results/GWAS/gwas_results.csv", index=False)

#---------------------------------------
# Part 3: Manhattan plot and Effect size vs statistical significance plot
#---------------------------------------
print(df_gwas_results.head(10))

# Plot 1: Toy GWAS Manhattan Plot
# - x-axis = locus index
# - Y-axis = −log₁₀(p-value)
# - Each dot = one genetic locus tested
# - Red dashed line = the Bonferroni significance threshold.
# Any dot above the line is statistically significant.
# this plot answers: Which loci are statistically significant?
plt.figure(figsize=(12, 6))
plt.scatter(
    range(len(df_gwas_results)),
    -np.log10(df_gwas_results["p_value"]),
    s=20
)
plt.axhline(-np.log10(0.05 / len(df_gwas_results)), color="red", linestyle="--")
plt.xlabel("Locus index")
plt.ylabel("-log10(p-value)")
plt.title("Toy GWAS Manhattan Plot")

os.makedirs("../figures/GWAS", exist_ok=True)
plt.savefig("../figures/GWAS/manhattan_plot.png", dpi=300, bbox_inches="tight")

plt.show()


#---------------------------------------
# Part 4: beta comparison - 2 plots
#---------------------------------------

# Extracting a numeric locus index from the name
df_gwas_results["locus_index"] = (
    df_gwas_results["locus"]
    .str.replace("Locus_", "")
    .astype(int)
)

# Renaming columns to make it clear
df_betas_renamed = df_betas.rename(columns={"beta": "beta_true"})

# After the merge, each row represents one locus, with:
# - beta_true → the true effect size used in the simulation
# - beta_hat → the effect estimated by GWAS
# - p_value → statistical significance of the GWAS estimate
df_compare = pd.merge(
    df_gwas_results,
    df_betas_renamed,
    left_on="locus_index",
    right_on="locus",
    how="inner"
)


# Plot 1: True vs Estimated β (colored by significance)
# - Bright points ->  Highly significant loci
# Usually Large |β_true| or Precisely estimated effects (small SE)
# - Faded points -> Non-significant loci
# Typically small effects or high uncertainty
# X-axis: true effect size (simulation truth)
# Y-axis: estimated effect size from GWAS
# Each dot = one locus.
# The red dashed line (y = x) represents perfect recovery:
# If a point lies exactly on this line → GWAS estimated the effect perfectly.
plt.figure(figsize=(8, 8))
plt.scatter(
    df_compare["beta_true"],
    df_compare["beta_hat"],
    c=-np.log10(df_compare["p_value"]),
    cmap="viridis",
    alpha=0.8
)
#  y = x (perfect recovery)
lims = [
    min(df_compare["beta_true"].min(), df_compare["beta_hat"].min()),
    max(df_compare["beta_true"].max(), df_compare["beta_hat"].max())
]
plt.plot(lims, lims, 'r--')
plt.colorbar(label="-log10(p-value)")
plt.plot(lims, lims, 'r--')
plt.xlabel("True β")
plt.ylabel("Estimated β")
plt.title("True vs Estimated β (colored by significance)")

os.makedirs("../figures/GWAS", exist_ok=True)
plt.savefig("../figures/GWAS/true_vs_estimated_beta.png", dpi=300, bbox_inches="tight")

plt.show()

# Correlation between true and estimated β -
# High correlation → good recovery of effect directions and sizes
# Low correlation → noisy or underpowered GWAS
# Interpretation guide:
# ~0 → no information
# 0.3–0.6 → moderate recovery
# 0.7 → strong recovery
corr = np.corrcoef(df_compare["beta_true"], df_compare["beta_hat"])[0, 1]
print(f"Correlation between true and estimated β: {corr:.3f}")


df_compare["beta_error"] = df_compare["beta_hat"] - df_compare["beta_true"]

# Plot 2: GWAS estimation error
# This plot isolates error directly:
# X-axis: true β
# Y-axis: estimation error
# The red horizontal line at 0 means: No error (perfect estimation)
plt.figure(figsize=(8, 6))
plt.scatter(df_compare["beta_true"], df_compare["beta_error"], alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("True β")
plt.ylabel("Estimation error (β_hat − β_true)")
plt.title("GWAS estimation error")

os.makedirs("../figures/GWAS", exist_ok=True)
plt.savefig("../figures/GWAS/gwas_estimation_error.png", dpi=300, bbox_inches="tight")

plt.show()

df_compare = df_compare[
    ["locus_index", "beta_hat", "beta_true", "p_adj"]
]
df_compare.to_csv("../results/GWAS/beta_comparison.csv", index=False)
print(df_compare.head())
#--------------------------------------------------------------
#--------------------------------------------------------------

def select_extreme_beta_loci(df_gwas_results):
    locus_max_beta = df_gwas_results.loc[
        df_gwas_results["beta_hat"].idxmax(), "locus"
    ]

    locus_min_beta = df_gwas_results.loc[
        df_gwas_results["beta_hat"].idxmin(), "locus"
    ]

    return {
        "Max betâ": locus_max_beta,
        "Min betâ": locus_min_beta
    }
def genotype_label_proportions(df_individuals, locus):
    data = []

    # סך כל החולים והבריאים (קבועים לכל הגנוטיפים)
    n_disease = (df_individuals["label"] == 1).sum()
    n_healthy = (df_individuals["label"] == 0).sum()

    for genotype in [0, 1, 2]:
        n_disease_g = (
            (df_individuals[locus] == genotype) &
            (df_individuals["label"] == 1)
        ).sum()

        n_healthy_g = (
            (df_individuals[locus] == genotype) &
            (df_individuals["label"] == 0)
        ).sum()

        disease_prop = n_disease_g / n_disease if n_disease > 0 else np.nan
        healthy_prop = n_healthy_g / n_healthy if n_healthy > 0 else np.nan

        data.append((healthy_prop, disease_prop))

    return data  # [(H0,D0), (H1,D1), (H2,D2)]

def plot_loci_genotype_proportions(df_individuals, loci_dict):
    genotypes = [0, 1, 2]
    width = 0.25

    plt.figure(figsize=(12, 6))

    x_base = np.arange(len(loci_dict)) * 4  # Spacing between loci

    for i, (label, locus) in enumerate(loci_dict.items()):
        proportions = genotype_label_proportions(df_individuals, locus)

        for j, genotype in enumerate(genotypes):
            healthy_prop, disease_prop = proportions[j]

            x_center = x_base[i] + j

            plt.bar(
                x_center - width / 2,
                healthy_prop,
                width,
                label="Healthy" if (i == 0 and j == 0) else "",
                color="tab:blue"
            )

            plt.bar(
                x_center + width / 2,
                disease_prop,
                width,
                label="Disease" if (i == 0 and j == 0) else "",
                color="tab:red"
            )

    # X ticks
    xticks = []
    xtick_labels = []

    for i, locus in enumerate(loci_dict.values()):
        for g in genotypes:
            xticks.append(x_base[i] + g)
            xtick_labels.append(f"{locus}\nG={g}")

    plt.xticks(xticks, xtick_labels, rotation=30)
    plt.ylabel("Proportion")
    plt.xlabel("Locus and genotype")
    plt.title("Genotype proportions by phenotype for extreme beta loci")
    plt.legend()
    plt.tight_layout()

    os.makedirs("../figures/GWAS", exist_ok=True)
    plt.savefig("../figures/GWAS/genotype_proportions_by_phenotype.png", dpi=300, bbox_inches="tight")

    plt.show()
loci = select_extreme_beta_loci(df_gwas_results)
plot_loci_genotype_proportions(df_individuals, loci)



