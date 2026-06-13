import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from genetic_utils import compute_epistasis_effect

# ----------------------------------
# PARAMETERS
# ----------------------------------

target_prev = 0.02
n_loci = 100
N = 100000

maf_min = 0.02
maf_max = 0.5
maf_decay = 6.0

epi_fraction_pairs = 0.1 #---------------------- change here when want 0% epistasis
epi_variance_fraction = 0.3 # epistasis target ratio (P)----change here to define epistasis ratio
sigma = 0.15


# ----------------------------------
# 1. CREATE MAIN EFFECTS
# ----------------------------------

def generate_betas(n_loci):

    betas = []

    for i in range(n_loci):

        r = np.random.rand()

        if r < 0.6:
            b = np.random.uniform(0.1, 0.2)

        elif r < 0.9:
            b = np.random.uniform(0.2, 0.5)

        else:
            b = np.random.uniform(0.5, 1)

        if np.random.rand() < 0.5:
            b *= -1

        betas.append(b)

    betas = np.array(betas)

    df_betas = pd.DataFrame({
        "locus": np.arange(n_loci),
        "beta": betas
    })

    return betas, df_betas


# ----------------------------------
# 2. CREATE MAF
# ----------------------------------

def generate_maf(betas, maf_min, maf_max, maf_decay):

    abs_beta = np.abs(betas)

    maf = maf_min + (maf_max - maf_min) * np.exp(-maf_decay * abs_beta)

    maf = np.clip(maf, maf_min, maf_max)

    return maf


# ----------------------------------
# 3. CREATE EPISTATIC PAIRS
# ----------------------------------

def generate_epistasis(n_loci, epi_fraction_pairs, sigma):

    pairs = list(itertools.combinations(range(n_loci), 2))

    n_pairs = len(pairs)

    epi_count = int(n_pairs * epi_fraction_pairs)

    epi_pairs_idx = np.random.choice(
        np.arange(n_pairs),
        epi_count,
        replace=False
    )

    gamma = np.zeros(n_pairs)

    gamma[epi_pairs_idx] = np.random.normal(
        0,
        sigma,
        epi_count
    )

    df_epistasis = pd.DataFrame({
        "i": [pairs[k][0] for k in range(n_pairs)],
        "j": [pairs[k][1] for k in range(n_pairs)],
        "gamma": gamma
    })

    return pairs, gamma, df_epistasis


# ----------------------------------
# 4. GENERATE GENOTYPES
# ----------------------------------

def generate_genotypes(N, n_loci, maf):

    genotypes = np.zeros((N, n_loci), dtype=int)

    for i in range(n_loci):

        p = maf[i]

        genotypes[:, i] = np.random.choice(
            [0, 1, 2],
            size=N,
            p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2]
        )

    return genotypes


# ----------------------------------
# 5. COMPUTE PRS without epistasis
# ----------------------------------

def compute_prs(genotypes, betas):

    return genotypes @ betas


# ----------------------------------
# 6. EPISTASIS SCALE
# ----------------------------------

# in genetic_utils.py

# ----------------------------------
# 7. COMPUTE EPISTATIC EFFECT
# ----------------------------------

# in genetic_utils.py

# ----------------------------------
# 8. LOGISTIC MODEL
# ----------------------------------

def generate_phenotype(PRS_total, target_prev):

    PRS_total_centered = PRS_total - PRS_total.mean()

    alpha = np.log(target_prev / (1 - target_prev))

    logit = alpha + PRS_total_centered

    P_D = 1 / (1 + np.exp(-logit))

    phenotype = np.random.binomial(1, P_D)

    return logit, P_D, phenotype


# ----------------------------------
# 9. BUILD DATAFRAME
# ----------------------------------

def build_population_dataframe(
        genotypes,
        PRS_total,
        logit,
        P_D,
        phenotype,
        n_loci):

    df_population = pd.DataFrame(
        genotypes,
        columns=[f"Locus_{i}" for i in range(n_loci)]
    )

    df_population["PRS_total"] = PRS_total
    df_population["logit"] = logit
    df_population["P(D)"] = P_D
    df_population["label"] = phenotype

    return df_population


# ----------------------------------
# 10. PLOTTING FUNCTIONS
# ----------------------------------

# 1. Distribution of β_i (effects)

def plot_beta_distribution(df_betas):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df_betas["beta"],
        bins=40,
        kde=True,
        color="blue",
        alpha=0.6
    )

    plt.title("Distribution of Main Effects βᵢ")
    plt.xlabel("βᵢ value")
    plt.ylabel("Frequency")

    plt.show()


# ----------------------------------
# 2. Distribution of γ_ij
# ----------------------------------

def plot_epistasis_distribution(epi_nonzero):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        epi_nonzero,
        bins=40,
        kde=True,
        color="purple",
        alpha=0.6
    )

    plt.title("Distribution of Epistatic Effects γᵢⱼ (non-zero only)")
    plt.xlabel("γᵢⱼ value")
    plt.ylabel("Frequency")

    plt.show()


# ----------------------------------
# 3. Distribution of total PRS
# ----------------------------------

def plot_prs_distribution(PRS_total):

    plt.figure(figsize=(10, 6))

    plt.hist(PRS_total, bins=50)

    plt.title("Distribution of total PRS (with epistasis)")
    plt.xlabel("PRS_total")
    plt.ylabel("Frequency")

    plt.show()


# ----------------------------------
# 4. Distribution of disease probabilities
# ----------------------------------

def plot_probability_distribution(P_D):

    plt.figure(figsize=(10, 6))

    plt.hist(P_D, bins=50)

    plt.title("Distribution of disease probabilities P(D)")
    plt.xlabel("P(D)")
    plt.ylabel("Frequency")

    plt.show()


# ----------------------------------
# 5. Histogram Healthy vs Diseased
# ----------------------------------

def plot_probability_by_label(df_individuals):

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df_individuals,
        x="P(D)",
        hue="label",
        bins=40,
        kde=True,
        alpha=0.6
    )

    plt.title("Histogram of P(D) for Healthy vs Diseased")
    plt.xlabel("P(D)")
    plt.ylabel("Count")

    plt.legend(
        title="Label",
        labels=["Diseased (1)", "Healthy (0)"]
    )

    plt.show()


# ----------------------------------
# 6. Print cases and controls
# ----------------------------------

def print_case_control_stats(df_individuals):

    num_total = len(df_individuals)

    num_cases = df_individuals["label"].sum()

    num_controls = num_total - num_cases

    print(
        f"Cases: {num_cases} "
        f"({num_cases / num_total * 100:.2f}%)"
    )

    print(
        f"Controls: {num_controls} "
        f"({num_controls / num_total * 100:.2f}%)"
    )


# ----------------------------------
# 7. PRS vs disease rate (deciles)
# ----------------------------------

def plot_prs_vs_disease_rate(df_individuals):

    df_individuals["decile"] = pd.qcut(
        df_individuals["PRS_total"],
        10,
        labels=False
    )

    calibration = df_individuals.groupby("decile").agg(
        mean_PRS=("PRS_total", "mean"),
        disease_rate=("label", "mean")
    ).reset_index()

    plt.figure(figsize=(7, 7))

    plt.scatter(
        calibration["mean_PRS"],
        calibration["disease_rate"],
        s=80
    )

    xmin = calibration["mean_PRS"].min()
    xmax = calibration["mean_PRS"].max()

    ymin = calibration["disease_rate"].min()
    ymax = calibration["disease_rate"].max()

    margin = 0.05 * (xmax - xmin)

    plt.xlim(xmin - margin, xmax + margin)

    plt.ylim(-0.01, 0.25)

    plt.xlabel("PRS_total")
    plt.ylabel("Observed disease rate")

    plt.title("PRS vs disease rate (deciles)")

    plt.grid(alpha=0.3)

    plt.show()

# ----------------------------------
# MAIN
# ----------------------------------

def main():

    betas, df_betas = generate_betas(n_loci)

    maf = generate_maf(
        betas,
        maf_min,
        maf_max,
        maf_decay
    )

    pairs, gamma, df_epistasis = generate_epistasis(
        n_loci,
        epi_fraction_pairs,
        sigma
    )

    genotypes = generate_genotypes(
        N,
        n_loci,
        maf
    )

    PRS = compute_prs(genotypes, betas)

    epi_effect, gamma = compute_epistasis_effect(
        genotypes,
        pairs,
        gamma,
        PRS,
        epi_variance_fraction
    )
    df_epistasis["gamma"] = gamma

    PRS_total = PRS + epi_effect

    logit, P_D, phenotype = generate_phenotype(
        PRS_total,
        target_prev
    )

    df_individuals = build_population_dataframe(
        genotypes,
        PRS_total,
        logit,
        P_D,
        phenotype,
        n_loci
    )

    df_individuals.to_csv("population.csv", index=False)
    df_betas.to_csv("real_betas.csv", index=False)
    df_epistasis.to_csv("real_epistasis.csv", index=False)

    # plot functions:
    plot_beta_distribution(df_betas)

    epi_nonzero = df_epistasis[
        df_epistasis["gamma"] != 0
        ]["gamma"]

    plot_epistasis_distribution(epi_nonzero)

    plot_prs_distribution(PRS_total)

    plot_probability_distribution(P_D)

    plot_probability_by_label(df_individuals)

    print_case_control_stats(df_individuals)

    plot_prs_vs_disease_rate(df_individuals)






if __name__ == "__main__":
    main()