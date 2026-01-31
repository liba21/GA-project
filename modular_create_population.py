import numpy as np
import pandas as pd
import itertools

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N = 100_000
N_LOCI = 100

# Target variance contributions
TARGET_VAR = {
    "main": 0.65,
    "directed_epi": 0.2,
    "pure_epi": 0.15
}
sigma_dir=0.05
sigma_pure=0.15

def split_loci(n_loci, frac_add=0.65, frac_dir=0.20):
    loci = np.random.permutation(n_loci)

    n_add = int(frac_add * n_loci)
    n_dir = int(frac_dir * n_loci)

    additive = loci[:n_add]
    directed = loci[n_add:n_add+n_dir]
    pure_epi = loci[n_add+n_dir:]

    return additive, directed, pure_epi

def sample_betas(n_loci, additive, directed, pure_epi):
    betas = np.zeros(n_loci)

    def draw_beta():
        r = np.random.rand()
        if r < 0.6:
            b = np.random.uniform(0.02, 0.08)
        elif r < 0.9:
            b = np.random.uniform(0.08, 0.20)
        else:
            b = np.random.uniform(0.20, 0.40)
        return b if np.random.rand() > 0.5 else -b

    for i in additive:
        betas[i] = draw_beta()

    for i in directed:
        betas[i] = draw_beta()

    # pure epistasis: β ≈ 0
    betas[pure_epi] = np.random.normal(0, 0.005, size=len(pure_epi))

    return betas

def sample_epistasis(betas, directed, pure_epi, sigma_dir=0.05, sigma_pure=0.15):
    pairs = list(itertools.combinations(range(len(betas)), 2))
    gamma = np.zeros(len(pairs))

    for idx, (i, j) in enumerate(pairs):

        # Directed epistasis
        if i in directed and j in directed:
            gamma[idx] = (
                0.3 * betas[i] * betas[j]
                + np.random.normal(0, sigma_dir)
            )

        # Pure epistasis
        elif i in pure_epi and j in pure_epi:
            gamma[idx] = np.random.normal(0, sigma_pure)

    return pairs, gamma

def generate_genotypes(n_ind, n_loci, maf=0.3):
    p = maf
    return np.random.choice(
        [0, 1, 2],
        size=(n_ind, n_loci),
        p=[(1-p)**2, 2*p*(1-p), p**2]
    )
def compute_prs_components(genotypes, betas, pairs, gamma):
    prs_main = genotypes @ betas

    prs_epi = np.zeros(len(genotypes))
    for idx, (i, j) in enumerate(pairs):
        if gamma[idx] != 0:
            prs_epi += gamma[idx] * genotypes[:, i] * genotypes[:, j]

    return prs_main, prs_epi

def scale_components(prs_main, prs_epi, target_main):
    v_main = np.var(prs_main)
    v_epi = np.var(prs_epi)

    s_main = np.sqrt(target_main / v_main)
    s_epi = np.sqrt((1 - target_main) / v_epi)

    return prs_main * s_main, prs_epi * s_epi

def compute_phenotype(prs_total, prevalence=0.05, top_frac=0.02):
    logit = prs_total
    p_d = 1 / (1 + np.exp(-logit))

    threshold = np.quantile(p_d, 1 - top_frac)
    label = (p_d >= threshold).astype(int)

    return logit, p_d, label

additive, directed, pure_epi = split_loci(N_LOCI)

betas = sample_betas(N_LOCI, additive, directed, pure_epi)

pairs, gamma = sample_epistasis(betas, directed, pure_epi)

genotypes = generate_genotypes(N, N_LOCI)

prs_main, prs_epi = compute_prs_components(genotypes, betas, pairs, gamma)

prs_main, prs_epi = scale_components(
    prs_main, prs_epi, TARGET_VAR["main"]
)

prs_total = prs_main + prs_epi

logit, P_D, label = compute_phenotype(prs_total)

df_individuals = pd.DataFrame(
    genotypes, columns=[f"Locus_{i}" for i in range(N_LOCI)]
)
df_individuals["logit"] = logit
df_individuals["P(D)"] = P_D
df_individuals["label"] = label

df_betas = pd.DataFrame({
    "locus": np.arange(N_LOCI),
    "beta": betas
})

df_epistasis = pd.DataFrame({
    "i": [p[0] for p in pairs],
    "j": [p[1] for p in pairs],
    "gamma": gamma
})

df_individuals.to_csv("population.csv", index=False)
df_betas.to_csv("real_betas.csv", index=False)
df_epistasis.to_csv("real_epistasis.csv", index=False)

var_main = np.var(prs_main)
var_epi  = np.var(prs_epi)

print("Main effects contribution:", var_main / (var_main + var_epi))
print("Epistasis contribution:", var_epi / (var_main + var_epi))

# ---------------------------
# 12. Plotting distributions
# ---------------------------

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# --------------------------------
# 1. Distribution of main effects β_i
# --------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(
    df_betas["beta"],
    bins=40,
    kde=True,
    color="steelblue",
    alpha=0.7
)
plt.title("Distribution of Main Genetic Effects (β)")
plt.xlabel("β value")
plt.ylabel("Count")
plt.show()

# --------------------------------
# 2. Distribution of epistatic effects γ_ij (non-zero only)
# --------------------------------
epi_nonzero = df_epistasis.loc[df_epistasis["gamma"] != 0, "gamma"]

plt.figure(figsize=(10, 6))
sns.histplot(
    epi_nonzero,
    bins=40,
    kde=True,
    color="purple",
    alpha=0.7
)
plt.title("Distribution of Epistatic Effects (γᵢⱼ, non-zero)")
plt.xlabel("γ value")
plt.ylabel("Count")
plt.show()

# --------------------------------
# 3. Distribution of total PRS (main + epistasis)
# --------------------------------
plt.figure(figsize=(10, 6))
plt.hist(prs_total, bins=50, color="gray", alpha=0.8)
plt.title("Distribution of Total PRS (Main + Epistasis)")
plt.xlabel("PRS_total")
plt.ylabel("Count")
plt.show()

# --------------------------------
# 4. Distribution of disease probability P(D)
# --------------------------------
plt.figure(figsize=(10, 6))
plt.hist(P_D, bins=50, color="darkred", alpha=0.8)
plt.title("Distribution of Disease Probability P(D)")
plt.xlabel("P(D)")
plt.ylabel("Count")
plt.show()

# --------------------------------
# 5. P(D) distribution by disease status (Healthy vs Diseased)
# --------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df_individuals,
    x="P(D)",
    hue="label",
    bins=40,
    kde=True,
    alpha=0.6,
    palette={0: "green", 1: "red"}
)
plt.title("Distribution of P(D) by Disease Status")
plt.xlabel("P(D)")
plt.ylabel("Count")
plt.legend(title="Label", labels=["Healthy (0)", "Diseased (1)"])
plt.show()
