import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# ---------------------------
# 1. Number of loci
# ---------------------------
# set the number of genetic loci to 100
n_loci = 100

# ---------------------------
# 2. Creating the main-effect table βᵢ
# ---------------------------
# For each locus, you sample a random number r.
# Based on r, you assign:
# - Small effect (0.02–0.08) with 60% probability
# - Medium effect (0.08–0.20) with 30% probability
# - Large effect (0.20–0.40) with 10% probability
# Then, with 50% probability, you flip the sign → making it protective (negative β).
betas = []
for i in range(n_loci):

    r = np.random.rand()

    if r < 0.6:  # Small effect
        b = np.random.uniform(0.02, 0.08)

    elif r < 0.9:  # Medium effect
        b = np.random.uniform(0.08, 0.20)

    else:  # Large effect
        b = np.random.uniform(0.20, 0.40)

    # 50% protective
    if np.random.rand() < 0.5:
        b *= -1

    betas.append(b)

# Finally, you store everything in a pandas table df_betas.
betas = np.array(betas)
df_betas = pd.DataFrame({"locus": np.arange(n_loci), "beta": betas})

# ---------------------------
# 3. Choosing 10% of all locus pairs to have epistasis
# ---------------------------
# All possible unique pairs (i,j) are generated:
pairs = list(itertools.combinations(range(n_loci), 2))
n_pairs = len(pairs)

epi_count = int(n_pairs * 0.10)  # 10%
# Out of all possible pairs, you sample 10% randomly to be "epistatic":
# These are the pairs where γᵢⱼ ≠ 0.
epi_pairs_idx = np.random.choice(np.arange(n_pairs), epi_count, replace=False)

# ---------------------------
# 4. Sampling γᵢⱼ ~ Normal(0, σ²)
# ---------------------------
# initialize all γ-values as zero (no interaction).
gamma = np.zeros(n_pairs)

# Then, for the selected epistatic pairs, sample: γᵢⱼ ~ Normal(0, σ²)
# with σ = 0.15 → moderately strong epistasis.
sigma = 0.15
gamma[epi_pairs_idx] = np.random.normal(0, sigma, epi_count)

# Ensuring a mix of positive and negative interactions
# by fliping the sign of ~50% of the epistatic values:
mask = np.random.rand(epi_count) < 0.5  # half negative
gamma[epi_pairs_idx[mask]] *= -1

# store the interactions in a table df_epistasis.
df_epistasis = pd.DataFrame({
    "i": [pairs[k][0] for k in range(n_pairs)],
    "j": [pairs[k][1] for k in range(n_pairs)],
    "gamma": gamma
})

# ---------------------------
# 5. Generating 100,000 diploid genotypes
# ---------------------------
# Each individual has 100 loci, each with 0/1/2 copies of the risk allele.
# Genotype probabilities:
# - 0 alleles: 25%
# - 1 allele: 50%
# - 2 alleles: 25%
# This simulates a realistic Hardy–Weinberg distribution.
N = 100000
genotypes = np.random.choice([0, 1, 2], size=(N, n_loci),
                             p=[0.25, 0.5, 0.25])

# ---------------------------
# 6. Calculating PRS without epistasis
# ---------------------------
# You multiply genotypes × main effects, by a matrix multiplication
# between genotypes matrix with betas vektor, and then sums it up.
PRS = genotypes @ betas

# ---------------------------
# 7. Adding epistatic effects
# ---------------------------
#
epi_effect = np.zeros(N)

# For each epistatic pair (i,j), you calculate: γij * xi * xj
for idx, (i, j) in enumerate(pairs):
    if gamma[idx] != 0:
        # and add it to the epistatic effect for each individual.
        epi_effect += gamma[idx] * genotypes[:, i] * genotypes[:, j]

# Sum all interaction contributions → PRS_total.
PRS_total = PRS + epi_effect

# ---------------------------
# 8. Computing logit and P(D)
# ---------------------------
# begin with a baseline prevalence P0 and calculate alpha accordingly:
p0 = 0.05
alpha = np.log(p0 / (1 - p0))

# later we will consider P0, and then we will do:
# logit = alpha + PRS_total.
logit = PRS_total
P_D = 1 / (1 + np.exp(-logit))

# ---------------------------
# 9. Building the population table
# ---------------------------
# create df_individuals with:
# - 100 genotype columns (Locus_0 ... Locus_99)
# - 1 column: logit
# - 1 column: P(D)
df_individuals = pd.DataFrame(genotypes, columns=[f"Locus_{i}" for i in range(n_loci)])
df_individuals["logit"] = logit
df_individuals["P(D)"] = P_D

# ---------------------------
# 10. Printing example rows
# ---------------------------
print("\n--- df_betas (main effects) ---")
print(df_betas.head())

print("\n--- df_epistasis (epistatic pairs) ---")
print(df_epistasis.head())

print("\n--- df_individuals (population table) ---")
print(df_individuals.head())

# ---------------------------
# 11. Plotting distributions
# ---------------------------
# 1. Distribution of total PRS (with epistasis):
plt.figure(figsize=(10, 6))
plt.hist(PRS_total, bins=50)
plt.title("Distribution of total PRS (with epistasis)")
plt.xlabel("PRS_total")
plt.ylabel("Frequency")
plt.show()

# 2. Distribution of disease probabilities P(D)
plt.figure(figsize=(10, 6))
plt.hist(P_D, bins=50)
plt.title("Distribution of disease probabilities P(D)")
plt.xlabel("P(D)")
plt.ylabel("Frequency")
plt.show()
