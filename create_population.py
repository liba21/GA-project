import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Parameters
# ----------------------------------
target_prev = 0.02 #---------------------------------------------------------------change here to control final deasise ratio
# set the number of genetic loci to 100
n_loci = 100
N = 100000

maf_min = 0.02
maf_max = 0.5
maf_decay = 6.0   # controls how fast MAF decreases with |β|

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

# ----------------------------------
# 2. Define MAF as a function of |β|
# ----------------------------------
abs_beta = np.abs(betas)
maf = maf_min + (maf_max - maf_min) * np.exp(-maf_decay * abs_beta)
maf = np.clip(maf, maf_min, maf_max)

# ---------------------------
# 3. Choosing 10% of all locus pairs to have epistasis
# ---------------------------
# All possible unique pairs (i,j) are generated:
pairs = list(itertools.combinations(range(n_loci), 2))
n_pairs = len(pairs)

epi_count = int(n_pairs * 0.1)  # 10% ~ 495 epistatic pairs --------------------------------------------- change here when want 0% epistasis
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
# returns gamma about between -0.45 and +0.45 in the vast majority of cases,
# and most of it −0.15≤γ≤0.15 .
sigma = 0.15
gamma[epi_pairs_idx] = np.random.normal(0, sigma, epi_count)

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
# - 0 alleles: 25% (p^2)
# - 1 allele: 50% (2pq)
# - 2 alleles: 25% (q^2)
# This simulates a Hardy–Weinberg distribution, with p = 0.5
# and q = 1 - p = 0.5.
#genotypes = np.random.choice([0, 1, 2], size=(N, n_loci),  p=[0.25, 0.5, 0.25])
genotypes = np.zeros((N, n_loci), dtype=int)

for i in range(n_loci):
    p = maf[i]
    genotypes[:, i] = np.random.choice(
        [0, 1, 2],
        size=N,
        p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2]
    )

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

# the part that responsible to keep the ratio between the main effects contribution
# to the epistasis contribution
var_main = np.var(PRS)
var_epi = np.var(epi_effect)

def epistasis_scale(var_main, var_epi, epi_fraction):
    """
    var_main    : variance of main effects
    var_epi     : variance of epistatic effects
    epi_fraction: desired fraction of epistasis (0 < p < 1)
    """
    assert 0 < epi_fraction < 1, "epi_fraction must be between 0 and 1"
    scale = np.sqrt((epi_fraction / (1 - epi_fraction)) * (var_main / var_epi))
    return scale

if var_epi > 0:
    # target_ratio = 0.9 * var_main / var_epi
    # scale = np.sqrt(target_ratio)
    p = 0.3  # epistasis target ratio ---------------------------------------------------------change here to define epistasis ratio
    scale = epistasis_scale(var_main, var_epi, p)

    # Scale epistatic effects to control variance contribution
    epi_effect *= scale
    gamma *= scale
    df_epistasis["gamma"] = gamma
else:
    # No epistasis → no scaling
    scale = 1.0

# Sum all interaction contributions → PRS_total
PRS_total = PRS + epi_effect

# ---------------------------
# 8. Computing logit and P(D)
# ---------------------------

# Center PRS_total
PRS_total_centered = PRS_total - PRS_total.mean()

# Set alpha for desired prevalence
alpha = np.log(target_prev / (1 - target_prev))

# Logistic model
logit = alpha + PRS_total_centered
P_D = 1 / (1 + np.exp(-logit))
phenotype = np.random.binomial(1, P_D)

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
df_individuals["label"] = phenotype


# ---------------------------
# 11. Printing example rows and sending population to csv file
# ---------------------------
print("\n--- df_betas (main effects) ---")
print(df_betas.head())

print("\n--- df_epistasis (epistatic pairs) ---")
print(df_epistasis.head())

print("\n--- df_individuals (population table) ---")
print(df_individuals.head())

var_main = np.var(PRS)
var_epi  = np.var(epi_effect)

print("Main effects contribution:", var_main / (var_main + var_epi))
print("Epistasis contribution:", var_epi / (var_main + var_epi))

df_individuals.to_csv("population.csv", index=False)
df_betas.to_csv("real_betas.csv", index=False)
df_epistasis.to_csv("real_epistasis.csv", index=False)


# ---------------------------
# 12. Plotting distributions
# ---------------------------
# 1. Distribution of β_i (effects):
plt.figure(figsize=(10, 6))
sns.histplot(df_betas["beta"], bins=40, kde=True, color="blue", alpha=0.6)
plt.title("Distribution of Main Effects βᵢ")
plt.xlabel("βᵢ value")
plt.ylabel("Frequency")
plt.show()

# 2. Distribution of γ_ij (epistatic effects):
epi_nonzero = df_epistasis[df_epistasis["gamma"] != 0]["gamma"]

plt.figure(figsize=(10, 6))
sns.histplot(epi_nonzero, bins=40, kde=True, color="purple", alpha=0.6)
plt.title("Distribution of Epistatic Effects γᵢⱼ  (non-zero only)")
plt.xlabel("γᵢⱼ value")
plt.ylabel("Frequency")
plt.show()


# 3. Distribution of total PRS (with epistasis):
plt.figure(figsize=(10, 6))
plt.hist(PRS_total, bins=50)
plt.title("Distribution of total PRS (with epistasis)")
plt.xlabel("PRS_total")
plt.ylabel("Frequency")
plt.show()

# 4. Distribution of disease probabilities P(D)
plt.figure(figsize=(10, 6))
plt.hist(P_D, bins=50)
plt.title("Distribution of disease probabilities P(D)")
plt.xlabel("P(D)")
plt.ylabel("Frequency")
plt.show()

# 5. Histograms
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
plt.legend(title="Label", labels=["Diseased (1)", "Healthy (0)"])
plt.show()

num_total = len(df_individuals)
num_cases = df_individuals["label"].sum()          # סך חולים
num_controls = num_total - num_cases              # סך בריאים

print(f"Cases: {num_cases} ({num_cases/num_total*100:.2f}%)")
print(f"Controls: {num_controls} ({num_controls/num_total*100:.2f}%)")
