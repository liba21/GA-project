import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score

# =========================================================
# LOAD DATA
# =========================================================

df_individuals = pd.read_csv("population.csv")
df_gwas = pd.read_csv("gwas_results.csv")
df_real = pd.read_csv("real_epistasis.csv")

locus_cols = [c for c in df_individuals.columns if c.startswith("Locus_")]
genotypes = df_individuals[locus_cols].values
y_true = df_individuals["label"].values

# =========================================================
# LOAD GWAS BETAS
# =========================================================

beta_map = dict(zip(df_gwas["locus"], df_gwas["beta_hat"]))

# =========================================================
# LOAD TRUE EPISTASIS
# =========================================================

true_gamma_map = {
    tuple(sorted((int(i), int(j)))): g
    for i, j, g in zip(df_real["i"], df_real["j"], df_real["gamma"])
    if g != 0
}

# =========================================================
# PRECOMPUTE MAIN PRS
# =========================================================

PRS_MAIN = np.zeros(len(df_individuals))

for locus, beta in beta_map.items():
    if locus in df_individuals.columns and not np.isnan(beta):
        PRS_MAIN += df_individuals[locus].values * beta

# =========================================================
# PRECOMPUTE ALL PAIRS
# =========================================================

print("Precomputing interactions...")

pair_matrix = {}
n_loci = len(locus_cols)

for i in range(n_loci):
    xi = df_individuals[f"Locus_{i}"].values - 1

    for j in range(i + 1, n_loci):
        xj = df_individuals[f"Locus_{j}"].values - 1
        pair_matrix[(i, j)] = xi * xj

all_pairs = list(pair_matrix.keys())

print("Precompute done.")
print("Total candidate pairs:", len(all_pairs))
print("Total true pairs:", len(true_gamma_map))

# =========================================================
# GA PARAMETERS
# =========================================================

POP_SIZE = 1000
N_GENERATIONS = 400

TOURNAMENT_K = 5
ELITE_SIZE = 10

FIXED_SIZE = 125

MUTATION_PAIR_PROB = 0.5
MUTATION_GAMMA_PROB = 0.5

GAMMA_MIN = -0.4
GAMMA_MAX = 0.4

# =========================================================
# INITIAL POPULATION
# =========================================================

def random_individual():
    pairs = random.sample(all_pairs, FIXED_SIZE)

    return [
        (pair, np.random.uniform(-0.1, 0.1))
        for pair in pairs
    ]

population = [random_individual() for _ in range(POP_SIZE)]

# =========================================================
# FITNESS
# =========================================================

def compute_fitness(individual):

    prs = PRS_MAIN.copy()

    for pair, gamma in individual:
        prs += gamma * pair_matrix[pair]

    auc = roc_auc_score(y_true, prs)

    return auc

# =========================================================
# RECOVERY METRICS
# =========================================================

def compute_recovery_metrics(individual, top_k=125):

    sorted_ind = sorted(
        individual,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    true_found = 0
    gamma_errors = []

    for pair, gamma_pred in sorted_ind:

        if pair in true_gamma_map:
            true_found += 1
            gamma_true = true_gamma_map[pair]
            gamma_errors.append(abs(gamma_pred - gamma_true))

    recall = true_found / len(true_gamma_map)

    if len(gamma_errors) > 0:
        gamma_err = np.mean(gamma_errors)
    else:
        gamma_err = np.nan

    return recall, gamma_err

# =========================================================
# TOURNAMENT SELECTION
# =========================================================

def tournament_select(population, scores):

    idxs = np.random.choice(len(population), TOURNAMENT_K)
    best_idx = max(idxs, key=lambda i: scores[i])

    return population[best_idx]

# =========================================================
# CLEAN DUPLICATES
# =========================================================

def fix_duplicates(individual):

    used = set()
    cleaned = []

    for pair, gamma in individual:
        if pair not in used:
            cleaned.append((pair, gamma))
            used.add(pair)

    while len(cleaned) < FIXED_SIZE:
        new_pair = random.choice(all_pairs)

        if new_pair not in used:
            cleaned.append(
                (new_pair, np.random.uniform(-0.1, 0.1))
            )
            used.add(new_pair)

    return cleaned

# =========================================================
# MUTATION
# =========================================================

def mutate(individual):

    individual = individual.copy()

    r = np.random.rand()

    # Replace one pair
    if r < MUTATION_PAIR_PROB:

        idx = np.random.randint(FIXED_SIZE)

        existing_pairs = {p for p, _ in individual}

        new_pair = random.choice(all_pairs)

        while new_pair in existing_pairs:
            new_pair = random.choice(all_pairs)

        individual[idx] = (
            new_pair,
            np.random.uniform(-0.1, 0.1)
        )

    # Mutate gamma
    else:

        idx = np.random.randint(FIXED_SIZE)

        pair, gamma = individual[idx]

        gamma += np.random.normal(0, 0.05)
        gamma = np.clip(gamma, GAMMA_MIN, GAMMA_MAX)

        individual[idx] = (pair, gamma)

    return individual

# =========================================================
# CROSSOVER
# =========================================================

def crossover(parent1, parent2):

    cut = np.random.randint(1, FIXED_SIZE - 1)

    child = parent1[:cut] + parent2[cut:]

    child = fix_duplicates(child)

    return child

# =========================================================
# MAIN LOOP
# =========================================================

best_solution = None
best_score = -np.inf

for gen in range(N_GENERATIONS):

    scores = [compute_fitness(ind) for ind in population]

    best_idx = np.argmax(scores)
    gen_best = scores[best_idx]
    gen_mean = np.mean(scores)

    if gen_best > best_score:
        best_score = gen_best
        best_solution = population[best_idx]

    recovery, gamma_err = compute_recovery_metrics(
        population[best_idx],
        top_k=125
    )

    print(
        f"Gen {gen:3d} | "
        f"Best={gen_best:.4f} | "
        f"Mean={gen_mean:.4f} | "
        f"Recall@125={recovery:.4f} | "
        f"GammaErr={gamma_err:.4f}"
    )

    # ELITISM
    elite_idx = np.argsort(scores)[-ELITE_SIZE:]
    new_population = [population[i] for i in elite_idx]

    # REPRODUCTION
    while len(new_population) < POP_SIZE:

        parent1 = tournament_select(population, scores)
        parent2 = tournament_select(population, scores)

        child = crossover(parent1, parent2)
        child = mutate(child)

        new_population.append(child)

    population = new_population

# =========================================================
# FINAL RESULTS
# =========================================================

print("\nFinal best AUC:", best_score)
print("Final best size:", len(best_solution))

final_recall, final_gamma_err = compute_recovery_metrics(
    best_solution,
    top_k=125
)

print("Final Recall@125:", final_recall)
print("Final Gamma Error:", final_gamma_err)
# =========================================================
# FINAL RESULTS
# =========================================================
import csv

def export_true_pairs(best_solution, filename="best_solution_true_pairs.csv"):

    rows = []

    for pair, gamma_pred in best_solution:

        # רק זוגות אמיתיים
        if pair in true_gamma_map:

            gamma_true = true_gamma_map[pair]

            rows.append([
                pair[0],
                pair[1],
                gamma_pred,
                gamma_true,
                abs(gamma_pred - gamma_true)
            ])

    # שמירה ל-CSV
    df_out = pd.DataFrame(
        rows,
        columns=["i", "j", "gamma_pred", "gamma_true", "abs_error"]
    )

    df_out.to_csv(filename, index=False)

    print(f"\nSaved {len(df_out)} true pairs to {filename}")

# הפעלה על הפתרון הטוב ביותר
export_true_pairs(best_solution)