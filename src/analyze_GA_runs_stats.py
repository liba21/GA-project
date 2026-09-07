import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# PARAMETERS
# =========================================================

RUN_FILES = [
    "GA_run_1_pairs.csv",
    "GA_run_2_pairs.csv",
    "GA_run_3_pairs.csv",
    "GA_run_4_pairs.csv",
    "GA_run_5_pairs.csv",
    "GA_run_6_pairs.csv"
]

OUTPUT_CSV = "../results/GA/pair_detection_statistics.csv"


# =========================================================
# LOAD TRUE EPISTASIS
# =========================================================

df_real = pd.read_csv("../data/real_epistasis.csv")

# Keep only true epistatic pairs (gamma != 0)
df_true = df_real[df_real["gamma"] != 0].copy()

# Make sure pair representation is sorted
df_true["i"] = df_true["i"].astype(int)
df_true["j"] = df_true["j"].astype(int)

df_true["pair"] = list(
    zip(
        df_true["i"],
        df_true["j"]
    )
)

print("Number of true epistatic pairs:", len(df_true))


# =========================================================
# LOAD GA RUNS
# =========================================================

run_data = []

for run_number, filename in enumerate(RUN_FILES, start=1):

    df_run = pd.read_csv(filename)

    # Make sure columns have the expected types
    df_run["i"] = df_run["i"].astype(int)
    df_run["j"] = df_run["j"].astype(int)

    # Make pair representation
    df_run["pair"] = list(
        zip(
            df_run["i"],
            df_run["j"]
        )
    )

    df_run["run"] = run_number

    run_data.append(df_run)


# =========================================================
# COMBINE ALL RUNS
# =========================================================

df_all_runs = pd.concat(
    run_data,
    ignore_index=True
)

print("Total GA results loaded:", len(df_all_runs))


# =========================================================
# CALCULATE STATISTICS FOR EACH TRUE PAIR
# =========================================================

results = []

for _, true_row in df_true.iterrows():

    i = true_row["i"]
    j = true_row["j"]
    gamma_true = true_row["gamma"]

    pair = (i, j)

    # All occurrences of this pair across all runs
    found = df_all_runs[
        df_all_runs["pair"] == pair
    ]

    times_found = len(found)

    detection_rate = (
        times_found / len(RUN_FILES)
    )

    # Mean predicted gamma only in runs where pair was found
    if times_found > 0:

        mean_gamma_pred = found["gamma_pred"].mean()

        gamma_error_when_found = abs(
            mean_gamma_pred - gamma_true
        )

    else:

        mean_gamma_pred = np.nan
        gamma_error_when_found = np.nan

    results.append([
        i,
        j,
        gamma_true,
        times_found,
        detection_rate,
        mean_gamma_pred,
        gamma_error_when_found
    ])


# =========================================================
# CREATE STATISTICS DATAFRAME
# =========================================================

df_statistics = pd.DataFrame(
    results,
    columns=[
        "i",
        "j",
        "gamma_true",
        "times_found",
        "detection_rate",
        "mean_gamma_pred",
        "gamma_error_when_found"
    ]
)


# =========================================================
# SORT BY TRUE GAMMA
# =========================================================

# Sort from weakest negative -> strongest negative
# -> weakest positive -> strongest positive
df_statistics = df_statistics.sort_values(
    by="gamma_true"
).reset_index(drop=True)


# =========================================================
# SAVE CSV
# =========================================================

df_statistics.to_csv(
    OUTPUT_CSV,
    index=False
)

print(f"\nSaved statistics to: {OUTPUT_CSV}")


# =========================================================
# PRINT SUMMARY
# =========================================================

print("\nDetection summary:")
print(
    f"Mean detection rate: "
    f"{df_statistics['detection_rate'].mean():.4f}"
)

print(
    f"Pairs found in all {len(RUN_FILES)} runs: "
    f"{(df_statistics['times_found'] == len(RUN_FILES)).sum()}"
)

print(
    f"Pairs never found: "
    f"{(df_statistics['times_found'] == 0).sum()}"
)


# =========================================================
# PLOT DETECTION FREQUENCY
# =========================================================

x = np.arange(len(df_statistics))

plt.figure(figsize=(18, 7))

plt.bar(
    x,
    df_statistics["times_found"]
)

plt.xlabel("True epistatic pairs (sorted by true gamma)")
plt.ylabel("Number of runs in which pair was found")
plt.title(
    "GA Pair Detection Frequency Across Runs"
)

plt.ylim(
    0,
    len(RUN_FILES)
)

plt.xticks([])

plt.tight_layout()

os.makedirs("../figures/GA", exist_ok=True)
plt.savefig("../figures/GA/GA_runs_stats.png", dpi=300, bbox_inches="tight")

plt.show()

print("Saved plot to: ../figures/GA/GA_runs_stats.png")