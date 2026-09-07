import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("../results/GA/best_solution_true_pairs.csv")

# Create an ID for each pair (for the X axis)
df["pair_id"] = df["i"].astype(str) + "," + df["j"].astype(str)

# Sorting to make the graph stable
df = df.sort_values("gamma_true").reset_index(drop=True)

x = np.arange(len(df))

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(16, 6))

# true gamma
plt.scatter(
    x,
    df["gamma_true"],
    label="True γ",
    alpha=0.8
)

# estimated gamma
plt.scatter(
    x,
    df["gamma_pred"],
    label="Predicted γ",
    alpha=0.8
)

# Connecting lines between truth and prediction
for i in range(len(df)):
    plt.plot(
        [x[i], x[i]],
        [df.loc[i, "gamma_true"], df.loc[i, "gamma_pred"]],
        color="gray",
        alpha=0.3
    )

# =========================================================
# STYLE
# =========================================================

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Epistatic pair index")
plt.ylabel("Gamma effect size")
plt.title("True vs Predicted Epistatic Effects (Best GA Solution)")
plt.legend()
plt.tight_layout()

os.makedirs("../figures/GA", exist_ok=True)
plt.savefig("../figures/GA/final_found_pairs_graph.png", dpi=300, bbox_inches="tight")

plt.show()