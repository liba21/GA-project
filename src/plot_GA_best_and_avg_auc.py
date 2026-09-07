import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "../results/GA/GA_auc_history.csv"
)


# ==============================
# PLOT
# ==============================

plt.figure(figsize=(8,5))


plt.plot(
    df["generation"],
    df["best_auc"],
    label="Best AUC",
    marker="o"
)


plt.plot(
    df["generation"],
    df["mean_auc"],
    label="Mean AUC",
    marker="o"
)


plt.xlabel(
    "Generation"
)

plt.ylabel(
    "AUC"
)

plt.title(
    "GA convergence: Best vs Mean AUC"
)


plt.legend()

plt.grid(True)

os.makedirs("../figures/GA", exist_ok=True)
plt.savefig("../figures/GA/GA_best_and_avg_auc.png", dpi=300, bbox_inches="tight")

plt.show()