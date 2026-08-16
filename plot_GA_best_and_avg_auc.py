import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "GA_auc_history.csv"
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

plt.show()