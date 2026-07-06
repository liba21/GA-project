import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("best_solution_true_pairs.csv")

# יצירת מזהה לכל זוג (לציר X)
df["pair_id"] = df["i"].astype(str) + "," + df["j"].astype(str)

# מיון כדי שהגרף יהיה יציב
df = df.sort_values("gamma_true").reset_index(drop=True)

x = np.arange(len(df))

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(16, 6))

# גמא אמיתי
plt.scatter(
    x,
    df["gamma_true"],
    label="True γ",
    alpha=0.8
)

# גמא שנלמד
plt.scatter(
    x,
    df["gamma_pred"],
    label="Predicted γ",
    alpha=0.8
)

# חיבור קווים בין אמת לניבוי (נותן תחושת error)
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

plt.show()