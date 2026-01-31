from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_individuals = pd.read_csv("population.csv")
df_gwas = pd.read_csv("gwas_results.csv")

# dictionary creation: locus -> beta_hat
beta_hat_dict = dict(
    zip(df_gwas["locus"], df_gwas["beta_hat"])
)

PRS_gwas = np.zeros(len(df_individuals))

for locus, beta_hat in beta_hat_dict.items():
    if not np.isnan(beta_hat):
        PRS_gwas += df_individuals[locus] * beta_hat
 #----------------------------------------------------------

y_true = df_individuals["label"]
y_score = PRS_gwas
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"GWAS-PRS ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve based on GWAS-estimated PRS")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ROC using true PRS (with epistasis)
auc_true = roc_auc_score(df_individuals["label"], df_individuals["logit"])

# ROC using GWAS PRS
auc_gwas = roc_auc_score(df_individuals["label"], PRS_gwas)

print(f"AUC true model: {auc_true:.3f}")
print(f"AUC GWAS PRS: {auc_gwas:.3f}")

# ------------------------------------------------------------------------------------
# recall and precision: 2 different ways to calculate them with different thresholds
# ------------------------------------------------------------------------------------
from sklearn.metrics import (
    roc_curve,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score
)
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Option 1: Threshold via ROC (Youden’s J)
# ------------------------------------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
threshold_youden = roc_thresholds[best_idx]

y_pred_youden = (y_score >= threshold_youden).astype(int)

precision_youden = precision_score(y_true, y_pred_youden)
recall_youden = recall_score(y_true, y_pred_youden)

print("=== Threshold via Youden’s J ===")
print(f"Threshold: {threshold_youden:.4f}")
print(f"Precision: {precision_youden * 100:.2f}%")
print(f"Recall:    {recall_youden * 100:.2f}%")

# ------------------------------------------------------------
# Precision–Recall Curve
# ------------------------------------------------------------
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
    y_true, y_score
)

avg_precision = average_precision_score(y_true, y_score)

plt.figure(figsize=(7, 6))
plt.plot(
    recall_curve,
    precision_curve,
    label=f"PRS (AP = {avg_precision:.3f})"
)

# Mark the two operating points
plt.scatter(
    recall_youden,
    precision_youden,
    color="red",
    label="Youden threshold",
    zorder=3
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (GWAS PRS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



