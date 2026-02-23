import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


# TEMP plot to compare global feature importance from Tree SHAP and Kernel SHAP
def plot_shap_comparison(comparison_df):
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=comparison_df, x="imp_tree", y="imp_kernel", alpha=0.7, edgecolor=None
    )

    max_val = max(comparison_df["imp_tree"].max(), comparison_df["imp_kernel"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Match (y=x)")

    p_corr, _ = pearsonr(comparison_df["imp_tree"], comparison_df["imp_kernel"])
    s_corr, _ = spearmanr(comparison_df["imp_tree"], comparison_df["imp_kernel"])

    plt.title(
        f"Tree SHAP vs Kernel SHAP Global Importance\n"
        f"Pearson: {p_corr:.3f} | Spearman: {s_corr:.3f}"
    )
    plt.xlabel("Tree SHAP Importance (Mean Abs Log-Odds)")
    plt.ylabel("Kernel SHAP Importance (Mean Abs Log-Odds)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()
