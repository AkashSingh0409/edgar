# %% [markdown]
# # Spatial Analysis & ML-Based Prediction of Global CO2 Emissions (EDGAR 1995, 1-degree grid)
# Follows the report's 9 experiments. Run as a notebook (`# %%` cells) or: `python edgar_co2_pipeline.py`
# Needs `edgar_core.py` in the same folder.  pip install -r requirements.txt

# %% [markdown]
# ## 0. Setup & configuration

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import edgar_core as ec

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
pd.set_option("display.width", 160)

# >>> USE YOUR REAL EDGAR FILE <<<
# Put your path here, e.g. REAL_DATA_PATH = "edgar_co2_data_1d_1995.csv".
# Metadata lines at the top are detected & skipped automatically (no hard-coded skiprows).
# Leave as None to use the synthetic EDGAR-like fallback.
REAL_DATA_PATH = None

# %% [markdown]
# ## Phase 1: Data understanding (source + metadata)

# %%
if REAL_DATA_PATH:
    raw, metadata = ec.load_edgar_csv(REAL_DATA_PATH)
    print(f"Loaded real EDGAR file: {REAL_DATA_PATH}")
else:
    print("[INFO] REAL_DATA_PATH is None -> using synthetic EDGAR-like data (1-degree grid, kg CO2/yr).")
    raw, metadata = ec.generate_synthetic_edgar(), "(synthetic - no metadata)"
print("--- metadata ---\n", metadata, "\n--- first rows ---")
print(raw.head())

# %% [markdown]
# ## Phase 2: Data cleaning

# %%
df, clean_summary = ec.prepare(raw)
print(pd.Series(clean_summary, name="count").to_frame())
print(f"\nUnique non-zero grid cells analysed: {len(df):,}")
print("Note: zero-emission cells are not part of the data, so conclusions describe recorded non-zero cells only.")

# %% [markdown]
# ## Phase 3: Exploratory & descriptive analysis

# %%
print(df[ec.COLS].describe(percentiles=[.25, .5, .75]).T)

# Fig 1 - spatial distribution
fig, ax = plt.subplots(figsize=(12, 5.5))
sc = ax.scatter(df.Longitude, df.Latitude, c=df.Log10_Emission, s=6, cmap="inferno")
plt.colorbar(sc, label="log10(CO2 emission, kg/yr)")
ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude",
       title="Fig 1: Global spatial distribution of recorded EDGAR CO2 emissions")
plt.tight_layout(); plt.show()

# Fig 2 - log-scale histogram (raw values span many orders of magnitude)
plt.figure(figsize=(8, 4))
sns.histplot(df.Log10_Emission, bins=50, kde=True, color="steelblue")
plt.title("Fig 2: CO2 emission distribution (log10 scale)"); plt.xlabel("log10(kg CO2 / year)")
plt.tight_layout(); plt.show()

# Boxplot (outlier detection)
plt.figure(figsize=(8, 3.5))
sns.boxplot(x=df.CO2_Emission, color="lightsteelblue"); plt.xscale("log")
plt.title("Boxplot of CO2 emission (log x-axis)"); plt.tight_layout(); plt.show()

# Fig 3 - correlation matrix
plt.figure(figsize=(6, 5))
sns.heatmap(df[["Longitude", "Latitude", "Abs_Latitude", "CO2_Emission", "Log10_Emission"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Fig 3: Correlation matrix"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 4: Feature engineering & PCA

# %%
print(df[["Longitude", "Latitude", "Abs_Latitude", "Hemisphere", "Distance_From_Equator", "Emission_Level"]].head())
print("\nEmission_Level (tertiles):\n", df.Emission_Level.value_counts())

pca = ec.pca_analysis(df)
for i, (v, c) in enumerate(zip(pca["explained"], pca["cumulative"]), 1):
    print(f"PC{i}: {v:.2%}  (cumulative {c:.2%})")
print(f"-> {pca['n_components']} component(s) retain >= 85% variance")

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(pca["cumulative"]) + 1), pca["cumulative"], marker="o")
plt.axhline(0.85, color="red", ls="--", label="85% variance")
plt.xticks(range(1, len(pca["cumulative"]) + 1)); plt.legend()
plt.xlabel("Principal components"); plt.ylabel("Cumulative explained variance")
plt.title("Fig 4: PCA cumulative explained variance"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 5: Hypothesis testing (Northern vs Southern Hemisphere)
# H0: mean emission is equal in both hemispheres.  H1: means differ.  alpha = 0.05

# %%
res = ec.hemisphere_test(df)
print(f"North: n={res['n_north']:,}, mean={res['mean_north']:,.0f} kg/yr")
print(f"South: n={res['n_south']:,}, mean={res['mean_south']:,.0f} kg/yr")
print(f"Welch t-test           : t = {res['t']:.3f}, p = {res['p']:.3e} -> {res['decision']}")
print(f"Robustness (log10 Welch): t = {res['t_log10']:.3f}, p = {res['p_log10']:.3e}")
print(f"Robustness (Mann-Whitney): U = {res['U']:,.0f}, p = {res['p_mannwhitney']:.3e}")
print("Caution: a statistical difference is NOT causal (population, industry, energy mix are omitted).")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=df, x="Hemisphere", y="CO2_Emission", estimator="mean", errorbar=None, ax=axes[0])
axes[0].set_title("Mean emission per cell")
sns.boxplot(data=df, x="Hemisphere", y="CO2_Emission", ax=axes[1]); axes[1].set_yscale("log")
axes[1].set_title("Distribution (log scale)")
plt.suptitle("Fig 5: Hemisphere-wise CO2 emission comparison"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 6: Traditional vs contemporary statistics

# %%
st = ec.descriptive_stats(df.CO2_Emission)
st_log = ec.descriptive_stats(df.Log10_Emission)
table = pd.DataFrame({
    "Traditional (raw kg/yr)": [st[k] for k in ["mean", "median", "std", "min", "max", "skewness",
                                               "excess_kurtosis", "IQR_upper_fence", "IQR_outliers"]],
}, index=["Mean", "Median", "Std dev", "Min", "Max", "Skewness", "Excess kurtosis",
          "IQR upper fence", "IQR outliers (cells)"])
print(table)
print(f"\nShapiro-Wilk (5,000-sample): W = {st['shapiro_W']:.4f}, p = {st['shapiro_p']:.3e} -> non-normal")
print(f"After log10 transform: skewness = {st_log['skewness']:.2f}, Shapiro W = {st_log['shapiro_W']:.4f}")
print("\nContemporary view - concentration of emissions:")
print(f"  Gini = {st['gini']:.3f} | top 1% of cells = {st['top1pct_share']:.1%} | "
      f"top 5% = {st['top5pct_share']:.1%} | top 10% = {st['top10pct_share']:.1%} of total")
print("IQR outliers are RETAINED: extreme cells are likely genuine high-emission areas, not errors.")

# %% [markdown]
# ## Phase 7: Model development (Longitude, Latitude -> CO2_Emission, 80/20 split)

# %%
results, models, (Xtr, Xte, ytr, yte), preds = ec.holdout_evaluation(df)
print(f"Train = {len(Xtr):,} | Test = {len(Xte):,}\n")
print(results.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

plt.figure(figsize=(6, 4))
sns.barplot(data=results, x="Model", y="R2", hue="Model", legend=False)
plt.axhline(0, color="k", lw=.8); plt.title("Fig 6: Regression model R² comparison (test set)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Phase 8: Model evaluation & 5-fold cross-validation

# %%
cv = ec.cross_validate_models(df)
print(cv.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print("\nNegative mean R2 = worse than always predicting the mean.")

# --- Extension implementing the report's own recommendations ---
# (a) log10 target tames the extreme skew; (b) spatially blocked validation (30x30 deg blocks)
# tests generalisation to UNSEEN regions instead of neighbouring cells of the training set.
print("\nRandom Forest on log10(emission): random vs spatially blocked 5-fold CV")
print(ec.log_target_cv_comparison(df).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print("A big gap between the two rows = optimistic scores caused by spatial autocorrelation/leakage.")

# %% [markdown]
# ## Phase 9: Model interpretation

# %%
rf = models["Random Forest"]
imp = ec.rf_feature_importance(rf)
print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("Importance is model-specific, NOT causal.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=imp, x="Feature", y="Importance", hue="Feature", legend=False, ax=axes[0])
axes[0].set_title("Fig 7: Random Forest feature importance")
axes[1].scatter(yte, preds["Random Forest"], s=6, alpha=.5)
lim = [yte.min(), max(yte.max(), preds["Random Forest"].max())]
axes[1].plot(lim, lim, "r--"); axes[1].set(xscale="log", yscale="log", xlabel="Actual", ylabel="Predicted",
                                           title="Fig 8: Random Forest actual vs predicted")
plt.tight_layout(); plt.show()

print("\nLargest emission cells:")
print(ec.top_cells(df, 10).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# %% [markdown]
# ## Phase 10: Case study - prioritising locations for investigation

# %%
# Cumulative-share (Lorenz-style) curve: how few cells carry most emissions?
srt = np.sort(df.CO2_Emission.values)[::-1]
cum_share = np.cumsum(srt) / srt.sum()
x = np.arange(1, len(srt) + 1) / len(srt)
plt.figure(figsize=(6, 4))
plt.plot(x * 100, cum_share * 100); plt.axvline(5, color="grey", ls=":")
plt.xlabel("Top X% of cells"); plt.ylabel("Share of total emissions (%)")
plt.title("Cumulative emission share (hotspot concentration)"); plt.tight_layout(); plt.show()

THRESH = 0.95  # flag the cells forming the top 5% by emission
priority = df[df.CO2_Emission >= df.CO2_Emission.quantile(THRESH)].sort_values("CO2_Emission", ascending=False)
print(f"Priority hotspot list: {len(priority):,} cells carry {priority.CO2_Emission.sum() / df.CO2_Emission.sum():.1%} of emissions")
priority[["Longitude", "Latitude", "CO2_Emission", "Hemisphere"]].to_csv("priority_hotspots.csv", index=False)
print("Saved -> priority_hotspots.csv  (next: join with population / industry / energy data)")
print("\nPipeline complete.")
