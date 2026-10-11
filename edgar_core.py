"""edgar_core.py - shared logic for the EDGAR CO2 capstone (pipeline + dashboard).

Data: EDGAR annual CO2 emissions on a 1x1 degree grid (kg CO2 / year).
Raw file = ~11 metadata lines followed by `Longitude,Latitude,CO2_Emission` rows.
"""
from __future__ import annotations

import io
import re
from typing import IO, Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
COLS = ["Longitude", "Latitude", "CO2_Emission"]
PCA_FEATURES = ["Longitude", "Latitude", "Abs_Latitude"]

# ----------------------------------------------------------------------------
# 1. Data sourcing: real EDGAR loader + synthetic fallback
# ----------------------------------------------------------------------------
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_DATA_LINE = re.compile(rf"^\s*{_NUM}\s*([,;\t])\s*{_NUM}\s*\1\s*{_NUM}\s*$")


def load_edgar_csv(source: Union[str, IO]) -> Tuple[pd.DataFrame, str]:
    """Read a raw EDGAR file (path, file-like text or bytes). Metadata lines are
    detected automatically (everything before the first numeric 3-column row), so
    the '11 lines' assumption is not hard-coded. Returns (raw_df, metadata_text)."""
    try:
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        else:
            content = source.read()
            text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    except OSError as exc:
        raise RuntimeError(f"Could not read EDGAR file: {exc}") from exc

    lines = text.splitlines()
    first, sep = None, ","
    for i, line in enumerate(lines):
        m = _DATA_LINE.match(line)
        if m:
            first, sep = i, m.group(1)
            break
    if first is None:
        raise ValueError("No 'Longitude,Latitude,CO2_Emission' numeric rows found in the file.")
    meta = "\n".join(lines[:first])
    df = pd.read_csv(io.StringIO("\n".join(lines[first:])), header=None, names=COLS, sep=sep,
                     engine="python", on_bad_lines="skip")
    return df, meta


# (lon, lat, sigma_deg, weight) - stylised emission regions
_HOTSPOTS = [(-85, 40, 9, 1.0), (-118, 34, 3, .5), (-96, 30, 3, .5), (5, 50, 8, .9),
             (30, 55, 10, .5), (115, 33, 8, 1.0), (135, 36, 3, .8), (127, 37, 2, .5),
             (78, 23, 8, .6), (47, 30, 6, .35), (28, -26, 4, .35), (-47, -23, 5, .4),
             (-60, -35, 5, .2), (135, -30, 10, .12), (-99, 20, 3, .25), (110, -7, 4, .2),
             (31, 30, 3, .2)]


def generate_synthetic_edgar(n_cells: int = 11749, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic EDGAR-like 1-degree grid: unique integer cells, heavy right-skew,
    concentrated hotspots (US, Europe, East Asia, India...), ~3:1 North:South split.
    Not a land mask - a few cells may fall over ocean. Replace with real data."""
    rng = np.random.default_rng(seed)
    lon, lat = np.arange(-179, 181), np.arange(-89, 90)
    LON, LAT = np.meshgrid(lon, lat)
    intensity = np.zeros(LON.shape)
    for x0, y0, s, w in _HOTSPOTS:
        intensity += w * np.exp(-((LON - x0) ** 2 + (LAT - y0) ** 2) / (2 * s ** 2))
    p = (0.01 + intensity).ravel()
    idx = rng.choice(p.size, size=n_cells, replace=False, p=p / p.sum())
    I = np.clip(intensity.ravel()[idx], 0, 1.5)
    # Smooth spatially-correlated random field (neighbouring cells behave alike, as in real data)
    field = np.zeros(LON.shape)
    for _ in range(120):
        cx, cy, sg = rng.uniform(-180, 180), rng.uniform(-60, 75), rng.uniform(3, 7)
        field += rng.normal() * np.exp(-((LON - cx) ** 2 + (LAT - cy) ** 2) / (2 * sg ** 2))
    field = ((field - field.mean()) / field.std()).ravel()[idx]
    log_e = 8.1 + 0.9 * I + 0.55 * field + rng.normal(0, 0.8, n_cells)
    # soft-saturate the upper tail towards log10(2.75e11)=11.44 (avoids many tied maxima)
    log_e = np.where(log_e > 10.0, 10.0 + 1.44 * np.tanh((log_e - 10.0) / 1.44), log_e)
    emission = np.clip(10 ** log_e, 8_530, 2.75e11)
    return pd.DataFrame({"Longitude": LON.ravel()[idx].astype(float),
                         "Latitude": LAT.ravel()[idx].astype(float),
                         "CO2_Emission": emission.round(0)}).reset_index(drop=True)


def write_edgar_style_csv(df: pd.DataFrame, path: str) -> None:
    """Write df with 11 EDGAR-like metadata lines (to test load_edgar_csv)."""
    meta = ["# EDGAR (synthetic test file)", "# Annual CO2 emissions 1995", "# Grid: 1 x 1 degree",
            "# Unit: kg CO2 per year", f"# Non-zero cells: {len(df)}"] + ["#"] * 6
    with open(path, "w") as fh:
        fh.write("\n".join(meta) + "\n")
        df[COLS].to_csv(fh, header=False, index=False)


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
def clean_edgar(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Numeric coercion, drop invalid / out-of-range / non-positive / duplicate rows."""
    n0 = len(raw)
    df = raw[COLS].apply(pd.to_numeric, errors="coerce")
    n_invalid = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    n_range = int((~(df.Longitude.between(-180, 180) & df.Latitude.between(-90, 90))).sum())
    df = df[df.Longitude.between(-180, 180) & df.Latitude.between(-90, 90)]
    n_nonpos = int((df.CO2_Emission <= 0).sum())
    df = df[df.CO2_Emission > 0]
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    return df, {"raw_rows": n0, "invalid_removed": n_invalid, "out_of_range_removed": n_range,
                "non_positive_removed": n_nonpos, "duplicates_removed": n_dup, "clean_rows": len(df)}


# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Abs_Latitude, Hemisphere, Distance_From_Equator, Emission_Level (tertiles), Log10_Emission.
    NB: as in the report, Distance_From_Equator == Abs_Latitude (perfectly collinear)."""
    f = df.copy()
    f["Abs_Latitude"] = f["Latitude"].abs()
    f["Hemisphere"] = np.where(f["Latitude"] >= 0, "Northern", "Southern")
    f["Distance_From_Equator"] = f["Abs_Latitude"]
    f["Emission_Level"] = pd.qcut(f["CO2_Emission"], 3, labels=["Low", "Medium", "High"])
    f["Log10_Emission"] = np.log10(f["CO2_Emission"])
    return f


def pca_analysis(df: pd.DataFrame, threshold: float = 0.85) -> dict:
    """VarianceThreshold -> StandardScaler -> PCA on the geographic features."""
    X = VarianceThreshold(0).fit_transform(df[PCA_FEATURES])
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(random_state=RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return {"explained": pca.explained_variance_ratio_, "cumulative": cum,
            "n_components": int(np.argmax(cum >= threshold) + 1), "scores": pca.transform(Xs)}


# ----------------------------------------------------------------------------
# 4. Statistics
# ----------------------------------------------------------------------------
def hemisphere_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """H0: mean emission equal in Northern/Southern cells. Welch t-test (as in the report),
    plus robustness checks that suit heavily skewed data: Welch on log10 and Mann-Whitney U."""
    n = df.loc[df.Hemisphere == "Northern", "CO2_Emission"]
    s = df.loc[df.Hemisphere == "Southern", "CO2_Emission"]
    if len(n) < 2 or len(s) < 2:
        raise ValueError(f"Need >=2 cells per hemisphere (North={len(n)}, South={len(s)}).")
    t, p = stats.ttest_ind(n, s, equal_var=False)
    tl, pl = stats.ttest_ind(np.log10(n), np.log10(s), equal_var=False)
    u, pu = stats.mannwhitneyu(n, s, alternative="two-sided")
    return {"n_north": len(n), "n_south": len(s), "mean_north": n.mean(), "mean_south": s.mean(),
            "t": t, "p": p, "decision": "Reject H0" if p < alpha else "Fail to reject H0",
            "t_log10": tl, "p_log10": pl, "U": u, "p_mannwhitney": pu}


def descriptive_stats(x: pd.Series, seed: int = RANDOM_STATE) -> dict:
    """Traditional + contemporary summary: moments, IQR outliers, Shapiro-Wilk, concentration."""
    q1, q3 = x.quantile([.25, .75])
    fence = q3 + 1.5 * (q3 - q1)
    sample = x.sample(min(5000, len(x)), random_state=seed)
    W, pW = stats.shapiro(sample)
    srt = np.sort(x.values)
    nn = len(srt)
    gini = (2 * np.sum(np.arange(1, nn + 1) * srt) / (nn * srt.sum())) - (nn + 1) / nn
    share = lambda frac: srt[-max(1, int(round(frac * nn))):].sum() / srt.sum()
    return {"count": nn, "total": x.sum(), "mean": x.mean(), "median": x.median(), "std": x.std(),
            "min": x.min(), "max": x.max(), "skewness": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
            "IQR_upper_fence": fence, "IQR_outliers": int((x > fence).sum()),
            "shapiro_W": W, "shapiro_p": pW, "gini": gini,
            "top1pct_share": share(.01), "top5pct_share": share(.05), "top10pct_share": share(.10)}


def top_cells(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t = df.nlargest(n, "CO2_Emission")[["Longitude", "Latitude", "CO2_Emission", "Hemisphere"]].copy()
    t["Share_of_Total_%"] = 100 * t["CO2_Emission"] / df["CO2_Emission"].sum()
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. Machine learning
# ----------------------------------------------------------------------------
def make_models() -> dict:
    return {"Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)}


def holdout_evaluation(df: pd.DataFrame, test_size: float = 0.2):
    """80/20 split (random_state=42) on [Longitude, Latitude] -> CO2_Emission."""
    X, y = df[["Longitude", "Latitude"]], df["CO2_Emission"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    models, rows, preds = make_models(), [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))), "R2": r2_score(yte, p)})
    return pd.DataFrame(rows), models, (Xtr, Xte, ytr, yte), preds


def cross_validate_models(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """5-fold CV (R2) per model, exactly as in the report (cv=5 -> contiguous unshuffled folds)."""
    X, y = df[["Longitude", "Latitude"]], df["CO2_Emission"]
    rows = []
    for name, m in make_models().items():
        s = cross_val_score(m, X, y, cv=cv, scoring="r2")
        rows.append({"Model": name, "Mean R2": s.mean(), "Std": s.std(),
                     **{f"Fold {i + 1}": v for i, v in enumerate(s)}})
    return pd.DataFrame(rows)


def spatial_blocks(df: pd.DataFrame, size_deg: int = 30) -> np.ndarray:
    """Integer id of the size_deg x size_deg geographic block containing each cell."""
    lon_b = np.floor((df.Longitude + 180) / size_deg).astype(int)
    lat_b = np.floor((df.Latitude + 90) / size_deg).astype(int)
    return (lon_b * 100 + lat_b).values


def log_target_cv_comparison(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Report's recommendations in action: log10 target + spatially blocked validation.
    Random Forest on [Lon, Lat] predicting log10(emission); R2 in log space.
    Shuffled KFold lets neighbouring cells sit in train AND test (optimistic);
    GroupKFold on 30-degree blocks tests generalisation to unseen regions."""
    X, y = df[["Longitude", "Latitude"]], df["Log10_Emission"]
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    schemes = {"Shuffled KFold (random)": KFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
                "Spatial-block GroupKFold": GroupKFold(n_splits)}
    rows = []
    for name, splitter in schemes.items():
        groups = spatial_blocks(df) if "Group" in name else None
        s = cross_val_score(rf, X, y, cv=splitter, groups=groups, scoring="r2")
        rows.append({"Validation scheme": name, "Mean R2 (log10)": s.mean(), "Std": s.std(),
                     **{f"Fold {i + 1}": v for i, v in enumerate(s)}})
    return pd.DataFrame(rows)


def rf_feature_importance(rf: RandomForestRegressor, cols=("Longitude", "Latitude")) -> pd.DataFrame:
    return (pd.DataFrame({"Feature": list(cols), "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def prepare(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """clean + feature engineering in one call."""
    clean, summary = clean_edgar(raw)
    return add_features(clean), summary
