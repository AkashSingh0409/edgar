"""EDGAR CO2 Spatial Analytics dashboard.

Run:  pip install -r requirements.txt   then   streamlit run app.py
Keep edgar_core.py in the same folder.

Upload the raw EDGAR CSV (metadata lines are skipped automatically) or use the
synthetic EDGAR-like fallback.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import edgar_core as ec

st.set_page_config(page_title="EDGAR CO2 Spatial Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def load_synthetic() -> pd.DataFrame:
    return ec.generate_synthetic_edgar()


@st.cache_data(show_spinner=False)
def load_uploaded(data: bytes):
    import io
    return ec.load_edgar_csv(io.BytesIO(data))


@st.cache_data(show_spinner="Training models...")
def run_models(df: pd.DataFrame):
    results, models, (_, Xte, _, yte), preds = ec.holdout_evaluation(df)
    return (results, ec.cross_validate_models(df), ec.log_target_cv_comparison(df),
            ec.rf_feature_importance(models["Random Forest"]), yte.values, preds["Random Forest"])


def fmt_mass(kg: float) -> str:
    """kg -> human-readable (Mt / Gt)."""
    if kg >= 1e12:
        return f"{kg / 1e12:,.2f} Gt"
    return f"{kg / 1e9:,.2f} Mt"


# ------------------------------------------------------------------ data loading
st.title("Global CO₂ Emissions - EDGAR 1° Grid")
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload EDGAR CSV (.csv)", type="csv")

source, meta = "Synthetic EDGAR-like data (fallback)", ""
try:
    if up is not None:
        raw, meta = load_uploaded(up.getvalue())
        source = f"Uploaded: {up.name}"
    else:
        raw = load_synthetic()
    full, clean_info = ec.prepare(raw)
    if len(full) < 50:
        raise ValueError(f"Only {len(full)} valid rows after cleaning - file looks wrong.")
except Exception as exc:
    st.error(f"Could not process the upload: {exc}. Showing synthetic data instead.")
    raw, meta, source = load_synthetic(), "", "Synthetic EDGAR-like data (fallback)"
    full, clean_info = ec.prepare(raw)

# ------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.success(source)
    st.header("Filters")
    hemis = st.multiselect("Hemisphere", ["Northern", "Southern"], default=["Northern", "Southern"])
    levels = st.multiselect("Emission level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    lon_r = st.slider("Longitude", -180, 180, (-180, 180))
    lat_r = st.slider("Latitude", -90, 90, (-90, 90))
    lo, hi = float(full.Log10_Emission.min()), float(full.Log10_Emission.max())
    e_r = st.slider("log10 emission (kg/yr)", lo, hi, (lo, hi), step=0.1)
    top_n = st.number_input("Top-N hotspot cells", 5, 200, 15, step=5)

view = full[full.Hemisphere.isin(hemis) & full.Emission_Level.isin(levels)
            & full.Longitude.between(*lon_r) & full.Latitude.between(*lat_r)
            & full.Log10_Emission.between(*e_r)]
if len(view) < 5:
    st.warning("Too few cells match the current filters - widen them in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ KPIs
k = st.columns(4)
k[0].metric("Grid cells", f"{len(view):,}", f"of {len(full):,}")
k[1].metric("Total emission", fmt_mass(view.CO2_Emission.sum()))
k[2].metric("Mean / median per cell", f"{view.CO2_Emission.mean() / 1e6:,.0f} / {view.CO2_Emission.median() / 1e6:,.0f} kt")
k[3].metric("Top-1% cells' share", f"{ec.descriptive_stats(view.CO2_Emission)['top1pct_share']:.0%}")

tab_map, tab_dist, tab_hemi, tab_hot, tab_ml = st.tabs(
    ["🗺️ Map", "📊 Distribution & stats", "🌓 Hemisphere test", "🔥 Hotspots", "🤖 Models"])

# ------------------------------------------------------------------ map
with tab_map:
    mp = view.assign(Size=(view.Log10_Emission - view.Log10_Emission.min() + .5))
    fig = px.scatter_geo(mp, lon="Longitude", lat="Latitude", color="Log10_Emission", size="Size",
                         size_max=8, color_continuous_scale="Inferno", projection="natural earth",
                         hover_data={"CO2_Emission": ":,.0f", "Size": False, "Log10_Emission": ":.2f"},
                         labels={"Log10_Emission": "log10 kg/yr"})
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Colour/size = log10(CO₂ kg/yr). Emissions are concentrated in a few regions, not uniform.")

# ------------------------------------------------------------------ distribution
with tab_dist:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(view, x="Log10_Emission", nbins=50, marginal="box",
                                     title="Emission distribution (log10 scale)"), width="stretch")
    with c2:
        S = ec.descriptive_stats(view.CO2_Emission)
        tbl = pd.DataFrame({"Measure": ["Mean", "Median", "Std dev", "Min", "Max", "Skewness",
                                        "Excess kurtosis", "IQR upper fence", "IQR outliers", "Gini",
                                        "Shapiro-Wilk W (≤5,000 sample)", "Shapiro p"],
                            "Value": [f"{S['mean']:,.0f}", f"{S['median']:,.0f}", f"{S['std']:,.0f}",
                                      f"{S['min']:,.0f}", f"{S['max']:,.0f}", f"{S['skewness']:.2f}",
                                      f"{S['excess_kurtosis']:.2f}", f"{S['IQR_upper_fence']:,.0f}",
                                      f"{S['IQR_outliers']:,}", f"{S['gini']:.3f}",
                                      f"{S['shapiro_W']:.4f}", f"{S['shapiro_p']:.2e}"]})
        st.dataframe(tbl, hide_index=True, width="stretch")
    st.caption("Values in kg CO₂/yr. IQR outliers are retained - they are likely genuine high-emission cells.")
    with st.expander("Data cleaning summary / file metadata"):
        st.json(clean_info)
        if meta:
            st.code(meta)

# ------------------------------------------------------------------ hemisphere test
with tab_hemi:
    try:
        r = ec.hemisphere_test(view)
        a, b, c = st.columns(3)
        a.metric("Welch t-test", f"t = {r['t']:.2f}", f"p = {r['p']:.2e}", delta_color="off")
        b.metric("Decision (α = 0.05)", r["decision"])
        c.metric("Mean N / S (kt per cell)", f"{r['mean_north'] / 1e6:,.0f} / {r['mean_south'] / 1e6:,.0f}")
        st.write(f"Robustness - Welch on log10: t = {r['t_log10']:.2f} (p = {r['p_log10']:.2e}); "
                 f"Mann-Whitney U p = {r['p_mannwhitney']:.2e}. "
                 f"Cells: North {r['n_north']:,}, South {r['n_south']:,}.")
        st.plotly_chart(px.box(view, x="Hemisphere", y="CO2_Emission", log_y=True, color="Hemisphere",
                               points=False, title="CO₂ per cell by hemisphere (log axis)"), width="stretch")
        st.info("A significant difference in means is not evidence that hemisphere *causes* emissions - "
                "population, industry and energy mix are omitted variables.")
    except ValueError as exc:
        st.warning(f"Test unavailable for current filters: {exc}")

# ------------------------------------------------------------------ hotspots
with tab_hot:
    tc = ec.top_cells(view, int(top_n))
    l, r_ = st.columns([3, 2])
    with l:
        st.subheader(f"Top {int(top_n)} emission cells")
        st.dataframe(tc.style.format({"CO2_Emission": "{:,.0f}", "Share_of_Total_%": "{:.2f}"}),
                     hide_index=True, width="stretch")
        pct = st.slider("Priority list = top X% of cells", 1, 20, 5)
        pri = view[view.CO2_Emission >= view.CO2_Emission.quantile(1 - pct / 100)] \
            .sort_values("CO2_Emission", ascending=False)
        st.write(f"**{len(pri):,} cells** carry **{pri.CO2_Emission.sum() / view.CO2_Emission.sum():.1%}** "
                 "of emissions in the current view.")
        st.download_button("Download priority hotspots (CSV)",
                           pri[["Longitude", "Latitude", "CO2_Emission", "Hemisphere"]].to_csv(index=False).encode(),
                           file_name="priority_hotspots.csv", mime="text/csv")
    with r_:
        srt = np.sort(view.CO2_Emission.values)[::-1]
        fig = go.Figure(go.Scatter(x=np.arange(1, len(srt) + 1) / len(srt) * 100,
                                   y=np.cumsum(srt) / srt.sum() * 100, mode="lines"))
        fig.update_layout(title="Cumulative emission share", xaxis_title="Top X% of cells",
                          yaxis_title="% of total emissions", height=380)
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ models
with tab_ml:
    st.caption("Models use the full cleaned dataset (Longitude, Latitude → CO₂), not the sidebar filters.")
    res, cv, cv_sp, imp, y_true, y_rf = run_models(full)
    a, b = st.columns(2)
    with a:
        st.subheader("Hold-out (80/20)")
        st.dataframe(res.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.subheader("5-fold cross-validation (R²)")
        st.dataframe(cv.style.format({c: "{:.4f}" for c in cv.columns if c != "Model"}),
                     hide_index=True, width="stretch")
    with b:
        st.subheader("Random vs spatially-blocked validation")
        st.dataframe(cv_sp.style.format({c: "{:.4f}" for c in cv_sp.columns if c != "Validation scheme"}),
                     hide_index=True, width="stretch")
        st.caption("Random Forest, log10 target. A gap between rows = optimism from spatial autocorrelation.")
        st.plotly_chart(px.bar(imp, x="Feature", y="Importance", title="Random Forest feature importance"),
                        width="stretch")
    fig = px.scatter(x=y_true, y=y_rf, log_x=True, log_y=True, opacity=.5,
                     labels={"x": "Actual", "y": "Predicted"}, title="Random Forest: actual vs predicted")
    lim = [max(1, y_true.min()), y_true.max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="red", dash="dash"), name="y = x"))
    st.plotly_chart(fig, width="stretch")
    st.info("Longitude/latitude alone give weak prediction - add population, GDP, industry and energy variables.")
