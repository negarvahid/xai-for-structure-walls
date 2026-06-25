"""
Structural Wall ML Explorer
===========================
Interactive comparison of interpretable and black-box ML models for
structural wall shear-capacity prediction.

Run with:
    source .venv/bin/activate
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import sys
import zipfile
import contextlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from working_version import (
    set_global_seed,
    load_and_clean_data,
    engineer_features,
    engineer_advanced_features,
    add_mean_encoding,
    impute_data,
    compare_imputation_strategies,
    analyze_skewness,
    transform_skewed_features,
    handle_outliers,
    compare_normalization_methods,
    normalize_data,
    compare_augmentation_strategies,
    augment_data,
    select_features,
    train_random_forest,
    train_xgboost,
    adjusted_r2,
    compute_aic_bic,
    bootstrap_confidence_intervals,
)
from nam_with_interactions import train_nam


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structural Wall ML Explorer",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# STYLING — editorial, monochromatic, generous whitespace
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

      html, body, [class*="css"], .stApp {
          font-family: 'Inter', -apple-system, system-ui, sans-serif;
          color: #0f172a;
      }

      /* Headings */
      h1, h2, h3, h4 {
          font-weight: 600;
          letter-spacing: -0.01em;
          color: #0f172a;
      }
      h1 { font-weight: 700; letter-spacing: -0.02em; }

      /* Subtle section captions */
      .stCaption, .st-emotion-cache-1b50dbm p {
          color: #64748b;
      }

      /* Metric cards */
      .metric-card {
          background: #ffffff;
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 18px 20px;
          transition: border-color .2s ease;
      }
      .metric-card:hover { border-color: #cbd5e1; }
      .metric-card.best  { border-color: #0f172a; border-width: 1px; box-shadow: inset 3px 0 0 #0f172a; }
      .metric-label      { font-size: 11px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; color: #64748b; }
      .metric-value      { font-size: 26px; font-weight: 600; color: #0f172a; margin-top: 4px; }
      .metric-sub        { font-size: 12px; color: #94a3b8; margin-top: 2px; }

      /* Tabs */
      div[data-testid="stTabs"] button { font-size: 14px; font-weight: 500; }
      div[data-testid="stTabs"] button[aria-selected="true"] { color: #0f172a; }

      /* Sidebar */
      section[data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
      section[data-testid="stSidebar"] .stButton>button {
          border-radius: 6px; font-weight: 500; letter-spacing: .02em;
      }

      /* Dividers less aggressive */
      hr { border-color: #e5e7eb !important; margin: 1.25rem 0 !important; }

      /* Dataframe polish */
      [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }

      /* Hide the default Streamlit menu/footer for a cleaner feel */
      #MainMenu { visibility: hidden; }
      footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Editorial palette used everywhere for model identity
MODEL_COLORS = {
    "Random Forest":    "#334155",   # slate
    "XGBoost":          "#b45309",   # amber
    "EBM":              "#991b1b",   # crimson
    "NAM":              "#1d4ed8",   # blue
    "NAM+Interactions": "#365314",   # forest
}

PLOT_BG = "#ffffff"
GRID    = "#e5e7eb"


def _style_fig(fig, height=380, legend_y=0.98):
    """Apply consistent minimal styling to every Plotly figure."""
    fig.update_layout(
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(family="Inter, system-ui, sans-serif", color="#0f172a", size=12),
        margin=dict(l=55, r=25, t=50, b=45),
        legend=dict(x=0.02, y=legend_y, bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#cbd5e1")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#cbd5e1")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _safe(s):
    return s.replace("/", "_").replace(" ", "_").replace("+", "_")


def inv_1d(vals, feat_name, scaler, all_feature_names):
    """Inverse-scale a single feature column (used for plotting in original units)."""
    if feat_name not in all_feature_names:
        return np.array(vals, dtype=float)
    col_idx = list(all_feature_names).index(feat_name)
    n_all   = len(all_feature_names)
    vals    = np.array(vals, dtype=float)
    X_tmp   = np.zeros((len(vals), n_all))
    X_tmp[:, col_idx] = vals
    return scaler.inverse_transform(X_tmp)[:, col_idx]


def build_results_zip(res, all_metrics, all_preds, y_test, feat_sel,
                      importances, term_names, sweep_df=None, pipeline_log=""):
    """Build an in-memory ZIP covering every tab's data."""
    scaler_   = res["scaler"]
    all_feat_ = res["all_feature_names"]
    X_test_   = res["X_test_sel"]
    ebm_obj   = res["ebm"]
    ci        = res["ci_results"]
    models_list = list(all_metrics.keys())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — COMPARISON
        # ══════════════════════════════════════════════════════════════════════

        # model_metrics.csv
        rows = []
        for name, m in all_metrics.items():
            rows.append({
                "model": name,
                "train_r2": round(m["train_r2"], 4),
                "test_r2": round(m["test_r2"], 4),
                "train_adj_r2": round(m["train_adj_r2"], 4),
                "test_adj_r2": round(m["test_adj_r2"], 4),
                "train_rmse": round(m["train_rmse"], 4),
                "test_rmse": round(m["test_rmse"], 4),
                "train_mae": round(m["train_mae"], 4),
                "test_mae": round(m["test_mae"], 4),
                "test_aic": round(m["test_aic"], 2),
                "test_bic": round(m["test_bic"], 2),
            })
        zf.writestr("tab1_comparison/model_metrics.csv",
                    pd.DataFrame(rows).to_csv(index=False))

        # predictions_and_residuals.csv
        pred_df = pd.DataFrame({"y_actual": y_test})
        for name, yp in all_preds.items():
            col = _safe(name).lower()
            pred_df[f"pred_{col}"] = yp
            pred_df[f"residual_{col}"] = y_test - yp
        zf.writestr("tab1_comparison/predictions_and_residuals.csv",
                    pred_df.to_csv(index=False))

        # bootstrap_confidence_intervals.csv
        ci_rows = [{
            "model": m,
            "r2_mean": round(ci[m]["r2_mean"], 4),
            "r2_lo_95": round(ci[m]["r2_lo"], 4),
            "r2_hi_95": round(ci[m]["r2_hi"], 4),
            "rmse_mean": round(ci[m]["rmse_mean"], 4),
            "rmse_lo_95": round(ci[m]["rmse_lo"], 4),
            "rmse_hi_95": round(ci[m]["rmse_hi"], 4),
        } for m in ci]
        zf.writestr("tab1_comparison/bootstrap_confidence_intervals.csv",
                    pd.DataFrame(ci_rows).to_csv(index=False))

        # nam_convergence.csv (both variants)
        tl, vl   = res["train_losses"],      res["val_losses"]
        tl_b, vl_b = res["train_losses_base"], res["val_losses_base"]
        conv_df = pd.DataFrame({
            "cycle": range(len(tl)),
            "nam_train_rmse": tl_b, "nam_val_rmse": vl_b,
            "nam_inter_train_rmse": tl, "nam_inter_val_rmse": vl,
        })
        zf.writestr("tab1_comparison/nam_convergence.csv",
                    conv_df.to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — EBM MAIN EFFECTS
        # ══════════════════════════════════════════════════════════════════════

        main_idxs = [i for i, tf in enumerate(ebm_obj.term_features_) if len(tf) == 1]
        ebm_global_ = ebm_obj.explain_global()
        ebm_imps    = ebm_obj.term_importances()

        # ebm_feature_importance.csv
        ebm_imp_rows = [
            {"feature": term_names[i], "importance": round(float(ebm_imps[i]), 6)}
            for i in sorted(main_idxs, key=lambda i: ebm_imps[i], reverse=True)
        ]
        zf.writestr("tab2_ebm_main_effects/ebm_feature_importance.csv",
                    pd.DataFrame(ebm_imp_rows).to_csv(index=False))

        # ebm_shape_data.csv — raw x/y for every main-effect term
        shape_rows = []
        for idx in main_idxs:
            feat = term_names[idx]
            data = ebm_global_.data(idx)
            try:
                x_raw = np.array(data["names"], dtype=float)
                y_sc  = np.array(data["scores"])
                n     = min(len(x_raw), len(y_sc))
                x_orig = inv_1d(x_raw[:n], feat, scaler_, all_feat_)
                for xi, yi in zip(x_orig, y_sc[:n]):
                    shape_rows.append({"feature": feat, "x_original_units": round(float(xi), 6),
                                       "shape_contribution": round(float(yi), 6)})
            except Exception:
                pass
        zf.writestr("tab2_ebm_main_effects/ebm_shape_data.csv",
                    pd.DataFrame(shape_rows).to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — EBM INTERACTIONS
        # ══════════════════════════════════════════════════════════════════════

        inter_idxs = [i for i, tf in enumerate(ebm_obj.term_features_) if len(tf) > 1]
        inter_sort = sorted(inter_idxs, key=lambda i: ebm_imps[i], reverse=True)
        feat_names_in_ = list(ebm_obj.feature_names_in_)

        # ebm_interaction_importance.csv
        inter_imp_rows = [{
            "rank": rank + 1,
            "pair": term_names[i],
            "feature_a": feat_names_in_[ebm_obj.term_features_[i][0]],
            "feature_b": feat_names_in_[ebm_obj.term_features_[i][1]],
            "importance": round(float(ebm_imps[i]), 6),
        } for rank, i in enumerate(inter_sort)]
        zf.writestr("tab3_ebm_interactions/ebm_interaction_importance.csv",
                    pd.DataFrame(inter_imp_rows).to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — NAM SHAPE FUNCTIONS
        # ══════════════════════════════════════════════════════════════════════

        for variant, contrib_key, label in [
            ("nam", "contributions_base", "NAM"),
            ("nam_interactions", "contributions", "NAM+Interactions"),
        ]:
            contribs_ = res[contrib_key]
            mean_abs_ = [float(np.abs(c).mean()) for c in contribs_]
            imp_rows  = [
                {"feature": feat_sel[i], "mean_abs_contribution": round(mean_abs_[i], 6)}
                for i in np.argsort(mean_abs_)[::-1]
            ]
            zf.writestr(f"tab4_nam_shapes/{variant}_feature_importance.csv",
                        pd.DataFrame(imp_rows).to_csv(index=False))

            shape_rows_ = []
            for i, feat in enumerate(feat_sel):
                try:
                    x_sc   = X_test_[:, i]
                    x_orig = inv_1d(x_sc, feat, scaler_, all_feat_)
                    y_c    = contribs_[i].ravel()
                    sidx   = np.argsort(x_orig)
                    for xi, yi in zip(x_orig[sidx], y_c[sidx]):
                        shape_rows_.append({"feature": feat,
                                            "x_original_units": round(float(xi), 6),
                                            "contribution": round(float(yi), 6)})
                except Exception:
                    pass
            zf.writestr(f"tab4_nam_shapes/{variant}_shape_data.csv",
                        pd.DataFrame(shape_rows_).to_csv(index=False))

        # NAM interaction contributions per pair
        inter_rows_ = []
        for label_, contrib_ in zip(res["pair_labels"], res["inter_contribs"]):
            ni, nj = label_.split(" x ")
            if ni not in feat_sel or nj not in feat_sel:
                continue
            ci_ = list(feat_sel).index(ni)
            cj_ = list(feat_sel).index(nj)
            xi  = inv_1d(X_test_[:, ci_], ni, scaler_, all_feat_)
            xj  = inv_1d(X_test_[:, cj_], nj, scaler_, all_feat_)
            c_  = contrib_.ravel()
            for a, b, cv in zip(xi, xj, c_):
                inter_rows_.append({"pair": label_,
                                    f"{ni}_original": round(float(a), 6),
                                    f"{nj}_original": round(float(b), 6),
                                    "contribution": round(float(cv), 6)})
        if inter_rows_:
            zf.writestr("tab4_nam_shapes/nam_interaction_contributions.csv",
                        pd.DataFrame(inter_rows_).to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 — DIAGNOSTICS
        # ══════════════════════════════════════════════════════════════════════

        resid_rows = []
        for name, yp in all_preds.items():
            r = y_test - yp
            sw_stat, sw_p = stats.shapiro(r) if 3 <= len(r) <= 5000 else (np.nan, np.nan)
            within_1sd = float(np.mean(np.abs((r - r.mean()) / (r.std(ddof=1) + 1e-12)) <= 1) * 100)
            within_2sd = float(np.mean(np.abs((r - r.mean()) / (r.std(ddof=1) + 1e-12)) <= 2) * 100)
            resid_rows.append({
                "model": name,
                "mean_residual": round(float(r.mean()), 6),
                "std_residual": round(float(r.std(ddof=1)), 6),
                "skewness": round(float(stats.skew(r)), 4),
                "excess_kurtosis": round(float(stats.kurtosis(r)), 4),
                "shapiro_wilk_stat": round(float(sw_stat), 4) if not np.isnan(sw_stat) else "",
                "shapiro_wilk_p": round(float(sw_p), 4) if not np.isnan(sw_p) else "",
                "within_1sd_pct": round(within_1sd, 1),
                "within_2sd_pct": round(within_2sd, 1),
            })
        zf.writestr("tab5_diagnostics/residual_diagnostics_summary.csv",
                    pd.DataFrame(resid_rows).to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6 — EBM INTERACTION SWEEP
        # ══════════════════════════════════════════════════════════════════════

        if sweep_df is not None:
            zf.writestr("tab6_ebm_sweep/ebm_interaction_sweep.csv",
                        sweep_df.to_csv(index=False))

        # ══════════════════════════════════════════════════════════════════════
        # TAB 7 — PIPELINE LOG
        # ══════════════════════════════════════════════════════════════════════

        if pipeline_log:
            zf.writestr("tab7_log/pipeline_log.txt", pipeline_log)

        # ══════════════════════════════════════════════════════════════════════
        # PNGs (all tabs)
        # ══════════════════════════════════════════════════════════════════════
        try:
            import plotly.io as pio
            colors_ci = [MODEL_COLORS[m] for m in models_list]

            # ── Tab 1: predicted vs actual ────────────────────────────────────
            for name, yp in all_preds.items():
                color = MODEL_COLORS[name]
                mn = float(min(y_test.min(), yp.min()))
                mx = float(max(y_test.max(), yp.max()))
                r2 = r2_score(y_test, yp)
                rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=yp, mode="markers",
                    marker=dict(color=color, size=7, opacity=0.75,
                                line=dict(color="white", width=0.5))))
                fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="#0f172a", dash="dot", width=1)))
                fig.update_layout(title=f"{name} · R² = {r2:.4f} · RMSE = {rmse:.4f}",
                    xaxis_title="Actual (V/√fc)", yaxis_title="Predicted (V/√fc)")
                _style_fig(fig, height=500)
                zf.writestr(f"tab1_comparison/pred_vs_actual_{_safe(name).lower()}.png",
                            pio.to_image(fig, format="png", scale=2))

            # ── Tab 1: model comparison bars ──────────────────────────────────
            for metric_key, metric_label in [
                ("test_r2", "Test R²"), ("test_adj_r2", "Test Adj R²"),
                ("test_rmse", "Test RMSE"), ("test_mae", "Test MAE"),
            ]:
                vals = [all_metrics[m][metric_key] for m in models_list]
                fig = go.Figure(go.Bar(x=models_list, y=vals,
                    marker_color=[MODEL_COLORS[m] for m in models_list],
                    text=[f"{v:.4f}" for v in vals], textposition="outside"))
                fig.update_layout(title=f"Model comparison — {metric_label}",
                                  yaxis_title=metric_label)
                _style_fig(fig, height=400)
                zf.writestr(f"tab1_comparison/model_comparison_{_safe(metric_label).lower()}.png",
                            pio.to_image(fig, format="png", scale=2))

            # ── Tab 1: bootstrap CI — R² and RMSE ────────────────────────────
            for ci_metric, ci_label in [("r2", "R²"), ("rmse", "RMSE")]:
                fig = go.Figure(go.Bar(
                    x=models_list,
                    y=[ci[m][f"{ci_metric}_mean"] for m in models_list],
                    error_y=dict(type="data", symmetric=False,
                        array=[ci[m][f"{ci_metric}_hi"] - ci[m][f"{ci_metric}_mean"] for m in models_list],
                        arrayminus=[ci[m][f"{ci_metric}_mean"] - ci[m][f"{ci_metric}_lo"] for m in models_list],
                        color="#0f172a", thickness=1.2),
                    marker_color=colors_ci,
                    text=[f'{ci[m][f"{ci_metric}_mean"]:.4f}' for m in models_list],
                    textposition="outside",
                ))
                fig.update_layout(title=f"{ci_label} ± 95% Bootstrap CI", yaxis_title=ci_label)
                _style_fig(fig, height=400)
                zf.writestr(f"tab1_comparison/bootstrap_ci_{ci_metric}.png",
                            pio.to_image(fig, format="png", scale=2))

            # ── Tab 1: NAM convergence ────────────────────────────────────────
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=tl_b, mode="lines+markers", name="NAM train",
                line=dict(color=MODEL_COLORS["NAM"], width=1.8, dash="dash")))
            fig.add_trace(go.Scatter(y=vl_b, mode="lines+markers", name="NAM val",
                line=dict(color=MODEL_COLORS["NAM"], width=1.8, dash="dot")))
            fig.add_trace(go.Scatter(y=tl, mode="lines+markers", name="NAM+Inter train",
                line=dict(color=MODEL_COLORS["NAM+Interactions"], width=2)))
            fig.add_trace(go.Scatter(y=vl, mode="lines+markers", name="NAM+Inter val",
                line=dict(color=MODEL_COLORS["NAM+Interactions"], width=2, dash="dot")))
            fig.update_layout(xaxis_title="Backfitting cycle", yaxis_title="RMSE (raw sum)")
            _style_fig(fig, height=360, legend_y=0.97)
            zf.writestr("tab1_comparison/nam_convergence.png",
                        pio.to_image(fig, format="png", scale=2))

            # ── Tab 2: EBM main-effect shapes ─────────────────────────────────
            main_sort_ = sorted(main_idxs, key=lambda i: ebm_imps[i], reverse=True)
            for idx in main_sort_:
                feat = term_names[idx]
                data = ebm_global_.data(idx)
                try:
                    x_raw = np.array(data["names"], dtype=float)
                    y_sc  = np.array(data["scores"])
                    n     = min(len(x_raw), len(y_sc))
                    x_orig = inv_1d(x_raw[:n], feat, scaler_, all_feat_)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_orig, y=y_sc[:n], mode="lines",
                        line=dict(color=MODEL_COLORS["EBM"], width=2.2),
                        fill="tozeroy", fillcolor="rgba(153,27,27,0.10)"))
                    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
                    fig.update_layout(title=f"EBM shape: {feat}  (importance={ebm_imps[idx]:.4f})",
                        xaxis_title=feat, yaxis_title="Shape contribution")
                    _style_fig(fig, height=350)
                    zf.writestr(f"tab2_ebm_main_effects/ebm_shape_{_safe(feat)}.png",
                                pio.to_image(fig, format="png", scale=2))
                except Exception:
                    pass

            # ── Tab 3: EBM interaction heatmaps ───────────────────────────────
            for idx in inter_sort:
                term   = term_names[idx]
                fi     = ebm_obj.term_features_[idx]
                ni, nj = feat_names_in_[fi[0]], feat_names_in_[fi[1]]
                data   = ebm_global_.data(idx)
                try:
                    _raw = data.get("scores") or data.get("values") or []
                    scores = np.array(_raw)
                    if scores.ndim != 2:
                        continue
                    if "names" in data and len(data["names"]) == 2:
                        bins_i = np.array(data["names"][0], dtype=float)
                        bins_j = np.array(data["names"][1], dtype=float)
                    elif "left_names" in data:
                        bins_i = np.array(data["left_names"], dtype=float)
                        bins_j = np.array(data["right_names"], dtype=float)
                    else:
                        bins_i = np.arange(scores.shape[0], dtype=float)
                        bins_j = np.arange(scores.shape[1], dtype=float)
                    xi_o = inv_1d(bins_i, ni, scaler_, all_feat_)
                    xj_o = inv_1d(bins_j, nj, scaler_, all_feat_)
                    vmax = float(np.abs(scores).max()) or 1.0
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(x=xi_o, y=xj_o, z=scores.T,
                        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                        colorbar=dict(title="Contribution", len=0.85, thickness=14)))
                    fig.update_layout(xaxis_title=f"{ni} (original units)",
                                      yaxis_title=f"{nj} (original units)")
                    _style_fig(fig, height=460)
                    zf.writestr(f"tab3_ebm_interactions/ebm_interaction_{_safe(term)}.png",
                                pio.to_image(fig, format="png", scale=2))
                except Exception:
                    pass

            # ── Tab 4: NAM shape plots (both variants) ────────────────────────
            for variant, contrib_key, color_, fill_ in [
                ("nam",          "contributions_base", MODEL_COLORS["NAM"],              "rgba(29,78,216,0.10)"),
                ("nam_inter",    "contributions",      MODEL_COLORS["NAM+Interactions"], "rgba(54,83,20,0.10)"),
            ]:
                contribs_ = res[contrib_key]
                mean_abs_ = [float(np.abs(c).mean()) for c in contribs_]
                for i, feat in enumerate(feat_sel):
                    try:
                        x_sc   = X_test_[:, i]
                        x_orig = inv_1d(x_sc, feat, scaler_, all_feat_)
                        y_c    = contribs_[i].ravel()
                        sidx   = np.argsort(x_orig)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_orig[sidx], y=y_c[sidx], mode="lines",
                            line=dict(color=color_, width=2.2),
                            fill="tozeroy", fillcolor=fill_))
                        fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
                        fig.update_layout(
                            title=f"{variant} shape: {feat}  (mean|contrib|={mean_abs_[i]:.4f})",
                            xaxis_title=feat, yaxis_title="Contribution")
                        _style_fig(fig, height=350)
                        zf.writestr(f"tab4_nam_shapes/{variant}_shape_{_safe(feat)}.png",
                                    pio.to_image(fig, format="png", scale=2))
                    except Exception:
                        pass

            # ── Tab 4: NAM interaction scatter plots ──────────────────────────
            for label_, contrib_ in zip(res["pair_labels"], res["inter_contribs"]):
                ni, nj = label_.split(" x ")
                if ni not in feat_sel or nj not in feat_sel:
                    continue
                ci__ = list(feat_sel).index(ni)
                cj__ = list(feat_sel).index(nj)
                xi   = inv_1d(X_test_[:, ci__], ni, scaler_, all_feat_)
                xj   = inv_1d(X_test_[:, cj__], nj, scaler_, all_feat_)
                c_   = contrib_.ravel()
                vmax = float(np.abs(c_).max()) or 1.0
                fig  = go.Figure(go.Scatter(x=xi, y=xj, mode="markers",
                    marker=dict(color=c_, colorscale="RdBu_r", cmin=-vmax, cmax=vmax,
                                colorbar=dict(title="Contribution", thickness=14),
                                size=8, opacity=0.85, line=dict(color="white", width=0.4)),
                    hovertemplate=f"{ni}=%{{x:.3f}}<br>{nj}=%{{y:.3f}}<br>contrib=%{{marker.color:.4f}}<extra></extra>"))
                fig.update_layout(xaxis_title=f"{ni} (original units)",
                                  yaxis_title=f"{nj} (original units)")
                _style_fig(fig, height=440)
                zf.writestr(f"tab4_nam_shapes/nam_interaction_{_safe(label_)}.png",
                            pio.to_image(fig, format="png", scale=2))

            # ── Tab 5: residual diagnostics — one set per model ───────────────
            for name, yp in all_preds.items():
                color = MODEL_COLORS[name]
                r     = y_test - yp
                mean_r, std_r = float(r.mean()), float(r.std(ddof=1))

                # residuals vs predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yp, y=r, mode="markers",
                    marker=dict(color=color, size=7, opacity=0.75,
                                line=dict(color="white", width=0.5))))
                fig.add_hline(y=0, line_dash="dot", line_color="#0f172a")
                fig.update_layout(title=f"{name} · Residuals vs Predicted",
                    xaxis_title="Predicted (V/√fc)", yaxis_title="Residual")
                _style_fig(fig, height=420)
                zf.writestr(f"tab5_diagnostics/residuals_vs_predicted_{_safe(name).lower()}.png",
                            pio.to_image(fig, format="png", scale=2))

                # residual histogram + normal fit
                xs  = np.linspace(r.min(), r.max(), 200)
                pdf = stats.norm.pdf(xs, loc=mean_r, scale=std_r)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=r, nbinsx=25, histnorm="probability density",
                    marker=dict(color=color, line=dict(color="white", width=0.8)), opacity=0.8))
                fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines",
                    line=dict(color="#0f172a", width=1.8, dash="dot")))
                fig.update_layout(title=f"{name} · Residual distribution",
                    xaxis_title="Residual", yaxis_title="Density")
                _style_fig(fig, height=400)
                zf.writestr(f"tab5_diagnostics/residual_histogram_{_safe(name).lower()}.png",
                            pio.to_image(fig, format="png", scale=2))

                # Q-Q plot
                qq_x, qq_y = stats.probplot(r, dist="norm", fit=False)
                slope_, intercept_, *_ = stats.linregress(qq_x, qq_y)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers",
                    marker=dict(color=color, size=7, opacity=0.8,
                                line=dict(color="white", width=0.5))))
                lx = np.array([qq_x.min(), qq_x.max()])
                fig.add_trace(go.Scatter(x=lx, y=intercept_ + slope_ * lx, mode="lines",
                    line=dict(color="#0f172a", width=1.4, dash="dot")))
                fig.update_layout(title=f"{name} · Q–Q plot",
                    xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
                _style_fig(fig, height=380)
                zf.writestr(f"tab5_diagnostics/residual_qq_{_safe(name).lower()}.png",
                            pio.to_image(fig, format="png", scale=2))

            # ── Tab 6: EBM sweep plots ────────────────────────────────────────
            if sweep_df is not None:
                xs_ = sweep_df["n_interactions"].tolist()
                for col, label, higher in [
                    ("test_r2",     "Test R²",      True),
                    ("test_adj_r2", "Test Adj R²",  True),
                    ("test_rmse",   "Test RMSE",    False),
                    ("test_mae",    "Test MAE",     False),
                    ("r2_gap",      "R² gap",       False),
                    ("rmse_gap",    "RMSE gap",     False),
                    ("ci_width_r2", "CI width R²",  False),
                    ("bias",        "|Mean residual|", False),
                    ("test_aic",    "AIC",          False),
                    ("test_bic",    "BIC",          False),
                ]:
                    best_i = sweep_df[col].idxmax() if higher else sweep_df[col].idxmin()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=xs_, y=sweep_df[col].tolist(),
                        mode="lines+markers",
                        line=dict(color=MODEL_COLORS["EBM"], width=2.2),
                        marker=dict(size=8)))
                    fig.add_trace(go.Scatter(
                        x=[sweep_df.loc[best_i, "n_interactions"]],
                        y=[sweep_df.loc[best_i, col]], mode="markers",
                        marker=dict(color="#0f172a", size=12, symbol="star"),
                        name=f"Best (n={sweep_df.loc[best_i,'n_interactions']})"))
                    fig.update_layout(title=f"EBM sweep — {label}",
                        xaxis_title="n_interactions", yaxis_title=label,
                        xaxis=dict(tickmode="linear", tick0=0, dtick=1))
                    _style_fig(fig, height=320)
                    zf.writestr(f"tab6_ebm_sweep/sweep_{_safe(label).lower()}.png",
                                pio.to_image(fig, format="png", scale=2))

        except Exception as e:
            zf.writestr("plots_error.txt",
                        f"PNG export failed: {e}\nInstall kaleido: pip install kaleido")

    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_DATA = "Database_Negar.xlsx"

with st.sidebar:
    st.markdown(
        "<h2 style='margin:0;font-weight:700;letter-spacing:-0.02em;'>Structural Wall ML</h2>"
        "<p style='margin:0 0 1rem 0;color:#64748b;font-size:13px;'>"
        "EBM &middot; NAM &middot; Random Forest &middot; XGBoost</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Configuration**")
    seed           = st.number_input("Random seed", value=42, step=1)
    test_size_pct  = st.slider("Test split (%)", 10, 40, 20, 5,
                               help="Percentage of data held out for testing")
    test_size      = test_size_pct / 100
    n_interactions = st.slider("EBM interaction pairs", 1, 10, 5,
                               help="How many pairwise interactions EBM is allowed to learn")

    st.divider()
    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    if "last_run" in st.session_state:
        cfg = st.session_state["last_run"]
        st.caption(
            f"Last run · seed {cfg['seed']} · test {cfg['test_size_pct']}% · "
            f"{cfg['n_interactions']} interactions"
        )

    if "results" in st.session_state:
        st.divider()
        st.markdown("**Export results**")
        if st.button("Build ZIP", use_container_width=True,
                     help="Packages all CSVs and PNGs into a single ZIP file"):
            with st.spinner("Building ZIP…"):
                try:
                    _res = st.session_state["results"]
                    zip_bytes = build_results_zip(
                        _res,
                        {
                            "Random Forest":    _res["rf_metrics"],
                            "XGBoost":          _res["xgb_metrics"],
                            "EBM":              _res["ebm_metrics"],
                            "NAM":              _res["nam_base_metrics"],
                            "NAM+Interactions": _res["nam_metrics"],
                        },
                        {
                            "Random Forest":    _res["rf_pred"],
                            "XGBoost":          _res["xgb_pred"],
                            "EBM":              _res["ebm_pred"],
                            "NAM":              _res["nam_base_pred"],
                            "NAM+Interactions": _res["nam_pred"],
                        },
                        _res["y_test"],
                        _res["feature_names_sel"],
                        _res["ebm"].term_importances(),
                        [
                            " × ".join(str(n) for n in tn) if isinstance(tn, (list, tuple)) else str(tn)
                            for tn in _res["ebm"].term_names_
                        ],
                        sweep_df=st.session_state.get("sweep_df"),
                        pipeline_log=st.session_state.get("pipeline_log", ""),
                    )
                    st.session_state["zip_bytes"] = zip_bytes
                except Exception as e:
                    st.error(f"ZIP build failed: {e}")

        if "zip_bytes" in st.session_state:
            st.download_button(
                label="Download results.zip",
                data=st.session_state["zip_bytes"],
                file_name="structural_wall_ml_results.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running pipeline… (2–4 min)")
def run_pipeline(seed, test_size, n_interactions, _data_source):
    """Cached wrapper that also captures stdout for the log tab."""
    log = io.StringIO()
    tee = type("Tee", (), {
        "write": lambda s, m: (log.write(m), sys.__stdout__.write(m)),
        "flush": lambda s: (sys.__stdout__.flush(),),
    })()
    with contextlib.redirect_stdout(tee):
        result = _run_pipeline_inner(seed, test_size, n_interactions, _data_source)
    result["pipeline_log"] = log.getvalue()
    return result


def _run_pipeline_inner(seed, test_size, n_interactions, _data_source):
    set_global_seed(seed)

    df = load_and_clean_data(_data_source)
    df = engineer_features(df)
    df = engineer_advanced_features(df)

    target_col = "V_sqrt_fc"
    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=seed
    )
    df_train = df.iloc[idx_tr].reset_index(drop=True)
    df_test  = df.iloc[idx_te].reset_index(drop=True)

    df_train, df_test = add_mean_encoding(df_train, df_test)

    best_impute, _ = compare_imputation_strategies(df_train)
    df_train, df_test, _ = impute_data(df_train, df_test, strategy=best_impute)

    _, skewed_cols = analyze_skewness(df_train)
    skewed_cols = [c for c in skewed_cols if c != target_col]
    df_train, df_test, _ = transform_skewed_features(
        df_train, df_test, method="yeojohnson", skewed_cols=skewed_cols
    )
    df_train, df_test, _ = handle_outliers(df_train, df_test, method="iqr")

    feature_cols = [
        c for c in df_train.select_dtypes(include=[np.number]).columns
        if c != target_col and c != "Max_Shear_Force_kN"
    ]
    X_train_raw = df_train[feature_cols].values
    X_test_raw  = df_test[feature_cols].values
    y_train     = df_train[target_col].values
    y_test      = df_test[target_col].values

    best_norm, _ = compare_normalization_methods(X_train_raw, y_train, random_state=seed)
    X_train_scaled, scaler = normalize_data(X_train_raw, method=best_norm)
    X_test_scaled = scaler.transform(X_test_raw)

    best_aug, _ = compare_augmentation_strategies(X_train_scaled, y_train)
    if best_aug != "none":
        X_train_aug, y_train_aug = augment_data(
            X_train_scaled, y_train, method=best_aug, n_augment=1, noise_level=0.03
        )
    else:
        X_train_aug, y_train_aug = X_train_scaled, y_train

    rf_model,  rf_metrics,  rf_pred  = train_random_forest(
        X_train_aug, X_test_scaled, y_train_aug, y_test, random_state=seed)
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(
        X_train_aug, X_test_scaled, y_train_aug, y_test, random_state=seed)

    X_train_sel, X_test_sel, feature_names_sel = select_features(
        X_train_aug, X_test_scaled, y_train_aug, feature_cols,
        rf_model, xgb_model, random_state=seed,
    )

    # EBM
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm = ExplainableBoostingRegressor(
        feature_names=feature_names_sel,
        max_bins=256, max_interaction_bins=32,
        interactions=n_interactions,
        learning_rate=0.01, min_samples_leaf=5, max_leaves=2,
        n_jobs=-1, random_state=seed,
    )
    ebm.fit(X_train_sel, y_train_aug)
    ebm_train_pred = ebm.predict(X_train_sel)
    ebm_pred       = ebm.predict(X_test_sel)

    # Complexity = main-effect features + realized pairwise interaction terms.
    n_inter_terms = sum(1 for tf in ebm.term_features_ if len(tf) > 1)
    p_ebm = len(feature_names_sel) + n_inter_terms   # predictors (for adjusted R²)
    k_ebm = p_ebm + 1                                # params incl. intercept (for AIC/BIC)
    ebm_r2_train = r2_score(y_train_aug, ebm_train_pred)
    ebm_r2_test  = r2_score(y_test, ebm_pred)
    ebm_metrics  = {
        "train_r2":     ebm_r2_train,
        "test_r2":      ebm_r2_test,
        "train_adj_r2": adjusted_r2(ebm_r2_train, len(y_train_aug), p_ebm),
        "test_adj_r2":  adjusted_r2(ebm_r2_test,  len(y_test),      p_ebm),
        "train_rmse":   float(np.sqrt(mean_squared_error(y_train_aug, ebm_train_pred))),
        "test_rmse":    float(np.sqrt(mean_squared_error(y_test, ebm_pred))),
        "train_mae":    float(mean_absolute_error(y_train_aug, ebm_train_pred)),
        "test_mae":     float(mean_absolute_error(y_test, ebm_pred)),
        "train_aic":    compute_aic_bic(y_train_aug, ebm_train_pred, k_ebm)[0],
        "test_aic":     compute_aic_bic(y_test, ebm_pred, k_ebm)[0],
        "train_bic":    compute_aic_bic(y_train_aug, ebm_train_pred, k_ebm)[1],
        "test_bic":     compute_aic_bic(y_test, ebm_pred, k_ebm)[1],
    }

    # Detect EBM interaction pairs — version-safe using term_features_
    feat_names_in = list(ebm.feature_names_in_)
    ebm_interaction_pairs = [
        (feat_names_in[tf[0]], feat_names_in[tf[1]])
        for tf in ebm.term_features_ if len(tf) > 1
    ]

    # Pure NAM (no interactions — ablation baseline)
    (nam_base_model, nam_base_metrics, nam_base_pred, contributions_base,
     (train_losses_base, val_losses_base), _, _) = train_nam(
        X_train_sel, X_test_sel, y_train_aug, y_test, list(feature_names_sel),
        hidden_units=[64, 32], epochs=300, random_state=seed, n_cycles=3,
        interaction_pairs=[],
    )

    # NAM with EBM-guided interactions
    (nam_model, nam_metrics, nam_pred, contributions,
     (train_losses, val_losses), inter_contribs, pair_labels) = train_nam(
        X_train_sel, X_test_sel, y_train_aug, y_test, list(feature_names_sel),
        hidden_units=[64, 32], epochs=300, random_state=seed, n_cycles=3,
        interaction_pairs=ebm_interaction_pairs,
    )

    # Bootstrap confidence intervals (1 000 resamples, 95 % CI)
    ci_results = bootstrap_confidence_intervals(
        y_test,
        {
            "Random Forest":    rf_pred,
            "XGBoost":          xgb_pred,
            "EBM":              ebm_pred,
            "NAM":              nam_base_pred,
            "NAM+Interactions": nam_pred,
        },
        n_bootstrap=1000,
        ci=95.0,
        random_state=seed,
    )

    return dict(
        ebm=ebm,
        ebm_pred=ebm_pred, rf_pred=rf_pred, xgb_pred=xgb_pred,
        nam_base_pred=nam_base_pred, nam_pred=nam_pred,
        y_test=y_test,
        rf_metrics=rf_metrics, xgb_metrics=xgb_metrics,
        ebm_metrics=ebm_metrics,
        nam_base_metrics=nam_base_metrics, nam_metrics=nam_metrics,
        contributions_base=contributions_base,
        contributions=contributions,
        inter_contribs=inter_contribs, pair_labels=pair_labels,
        train_losses_base=train_losses_base, val_losses_base=val_losses_base,
        train_losses=train_losses, val_losses=val_losses,
        X_test_sel=X_test_sel, X_train_sel=X_train_sel,
        y_train_aug=y_train_aug,
        feature_names_sel=list(feature_names_sel),
        scaler=scaler, all_feature_names=feature_cols,
        ebm_interaction_pairs=ebm_interaction_pairs,
        ci_results=ci_results,
    )


def run_ebm_interaction_sweep(X_train_sel, X_test_sel, y_train_aug, y_test,
                               feature_names_sel, seed, max_interactions=5):
    """
    Train EBM for n_interactions in [0 .. max_interactions] and collect metrics.
    Reuses already-prepared data — no re-preprocessing, typically <1 min total.
    """
    from interpret.glassbox import ExplainableBoostingRegressor

    records = []
    for n in range(0, max_interactions + 1):
        ebm = ExplainableBoostingRegressor(
            feature_names=feature_names_sel,
            max_bins=256, max_interaction_bins=32,
            interactions=n,
            learning_rate=0.01, min_samples_leaf=5, max_leaves=2,
            n_jobs=-1, random_state=seed,
        )
        ebm.fit(X_train_sel, y_train_aug)
        train_pred = ebm.predict(X_train_sel)
        test_pred  = ebm.predict(X_test_sel)

        # Complexity scales with the interaction terms the EBM actually realized,
        # so AIC/BIC/Adj R² penalize larger n_interactions.
        n_inter_terms = sum(1 for tf in ebm.term_features_ if len(tf) > 1)
        p = len(feature_names_sel) + n_inter_terms   # predictors (for adjusted R²)
        k = p + 1                                    # params incl. intercept (for AIC/BIC)
        n_tr, n_te = len(y_train_aug), len(y_test)

        r2_tr  = r2_score(y_train_aug, train_pred)
        r2_te  = r2_score(y_test,      test_pred)
        rmse_tr = float(np.sqrt(mean_squared_error(y_train_aug, train_pred)))
        rmse_te = float(np.sqrt(mean_squared_error(y_test,      test_pred)))
        mae_te  = float(mean_absolute_error(y_test, test_pred))
        aic_te  = compute_aic_bic(y_test, test_pred, k)[0]
        bic_te  = compute_aic_bic(y_test, test_pred, k)[1]
        residuals = y_test - test_pred

        # Bootstrap CI width (proxy for prediction stability)
        ci = bootstrap_confidence_intervals(
            y_test, {"EBM": test_pred}, n_bootstrap=500, ci=95.0, random_state=seed
        )["EBM"]
        ci_width_r2   = float(ci["r2_hi"]   - ci["r2_lo"])
        ci_width_rmse = float(ci["rmse_hi"] - ci["rmse_lo"])

        sw_stat, sw_p = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 else (np.nan, np.nan)

        records.append({
            "n_interactions":   n,
            "train_r2":         round(r2_tr,  4),
            "test_r2":          round(r2_te,  4),
            "train_adj_r2":     round(adjusted_r2(r2_tr, n_tr, p), 4),
            "test_adj_r2":      round(adjusted_r2(r2_te, n_te, p), 4),
            "r2_gap":           round(r2_tr - r2_te, 4),          # overfitting proxy
            "train_rmse":       round(rmse_tr, 4),
            "test_rmse":        round(rmse_te, 4),
            "rmse_gap":         round(rmse_te - rmse_tr, 4),
            "test_mae":         round(mae_te,  4),
            "test_aic":         round(aic_te,  2),
            "test_bic":         round(bic_te,  2),
            "ci_width_r2":      round(ci_width_r2,   4),   # narrower = more stable
            "ci_width_rmse":    round(ci_width_rmse,  4),
            "bias":             round(float(np.abs(residuals.mean())), 6),
            "residual_std":     round(float(residuals.std(ddof=1)), 6),
            "shapiro_p":        round(float(sw_p), 4) if not np.isnan(sw_p) else None,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running pipeline…"):
        st.cache_resource.clear()
        result = run_pipeline(seed, test_size, n_interactions, LOCAL_DATA)
        st.session_state["results"]      = result
        st.session_state["pipeline_log"] = result.get("pipeline_log", "")
        st.session_state["last_run"]     = dict(
            seed=seed, test_size_pct=test_size_pct, n_interactions=n_interactions
        )
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.markdown("# Structural Wall ML Explorer")
    st.markdown(
        "<p style='font-size:16px; color:#475569; max-width:720px;'>"
        "Train and compare four machine-learning models on your structural "
        "wall dataset, then interrogate their shape functions, interactions, "
        "and residual behaviour interactively."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    c1, c2, c3, c4, c5 = st.columns(5)
    for c, (name, desc) in zip(
        [c1, c2, c3, c4, c5],
        [
            ("Random Forest", "Nonlinear ensemble baseline"),
            ("XGBoost",       "Gradient-boosted trees"),
            ("EBM",           "Glass-box additive model"),
            ("NAM",           "Neural additive — no interactions"),
            ("NAM+Inter",     "Neural additive + EBM-guided pairs"),
        ],
    ):
        c.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Model</div>"
            f"<div class='metric-value' style='font-size:18px;'>{name}</div>"
            f"<div class='metric-sub'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")
    st.info("Click **Run pipeline** in the sidebar to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "rf_metrics", "xgb_metrics", "ebm_metrics", "nam_base_metrics", "nam_metrics",
    "rf_pred", "xgb_pred", "ebm_pred", "nam_base_pred", "nam_pred", "ci_results",
    "X_train_sel", "y_train_aug",
}
if not REQUIRED_KEYS.issubset(st.session_state["results"].keys()):
    del st.session_state["results"]
    st.cache_resource.clear()
    st.warning("Cached results are out of date. Please re-run the pipeline.")
    st.stop()

res          = st.session_state["results"]
ebm          = res["ebm"]
scaler       = res["scaler"]
all_feat     = res["all_feature_names"]
feat_sel     = res["feature_names_sel"]
ebm_global   = ebm.explain_global()
importances  = ebm.term_importances()
y_test       = res["y_test"]

raw_term_names   = ebm.term_names_
term_features_   = ebm.term_features_
feature_names_in = list(ebm.feature_names_in_)


def _term_label(idx):
    tn = raw_term_names[idx]
    return " × ".join(str(n) for n in tn) if isinstance(tn, (list, tuple)) else str(tn)


def _is_interaction(idx):
    return len(term_features_[idx]) > 1


def _term_feature_names(idx):
    fi = term_features_[idx]
    return feature_names_in[fi[0]], feature_names_in[fi[1]]


term_names = [_term_label(i) for i in range(len(raw_term_names))]

ALL_METRICS = {
    "Random Forest":    res["rf_metrics"],
    "XGBoost":          res["xgb_metrics"],
    "EBM":              res["ebm_metrics"],
    "NAM":              res["nam_base_metrics"],
    "NAM+Interactions": res["nam_metrics"],
}
ALL_PREDS = {
    "Random Forest":    res["rf_pred"],
    "XGBoost":          res["xgb_pred"],
    "EBM":              res["ebm_pred"],
    "NAM":              res["nam_base_pred"],
    "NAM+Interactions": res["nam_pred"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HEADER + TABS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Structural Wall ML Explorer")
st.markdown(
    "<p style='color:#64748b; font-size:14px; margin-top:-12px;'>"
    "Model comparison, interpretability, and residual diagnostics."
    "</p>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Comparison",
    "EBM — Main effects",
    "EBM — Interactions",
    "NAM — Shape functions",
    "Diagnostics",
    "EBM Interaction Sweep",
    "Log",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Model comparison")
    st.caption("All values computed on the held-out test set unless marked Train. "
               "Highlighted value = best per column.")

    # Headline cards
    best_r2 = max(m["test_r2"] for m in ALL_METRICS.values())
    cols = st.columns(5)
    for col, (name, m) in zip(cols, ALL_METRICS.items()):
        is_best = abs(m["test_r2"] - best_r2) < 1e-6
        cls = "metric-card best" if is_best else "metric-card"
        col.markdown(
            f"<div class='{cls}'>"
            f"<div class='metric-label'>{name}</div>"
            f"<div class='metric-value'>R² {m['test_r2']:.4f}</div>"
            f"<div class='metric-sub'>Adj R² {m['test_adj_r2']:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Full metrics table
    rows = []
    for name, m in ALL_METRICS.items():
        rows.append({
            "Model":        name,
            "Train R²":     round(m["train_r2"],    4),
            "Test R²":      round(m["test_r2"],     4),
            "Train Adj R²": round(m["train_adj_r2"],4),
            "Test Adj R²":  round(m["test_adj_r2"], 4),
            "Train RMSE":   round(m["train_rmse"],  4),
            "Test RMSE":    round(m["test_rmse"],   4),
            "Train MAE":    round(m["train_mae"],   4),
            "Test MAE":     round(m["test_mae"],    4),
            "Test AIC":     round(m["test_aic"],    1),
            "Test BIC":     round(m["test_bic"],    1),
        })
    df_metrics = pd.DataFrame(rows)

    higher_better = {"Train R²", "Test R²", "Train Adj R²", "Test Adj R²"}
    lower_better  = {"Train RMSE", "Test RMSE", "Train MAE", "Test MAE",
                     "Test AIC", "Test BIC"}

    def _highlight(df):
        styled = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            if col == "Model":
                continue
            if col in higher_better:
                best_idx = df[col].idxmax()
            elif col in lower_better:
                best_idx = df[col].idxmin()
            else:
                continue
            styled.loc[best_idx, col] = "background-color: #f1f5f9; font-weight: 600; color: #0f172a;"
        return styled

    st.dataframe(df_metrics.style.apply(_highlight, axis=None),
                 use_container_width=True, hide_index=True)

    st.divider()

    # Bootstrap CIs
    st.subheader("Bootstrap confidence intervals")
    st.caption("95% CI over 1 000 resamples of the test set. Narrower = more stable.")

    ci = res["ci_results"]
    models_ci = list(ci.keys())
    colors_ci = [MODEL_COLORS[m] for m in models_ci]

    col_r2, col_rmse = st.columns(2)

    with col_r2:
        fig = go.Figure(go.Bar(
            x=models_ci,
            y=[ci[m]["r2_mean"] for m in models_ci],
            error_y=dict(
                type="data", symmetric=False,
                array=    [ci[m]["r2_hi"]   - ci[m]["r2_mean"] for m in models_ci],
                arrayminus=[ci[m]["r2_mean"] - ci[m]["r2_lo"]  for m in models_ci],
                color="#0f172a", thickness=1.2,
            ),
            marker_color=colors_ci,
            text=[f'{ci[m]["r2_mean"]:.4f}' for m in models_ci],
            textposition="outside",
        ))
        fig.update_layout(title="R² ± 95% CI", yaxis_title="R²")
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    with col_rmse:
        fig = go.Figure(go.Bar(
            x=models_ci,
            y=[ci[m]["rmse_mean"] for m in models_ci],
            error_y=dict(
                type="data", symmetric=False,
                array=    [ci[m]["rmse_hi"]   - ci[m]["rmse_mean"] for m in models_ci],
                arrayminus=[ci[m]["rmse_mean"] - ci[m]["rmse_lo"]  for m in models_ci],
                color="#0f172a", thickness=1.2,
            ),
            marker_color=colors_ci,
            text=[f'{ci[m]["rmse_mean"]:.4f}' for m in models_ci],
            textposition="outside",
        ))
        fig.update_layout(title="RMSE ± 95% CI", yaxis_title="RMSE")
        st.plotly_chart(_style_fig(fig), use_container_width=True)

    ci_rows = [{
        "Model":        m,
        "R² mean":      round(ci[m]["r2_mean"],   4),
        "R² 95% CI":    f'[{ci[m]["r2_lo"]:.4f}, {ci[m]["r2_hi"]:.4f}]',
        "RMSE mean":    round(ci[m]["rmse_mean"], 4),
        "RMSE 95% CI":  f'[{ci[m]["rmse_lo"]:.4f}, {ci[m]["rmse_hi"]:.4f}]',
    } for m in models_ci]
    st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Predicted vs actual
    st.subheader("Predicted vs actual")
    model_choice = st.radio("Model", list(ALL_PREDS.keys()), horizontal=True,
                            label_visibility="collapsed")
    yp    = ALL_PREDS[model_choice]
    color = MODEL_COLORS[model_choice]
    r2    = r2_score(y_test, yp)
    rmse  = float(np.sqrt(mean_squared_error(y_test, yp)))
    mn, mx = float(min(y_test.min(), yp.min())), float(max(y_test.max(), yp.max()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=yp, mode="markers",
        marker=dict(color=color, size=7, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        name=model_choice,
        hovertemplate="Actual=%{x:.3f}<br>Predicted=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color="#0f172a", dash="dot", width=1),
        name="Perfect fit",
    ))
    fig.update_layout(
        title=f"{model_choice} · Test R² = {r2:.4f} · RMSE = {rmse:.4f}",
        xaxis_title="Actual (V/√fc)",
        yaxis_title="Predicted (V/√fc)",
    )
    st.plotly_chart(_style_fig(fig, height=500), use_container_width=True)

    st.divider()

    # Side-by-side metric bar
    metric_bar = st.selectbox(
        "Compare models on:",
        ["Test R²", "Test Adj R²", "Test RMSE", "Test MAE", "Test AIC", "Test BIC"],
    )
    key = metric_bar.lower().replace(" ", "_").replace("²", "2")
    vals = [ALL_METRICS[m][key] for m in ALL_METRICS]
    fig = go.Figure(go.Bar(
        x=list(ALL_METRICS.keys()), y=vals,
        marker_color=list(MODEL_COLORS.values()),
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(title=metric_bar, yaxis_title=metric_bar)
    st.plotly_chart(_style_fig(fig), use_container_width=True)

    # NAM convergence — both variants
    st.divider()
    st.subheader("NAM backfitting convergence")
    st.caption("Solid = NAM+Interactions · Dashed = NAM (no interactions)")
    tl,    vl    = res["train_losses"],      res["val_losses"]
    tl_b,  vl_b  = res["train_losses_base"], res["val_losses_base"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=tl_b, mode="lines+markers", name="NAM train",
                             line=dict(color=MODEL_COLORS["NAM"], width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(y=vl_b, mode="lines+markers", name="NAM val",
                             line=dict(color=MODEL_COLORS["NAM"], width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(y=tl, mode="lines+markers", name="NAM+Inter train",
                             line=dict(color=MODEL_COLORS["NAM+Interactions"], width=2)))
    fig.add_trace(go.Scatter(y=vl, mode="lines+markers", name="NAM+Inter val",
                             line=dict(color=MODEL_COLORS["NAM+Interactions"], width=2, dash="dot")))
    fig.update_layout(xaxis_title="Backfitting cycle", yaxis_title="RMSE (raw sum)")
    st.plotly_chart(_style_fig(fig, height=360, legend_y=0.97), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EBM MAIN EFFECTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("EBM main-effect shape functions")
    st.caption("How each feature alone pushes the prediction, with all other "
               "features held constant. X-axis in original physical units.")

    main_idxs  = [i for i in range(len(term_names)) if not _is_interaction(i)]
    main_sort  = sorted(main_idxs, key=lambda i: importances[i], reverse=True)
    main_names = [term_names[i] for i in main_sort]

    selected = st.multiselect(
        "Features (sorted by importance)",
        options=main_names, default=main_names[:6],
    )

    if not selected:
        st.info("Select at least one feature above.")
    else:
        cols = st.columns(min(3, len(selected)))
        for i, feat in enumerate(selected):
            idx  = term_names.index(feat)
            data = ebm_global.data(idx)
            with cols[i % 3]:
                try:
                    x_raw  = np.array(data["names"], dtype=float)
                    y_vals = np.array(data["scores"])
                    n      = min(len(x_raw), len(y_vals))
                    x_orig = inv_1d(x_raw[:n], feat, scaler, all_feat)
                    y_vals = y_vals[:n]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_orig, y=y_vals, mode="lines",
                        line=dict(color=MODEL_COLORS["EBM"], width=2.2),
                        fill="tozeroy",
                        fillcolor="rgba(153,27,27,0.10)",
                    ))
                    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{feat}</b>"
                                 f"<br><sup style='color:#64748b;'>"
                                 f"importance {importances[idx]:.4f}</sup>",
                            font_size=13,
                        ),
                        xaxis_title=feat,
                        yaxis_title="Shape contribution",
                        showlegend=False,
                    )
                    st.plotly_chart(_style_fig(fig, height=300), use_container_width=True)
                except Exception as e:
                    st.error(f"{feat}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EBM INTERACTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("EBM pairwise interaction surfaces")
    st.caption("Red = pair pushes prediction up · Blue = down · White ≈ no joint effect.")

    inter_idxs = [i for i in range(len(term_names)) if _is_interaction(i)]
    if not inter_idxs:
        st.warning("No interaction terms — raise 'EBM interaction pairs' in the sidebar and re-run.")
    else:
        inter_sort = sorted(inter_idxs, key=lambda i: importances[i], reverse=True)

        st.markdown("**Ranked interaction pairs**")
        summary = pd.DataFrame([
            {
                "Rank":       rank + 1,
                "Pair":       term_names[i],
                "Feature A":  _term_feature_names(i)[0],
                "Feature B":  _term_feature_names(i)[1],
                "Importance": round(float(importances[i]), 5),
            }
            for rank, i in enumerate(inter_sort)
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.divider()

        for idx in inter_sort:
            term   = term_names[idx]
            ni, nj = _term_feature_names(idx)
            data   = ebm_global.data(idx)
            imp    = importances[idx]

            with st.expander(f"{term}   ·   importance = {imp:.5f}", expanded=True):
                try:
                    _raw = data.get("scores")
                    if _raw is None: _raw = data.get("values")
                    if _raw is None: _raw = []
                    scores = np.array(_raw)
                    if scores.ndim != 2:
                        raise ValueError(
                            f"Expected 2-D scores, got shape {scores.shape}. "
                            f"Keys: {list(data.keys())}"
                        )

                    if "names" in data and isinstance(data["names"], (list, tuple)) and len(data["names"]) == 2:
                        bins_i = np.array(data["names"][0], dtype=float)
                        bins_j = np.array(data["names"][1], dtype=float)
                    elif "left_names" in data and "right_names" in data:
                        bins_i = np.array(data["left_names"], dtype=float)
                        bins_j = np.array(data["right_names"], dtype=float)
                    elif "bin_labels" in data and len(data["bin_labels"]) == 2:
                        bins_i = np.array(data["bin_labels"][0], dtype=float)
                        bins_j = np.array(data["bin_labels"][1], dtype=float)
                    else:
                        bins_i = np.arange(scores.shape[0], dtype=float)
                        bins_j = np.arange(scores.shape[1], dtype=float)

                    xi_orig = inv_1d(bins_i, ni, scaler, all_feat)
                    xj_orig = inv_1d(bins_j, nj, scaler, all_feat)
                    vmax    = float(np.abs(scores).max()) or 1.0

                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(
                        x=xi_orig, y=xj_orig, z=scores.T,
                        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
                        colorbar=dict(title="Contribution", len=0.85, thickness=14),
                    ))
                    fig.add_trace(go.Contour(
                        x=xi_orig, y=xj_orig, z=scores.T,
                        showscale=False, ncontours=7,
                        contours=dict(coloring="none", showlabels=True,
                                      labelfont=dict(size=9, color="#0f172a")),
                        line=dict(color="#0f172a", width=0.6),
                    ))
                    fig.update_layout(
                        xaxis_title=f"{ni}  (original units)",
                        yaxis_title=f"{nj}  (original units)",
                    )
                    st.plotly_chart(_style_fig(fig, height=460), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render {term}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NAM SHAPES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    X_test_sel     = res["X_test_sel"]
    inter_contribs = res["inter_contribs"]
    pair_labels    = res["pair_labels"]

    # Toggle between pure NAM and NAM+Interactions shapes
    nam_variant = st.radio(
        "Show shapes for",
        ["NAM (no interactions)", "NAM+Interactions"],
        horizontal=True,
        label_visibility="collapsed",
    )
    using_base = nam_variant == "NAM (no interactions)"
    contributions = res["contributions_base"] if using_base else res["contributions"]
    variant_color = MODEL_COLORS["NAM"] if using_base else MODEL_COLORS["NAM+Interactions"]
    variant_fill  = "rgba(29,78,216,0.10)" if using_base else "rgba(54,83,20,0.10)"

    mean_abs = [float(np.abs(c).mean()) for c in contributions]
    order    = np.argsort(mean_abs)[::-1]

    st.subheader("NAM single-feature shape functions")
    st.caption("Learned by backfitting — each curve is one MLP's contribution "
               "with every other feature's effect already removed.")

    selected_nam = st.multiselect(
        "Features (sorted by mean |contribution|)",
        options=[feat_sel[i] for i in order],
        default=[feat_sel[i] for i in order[:6]],
    )

    if not selected_nam:
        st.info("Select at least one feature above.")
    else:
        cols = st.columns(min(3, len(selected_nam)))
        for i, feat in enumerate(selected_nam):
            j         = list(feat_sel).index(feat)
            x_scaled  = X_test_sel[:, j]
            x_orig    = inv_1d(x_scaled, feat, scaler, all_feat)
            y_contrib = contributions[j].ravel()
            sort_idx  = np.argsort(x_orig)

            with cols[i % 3]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_orig[sort_idx], y=y_contrib[sort_idx],
                    mode="lines",
                    line=dict(color=variant_color, width=2.2),
                    fill="tozeroy", fillcolor=variant_fill,
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
                fig.update_layout(
                    title=dict(
                        text=f"<b>{feat}</b>"
                             f"<br><sup style='color:#64748b;'>"
                             f"mean |contrib| {mean_abs[j]:.4f}</sup>",
                        font_size=13,
                    ),
                    xaxis_title=feat,
                    yaxis_title="Contribution",
                    showlegend=False,
                )
                st.plotly_chart(_style_fig(fig, height=300), use_container_width=True)

    # NAM interaction scatter plots
    if pair_labels:
        st.divider()
        st.subheader("NAM interaction contributions (EBM-guided pairs)")
        st.caption("Each dot is one test sample; colour is the 2-input MLP's contribution.")

        for label, contrib in zip(pair_labels, inter_contribs):
            ni, nj = label.split(" x ")
            if ni not in feat_sel or nj not in feat_sel:
                continue
            ci_ = list(feat_sel).index(ni)
            cj_ = list(feat_sel).index(nj)
            xi  = inv_1d(X_test_sel[:, ci_], ni, scaler, all_feat)
            xj  = inv_1d(X_test_sel[:, cj_], nj, scaler, all_feat)
            c   = contrib.ravel()
            vmax = float(np.abs(c).max()) or 1.0

            with st.expander(f"{label}", expanded=True):
                fig = go.Figure(go.Scatter(
                    x=xi, y=xj, mode="markers",
                    marker=dict(
                        color=c, colorscale="RdBu_r",
                        cmin=-vmax, cmax=vmax,
                        colorbar=dict(title="Contribution", thickness=14),
                        size=8, opacity=0.85,
                        line=dict(color="white", width=0.4),
                    ),
                    hovertemplate=(
                        f"{ni}=%{{x:.3f}}<br>"
                        f"{nj}=%{{y:.3f}}<br>"
                        "contrib=%{marker.color:.4f}<extra></extra>"
                    ),
                ))
                fig.update_layout(
                    xaxis_title=f"{ni}  (original units)",
                    yaxis_title=f"{nj}  (original units)",
                )
                st.plotly_chart(_style_fig(fig, height=440), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RESIDUAL DIAGNOSTICS  (credibility feature)
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Residual diagnostics")
    st.caption(
        "Standard regression validation checks. A trustworthy model should show "
        "residuals that are (i) unbiased around zero, (ii) homoscedastic across "
        "the predicted range, and (iii) approximately normally distributed."
    )

    diag_model = st.radio(
        "Model", list(ALL_PREDS.keys()),
        horizontal=True, label_visibility="collapsed", key="diag_model",
    )
    yp        = ALL_PREDS[diag_model]
    color     = MODEL_COLORS[diag_model]
    residuals = y_test - yp
    std_resid = (residuals - residuals.mean()) / (residuals.std(ddof=1) + 1e-12)

    # ── Summary metrics ──
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if 3 <= len(residuals) <= 5000 else (np.nan, np.nan)
    mean_r   = float(residuals.mean())
    std_r    = float(residuals.std(ddof=1))
    skew_r   = float(stats.skew(residuals))
    kurt_r   = float(stats.kurtosis(residuals))  # excess

    # ± 1 SD coverage as a sanity check on homoscedastic/normal behaviour
    within_1sd = float(np.mean(np.abs(std_resid) <= 1.0) * 100)
    within_2sd = float(np.mean(np.abs(std_resid) <= 2.0) * 100)

    cards = [
        ("Mean residual",      f"{mean_r:+.4f}", "bias around 0"),
        ("Residual σ",         f"{std_r:.4f}",    "spread"),
        ("Skewness",           f"{skew_r:+.3f}",  "symmetry (0 = ideal)"),
        ("Excess kurtosis",    f"{kurt_r:+.3f}",  "tails (0 = Gaussian)"),
        ("Shapiro–Wilk p",     f"{shapiro_p:.3f}" if not np.isnan(shapiro_p) else "—",
                                "normality (>0.05 = normal)"),
        ("Within ±2σ",         f"{within_2sd:.1f}%", "ideal ≈ 95%"),
    ]
    ccols = st.columns(len(cards))
    for col, (label, value, sub) in zip(ccols, cards):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value' style='font-size:20px;'>{value}</div>"
            f"<div class='metric-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Residual vs Predicted (homoscedasticity) ──
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yp, y=residuals, mode="markers",
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            hovertemplate="Pred=%{x:.3f}<br>Residual=%{y:.3f}<extra></extra>",
            name="Residuals",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#0f172a")
        # LOESS-ish local mean via binning for trend overlay
        try:
            order_ = np.argsort(yp)
            yp_s, r_s = yp[order_], residuals[order_]
            bins = np.linspace(yp_s.min(), yp_s.max(), 12)
            centres, means = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (yp_s >= lo) & (yp_s < hi)
                if mask.sum() >= 3:
                    centres.append((lo + hi) / 2)
                    means.append(r_s[mask].mean())
            if centres:
                fig.add_trace(go.Scatter(
                    x=centres, y=means, mode="lines",
                    line=dict(color="#0f172a", width=1.8),
                    name="Local mean",
                ))
        except Exception:
            pass
        fig.update_layout(
            title="Residuals vs predicted",
            xaxis_title="Predicted (V/√fc)",
            yaxis_title="Residual (actual − predicted)",
        )
        st.plotly_chart(_style_fig(fig, height=420), use_container_width=True)

    # ── Residual distribution with normal overlay ──
    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=25, histnorm="probability density",
            marker=dict(color=color, line=dict(color="white", width=0.8)),
            opacity=0.8, name="Residuals",
        ))
        # Fitted normal curve
        xs = np.linspace(residuals.min(), residuals.max(), 200)
        pdf = stats.norm.pdf(xs, loc=mean_r, scale=std_r)
        fig.add_trace(go.Scatter(
            x=xs, y=pdf, mode="lines",
            line=dict(color="#0f172a", width=1.8, dash="dot"),
            name=f"𝒩({mean_r:+.3f}, {std_r:.3f})",
        ))
        fig.update_layout(
            title="Residual distribution",
            xaxis_title="Residual",
            yaxis_title="Density",
        )
        st.plotly_chart(_style_fig(fig, height=420, legend_y=0.97), use_container_width=True)

    # ── Q-Q plot ──
    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        qq_x, qq_y = stats.probplot(residuals, dist="norm", fit=False)
        slope, intercept, *_ = stats.linregress(qq_x, qq_y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=qq_x, y=qq_y, mode="markers",
            marker=dict(color=color, size=7, opacity=0.8,
                        line=dict(color="white", width=0.5)),
            name="Residual quantiles",
        ))
        line_x = np.array([qq_x.min(), qq_x.max()])
        fig.add_trace(go.Scatter(
            x=line_x, y=intercept + slope * line_x, mode="lines",
            line=dict(color="#0f172a", width=1.4, dash="dot"),
            name="Reference",
        ))
        fig.update_layout(
            title="Q–Q plot (residuals vs normal)",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
        )
        st.plotly_chart(_style_fig(fig, height=380, legend_y=0.97), use_container_width=True)

    # ── Error by prediction magnitude (binned absolute error) ──
    with col_d:
        abs_err = np.abs(residuals)
        try:
            bins = np.quantile(yp, np.linspace(0, 1, 6))
            bins = np.unique(bins)
            labels_b, means_b, stds_b = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (yp >= lo) & (yp <= hi)
                if m.sum() >= 2:
                    labels_b.append(f"{lo:.2f}–{hi:.2f}")
                    means_b.append(abs_err[m].mean())
                    stds_b.append(abs_err[m].std(ddof=1))
            fig = go.Figure(go.Bar(
                x=labels_b, y=means_b,
                error_y=dict(type="data", array=stds_b, color="#0f172a", thickness=1.2),
                marker_color=color,
                text=[f"{v:.3f}" for v in means_b], textposition="outside",
            ))
            fig.update_layout(
                title="Mean |error| by prediction quintile",
                xaxis_title="Predicted range",
                yaxis_title="Mean absolute error",
            )
            st.plotly_chart(_style_fig(fig, height=380), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute binned error: {e}")

    # ── Interpretation panel ──
    st.divider()
    interpretation = []
    if abs(mean_r) > 0.05 * std_r:
        interpretation.append(
            f"• Mean residual is **{mean_r:+.4f}** — slight bias; an unbiased "
            f"model would be near 0 relative to σ = {std_r:.4f}."
        )
    else:
        interpretation.append(f"• Residuals are **unbiased** (mean ≈ 0 relative to σ).")

    if not np.isnan(shapiro_p):
        if shapiro_p > 0.05:
            interpretation.append(
                f"• Shapiro–Wilk p = **{shapiro_p:.3f}** — residuals are "
                f"consistent with a normal distribution."
            )
        else:
            interpretation.append(
                f"• Shapiro–Wilk p = **{shapiro_p:.3f}** (< 0.05) — residuals "
                f"deviate from normality; check the Q–Q plot tails."
            )

    if abs(skew_r) > 0.5:
        interpretation.append(
            f"• Skewness = **{skew_r:+.2f}** — distribution is asymmetric "
            f"({'right' if skew_r > 0 else 'left'}-tailed)."
        )
    if kurt_r > 1:
        interpretation.append(
            f"• Excess kurtosis = **{kurt_r:+.2f}** — heavier tails than Gaussian; "
            f"a few predictions miss badly."
        )

    interpretation.append(
        f"• Coverage: **{within_1sd:.1f}%** within ±1σ (ideal 68%), "
        f"**{within_2sd:.1f}%** within ±2σ (ideal 95%)."
    )

    st.markdown("**Interpretation**")
    st.markdown("\n".join(interpretation))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EBM INTERACTION SWEEP
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("EBM interaction sweep")
    st.caption(
        "Trains EBM six times (0–5 interaction pairs) on the same preprocessed data. "
        "Typically completes in under a minute. Helps identify whether adding interactions "
        "genuinely improves generalisation or just overfits."
    )

    sweep_btn = st.button("Run sweep (0–5 interactions)", type="primary")
    if sweep_btn:
        with st.spinner("Sweeping EBM interactions 0 → 5…"):
            sweep_df = run_ebm_interaction_sweep(
                res["X_train_sel"], res["X_test_sel"],
                res["y_train_aug"], res["y_test"],
                res["feature_names_sel"], seed=seed,
            )
            st.session_state["sweep_df"] = sweep_df

    if "sweep_df" not in st.session_state:
        st.info("Click **Run sweep** above to begin.")
    else:
        sw = st.session_state["sweep_df"]
        xs = sw["n_interactions"].tolist()

        def _line(y_col, title, y_label, higher_better=True):
            best_i = sw[y_col].idxmax() if higher_better else sw[y_col].idxmin()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=sw[y_col].tolist(), mode="lines+markers",
                line=dict(color=MODEL_COLORS["EBM"], width=2.2),
                marker=dict(size=8),
                hovertemplate=f"n=%{{x}}<br>{y_col}=%{{y:.4f}}<extra></extra>",
            ))
            # Highlight best point
            fig.add_trace(go.Scatter(
                x=[sw.loc[best_i, "n_interactions"]],
                y=[sw.loc[best_i, y_col]],
                mode="markers",
                marker=dict(color="#0f172a", size=12, symbol="star"),
                name=f"Best (n={sw.loc[best_i,'n_interactions']})",
            ))
            fig.update_layout(title=title, xaxis_title="n_interactions",
                              yaxis_title=y_label,
                              xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            return _style_fig(fig, height=320)

        # ── Row 1: R² and Adj R² ──────────────────────────────────────────────
        st.markdown("#### Fit quality")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_line("test_r2",     "Test R²",     "R²"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(_line("test_adj_r2", "Test Adj R²", "Adj R²"),
                            use_container_width=True)

        # ── Row 2: RMSE and MAE ──────────────────────────────────────────────
        st.markdown("#### Error magnitude")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_line("test_rmse", "Test RMSE", "RMSE", higher_better=False),
                            use_container_width=True)
        with c2:
            st.plotly_chart(_line("test_mae",  "Test MAE",  "MAE",  higher_better=False),
                            use_container_width=True)

        # ── Row 3: Train–test gap (overfitting) ──────────────────────────────
        st.markdown("#### Generalisation gap  (train − test)")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=xs, y=sw["r2_gap"].tolist(),
                marker_color=[
                    "#991b1b" if v > sw["r2_gap"].median() else "#334155"
                    for v in sw["r2_gap"]
                ],
                text=[f"{v:.4f}" for v in sw["r2_gap"]], textposition="outside",
            ))
            fig.update_layout(title="R² gap (train − test, lower = less overfit)",
                              xaxis_title="n_interactions", yaxis_title="R² gap",
                              xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(_style_fig(fig, height=320), use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=xs, y=sw["rmse_gap"].tolist(),
                marker_color=[
                    "#991b1b" if v > sw["rmse_gap"].median() else "#334155"
                    for v in sw["rmse_gap"]
                ],
                text=[f"{v:.4f}" for v in sw["rmse_gap"]], textposition="outside",
            ))
            fig.update_layout(title="RMSE gap (test − train, lower = less overfit)",
                              xaxis_title="n_interactions", yaxis_title="RMSE gap",
                              xaxis=dict(tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(_style_fig(fig, height=320), use_container_width=True)

        # ── Row 4: Confidence & bias ─────────────────────────────────────────
        st.markdown("#### Stability & bias")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(
                _line("ci_width_r2", "Bootstrap CI width — R²",
                      "CI width (narrower = more stable)", higher_better=False),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _line("bias", "|Mean residual| — bias",
                      "|mean residual| (lower = less bias)", higher_better=False),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                _line("residual_std", "Residual σ",
                      "σ (lower = less spread)", higher_better=False),
                use_container_width=True,
            )

        # ── Row 5: Information criteria ──────────────────────────────────────
        st.markdown("#### Information criteria")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_line("test_aic", "AIC", "AIC (lower = better)",
                                  higher_better=False), use_container_width=True)
        with c2:
            st.plotly_chart(_line("test_bic", "BIC", "BIC (lower = better)",
                                  higher_better=False), use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Full sweep table")

        # Compute marginal gains row-over-row
        disp = sw.copy()
        disp.insert(0, "Δ test R²",   disp["test_r2"].diff().round(4))
        disp.insert(0, "Δ test RMSE", disp["test_rmse"].diff().round(4))

        def _hl_sweep(df):
            styled = pd.DataFrame("", index=df.index, columns=df.columns)
            higher = {"test_r2", "test_adj_r2", "train_r2", "train_adj_r2"}
            lower  = {"test_rmse", "train_rmse", "test_mae", "test_aic", "test_bic",
                      "r2_gap", "rmse_gap", "ci_width_r2", "ci_width_rmse", "bias", "residual_std"}
            for col in df.columns:
                if col in higher:
                    styled.loc[df[col].idxmax(), col] = "background-color:#f1f5f9;font-weight:600;"
                elif col in lower:
                    styled.loc[df[col].idxmin(), col] = "background-color:#f1f5f9;font-weight:600;"
            return styled

        st.dataframe(disp.style.apply(_hl_sweep, axis=None),
                     use_container_width=True, hide_index=True)

        # ── Recommendation ────────────────────────────────────────────────────
        st.divider()
        best_r2_n  = int(sw.loc[sw["test_r2"].idxmax(),   "n_interactions"])
        best_rmse_n = int(sw.loc[sw["test_rmse"].idxmin(), "n_interactions"])
        best_gap_n  = int(sw.loc[sw["r2_gap"].idxmin(),   "n_interactions"])

        st.markdown("**Recommendation**")
        if best_r2_n == best_rmse_n:
            rec = f"n = **{best_r2_n}** maximises Test R² and minimises RMSE simultaneously."
        else:
            rec = (f"Test R² peaks at n = **{best_r2_n}**, RMSE is lowest at n = **{best_rmse_n}**. "
                   f"Smallest train–test gap is at n = **{best_gap_n}**.")
        st.markdown(rec + f"  Current pipeline uses n = **{n_interactions}**.")

        # Download sweep CSV
        st.download_button(
            "Download sweep_results.csv",
            data=sw.to_csv(index=False),
            file_name="ebm_interaction_sweep.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PIPELINE LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Pipeline log")
    st.caption("Everything the pipeline printed during the last run.")

    log = st.session_state.get("pipeline_log", "")

    if not log:
        st.info("No log yet — run the pipeline first.")
    else:
        st.download_button(
            label="Download log",
            data=log, file_name="pipeline_log.txt", mime="text/plain",
        )

        search = st.text_input("Filter lines", placeholder="e.g. R²  or  RMSE  or  Cycle")
        lines  = log.splitlines()
        if search:
            lines = [l for l in lines if search.lower() in l.lower()]
            st.caption(f"{len(lines)} matching lines")

        st.markdown(
            "<div style='"
            "background:#0f172a; color:#e2e8f0; font-family:ui-monospace,monospace; "
            "font-size:12.5px; line-height:1.6; padding:16px 20px; "
            "border-radius:8px; overflow-x:auto; white-space:pre-wrap; "
            "max-height:600px; overflow-y:auto;"
            f"'>{chr(10).join(lines) if lines else '(no matching lines)'}</div>",
            unsafe_allow_html=True,
        )
