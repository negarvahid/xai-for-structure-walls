#!/usr/bin/env python3
"""
Comprehensive Data Engineering & ML Pipeline for Structural Wall Dataset
- Multiple imputation strategies
- Normalization techniques
- Baseline models: XGBoost, Random Forest
- Neural Additive Model (NAM)
"""

import os
import random

# Ensure headless matplotlib + writable cache (important on locked-down systems)
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_ROOT_DIR, ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility across python/numpy (and downstream libs)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def adjusted_r2(r2: float, n: int, k: int) -> float:
    """Compute adjusted R² given R², number of samples n, and number of features k.

    Penalises adding predictors that don't improve the model:
        adj_R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)
    Returns NaN if the denominator is zero or negative.
    """
    if n <= k + 1:
        return float('nan')
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def compute_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int):
    """Compute AIC and BIC for a regression model using the Gaussian log-likelihood.

    Formulas (lower = better; penalise complexity differently):
        AIC = n·ln(RSS/n) + 2·k
        BIC = n·ln(RSS/n) + k·ln(n)

    k is the effective number of parameters (here: number of input features + 1 for
    the implicit intercept/bias term).  For black-box tree / boosting models this is
    a lower-bound approximation — the relative ranking across models is still useful.
    Returns (nan, nan) if the residual sum of squares is non-positive.
    """
    n = len(y_true)
    rss = float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))
    if rss <= 0 or n <= 0:
        return float('nan'), float('nan')
    log_term = n * np.log(rss / n)
    aic = log_term + 2 * k
    bic = log_term + k * np.log(n)
    return aic, bic


# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================

def load_and_clean_data(filepath):
    """Load Excel file and clean column names."""
    # Read raw data (skip header rows)
    df_raw = pd.read_excel(filepath, header=None)
    
    # Data starts at row 3 (0-indexed)
    # Row 0-2 are headers with subheaders
    df = df_raw.iloc[3:].copy().reset_index(drop=True)
    
    # Define column names based on the structure observed
    columns = [
        'Paper_No', 'Specimen', 'Thickness_mm', 'Length_mm', 'Height_mm',
        'Shear_Span_Ratio', 'Aspect_Ratio', 'Walls_Cross_Section', 'Failure_Type',
        'Curvature_Type', 'Axial_Load_Ratio', 'fc_MPa', 'rho_bl_pct', 'fybl_MPa',
        'rho_sh_pct', 'rho_t_pct', 'fy_MPa', 'rho_l_pct', 'fyl_MPa',
        'Unknown_1', 'Loading_Protocol', 'Ab_Ag', 'lw_tw', 'Unknown_2',
        'rho_l_fyl_fc', 'rho_t_fyt_fc', 'rho_sh_fysh_fc', 'rho_bl_fybl_fc',
        'Unknown_3', 'Max_Shear_Force_kN', 'V_sqrt_fc'
    ]
    df.columns = columns
    
    # Drop completely empty/unknown columns
    drop_cols = ['Unknown_1', 'Unknown_2', 'Unknown_3', 'Loading_Protocol', 'Paper_No']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Convert numeric columns
    numeric_cols = [
        'Thickness_mm', 'Length_mm', 'Height_mm', 'Shear_Span_Ratio', 
        'Aspect_Ratio', 'Walls_Cross_Section', 'Failure_Type', 'Curvature_Type',
        'Axial_Load_Ratio', 'fc_MPa', 'rho_bl_pct', 'fybl_MPa', 'rho_sh_pct',
        'rho_t_pct', 'fy_MPa', 'rho_l_pct', 'fyl_MPa', 'Ab_Ag', 'lw_tw',
        'rho_l_fyl_fc', 'rho_t_fyt_fc', 'rho_sh_fysh_fc', 'rho_bl_fybl_fc',
        'Max_Shear_Force_kN', 'V_sqrt_fc'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where target is missing
    df = df.dropna(subset=['V_sqrt_fc'])
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

def eda_report(df):
    """Generate EDA report."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    
    print(f"\nMissing Values (%):")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing_pct[missing_pct > 0])
    
    print(f"\nNumeric Summary Statistics:")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.describe().T[['count', 'mean', 'std', 'min', 'max']])
    
    return missing_pct


def plot_eda(df, save_path='eda_plots.png'):
    """Create EDA visualizations."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Missing values heatmap
    ax1 = axes[0, 0]
    missing = df.isnull()
    sns.heatmap(missing.iloc[:100, :], cbar=True, ax=ax1, cmap='viridis')
    ax1.set_title('Missing Values Pattern (First 100 rows)')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Samples')
    
    # 2. Target distribution
    ax2 = axes[0, 1]
    df['V_sqrt_fc'].hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7)
    ax2.set_title('Target Distribution: V/sqrt(fc)')
    ax2.set_xlabel('V/sqrt(fc)')
    ax2.set_ylabel('Frequency')
    
    # 3. Correlation heatmap (top features)
    ax3 = axes[1, 0]
    top_numeric = numeric_df.dropna(axis=1, how='all').iloc[:, :12]
    corr = top_numeric.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title('Feature Correlations')
    
    # 4. Feature distributions
    ax4 = axes[1, 1]
    key_features = ['fc_MPa', 'Axial_Load_Ratio', 'Aspect_Ratio', 'rho_l_fyl_fc']
    available = [f for f in key_features if f in df.columns]
    if available:
        df[available].hist(bins=30, ax=ax4, alpha=0.7)
    ax4.set_title('Key Feature Distributions')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nEDA plots saved to {save_path}")


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create derived features."""
    df = df.copy()
    
    # Geometric features
    if 'Length_mm' in df.columns and 'Thickness_mm' in df.columns:
        df['Area_mm2'] = df['Length_mm'] * df['Thickness_mm']
    
    if 'Height_mm' in df.columns and 'Length_mm' in df.columns:
        df['H_L_ratio'] = df['Height_mm'] / df['Length_mm'].replace(0, np.nan)
    
    # Reinforcement indices (total reinforcement indicator)
    rho_cols = ['rho_l_pct', 'rho_t_pct', 'rho_sh_pct', 'rho_bl_pct']
    available_rho = [c for c in rho_cols if c in df.columns]
    if available_rho:
        df['Total_Rho'] = df[available_rho].sum(axis=1, skipna=True)
    
    # Normalized reinforcement ratios
    ratio_cols = ['rho_l_fyl_fc', 'rho_t_fyt_fc', 'rho_sh_fysh_fc', 'rho_bl_fybl_fc']
    available_ratios = [c for c in ratio_cols if c in df.columns]
    if available_ratios:
        df['Sum_Rho_Ratio'] = df[available_ratios].sum(axis=1, skipna=True)
    
    # Interaction terms
    if 'fc_MPa' in df.columns and 'Axial_Load_Ratio' in df.columns:
        df['fc_x_ALR'] = df['fc_MPa'] * df['Axial_Load_Ratio']
    
    if 'Aspect_Ratio' in df.columns and 'Shear_Span_Ratio' in df.columns:
        df['AR_x_SSR'] = df['Aspect_Ratio'] * df['Shear_Span_Ratio']
    
    print(f"\nEngineered {len(df.columns)} total features")
    return df


def engineer_advanced_features(df):
    """
    Advanced feature engineering for SCARCE DATA.
    Creates domain-specific, physics-based, and interaction features.
    """
    df = df.copy()
    n_original = len(df.columns)
    
    print("\n" + "="*70)
    print("ADVANCED FEATURE ENGINEERING FOR SCARCE DATA")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1. PHYSICS-BASED FEATURES (Structural Engineering Domain Knowledge)
    # -------------------------------------------------------------------------
    print("\n[1] Physics-Based Features...")
    
    # Shear capacity indicators (ACI 318 inspired)
    if all(c in df.columns for c in ['fc_MPa', 'Thickness_mm', 'Length_mm']):
        # Concrete contribution to shear (simplified)
        df['Vc_proxy'] = np.sqrt(df['fc_MPa']) * df['Thickness_mm'] * df['Length_mm'] / 1000
    
    # Reinforcement contribution proxies
    if all(c in df.columns for c in ['rho_t_pct', 'fy_MPa', 'Thickness_mm', 'Length_mm']):
        df['Vs_proxy'] = df['rho_t_pct'] * df['fy_MPa'] * df['Thickness_mm'] * df['Length_mm'] / 100000
    
    # Out-of-plane slenderness (hw / tw)
    if all(c in df.columns for c in ['Height_mm', 'Thickness_mm']):
        df['Out_of_plane_slenderness'] = df['Height_mm'] / df['Thickness_mm']
    
    # Confinement effectiveness
    if all(c in df.columns for c in ['rho_sh_pct', 'rho_bl_pct']):
        # Updated confinement index: simple sum rho_sh + rho_bl
        df['Confinement_Index'] = df['rho_sh_pct'] + 2 * df['rho_bl_pct']
    
    # Axial stress
    if all(c in df.columns for c in ['Axial_Load_Ratio', 'fc_MPa']):
        df['Axial_Stress'] = df['Axial_Load_Ratio'] * df['fc_MPa']
    
    # -------------------------------------------------------------------------
    # 2. GEOMETRIC FEATURES
    # -------------------------------------------------------------------------
    print("[2] Geometric Features...")
    
    if all(c in df.columns for c in ['Length_mm', 'Thickness_mm', 'Height_mm']):
        df['Volume_mm3'] = df['Length_mm'] * df['Thickness_mm'] * df['Height_mm']
        df['Surface_Area'] = 2 * (df['Length_mm'] * df['Thickness_mm'] + 
                                   df['Length_mm'] * df['Height_mm'] + 
                                   df['Thickness_mm'] * df['Height_mm'])
        df['Compactness'] = df['Volume_mm3'] / (df['Surface_Area'] + 1e-10)
    
    # Section modulus proxy
    if all(c in df.columns for c in ['Thickness_mm', 'Length_mm']):
        df['Section_Modulus'] = df['Thickness_mm'] * (df['Length_mm'] ** 2) / 6
        df['Moment_Inertia'] = df['Thickness_mm'] * (df['Length_mm'] ** 3) / 12
    
    # -------------------------------------------------------------------------
    # 3. REINFORCEMENT FEATURES
    # -------------------------------------------------------------------------
    print("[3] Reinforcement Features...")
    
    rho_cols = ['rho_l_pct', 'rho_t_pct', 'rho_sh_pct', 'rho_bl_pct']
    available_rho = [c for c in rho_cols if c in df.columns]
    
    if len(available_rho) >= 2:
        # Total reinforcement
        df['Total_Rho'] = df[available_rho].sum(axis=1)
        
        # Reinforcement ratios (longitudinal / transverse).
        # If transverse steel is zero, set ratio to NaN (will be handled by imputation later).
        if 'rho_l_pct' in df.columns and 'rho_t_pct' in df.columns:
            mask = df['rho_t_pct'] != 0
            df['Rho_L_T_Ratio'] = np.nan
            df.loc[mask, 'Rho_L_T_Ratio'] = df.loc[mask, 'rho_l_pct'] / df.loc[mask, 'rho_t_pct']
        
        # Reinforcement balance
        df['Rho_Std'] = df[available_rho].std(axis=1)
        df['Rho_Max'] = df[available_rho].max(axis=1)
        df['Rho_Min'] = df[available_rho].min(axis=1)
        df['Rho_Range'] = df['Rho_Max'] - df['Rho_Min']
    
    # Mechanical reinforcement ratio (omega)
    if all(c in df.columns for c in ['rho_l_pct', 'fyl_MPa', 'fc_MPa']):
        df['Omega_l'] = (df['rho_l_pct'] / 100) * df['fyl_MPa'] / df['fc_MPa']
    
    if all(c in df.columns for c in ['rho_t_pct', 'fy_MPa', 'fc_MPa']):
        df['Omega_t'] = (df['rho_t_pct'] / 100) * df['fy_MPa'] / df['fc_MPa']
    
    # -------------------------------------------------------------------------
    # 4. INTERACTION FEATURES
    # -------------------------------------------------------------------------
    print("[4] Interaction Features...")
    
    # Key interactions based on structural mechanics
    interactions = [
        ('fc_MPa', 'Axial_Load_Ratio'),
        ('fc_MPa', 'Aspect_Ratio'),
        ('fc_MPa', 'Shear_Span_Ratio'),
        ('Aspect_Ratio', 'Axial_Load_Ratio'),
        ('Shear_Span_Ratio', 'Axial_Load_Ratio'),
        ('Aspect_Ratio', 'Shear_Span_Ratio'),
        ('Total_Rho', 'fc_MPa') if 'Total_Rho' in df.columns else None,
        ('Out_of_plane_slenderness', 'Axial_Load_Ratio') if 'Out_of_plane_slenderness' in df.columns else None,
    ]
    
    for pair in interactions:
        if pair is None:
            continue
        col1, col2 = pair
        if col1 in df.columns and col2 in df.columns:
            # Multiplication
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            # Division (both ways can be meaningful)
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-10)
    
    # -------------------------------------------------------------------------
    # 5. POLYNOMIAL FEATURES (selective)
    # -------------------------------------------------------------------------
    print("[5] Polynomial Features...")
    
    # Square terms for key features
    poly_cols = ['fc_MPa', 'Axial_Load_Ratio', 'Aspect_Ratio', 'Shear_Span_Ratio']
    for col in poly_cols:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    # -------------------------------------------------------------------------
    # 6. RATIO FEATURES
    # -------------------------------------------------------------------------
    print("[6] Ratio Features...")
    
    # Normalized reinforcement indices
    ratio_cols = ['rho_l_fyl_fc', 'rho_t_fyt_fc', 'rho_sh_fysh_fc', 'rho_bl_fybl_fc']
    available_ratios = [c for c in ratio_cols if c in df.columns]
    
    if len(available_ratios) >= 2:
        df['Sum_Rho_Ratio'] = df[available_ratios].sum(axis=1)
        df['Mean_Rho_Ratio'] = df[available_ratios].mean(axis=1)
        df['Max_Rho_Ratio'] = df[available_ratios].max(axis=1)
    
    # -------------------------------------------------------------------------
    # 7. CATEGORICAL ENCODING FEATURES  (moved to add_mean_encoding)
    # -------------------------------------------------------------------------
    # Group-mean encoding is computed AFTER the train/test split to avoid
    # leakage — test-set rows must not influence the group means seen by the
    # training set.  Call add_mean_encoding(df_train, df_test) instead.
    print("[7] Categorical Encoding... (deferred to post-split — see add_mean_encoding)")
    
    # -------------------------------------------------------------------------
    # 8. BINNED FEATURES (capture non-linearity)
    # -------------------------------------------------------------------------
    print("[8] Binned Features...")
    
    bin_cols = ['Axial_Load_Ratio', 'Aspect_Ratio', 'fc_MPa']
    for col in bin_cols:
        if col in df.columns:
            df[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    
    n_new = len(df.columns) - n_original
    print(f"\n>> Created {n_new} new features (total: {len(df.columns)})")
    
    return df


def add_mean_encoding(df_train, df_test):
    """Compute group-mean encoding features using TRAINING data only.

    Group means are computed exclusively from df_train, then mapped onto
    both df_train and df_test.  This prevents test-set rows from influencing
    the statistics seen by the model during training (target-leakage via
    group statistics).

    Features added
    --------------
    For Failure_Type groups  : {col}_by_Failure_mean, {col}_Failure_diff
      cols: fc_MPa, Axial_Load_Ratio, Aspect_Ratio
    For Walls_Cross_Section groups : {col}_by_Section_mean
      cols: fc_MPa, Total_Rho
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    # --- Failure_Type group means ----------------------------------------
    if 'Failure_Type' in df_train.columns:
        for col in ['fc_MPa', 'Axial_Load_Ratio', 'Aspect_Ratio']:
            if col not in df_train.columns:
                continue
            group_means = df_train.groupby('Failure_Type')[col].mean()
            overall_mean = df_train[col].mean()

            # Map train
            df_train[f'{col}_by_Failure_mean'] = (
                df_train['Failure_Type'].map(group_means).fillna(overall_mean)
            )
            df_train[f'{col}_Failure_diff'] = df_train[col] - df_train[f'{col}_by_Failure_mean']

            # Map test (unseen groups fall back to overall training mean)
            df_test[f'{col}_by_Failure_mean'] = (
                df_test['Failure_Type'].map(group_means).fillna(overall_mean)
            )
            df_test[f'{col}_Failure_diff'] = df_test[col] - df_test[f'{col}_by_Failure_mean']

    # --- Walls_Cross_Section group means ---------------------------------
    if 'Walls_Cross_Section' in df_train.columns:
        for col in ['fc_MPa', 'Total_Rho']:
            if col not in df_train.columns:
                continue
            group_means  = df_train.groupby('Walls_Cross_Section')[col].mean()
            overall_mean = df_train[col].mean()

            df_train[f'{col}_by_Section_mean'] = (
                df_train['Walls_Cross_Section'].map(group_means).fillna(overall_mean)
            )
            df_test[f'{col}_by_Section_mean'] = (
                df_test['Walls_Cross_Section'].map(group_means).fillna(overall_mean)
            )

    n_new = sum(1 for c in df_train.columns if '_by_Failure' in c or '_Failure_diff' in c or '_by_Section' in c)
    print(f"  Mean encoding added {n_new} features (train means only, safe for test set)")
    return df_train, df_test


def augment_data(X, y, method='noise', n_augment=1, noise_level=0.05):
    """
    Data augmentation for regression with scarce data.
    
    Methods:
    - 'noise': Add Gaussian noise to features
    - 'smogn': SMOGN-like oversampling for rare target values
    - 'mixup': Mixup augmentation (interpolate between samples)
    - 'bootstrap': Bootstrap resampling with noise
    """
    X_aug = X.copy()
    y_aug = y.copy()
    
    print(f"\n[Data Augmentation] Method: {method}, Original samples: {len(y)}")
    
    if method == 'noise':
        # Add Gaussian noise to features
        for _ in range(n_augment):
            noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0)
            X_noisy = X + noise
            X_aug = np.vstack([X_aug, X_noisy])
            y_aug = np.concatenate([y_aug, y])
    
    elif method == 'smogn':
        # Oversample rare target values (tails of distribution)
        # Identify rare samples (in tails)
        q_low, q_high = np.percentile(y, [10, 90])
        rare_mask = (y <= q_low) | (y >= q_high)
        X_rare = X[rare_mask]
        y_rare = y[rare_mask]
        
        for _ in range(n_augment * 2):  # More augmentation for rare samples
            if len(X_rare) > 1:
                # Interpolate between rare samples
                idx1, idx2 = np.random.choice(len(X_rare), 2, replace=False)
                alpha = np.random.uniform(0.3, 0.7)
                X_new = alpha * X_rare[idx1] + (1 - alpha) * X_rare[idx2]
                y_new = alpha * y_rare[idx1] + (1 - alpha) * y_rare[idx2]
                
                # Add small noise
                X_new += np.random.normal(0, noise_level * 0.5, X_new.shape) * np.std(X, axis=0)
                
                X_aug = np.vstack([X_aug, X_new.reshape(1, -1)])
                y_aug = np.append(y_aug, y_new)
    
    elif method == 'mixup':
        # Mixup: interpolate between random pairs
        for _ in range(n_augment * len(y) // 2):
            idx1, idx2 = np.random.choice(len(y), 2, replace=False)
            alpha = np.random.beta(0.4, 0.4)  # Beta distribution for mixup
            
            X_new = alpha * X[idx1] + (1 - alpha) * X[idx2]
            y_new = alpha * y[idx1] + (1 - alpha) * y[idx2]
            
            X_aug = np.vstack([X_aug, X_new.reshape(1, -1)])
            y_aug = np.append(y_aug, y_new)
    
    elif method == 'bootstrap':
        # Bootstrap with noise
        for _ in range(n_augment):
            # Sample with replacement
            idx = np.random.choice(len(y), len(y), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            # Add noise
            noise = np.random.normal(0, noise_level, X_boot.shape) * np.std(X, axis=0)
            X_boot = X_boot + noise
            
            X_aug = np.vstack([X_aug, X_boot])
            y_aug = np.concatenate([y_aug, y_boot])
    
    print(f"  Augmented samples: {len(y_aug)} ({len(y_aug) - len(y)} new)")
    return X_aug, y_aug


def compare_augmentation_strategies(X, y):
    """Compare different data augmentation strategies."""
    
    methods = ['none', 'noise', 'smogn', 'mixup', 'bootstrap']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING DATA AUGMENTATION STRATEGIES")
    print("="*70)
    
    for method in methods:
        print(f"\nTesting {method}...")
        
        if method == 'none':
            X_use, y_use = X, y
        else:
            X_use, y_use = augment_data(X, y, method=method, n_augment=1, noise_level=0.03)
        
        # Use repeated K-fold for more robust estimate
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # Repeated 5-fold CV
        scores = []
        for seed in range(3):  # 3 repetitions
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            fold_scores = cross_val_score(model, X_use, y_use, cv=kf, scoring='neg_mean_squared_error')
            scores.extend(fold_scores)
        
        rmse = np.sqrt(-np.mean(scores))
        rmse_std = np.sqrt(-np.array(scores)).std()
        
        results[method] = {'rmse': rmse, 'rmse_std': rmse_std}
        print(f"  CV RMSE: {rmse:.4f} (+/- {rmse_std:.4f})")
    
    best_method = min(results, key=lambda x: results[x]['rmse'])
    print(f"\n>> Best augmentation: {best_method}")
    return best_method, results


def small_data_cv(X, y, model, n_splits=5, n_repeats=5):
    """
    Cross-validation strategies optimized for small datasets.
    Uses Repeated K-Fold and optionally Leave-One-Out.
    """
    from sklearn.model_selection import RepeatedKFold, LeaveOneOut
    
    results = {}
    
    # Repeated K-Fold
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_squared_error')
    results['repeated_kfold'] = {
        'rmse': np.sqrt(-scores.mean()),
        'rmse_std': np.sqrt(-scores).std()
    }
    
    # Leave-One-Out (only if dataset is small enough)
    if len(y) <= 200:
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
        results['loo'] = {
            'rmse': np.sqrt(-scores.mean()),
            'rmse_std': np.sqrt(-scores).std()
        }
    
    return results


# ============================================================================
# 4. MISSING DATA IMPUTATION STRATEGIES
# ============================================================================

def _build_imputer(strategy, n_neighbors=5):
    """Return an unfitted imputer for the given strategy name."""
    if strategy == 'mean':
        return SimpleImputer(strategy='mean')
    elif strategy == 'median':
        return SimpleImputer(strategy='median')
    elif strategy == 'knn':
        return KNNImputer(n_neighbors=n_neighbors, weights='distance')
    elif strategy == 'iterative':
        return IterativeImputer(max_iter=20, random_state=42)
    elif strategy == 'rf':
        return IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1),
            max_iter=10, random_state=42,
        )
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")


def impute_data(df_train, df_test=None, strategy='knn', n_neighbors=5):
    """Fit imputer on df_train only, then transform both df_train and df_test.

    Fitting on training data only prevents test-set statistics (means, KNN
    neighbours, MICE chain) from leaking into the training set.
    The target column is excluded from imputation — it should have no
    missing values after load_and_clean_data drops those rows.

    Returns
    -------
    df_train_imp : DataFrame
    df_test_imp  : DataFrame  (None if df_test was None)
    imputer      : fitted imputer object
    """
    target_col = 'V_sqrt_fc'

    def _impute_df(df_in, imputer, fit=False):
        df_in = df_in.copy()
        numeric_cols     = [c for c in df_in.select_dtypes(include=[np.number]).columns
                            if c != target_col]
        non_numeric_cols = df_in.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric_data = df_in[non_numeric_cols].copy() if non_numeric_cols else None

        X = df_in[numeric_cols].values
        X_imp = imputer.fit_transform(X) if fit else imputer.transform(X)

        df_out = pd.DataFrame(X_imp, columns=numeric_cols, index=df_in.index)
        if target_col in df_in.columns:
            df_out[target_col] = df_in[target_col].values
        if non_numeric_data is not None:
            for col in non_numeric_cols:
                df_out[col] = non_numeric_data[col].values
        return df_out

    imputer = _build_imputer(strategy, n_neighbors)

    before = df_train.select_dtypes(include=[np.number]).isnull().sum().sum()
    df_train_imp = _impute_df(df_train, imputer, fit=True)
    after  = df_train_imp.select_dtypes(include=[np.number]).isnull().sum().sum()
    print(f"Imputed training data using '{strategy}'  ({before} → {after} missing values)")

    df_test_imp = None
    if df_test is not None:
        df_test_imp = _impute_df(df_test, imputer, fit=False)
        after_test = df_test_imp.select_dtypes(include=[np.number]).isnull().sum().sum()
        print(f"  Applied to test set ({after_test} missing values remaining)")

    return df_train_imp, df_test_imp, imputer


def compare_imputation_strategies(df_train, target_col='V_sqrt_fc'):
    """Compare imputation strategies via cross-validation using sklearn Pipelines.

    **No leakage**: each CV fold fits the imputer only on its own fold's training
    split, so test-fold statistics never influence the imputed values.  This is
    the correct procedure used by experienced data scientists.

    Parameters
    ----------
    df_train : DataFrame   — training data only (do NOT pass the full dataset)
    target_col : str

    Returns
    -------
    best_strategy : str
    results       : dict  {strategy: {'rmse': float, 'rmse_std': float}}
    """
    from sklearn.pipeline import Pipeline

    strategies = ['mean', 'median', 'knn', 'iterative']
    results = {}

    feature_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    X = df_train[feature_cols].values
    y = df_train[target_col].values

    print("\n" + "="*70)
    print("COMPARING IMPUTATION STRATEGIES  (Pipeline-based CV, no leakage)")
    print("="*70)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for strategy in strategies:
        print(f"\nTesting {strategy} imputation...")
        try:
            pipe = Pipeline([
                ('imputer', _build_imputer(strategy)),
                ('model',   RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
            ])
            scores = cross_val_score(pipe, X, y, cv=cv,
                                     scoring='neg_mean_squared_error', n_jobs=1)
            rmse     = float(np.sqrt(-scores.mean()))
            rmse_std = float(np.sqrt(-scores).std())
            results[strategy] = {'rmse': rmse, 'rmse_std': rmse_std}
            print(f"  CV RMSE: {rmse:.4f} (+/- {rmse_std:.4f})")
        except Exception as e:
            print(f"  Failed: {e}")
            results[strategy] = {'rmse': np.inf, 'rmse_std': 0}

    best_strategy = min(results, key=lambda x: results[x]['rmse'])
    print(f"\n>> Best imputation strategy: {best_strategy}")
    return best_strategy, results


# ============================================================================
# 5. SKEWNESS & OUTLIER HANDLING
# ============================================================================

from scipy import stats
from scipy.stats import skew, boxcox, yeojohnson

def analyze_skewness(df, threshold=1.0):
    """Analyze skewness of numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewness = {}
    
    print("\n" + "="*70)
    print("SKEWNESS ANALYSIS")
    print("="*70)
    print(f"{'Feature':<25} {'Skewness':>10} {'Status':<15}")
    print("-"*50)
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 10:
            sk = skew(data)
            skewness[col] = sk
            status = "HIGHLY SKEWED" if abs(sk) > threshold else "OK"
            if abs(sk) > threshold:
                print(f"{col:<25} {sk:>10.3f} {status:<15}")
    
    skewed_features = [k for k, v in skewness.items() if abs(v) > threshold]
    print(f"\n>> {len(skewed_features)} highly skewed features (|skew| > {threshold})")
    return skewness, skewed_features


def transform_skewed_features(df_train, df_test=None, method='log', skewed_cols=None, threshold=1.0):
    """Transform skewed features — fit parameters on df_train, apply to both.

    All shift values, Box-Cox / Yeo-Johnson lambdas, and power exponents are
    estimated from df_train only.  df_test receives the same transformation
    using the training-set parameters, preventing test statistics from leaking.

    Methods: 'log', 'sqrt', 'boxcox', 'yeojohnson', 'reciprocal', 'power'
    """
    df_train = df_train.copy()
    if df_test is not None:
        df_test = df_test.copy()

    if skewed_cols is None:
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        skewed_cols = [
            col for col in numeric_cols
            if len(df_train[col].dropna()) > 10 and abs(skew(df_train[col].dropna())) > threshold
        ]

    transformed_cols = []
    transform_info   = {}

    def _apply(df, col, info):
        """Apply a pre-fitted transform described by *info* to *df*."""
        m = info['method']
        if m == 'log':
            df[col] = np.log1p(df[col] + info['shift'])
        elif m == 'sqrt':
            df[col] = np.sqrt(df[col] + info['shift'])
        elif m == 'boxcox':
            from scipy.special import boxcox1p
            mask = ~df[col].isnull()
            df.loc[mask, col] = boxcox(df.loc[mask, col].values + info['shift'],
                                       lmbda=info['lambda'])
        elif m == 'yeojohnson':
            from scipy.stats import yeojohnson as _yj
            mask = ~df[col].isnull()
            # When lmbda is supplied, yeojohnson() returns only the
            # transformed array (not a 2-tuple), so assign directly.
            df.loc[mask, col] = _yj(df.loc[mask, col].values, lmbda=info['lambda'])
        elif m == 'reciprocal':
            df[col] = 1 / (df[col] + 1e-10)
        elif m == 'power':
            df[col] = np.power(df[col] + info['shift'], info['exponent'])
        return df

    for col in skewed_cols:
        try:
            if method == 'log':
                shift = max(0, -df_train[col].min() + 1) if df_train[col].min() <= 0 else 0
                transform_info[col] = {'method': 'log', 'shift': shift}

            elif method == 'sqrt':
                shift = abs(df_train[col].min()) if df_train[col].min() < 0 else 0
                transform_info[col] = {'method': 'sqrt', 'shift': shift}

            elif method == 'boxcox':
                shift = abs(df_train[col].min()) + 1 if df_train[col].min() <= 0 else 0
                mask  = ~df_train[col].isnull()
                _, lmbda = boxcox(df_train.loc[mask, col].values + shift)
                transform_info[col] = {'method': 'boxcox', 'lambda': lmbda, 'shift': shift}

            elif method == 'yeojohnson':
                mask = ~df_train[col].isnull()
                _, lmbda = yeojohnson(df_train.loc[mask, col].values)
                transform_info[col] = {'method': 'yeojohnson', 'lambda': lmbda}

            elif method == 'reciprocal':
                transform_info[col] = {'method': 'reciprocal'}

            elif method == 'power':
                shift = abs(df_train[col].min()) if df_train[col].min() < 0 else 0
                transform_info[col] = {'method': 'power', 'shift': shift, 'exponent': 0.25}

            # Apply to train
            df_train = _apply(df_train, col, transform_info[col])
            # Apply same fitted params to test
            if df_test is not None and col in df_test.columns:
                df_test = _apply(df_test, col, transform_info[col])

            transformed_cols.append(col)
        except Exception as e:
            print(f"  Warning: Could not transform {col}: {e}")

    print(f"\nTransformed {len(transformed_cols)} features using '{method}' (train params applied to test)")
    if df_test is not None:
        return df_train, df_test, transform_info
    return df_train, None, transform_info


def handle_outliers(df_train, df_test=None, method='clip', threshold=3.0, cols=None):
    """Fit outlier bounds on df_train only, then clip both df_train and df_test.

    All statistics (mean/std, IQR quartiles, percentiles, MAD) are computed
    exclusively from df_train, preventing any test-set information from leaking
    into the training distribution.

    Methods:
    - 'clip': Clip to threshold × std from mean
    - 'iqr': Q1 - 1.5*IQR … Q3 + 1.5*IQR
    - 'winsorize': 1st/99th percentile of training data
    - 'robust_zscore': median ± threshold × MAD/0.6745
    - 'log_clip': log-space clip then inverse-transform
    - 'zscore_remove': marks outliers only (no rows dropped)

    Returns
    -------
    df_train_out : DataFrame
    df_test_out  : DataFrame or None
    outlier_info : dict
    """
    df_train = df_train.copy()
    df_test  = df_test.copy() if df_test is not None else None

    if cols is None:
        cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info = {}
    bounds = {}  # store per-column clip bounds for applying to test

    for col in cols:
        data = df_train[col].dropna()
        if len(data) < 10:
            continue

        original_min, original_max = data.min(), data.max()
        n_outliers = 0
        lower = upper = None  # will be set by each method

        if method == 'clip':
            mean, std = data.mean(), data.std()
            lower = mean - threshold * std
            upper = mean + threshold * std

        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

        elif method == 'winsorize':
            lower = data.quantile(0.01)
            upper = data.quantile(0.99)

        elif method == 'robust_zscore':
            median = data.median()
            mad = np.median(np.abs(data - median))
            if mad == 0:
                mad = 1e-10
            lower = median - threshold * mad / 0.6745
            upper = median + threshold * mad / 0.6745

        elif method == 'log_clip':
            min_val = data.min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            log_data = np.log1p(data + shift)
            mean_l, std_l = log_data.mean(), log_data.std()
            log_lower = mean_l - threshold * std_l
            log_upper = mean_l + threshold * std_l
            # store in log-space
            bounds[col] = {'method': 'log_clip', 'shift': shift,
                           'log_lower': log_lower, 'log_upper': log_upper}
            # apply to train
            log_col_tr = np.log1p(df_train[col] + shift)
            n_outliers = ((log_col_tr < log_lower) | (log_col_tr > log_upper)).sum()
            df_train[col] = np.expm1(log_col_tr.clip(log_lower, log_upper)) - shift
            # apply to test
            if df_test is not None and col in df_test.columns:
                log_col_te = np.log1p(df_test[col] + shift)
                df_test[col] = np.expm1(log_col_te.clip(log_lower, log_upper)) - shift
            if n_outliers > 0:
                outlier_info[col] = {
                    'method': method, 'n_outliers': int(n_outliers),
                    'original_range': (original_min, original_max),
                    'new_range': (df_train[col].min(), df_train[col].max()),
                }
            continue  # already applied, skip generic clip below

        elif method == 'zscore_remove':
            z_scores = np.abs(stats.zscore(data))
            n_outliers = int((z_scores >= threshold).sum())
            outlier_info[col] = {'method': method, 'n_outliers': n_outliers}
            continue  # no clipping for this method

        # Generic clip path (clip / iqr / winsorize / robust_zscore)
        if lower is not None and upper is not None:
            n_outliers = int(((df_train[col] < lower) | (df_train[col] > upper)).sum())
            df_train[col] = df_train[col].clip(lower, upper)
            bounds[col] = {'lower': lower, 'upper': upper}
            if df_test is not None and col in df_test.columns:
                df_test[col] = df_test[col].clip(lower, upper)

        if n_outliers > 0:
            outlier_info[col] = {
                'method': method,
                'n_outliers': n_outliers,
                'original_range': (original_min, original_max),
                'new_range': (df_train[col].min(), df_train[col].max()),
            }

    total_outliers = sum(info.get('n_outliers', 0) for info in outlier_info.values())
    print(f"\nHandled outliers using '{method}' method (bounds fitted on train only)")
    print(f"  - Total outliers treated: {total_outliers}")
    return df_train, df_test, outlier_info


def compare_transformations(df, target_col='V_sqrt_fc'):
    """Compare different transformation strategies."""
    
    transform_methods = ['log', 'sqrt', 'boxcox', 'yeojohnson']
    outlier_methods = ['clip', 'iqr', 'winsorize', 'robust_zscore']
    
    results = []
    
    print("\n" + "="*70)
    print("COMPARING TRANSFORMATION STRATEGIES")
    print("="*70)
    
    # Get skewed columns (excluding target)
    _, skewed_cols = analyze_skewness(df)
    skewed_cols = [c for c in skewed_cols if c != target_col]
    
    # Baseline (no transformation)
    print("\n--- Baseline (no transformation) ---")
    df_baseline = df.copy()
    feature_cols = [c for c in df_baseline.select_dtypes(include=[np.number]).columns 
                   if c != target_col]
    X = df_baseline[feature_cols].values
    y = df_baseline[target_col].values
    
    # Handle any remaining NaNs
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    baseline_rmse = np.sqrt(-scores.mean())
    print(f"  CV RMSE: {baseline_rmse:.4f}")
    results.append({'method': 'baseline', 'rmse': baseline_rmse})
    
    # Test transform methods
    print("\n--- Transformation Methods ---")
    for method in transform_methods:
        try:
            df_transformed, _, _ = transform_skewed_features(df.copy(), method=method, skewed_cols=skewed_cols)
            
            feature_cols = [c for c in df_transformed.select_dtypes(include=[np.number]).columns 
                          if c != target_col]
            X = df_transformed[feature_cols].values
            y = df_transformed[target_col].values
            
            X = imputer.fit_transform(X)
            
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            rmse = np.sqrt(-scores.mean())
            print(f"  {method:<15} CV RMSE: {rmse:.4f}")
            results.append({'method': f'transform_{method}', 'rmse': rmse})
        except Exception as e:
            print(f"  {method:<15} Failed: {e}")
    
    # Test outlier methods
    print("\n--- Outlier Handling Methods ---")
    for method in outlier_methods:
        try:
            df_cleaned, _, _ = handle_outliers(df.copy(), method=method)
            
            feature_cols = [c for c in df_cleaned.select_dtypes(include=[np.number]).columns 
                          if c != target_col]
            X = df_cleaned[feature_cols].values
            y = df_cleaned[target_col].values
            
            X = imputer.fit_transform(X)
            
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            rmse = np.sqrt(-scores.mean())
            print(f"  {method:<15} CV RMSE: {rmse:.4f}")
            results.append({'method': f'outlier_{method}', 'rmse': rmse})
        except Exception as e:
            print(f"  {method:<15} Failed: {e}")
    
    # Combined: best transform + best outlier handling
    print("\n--- Combined (transform + outlier) ---")
    try:
        df_combined, _, _ = transform_skewed_features(df.copy(), method='yeojohnson', skewed_cols=skewed_cols)
        df_combined, _, _ = handle_outliers(df_combined, method='iqr')
        
        feature_cols = [c for c in df_combined.select_dtypes(include=[np.number]).columns 
                      if c != target_col]
        X = df_combined[feature_cols].values
        y = df_combined[target_col].values
        
        X = imputer.fit_transform(X)
        
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        print(f"  yeojohnson+iqr  CV RMSE: {rmse:.4f}")
        results.append({'method': 'combined_yeo_iqr', 'rmse': rmse})
    except Exception as e:
        print(f"  Combined Failed: {e}")
    
    # Find best
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['rmse'].idxmin()]
    print(f"\n>> Best strategy: {best['method']} (RMSE: {best['rmse']:.4f})")
    
    return results_df


def plot_distributions(df, cols=None, save_path='distributions.png'):
    """Plot before/after distribution comparisons."""
    if cols is None:
        # Get most skewed columns
        skewness, skewed_cols = analyze_skewness(df, threshold=0.5)
        cols = skewed_cols[:8]  # Top 8
    
    n_cols = len(cols)
    if n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_cols, 3, figsize=(12, 3*n_cols))
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(cols):
        data = df[col].dropna()
        
        # Original
        axes[i, 0].hist(data, bins=30, edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'{col}\nOriginal (skew={skew(data):.2f})')
        
        # Log transform
        min_val = data.min()
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        log_data = np.log1p(data + shift)
        axes[i, 1].hist(log_data, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[i, 1].set_title(f'Log Transform (skew={skew(log_data):.2f})')
        
        # Yeo-Johnson
        yj_data, _ = yeojohnson(data)
        axes[i, 2].hist(yj_data, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[i, 2].set_title(f'Yeo-Johnson (skew={skew(yj_data):.2f})')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nDistribution plots saved to {save_path}")


# ============================================================================
# 6. NORMALIZATION STRATEGIES
# ============================================================================

def normalize_data(X, method='standard'):
    """
    Apply different normalization methods.
    
    Methods:
    - 'standard': StandardScaler (z-score normalization)
    - 'minmax': MinMaxScaler (0-1 scaling)
    - 'robust': RobustScaler (median and IQR based)
    - 'quantile': QuantileTransformer (uniform/gaussian output)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def compare_normalization_methods(X, y, random_state: int = 42):
    """Compare normalization methods."""
    methods = ['standard', 'minmax', 'robust', 'quantile']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING NORMALIZATION METHODS")
    print("="*70)
    
    for method in methods:
        X_scaled, _ = normalize_data(X, method=method)
        
        model = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        
        results[method] = rmse
        print(f"{method:12s} CV RMSE: {rmse:.4f}")
    
    best_method = min(results, key=results.get)
    print(f"\n>> Best normalization: {best_method}")
    return best_method, results


# ============================================================================
# 6. BASELINE MODELS: XGBoost & Random Forest
# ============================================================================

def train_random_forest(X_train, X_test, y_train, y_test, **kwargs):
    """Train and evaluate Random Forest."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    _train_r2 = r2_score(y_train, y_pred_train)
    _test_r2  = r2_score(y_test,  y_pred_test)
    _k = X_train.shape[1] + 1  # features + intercept
    _train_aic, _train_bic = compute_aic_bic(y_train, y_pred_train, _k)
    _test_aic,  _test_bic  = compute_aic_bic(y_test,  y_pred_test,  _k)
    metrics = {
        'train_rmse':    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse':     np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'train_mae':     mean_absolute_error(y_train, y_pred_train),
        'test_mae':      mean_absolute_error(y_test,  y_pred_test),
        'train_r2':      _train_r2,
        'test_r2':       _test_r2,
        'train_adj_r2':  adjusted_r2(_train_r2, len(y_train), X_train.shape[1]),
        'test_adj_r2':   adjusted_r2(_test_r2,  len(y_test),  X_test.shape[1]),
        'train_aic':     _train_aic,
        'test_aic':      _test_aic,
        'train_bic':     _train_bic,
        'test_bic':      _test_bic,
    }

    return model, metrics, y_pred_test


def train_xgboost(X_train, X_test, y_train, y_test, **kwargs):
    """Train and evaluate XGBoost."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    _train_r2 = r2_score(y_train, y_pred_train)
    _test_r2  = r2_score(y_test,  y_pred_test)
    _k = X_train.shape[1] + 1
    _train_aic, _train_bic = compute_aic_bic(y_train, y_pred_train, _k)
    _test_aic,  _test_bic  = compute_aic_bic(y_test,  y_pred_test,  _k)
    metrics = {
        'train_rmse':    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse':     np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'train_mae':     mean_absolute_error(y_train, y_pred_train),
        'test_mae':      mean_absolute_error(y_test,  y_pred_test),
        'train_r2':      _train_r2,
        'test_r2':       _test_r2,
        'train_adj_r2':  adjusted_r2(_train_r2, len(y_train), X_train.shape[1]),
        'test_adj_r2':   adjusted_r2(_test_r2,  len(y_test),  X_test.shape[1]),
        'train_aic':     _train_aic,
        'test_aic':      _test_aic,
        'train_bic':     _train_bic,
        'test_bic':      _test_bic,
    }

    return model, metrics, y_pred_test


# ============================================================================
# 7. NEURAL ADDITIVE MODEL (NAM)
# ============================================================================

def train_nam(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    hidden_units=[64, 32],
    epochs=300,
    lr=0.001,
    batch_size=32,
    random_state: int = 42,
    n_cycles: int = 3,
    mlp_alpha: float = 0.05,   # L2 penalty on MLP weights; higher → smoother shapes, less overfit
):
    """
    Train Neural Additive Model (NAM) with backfitting.

    NAM: y = bias + Σ_i f_i(x_i)
    where each f_i is a small MLP trained on the partial residual
    y - Σ_{j≠i} f_j(x_j).  Running multiple backfitting cycles lets
    networks learn complementary, non-redundant shape functions instead
    of all competing to explain the same variance independently.

    Improvements over the original version
    ---------------------------------------
    * Backfitting (n_cycles): each MLP fits the partial residual, not raw y
    * hidden_units / epochs parameters are now actually used
    * RidgeCV replaces fixed Ridge(alpha=1.0) — alpha is cross-validated
    * n_iter_no_change raised to 20 for more patient early stopping
    * Per-cycle RMSE is tracked and returned for the training-curve plot
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import RidgeCV

    n_features = X_train.shape[1]

    # Current contribution of each feature network (updated each cycle)
    contributions_train = np.zeros((X_train.shape[0], n_features))
    contributions_test  = np.zeros((X_test.shape[0],  n_features))
    feature_models      = [None] * n_features

    train_losses, val_losses = [], []

    print(f"\nTraining NAM with {n_features} feature networks "
          f"({n_cycles} backfitting cycles)...")

    for cycle in range(n_cycles):
        print(f"\n  [Cycle {cycle + 1}/{n_cycles}]")

        for i in range(n_features):
            X_i_train = X_train[:, i:i+1]
            X_i_test  = X_test[:,  i:i+1]

            # Partial residual: y minus every other feature's contribution
            other_contrib = contributions_train.sum(axis=1) - contributions_train[:, i]
            partial_residual = y_train - other_contrib

            mlp = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_units),
                activation='relu',
                solver='adam',
                alpha=mlp_alpha,
                max_iter=epochs,
                random_state=random_state + i + cycle * n_features,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
            )
            mlp.fit(X_i_train, partial_residual)

            contributions_train[:, i] = mlp.predict(X_i_train)
            contributions_test[:,  i] = mlp.predict(X_i_test)
            feature_models[i] = mlp

            if (i + 1) % 10 == 0 or (i + 1) == n_features:
                print(f"    Trained {i + 1}/{n_features} feature networks...")

        # Track raw-sum RMSE at the end of each cycle (before Ridge)
        cycle_train_rmse = np.sqrt(mean_squared_error(y_train, contributions_train.sum(axis=1)))
        cycle_val_rmse   = np.sqrt(mean_squared_error(y_test,  contributions_test.sum(axis=1)))
        train_losses.append(cycle_train_rmse)
        val_losses.append(cycle_val_rmse)
        print(f"    Cycle {cycle + 1} RMSE — Train: {cycle_train_rmse:.4f}, Val: {cycle_val_rmse:.4f}")

    # Final combination with cross-validated Ridge
    combiner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    combiner.fit(contributions_train, y_train)
    print(f"\n  RidgeCV selected alpha = {combiner.alpha_:.4g}")

    y_pred_train = combiner.predict(contributions_train)
    y_pred_test  = combiner.predict(contributions_test)

    contributions = [
        contributions_test[:, i:i+1] * combiner.coef_[i]
        for i in range(n_features)
    ]

    _train_r2 = r2_score(y_train, y_pred_train)
    _test_r2  = r2_score(y_test,  y_pred_test)
    _k = X_train.shape[1] + 1
    _train_aic, _train_bic = compute_aic_bic(y_train, y_pred_train, _k)
    _test_aic,  _test_bic  = compute_aic_bic(y_test,  y_pred_test,  _k)
    metrics = {
        'train_rmse':    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse':     np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'train_mae':     mean_absolute_error(y_train, y_pred_train),
        'test_mae':      mean_absolute_error(y_test,  y_pred_test),
        'train_r2':      _train_r2,
        'test_r2':       _test_r2,
        'train_adj_r2':  adjusted_r2(_train_r2, len(y_train), X_train.shape[1]),
        'test_adj_r2':   adjusted_r2(_test_r2,  len(y_test),  X_test.shape[1]),
        'train_aic':     _train_aic,
        'test_aic':      _test_aic,
        'train_bic':     _train_bic,
        'test_bic':      _test_bic,
    }

    class SklearnNAM:
        def __init__(self, feature_models, combiner):
            self.feature_models = feature_models
            self.combiner = combiner

        def predict(self, X):
            contribs = np.zeros((X.shape[0], len(self.feature_models)))
            for i, mdl in enumerate(self.feature_models):
                contribs[:, i] = mdl.predict(X[:, i:i+1])
            return self.combiner.predict(contribs)

    model = SklearnNAM(feature_models, combiner)
    return model, metrics, y_pred_test, contributions, (train_losses, val_losses)


# ============================================================================
# 8. EXPLAINABLE BOOSTING MACHINE (EBM)
# ============================================================================

def train_ebm(X_train, X_test, y_train, y_test, feature_names, random_state: int = 42):
    """
    Train Explainable Boosting Machine (EBM).
    
    EBM is a glass-box model that captures:
    - Main effects: y = Σ fᵢ(xᵢ)  
    - Pairwise interactions: + Σ fᵢⱼ(xᵢ, xⱼ)
    
    Best of both worlds: interpretable + captures interactions.
    """
    from interpret.glassbox import ExplainableBoostingRegressor
    
    print("\nTraining Explainable Boosting Machine (EBM)...")
    print("  (captures main effects + pairwise interactions)")
    
    # EBM with tuned parameters for this dataset size
    ebm = ExplainableBoostingRegressor(
        feature_names=feature_names,
        max_bins=256,              # Granularity of feature binning
        max_interaction_bins=32,   # Bins for interaction terms
        interactions=5,            # Reduced from 15 — fewer pairs prevents memorising training noise
        learning_rate=0.01,
        min_samples_leaf=5,        # Increased from 3 — tighter regularisation for small dataset
        max_leaves=2,              # Reduced from 3 — shallower interaction surfaces reduce overfit
        n_jobs=-1,
        random_state=random_state
    )
    
    ebm.fit(X_train, y_train)
    
    y_pred_train = ebm.predict(X_train)
    y_pred_test = ebm.predict(X_test)
    
    _train_r2 = r2_score(y_train, y_pred_train)
    _test_r2  = r2_score(y_test,  y_pred_test)
    _k = X_train.shape[1] + 1
    _train_aic, _train_bic = compute_aic_bic(y_train, y_pred_train, _k)
    _test_aic,  _test_bic  = compute_aic_bic(y_test,  y_pred_test,  _k)
    metrics = {
        'train_rmse':    np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse':     np.sqrt(mean_squared_error(y_test,  y_pred_test)),
        'train_mae':     mean_absolute_error(y_train, y_pred_train),
        'test_mae':      mean_absolute_error(y_test,  y_pred_test),
        'train_r2':      _train_r2,
        'test_r2':       _test_r2,
        'train_adj_r2':  adjusted_r2(_train_r2, len(y_train), X_train.shape[1]),
        'test_adj_r2':   adjusted_r2(_test_r2,  len(y_test),  X_test.shape[1]),
        'train_aic':     _train_aic,
        'test_aic':      _test_aic,
        'train_bic':     _train_bic,
        'test_bic':      _test_bic,
    }

    # Get feature importances
    importances = ebm.term_importances()
    term_names = ebm.term_names_
    
    print(f"\n  Top 10 important terms (features + interactions):")
    importance_order = np.argsort(importances)[::-1]
    for i in range(min(10, len(importance_order))):
        idx = importance_order[i]
        print(f"    {i+1}. {term_names[idx]}: {importances[idx]:.4f}")
    
    return ebm, metrics, y_pred_test


def plot_ebm_interpretability(
    ebm,
    feature_names,
    save_path='ebm_interpretability.png',
    scaler=None,
    all_feature_names=None,
):
    """Plot EBM feature shape functions and interactions.

    If *scaler* and *all_feature_names* are provided, the bin boundary values
    on the x-axis are inverse-transformed to original physical units.
    """
    ebm_global = ebm.explain_global()
    importances = ebm.term_importances()
    term_names  = ebm.term_names_

    importance_order = np.argsort(importances)[::-1]

    n_plots = min(12, len(term_names))
    n_cols  = 3
    n_rows  = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    # Build a lookup: feature name → its index in the scaler's feature space
    name_to_full_idx = {}
    if scaler is not None and all_feature_names is not None:
        name_to_full_idx = {name: i for i, name in enumerate(all_feature_names)}

    def _inv_1d(vals, feat_name):
        """Inverse-transform a 1-D array of bin values for one feature."""
        if not name_to_full_idx or feat_name not in name_to_full_idx:
            return vals
        col_idx = name_to_full_idx[feat_name]
        n_all   = len(all_feature_names)
        X_tmp   = np.zeros((len(vals), n_all))
        X_tmp[:, col_idx] = vals
        return scaler.inverse_transform(X_tmp)[:, col_idx]

    for i in range(n_plots):
        idx       = importance_order[i]
        term_name = term_names[idx]
        data      = ebm_global.data(idx)
        ax        = axes[i]

        if ' x ' in term_name:
            # Interaction term — show as text summary
            ax.text(0.5, 0.5,
                    f"Interaction:\n{term_name}\n\nImportance: {importances[idx]:.4f}",
                    ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        else:
            # Main effect — plot shape function with original-unit x-axis
            if 'names' in data and 'scores' in data:
                x_vals = np.atleast_1d(data['names']).astype(float)
                y_vals = np.atleast_1d(data['scores'])
                n = min(len(x_vals), len(y_vals))
                if n == 0:
                    ax.text(0.5, 0.5, f'No data\n{term_name}',
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    x_vals, y_vals = x_vals[:n], y_vals[:n]
                    # Inverse-transform to original units
                    x_orig = _inv_1d(x_vals, term_name)

                    ax.plot(x_orig, y_vals, 'b-', linewidth=2)
                    ax.fill_between(x_orig, y_vals, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax.set_xlabel(term_name, fontsize=8)

                    # Tick labels: 5 evenly-spaced x ticks in original units
                    tick_pos = np.linspace(0, n - 1, min(5, n), dtype=int)
                    ax.set_xticks(x_orig[tick_pos])
                    ax.set_xticklabels(
                        [f'{x_orig[j]:.2g}' for j in tick_pos],
                        fontsize=7, rotation=45,
                    )

        ax.set_title(f'{term_name}\n(imp: {importances[idx]:.3f})', fontsize=9)
        ax.tick_params(axis='both', labelsize=7)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('EBM Shape Functions — x in original feature units', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nEBM interpretability plots saved to {save_path}")


def _inverse_transform_cols(X_sel, scaler, all_feature_names, sel_feature_names):
    """Inverse-transform a subset of scaled columns back to original units.

    Works for any sklearn scaler (StandardScaler, RobustScaler, etc.) because
    they all transform columns independently — the values in other columns are
    irrelevant and we can fill them with zeros.
    """
    n_samples = X_sel.shape[0]
    n_all     = len(all_feature_names)

    name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
    sel_indices = [name_to_idx[f] for f in sel_feature_names if f in name_to_idx]

    if len(sel_indices) != len(sel_feature_names):
        return X_sel  # fallback: return as-is if names don't match

    X_full = np.zeros((n_samples, n_all))
    X_full[:, sel_indices] = X_sel
    X_full_inv = scaler.inverse_transform(X_full)
    return X_full_inv[:, sel_indices]


def plot_nam_interpretability(
    X_for_plot,
    contributions,
    feature_names,
    save_path='nam_interpretability.png',
    scaler=None,
    all_feature_names=None,
):
    """Plot NAM feature shape functions.

    If *scaler* and *all_feature_names* are provided, the x-axis is shown in
    original physical units; otherwise scaled values are used.
    """
    if scaler is not None and all_feature_names is not None:
        X_for_plot = _inverse_transform_cols(
            X_for_plot, scaler, all_feature_names, feature_names
        )

    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        if i < len(contributions):
            x_vals = X_for_plot[:, i]
            y_vals = contributions[i].flatten()

            sort_idx = np.argsort(x_vals)
            ax.scatter(x_vals[sort_idx], y_vals[sort_idx], alpha=0.3, s=10)
            ax.set_xlabel(name, fontsize=8)
            ax.set_ylabel('f(x)', fontsize=8)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', labelsize=7)

    for i in range(len(feature_names), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('NAM Feature Shape Functions (original feature units)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nNAM interpretability plots saved to {save_path}")


# ============================================================================
# 8. MODEL COMPARISON & VISUALIZATION
# ============================================================================

def compare_models(y_test, predictions_dict, save_path='model_comparison.png'):
    """Compare model predictions."""
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(5*len(predictions_dict), 5))
    if len(predictions_dict) == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        ax.set_xlabel('Actual V/sqrt(fc)')
        ax.set_ylabel('Predicted V/sqrt(fc)')
        ax.set_title(f'{name}\nRMSE={rmse:.3f}, R²={r2:.3f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nModel comparison plots saved to {save_path}")


def plot_feature_importance(model, feature_names, model_name, save_path='feature_importance.png'):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFeature importance plot saved to {save_path}")


def print_feature_importance_summary(rf_model, xgb_model, contributions, feature_names,
                                      nam_feature_names=None, ebm_model=None, top_k=15):
    """Print and save top-k important features for RF, XGBoost, NAM, and EBM.

    *feature_names*     — names aligned with RF and XGBoost (full pre-selection set).
    *nam_feature_names* — names aligned with NAM contributions and EBM terms.
                          Defaults to *feature_names* if not supplied (backward compat).
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE BY MODEL (top-%d)" % top_k)
    print("="*70)

    if nam_feature_names is None:
        nam_feature_names = feature_names

    # Random Forest
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    rf_top = [feature_names[i] for i in idx]
    print("\n  Random Forest (trained on full %d-feature set):" % len(feature_names))
    for i, name in enumerate(rf_top, 1):
        print("    %2d. %s (%.4f)" % (i, name, imp[idx[i-1]]))

    # XGBoost
    imp = xgb_model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    xgb_top = [feature_names[i] for i in idx]
    print("\n  XGBoost (trained on full %d-feature set):" % len(feature_names))
    for i, name in enumerate(xgb_top, 1):
        print("    %2d. %s (%.4f)" % (i, name, imp[idx[i-1]]))

    # NAM — contributions are aligned with nam_feature_names (selected set)
    nam_imp = np.array([np.abs(contrib).mean() for contrib in contributions])
    idx = np.argsort(nam_imp)[::-1][:top_k]
    nam_top = [nam_feature_names[i] for i in idx]
    print("\n  NAM — mean |contribution|  (trained on selected %d-feature set):" % len(nam_feature_names))
    for i, name in enumerate(nam_top, 1):
        print("    %2d. %s (%.4f)" % (i, name, nam_imp[idx[i-1]]))

    # EBM: term importances (main-effect terms only for direct feature comparison)
    if ebm_model is not None and hasattr(ebm_model, 'term_importances'):
        term_imp   = ebm_model.term_importances()
        term_names = ebm_model.term_names_
        main_effects = [
            (name.strip(), term_imp[i])
            for i, name in enumerate(term_names)
            if ' x ' not in name and name.strip() in nam_feature_names
        ]
        main_effects.sort(key=lambda x: -x[1])
        ebm_top = [name for name, _ in main_effects[:top_k]]
        print("\n  EBM — main-effect terms  (trained on selected %d-feature set):" % len(nam_feature_names))
        for i, (name, val) in enumerate(main_effects[:top_k], 1):
            print("    %2d. %s (%.4f)" % (i, name, val))
    else:
        ebm_top = []
    
    # Save to CSV (one row per model, columns = rank 1..top_k)
    max_len = max(len(rf_top), len(xgb_top), len(nam_top), len(ebm_top) if ebm_top else 0)
    rows = []
    for label, top in [('Random Forest', rf_top), ('XGBoost', xgb_top), ('NAM', nam_top), ('EBM', ebm_top)]:
        row = {'Model': label}
        for r in range(max_len):
            row['Rank_%d' % (r+1)] = top[r] if r < len(top) else ''
        rows.append(row)
    pd.DataFrame(rows).to_csv('feature_importance_by_model.csv', index=False)
    print("\n  Saved to feature_importance_by_model.csv")


def plot_training_curves(train_losses, val_losses, save_path='nam_training.png'):
    """Plot NAM training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('NAM Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 9. FEATURE SELECTION
# ============================================================================

# Structural engineering parameters that must never be dropped regardless of
# what the statistical filters decide.
PROTECTED_FEATURES = [
    'fc_MPa', 'Axial_Load_Ratio', 'Aspect_Ratio', 'Shear_Span_Ratio',
    'rho_t_pct', 'rho_l_pct', 'fy_MPa', 'fybl_MPa', 'fyl_MPa',
    'Thickness_mm', 'Length_mm', 'Height_mm',
]


def _remove_low_variance(X, feature_names, threshold=0.01, protected=None):
    """Step 1a: Drop features whose variance falls below *threshold*.

    Protected features are never dropped regardless of variance.
    Returns the filtered array, kept feature names, and a boolean mask.
    """
    from sklearn.feature_selection import VarianceThreshold
    protected = set(protected or [])

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    mask = selector.get_support().copy()

    # Reinstate any protected feature the filter wanted to drop
    for i, name in enumerate(feature_names):
        if name in protected:
            mask[i] = True

    kept    = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    dropped = [feature_names[i] for i in range(len(feature_names)) if not mask[i]]
    if dropped:
        print(f"    Dropped {len(dropped)} near-zero-variance features: {dropped}")
    else:
        print(f"    No near-zero-variance features found.")
    return X[:, mask], kept, mask


def _remove_correlated(X, feature_names, threshold=0.92, importances=None, protected=None):
    """Step 1b: For each correlated pair (|r| > threshold) drop the less important one.

    *importances* is a 1-D array aligned with *feature_names* used to decide
    which member of a correlated pair to keep.  Protected features are never
    dropped even if they are the less-important member.
    Returns the filtered array, kept names, and a boolean mask.
    """
    protected = set(protected or [])
    n   = len(feature_names)
    imp = importances if importances is not None else np.ones(n)
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)

    drop = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if corr[i, j] > threshold:
                i_prot = feature_names[i] in protected
                j_prot = feature_names[j] in protected
                if j_prot and not i_prot:
                    drop.add(i)
                    break                     # i is eliminated; stop its j-loop
                elif not j_prot:
                    if imp[i] >= imp[j]:
                        drop.add(j)           # keep i, drop j
                    else:
                        drop.add(i)
                        break                 # i is eliminated; move to next i
                # if both protected: keep both

    mask    = np.array([i not in drop for i in range(n)])
    kept    = [feature_names[i] for i in range(n) if mask[i]]
    dropped = [feature_names[i] for i in range(n) if not mask[i]]
    if dropped:
        preview = dropped[:10]
        suffix  = f" …+{len(dropped)-10} more" if len(dropped) > 10 else ""
        print(f"    Dropped {len(dropped)} highly-correlated features: {preview}{suffix}")
    else:
        print(f"    No highly-correlated features found.")
    return X[:, mask], kept, mask


def _rfecv_cutoff(X, y, feature_names, protected=None, random_state=42):
    """Step 3: Recursive Feature Elimination with Cross-Validation.

    Finds the number of features that maximises CV R² on the training set.
    Protected features are pinned back into the support if RFECV would have
    excluded them.
    Returns the filtered array, kept names, boolean mask, and the fitted RFECV.
    """
    from sklearn.feature_selection import RFECV

    protected = set(protected or [])
    estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring='r2',
        min_features_to_select=5,
        n_jobs=-1,
    )
    rfecv.fit(X, y)

    support = rfecv.support_.copy()
    for i, name in enumerate(feature_names):
        if name in protected:
            support[i] = True

    kept = [feature_names[i] for i in range(len(feature_names)) if support[i]]
    n_auto = int(rfecv.support_.sum())
    n_pinned = len(kept) - n_auto
    print(f"    RFECV optimal: {n_auto} features  |  {n_pinned} extra pinned by domain rules  →  {len(kept)} total")
    return X[:, support], kept, support, rfecv


def _stability_selection(X, y, feature_names, n_bootstrap=20, top_n=30,
                          threshold=0.6, random_state=42, protected=None):
    """Step 4: Keep only features that appear in the top-*top_n* importance
    ranking in at least *threshold* fraction of bootstrap resamples.

    This filters out features whose apparent importance is artefact of a
    particular train/test split.  Protected features are always kept.
    Returns the filtered array, kept names, and a boolean mask.
    """
    protected = set(protected or [])
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    votes = np.zeros(n_features)

    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        rf  = RandomForestRegressor(n_estimators=100, random_state=random_state + b, n_jobs=-1)
        rf.fit(X[idx], y[idx])
        top_idx = set(np.argsort(rf.feature_importances_)[::-1][:top_n])
        for i in top_idx:
            votes[i] += 1

    freq = votes / n_bootstrap
    mask = freq >= threshold

    for i, name in enumerate(feature_names):
        if name in protected:
            mask[i] = True

    kept    = [feature_names[i] for i in range(n_features) if mask[i]]
    dropped = [feature_names[i] for i in range(n_features) if not mask[i]]
    print(f"    Kept {len(kept)} stable features  (≥{threshold*100:.0f}% of {n_bootstrap} bootstrap runs)")
    if dropped:
        print(f"    Dropped {len(dropped)} unstable features: {dropped[:10]}{'…' if len(dropped)>10 else ''}")
    return X[:, mask], kept, mask


def select_features(X_train, X_test, y_train, feature_names,
                    rf_model, xgb_model,
                    corr_threshold=0.92,
                    var_threshold=0.01,
                    n_bootstrap=20,
                    stability_threshold=0.6,
                    protected=None,
                    random_state=42):
    """Multi-stage feature selection pipeline.

    Stages (applied to training data only; test data is filtered consistently):
      1a. Remove near-zero-variance features.
      1b. Remove one member of each highly-correlated pair (keep the more
          important one according to consensus importance).
       2. Compute RF + XGBoost consensus importance score.
       3. RFECV: find the cross-validated optimal feature count.
       4. Stability selection: keep only features that are consistently
          important across bootstrap resamples.

    Domain-protected features (PROTECTED_FEATURES) are never removed at any
    stage.

    Returns
    -------
    X_train_sel : ndarray  — training matrix with selected features
    X_test_sel  : ndarray  — test matrix with the same features
    feat_sel    : list[str] — selected feature names
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    print(f"\n  Starting with {len(feature_names)} features, {len(X_train)} training samples")

    if protected is None:
        protected = PROTECTED_FEATURES
    protected_present = [p for p in protected if p in feature_names]
    print(f"  Domain-protected features ({len(protected_present)}): {protected_present}")

    # ---- Step 1a: variance filter ----------------------------------------
    print("\n  [Step 1a] Near-zero variance filter...")
    X_tr, feat, var_mask = _remove_low_variance(X_train, feature_names, var_threshold, protected_present)
    X_te = X_test[:, var_mask]
    print(f"    → {len(feat)} features remain")

    # ---- Step 2: consensus importance (on variance-filtered set) ----------
    print("\n  [Step 2] RF + XGBoost consensus importance...")
    orig_idx    = [feature_names.index(f) for f in feat]
    rf_imp      = rf_model.feature_importances_[orig_idx]
    xgb_imp     = xgb_model.feature_importances_[orig_idx]
    rf_norm     = rf_imp  / (rf_imp.sum()  + 1e-10)
    xgb_norm    = xgb_imp / (xgb_imp.sum() + 1e-10)
    consensus   = (rf_norm + xgb_norm) / 2
    top5        = np.argsort(consensus)[::-1][:5]
    print(f"    Top 5 consensus features: {[feat[i] for i in top5]}")

    # ---- Step 1b: correlation filter (guided by consensus importance) -----
    print(f"\n  [Step 1b] Correlation filter  (|r| > {corr_threshold})...")
    X_tr, feat, corr_mask = _remove_correlated(X_tr, feat, corr_threshold, consensus, protected_present)
    X_te = X_te[:, corr_mask]
    print(f"    → {len(feat)} features remain")

    # ---- Step 3: RFECV ----------------------------------------------------
    print("\n  [Step 3] RFECV — cross-validated optimal feature count...")
    X_tr, feat, rfe_mask, _ = _rfecv_cutoff(X_tr, y_train, feat, protected_present, random_state)
    X_te = X_te[:, rfe_mask]
    print(f"    → {len(feat)} features remain")

    # ---- Step 4: stability selection --------------------------------------
    print(f"\n  [Step 4] Stability selection  ({n_bootstrap} bootstrap runs)...")
    X_tr, feat, stab_mask = _stability_selection(
        X_tr, y_train, feat,
        n_bootstrap=n_bootstrap,
        top_n=min(len(feat), 25),
        threshold=stability_threshold,
        random_state=random_state,
        protected=protected_present,
    )
    X_te = X_te[:, stab_mask]

    print(f"\n  ✓ Final feature set: {len(feat)} features  (was {len(feature_names)})")
    print(f"    n/p ratio: {len(X_tr)}/{len(feat)} = {len(X_tr)/len(feat):.1f}  (was {len(X_tr)/len(feature_names):.1f})")
    print(f"    Selected: {feat}")
    return X_tr, X_te, feat


# ============================================================================
# 10. MODEL STABILITY: BOOTSTRAP CONFIDENCE INTERVALS & LEARNING CURVES
# ============================================================================

def bootstrap_confidence_intervals(
    y_test,
    predictions: dict,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict:
    """Estimate metric uncertainty via bootstrap resampling of the test set.

    For each model, we resample the (y_test, y_pred) pairs *with replacement*
    n_bootstrap times and compute RMSE and R² on every resample.  The central
    interval [α/2, 1-α/2] of that distribution is the confidence interval.

    This quantifies how much the reported metric could change if we had drawn
    a different test set — a form of model stability assessment that does NOT
    require re-training.

    Parameters
    ----------
    y_test      : array-like  Ground-truth test targets
    predictions : dict        {model_name: y_pred array}
    n_bootstrap : int         Number of resamples (1 000 is sufficient)
    ci          : float       Coverage in percent, e.g. 95.0
    random_state: int

    Returns
    -------
    ci_results : dict  {model_name: {'rmse_mean', 'rmse_lo', 'rmse_hi',
                                      'r2_mean',   'r2_lo',   'r2_hi'}}
    """
    rng       = np.random.RandomState(random_state)
    y_test    = np.asarray(y_test)
    alpha     = (100.0 - ci) / 2.0
    n         = len(y_test)
    ci_results = {}

    print("\n" + "="*70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS  (n={n_bootstrap}, CI={ci:.0f}%)")
    print("="*70)
    print(f"  {'Model':<20} {'RMSE':>8}  {'CI RMSE':>20}  {'R²':>7}  {'CI R²':>20}")
    print("  " + "-"*78)

    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred)
        rmse_boot, r2_boot = [], []

        for _ in range(n_bootstrap):
            idx      = rng.randint(0, n, size=n)
            y_b      = y_test[idx]
            yp_b     = y_pred[idx]
            rmse_boot.append(np.sqrt(mean_squared_error(y_b, yp_b)))
            r2_boot.append(r2_score(y_b, yp_b))

        rmse_arr = np.array(rmse_boot)
        r2_arr   = np.array(r2_boot)

        rmse_lo, rmse_hi = np.percentile(rmse_arr, [alpha, 100 - alpha])
        r2_lo,   r2_hi   = np.percentile(r2_arr,   [alpha, 100 - alpha])

        ci_results[name] = {
            'rmse_mean': float(rmse_arr.mean()),
            'rmse_lo':   float(rmse_lo),
            'rmse_hi':   float(rmse_hi),
            'r2_mean':   float(r2_arr.mean()),
            'r2_lo':     float(r2_lo),
            'r2_hi':     float(r2_hi),
        }

        print(
            f"  {name:<20} {rmse_arr.mean():>8.4f}  "
            f"[{rmse_lo:.4f}, {rmse_hi:.4f}]  "
            f"{r2_arr.mean():>7.4f}  "
            f"[{r2_lo:.4f}, {r2_hi:.4f}]"
        )

    return ci_results


def plot_learning_curves(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = 'Model',
    cv: int = 5,
    n_jobs: int = -1,
    save_path: str = 'learning_curves.png',
    random_state: int = 42,
) -> None:
    """Plot training and cross-validated validation scores vs training-set size.

    Learning curves reveal:
      • Underfitting  — both train and val scores are low (high bias)
      • Overfitting   — large gap between train and val scores (high variance)
      • Saturation    — adding more data stops helping (plateau on the right)

    Uses sklearn's learning_curve helper so the scaler / preprocessing inside
    `estimator` is NOT refitted here — call this after the full preprocessing
    is done and pass the already-preprocessed arrays.

    Parameters
    ----------
    estimator  : fitted or unfitted sklearn-compatible estimator
    X_train    : np.ndarray — preprocessed training features
    y_train    : np.ndarray — training targets
    model_name : str
    cv         : int — number of cross-validation folds
    save_path  : str
    """
    from sklearn.model_selection import learning_curve

    train_sizes_rel = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        train_sizes=train_sizes_rel,
        cv=cv,
        scoring='r2',
        n_jobs=n_jobs,
        random_state=random_state,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes_abs, train_mean, 'o-', color='steelblue', label='Train R²')
    ax.fill_between(train_sizes_abs,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='steelblue')
    ax.plot(train_sizes_abs, val_mean, 'o-', color='darkorange', label='CV Val R²')
    ax.fill_between(train_sizes_abs,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='darkorange')

    ax.set_xlabel('Training set size')
    ax.set_ylabel('R²')
    ax.set_title(f'Learning Curves — {model_name}')
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=min(0, val_mean.min() - 0.1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Learning curves saved → {save_path}")


# ============================================================================
# 11. MAIN PIPELINE
# ============================================================================

def main(filepath, seed: int = 42):
    """Run the complete pipeline.

    Preprocessing order (no leakage):
      1. Load & clean
      2. EDA (before any transformation)
      3. Feature engineering (deterministic, no statistics fitted)
      4. Train / test split  ← ALL fitted statistics come AFTER this line
      5. Mean encoding       (fit group means on train, map to both)
      6. Imputation          (fit imputer on train, transform both)
      7. Skewness transforms (fit lambdas on train, apply to both)
      8. Outlier handling    (fit bounds on train, clip both)
      9. Normalisation       (fit scaler on train, transform both)
     10. Augmentation        (train only)
     11. Model training, feature selection, bootstrap CI, learning curves
    """
    print("="*70)
    print("STRUCTURAL WALL DATA ENGINEERING & ML PIPELINE")
    print("="*70)

    set_global_seed(seed)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    print("\n[1] Loading and cleaning data...")
    df = load_and_clean_data(filepath)

    # ── 2. EDA (on raw data — no transformation applied yet) ─────────────────
    print("\n[2] Exploratory Data Analysis...")
    missing_pct = eda_report(df)
    plot_eda(df, save_path='eda_plots.png')

    # ── 3. Feature engineering (deterministic, no statistics fitted) ──────────
    print("\n[3] Basic Feature Engineering...")
    df = engineer_features(df)

    print("\n[3b] Advanced Feature Engineering (Physics-based, Interactions, Polynomials)...")
    df = engineer_advanced_features(df)

    # ── 4. Train / test split (ALL fitted transforms come AFTER this) ─────────
    print("\n[4] Splitting into train / test sets...")
    target_col  = 'V_sqrt_fc'
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col and c != 'Max_Shear_Force_kN']

    y_all    = df[target_col].values
    idx_all  = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx_all, test_size=0.2, random_state=seed)

    df_train = df.iloc[idx_tr].reset_index(drop=True)
    df_test  = df.iloc[idx_te].reset_index(drop=True)
    print(f"  Train: {len(df_train)} rows  |  Test: {len(df_test)} rows")

    # ── 5. Mean encoding (fit on train only) ──────────────────────────────────
    print("\n[5] Applying mean encoding (train-only)...")
    df_train, df_test = add_mean_encoding(df_train, df_test)

    # Refresh feature list after mean encoding adds new columns
    feature_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns
                    if c != target_col and c != 'Max_Shear_Force_kN']

    # ── 6. Imputation ─────────────────────────────────────────────────────────
    print("\n[6] Comparing Imputation Strategies (Pipeline CV on train only)...")
    best_impute, impute_results = compare_imputation_strategies(df_train)

    print(f"\n[6b] Applying '{best_impute}' imputation...")
    df_train, df_test, _imputer = impute_data(df_train, df_test, strategy=best_impute)

    # ── 7. Skewness analysis + transforms ─────────────────────────────────────
    print("\n[7] Analyzing Feature Distributions (train set)...")
    skewness, skewed_cols = analyze_skewness(df_train)
    skewed_cols = [c for c in skewed_cols if c != target_col]

    print("\n[7b] Applying Yeo-Johnson transforms (fitted on train)...")
    df_train, df_test, transform_info = transform_skewed_features(
        df_train, df_test, method='yeojohnson', skewed_cols=skewed_cols
    )

    # ── 8. Outlier handling ───────────────────────────────────────────────────
    print("\n[8] Handling Outliers (bounds fitted on train)...")
    df_train, df_test, outlier_info = handle_outliers(df_train, df_test, method='iqr')

    # Plot post-transform distributions (train set)
    plot_distributions(df_train, cols=skewed_cols[:8], save_path='distributions.png')

    # ── Build X / y arrays ────────────────────────────────────────────────────
    feature_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns
                    if c != target_col and c != 'Max_Shear_Force_kN']

    X_train_raw = df_train[feature_cols].values
    X_test_raw  = df_test[feature_cols].values
    y_train     = df_train[target_col].values
    y_test      = df_test[target_col].values
    feature_names = feature_cols

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Train samples: {len(y_train)}  |  Test samples: {len(y_test)}")

    # ── 9. Normalisation (fit on train, apply to both) ────────────────────────
    print("\n[9] Comparing Normalization Methods (CV on train only)...")
    best_norm, norm_results = compare_normalization_methods(X_train_raw, y_train, random_state=seed)

    print(f"\n[9b] Applying '{best_norm}' normalisation...")
    X_train_scaled, scaler = normalize_data(X_train_raw, method=best_norm)
    X_test_scaled          = scaler.transform(X_test_raw)

    # ── 10. Augmentation (train only) ─────────────────────────────────────────
    print("\n[10] Comparing Data Augmentation Strategies...")
    best_aug, aug_results = compare_augmentation_strategies(X_train_scaled, y_train)

    if best_aug != 'none':
        print(f"\n[10b] Applying {best_aug} augmentation...")
        X_train_aug, y_train_aug = augment_data(
            X_train_scaled, y_train, method=best_aug, n_augment=1, noise_level=0.03
        )
    else:
        X_train_aug, y_train_aug = X_train_scaled, y_train

    X_test = X_test_scaled
    
    # ── 11. Train baseline models ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("[11] TRAINING BASELINE MODELS")
    print("="*70)
    
    print("\n--- Random Forest ---")
    rf_model, rf_metrics, rf_pred = train_random_forest(
        X_train_aug, X_test, y_train_aug, y_test, random_state=seed
    )
    print(f"Train RMSE: {rf_metrics['train_rmse']:.4f}, Test RMSE: {rf_metrics['test_rmse']:.4f}")
    print(f"Train R²: {rf_metrics['train_r2']:.4f}, Test R²: {rf_metrics['test_r2']:.4f}")
    print(f"Train Adj-R²: {rf_metrics['train_adj_r2']:.4f}, Test Adj-R²: {rf_metrics['test_adj_r2']:.4f}")
    print(f"Test AIC: {rf_metrics['test_aic']:.2f}, Test BIC: {rf_metrics['test_bic']:.2f}")
    
    print("\n--- XGBoost ---")
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(
        X_train_aug, X_test, y_train_aug, y_test, random_state=seed
    )
    print(f"Train RMSE: {xgb_metrics['train_rmse']:.4f}, Test RMSE: {xgb_metrics['test_rmse']:.4f}")
    print(f"Train R²: {xgb_metrics['train_r2']:.4f}, Test R²: {xgb_metrics['test_r2']:.4f}")
    print(f"Train Adj-R²: {xgb_metrics['train_adj_r2']:.4f}, Test Adj-R²: {xgb_metrics['test_adj_r2']:.4f}")
    print(f"Test AIC: {xgb_metrics['test_aic']:.2f}, Test BIC: {xgb_metrics['test_bic']:.2f}")

    # --- Feature selection ---------------------------------------------------
    # Run after RF + XGBoost so their importances can guide the consensus score.
    # NAM and EBM are then trained on the reduced feature set, which fixes the
    # Adj-R² collapse caused by having 83 features against a small test set.
    X_train_sel, X_test_sel, feature_names_sel = select_features(
        X_train_aug, X_test, y_train_aug, feature_names,
        rf_model, xgb_model,
        random_state=seed,
    )

    # ── 12. Train NAM ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("[12] TRAINING NEURAL ADDITIVE MODEL (NAM)")
    print("="*70)
    
    nam_model, nam_metrics, nam_pred, contributions, (train_losses, val_losses) = train_nam(
        X_train_sel, X_test_sel, y_train_aug, y_test, feature_names_sel,
        hidden_units=[64, 32], epochs=300, lr=0.001, batch_size=32, random_state=seed, n_cycles=3
    )
    print(f"\nTrain RMSE: {nam_metrics['train_rmse']:.4f}, Test RMSE: {nam_metrics['test_rmse']:.4f}")
    print(f"Train R²: {nam_metrics['train_r2']:.4f}, Test R²: {nam_metrics['test_r2']:.4f}")
    print(f"Train Adj-R²: {nam_metrics['train_adj_r2']:.4f}, Test Adj-R²: {nam_metrics['test_adj_r2']:.4f}")
    print(f"Test AIC: {nam_metrics['test_aic']:.2f}, Test BIC: {nam_metrics['test_bic']:.2f}")
    
    # ── 13. Train EBM ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("[13] TRAINING EXPLAINABLE BOOSTING MACHINE (EBM)")
    print("="*70)
    
    try:
        ebm_model, ebm_metrics, ebm_pred = train_ebm(
            X_train_sel, X_test_sel, y_train_aug, y_test, feature_names_sel, random_state=seed
        )
        print(f"\nTrain RMSE: {ebm_metrics['train_rmse']:.4f}, Test RMSE: {ebm_metrics['test_rmse']:.4f}")
        print(f"Train R²: {ebm_metrics['train_r2']:.4f}, Test R²: {ebm_metrics['test_r2']:.4f}")
        print(f"Train Adj-R²: {ebm_metrics['train_adj_r2']:.4f}, Test Adj-R²: {ebm_metrics['test_adj_r2']:.4f}")
        print(f"Test AIC: {ebm_metrics['test_aic']:.2f}, Test BIC: {ebm_metrics['test_bic']:.2f}")
        ebm_available = True
    except ImportError:
        print("\n  interpretml not installed. Run: pip install interpret")
        print("  Skipping EBM...")
        ebm_available = False
        ebm_metrics = {
            'train_rmse': np.nan, 'test_rmse': np.nan,
            'train_mae':  np.nan, 'test_mae':  np.nan,
            'train_r2':   np.nan, 'test_r2':   np.nan,
            'train_adj_r2': np.nan, 'test_adj_r2': np.nan,
            'train_aic':  np.nan, 'test_aic':  np.nan,
            'train_bic':  np.nan, 'test_bic':  np.nan,
        }
        ebm_pred = np.zeros_like(y_test)
    
    # ── 14. Visualizations ────────────────────────────────────────────────────
    print("\n[14] Generating Visualizations...")
    
    # Model comparison
    predictions = {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'NAM': nam_pred
    }
    if ebm_available:
        predictions['EBM'] = ebm_pred
    compare_models(y_test, predictions, save_path='model_comparison.png')
    
    # Feature importance
    plot_feature_importance(rf_model, feature_names, 'Random Forest', 'rf_importance.png')
    plot_feature_importance(xgb_model, feature_names, 'XGBoost', 'xgb_importance.png')
    
    # NAM interpretability
    # Use inverse-transformed coordinates on the x-axis where possible,
    # so that the shapes are shown in a more interpretable space than the
    # normalized coordinates used for training.
    # NAM and EBM were trained on the selected feature set (X_test_sel).
    # Pass the scaler and full feature list so both plot functions can
    # inverse-transform the x-axis back to original physical units.
    plot_nam_interpretability(
        X_test_sel, contributions, feature_names_sel,
        save_path='nam_interpretability.png',
        scaler=scaler,
        all_feature_names=feature_names,
    )
    plot_training_curves(train_losses, val_losses, 'nam_training.png')

    # EBM interpretability
    if ebm_available:
        try:
            plot_ebm_interpretability(
                ebm_model, feature_names_sel,
                save_path='ebm_interpretability.png',
                scaler=scaler,
                all_feature_names=feature_names,
            )
        except Exception as e:
            print(f"  Could not generate EBM plots: {e}")

    # Feature importance summary (all models)
    # RF/XGB used the full feature set; NAM/EBM used the selected subset.
    print_feature_importance_summary(
        rf_model, xgb_model, contributions, feature_names,
        nam_feature_names=feature_names_sel,
        ebm_model=ebm_model if ebm_available else None,
        top_k=15
    )
    
    # ── 15. Bootstrap confidence intervals ───────────────────────────────────
    print("\n[15] Bootstrap Confidence Intervals (test-set stability)...")
    _boot_preds = {
        'Random Forest': rf_pred,
        'XGBoost':       xgb_pred,
        'NAM':           nam_pred,
    }
    if ebm_available:
        _boot_preds['EBM'] = ebm_pred
    ci_results = bootstrap_confidence_intervals(
        y_test, _boot_preds, n_bootstrap=1000, ci=95.0, random_state=seed
    )

    # ── 16. Learning curves (RF and XGBoost on full training set) ────────────
    print("\n[16] Generating Learning Curves...")
    try:
        from sklearn.ensemble import RandomForestRegressor as _RF
        _rf_lc = _RF(n_estimators=100, max_depth=15, min_samples_leaf=2,
                     random_state=seed, n_jobs=-1)
        plot_learning_curves(
            _rf_lc, X_train_aug, y_train_aug,
            model_name='Random Forest',
            cv=5, save_path='lc_random_forest.png', random_state=seed,
        )
    except Exception as e:
        print(f"  RF learning curve failed: {e}")

    try:
        from xgboost import XGBRegressor as _XGB
        _xgb_lc = _XGB(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        plot_learning_curves(
            _xgb_lc, X_train_aug, y_train_aug,
            model_name='XGBoost',
            cv=5, save_path='lc_xgboost.png', random_state=seed,
        )
    except Exception as e:
        print(f"  XGB learning curve failed: {e}")

    # ── 17. Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    models_list = ['Random Forest', 'XGBoost', 'NAM']
    metrics_list = [rf_metrics, xgb_metrics, nam_metrics]
    
    if ebm_available:
        models_list.append('EBM')
        metrics_list.append(ebm_metrics)
    
    results_df = pd.DataFrame({
        'Model':         models_list,
        'Train_RMSE':    [m['train_rmse']    for m in metrics_list],
        'Test_RMSE':     [m['test_rmse']     for m in metrics_list],
        'Train_R2':      [m['train_r2']      for m in metrics_list],
        'Test_R2':       [m['test_r2']       for m in metrics_list],
        'Train_Adj_R2':  [m['train_adj_r2']  for m in metrics_list],
        'Test_Adj_R2':   [m['test_adj_r2']   for m in metrics_list],
        'Train_MAE':     [m['train_mae']     for m in metrics_list],
        'Test_MAE':      [m['test_mae']      for m in metrics_list],
        'Test_AIC':      [m['test_aic']      for m in metrics_list],
        'Test_BIC':      [m['test_bic']      for m in metrics_list],
    })
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to model_results.csv")
    
    return {
        'df_train': df_train,
        'df_test':  df_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'models': {'rf': rf_model, 'xgb': xgb_model, 'nam': nam_model},
        'metrics': {'rf': rf_metrics, 'xgb': xgb_metrics, 'nam': nam_metrics},
        'imputation_strategy': best_impute,
        'normalization_method': best_norm,
    }


if __name__ == "__main__":
    import argparse

    _dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(_dir, "Database_Negar.xlsx")

    parser = argparse.ArgumentParser(description="Structural wall ML pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default=default_data,
        help="Path to Database_Negar.xlsx (default: next to this script)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find data file at '{args.data}'. "
            f"Place 'Database_Negar.xlsx' next to working_version.py, "
            f"or pass --data /full/path/to/Database_Negar.xlsx"
        )

    results = main(args.data, seed=args.seed)