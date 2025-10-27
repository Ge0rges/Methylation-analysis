import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, chi2, f_classif, VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample


def bootstrap_pca_loadings(X, n_components=2, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    """
    Perform bootstrap resampling on PCA to compute confidence intervals for loadings.
    
    Parameters:
    -----------
    X : pandas DataFrame or numpy array
        The feature matrix (samples x features). Should NOT include non-feature columns.
    n_components : int, default=2
        Number of principal components to compute
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'original_loadings': DataFrame of loadings from original data
        - 'bootstrap_loadings': 3D array of all bootstrap loadings (n_bootstrap x n_features x n_components)
        - 'loading_means': DataFrame of mean loadings across bootstraps
        - 'loading_std': DataFrame of standard deviations of loadings
        - 'confidence_intervals': DataFrame with lower and upper bounds for each loading
        - 'original_explained_variance': Explained variance ratios from original PCA
    """
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert to numpy array if DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = X
        feature_names = [f"Feature_{i}" for i in range(X_array.shape[1])]
    
    n_samples, n_features = X_array.shape
    
    # Fit PCA on original data
    pca_original = PCA(n_components=n_components)
    pca_original.fit(X_array)
    original_loadings = pca_original.components_.T  # Shape: (n_features, n_components)
    original_variance = pca_original.explained_variance_ratio_
    
    # Store bootstrap loadings
    bootstrap_loadings = np.zeros((n_bootstrap, n_features, n_components))
    
    # Perform bootstrap
    for i in range(n_bootstrap):
        # Resample with replacement (sample indices, not rows directly)
        indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
        X_boot = X_array[indices, :]
        
        # Fit PCA on bootstrap sample
        pca_boot = PCA(n_components=n_components)
        pca_boot.fit(X_boot)
        loadings_boot = pca_boot.components_.T
        
        # Handle sign ambiguity: align bootstrap loadings with original
        # by maximizing correlation with original loadings
        for j in range(n_components):
            if np.dot(original_loadings[:, j], loadings_boot[:, j]) < 0:
                loadings_boot[:, j] *= -1
        
        bootstrap_loadings[i, :, :] = loadings_boot
    
    # Calculate statistics
    loading_means = np.mean(bootstrap_loadings, axis=0)
    loading_std = np.std(bootstrap_loadings, axis=0)
    
    # Calculate confidence intervals using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bounds = np.percentile(bootstrap_loadings, lower_percentile, axis=0)
    upper_bounds = np.percentile(bootstrap_loadings, upper_percentile, axis=0)
    
    # Create DataFrames for output
    component_names = [f"PC{i+1}" for i in range(n_components)]
    
    original_loadings_df = pd.DataFrame(
        original_loadings, 
        columns=component_names, 
        index=feature_names
    )
    
    loading_means_df = pd.DataFrame(
        loading_means, 
        columns=component_names, 
        index=feature_names
    )
    
    loading_std_df = pd.DataFrame(
        loading_std, 
        columns=component_names, 
        index=feature_names
    )
    
    # Create confidence interval DataFrame
    ci_data = []
    for i, feat in enumerate(feature_names):
        for j, comp in enumerate(component_names):
            ci_data.append({
                'Feature': feat,
                'Component': comp,
                'Original_Loading': original_loadings[i, j],
                'Mean_Loading': loading_means[i, j],
                'Std_Loading': loading_std[i, j],
                'CI_Lower': lower_bounds[i, j],
                'CI_Upper': upper_bounds[i, j],
                'Significant': not (lower_bounds[i, j] <= 0 <= upper_bounds[i, j])
            })
    
    confidence_intervals_df = pd.DataFrame(ci_data)
    
    results = {
        'original_loadings': original_loadings_df,
        'bootstrap_loadings': bootstrap_loadings,
        'loading_means': loading_means_df,
        'loading_std': loading_std_df,
        'confidence_intervals': confidence_intervals_df,
        'original_explained_variance': original_variance
    }
    
    return results


def do_feature_selection(X: pd.DataFrame, y: pd.Series, top_percentile: float =0.1) -> pl.DataFrame:
    """
    Perform feature selection using scikit-learn's mutual_info_classif, chi2, and f_classif, in combination with SelectPercentile.
    First use VarianceThreshold to remove zero-variance features.
    Result is a Polars DataFrame with one row per feature, and columns indicating whether the feature was selected by each method.
    """
    
    # Remove zero-variance features
    var_thresh = VarianceThreshold()
    X_var = var_thresh.fit_transform(X)
    features_retained = X.columns[var_thresh.get_support()].tolist()
    
    # Define tests
    tests = {
        'mutual_info': mutual_info_classif,
        'chi2': chi2,
        'f_classif': f_classif
    }
    
    # Perform feature selection for each test
    results = {'contig': [f.split(",")[0][2:-1] for f in features_retained],
               'position': [int(f.split(",")[1]) for f in features_retained],
               'strand': [f.split(",")[2][:-1] == "true" for f in features_retained]}

    for test_name, test_func in tests.items():
        selector = SelectPercentile(test_func, percentile=top_percentile * 100)
        selector.fit(X_var, y)
        results[test_name] = selector.get_support().tolist()
    
    # Add false if feature was removed by variance threshold
    for f in X.columns:
        if f not in features_retained:
            results['contig'].append(f.split(",")[0][2:-1])
            results['position'].append(int(f.split(",")[1]))
            results['strand'].append(f.split(",")[2][:-1] == "true")
            for test_name in tests.keys():
                results[test_name].append(False)
    
    return pl.DataFrame(results)


def bootstrap_pls(df, n_boot=1000, random_state=42):
    """
    Bootstrap estimates for PLS feature loadings
    """
    _, n_components = pls_cv_n_components(df, n_components_range=range(1,8))
    
    pivot_df = df.pivot_table(
        index='feature', 
        columns=['salinity', 'control', 'step'], 
        values='value', 
        fill_value=0
    )
    X = pivot_df.T.values
    features = pivot_df.index
    
    salinity_vals = np.array([col[0] for col in pivot_df.columns])
    control_vals = np.array([col[1] for col in pivot_df.columns])
    step_vals = np.array([col[2] for col in pivot_df.columns])
    le_sal = LabelEncoder()
    le_ctrl = LabelEncoder()
    sal_encoded = le_sal.fit_transform(salinity_vals)
    ctrl_encoded = le_ctrl.fit_transform(control_vals)
    Y = np.column_stack([sal_encoded, ctrl_encoded, step_vals])
    rng = np.random.default_rng(random_state)
    loadings_boot = np.zeros((n_boot, len(features), n_components))
    vip_boot = np.zeros((n_boot, len(features)))

    for i in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        X_resample = X[idx]
        Y_resample = Y[idx]
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_resample, Y_resample)
        loadings_boot[i] = pls.x_loadings_
        vip_boot[i] = vip(X_resample, Y_resample, pls)

    ci_low = np.percentile(loadings_boot, 2.5, axis=0)
    ci_high = np.percentile(loadings_boot, 97.5, axis=0)
    mean_loadings = np.mean(loadings_boot, axis=0)
    feature_importance = pd.DataFrame(mean_loadings, columns=[f'Component_{i+1}' for i in range(n_components)], index=features)
    feature_importance_ci = pd.DataFrame(ci_high-ci_low, columns=[f'CI_{i+1}' for i in range(n_components)], index=features)
    
    mean_vip = vip_boot.mean(axis=0)
    vip_scores = pd.Series(mean_vip, index=features, name="VIP")
    important = vip_scores[vip_scores >= 1]

    # Filter out any feature who's CI includes zero for all components
    feature_importance = feature_importance[(feature_importance_ci != 0).any(axis=1)]
    feature_importance_ci = feature_importance_ci.loc[feature_importance.index]
    
    # Filter out any feature who's VIP score is not important
    feature_importance = feature_importance.loc[feature_importance.index.isin(important.index)]
    feature_importance_ci = feature_importance_ci.loc[feature_importance.index]
    
    return feature_importance, feature_importance_ci


def vip(X, Y, pls):
    T = pls.x_scores_
    W = pls.x_rotations_        # not .x_weights_
    Q = pls.y_loadings_
    p, h = W.shape

    s = np.diag(T.T @ T @ Q.T @ Q)          # SS of Y explained per comp
    total_s = s.sum()

    vips = np.zeros(p)
    for i in range(p):
        weight = ((W[i] / np.linalg.norm(W, axis=0)) ** 2)
        vips[i] = np.sqrt(p * (s @ weight) / total_s)
    return vips 


def pls_cv_n_components(df, n_components_range=range(1,8)):
    """
    Cross-validate the number of PLS components via KFold
    """
    pivot_df = df.pivot_table(
        index='feature', 
        columns=['salinity', 'control', 'step'], 
        values='value', 
        fill_value=0
    )
    X = pivot_df.T.values
    salinity_vals = np.array([col[0] for col in pivot_df.columns])
    control_vals = np.array([col[1] for col in pivot_df.columns])
    step_vals = np.array([col[2] for col in pivot_df.columns])
    le_sal = LabelEncoder()
    le_ctrl = LabelEncoder()
    sal_encoded = le_sal.fit_transform(salinity_vals)
    ctrl_encoded = le_ctrl.fit_transform(control_vals)
    Y = np.column_stack([sal_encoded, ctrl_encoded, step_vals])
    
    cv_scores = []
    for n_comp in n_components_range:
        kf = KFold(n_splits=len(X), shuffle=True, random_state=42)
        mse_folds = []
        for train_idx, test_idx in kf.split(X):
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X[train_idx], Y[train_idx])
            Y_pred = pls.predict(X[test_idx])
            mse_folds.append(mean_squared_error(Y[test_idx], Y_pred))
        mean_mse = np.mean(mse_folds)
        cv_scores.append(mean_mse)
    best_n_comp = n_components_range[np.argmin(cv_scores)]
    return cv_scores, best_n_comp