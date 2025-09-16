import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ttest_ind, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def do_mutual_information(X, y):
    """
    Identify features significantly predictive of binary class using:
    - Mutual information
    - Welch's t-test (absolute statistic)
    
    Returns all features with t-test p-value below threshold,
    along with MI score, t-statistic, and consensus rank.
    """
    
    # 1. Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_table = pd.DataFrame({
        "contig": [x.split(",")[0][2:-1] for x in X.columns],
        "position": [int(x.split(",")[1]) for x in X.columns],
        "strand": [bool(x.split(",")[2]) for x in X.columns],
        "mi_score": mi_scores
    }).sort_values('mi_score', ascending=False)
    
    return mi_table


def do_t_test(X, y, p_threshold=0.05):
    # Welch's t-test, do multiple test correction then drop insignificant features
    class_0, class_1 = X[y == 0], X[y == 1]
    t_results = [ttest_ind(class_0[col], class_1[col], equal_var=False) for col in X.columns]
    abs_t_stats = [abs(t.statistic) for t in t_results]
    t_pvals = [t.pvalue for t in t_results]
    
    reject_mask, t_pvals_corrected, _, _ = multipletests(t_pvals, method='fdr_tsbh', alpha=p_threshold)
    significant = pd.DataFrame({
        "feature": X.columns,
        "t_stat": abs_t_stats,
        "p_value": t_pvals_corrected
    })
    
    significant = significant[reject_mask].sort_values('t_stat', ascending=False)

    return significant
    

def do_spearmanr(df, p_threshold=0.05):
    """
    Identify features predictive of experimental variables
    """
    
    # Create pivot table
    pivot_df = df.pivot_table(
        index='feature', 
        columns=['salinity', 'control', 'step'], 
        values='value', 
        fill_value=0
    )
    
    # Calculate correlations with each experimental variable
    results = []
    
    for feature in pivot_df.index:
        feature_values = pivot_df.loc[feature].values
        
        # Extract experimental variables from column names
        salinity_vals = [col[0] for col in pivot_df.columns]
        control_vals = [col[1] for col in pivot_df.columns]
        step_vals = [col[2] for col in pivot_df.columns]
        
        # Calculate Spearman correlations (robust to non-normal data)
        sal_corr, sal_p = spearmanr(feature_values, salinity_vals)
        ctrl_corr, ctrl_p = spearmanr(feature_values, control_vals)
        step_corr, step_p = spearmanr(feature_values, step_vals)
        
        results.append({
            'feature': feature,
            'salinity_corr': sal_corr,
            'salinity_pval': sal_p,
            'control_corr': ctrl_corr,
            'control_pval': ctrl_p,
            'step_corr': step_corr,
            'step_pval': step_p
        })
        
    results = pd.DataFrame(results)
    
    # Do multiple test correction for all variables together
    _, pvals_corrected, _, _ = multipletests(results[['salinity_pval', 'control_pval', 'step_pval']].values.flatten(), method='fdr_tsbh', alpha=p_threshold)
    results[['salinity_pval', 'control_pval', 'step_pval']] = pvals_corrected.reshape(-1, 3)

    # Filter for significance
    results = results[results[['salinity_pval', 'control_pval', 'step_pval']].min(axis=1) < p_threshold]
    
    return results


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