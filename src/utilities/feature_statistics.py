import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_selection import SelectFdr, mutual_info_classif, chi2, f_classif, VarianceThreshold, SelectPercentile
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample


def do_feature_selection(X: pd.DataFrame, y: pd.Series, alpha: float, top_percentile: int) -> pl.DataFrame:
    """
    Perform feature selection using scikit-learn's mutual_info_classif, chi2, and f_classif, in combination with SelectPercentile.
    First use VarianceThreshold to remove zero-variance features.
    Result is a Polars DataFrame with one row per feature, and columns indicating whether the feature was selected by each method.
    """
    
    # Remove zero-variance features
    var_thresh = VarianceThreshold(threshold=0.000625)
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
        if test_func == mutual_info_classif: # Doesn't have p-values, so use percentile
            selector = SelectPercentile(mutual_info_classif, percentile=top_percentile)
            selector.fit(X_var, y)
            results[test_name] = selector.get_support().tolist()
        else:
            selector = SelectFdr(test_func, alpha=alpha)
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


def run_pls(df, n_components=2):
    """
    Run PLS on 2 components and return the loadings
    """
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
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    
    loadings = pd.DataFrame(
        pls.x_loadings_, 
        columns=[f'Component_{i+1}' for i in range(n_components)], 
        index=features
    )
    
    return loadings
