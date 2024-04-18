import pandas as pd
import seaborn as sns
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from adjustText import adjust_text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def likelihood_ratio_test(modkit_score, num_tests, alpha=0.05):
    """
    Perform the likelihood ratio test using ModKit scores with Bonferroni correction.

    Parameters:
    modkit_score (float): The ModKit score for the test.
    num_tests (int): The number of tests for Bonferroni correction.
    alpha (float): The significance level.

    Returns:
    tuple: A tuple (bool, float) representing if the test is significant and the corrected p-value.
    """
    test_statistic = 2 * modkit_score
    raw_p_value = chi2.sf(test_statistic, 2)
    corrected_p_value = min(raw_p_value * num_tests, 1.0)
    return corrected_p_value < alpha


def perform_pca_clustering(df, features, n_clusters=4, add_text=False):
    """
    Cluster and visualize methylation data by function.

    Parameters:
    df (DataFrame): DataFrame containing the methylation data.
    features (list): List of columns representing methylation levels.
    group_var (str): Column name representing the function.
    n_clusters (int): Number of clusters for KMeans clustering.
    """

    df.drop_duplicates(subset=['sample', 'name'], inplace=True)
    o_df = df.copy()

    # Initialize a LabelEncoder
    label_encoder = LabelEncoder()

    # Loop through each column to find categorical columns
    for column in features:
        if df[column].dtype == 'object':
            # Convert categorical columns to integers
            df[column] = label_encoder.fit_transform(df[column])

    # Data Preparation
    data = df[features].copy()

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_results, index=data.index, columns=['PC1', 'PC2'])

    # Add sample info back
    pca_df['sample'] = o_df.loc[pca_df.index, 'sample']
    pca_df['function'] = o_df.loc[pca_df.index, 'function']

    # Get the loadings
    loadings = pca.components_.T

    # Create a DataFrame of loadings
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=features)

    # Plot the loadings
    plt.figure(figsize=(12, 6))
    sns.heatmap(loading_matrix, annot=True, cmap='coolwarm')
    plt.title('PCA Loadings')
    plt.show()

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(pca_df[['PC1', 'PC2']])
    pca_df['cluster'] = kmeans.labels_

    # Visualization
    plt.figure(figsize=(15, 15))
    ax = sns.scatterplot(x='PC1', y='PC2', hue='cluster', style='sample', data=pca_df, palette="husl")

    # Prepare text annotations
    if add_text:
        texts = []
        for line in range(0, pca_df.shape[0]):
            if pca_df.PC2[line] > 1:
                texts.append(ax.text(pca_df.PC1[line], pca_df.PC2[line], pca_df['function'][line][:30],
                                 horizontalalignment='left', size='small', color='black'))

        # Automatically adjust text
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'), lim=10000, expand_text=(1.6, 1.6),
                    expand_points=(1.6, 1.6), expand_objects=(1.5, 1.5))

    plt.title(f'PCA and Clustering')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

    return


def perform_linear_regression(df, sample_cols, feature_cols):
    """
    Predict average methylation fraction based on the sample using Linear Regression.
    This function calculates the methylation fractions from multiple columns.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    sample_col (str): Column name representing the sample.
    feature_cols (list): List of column names representing methylation fractions.

    Returns:
    dict: Dictionary containing model performance metrics.
    """
    # Initialize a LabelEncoder
    label_encoder = LabelEncoder()

    # Loop through each column to find categorical columns
    for column in sample_cols:
        if df[column].dtype == 'object':
            # Convert categorical columns to integers
            df[column] = label_encoder.fit_transform(df[column])

    # Preparing data
    X = df[sample_cols]
    y = df[feature_cols]

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Creating and fitting the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = model.predict(X_test)

    correlation, p_value = spearmanr(df[sample_cols], df[feature_cols])

    return {
        "R-squared": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "Spearman correlation": correlation,
        "Spearman p_value": p_value
    }
