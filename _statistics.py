import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm

def logistic_regression_pvalue(df, p_value_threshold=0.05):
    """
    Multinomial Logistic Regression for Methylation Types Analysis

    :param df:
    :type df:
    :return:
    :rtype:
    """
    # Convert names to categories
    df.iloc[:, 0] = df['name'].astype('category').cat.codes
    df.iloc[:, -1] = df['sample'].astype('category').cat.codes

    df['sample'] = df['sample'].astype(int)
    df['name'] = df['name'].astype(int)

    # assert that all rows have at least 1 methylation of any type
    assert all(df.iloc[:, 1:-1].sum(axis=1) > 0), "All rows must have at least 1 methylation of any type"

    # Convert each count to own row as in https://stackoverflow.com/questions/78584847/convert-count-row-to-one-hot-encoding-efficiently/78584909
    num_cols = df.columns[1:-1]
    a = df[num_cols].to_numpy()
    idx = np.repeat(np.arange(a.shape[0]), a.sum(1))
    cols = np.repeat(np.tile(np.arange(a.shape[1]), a.shape[0]), a.flat)
    b = np.zeros((len(idx), len(num_cols)), dtype=int)
    b[np.arange(len(idx)), cols] = 1

    df = df.iloc[idx]
    df.loc[:, num_cols] = b

    # Get features
    X = df['name'] + df['sample']
    y = df.iloc[:, 1:-1]

    # Get restricted features
    df_restricted = df[df["sample"] == df["sample"].unique()[0]]
    X_restricted = df_restricted['name'] + df_restricted['sample']
    y_restricted = df_restricted.iloc[:, 1:-1]

    del df_restricted
    del df
    del a
    del b
    del cols
    del idx

    # Add constant to X_train for intercept
    sm.add_constant(X)

    # Fit the logistic regression models
    model = sm.MNLogit(y, X).fit()
    restricted_model = sm.MNLogit(y_restricted, X_restricted).fit()

    # Get the rao score
    score_test_result = model.compare_lm_test(restricted_model, use_lr=True)

    return score_test_result[1] < p_value_threshold


def r_rao_score_test(df, p_value_threshold=0.05):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    # Ensure the 'sample' column is treated as a categorical variable
    df.loc[:, 'sample'] = df['sample'].astype('category').cat.codes
    df.loc[:, 'name'] = df['name'].astype('category').cat.codes

    df = df.astype(int)

    # Define the R function for fitting the model and performing the Rao score test
    r_script = """
    function(df) {
        library(VGAM)
        library(data.table)
        
        # Convert data frame to data table for faster processing
        df <- as.data.table(df)
        
        # Ensure 'sample' is a factor
        df[, sample := as.factor(sample)]
        
        # Fit the multinomial logit model in parallel
        print("Starting regression...")
        fit <- vglm(cbind(`21839`, a, m, Ncanonical) ~ sample, family = multinomial, data = df, parallel = TRUE)

        # Extracting the score vector and p values
        # print("Getting score...")
        # score_vector <- score.stat(fit)
        print("Getting p values...")
        p_values <- summary(fit, score0 = TRUE)@coef3[, "Pr(>|z|)"]
        print("Done")
        
        # Print p-values
        return(p_values)
    }
    """

    # Convert the pandas DataFrame to R DataFrame
    with (robjects.default_converter + pandas2ri.converter).context():
        r_combined_df = robjects.conversion.get_conversion().py2rpy(df)

        # Create the R function
        r_function = robjects.r(r_script)

        # Call the R function with the combined DataFrame
        result = r_function(r_combined_df)

        # Extract results
        p_value = result

    return all(p_value < p_value_threshold)


def willis_dmr_test_r(combined_methyl_data):
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import default_converter

    Y = combined_methyl_data.drop(columns=["name", "sample"])
    X = pd.get_dummies(combined_methyl_data["sample"], dtype=int)

    # Check X, Y and combined_methyl_data have the same number of rows
    assert X.shape[0] == Y.shape[0] == combined_methyl_data.shape[
        0], "X, Y and combined_methyl_data have different number of rows"

    # Check that for any row the value of sample in combined_methyl_data is True in the corresponding column in X
    for index, row in combined_methyl_data.iterrows():
        sample_value = row["sample"]
        assert X.loc[index, sample_value] == 1, f"One-hot encoding failed for row {index}"
        assert X.loc[
                   index, X.columns != sample_value].sum() == 0, f"One-hot encoding failed for row {index}: More than one column has value 1"

    # Call R function
    r = robjects.r
    r['source']('R/get_multinom_score.R')
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        get_multinom_score = robjects.globalenv['get_multinom_score']
        result = get_multinom_score(np.array(X), np.array(Y), strong=False, j=0)
    return result


def willis_dmr_test(combined_methyl_data):
    """
    Conduct a differential methylation hypothesis test.

    Parameters:
    X (np.array): Matrix of covariates for the samples, size (n_samples, n_features).
    Y (np.array): Counts of each methylation pattern, size (n_samples, n_patterns).

    Returns:
    float: The p-value of the test.
    """

    Y = combined_methyl_data.drop(columns=["name", "sample"])
    X = pd.get_dummies(combined_methyl_data["sample"], dtype=int)

    X, Y = np.array(X, dtype=float).T, np.array(Y, dtype=float)

    # Some variables - Shape goes (row x column)
    J = Y.shape[1]
    p = X.shape[0]
    n = X.shape[1]
    b = (J - 1) * (p + 1)

    # Define P_ij
    def P_ij(theta_hat, i, j):
        if j < J - 1:
            numerator = np.exp(theta_hat.T[j * (p + 1)] + X.T[i] @ theta_hat.T[j * (p + 1) + 1:(j + 1) * (p + 1)])
            denominator = 1
            for k in range(0, J - 1):
                denominator += np.exp(
                    theta_hat.T[k * (p + 1)] + X.T[i] @ theta_hat.T[k * (p + 1) + 1:(k + 1) * (p + 1)])

            return numerator / denominator

        elif j == J - 1:
            denominator = 1
            for k in range(0, J - 1):
                denominator += np.exp(
                    theta_hat.T[k * (p + 1)] + X.T[i] @ theta_hat.T[k * (p + 1) + 1:(k + 1) * (p + 1)])
            return 1 / denominator

    # Define the likelihood function to be maximized
    def likelihood_vectorized(theta):
        theta_hat = theta.reshape((J - 1, p + 1))

        intercepts = theta_hat[:, 0]
        coefficients = theta_hat[:, 1:]

        linear_comb = intercepts + np.dot(X.T, coefficients.T)
        exp_comb = np.exp(linear_comb)

        sum_exp_comb = np.sum(exp_comb, axis=1, keepdims=True)
        P = np.hstack((exp_comb / (1 + sum_exp_comb), 1 / (1 + sum_exp_comb)))

        log_P = np.log(P)
        return np.sum(Y * log_P)

    init_theta = np.zeros(b)
    result = minimize(lambda theta: -likelihood_vectorized(theta), init_theta, method='BFGS')
    theta_hat = result.x

    # Calculate score function
    s_theta = np.zeros(b)  # (J - 1) * (p + 1)
    for j in range(0, J - 1):
        for i in range(0, n):
            Ni = np.sum(Y[i])
            X_prime = np.hstack([1, X.T[i]]).T
            # This give a p+1 vector the which corresponds to the
            # (j − 1)(p + 1) + 1, (j − 1)(p + 1) + 2, . . . , (j − 1)(p + 1) + p, j(p + 1)-th elements of S
            matrix = (Y[i, j] - Ni * P_ij(theta_hat, i, j)) * X_prime.T
            s_theta[j * (p + 1):(j + 1) * (p + 1)] += matrix

    # Initialize D hat Y
    D_hat_Y = np.zeros((b, b))

    # For rows (j − 1)(p + 1) + 1 through j(p + 1) and columns (k − 1)(p + 1) + 1 through j(k + 1), populate this submatrix
    for j in range(0, J - 1):
        for k in range(0, J - 1):
            submatrix = np.zeros((p + 1, p + 1))
            for i in range(0, n):
                Ni = np.sum(Y[i])
                X_prime = np.hstack([1, X.T[i]]).T
                submatrix += (Y[i, j] - Ni * P_ij(theta_hat, i, j)) * (
                            Y[i, k] - Ni * P_ij(theta_hat, i, k)) * X_prime.T @ X_prime

            row_start = j * (p + 1)
            row_end = (j + 1) * (p + 1)
            col_start = k * (p + 1)
            col_end = (k + 1) * (p + 1)

            D_hat_Y[row_start:row_end, col_start:col_end] = submatrix

    # Create I hat Y
    I_hat_Y = np.zeros((b, b))
    for j in range(0, J - 1):
        # Diagonal submatrix for j == k
        submatrix_diag = np.zeros((p + 1, p + 1))
        for i in range(0, n):
            Ni = np.sum(Y[i])
            X_prime = np.hstack([1, X.T[i]]).T
            submatrix_diag += -Ni * P_ij(theta_hat, i, j) * (P_ij(theta_hat, i, j) - 1) * X_prime.T @ X_prime

        row_start = j * (p + 1)
        row_end = (j + 1) * (p + 1)
        col_start = row_start
        col_end = row_end

        I_hat_Y[row_start:row_end, col_start:col_end] = submatrix_diag

        # Off-diagonal submatrices for j != k
        for k in range(0, J - 1):
            if j != k:
                submatrix_off_diag = np.zeros((p + 1, p + 1))
                for i in range(n):
                    Ni = np.sum(Y[i])
                    submatrix_off_diag += Ni * P_ij(theta_hat, i, j) * P_ij(theta_hat, i, k)

                col_start_off = k * (p + 1)
                col_end_off = (k + 1) * (p + 1)

                I_hat_Y[row_start:row_end, col_start_off:col_end_off] = -submatrix_off_diag

    # Create H2
    H2 = np.zeros((p * (J - 1), b))
    for j in range(0, J - 1):
        row_start = j * p
        row_end = (j + 1) * p
        col_start = j * (p + 1) + 1
        col_end = (j + 1) * (p + 1)

        # Place the identity matrix
        H2[row_start:row_end, col_start:col_end] = np.eye(p)

    # Calculate T strong
    inverse_I_hat_Y = np.linalg.inv(I_hat_Y)
    t_strong = s_theta.T @ inverse_I_hat_Y @ H2.T @ np.linalg.inv(
        H2 @ inverse_I_hat_Y @ D_hat_Y @ inverse_I_hat_Y @ H2.T) @ H2 @ inverse_I_hat_Y @ s_theta

    # Calculate p_value
    df = p * (J - 1)  # Degrees of freedom
    p_value = stats.chi2.sf(t_strong, df)

    return p_value


def modkit_llr(modkit_score, num_tests, alpha=0.05):
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


def paired_t_test(df, p_value_threshold=0.05):
    """
    Perform a paired t-test on the methylation data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the methylation data.

    Returns:
    Bool: Wehether any one methylation column is different.
    """

    # Extract the two samples
    samples = df['sample'].unique()
    assert len(samples) == 2, "The DataFrame must contain exactly two samples for a paired t-test"

    # Do a paired t-test for each methylation column
    p_values = []
    for column in df.columns[1:-1]:
        sample1 = df[df['sample'] == samples[0]][column]
        sample2 = df[df['sample'] == samples[1]][column]

        # Perform the t-test
        _, p_value = stats.ttest_rel(sample1, sample2)
        p_values.append((column, p_value < p_value_threshold))

    # If any -_values are significant return true
    return any([p_value for _, p_value in p_values])
