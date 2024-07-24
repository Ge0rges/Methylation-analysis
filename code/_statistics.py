import math
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
    # Convert strings to codes
    df.iloc[:, 0] = df['name'].astype('category').cat.codes
    df.iloc[:, -1] = df['sample'].astype('category').cat.codes

    df['sample'] = df['sample'].astype(int)
    df['name'] = df['name'].astype(int)

    # assert that all rows have at least 1 methylation of any type
    assert all(df.iloc[:, 1:-1].sum(axis=1) > 0), "All rows must have at least 1 methylation of any type"
    assert df["sample"].nunique() == 2, "Only pairwise comparisons supported"

    # Convert each count to own row as in https://stackoverflow.com/questions/78584847/convert-count-row-to-one-hot-encoding-efficiently/78584909
    num_cols = df.columns[1:-1]
    a = df[num_cols].to_numpy()
    idx = np.repeat(np.arange(a.shape[0]), a.sum(1))
    cols = np.repeat(np.tile(np.arange(a.shape[1]), a.shape[0]), a.flat)
    b = np.zeros((len(idx), len(num_cols)), dtype=int)
    b[np.arange(len(idx)), cols] = 1

    df = df.iloc[idx]
    df.loc[:, num_cols] = b

    del a
    del b
    del cols
    del idx

    # Make each sample a column
    df = pd.get_dummies(df, columns=['sample'], prefix='sample', dtype=int)

    # Get features
    X = pd.concat([df['name'], df['sample_0'], df['sample_1']], axis=1)
    X_restricted = pd.concat([df['name'], df['sample_0']], axis=1)
    y = df.iloc[:, 1:-2]

    del df

    # Add constant for intercept
    X = sm.add_constant(X)
    X_restricted = sm.add_constant(X_restricted)

    # Fit the logistic regression models
    model = sm.MNLogit(y, X).fit()
    restricted_model = sm.MNLogit(y, X_restricted).fit()

    # Get the rao score
    params_r = np.concatenate((restricted_model.params.to_numpy(), np.zeros((1, X.shape[1]-1))), axis=0).ravel("f")
    score_test_result = model.score_test(params_constrained=params_r, k_constraints=X.shape[1]-1)

    # Make the restriction matrix
    num_classes = y.shape[1] - 1
    num_params_per_class = X.shape[1]

    R = np.zeros((num_classes, num_classes*num_params_per_class))
    for i in range(num_classes):
        R[i, i * num_params_per_class + 3] = 1

    wald_test = model.wald_test(R)

    print(f"Difference between p_values {score_test_result[1] - wald_test.pvalue}")
    assert wald_test.pvalue < score_test_result[1]

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
    from rpy2.robjects.packages import importr

    Y = combined_methyl_data.drop(columns=["name", "sample"]).to_numpy()
    X = pd.get_dummies(combined_methyl_data["sample"], dtype=int).to_numpy()

    # Check X, Y and df have the same number of rows
    assert X.shape[0] == Y.shape[0] == combined_methyl_data.shape[
        0], "X, Y and df have different number of rows"
    
    # If there are any empty cols in Y remove them
    Y = Y[:, Y.any(0)]

    print(X)
    print(Y)
    # Call R function
    raobust = importr('raoBust')
    numpy2ri.activate()
    np_cv_rules = default_converter + numpy2ri.converter
    with np_cv_rules.context():
        result = raobust.multinom_test(X, Y, strong=True, j=False, penalty=False)
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

    # Save all the matrices to a csv
    pd.DataFrame(X).to_csv("X.csv")
    pd.DataFrame(Y).to_csv("Y.csv")
    pd.DataFrame(theta_hat).to_csv("theta_hat.csv")
    pd.DataFrame(s_theta).to_csv("s_theta.csv")
    pd.DataFrame(D_hat_Y).to_csv("D_hat_Y.csv")
    pd.DataFrame(I_hat_Y).to_csv("I_hat_Y.csv")
    pd.DataFrame(H2).to_csv("H2.csv")

    # Calculate T strong
    inverse_I_hat_Y = np.linalg.inv(I_hat_Y)
    t_strong = s_theta.T @ inverse_I_hat_Y @ H2.T @ np.linalg.inv(
        H2 @ inverse_I_hat_Y @ D_hat_Y @ inverse_I_hat_Y @ H2.T) @ H2 @ inverse_I_hat_Y @ s_theta

    # Calculate p_value
    df = p * (J - 1)  # Degrees of freedom
    p_value = stats.chi2.sf(t_strong, df)

    return p_value


def modkit_llr(modkit_score, num_tests, p_value_threshold=0.05):
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
    raw_p_value = stats.chi2.sf(test_statistic, 2)
    corrected_p_value = min(raw_p_value * num_tests, 1.0)
    return corrected_p_value < p_value_threshold


def pearson_chi_squared(df, p_value_threshold=0.05):
    # Perform a Pearson's Chi-squared test
    # https://online.stat.psu.edu/stat504/book/export/html/720
    # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    df = df.drop(["sample"])

    return stats.chi2_contingency(df.collect()).pvalue < p_value_threshold


def fisher_exact_test(df, p_value_threshold=0.05):
    # Perform a Fischer's exact test
    # https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    # https://github.com/scipy/scipy/issues/7099
    def untab(table):
        r, c = table.shape
        x = []
        y = []
        for i in range(r):
            for j in range(c):
                x += ([i] * table[i, j])
                y += ([j] * table[i, j])
        return np.asarray(x), np.asarray(y)

    def statistic(x, y):
        table = stats.contingency.crosstab(x, y)[1]
        return stats.contingency.chi2_contingency(table).statistic

    # Check that the data is in the correct format
    assert df['name'].nunique == 1, "Fischer exact test is designed for one gene in many samples"

    df.loc[:, 'name'] = df['name'].astype('category').cat.codes

    observed = np.asarray(df.drop(columns="sample"))

    rowsums, colsums = stats.contingency.margins(observed)
    rng = np.random.default_rng(2395834589245)
    X = stats.random_table(rowsums.ravel(), colsums.ravel(), seed=rng)

    n_mc_samples = 9999
    null_distribution = []
    for i in range(n_mc_samples):
        table = X.rvs()
        null_distribution.append(statistic(table))
    null_distribution = np.asarray(null_distribution)

    n_extreme = np.sum(null_distribution >= statistic(observed))
    pvalue = (n_extreme + 1) / (n_mc_samples + 1)

    return pvalue < p_value_threshold
