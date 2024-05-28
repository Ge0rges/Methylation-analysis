import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.optimize import minimize
import statsmodels.api as sm



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
            numerator = np.exp(theta_hat.T[j * (p+1)] + X.T[i] @ theta_hat.T[j * (p+1) + 1:(j+1)*(p+1)])
            denominator = 1
            for k in range(0, J-1):
                denominator += np.exp(theta_hat.T[k * (p+1)] + X.T[i] @ theta_hat.T[k * (p+1) + 1:(k+1)*(p+1)])

            return numerator / denominator

        elif j == J - 1:
            denominator = 1
            for k in range(0, J-1):
                denominator += np.exp(theta_hat.T[k * (p+1)] + X.T[i] @ theta_hat.T[k * (p+1) + 1:(k+1)*(p+1)])
            return 1/denominator

    # Define the likelihood function to be maximized
    def likelihood_vectorized(theta):
        theta_hat = theta.reshape((J-1, p+1))

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
    for j in range(0, J-1):
        for i in range(0, n):
            Ni = np.sum(Y[i])
            X_prime = np.hstack([1, X.T[i]]).T
            # This give a p+1 vector the which corresponds to the
            # (j − 1)(p + 1) + 1, (j − 1)(p + 1) + 2, . . . , (j − 1)(p + 1) + p, j(p + 1)-th elements of S
            matrix = (Y[i, j] - Ni * P_ij(theta_hat, i, j)) * X_prime.T
            s_theta[j*(p+1):(j+1)*(p+1)] += matrix

    # Initialize D hat Y
    D_hat_Y = np.zeros((b, b))

    # For rows (j − 1)(p + 1) + 1 through j(p + 1) and columns (k − 1)(p + 1) + 1 through j(k + 1), populate this submatrix
    for j in range(0, J-1):
        for k in range(0, J-1):
            submatrix = np.zeros((p+1, p+1))
            for i in range(0, n):
                Ni = np.sum(Y[i])
                X_prime = np.hstack([1, X.T[i]]).T
                submatrix += (Y[i, j] - Ni*P_ij(theta_hat, i, j)) * (Y[i, k] - Ni*P_ij(theta_hat, i, k)) * X_prime.T @ X_prime

            row_start = j * (p + 1)
            row_end = (j + 1) * (p + 1)
            col_start = k * (p + 1)
            col_end = (k + 1) * (p + 1)

            D_hat_Y[row_start:row_end, col_start:col_end] = submatrix

    # Create I hat Y
    I_hat_Y = np.zeros((b, b))
    for j in range(0, J-p):
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
        for k in range(0, J-p):
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
    for j in range(0, J-1):
        row_start = j * p
        row_end = (j+1) * p
        col_start = j * (p+1) + 1
        col_end = (j+1) * (p + 1)

        # Place the identity matrix
        H2[row_start:row_end, col_start:col_end] = np.eye(p)

    # Calculate T strong
    inverse_I_hat_Y = np.linalg.inv(I_hat_Y)
    t_strong = s_theta.T @ inverse_I_hat_Y @ H2.T @ np.linalg.inv(H2 @ inverse_I_hat_Y @ D_hat_Y @ inverse_I_hat_Y @ H2.T) @ H2 @ inverse_I_hat_Y @ s_theta

    # Calculate p_value
    df = p * (J - 1)  # Degrees of freedom
    p_value = chi2.sf(t_strong, df)

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
