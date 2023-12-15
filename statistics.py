from scipy.stats import chi2


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
