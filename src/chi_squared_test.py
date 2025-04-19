import numpy as np
from scipy.stats import chi2_contingency

# # Polaribacter contig and prophage GATC - p = 0.1129
# a, L1 = 32, (45419 - 36496)   # Contig
# b, L2 = 180, 36496            # Prophage

# # Psychromonas MAG contig and prophage GATC - p = 0.0000000130
# a, L1 = 35280 , 4938485    # MAG
# b, L2 = 144, 32300         # Prophage

# Pelagibacter contig and prophage GGATC - p = 0.0000064136
a, L1 = 23, (88007 - 63960)   # Contig
b, L2 = 97, 36496            # Prophage

# Pelagibacter MAG and prophage GGATC - p = 2.8758368063512473e-25
a, L1 = 1302, 1403101   # MAG
b, L2 = 97, 36496       # Prophage

observed = np.array([
    [a,       b      ],   # motif counts
    [L1 - a,  L2 - b ]    # non-motif counts
])

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi² statistic:           {chi2:.4f}")
print(f"Degrees of freedom:      {dof}")
print(f"p‑value:                 {p_value}")
print("Expected frequencies:\n", expected)