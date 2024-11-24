import numpy as np
import matplotlib.pyplot as plt

# A is model accuracy
# X is filtering threshold of model output probability
# N is number of samples required to achieve a certain confidence level of 95% (log odds: 2.944


# Define the function
def compute_N(X):
    return np.ceil(2.944 / np.log(X / (1 - X)))


# List of A values
A_values = [0.85, 0.97, 0.98]

# Generate X values avoiding the boundaries 0 and 1
min = 0.55
X = np.linspace(min, 0.99, 1000)

# Plot the function for each A
plt.figure(figsize=(10, 6))
for A in A_values:
    # Compute N for the current A
    X_A = A * X  # Multiply X by A
    valid_indices = (X_A > min) & (X_A < 1)  # Ensure X_A remains in valid range
    X_valid = X_A[valid_indices]
    N = compute_N(X_valid)

    # Plot the result
    plt.plot(X_valid, N, label=f'A = {A}')

# Customize the plot
plt.xlabel(r'$X$', fontsize=14)
plt.ylabel(r'$N$', fontsize=14)
plt.title('Plot of N vs X for different A values', fontsize=16)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
