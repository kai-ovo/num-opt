import numpy as np
import matplotlib.pyplot as plt
from proximal import ppa, pgm, apgm, admm, forward_backward_splitting, fista, linearized_admm, douglas_rachford_splitting

# Define the Lasso problem
def generate_lasso_data(n_samples=100, n_features=50, noise=0.1):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + noise * np.random.randn(n_samples)
    return X, y, true_coef

# Define the Lasso objective: f(x) = 1/2 ||Ax - b||^2 + lambda * ||x||_1
def lasso_grad_f(A, b, x):
    return A.T @ (A @ x - b)

def lasso_prox_g(x, lmbda=1.0):
    return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

def lasso_prox_f(x, A, b, step_size):
    return np.linalg.solve(np.eye(A.shape[1]) + step_size * A.T @ A, x + step_size * A.T @ b)

def prox_f_wrapper(A, b):
    return lambda x, step_size: lasso_prox_f(x, A, b, step_size)


# Define the proximal operator for the ADMM approach
def prox_l2(x, step_size):
    return x / (1 + step_size)

def prox_l1(x, step_size):
    return np.sign(x) * np.maximum(np.abs(x) - step_size, 0)

# Generate data for the Lasso problem
X, y, true_coef = generate_lasso_data()
n_samples, n_features = X.shape
lmbda = 0.1

# Initial guess
x0 = np.zeros(n_features)

# # Proximal operator wrapper for methods that expect one argument
# def prox_g_wrapper(prox_g, lmbda):
#     return lambda z: prox_g(z, lmbda)

# Solve the Lasso problem using various proximal methods
methods = {
    'PPA': ppa,
    'PGM': pgm,
    'APGM': apgm,
    'FBS': forward_backward_splitting,
    'FISTA': fista,
    'ADMM': admm,
    'Linearized ADMM': linearized_admm,
    'Douglas-Rachford': douglas_rachford_splitting
}

# Store the solutions and errors for each method
solutions = {}
errors = {}

for name, method in methods.items():
    if name in ['ADMM', 'Linearized ADMM']:
        x, err = method(prox_l2, prox_l1, X, y, rho=1.0, x0=x0)
    elif name in ['PPA']:
        x, err = method(lasso_prox_g, x0)
    elif name in ['Douglas-Rachford']:
        x, err = method(prox_f_wrapper(X, y), lasso_prox_g, x0)
    else:
        x, err = method(lambda z: lasso_grad_f(X, y, z), lasso_prox_g, x0)
    
    solutions[name] = x
    errors[name] = err
    print(f"{name} solution: {x}")
    
# Plot the convergence for each method
plt.figure(figsize=(10, 6))
for name, err in errors.items():
    plt.plot(err, label=name)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Convergence of Proximal Methods for Lasso')
plt.legend()
plt.grid(True)
plt.show()
