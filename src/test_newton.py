import numpy as np
import matplotlib.pyplot as plt
from newton import newtons, modified_newtons, bfgs, lbfgs, dfp, inexact_newtons, gauss_newton

# Example function: Rosenbrock's function
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

# Gradient of Rosenbrock's function
def grad_rosenbrock(x):
    grad_x0 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    grad_x1 = 200 * (x[1] - x[0] ** 2)
    return np.array([grad_x0, grad_x1])

# Hessian of Rosenbrock's function
def hess_rosenbrock(x):
    hess_00 = 2 - 400 * x[1] + 1200 * x[0] ** 2
    hess_01 = -400 * x[0]
    hess_10 = hess_01
    hess_11 = 200
    return np.array([[hess_00, hess_01], [hess_10, hess_11]])

# Initial guess
x0 = np.array([-1.2, 1.0])

# Plot convergence for each method
plt.figure(figsize=(10, 7))

# Test Newton's Method
x_newton, hist_newton = newtons(rosenbrock, grad_rosenbrock, hess_rosenbrock, x0)
plt.plot(hist_newton, label="Newton's Method")
print(f"Newton's Method Solution: {x_newton}")

# Test Modified Newton's Method
x_mod_newton, hist_mod_newton = modified_newtons(rosenbrock, grad_rosenbrock, hess_rosenbrock, x0)
plt.plot(hist_mod_newton, label="Modified Newton's Method")
print(f"Modified Newton's Method Solution: {x_mod_newton}")

# Test BFGS Method
x_bfgs, hist_bfgs = bfgs(rosenbrock, grad_rosenbrock, x0)
plt.plot(hist_bfgs, label="BFGS Method")
print(f"BFGS Method Solution: {x_bfgs}")

# Test L-BFGS Method
x_lbfgs, hist_lbfgs = lbfgs(rosenbrock, grad_rosenbrock, x0)
plt.plot(hist_lbfgs, label="L-BFGS Method")
print(f"L-BFGS Method Solution: {x_lbfgs}")

# Test DFP Method
x_dfp, hist_dfp = dfp(rosenbrock, grad_rosenbrock, x0)
plt.plot(hist_dfp, label="DFP Method")
print(f"DFP Method Solution: {x_dfp}")

# Test Inexact Newton's Method
x_inexact_newton, hist_inexact_newton = inexact_newtons(rosenbrock, grad_rosenbrock, hess_rosenbrock, x0)
plt.plot(hist_inexact_newton, label="Inexact Newton's Method")
print(f"Inexact Newton's Method Solution: {x_inexact_newton}")

# Display the convergence plot
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Relative Error (log scale)')
plt.title('Convergence of Optimization Methods')
plt.legend()
plt.grid(True)
plt.show()
