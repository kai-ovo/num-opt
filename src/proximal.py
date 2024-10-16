import numpy as np

def ppa(prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Proximal Point Algorithm (PPA)
    
    Inputs:
    - prox_g: Proximal operator of g
    - x0: initialization
    - step_size: Step size (1 / regularization parameter)
    - tol: convergence tolerance 
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        x_new = prox_g(x, step_size)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, errors

def pgm(grad_f, prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Proximal Gradient Method (PGM)
    
    Inputs:
    - grad_f: Gradient of f
    - prox_g: Proximal operator of g
    - x0: initialization
    - step_size: step size
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        x_new = prox_g(x - step_size * grad, step_size)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, errors

def apgm(grad_f, prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Accelerated Proximal Gradient Method (APGM)
    
    Inputs:
    - grad_f: Gradient of f
    - prox_g: Proximal operator of g
    - x0: initialization
    - step_size: Step size for gradient updates
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    y = x0
    t = 1
    errors = []
    for _ in range(max_iter):
        grad = grad_f(y)
        x_new = prox_g(y - step_size * grad, step_size)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + (t - 1) / t_new * (x_new - x)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
        t = t_new
    return x, errors

def admm(prox_f, prox_g, A, b, rho=1.0, x0=None, z0=None, u0=None, tol=1e-6, max_iter=100):
    """
    Alternating Direction Method of Multipliers (ADMM)
    
    Inputs:
    - prox_f: Proximal operator of f
    - prox_g: Proximal operator of g
    - A: Linear operator matrix
    - b: Offset vector in the constraint
    - rho: Penalty parameter for the augmented Lagrangian
    - x0: Initial guess for x
    - z0: Initial guess for z
    - u0: Initial guess for dual variable u
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    m, n = A.shape
    if x0 is None:
        x0 = np.zeros(n)
    if z0 is None:
        z0 = np.zeros(m)
    if u0 is None:
        u0 = np.zeros(m)
    
    x = x0
    z = z0
    u = u0
    errors = []
    
    for _ in range(max_iter):
        x_new = prox_f(x - A.T @ (u + rho * z), 1 / rho)
        z_new = prox_g(A @ x_new + u / rho, 1 / rho)
        u_new = u + rho * (A @ x_new - z_new)
        
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        
        if error < tol:
            break
        
        x, z, u = x_new, z_new, u_new
    
    return x, errors

def forward_backward_splitting(grad_f, prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Forward-Backward Splitting (FBS)
    
    Inputs:
    - grad_f: Gradient of f
    - prox_g: Proximal operator of g
    - x0: Initial guess for the optimization variable
    - step_size: Step size for gradient updates
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        x_new = prox_g(x - step_size * grad, step_size)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, errors

def fista(grad_f, prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) 
    
    Inputs:
    - grad_f: Gradient of f
    - prox_g: Proximal operator of g
    - x0: Initial guess for the optimization variable
    - step_size: Step size for gradient updates
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    y = x0
    t = 1
    errors = []
    for _ in range(max_iter):
        grad = grad_f(y)
        x_new = prox_g(y - step_size * grad, step_size)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + (t - 1) / t_new * (x_new - x)
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        if error < tol:
            break
        x = x_new
        t = t_new
    return x, errors

def linearized_admm(prox_f, prox_g, A, b, rho=1.0, x0=None, z0=None, u0=None, tol=1e-6, max_iter=100):
    """
    Linearized ADMM
    
    Inputs:
    - prox_f: Proximal operator of f
    - prox_g: Proximal operator of g
    - A: Linear operator matrix
    - b: Offset vector in the constraint
    - rho: Penalty parameter for the augmented Lagrangian
    - x0: Initial guess for x
    - z0: Initial guess for z
    - u0: Initial guess for the dual
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    m, n = A.shape
    if x0 is None:
        x0 = np.zeros(n)
    if z0 is None:
        z0 = np.zeros(m)
    if u0 is None:
        u0 = np.zeros(m)
    
    x = x0
    z = z0
    u = u0
    errors = []
    
    for _ in range(max_iter):
        x_new = prox_f(x - A.T @ (u + rho * z), 1 / rho)
        z_new = prox_g(A @ x_new + u / rho, 1 / rho)
        u_new = u + rho * (A @ x_new - z_new)
        
        error = np.linalg.norm(x_new - x)
        errors.append(error)
        
        if error < tol:
            break
        
        x, z, u = x_new, z_new, u_new
    
    return x, errors

def douglas_rachford_splitting(prox_f, prox_g, x0, step_size=1.0, tol=1e-6, max_iter=100):
    """
    Douglas-Rachford Splitting method 
    
    Inputs:
    - prox_f: Proximal operator of f
    - prox_g: Proximal operator of g
    - x0: Initial guess for the optimization variable
    - step_size: Step size for proximal updates
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: List of relative errors over iterations
    """
    x = x0
    y = x0
    errors = []
    for _ in range(max_iter):
        y_new = prox_f(y, step_size)
        x_new = prox_g(2 * y_new - y, step_size)
        y = y + (x_new - y_new)
        error = np.linalg.norm(y_new - x_new)
        errors.append(error)
        if error < tol:
            break
    return y, errors
