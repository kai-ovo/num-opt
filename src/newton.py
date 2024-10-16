import numpy as np

def newtons(grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Basic Newton's Method for unconstrained opt
    
    Inputs:
    - grad_f: Gradient of f
    - hess_f: Hessian of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: rel. err. history
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        step = np.linalg.solve(hess, -grad)
        x = x + step
        error = np.linalg.norm(grad)
        errors.append(error)
        if error < tol:
            break
    return x, errors

def modified_newtons(grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Modified Newton's Method for optimization with Hessian modification.
    
    Inputs:
    - f: callable objective 
    - grad_f: Gradient of f
    - hess_f: Hessian of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: rel. err. history
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        eigvals = np.linalg.eigvals(hess)
        if np.any(eigvals <= 0):  # Ensure positive definiteness
            hess += np.eye(hess.shape[0]) * 1e-5
        step = np.linalg.solve(hess, -grad)
        x = x + step
        error = np.linalg.norm(grad)
        errors.append(error)
        if error < tol:
            break
    return x, errors

def bfgs( grad_f, x0, tol=1e-6, max_iter=100):
    """
    BFGS
    
    Inputs:
    - f: Objective function
    - grad_f: Gradient of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: rel. err. history
    """
    x = x0
    n = len(x)
    H = np.eye(n)
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        error = np.linalg.norm(grad)
        errors.append(error)
        if error < tol:
            break
        p = -H @ grad
        alpha = 1.0  # Line search
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - grad
        rho = 1.0 / (y.T @ s)
        H = (np.eye(n) - rho * np.outer(s, y)) @ H @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    return x, errors

def lbfgs(grad_f, x0, tol=1e-6, max_iter=100, m=10):
    """
    L-BFGS
    
    Inputs:
    - grad_f: Gradient of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - m: Memory size for storing updates
    
    Output:
    - x: The point that minimizes the function
    - errors: rel. err. hist.
    """
    x = x0
    errors = []
    s_list = []
    y_list = []
    rho_list = []
    
    for _ in range(max_iter):
        grad = grad_f(x)
        error = np.linalg.norm(grad)
        errors.append(error)
        if error < tol:
            break
        
        if len(s_list) > 0:
            q = grad
            alpha = []
            for s, y, rho in reversed(list(zip(s_list, y_list, rho_list))):
                alpha_k = rho * np.dot(s, q)
                q = q - alpha_k * y
                alpha.append(alpha_k)
            
            r = q
            for s, y, rho, alpha_k in zip(s_list, y_list, rho_list, reversed(alpha)):
                beta = rho * np.dot(y, r)
                r = r + s * (alpha_k - beta)
            direction = -r
        else:
            direction = -grad
        
        alpha = 1.0  # Line search
        x_new = x + alpha * direction
        s = x_new - x
        y = grad_f(x_new) - grad
        rho = 1.0 / np.dot(s, y)
        
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)
        
        s_list.append(s)
        y_list.append(y)
        rho_list.append(rho)
        
        x = x_new
    return x, errors

def dfp(grad_f, x0, tol=1e-6, max_iter=100):
    """
    DFP (Davidon-Fletcher-Powell) method
    
    Inputs:
    - grad_f: Gradient of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: The point that minimizes the function
    - errors: relative error history
    """
    x = x0
    n = len(x)
    H = np.eye(n)
    errors = []
    
    for _ in range(max_iter):
        grad = grad_f(x)
        error = np.linalg.norm(grad)
        errors.append(error)
        if error < tol:
            break
        
        p = -H @ grad
        alpha = 1.0  # Line search
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - grad
        H = H + (np.outer(s, s) / (s.T @ y)) - (H @ np.outer(y, y) @ H) / (y.T @ H @ y)
        
        x = x_new
    return x, errors

def inexact_newtons(grad_f, hess_f, x0, tol=1e-6, max_iter=100, cg_tol=1e-2):
    """
    Inexact Newton's Method with cg
    
    Inputs:
    - grad_f: Gradient of f
    - hess_f: Hessian of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    - cg_tol: Tolerance for the conjugate gradient solver
    
    Output:
    - x: The point that minimizes the function
    - errors: relatvie error history
    """
    x = x0
    errors = []
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        def CG(A, b, tol=cg_tol):
            r = b - A @ np.zeros_like(b)
            p = r.copy()
            x_cg = np.zeros_like(b)
            for _ in range(len(b)):
                Ap = A @ p
                alpha = r.T @ r / (p.T @ Ap)
                x_cg += alpha * p
                r_new = r - alpha * Ap
                if np.linalg.norm(r_new) < tol:
                    break
                beta = r_new.T @ r_new / (r.T @ r)
                p = r_new + beta * p
                r = r_new
            return x_cg
        
        step = CG(hess, -grad)
        x = x + step
        
        error = np.linalg.norm(grad)
        errors.append(error)
        
        if error < tol:
            break
    return x, errors

def gauss_newton(f, jacobian_f, x0, tol=1e-6, max_iter=100):
    """
    Gauss-Newton Method for non-linear least squares optimization.
    Basic idea: linearize then Newton
    
    Inputs:
    - f: objective function (in this case the least squares residual)
    - jacobian_f: Jacobian of the objective function
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum number of iterations
    
    Output:
    - x: the returned approximate minimizer
    - errors: error history
    """
    x = x0
    errors = []
    
    for _ in range(max_iter):
        residuals = f(x)
        J = jacobian_f(x)

        step = np.linalg.solve(J.T @ J, -J.T @ residuals)

        x = x + step

        error = np.linalg.norm(residuals)
        errors.append(error)

        if error < tol:
            break
    
    return x, errors

