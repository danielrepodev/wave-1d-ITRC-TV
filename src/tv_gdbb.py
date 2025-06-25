import numpy as np
import tqdm


def minimize_TV(u, A, alpha_TV, beta, initial_guess=None, tol=None, maxiter=10**4, callback=None, silent=False, desc="Minimizing TV"):
    """Minimize the TV functional for u using gradient descent with
    Barzilai-Borwein choice of steplength, approximating the absolute
    value function |v| with sqrt(v^2 + beta).
    Forward problem is denoted by Av = u + epsilon.

    Parameters:
        u: Right-hand side of forward equation
        A: Forward operator
        alpha_TV: Regularization parameter
        beta: Smoothing parameter for the absolute value approximation
    
    Output: numpy array
    """

    # Set initial guess to zero unless provided
    if initial_guess is None:
        initial_guess = np.zeros(u.size)

    # Initialize v
    v = initial_guess

    # Gradient descent
    for n in tqdm.tqdm(range(maxiter), disable=silent, desc=desc):

        # Compute gradient
        v_p = np.roll(v, -1)
        v_m = np.roll(v, 1)
        v_p[-1] = 1  # We assume that c=1 outside [0,L]
        v_m[0] = 1
        term1 = (-2 * v_p + 2 * v) / np.sqrt(v_p**2 - 2 * v_p * v + v**2 + beta)
        term2 = (2 * v - 2 * v_m) / np.sqrt(v**2 - 2 * v * v_m + v_m**2 + beta)
        grad_TV = term1 + term2  # TV gradient term
        grad_DF = (
            2 * A.T @ ((A @ v) - u)
        )  # Data fidelity gradient term
        grad = grad_DF + alpha_TV * grad_TV

        # Calculate steplength using Barzilai-Borwein
        if n == 0:
            l = 10 ** (-4)
        else:
            delta_v = v - v_old
            delta_g = grad - grad_old
            l = np.dot(delta_v, delta_v) / np.dot(delta_v, delta_g)

        # Store variables
        v_old = v
        grad_old = grad

        # Update v
        v = v - l * grad

        # Callback
        if callback is not None:
            callback(A, v, v_old, n)

        # Early stopping criterion (if given)
        if tol is not None:
            if np.linalg.norm(v - v_old) < tol:
                print(f"Early stop on iteration {n}")
                break

    return v
