import numpy as np
import scipy.integrate
from src.tv_gdbb import minimize_TV


def reg_diff(ITRC, u, **kwargs):
    """Computes TV regularized derivatives using Barzilai-Borwein
    gradient descent from `src.tv_gdbb.py`.
    Parameters in **kwargs are passed to `minimize_TV`.
    
    Parameters:
        ITRC: ITRC object
        u: function to be differentiated (numpy array)

    Output: numpy array
    """
    n = len(ITRC.r_mesh)

    def A_func(v):
        I = scipy.integrate.trapezoid(v, x=ITRC.r_mesh)
        ret = scipy.integrate.cumulative_trapezoid(v, x=ITRC.r_mesh)
        ret = np.concat([ret, np.full(shape=1, fill_value=I)])
        return ret

    def AT_func(v):
        I = scipy.integrate.trapezoid(v, x=ITRC.r_mesh)
        ret = I - scipy.integrate.cumulative_trapezoid(v, x=ITRC.r_mesh)
        ret = np.concat([ret, np.zeros(1)])
        return ret

    # Create integral operator
    A = scipy.sparse.linalg.LinearOperator(
        shape=(n, n), matvec=A_func, rmatvec=AT_func, dtype=np.float64
    )
    
    # Naive differentiation as initial guess
    initial_guess = np.diff(u, append=[u[-1]]) / ITRC.dt

    # Run gradient descent
    d = minimize_TV(
        u=u,
        A=A,
        initial_guess=initial_guess,
        **kwargs
    )

    return d
