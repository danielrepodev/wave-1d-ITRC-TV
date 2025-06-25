import numpy as np
import scipy.integrate
import scipy.sparse.linalg
from src.simulation import pw_const, inverse_pw_lin
from src.reg_diff import reg_diff
import tqdm


def oper_to_matrix(L, dim, *args):
    """Transforms linear operator (python function) to its corresponding
    matrix (numpy array).

    Parameters:
        L: linear operator (vectorized function)
        dim: dimension (length of operand)

    Output: square matrix of dimension dim * dim
    """
    basis = [np.zeros(dim) for i in range(dim)]
    for i in range(dim):
        basis[i][i] = 1
    images = []
    for elem in basis:
        if not args:
            images.append(L(elem))
        else:
            images.append(L(elem, *args))
    ret = np.array(images).T
    return ret


class ITRC:
    """Class implementing the Iterative Time Reversal Control algorithm
    to compute a regularized solution for the 1+1 dimensional wave equation."""

    def __init__(self, L, n_x, T, t_mesh, Lambda):
        # Passed attributes
        self.L = L
        self.n_x = n_x  
        self.T = T
        self.Lambda = Lambda

        # Calculated attributes
        self.x_mesh = np.linspace(0, L, n_x)
        self.t_mesh = t_mesh
        self.dt = self.t_mesh[1] - self.t_mesh[0]
        self.r_mesh = self.t_mesh[self.t_mesh <= self.T]  # Subset of t_mesh

    def oper_B(self, f):
        """
        Integrate f from t to T if t<T.
        Otherwise return 0.

        Parameters:
            f: Function to be integrated

        Output: numpy array
        """

        # Get maximal index less than or equal to T
        mesh_argmax = self.t_mesh[self.t_mesh <= self.T].argmax()

        # Calculate integrals
        ret = np.zeros(self.t_mesh.size)
        for i in range(mesh_argmax):
            t_mesh_trunc = self.t_mesh[i : mesh_argmax + 1]
            f_trunc = f[i : mesh_argmax + 1]
            ret[i] = scipy.integrate.trapezoid(y=f_trunc, x=t_mesh_trunc)

        return ret

    def oper_Pr(self, f, r_ind):
        """
        Projection operator. Result is supported on [T-r,T],
        where r=r_mesh[i].

        Parameters:
            f: Function to be projected (numpy array)
            r_ind: Index of r on r_mesh (int)

        Output: numpy array
        """

        r_i = self.r_mesh[r_ind]
        ind1 = self.t_mesh >= self.T - r_i
        ind2 = self.t_mesh <= self.T
        ind = ind1 & ind2
        ret = np.zeros(len(f))
        ret[ind] = f[ind]

        return ret

    def oper_J(self, f, callback=None, max_calls=1):
        """
        Time filter operator. Integrates a truncated f
        for all values of t. Returns an array of function
        values on t_mesh.

        Parameters:
            f: Function to be integrated (numpy array)
            callback: Function to call inside integration loop

        Output: numpy array
        """

        # Initializing
        ret = np.zeros(self.t_mesh.size)
        called = 0

        # First step
        ret[0] = 0.5 * scipy.integrate.trapezoid(y=f, x=self.t_mesh)

        # Calculate integrals
        for t in range(1, len(ret) // 2 + 1):
            ret[t] = 0.5 * scipy.integrate.trapezoid(y=f[t:-t], x=self.t_mesh[t:-t])
            if (callback is not None) and (called < max_calls):
                call_result = callback(ret, t, f)
                called = called + call_result

        return ret

    def oper_R(self, f):
        """
        Time reversal operator. Maps f(t) to f(2T-t).

        Parameters:
            f: Function to be reversed (numpy array)

        Output: numpy array
        """
        ret = np.flip(f)
        return ret

    def oper_K(self, f, Lambda):
        """
        Integral kernel operator. Composition and linear combination
        of J, R and the Neumann-to-Dirichlet map.

        Parameters:
            f: Function to be mapped (numpy array)
            Lambda: Neumann-to-Dirichlet map of the Neumann problem (numpy array)

        Output: numpy array
        """
        ret = self.oper_J(Lambda @ f) - self.oper_R(
            Lambda @ self.oper_R(self.oper_J(f))
        )
        return ret

    # Matrix constructors (unused in final analysis)

    def matrix_B(self):
        """Time integral matrix. Square matrix with side length calculated from t_mesh."""
        ret = oper_to_matrix(self.oper_B, dim=self.t_mesh.size)
        return ret

    def matrix_Pr(self, i):
        """Time projection matrix. Square matrix with side length calculated from t_mesh."""
        ret = oper_to_matrix(self.oper_Pr, self.t_mesh.size, i)
        return ret

    def matrix_J(self):
        """Time filter matrix. Square matrix with side length calculated from t_mesh."""
        ret = oper_to_matrix(self.oper_J, dim=self.t_mesh.size)
        return ret

    def matrix_R(self):
        """Time reversal matrix. Square matrix with side length calculated from t_mesh."""
        ret = oper_to_matrix(self.oper_R, dim=self.t_mesh.size)
        return ret

    def matrix_K(self, Lambda):
        """Integration kernel matrix. Square matrix with side length calculated from t_mesh."""
        ret = oper_to_matrix(self.oper_K, self.t_mesh.size, Lambda)
        return ret

    def regularized_f_matrix(self, r_ind, alpha_vol):
        """
        Minimizer of regularization functional.
        Calculated by constructing matrices and using numpy.linalg.solve.

        Parameters:
            r_ind: index of r
            alpha: regularization parameter
        """

        Pr = self.matrix_Pr(r_ind)
        B = self.matrix_B()
        K = self.matrix_K(self.Lambda)
        LHS = Pr @ K @ Pr + alpha_vol * np.eye(K.shape[1])
        RHS = Pr @ B @ np.ones(self.t_mesh.shape)
        if not Pr.any():
            # If r = 0, function f should be zero
            return np.zeros(shape=(LHS.shape[1],))
        ret = np.linalg.solve(LHS, RHS)
        return ret

    def as_LO(self, oper):
        """Wrapper for the constructor of scipy.sparse.linalg.LinearOperator.
        Takes size of t_mesh into account."""
        n = len(self.t_mesh)
        return scipy.sparse.linalg.LinearOperator(
            shape=(n, n), matvec=oper, dtype=np.float64
        )

    def construct_operators(self, alpha_vol, r_ind):
        """Forms the regularization equation left- and
        right-hand sides as LHS = Pr K Pr + alpha
        and RHS = Pr B 1.

        Parameters:
            alpha: value of volume parameter
            r_ind: index of r on r_mesh

        Output:
            LHS_op: linear operator
            RHS: numpy array
        """
        if r_ind == len(self.r_mesh) - 1:
            r_ind -= 1
        B_op = self.as_LO(self.oper_B)
        
        # Two operators $P_r$ appear on the LHS.
        # They are instantiated separately here, in case
        # one wants to use subsequent points on the r-mesh.
        # In the final analysis, the operators are identical.
        curry_Pr1 = lambda v: self.oper_Pr(v, r_ind + 1)
        Pr_op1 = self.as_LO(curry_Pr1)
        curry_Pr2 = lambda v: self.oper_Pr(v, r_ind + 1)
        Pr_op2 = self.as_LO(curry_Pr2)

        curry_K = lambda v: self.oper_K(v, self.Lambda)
        K_op = self.as_LO(curry_K)
        alpha_op = scipy.sparse.linalg.aslinearoperator(
            alpha_vol * np.eye(self.t_mesh.size)
        )
        LHS_op = Pr_op2 @ K_op @ Pr_op1 + alpha_op
        RHS = Pr_op2 @ B_op @ np.ones(self.t_mesh.shape)

        return LHS_op, RHS

    def regularized_f_CG(self, r_ind, alpha_vol, precondition=None, **kwargs):
        """
        Minimizer of regularization functional.
        Calculated using the Conjugate Gradient method. Parameters in **kwargs
        are passed to scipy.sparse.linalg.cg.

        Parameters:
            r_ind: index of r
            alpha: regularization parameter
            precondition: pass "jac" to use Jacobi preconditioning,
            or "iLU" to use sparse iLU preconditioning. Not used in
            final analysis.
        """

        LHS_op, RHS = self.construct_operators(alpha_vol, r_ind)

        if precondition == "jac":
            M_diag = self.jacobi_preconditioner_diag(alpha_vol=alpha_vol)
            M_r_diag = M_diag
            if r_ind < len(self.r_mesh) - 1:
                M_r_diag[r_ind + 1 :] = 0
            M_r = np.diag(M_r_diag)
        elif precondition == "iLU":
            vecs = []
            for v in np.eye(len(self.t_mesh)):
                vecs.append(LHS_op @ v)
            LHS_arr = np.array(vecs).T
            LHS_iLU = scipy.sparse.linalg.spilu(LHS_arr)
            M_r = self.as_LO(oper=LHS_iLU.solve)
        else:
            M_r = None

        # Conjugate Gradient method
        res, exit_code = scipy.sparse.linalg.cg(LHS_op, RHS, M=M_r, **kwargs)

        return res, exit_code

    def jacobi_preconditioner_diag(self, alpha_vol):
        """Computes a diagonal (Jacobi) preconditioner for use with the
        conjugate gradient method. Not used in the final analysis."""
        LHS_op, _ = self.construct_operators(
            alpha_vol=alpha_vol, r_ind=len(self.r_mesh) - 2
        )
        vecs = []
        for v in np.eye(len(self.t_mesh)):
            vecs.append(LHS_op @ v)
        LHS_arr = np.array(vecs).T
        M_diag = 1 / np.diag(LHS_arr)
        return M_diag

    def travel_time_volume(self, callback=None, initial=None, silent=False, **kwargs):
        """Calculates travel time volumes for every r using the
        Blagovestchenskii identities. Parameters in **kwargs
        are passed to scipy.sparse.linalg.cg through regularized_f_CG.

        Parameters:
            callback: optional function to call after each mesh point, taking
            iteration number and current result as inputs. Signature: (int, numpy.array)
            initial: initial guess for the conjugate gradient method (string).
            Currently only accepts "delta", which only works for
            constant c. If not given, uses zeroes.

        Output: numpy array
        """

        # Use attribute alpha if not specified
        # (Other values may be used for plotting)
        B1 = self.oper_B(np.ones(self.t_mesh.shape))
        ret = np.zeros(self.r_mesh.size)
        called = False

        for i in tqdm.tqdm(
            range(len(self.r_mesh)), disable=silent, desc="Calculating volumes"
        ):

            if (initial == "delta") and (i < len(self.r_mesh) - 1):
                # Use an initial guess. Not used in final analysis.

                # Use delta as an initial guess
                delta_arr = np.zeros(self.t_mesh.shape)
                delta_arr[len(delta_arr) // 2 - i] = 1
                x0 = delta_arr / (
                    2 * self.dt
                )  # Dividing by 2 here somehow reduces numerical instability
                f_r, _ = self.regularized_f_CG(r_ind=i, x0=x0, **kwargs)
            else:
                # Compute f_r using an initial guess of zeroes
                f_r, _ = self.regularized_f_CG(r_ind=i, **kwargs)
            ret[i] = scipy.integrate.trapezoid(y=f_r * B1, x=self.t_mesh)

            # Check optional callback
            if (callback is not None) and (not called):
                called = callback(i, ret)

        return ret

    def travel_time_wave_speed(
        self, volume, alpha_TV=1e-1, beta=1e-8, tol=1e-5, maxiter=10**5, silent=False
    ):
        """Calculates the wave speed v in travel time coordinates from
        an array of volumes.

        Parameters:
            volume: array of volumes on r_mesh
            alpha_TV: value of TV parameter
            beta: value of absolute value smoothing constant
            tol: stopping tolerance for Barzilai-Borwein gradient descent
            maxiter: maximum number of iterations for Barzilai-Borwein gradient descent

        Output: numpy array
        """

        V_prime = reg_diff(
            ITRC=self,
            u=volume,
            alpha_TV=alpha_TV,
            beta=beta,
            tol=tol,
            maxiter=maxiter,
            silent=silent,
            desc="Calculating TV derivative",
        )
        v = 1 / V_prime
        return v

    def chi_arr(self, v_arr):
        """Compute values of chi on r_mesh. These give an x-axis value
        for r_mesh values. Note that the output values do not
        need to lie on x_mesh.

        Parameters:
            v_arr: array of travel time wave speeds on r_mesh

        Output: numpy array
        """

        ret = np.zeros(self.r_mesh.shape)  # array
        for i in range(len(self.r_mesh)):
            r_mesh_trunc = self.r_mesh[0:i]
            ret[i] = scipy.integrate.trapezoid(y=v_arr[0:i], x=r_mesh_trunc)
        return ret

    def wave_speed(self, v_arr):
        """Calculates the wave speed c from the travel time wave speed v
        using piecewise interpolation techniques from `src.simulation`.

        Parameters:
            v_arr: array of travel time wave speeds on r_mesh

        Output: numpy array
        """

        chi_arr = self.chi_arr(v_arr)
        v_func = pw_const(self.r_mesh, v_arr)
        chi_inverse = inverse_pw_lin(self.r_mesh, chi_arr)
        ret = v_func(chi_inverse(self.x_mesh))
        return ret

    def run(self, alpha_vol, alpha_TV, CG_maxiter=20, silent=False):
        """Runs entire ITRC algorithm.
        Returns values of c at each point on x_mesh.

        Parameters:
            alpha_vol: value of volume parameter
            alpha_TV: value of TV parameter
            CG_maxiter: maximum iteration number for
            the conjugate gradient method

        Output: numpy array
        """
        volume = self.travel_time_volume(
            atol=1e-6, rtol=1e-5, maxiter=CG_maxiter, alpha_vol=alpha_vol, silent=silent
        )
        v_arr = self.travel_time_wave_speed(volume, alpha_TV=alpha_TV, silent=silent)
        ret = self.wave_speed(v_arr)

        return ret, volume, v_arr

    def reconstruction_error(self, rec, c):
        """Calculates relative L^2 error of the ITRC reconstruction of c.

        Parameters:
            rec: Reconstruction (array)
            c: Wave speed (function on arrays)

        Output: float
        """

        c_arr = c(self.x_mesh)
        rel_err = np.linalg.norm(rec - c_arr) / np.linalg.norm(c_arr)
        return rel_err

    def parameter_grid_search(self, c, alpha_vol_grid, alpha_TV_grid):
        """
        Runs a grid search to find optimal regularization parameter values
        for the ITRC-TV algorithm.
        Parameters:
            c: wave speed (function on arrays)
            alpha_vol_grid: array of volume parameter values
            alpha_TV_grid: array of TV parameter values
        """

        # Initialize 2D array of errors with value np.inf everywhere
        errs = np.full((alpha_vol_grid.size, alpha_TV_grid.size), np.inf)
        for i, alpha_vol in enumerate(alpha_vol_grid):
            for j, alpha_TV in enumerate(alpha_TV_grid):
                c_rec, _, _ = self.run(alpha_vol=alpha_vol, alpha_TV=alpha_TV)
                rel_err = self.reconstruction_error(c_rec, c)
                errs[i, j] = rel_err

        # Get minimum
        vol_min, TV_min = np.unravel_index(errs.argmin(), errs.shape)
        return (alpha_vol_grid[vol_min], alpha_TV_grid[TV_min], errs)
