import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import src.simulation as simulation
import scipy


def init_gridplot(hori_labels, vert_labels):
    """Initialize plotting grid with margins. Returns a figure (`fig`) and a gridspec (`g`).
    Subplots can be added with `fig.add_subplot(g[i,j])`.
    Use within another plotting function.

    Parameters:
        hori_labels: horizontal axis labels, list of str
        vert_labels: vertical axis labels, list of str
    """

    fig = plt.figure(layout="constrained")
    nrows = 1 + len(vert_labels)
    ncols = 1 + len(hori_labels)
    w_ratios = [1 / 16] + [1 for i in range(ncols - 1)]
    h_ratios = [1 for i in range(nrows - 1)] + [1 / 16]
    g = fig.add_gridspec(nrows, ncols, width_ratios=w_ratios, height_ratios=h_ratios)

    # Create side panels to display alpha
    left_sides = []
    for i in range(nrows - 1):
        side = fig.add_subplot(g[i, 0])
        left_sides.append(side)
    bottom_sides = []
    for i in range(1, ncols):
        side = fig.add_subplot(g[nrows - 1, i])
        bottom_sides.append(side)

    for i, side in enumerate(left_sides):
        side.tick_params(size=0)
        side.set_xticklabels([])
        side.set_yticklabels([])
        spine_directions = ["top", "bottom", "right"]
        for sd in spine_directions:
            side.spines[sd].set_visible(False)
        side.set_ylabel(vert_labels[i])

    for i, side in enumerate(bottom_sides):
        side.tick_params(size=0)
        side.set_xticklabels([])
        side.set_yticklabels([])
        spine_directions = ["top", "right", "left"]
        for sd in spine_directions:
            side.spines[sd].set_visible(False)
        side.set_xlabel(hori_labels[i])

    return fig, g


def f_alpha_r_plot(
    ITRC, r_ind, alphas, solver_scale=10, a=None, cmax=1, method="CG", **kwargs
):
    """Plot boundary sources corresponding to input parameter values, and the
    arising solutions to the wave Neumann IBVP. Computes the boundary sources
    and the solutions to the forward problem.

    Parameters:
        ITRC: ITRC object
        r_ind: index of r on ITRC.r_mesh
        alphas: list of regularization parameter values
        solver_scale: scale of denser solver mesh to use when computing IBVP solution
        a: wave speed squared
        cmax: maximum value of c
        method: choice of "CG" or "Matrix". Indicates which method to use in computation.

        Other keyword arguments are passed to the iterative solver for f."""

    vert_labels = [rf"$\alpha_{{Vol}}={alpha:.2g}$" for alpha in alphas]
    hori_labels = [
        rf"$f$",
        rf"$u^f(T)$",
    ]
    fig, g = init_gridplot(hori_labels=hori_labels, vert_labels=vert_labels)

    # indicator_arr = np.zeros(ITRC.x_mesh.size)
    # indicator_arr[0:r_ind+1] = 1

    for i, alpha_vol in enumerate(alphas):

        # Form operators
        LHS_op, RHS = ITRC.construct_operators(alpha_vol, r_ind)

        if method == "CG":
            res, exit_code = ITRC.regularized_f_CG(
                r_ind=r_ind, alpha_vol=alpha_vol, **kwargs
            )
            print(f"Error in CG result = {np.max(np.abs(LHS_op @ res - RHS))}")
            print(f"{exit_code = }")
        elif method == "Matrix":
            res = ITRC.regularized_f_matrix(r_ind=r_ind, alpha_vol=alpha_vol)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # For efficiency, we plot a cubic spline interpolation of the result instead of using
        # the bump-function basis. The bump basis elements are slightly wide cubic interpolants,
        # but the difference is small (of order 1e-16).
        minus_res_spl = scipy.interpolate.CubicHermiteSpline(
            ITRC.t_mesh, -res, dydx=np.zeros(ITRC.t_mesh.shape)
        )
        r = ITRC.r_mesh[r_ind]

        ax_L = fig.add_subplot(g[i, 1])
        ax_R = fig.add_subplot(g[i, 2])
        if a is not None:
            # Plot regularized result and the related solution to
            # the wave equation
            u, ts, xs = simulation.solver(
                (1 / solver_scale) * ITRC.dt,
                ITRC.T,
                h0=minus_res_spl,
                L=ITRC.L,
                a=a,
                cmax=cmax,
            )
            max_ind = len(ts) - 1
            indicator_arr = xs <= ITRC.r_mesh[r_ind]
            ax_L.plot(ITRC.t_mesh, res, label=f"{method} result")
            ax_R.plot(xs[:max_ind], u[:max_ind], label="u^f(T)")
            ax_R.plot(xs[:max_ind], indicator_arr[:max_ind], "--", label="1_{M{r}}")

        else:
            # Only plot the regularized result
            ax_L.plot(ITRC.t_mesh, res, label=f"{method} result")
            if len(alphas) == 1:
                fig.suptitle(f"{alpha_vol = }, {r = }")
                ax_L.legend()

    if len(alphas) == 1:
        return res


def volume_grid(Vol_recs, t_mesh, alpha_vols, sigmas):
    """Plot grid of volume reconstructions. Volumes
    must be computed beforehand.

    Parameters:
        Vol_recs: dictionary of reconstructions
        t_mesh: mesh common to all reconstructions (numpy array)
        alpha_vols: list of values of volume parameter
        sigmas: list of noise standard deviations
    """
    nrows = len(alpha_vols)
    ncols = len(sigmas)
    vert_labels = [rf"$\alpha_{{Vol}}={alpha_vols[i]:.2g}$" for i in range(ncols)]
    hori_labels = [rf"$\sigma={sigmas[i]:.2g}$" for i in range(nrows)]

    fig, g = init_gridplot(hori_labels=hori_labels, vert_labels=vert_labels)

    # Create figures
    for i in range(nrows):
        for j in range(ncols):
            Vol_rec_list = Vol_recs[i][j]
            ax = fig.add_subplot(g[i, j + 1])
            for Vol_rec in Vol_rec_list:
                ax.plot(t_mesh, Vol_rec)
    plt.show()


def reg_diff_plot(ITRC, volume, d, num, alpha_vol, alpha_TV):
    """Plot volume against cumulative integral of regularized derivative.
    Both volume and derivative must be computed beforehand,
    but the cumulative integral is calculated by this function.
    Ideally, the integrated derivative should be a denoised version
    of the volume.

    Parameters:
        ITRC: ITRC object
        volume: reconstructed volume
        d: derivative (precomputed)
        num: number of reconstruction to display in title
        alpha_vol: value of volume parameter
        alpha_TV: value of TV parameter
    """
    re_int = scipy.integrate.cumulative_trapezoid(y=d, x=ITRC.r_mesh, initial=0)
    plt.plot(ITRC.r_mesh, re_int, label="Integral of TV derivative")
    plt.plot(ITRC.r_mesh, volume, label="Volume")
    plt.xlabel("r")
    plt.ylabel("V")
    plt.suptitle("Volume vs Integrated TV derivative")
    plt.title(
        rf"$c = c_{num}$, $(\alpha_{{Vol}}, \alpha_{{TV}})$ = ({alpha_vol}, {alpha_TV})"
    )
    plt.legend()


def c_comparison_plot(ITRC, c_rec, c, num, alpha_vol, alpha_TV, sigma=None, ax=None):
    """Plot reconstruction of c against the true c.

    Parameters:
        ITRC: ITRC object
        c_rec: reconstruction of c (numpy array)
        c: true wave speed function (vectorized function)
        num: number of reconstruction to display in title
        alpha_vols: list of values of volume parameter
        alpha_TVs: list of values of TV parameter
        sigmas: noise standard deviation
        ax: existing axis object to draw onto (optional)
    """
    if ax == None:
        if sigma is None:
            sigma = "undefined"
        plt.plot(ITRC.x_mesh, c_rec, label="c_rec")
        plt.plot(ITRC.x_mesh, c(ITRC.x_mesh), "--", label="c")
        plt.suptitle(rf"Comparison of $c$ and reconstruction")
        plt.title(
            rf"$c = c_{num}$, $\sigma$ = {sigma}, $(\alpha_{{Vol}}, \alpha_{{TV}})$ = ({alpha_vol}, {alpha_TV})"
        )
        plt.xlabel("x")
        plt.ylabel("c")
        plt.ylim(bottom=0)
        plt.legend()
    else:
        ax.plot(ITRC.x_mesh, c_rec, label="c_rec")
        ax.plot(ITRC.x_mesh, c(ITRC.x_mesh), "--", label="c")
        ax.set_ylim(bottom=0)


def c_comparison_grid(ITRC, c_recs, c, num, alpha_vols, alpha_TVs):
    """Plot grid of comparisons with varying values of alpha. Calls `c_comparison_plot`.

    Parameters:
        ITRC: ITRC object
        c_recs: dictionary of reconstructions
        c: true wave speed function
        num: number of c (will be deprecated)
        alpha_vols: list of values of volume parameter
        alpha_TVs: list of values of TV parameter
    """

    nrows = len(alpha_TVs)
    ncols = len(alpha_vols)
    vert_labels = [rf"$\alpha_{{Vol}}={alpha_vols[i]:.2g}$" for i in range(ncols)]
    hori_labels = [rf"$\alpha_{{TV}}={alpha_TVs[i]:.2g}$" for i in range(nrows)]

    fig, g = init_gridplot(hori_labels=hori_labels, vert_labels=vert_labels)

    # Create figures
    for i in range(nrows):
        for j in range(ncols):
            c_rec = c_recs[i][j]
            alpha_vol = alpha_vols[i]
            alpha_TV = alpha_TVs[j]
            ax = fig.add_subplot(g[i, j + 1])
            c_comparison_plot(ITRC, c_rec, c, num, alpha_vol, alpha_TV, ax=ax)
    plt.show()


# ----------
# Unused plotting functions


def LHS_delta_plot(ITRC, delta_func, r_delta, r_proj, ind=None):  # TODO: deprecate?
    fig, axs = plt.subplots(3, 3)
    if ind is None:
        ind = np.arange(len(ITRC.t_mesh))

    axs[0, 0].plot(
        ITRC.t_mesh[ind],
        delta_func(ITRC.t_mesh - r_delta)[ind],
        "o-",
        label=r"$\delta (\cdot - r_0)$",
    )
    axs[0, 1].plot(
        ITRC.t_mesh[ind],
        (ITRC.Lambda @ delta_func(ITRC.t_mesh - r_delta))[ind],
        "o-",
        label=r"$\Lambda \delta (\cdot - r_0)$",
    )
    axs[0, 2].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_J(delta_func(ITRC.t_mesh - r_delta))[ind],
        "o-",
        label=r"$J \delta (\cdot - r_0)$",
    )
    axs[1, 0].plot(
        ITRC.t_mesh[ind],
        (ITRC.Lambda @ ITRC.oper_R(ITRC.oper_J(delta_func(ITRC.t_mesh - r_delta))))[
            ind
        ],
        "o-",
        label=r"$\Lambda R J \delta (\cdot - r_0)$",
    )
    axs[1, 1].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_J(ITRC.Lambda @ delta_func(ITRC.t_mesh - r_delta))[ind],
        "o-",
        label=r"$J \Lambda \delta (\cdot - r_0)$",
    )
    axs[1, 1].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_R(
            ITRC.Lambda @ ITRC.oper_R(ITRC.oper_J(delta_func(ITRC.t_mesh - r_delta)))
        )[ind],
        "o-",
        label=r"$R \Lambda R J \delta (\cdot - r_0)$",
    )
    axs[1, 2].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_K(delta_func(ITRC.t_mesh - r_delta), Lambda=ITRC.Lambda)[ind],
        "o-",
        label=r"$K \delta (\cdot - r_0)$",
    )
    axs[2, 0].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_Pr(delta_func(ITRC.t_mesh - r_delta), i=r_proj)[ind],
        "o-",
        label=r"$P_r \delta (\cdot - r_0)$",
    )
    axs[2, 1].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_K(
            ITRC.oper_Pr(delta_func(ITRC.t_mesh - r_delta), i=r_proj),
            Lambda=ITRC.Lambda,
        )[ind],
        "o-",
        label=r"$K P_r \delta (\cdot - r_0)$",
    )
    axs[2, 2].plot(
        ITRC.t_mesh[ind],
        ITRC.oper_Pr(
            ITRC.oper_K(
                ITRC.oper_Pr(delta_func(ITRC.t_mesh - r_delta), i=r_proj),
                Lambda=ITRC.Lambda,
            ),
            i=r_proj,
        )[ind],
        "o-",
        label=r"$P_r K P_r \delta (\cdot - r_0)$",
    )

    for i in range(3):
        for j in range(3):
            axs[i, j].legend()


def c_rec_plot(ITRC, c_rec, volume, v_arr):  # TODO: deprecate?
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(ITRC.x_mesh, c_rec, label="c_rec")
    axs[1].plot(ITRC.r_mesh, volume, label="volume")
    axs[2].plot(ITRC.r_mesh, v_arr, label="v_arr")

    for ax in axs:
        ax.legend()


def LHS_RHS_plot(ITRC, alpha, r_ind, **kwargs):  # TODO: deprecate?
    LHS_op, RHS = ITRC.construct_operators(alpha=alpha, r_ind=r_ind)
    res, _ = ITRC.regularized_f_CG(r_ind, alpha, **kwargs)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(ITRC.t_mesh, RHS, label="RHS")
    axs[0, 1].plot(ITRC.t_mesh, LHS_op @ res, label="LHS_op @ CG result")
    axs[1, 0].plot(ITRC.t_mesh, RHS - LHS_op @ res, label="Next CG direction")
    axs[1, 1].plot(ITRC.t_mesh, res, label="CG result")

    fig.suptitle(
        f"Comparison of RHS, LHS and regularization result with {alpha = }, {r_ind = }"
    )
    for i in range(2):
        for j in range(2):
            axs[i, j].legend()

    # Return res to check values
    return LHS_op, RHS, res
