import numpy as np
from functools import partial
from math import ceil
from scipy.interpolate import KroghInterpolator


# All functions except simulate_Lambda are by Lauri Oksanen (https://github.com/l-oksanen/oprecnn/).
# The function `solver` has been changed to accommodate mesh refinement.


def solver(
    dt,
    T,
    tmin=0,
    f=None,
    v0=None,
    v1=None,
    h0=None,
    h1=None,
    a=None,
    cmax=1,
    L=1,
    C=1,
    user_action=None,
    refinement=None
):
    """
    Solve u_tt=a*u_xx + f on (0,T) x (0,L)
    with initial conditions u=v0 and ut=v1 at t=0
    and boundary conditions ux=h0 and ux=h1 at x=0 and x=L, respectively.
    Time step is given by dt, the Courant number by C,
    and cmax is the maximum of the speed of sound i.e. max(sqrt(a)).
    Function user_action(u, t, x, n) is called at each time step.
    Here u is the solution at time t[n] in the mesh x.

    Returns u, t, x where u is the the solution at t=T in the mesh x
    and t is the mesh in time.

    """
    # Defaults
    f = (lambda t, x: 0) if f is None else f

    def init(param, val):
        if param is None:
            return lambda x: val if np.isscalar(x) else val * np.ones_like(x)
        else:
            return param

    v0 = init(v0, 0)
    v1 = init(v1, 0)
    h0 = init(h0, 0)
    h1 = init(h1, 0)
    a = init(a, cmax**2)

    user_action = (lambda u, t, x, n: False) if user_action is None else user_action

    # Meshes
    Nt = int(ceil((T / 2) / dt)) * 2  # Ensure that T is a mesh point
    # The following refinements is a modification from Lauri Oksanen's work.
    # If refinement is enabled, increase number of points after Nt has been calculated
    if refinement is not None:
        Nt = Nt * refinement
    t, dt = np.linspace(tmin, T, Nt + 1, retstep=True)  # mesh in time
    Nx = int(round(C * L / (dt * cmax)))
    x, dx = np.linspace(0, L, Nx + 1, retstep=True)  # mesh in space

    # Help variables in the scheme
    dt2 = dt**2
    dd2 = dt2 / dx**2

    # Storage arrays
    u_np1 = np.zeros(Nx + 1)  # solution at n+1
    u_n = np.zeros(Nx + 1)  # solution at n
    u_nm1 = np.zeros(Nx + 1)  # solution at n-1

    # At n=0 load initial condition
    u_nm1 = v0(x)
    user_action(u_nm1, t, x, 0)

    # At n=1 use special formula
    u_n[0] = (
        u_nm1[0]
        + dt * v1(x[0])
        + dd2 * a(x[0]) * (u_nm1[1] - u_nm1[0] - dx * h0(t[0]))
        + 0.5 * dt2 * f(t[0], x[0])
    )
    u_n[Nx] = (
        u_nm1[Nx]
        + dt * v1(x[Nx])
        + dd2 * a(x[Nx]) * (u_nm1[Nx - 1] - u_nm1[Nx] + dx * h1(t[0]))
        + 0.5 * dt2 * f(t[0], x[Nx])
    )
    u_n[1:-1] = (
        u_nm1[1:-1]
        + dt * v1(x[1:-1])
        + 0.5 * dd2 * a(x[1:-1]) * (u_nm1[2:] - 2 * u_nm1[1:-1] + u_nm1[:-2])
        + 0.5 * dt2 * f(t[0], x[1:-1])
    )
    user_action(u_n, t, x, 1)

    # Compute u_np1 given u_n and u_nm1
    for n in range(1, Nt):
        u_np1[0] = (
            -u_nm1[0]
            + 2 * u_n[0]
            + 2 * dd2 * a(x[0]) * (u_n[1] - u_n[0] - dx * h0(t[n]))
            + dt2 * f(t[n], x[0])
        )
        u_np1[Nx] = (
            -u_nm1[Nx]
            + 2 * u_n[Nx]
            + 2 * dd2 * a(x[Nx]) * (u_n[Nx - 1] - u_n[Nx] + dx * h1(t[n]))
            + dt2 * f(t[n], x[Nx])
        )
        u_np1[1:-1] = (
            -u_nm1[1:-1]
            + 2 * u_n[1:-1]
            + dd2 * a(x[1:-1]) * (u_n[2:] - 2 * u_n[1:-1] + u_n[:-2])
            + dt2 * f(t[n], x[1:-1])
        )
        if user_action(u_np1, t, x, n + 1):
            break
        # Swap storage arrays for next step
        u_nm1, u_n, u_np1 = u_n, u_np1, u_nm1

    return u_n, t, x


def bump(x, a=0, b=0.4, radius=0.2, deg=3, coefficient=1):
    a0 = a
    b0 = b
    a1 = a0 + radius
    b1 = b0 - radius

    assert a1 <= b1, "a + radius must be less than b - radius"

    pts1 = np.concatenate((a0 * np.ones(deg), a1 * np.ones(deg)))
    vals1 = np.zeros(2 * deg)
    vals1[deg] = 1
    up = KroghInterpolator(pts1, vals1)

    pts2 = np.concatenate((b1 * np.ones(deg), b0 * np.ones(deg)))
    vals2 = np.zeros(2 * deg)
    vals2[0] = 1
    down = KroghInterpolator(pts2, vals2)

    return coefficient * np.piecewise(
        x,
        [
            np.logical_and(x > a0, x < a1),
            np.logical_and(x >= a1, x <= b1),
            np.logical_and(x > b1, x < b0),
        ],
        [up, 1, down, 0],
    )


def compute_Lambda_h(c, dt, T, L, tmin=0, h0=bump, cmax=1, solver_scale=1):
    def a(x):
        return c(x) ** 2

    Lambda_h = []

    def save_step(un, t, x, n):
        Lambda_h.append(un[0])

    _, ts, xs = solver(
        dt, T, tmin=tmin, h0=h0, L=L, a=a, cmax=cmax, user_action=save_step, refinement=solver_scale
    )
    Lambda_h = Lambda_h[0::solver_scale]  # Subsample the solution
    return np.array(Lambda_h), ts, xs


def time_translate(x):
    n = len(x)
    X = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            X[:, 0] = x
        else:
            X[i:, i] = x[:-i]
    return X


def cell_condlist(x, cells):
    n = np.size(cells) - 1
    condlist = []
    for i in range(n):
        condlist.append(np.logical_and(x >= cells[i], x < cells[i + 1]))
    return condlist


def pw_const(cells, vals):
    """Create function from piecewise values"""

    def fun(x):
        return np.piecewise(x, cell_condlist(x, cells), vals)

    return fun


def integral_pw_const_node_vals(cells, vals):
    """Compute the values of the integral of a piecewise constant function on cell boundaries"""
    n = np.size(cells) - 1
    out = np.zeros(n + 1)
    for i in range(1, n + 1):
        dx = cells[i] - cells[i - 1]
        out[i] = out[i - 1] + dx * vals[i - 1]
    return out


def inverse_pw_lin(cells, node_vals):
    """Create inverse function of a piecewise linear function given by node values"""
    n = np.size(cells) - 1

    def piece(i):
        dy = node_vals[i + 1] - node_vals[i]
        dx = cells[i + 1] - cells[i]
        return lambda y: (y - node_vals[i]) * dx / dy + cells[i]

    pieces = [piece(i) for i in range(n)]

    def fun(y):
        return np.piecewise(y, cell_condlist(y, node_vals), pieces)

    return fun


def simulate_Lambda(c, dt=0.02, solver_scale=40, T=2.5, L=5, cmax=1):
    r = dt  # / 2
    rad = 1 - 10 ** (-16)
    minus_h0 = partial(bump, a=-r, b=r, radius=rad * r, deg=3, coefficient=-1)

    # Compute the first column of Lambda
    # (half of the bump is clipped out of the range of t)
    Lambda_h0, _, _ = compute_Lambda_h(
        c, dt, 2 * T, L, h0=minus_h0, cmax=cmax, solver_scale=solver_scale
    )

    # Compute the rest of Lambda
    minus_h = partial(bump, a=0, b=2 * r, radius=rad * r, deg=3, coefficient=-1)
    Lambda_h, dense_t_mesh, _ = compute_Lambda_h(
        c, dt, 2 * T, L, h0=minus_h, cmax=cmax, solver_scale=solver_scale
    )
    Lambda = time_translate(Lambda_h)  # Translate the solution
    Lambda = np.concatenate(
        [Lambda_h0[:, np.newaxis], Lambda[:, :-1]], axis=1
    )  # Combine with the first column, drop the last (out of bounds)

    return Lambda, dense_t_mesh
