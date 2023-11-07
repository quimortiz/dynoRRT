"""
This program presents a minimal example of using the QP solver quadprog.
sudo apt install robotpkg-py35-quadprog
"""

import quadprog
import numpy as np
from numpy.linalg import inv, pinv, norm, eig, svd

# %jupyter_snippet matrices
A = np.random.rand(5, 5) * 2 - 1
A = A @ A.T  ### Make it positive symmetric
b = np.random.rand(5)

C = np.random.rand(10, 5)
d = np.random.rand(10)
# %end_jupyter_snippet

assert np.all(eig(A)[0] > 0)

# %jupyter_snippet solve
[x, cost, _, niter, lag, iact] = quadprog.solve_qp(
    A, b, C.T, d
)  # Notice that C.T is passed instead of C
# %end_jupyter_snippet

assert np.isclose(x @ A @ x / 2 - b @ x, cost)
assert np.all((C @ x - d) >= -1e-6)  # Check primal KKT condition
assert np.all(np.isclose((C @ x - d)[iact - 1], 0))  # Check primal complementarity
assert np.all(np.isclose((A @ x - b - lag @ C), 0))  # Check dual KKT condition
assert np.all(
    np.isclose(lag[[not i in iact - 1 for i in range(len(lag))]], 0)
)  # Check dual complementarity
