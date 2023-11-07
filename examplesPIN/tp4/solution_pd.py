import pinocchio as pin
import numpy as np
import matplotlib.pylab as plt

plt.ion()
from numpy.linalg import inv, pinv, norm
import time
from tp4.robot_hand import RobotHand
from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# %jupyter_snippet robothand
robot = RobotHand()
viz = MeshcatVisualizer(robot, url="classical")
viz.display(robot.q0)
# %end_jupyter_snippet

# Initial state position+velocity
# %jupyter_snippet init
q = robot.q0.copy()
vq = np.zeros(robot.model.nv)
# %end_jupyter_snippet

# %jupyter_snippet hyper
# Hyperparameters for the control and the simu
Kp = 50.0  # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)  # derivative gain (D of PD)
dt = 1e-3  # simulation timestep
# %end_jupyter_snippet

### Examples for computing the generalized mass matrix and dynamic bias.
# %jupyter_snippet mass
M = pin.crba(robot.model, robot.data, q)
b = pin.nle(robot.model, robot.data, q, vq)
# %end_jupyter_snippet

### Example to compute the forward dynamics from M and b
# %jupyter_snippet dyninv
tauq = np.random.rand(robot.model.nv)
aq = inv(M) @ (tauq - b)
# %end_jupyter_snippet
# Alternatively, call the ABA algorithm
aq_bis = pin.aba(robot.model, robot.data, q, vq, tauq)
print(f"Sanity check, should be 0 ... {norm(aq-aq_bis)}")

### Example to integrate an acceleration.
# %jupyter_snippet integrate
vq += aq * dt
q = pin.integrate(robot.model, q, vq * dt)
# %end_jupyter_snippet

### Reference trajectory
# %jupyter_snippet trajref
from tp4.traj_ref import TrajRef

qdes = TrajRef(
    robot.q0,
    omega=np.array([0, 0.1, 1, 1.5, 2.5, -1, -1.5, -2.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    amplitude=1.5,
)
# %end_jupyter_snippet

# %jupyter_snippet loop
hq = []  ### For storing the logs of measured trajectory q
hqdes = []  ### For storing the logs of desired trajectory qdes
for i in range(10000):
    t = i * dt

    # Compute the model.
    M = pin.crba(robot.model, robot.data, q)
    b = pin.nle(robot.model, robot.data, q, vq)

    # Compute the PD control.
    tauq = -Kp * (q - qdes(t)) - Kv * (vq - qdes.velocity(t)) + qdes.acceleration(t)

    # Simulated the resulting acceleration (forward dynamics
    aq = inv(M) @ (tauq - b)

    # Integrate the acceleration.
    vq += aq * dt
    q = pin.integrate(robot.model, q, vq * dt)

    # Display every TDISP iterations.
    TDISP = 50e-3  # Display every 50ms
    if not i % int(TDISP / dt):  # Only display once in a while ...
        viz.display(q)
        time.sleep(TDISP)

    # Log the history.
    hq.append(q.copy())
    hqdes.append(qdes.copy())

# %end_jupyter_snippet
